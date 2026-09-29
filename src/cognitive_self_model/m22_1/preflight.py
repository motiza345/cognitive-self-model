"""Execute the frozen M22.1 preflight and write its artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import certificate_from_protocol, render_readme, write_json
from .config import config_sha256, hook_name, load_config
from .direction import orthogonal_direction, primary_direction, vector_sha256
from .leakage import audit_package_imports
from .loader import direction_tensor, forward_record, load_qwen
from .outcome import resolve_outcome_tokens
from .prompts import assert_split_integrity, frozen_prompts, prompt_manifest_sha256
from .protocol import candidate_layers, execute_protocol
from .runtime_info import git_identity


def _limitations(bundle: Any, outcome: dict[str, Any], protocol_result: dict[str, Any]) -> list[str]:
    notes = [
        "GPU is not required. Scoring uses float32.",
        "Only the pre-declared layer subset was searched. Unsearched layers are unknown.",
        "One primary direction and one orthogonal control were used. This is not a circuit discovery.",
        "Discovery selects the layer. Validation and replication are the only confirmatory splits.",
        "Each split is small. Intervals are descriptive.",
        "Dose-response shape and the directionality label are not status gates.",
        "No predictor was trained.",
    ]
    if bundle.device != "cuda":
        notes.append(f"This run used device {bundle.device}.")
    if not outcome["single_piece"]:
        notes.append(
            "At least one outcome string tokenized to more than one piece. The pre-declared rule kept the first id."
        )
    if not bundle.revision_pinned:
        notes.append(
            "Model revision was not pinned"
            + (f" ({bundle.revision_error})." if bundle.revision_error else ".")
            + " VALIDATED_FOR_M22 is unavailable until a revision is pinned."
        )
    if protocol_result["status"] != "VALIDATED_FOR_M22":
        notes.append("This artifact does not authorize an M22.2 dataset.")
    return notes


def run_preflight(output_dir: Path | None = None, loader=None) -> dict[str, Any]:
    """Run the frozen preflight.

    The default loader is ``load_qwen``. M22.1-R passes its own fail-closed
    loader. Callers that omit ``loader`` keep the historical path.
    """
    config = load_config()
    load_model = load_qwen if loader is None else loader
    identity = git_identity()
    if identity["branch"] == "main":
        raise RuntimeError("M22.1 must not be executed on main.")
    prompts = frozen_prompts()
    assert_split_integrity(prompts)
    leakage = audit_package_imports()
    bundle = load_model(config)
    layers = candidate_layers(bundle.n_layers, int(config["n_candidate_layers"]))
    primary = primary_direction(bundle.d_model, int(config["direction_seed"]))
    regenerated = primary_direction(bundle.d_model, int(config["direction_seed"]))
    control = orthogonal_direction(bundle.d_model, int(config["control_direction_seed"]), primary)
    definition_ok = bool(np.array_equal(primary, regenerated) and vector_sha256(primary) == vector_sha256(regenerated))
    outcome = resolve_outcome_tokens(
        bundle.model.tokenizer,
        str(config["outcome"]["positive_text"]),
        str(config["outcome"]["negative_text"]),
    )
    manifest = {
        "milestone": "M22.1",
        "git_branch": identity["branch"],
        "git_commit": identity["commit"],
        "git_dirty": identity["dirty"],
        "origin": identity["origin"],
        "model_id": bundle.model_id,
        "model_revision": bundle.model_revision,
        "revision_pinned": bundle.revision_pinned,
        "revision_source": bundle.revision_source,
        "revision_error": bundle.revision_error,
        "load_used_revision": bundle.load_used_revision,
        "library_versions": bundle.versions,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "n_layers": bundle.n_layers,
        "d_model": bundle.d_model,
        "candidate_layers": layers,
        "hook_template": config["hook_template"],
        "token_position": config["token_position"],
        "direction_seed": config["direction_seed"],
        "control_direction_seed": config["control_direction_seed"],
        "direction_sha256": vector_sha256(primary),
        "control_direction_sha256": vector_sha256(control),
        "direction_rule": config["direction_rule"],
        "magnitude_grid": config["magnitude_grid"],
        "primary_alpha": config["primary_alpha"],
        "outcome": outcome,
        "prompt_count": len(prompts),
        "prompts": [prompt.to_dict() for prompt in prompts],
        "prompt_manifest_sha256": prompt_manifest_sha256(prompts),
        "config_sha256": config_sha256(config),
        "splits": {
            role: [prompt.prompt_id for prompt in prompts if prompt.role == role]
            for role in ("discovery", "validation", "replication")
        },
        "gpu_required": False,
    }
    destination = output_dir if output_dir is not None else Path("reports") / "M22_1"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "manifest.json", manifest)

    tensors = {
        "primary": direction_tensor(primary, bundle.device),
        "control": direction_tensor(control, bundle.device),
    }
    progress = {"n": 0}

    def measure(prompt: Any, *, hook: bool, layer: int | None, alpha: float, direction_id: str) -> dict[str, Any]:
        progress["n"] += 1
        if progress["n"] == 1 or progress["n"] % 10 == 0:
            print(
                f"forward {progress['n']}: role={prompt.role} hook={hook} layer={layer} alpha={alpha} direction={direction_id}",
                flush=True,
            )
        vector = None if direction_id in {None, "none"} else tensors[direction_id]
        name = None if layer is None else hook_name(int(layer), config)
        return forward_record(
            bundle,
            text=prompt.text,
            prepend_bos=bool(config["prepend_bos"]),
            hook=hook,
            layer=layer,
            alpha=float(alpha),
            direction=vector,
            positive_id=int(outcome["positive_token_id"]),
            negative_id=int(outcome["negative_token_id"]),
            expected_hook_name=name,
        )

    def on_freeze(frozen: dict[str, Any]) -> None:
        payload = dict(frozen)
        payload["direction_sha256"] = manifest["direction_sha256"]
        payload["config_sha256"] = manifest["config_sha256"]
        payload["prompt_manifest_sha256"] = manifest["prompt_manifest_sha256"]
        payload["model_revision"] = manifest["model_revision"]
        write_json(destination / "frozen_candidate.json", payload)

    protocol_result = execute_protocol(
        prompts,
        n_layers=bundle.n_layers,
        measure=measure,
        config=config,
        revision_pinned=bundle.revision_pinned,
        definition_reproducible=definition_ok,
        leakage_pass=bool(leakage["pass"]),
        on_freeze=on_freeze,
    )
    limitations = _limitations(bundle, outcome, protocol_result)
    certificate = certificate_from_protocol(
        protocol_result=protocol_result,
        manifest=manifest,
        limitations=limitations,
    )
    certificate["import_audit"] = leakage
    results = {
        "status": protocol_result["status"],
        "candidate_layers": protocol_result["candidate_layers"],
        "discovery_means": protocol_result["discovery_means"],
        "selected_layer": protocol_result["selected_layer"],
        "null": protocol_result["null"],
        "detectability": protocol_result["detectability"],
        "dose_response": protocol_result["dose_response"],
        "directionality": protocol_result["directionality"],
        "evidence": protocol_result["evidence"],
        "discovery_rows": protocol_result["discovery_rows"],
        "validation_rows": None if protocol_result["validation"] is None else protocol_result["validation"]["rows"],
        "validation_control_rows": None
        if protocol_result["validation"] is None
        else protocol_result["validation"]["control_rows"],
        "replication_rows": None if protocol_result["replication"] is None else protocol_result["replication"]["rows"],
        "replication_control_rows": None
        if protocol_result["replication"] is None
        else protocol_result["replication"]["control_rows"],
    }
    write_json(destination / "intervention_preflight_results.json", results)
    write_json(destination / "certificate.json", certificate)
    validated_path = destination / "validated_intervention.json"
    rejected_path = destination / "intervention_not_validated.json"
    if certificate["status"] == "VALIDATED_FOR_M22":
        write_json(validated_path, certificate)
        if rejected_path.exists():
            rejected_path.unlink()
    else:
        write_json(rejected_path, {"status": certificate["status"], "certificate": "certificate.json"})
        if validated_path.exists():
            validated_path.unlink()
    (destination / "README.md").write_text(render_readme(certificate), encoding="utf-8")
    return certificate
