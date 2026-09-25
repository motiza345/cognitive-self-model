"""Small JSON artifacts for the M22.1 preflight. No model weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean_pair(directionality: dict[str, Any] | None, split: str) -> str:
    if not isinstance(directionality, dict) or not isinstance(directionality.get(split), dict):
        return "not measured"
    block = directionality[split]
    positive = block.get("mean_delta_positive_alpha")
    negative = block.get("mean_delta_negative_alpha")
    if positive is None or negative is None:
        return "not measured"
    sign_blind = 0.5 * (float(positive) + float(negative))
    return (
        f"+alpha {float(positive):.6f}, -alpha {float(negative):.6f}, "
        f"sign-blind average {sign_blind:.6f}, class {block.get('label')}"
    )


def render_readme(certificate: dict[str, Any]) -> str:
    status = certificate["status"]
    limitations = "\n".join(f"- {item}" for item in certificate.get("limitations", []))
    specificity = certificate.get("specificity") or {}
    direction = certificate.get("directionality") if isinstance(certificate.get("directionality"), dict) else None
    null = certificate.get("null_equivalence") or {}
    return f"""# M22.1 causal intervention preflight

Status: `{status}`

This file records one preflight of an additive last-token residual intervention on `{certificate.get("model_id")}`. It is not a self-model, not a training run, and not permission to build the M22.2 dataset unless the status is `VALIDATED_FOR_M22`.

## What was tested

- Hook family: `{certificate.get("hook")}` at token position `{certificate.get("token_position")}`.
- Outcome: `{certificate.get("outcome", {}).get("type")}` with token ids `{certificate.get("outcome", {}).get("token_ids")}`.
- Candidate layers: `{certificate.get("candidate_layers")}`.
- Frozen layer: `{certificate.get("selected_layer")}`.
- Magnitude grid: `{certificate.get("magnitude_grid")}`.
- Validation directionality (not a gate): `{_mean_pair(direction, "validation")}`.
- Replication directionality (not a gate): `{_mean_pair(direction, "replication")}`.

## What passed or failed

Null equivalence pass: `{null.get("pass")}` (max abs difference `{null.get("max_abs")}`).
Hook integrity pass: `{certificate.get("hook_integrity_pass")}`.
Primary-direction mean paired delta at alpha = +1: validation `{specificity.get("validation_target_mean")}`, replication `{specificity.get("replication_target_mean")}`.
Orthogonal-control mean paired delta at alpha = +1: validation `{specificity.get("validation_control_mean")}`, replication `{specificity.get("replication_control_mean")}`.
The frozen status rule requires the primary absolute mean to exceed the control absolute mean on both confirmatory splits. This run's status is `{status}`.
Dose response is descriptive and is not a pass/fail gate.

## Established

Only the contents of `certificate.json` for this commit, model revision, prompt manifest, and config hash. A status of `VALIDATED_FOR_M22` means that specific intervention produced a paired outcome contrast that survived the pre-declared validation, replication, null, and control checks.

## Not established

- No self-model was trained or shown to predict the outcome.
- The checkpoint was not shown to understand itself, to be self-aware, or to self-improve.
- The intervention is not a universal causal mechanism and is not a discovered circuit.
- Earlier synthetic temporal forecasts and the epistemic benchmark's invalidity score are not evidence for this result.
- Directionality and dose shape are reported separately from the status gate.

## Limitations

{limitations if limitations else "- None recorded."}

## Next permitted step

{certificate.get("next_step")}
"""


def certificate_from_protocol(
    *,
    protocol_result: dict[str, Any],
    manifest: dict[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    status = protocol_result["status"]
    if status == "VALIDATED_FOR_M22":
        next_step = "M22.2 intervention/outcome dataset construction is authorized for this frozen intervention only."
    elif status == "CANDIDATE":
        next_step = "Do not build M22.2. The intervention is technically suggestive but is not frozen; see limitations."
    else:
        next_step = "Do not build M22.2. The smallest justified next change is a new pre-registered preflight, not a larger model."
    detectability = protocol_result.get("detectability") or {}
    return {
        "status": status,
        "milestone": "M22.1",
        "model_id": manifest["model_id"],
        "model_revision": manifest.get("model_revision"),
        "hook": manifest["hook_template"],
        "selected_hook": None
        if protocol_result.get("selected_layer") is None
        else manifest["hook_template"].format(layer=protocol_result["selected_layer"]),
        "token_position": "last",
        "candidate_layers": protocol_result.get("candidate_layers"),
        "selected_layer": protocol_result.get("selected_layer"),
        "magnitude_grid": manifest["magnitude_grid"],
        "outcome": manifest["outcome"],
        "null_equivalence": protocol_result.get("null"),
        "hook_integrity_pass": protocol_result["evidence"]["hook_integrity_pass"],
        "detectability": detectability,
        "dose_response": protocol_result.get("dose_response"),
        "directionality": protocol_result.get("directionality"),
        "specificity": {
            "control": "orthogonal direction, same layer, primary alpha",
            "validation_target_mean": protocol_result["evidence"]["validation_mean"],
            "validation_control_mean": protocol_result["evidence"]["validation_control_mean"],
            "validation_paired_difference_mean": protocol_result["evidence"]["validation_paired_difference_mean"],
            "replication_target_mean": protocol_result["evidence"]["replication_mean"],
            "replication_control_mean": protocol_result["evidence"]["replication_control_mean"],
            "replication_paired_difference_mean": protocol_result["evidence"]["replication_paired_difference_mean"],
        },
        "validation_split": "validation",
        "replication_split": "replication",
        "discovery_split": "discovery",
        "evidence": protocol_result["evidence"],
        "floor": protocol_result["floor"],
        "limitations": limitations,
        "next_step": next_step,
        "self_model_trained": False,
        "config_sha256": manifest["config_sha256"],
        "prompt_manifest_sha256": manifest["prompt_manifest_sha256"],
        "git_commit": manifest.get("git_commit"),
    }
