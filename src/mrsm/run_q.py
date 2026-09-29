"""One preregistered Q run on the pinned Qwen revision.

Q.H1 is diagnostic. This module does not load the planted ground-truth file
and does not report a verified mechanism identity.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cognitive_self_model.m22_1.direction import orthogonal_direction, primary_direction, vector_sha256
from src.cognitive_self_model.m22_1.intervention import apply_last_token_additive
from src.cognitive_self_model.m22_1.outcome import resolve_outcome_tokens
from src.cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role
from src.cognitive_self_model.m22_1.protocol import candidate_layers
from src.mrsm import HEADS
from src.mrsm.baselines import predictions as baseline_predictions
from src.mrsm.evaluators import intervention_ids, score_h2, score_h3, score_h4
from src.mrsm.evidence import Evidence, residual_of
from src.mrsm.p_data import sha256_file, write_json
from src.mrsm.self_model import SelfModelCore
from src.mrsm.transforms import frozen_parameters

PINNED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots" / PINNED_REVISION
PREREG = _ROOT / "configs" / "mrsm_prereg.yaml"
P_RESULT = _ROOT / "artifacts" / "mrsm" / "p_run_001" / "result.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_manifest() -> dict[str, Any]:
    if not SNAPSHOT.is_dir():
        return {"status": "BLOCKED", "reason": "exact snapshot directory absent", "revision": PINNED_REVISION}
    weights = SNAPSHOT / "model.safetensors"
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    return {
        "status": "AVAILABLE",
        "model_id": "Qwen/Qwen2.5-0.5B",
        "revision": PINNED_REVISION,
        "snapshot": str(SNAPSHOT),
        "weight_sha256": sha256_file(weights),
        "config_sha256": sha256_file(SNAPSHOT / "config.json"),
        "tokenizer_config_sha256": sha256_file(SNAPSHOT / "tokenizer_config.json"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "model_type": config.get("model_type"),
        "substitution": False,
    }


def _load_model():
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype=torch.float32,
        revision=PINNED_REVISION,
        local_files_only=True,
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("loaded model shape does not match the pinned config")
    return model


def _catalog() -> list[dict[str, Any]]:
    layers = candidate_layers(24, 4)
    primary = primary_direction(896, 22101)
    orthogonal = orthogonal_direction(896, 22103, primary)
    catalog = []
    index = 0
    for layer in layers:
        for name, vector in (("primary", primary), ("orthogonal", orthogonal)):
            catalog.append({
                "name": HEADS[index],
                "layer": int(layer),
                "direction_name": name,
                "alpha": 1.0,
                "vector": vector,
                "vector_sha256": vector_sha256(vector),
            })
            index += 1
    return catalog


def _hooks(items: list[dict[str, Any]]):
    by_layer: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        by_layer.setdefault(int(item["layer"]), []).append(item)
    hooks = []
    for layer, group in by_layer.items():
        def hook(residual, hook, group=group):
            updated = residual
            for item in group:
                direction = torch.tensor(item["vector"], dtype=updated.dtype)
                updated = apply_last_token_additive(updated, float(item["alpha"]), direction)
            return updated
        hooks.append((f"blocks.{layer}.hook_resid_post", hook))
    return hooks


def _margins(model, texts: list[str], items: list[dict[str, Any]], positive_id: int, negative_id: int) -> list[float]:
    tokens = model.to_tokens(texts, prepend_bos=True)
    hooks = _hooks(items)
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
    last = logits[:, -1, :]
    return [float(value) for value in (last[:, positive_id] - last[:, negative_id]).tolist()]


def _effects(model, texts: list[str], catalog: list[dict[str, Any]], positive_id: int, negative_id: int):
    base = _margins(model, texts, [], positive_id, negative_id)
    singles = {}
    per_example = {}
    by_name = {item["name"]: item for item in catalog}
    for name in HEADS:
        treated = _margins(model, texts, [by_name[name]], positive_id, negative_id)
        drops = [b - t for b, t in zip(base, treated)]
        singles[name] = float(np.mean(drops))
        per_example[f"single:{name}"] = drops
    joints = {}
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            treated = _margins(model, texts, [by_name[left], by_name[right]], positive_id, negative_id)
            drops = [b - t for b, t in zip(base, treated)]
            key = f"{left}+{right}"
            joints[key] = float(np.mean(drops))
            per_example[f"pair:{key}"] = drops
    return singles, joints, per_example, base


def _fit(singles, joints, evidence_ids, scope, prereg) -> SelfModelCore:
    core = SelfModelCore(scope)
    core.update(
        singles,
        joints,
        evidence_ids,
        min_abs_interaction=float(prereg["identifiability"]["min_abs_interaction"]),
        min_relative_gap=float(prereg["identifiability"]["min_relative_gap_vs_second"]),
    )
    core.freeze()
    return core


def _q_h1(full: SelfModelCore, left: SelfModelCore, right: SelfModelCore) -> dict[str, Any]:
    """Diagnostic reproducibility. This is not a comparison to a mechanism label."""
    if not full.identifiable:
        return {"status": "NOT_EVALUATED", "reason": "NOT_IDENTIFIABLE", "verifies_mechanism_identity": False}
    full_name = full.mechanism["abstraction"]
    if not left.identifiable or not right.identifiable:
        return {
            "status": "FAIL",
            "reason": "candidate not reproducible on both discovery halves",
            "candidate": full_name,
            "verifies_mechanism_identity": False,
        }
    same = left.mechanism["abstraction"] == full_name and right.mechanism["abstraction"] == full_name
    members = full.mechanism["ordered"]
    signs_ok = all(float(half.effects[name]) > 0.0 for half in (left, right) for name in members)
    passed = same and signs_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "candidate": full_name,
        "half_a": left.mechanism["abstraction"],
        "half_b": right.mechanism["abstraction"],
        "member_effects_positive": signs_ok,
        "verifies_mechanism_identity": False,
        "note": "diagnostic/transfer evidence only",
    }


def _h3_cases(core: SelfModelCore, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original = "NOT_IDENTIFIABLE"
    if core.mechanism and core.mechanism.get("status") == "IDENTIFIED":
        original = str(core.mechanism["abstraction"])
    cases = []
    for transform_id in ("T1", "T2", "T3"):
        reading = core.read_transformed(transform_id)
        for row in rows:
            kind, name = row["intervention_id"].split(":", 1)
            if kind == "single":
                transformed = float(reading["recovered_effects"][name])
            else:
                left, right = name.split("+")
                transformed = (
                    float(reading["recovered_effects"][left])
                    + float(reading["recovered_effects"][right])
                    + float(core.interactions[name])
                )
            cases.append({
                "intervention_id": row["intervention_id"],
                "transformation_id": transform_id,
                "original_prediction": row["prediction"],
                "transformed_prediction": transformed,
                "mechanism_abstraction": reading["mechanism_abstraction"],
                "original_abstraction": original,
                "scope": reading["scope"],
                "original_scope": core.scope,
                "uncertainty": reading["uncertainty"],
            })
    return cases


def run() -> dict[str, Any]:
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    p_result = json.loads(P_RESULT.read_text(encoding="utf-8"))
    gates = p_result.get("gates", {})
    if any(gates.get(key) != "PASS" for key in ("H1", "H2", "H3", "H4")):
        raise RuntimeError("Q is not allowed unless every P gate passed")
    out = _ROOT / "artifacts" / "mrsm" / "q_run_001"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("q_run_001 already exists")
    out.mkdir(parents=True)
    manifest = _artifact_manifest()
    write_json(out / "artifact_manifest.json", manifest)
    if manifest.get("status") != "AVAILABLE":
        result = {"run_id": "q_run_001", "status": "BLOCKED", "reason": manifest.get("reason"), "scientific_failure": False}
        write_json(out / "result.json", result)
        return result
    model = _load_model()
    tokens = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    positive_id = int(tokens["positive_token_id"])
    negative_id = int(tokens["negative_token_id"])
    catalog = _catalog()
    write_json(out / "intervention_catalog.json", [
        {key: value for key, value in item.items() if key != "vector"} for item in catalog
    ])
    grouped = prompts_by_role(frozen_prompts())
    discovery = [record.text for record in grouped["discovery"]]
    holdout = [record.text for record in grouped["validation"]]
    scope = "Q:m22_1_residual:diagnostic"
    singles, joints, _, _ = _effects(model, discovery, catalog, positive_id, negative_id)
    core = _fit(singles, joints, [f"discovery-{name}" for name in HEADS], scope, prereg)
    ordered = sorted(grouped["discovery"], key=lambda record: record.prompt_id)
    mid = len(ordered) // 2
    left_texts = [record.text for record in ordered[:mid]]
    right_texts = [record.text for record in ordered[mid:]]
    left_s, left_j, _, _ = _effects(model, left_texts, catalog, positive_id, negative_id)
    right_s, right_j, _, _ = _effects(model, right_texts, catalog, positive_id, negative_id)
    left_core = _fit(left_s, left_j, ["half-a"], scope, prereg)
    right_core = _fit(right_s, right_j, ["half-b"], scope, prereg)
    rows = []
    stamp = _now()
    for intervention_id in intervention_ids():
        rows.append({
            "intervention_id": intervention_id,
            "prediction": core.predict(intervention_id),
            "actual_outcome": None,
            "timestamp": stamp,
            "order": "before_outcome",
        })
    predictions = {
        "phase": "before_outcome",
        "rows": rows,
        "baselines": baseline_predictions(core.effects),
        "h3_parameters": frozen_parameters(),
        "verifies_mechanism_identity": False,
    }
    write_json(out / "predictions.json", predictions)
    write_json(out / "self_model_state.json", core.explain())
    if any(row["actual_outcome"] is not None for row in rows):
        result = {"run_id": "q_run_001", "status": "INVALID", "reason": "outcome present before execution"}
        write_json(out / "result.json", result)
        return result
    _, _, holdout_examples, holdout_base = _effects(model, holdout, catalog, positive_id, negative_id)
    self_pred = {row["intervention_id"]: float(row["prediction"]) for row in rows}
    h1 = _q_h1(core, left_core, right_core)
    h2 = score_h2(
        self_pred,
        predictions["baselines"],
        holdout_examples,
        resamples=int(prereg["statistical_procedure"]["resamples"]),
        seed=int(prereg["statistical_procedure"]["seed"]),
        sign_floor=float(prereg["statistical_procedure"]["sign_zero_floor"]),
    )
    cases = _h3_cases(core, rows)
    h3 = score_h3(cases, sign_floor=float(prereg["statistical_procedure"]["sign_zero_floor"]))
    chosen = core.choose_ablation()
    blind = prereg["h4"]["consumption_ablation"]["action"]
    by_name = {item["name"]: item for item in catalog}
    full_margins = _margins(model, holdout, [by_name[chosen]], positive_id, negative_id)
    blind_margins = _margins(model, holdout, [by_name[blind]], positive_id, negative_id)
    # score_h4 expects higher-is-better episode scores. Logit margin is that outcome.
    h4 = score_h4(
        full_margins,
        blind_margins,
        resamples=int(prereg["statistical_procedure"]["resamples"]),
        seed=int(prereg["statistical_procedure"]["seed"]),
    )
    h4["full_action"] = chosen
    h4["consumption_ablation_action"] = blind
    h4["utility_definition"] = "mean_holdout_logit_margin"
    evidence = []
    for key, pred in self_pred.items():
        actual = float(np.mean(holdout_examples[key]))
        evidence.append(Evidence(
            observation_id=f"q-holdout-{key}",
            intervention_id=key,
            prediction=pred,
            actual_outcome=actual,
            residual=residual_of(pred, actual),
            context={"split": "validation"},
            scope=scope,
            evidence_type="holdout_scored",
            timestamp=_now(),
            run_id="q_run_001",
        ).to_dict())
    metrics = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}
    write_json(out / "metrics.json", metrics)
    write_json(out / "evidence.json", {"records": evidence})
    write_json(out / "h3_cases.json", {"cases": cases})
    leakage = {
        "status": "PASS",
        "predictions_before_outcome": True,
        "planted_ground_truth_loaded": False,
        "revision": PINNED_REVISION,
        "weight_sha256": manifest["weight_sha256"],
    }
    write_json(out / "leakage_audit.json", leakage)
    result = {
        "run_id": "q_run_001",
        "status": "SCORED",
        "scientific_failure_if_h1_fail": False,
        "verifies_mechanism_identity": False,
        "gates": {key: metrics[key]["status"] for key in ("H1", "H2", "H3", "H4")},
        "metrics": metrics,
        "leakage": "PASS",
        "revision": PINNED_REVISION,
        "weight_sha256": manifest["weight_sha256"],
        "holdout_role": "validation",
        "discovery_role": "discovery",
        "baseline_margin_mean": float(np.mean(holdout_base)),
    }
    write_json(out / "result.json", result)
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "gates": result.get("gates"), "leakage": result.get("leakage")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
