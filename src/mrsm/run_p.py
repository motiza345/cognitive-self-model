"""One preregistered P scientific run.

Prediction is serialized before the holdout set is executed. Scoring loads the
planted ground-truth file only after that barrier.
"""

from __future__ import annotations

import json
import platform
import subprocess
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

import mrsm_p_v1  # noqa: E402

from src.mrsm import HEADS
from src.mrsm.baselines import predictions as baseline_predictions
from src.mrsm.evaluators import intervention_ids, score_h1, score_h2, score_h3, score_h4
from src.mrsm.evidence import Evidence, residual_of
from src.mrsm.interventions import discovery_effects, execute, parse_head
from src.mrsm.leakage import audit_predictions, audit_sources, audit_state
from src.mrsm.p_data import dataset, sha256_file, write_json
from src.mrsm.self_model import SelfModelCore
from src.mrsm.transforms import frozen_parameters

PREREG = _ROOT / "configs" / "mrsm_prereg.yaml"
GROUND_TRUTH = _ROOT / "artifacts" / "mrsm" / "planted_ground_truth.json"
MANIFEST = _ROOT / "artifacts" / "mrsm" / "holdout_manifest.json"
CHECKPOINT_MANIFEST = _ROOT / "artifacts" / "mrsm" / "construction" / "checkpoint_manifest.json"
CONSTRUCTION_RECORD = _ROOT / "artifacts" / "mrsm" / "construction" / "construction_record.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(args: list[str]) -> str:
    out = subprocess.run(["git", *args], cwd=_ROOT, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _environment() -> dict[str, Any]:
    import importlib.metadata

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "ABSENT"

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "transformer_lens": version("transformer-lens"),
        "transformers": version("transformers"),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
    }


def _load_model(prereg: dict[str, Any]):
    binding = _load_yaml(CHECKPOINT_MANIFEST) if CHECKPOINT_MANIFEST.is_file() else None
    if not isinstance(binding, dict) or not binding.get("sha256"):
        raise RuntimeError("checkpoint manifest is missing; construction is not frozen")
    path = _ROOT / prereg["model"]["arm_p_checkpoint"]
    digest = sha256_file(path)
    if digest != binding["sha256"]:
        raise RuntimeError("checkpoint hash does not match the construction manifest")
    cfg = mrsm_p_v1.Config()
    mrsm_p_v1.seed_all(int(prereg["seeds"]["construction_model_seed"]))
    model = mrsm_p_v1.build_model(cfg)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, digest


def _fit(model, prereg: dict[str, Any], run_id: str) -> tuple[SelfModelCore, dict[str, Any], list[Evidence]]:
    discovery = prereg["splits"]["discovery"]
    ds = dataset(int(discovery["n"]), int(discovery["seed"]))
    singles, joints = discovery_effects(model, ds)
    evidence: list[Evidence] = []
    stamp = _now()
    for key, value in {**{f"single:{name}": effect for name, effect in singles.items()}, **{f"pair:{name}": effect for name, effect in joints.items()}}.items():
        evidence.append(
            Evidence(
                observation_id=f"discovery-{key}",
                intervention_id=key,
                prediction=None,
                actual_outcome=value,
                residual=None,
                context={"split": "discovery"},
                scope=prereg["scope"],
                evidence_type="discovery_observation",
                timestamp=stamp,
                run_id=run_id,
            )
        )
    core = SelfModelCore(prereg["scope"])
    core.update(
        singles,
        joints,
        [item.observation_id for item in evidence],
        min_abs_interaction=float(prereg["identifiability"]["min_abs_interaction"]),
        min_relative_gap=float(prereg["identifiability"]["min_relative_gap_vs_second"]),
    )
    core.freeze()
    fit_inputs = {
        "head_effects": singles,
        "joint_effects": joints,
        "expected_heads": list(HEADS),
        "discovery_ids": [f"discovery-{seed}-{i}" for i in range(int(discovery["n"])) for seed in [int(discovery["seed"])]],
        "holdout_outcomes_present": False,
    }
    return core, fit_inputs, evidence


def _predict(core: SelfModelCore, prereg: dict[str, Any], run_id: str) -> dict[str, Any]:
    rows = []
    stamp = _now()
    for intervention_id in intervention_ids():
        pred = core.predict(intervention_id)
        rows.append(
            {
                "intervention_id": intervention_id,
                "prediction": pred,
                "actual_outcome": None,
                "timestamp": stamp,
                "order": "before_outcome",
                "mechanism_abstraction": None if core.mechanism is None else core.mechanism.get("abstraction", "NOT_IDENTIFIABLE"),
                "scope": core.scope,
                "uncertainty": core.uncertainty,
            }
        )
    return {
        "phase": "before_outcome",
        "run_id": run_id,
        "seed": prereg["seeds"]["holdout_seed"],
        "rows": rows,
        "self_model": core.explain(),
        "baselines": baseline_predictions(core.effects),
        "h3_parameters": frozen_parameters(),
    }


def _h3_cases(core: SelfModelCore, predictions: dict[str, Any]) -> list[dict[str, Any]]:
    original_abstraction = "NOT_IDENTIFIABLE"
    if core.mechanism and core.mechanism.get("status") == "IDENTIFIED":
        original_abstraction = str(core.mechanism["abstraction"])
    cases = []
    by_id = {row["intervention_id"]: row for row in predictions["rows"]}
    for transform_id in ("T1", "T2", "T3"):
        reading = core.read_transformed(transform_id)
        for intervention_id, row in by_id.items():
            kind, name = intervention_id.split(":", 1)
            if kind == "single":
                transformed = float(reading["recovered_effects"][name])
            else:
                left, right = name.split("+")
                transformed = (
                    float(reading["recovered_effects"][left])
                    + float(reading["recovered_effects"][right])
                    + float(core.interactions[name])
                )
            cases.append(
                {
                    "intervention_id": intervention_id,
                    "transformation_id": transform_id,
                    "original_prediction": row["prediction"],
                    "transformed_prediction": transformed,
                    "transformed_coordinate": reading["transformed_coordinate"],
                    "mechanism_abstraction": reading["mechanism_abstraction"],
                    "original_abstraction": original_abstraction,
                    "scope": reading["scope"],
                    "original_scope": core.scope,
                    "uncertainty": reading["uncertainty"],
                }
            )
    return cases


def _execute_holdout(model, prereg: dict[str, Any]) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    holdout = prereg["splits"]["holdout"]
    ds = dataset(int(holdout["n"]), int(holdout["seed"]))
    raw = ds.tokens.numpy().tobytes() + ds.targets.numpy().tobytes()
    import hashlib

    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["hash"] or manifest["sample_ids"] != [f"holdout-{holdout['seed']}-{i}" for i in range(int(holdout["n"]))]:
        raise RuntimeError("holdout manifest does not match the regenerated split")
    actual_margin: dict[str, list[float]] = {}
    correct_by_head: dict[str, list[int]] = {}
    for name in HEADS:
        result = execute(model, ds, "intervention", [parse_head(name)], with_representation=False)
        actual_margin[f"single:{name}"] = result["behavioral_delta"]["per_example_margin_drop"]
        correct_by_head[name] = result["observable_outcome"]["correct"]
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            result = execute(
                model, ds, "counterfactual", [parse_head(left), parse_head(right)], with_representation=False
            )
            actual_margin[f"pair:{left}+{right}"] = result["behavioral_delta"]["per_example_margin_drop"]
    return actual_margin, correct_by_head, manifest


def run(output_dir: Path | None = None) -> dict[str, Any]:
    prereg = _load_yaml(PREREG)
    if prereg.get("status") != "PREREGISTERED":
        raise RuntimeError("preregistration is not frozen")
    if not CONSTRUCTION_RECORD.is_file():
        raise RuntimeError("P construction record is missing")
    construction = json.loads(CONSTRUCTION_RECORD.read_text(encoding="utf-8"))
    if construction.get("budget_outcome") != "DONE":
        raise RuntimeError("P construction is not DONE; DEC-011 forbids starting MRSM")
    run_id = "p_run_001"
    out = output_dir if output_dir is not None else _ROOT / "artifacts" / "mrsm" / run_id
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} already exists; refuse to overwrite a scientific run")
    out.mkdir(parents=True)
    model, checkpoint_sha = _load_model(prereg)
    core, fit_inputs, discovery_evidence = _fit(model, prereg, run_id)
    predictions = _predict(core, prereg, run_id)
    write_json(out / "predictions.json", predictions)
    write_json(out / "self_model_state.json", core.explain())
    write_json(out / "fit_inputs.json", fit_inputs)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_audit = audit_sources()
    state_audit = audit_state(core.explain(), fit_inputs, manifest["sample_ids"])
    prediction_audit = audit_predictions(predictions)
    h3_params_ok = predictions["h3_parameters"]["T3"]["matrix_sha256"] == prereg["h3"]["t3"]["matrix_sha256"]
    leakage = {
        "source": source_audit,
        "state": state_audit,
        "predictions": prediction_audit,
        "h3_parameters_match_prereg": h3_params_ok,
        "ground_truth_loaded_before_prediction": False,
        "holdout_outcomes_loaded_before_prediction": False,
        "status": "PASS"
        if source_audit["pass"] and state_audit["pass"] and prediction_audit["pass"] and h3_params_ok
        else "FAIL",
    }
    write_json(out / "leakage_audit.json", leakage)
    if leakage["status"] != "PASS":
        result = {"run_id": run_id, "status": "INVALID", "reason": "leakage", "leakage": leakage}
        write_json(out / "result.json", result)
        return result

    actual, correct_by_head, manifest = _execute_holdout(model, prereg)
    leakage["holdout_outcomes_loaded_before_prediction"] = False
    write_json(out / "outcomes.json", {"margin_drop": actual, "correct_by_head": correct_by_head})
    self_pred = {row["intervention_id"]: float(row["prediction"]) for row in predictions["rows"]}
    actual_mean = {key: float(np.mean(values)) for key, values in actual.items()}
    evidence = [item.to_dict() for item in discovery_evidence]
    stamp = _now()
    for key, pred in self_pred.items():
        evidence.append(
            Evidence(
                observation_id=f"holdout-{key}",
                intervention_id=key,
                prediction=pred,
                actual_outcome=actual_mean[key],
                residual=residual_of(pred, actual_mean[key]),
                context={"split": "holdout"},
                scope=prereg["scope"],
                evidence_type="holdout_scored",
                timestamp=stamp,
                run_id=run_id,
            ).to_dict()
        )
    write_json(out / "evidence.json", {"records": evidence})
    falsifications = []
    tolerance = max(1.0, 2.0 * float(core.uncertainty or 0.0))
    for record in evidence:
        if record["evidence_type"] != "holdout_scored":
            continue
        item = Evidence(**record)
        falsifications.append(core.falsify(item, tolerance=tolerance))
    write_json(out / "falsification.json", {"tolerance": tolerance, "records": falsifications, "mechanism_changed": False})

    ground_truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    leakage["ground_truth_loaded_at_scoring"] = True
    ordered = tuple(ground_truth["ordered_edge"])
    h1 = score_h1(
        core.mechanism,
        {name: actual_mean[f"single:{name}"] for name in HEADS},
        ground_truth_ordered=(str(ordered[0]), str(ordered[1])),
        sign_floor=float(prereg["statistical_procedure"]["sign_zero_floor"]),
    )
    h2 = score_h2(
        self_pred,
        predictions["baselines"],
        actual,
        resamples=int(prereg["statistical_procedure"]["resamples"]),
        seed=int(prereg["statistical_procedure"]["seed"]),
        sign_floor=float(prereg["statistical_procedure"]["sign_zero_floor"]),
    )
    cases = _h3_cases(core, predictions)
    write_json(out / "h3_cases.json", {"cases": cases})
    h3 = score_h3(cases, sign_floor=float(prereg["statistical_procedure"]["sign_zero_floor"]))
    chosen = core.choose_ablation()
    blind = prereg["h4"]["consumption_ablation"]["action"]
    h4 = score_h4(
        correct_by_head[chosen],
        correct_by_head[blind],
        resamples=int(prereg["statistical_procedure"]["resamples"]),
        seed=int(prereg["statistical_procedure"]["seed"]),
    )
    h4["full_action"] = chosen
    h4["consumption_ablation_action"] = blind
    metrics = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "baselines": h2["baseline_mae"]}
    write_json(out / "metrics.json", metrics)
    env = _environment()
    env["checkpoint_sha256"] = checkpoint_sha
    env["holdout_hash"] = manifest["hash"]
    env["preregistration_id"] = prereg["preregistration_id"]
    write_json(out / "environment.json", env)
    artifact_hashes = {
        name: sha256_file(out / name)
        for name in (
            "predictions.json",
            "outcomes.json",
            "self_model_state.json",
            "evidence.json",
            "metrics.json",
            "leakage_audit.json",
        )
    }
    result = {
        "run_id": run_id,
        "status": "SCORED",
        "seed": prereg["seeds"]["holdout_seed"],
        "checkpoint_sha256": checkpoint_sha,
        "holdout_hash": manifest["hash"],
        "gates": {key: metrics[key]["status"] for key in ("H1", "H2", "H3", "H4")},
        "metrics": metrics,
        "leakage": leakage["status"],
        "artifact_hashes": artifact_hashes,
        "git_commit": env["git_commit"],
        "falsification_changed_mechanism": False,
    }
    write_json(out / "result.json", result)
    write_json(out / "leakage_audit.json", leakage)
    return result


def main() -> int:
    result = run()
    print(json.dumps({"status": result["status"], "gates": result.get("gates"), "leakage": result.get("leakage")}, indent=2))
    return 0 if result["status"] in {"SCORED", "INVALID"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
