"""Execute the frozen M23-G benchmark. Does not change the protocol or the belief class.

Evaluation predictions are written before evaluation outcomes exist.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_m23_g_protocol import CELLS, catalog
from src.cognitive_self_model.m23.belief import belief_from_observations, update_belief
from src.cognitive_self_model.m23.m22_reuse import (
    logit_margin,
    make_resid_hook,
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)
from src.cognitive_self_model.m23.stats import paired_mean_ci

D1_SHA = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
D2_SHA = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
SWAP = {
    "M22.1-D1-L0": "M22.1-D2-L0",
    "M22.1-D2-L0": "M22.1-D1-L0",
    "M22.1-D1-L8": "M22.1-D2-L8",
    "M22.1-D2-L8": "M22.1-D1-L8",
    "M22.1-D1-L15": "M22.1-D2-L15",
    "M22.1-D2-L15": "M22.1-D1-L15",
}
METHODS = ("updated", "no_update", "constant_effect", "outcome_only", "shuffled")


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log = path.parent / "order_log.jsonl"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": str(path.relative_to(ROOT)), "utc": stamp, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}) + "\n")
    return stamp


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _prompts(partition: str) -> list[dict[str, str]]:
    return sorted((row for row in catalog() if row["partition"] == partition), key=lambda row: row["prompt_id"])


def _directions() -> dict[str, Any]:
    d1 = primary_direction(896, 22101)
    d2 = orthogonal_direction(896, 22103, d1)
    if vector_sha256(d1) != D1_SHA or vector_sha256(d2) != D2_SHA:
        raise SystemExit("direction hash does not match the existing M22.1 vectors")
    return {"D1": d1, "D2": d2}


def _forward(model, runner, text: str, positive_id: int, negative_id: int, hook: dict[str, Any] | None):
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    recorder: dict[str, Any] = {}
    with torch.no_grad():
        if hook is None:
            logits = runner._as_logits(model(tokens))
            return logit_margin(logits, positive_id, negative_id), recorder
        direction = torch.tensor(hook["vector"], dtype=torch.float32)
        hook_fn = make_resid_hook(1.0, direction, recorder)
        hook_name = f"blocks.{int(hook['layer'])}.hook_resid_post"
        logits = runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)]))
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("hook did not fire once")
    if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
        raise RuntimeError("hook did not stay on the last token")
    return logit_margin(logits, positive_id, negative_id), recorder


def _measure(model, runner, tokens, directions, prompts) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        baseline, _empty = _forward(
            model, runner, prompt["text"], tokens["positive_token_id"], tokens["negative_token_id"], None
        )
        for cell in CELLS:
            intervened, recorder = _forward(
                model,
                runner,
                prompt["text"],
                tokens["positive_token_id"],
                tokens["negative_token_id"],
                {"layer": cell["layer"], "vector": directions[cell["direction_id"]]},
            )
            rows.append(
                {
                    "intervention_id": cell["intervention_id"],
                    "layer": cell["layer"],
                    "direction_id": cell["direction_id"],
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt["family"],
                    "partition": prompt["partition"],
                    "alpha": 1.0,
                    "baseline_output": float(baseline),
                    "intervened_output": float(intervened),
                    "observed_effect": float(intervened - baseline),
                    "pre_dot": float(recorder["pre_dot"]),
                }
            )
        print(f"{prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["intervention_id"], row["prompt_id"]): row for row in rows}


def _prediction_row(method: str, cell: dict[str, Any], prompt: dict[str, str], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "intervention_id": cell["intervention_id"],
        "prompt_id": prompt["prompt_id"],
        "alpha": 1.0,
        "predicted_effect": prediction["predicted_effect"],
        "predicted_direction": prediction["predicted_direction"],
        "uncertainty": prediction["uncertainty"],
        "model_version": prediction["model_version"],
        "evidence_version": prediction["evidence_version"],
        "outcome_present": False,
    }


def _scalar_prediction(value: float) -> dict[str, Any]:
    return {
        "predicted_effect": float(value),
        "predicted_direction": _sign(value),
        "uncertainty": None,
        "model_version": None,
        "evidence_version": None,
    }


def _apply(belief, outcomes: dict[tuple[str, str], dict[str, Any]], source_id: str, prompts, prefix: str):
    current = belief
    history = []
    for prompt in prompts:
        evidence_id = f"{prefix}{source_id}:{prompt['prompt_id']}"
        observed = float(outcomes[(source_id, prompt["prompt_id"])]["observed_effect"])
        current = update_belief(current, observed, evidence_id)
        history.append(current.update_history[-1])
    return current, history


def _mae(rows: list[dict[str, Any]], method: str) -> float:
    errors = [abs(float(row["observed_effect"]) - float(row["predictions"][method]["predicted_effect"])) for row in rows]
    return _mean(errors)


def _paired(rows: list[dict[str, Any]], left: str, right: str) -> list[float]:
    return [
        abs(float(row["observed_effect"]) - float(row["predictions"][left]["predicted_effect"]))
        - abs(float(row["observed_effect"]) - float(row["predictions"][right]["predicted_effect"]))
        for row in rows
    ]


def _sign_stats(rows: list[dict[str, Any]], method: str) -> dict[str, int]:
    comparable = 0
    agree = 0
    zero = 0
    for row in rows:
        predicted = int(row["predictions"][method]["predicted_direction"])
        observed = _sign(float(row["observed_effect"]))
        if predicted == 0 or observed == 0:
            zero += 1
            continue
        comparable += 1
        if predicted == observed:
            agree += 1
    return {"n_agree": agree, "n_comparable": comparable, "n_zero_sign": zero}


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    payload = {}
    for name in sorted(grouped):
        group = grouped[name]
        differences = _paired(group, "no_update", "updated")
        interval = paired_mean_ci(differences)
        payload[name] = {
            "n": len(group),
            "mae_updated": _mae(group, "updated"),
            "mae_no_update": _mae(group, "no_update"),
            "mae_constant_effect": _mae(group, "constant_effect"),
            "mae_outcome_only": _mae(group, "outcome_only"),
            "mae_shuffled": _mae(group, "shuffled"),
            "paired_mean": interval["mean"],
            "interval_low": interval["low"],
            "interval_high": interval["high"],
            "interval_class": interval["class"],
            "sign_updated": _sign_stats(group, "updated"),
            "secondary_not_used_for_verdict": True,
        }
    return payload


def _verdict(primary_class: str, shuffled_class: str) -> str:
    if primary_class == "CI_POSITIVE" and shuffled_class == "CI_POSITIVE":
        return "BASELINE_MATCH"
    if primary_class == "CI_POSITIVE":
        return "UPDATE_IMPROVES"
    if primary_class == "CI_NEGATIVE":
        return "UPDATE_HURTS"
    return "INCONCLUSIVE"


def main() -> None:
    raw = ROOT / "reports" / "m23_g_raw"
    if (raw / "evaluation_outcomes.json").exists():
        raise SystemExit("M23-G evaluation outcomes already exist")
    if raw.exists() and any(raw.iterdir()):
        raise SystemExit("partial M23-G raw files exist; refusing to mix runs")
    protocol_hash = hashlib.sha256((ROOT / "reports" / "M23_G_PREREGISTRATION.md").read_bytes()).hexdigest()
    runner = _runner()
    model, _device = runner.load_model()
    if int(model.cfg.d_model) != 896:
        raise SystemExit("model dimension changed")
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise SystemExit("token ids changed")
    directions = _directions()
    cells = {cell["intervention_id"]: cell for cell in CELLS}

    train_prompts = _prompts("train")
    validation_prompts = _prompts("validation")
    evaluation_prompts = _prompts("evaluation")
    if {row["prompt_id"] for row in evaluation_prompts} & {row["prompt_id"] for row in train_prompts + validation_prompts}:
        raise SystemExit("evaluation prompts overlap train or validation")

    train_rows = _measure(model, runner, tokens, directions, train_prompts)
    train_index = _index(train_rows)
    _write(raw / "train_outcomes.json", {"rows": train_rows})
    initial = {}
    initial_snapshot = {}
    for cell in CELLS:
        observations = [float(train_index[(cell["intervention_id"], prompt["prompt_id"])]["observed_effect"]) for prompt in train_prompts]
        belief = belief_from_observations(
            cell["intervention_id"],
            "last_token_additive",
            observations,
            validity_scope={"status": "IN_SCOPE", "hook": f"blocks.{cell['layer']}.hook_resid_post", "alpha": 1.0},
        )
        initial[cell["intervention_id"]] = belief
        prediction = belief.predict(1.0)
        initial_snapshot[cell["intervention_id"]] = {
            "predicted_effect": prediction["predicted_effect"],
            "predicted_direction": prediction["predicted_direction"],
            "uncertainty": prediction["uncertainty"],
            "model_version": prediction["model_version"],
            "n_train": len(observations),
            "evidence_ids": [f"{cell['intervention_id']}:{prompt['prompt_id']}" for prompt in train_prompts],
        }
    _write(raw / "initial_beliefs.json", initial_snapshot)

    validation_prediction_path = raw / "validation_predictions.json"
    validation_outcome_path = raw / "validation_outcomes.json"
    if validation_outcome_path.exists():
        raise SystemExit("validation outcomes exist before predictions")
    validation_predictions = []
    for cell in CELLS:
        prediction = initial[cell["intervention_id"]].predict(1.0)
        for prompt in validation_prompts:
            validation_predictions.append(_prediction_row("initial", cell, prompt, prediction))
    if any(row.get("outcome_present") for row in validation_predictions):
        raise SystemExit("validation prediction carries an outcome")
    validation_prediction_stamp = _write(validation_prediction_path, {"rows": validation_predictions})
    if validation_outcome_path.exists():
        raise SystemExit("validation outcomes appeared before the forward")

    validation_rows = _measure(model, runner, tokens, directions, validation_prompts)
    validation_index = _index(validation_rows)
    validation_outcome_stamp = _write(validation_outcome_path, {"rows": validation_rows})
    validation_order_ok = validation_prediction_path.stat().st_mtime_ns <= validation_outcome_path.stat().st_mtime_ns

    updated = {}
    histories = {}
    for cell in CELLS:
        before_effect = float(initial[cell["intervention_id"]].predicted_effect)
        before_version = int(initial[cell["intervention_id"]].version)
        current, history = _apply(initial[cell["intervention_id"]], validation_index, cell["intervention_id"], validation_prompts, "")
        if float(initial[cell["intervention_id"]].predicted_effect) != before_effect or int(initial[cell["intervention_id"]].version) != before_version:
            raise SystemExit("update mutated the original belief")
        updated[cell["intervention_id"]] = current
        histories[cell["intervention_id"]] = history
    _write(raw / "update_history.json", histories)

    evaluation_prediction_path = raw / "evaluation_predictions.json"
    evaluation_outcome_path = raw / "evaluation_outcomes.json"
    if evaluation_outcome_path.exists():
        raise SystemExit("evaluation outcomes exist before predictions")
    train_effects = [float(row["observed_effect"]) for row in train_rows]
    if len(train_effects) != 72:
        raise SystemExit("train evidence count is not 72")
    constant = _mean(train_effects)
    shuffled = {}
    shuffled_history = {}
    for cell in CELLS:
        source = SWAP[cell["intervention_id"]]
        belief, history = _apply(initial[cell["intervention_id"]], validation_index, source, validation_prompts, "shuffle-")
        if float(initial[cell["intervention_id"]].predicted_effect) != float(initial_snapshot[cell["intervention_id"]]["predicted_effect"]):
            raise SystemExit("shuffle mutated the original belief")
        shuffled[cell["intervention_id"]] = belief
        shuffled_history[cell["intervention_id"]] = history
    evaluation_predictions = []
    for cell in CELLS:
        methods = {
            "updated": updated[cell["intervention_id"]].predict(1.0),
            "no_update": initial[cell["intervention_id"]].predict(1.0),
            "constant_effect": _scalar_prediction(constant),
            "outcome_only": _scalar_prediction(constant),
            "shuffled": shuffled[cell["intervention_id"]].predict(1.0),
        }
        for prompt in evaluation_prompts:
            for method in METHODS:
                evaluation_predictions.append(_prediction_row(method, cell, prompt, methods[method]))
    if len(evaluation_predictions) != 360:
        raise SystemExit("evaluation prediction count is not 360")
    if any("observed_effect" in row or row["outcome_present"] for row in evaluation_predictions):
        raise SystemExit("evaluation prediction contains an outcome")
    evaluation_prediction_stamp = _write(evaluation_prediction_path, {"rows": evaluation_predictions, "constant_train_mean": constant})
    if evaluation_outcome_path.exists():
        raise SystemExit("evaluation outcomes appeared before the forward")

    evaluation_rows = _measure(model, runner, tokens, directions, evaluation_prompts)
    evaluation_outcome_stamp = _write(evaluation_outcome_path, {"rows": evaluation_rows})
    evaluation_order_ok = evaluation_prediction_path.stat().st_mtime_ns <= evaluation_outcome_path.stat().st_mtime_ns
    _write(raw / "shuffled_update_history.json", shuffled_history)

    by_key = {(row["method"], row["intervention_id"], row["prompt_id"]): row for row in evaluation_predictions}
    scored = []
    for row in sorted(evaluation_rows, key=lambda item: (item["intervention_id"], item["prompt_id"])):
        predictions = {}
        for method in METHODS:
            saved = by_key[(method, row["intervention_id"], row["prompt_id"])]
            predictions[method] = {
                "predicted_effect": saved["predicted_effect"],
                "predicted_direction": saved["predicted_direction"],
                "uncertainty": saved["uncertainty"],
                "model_version": saved["model_version"],
            }
        scored.append({**row, "predictions": predictions})
    differences = _paired(scored, "no_update", "updated")
    primary = paired_mean_ci(differences)
    shuffled_differences = _paired(scored, "no_update", "shuffled")
    shuffled_interval = paired_mean_ci(shuffled_differences)
    verdict = _verdict(str(primary["class"]), str(shuffled_interval["class"]))
    contradiction = {"n_updates": 0, "sign_mismatch": 0, "outside_interval": 0, "critical": 0, "reasons": {}}
    versions = {}
    for cell_id, history in histories.items():
        versions[cell_id] = {
            "initial_version": 1,
            "final_version": history[-1]["model_version_after"],
            "final_predicted_effect": history[-1]["belief_after"]["predicted_effect"],
            "final_uncertainty": history[-1]["belief_after"]["uncertainty"],
            "final_status": history[-1]["belief_after"]["validity_scope"]["status"],
            "inflation": history[-1]["belief_after"]["inflation"],
        }
        for record in history:
            contradiction["n_updates"] += 1
            evidence = record["evidence"]
            contradiction["sign_mismatch"] += int(bool(evidence["sign_mismatch"]))
            contradiction["outside_interval"] += int(bool(evidence["outside_interval"]))
            contradiction["critical"] += int(bool(evidence["critical"]))
            reason = record["update_reason"]
            contradiction["reasons"][reason] = contradiction["reasons"].get(reason, 0) + 1
    leakage = {
        "validation_predictions_before_outcomes": validation_order_ok,
        "evaluation_predictions_before_outcomes": evaluation_order_ok,
        "evaluation_predictions_lack_outcomes": True,
        "original_belief_unchanged": True,
        "constant_uses_train_only": True,
        "shuffled_is_within_layer_direction_swap": set(SWAP) == {cell["intervention_id"] for cell in CELLS},
        "evaluation_disjoint_from_train_and_validation": True,
        "alpha_frozen_at_plus_one": all(row["alpha"] == 1.0 for row in evaluation_rows),
        "six_cells_retained": [cell["intervention_id"] for cell in CELLS],
    }
    leakage_ok = all(value is True or isinstance(value, list) for value in leakage.values())
    payload = {
        "protocol_sha256": protocol_hash,
        "protocol_status_preserved": "READY_FOR_EXECUTION",
        "verdict": verdict,
        "verdict_is_not_a_self_model_claim": True,
        "alpha": 1.0,
        "cells": [cell["intervention_id"] for cell in CELLS],
        "n_evaluation_rows": len(scored),
        "primary": {
            "definition": "abs(no_update error) - abs(updated error)",
            "mean": primary["mean"],
            "low": primary["low"],
            "high": primary["high"],
            "class": primary["class"],
            "differences": differences,
        },
        "mae": {method: _mae(scored, method) for method in METHODS},
        "sign": {method: _sign_stats(scored, method) for method in METHODS},
        "shuffled_vs_no_update": {
            "mean": shuffled_interval["mean"],
            "low": shuffled_interval["low"],
            "high": shuffled_interval["high"],
            "class": shuffled_interval["class"],
        },
        "constant_equals_outcome_only": leakage["constant_uses_train_only"] and _mae(scored, "constant_effect") == _mae(scored, "outcome_only"),
        "constant_train_mean": constant,
        "contradiction": contradiction,
        "belief_versions": versions,
        "per_cell": _group(scored, "intervention_id"),
        "per_family": _group(scored, "family"),
        "timestamps": {
            "validation_predictions_utc": validation_prediction_stamp,
            "validation_outcomes_utc": validation_outcome_stamp,
            "evaluation_predictions_utc": evaluation_prediction_stamp,
            "evaluation_outcomes_utc": evaluation_outcome_stamp,
        },
        "leakage": leakage,
        "leakage_ok": leakage_ok,
    }
    _write(ROOT / "reports" / "M23_G_RESULTS.json", payload)
    fieldnames = [
        "episode_id",
        "partition",
        "intervention_id",
        "layer",
        "direction_id",
        "prompt_id",
        "family",
        "alpha",
        "observed_effect",
        "baseline_output",
        "intervened_output",
        "pre_dot",
        "method",
        "predicted_effect",
        "predicted_direction",
        "uncertainty",
        "model_version",
    ]
    episode_rows = []
    prediction_lookup = {
        ("validation", row["intervention_id"], row["prompt_id"]): row for row in validation_predictions
    }
    for partition_rows, partition in (
        (train_rows, "train"),
        (validation_rows, "validation"),
        (evaluation_rows, "evaluation"),
    ):
        for row in partition_rows:
            base = {
                "episode_id": f"{partition}:{row['intervention_id']}:{row['prompt_id']}",
                "partition": partition,
                "intervention_id": row["intervention_id"],
                "layer": row["layer"],
                "direction_id": row["direction_id"],
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "alpha": row["alpha"],
                "observed_effect": row["observed_effect"],
                "baseline_output": row["baseline_output"],
                "intervened_output": row["intervened_output"],
                "pre_dot": row["pre_dot"],
            }
            if partition == "train":
                episode_rows.append({**base, "method": "", "predicted_effect": "", "predicted_direction": "", "uncertainty": "", "model_version": ""})
            elif partition == "validation":
                saved = prediction_lookup[(partition, row["intervention_id"], row["prompt_id"])]
                episode_rows.append(
                    {
                        **base,
                        "method": "initial",
                        "predicted_effect": saved["predicted_effect"],
                        "predicted_direction": saved["predicted_direction"],
                        "uncertainty": saved["uncertainty"],
                        "model_version": saved["model_version"],
                    }
                )
            else:
                for method in METHODS:
                    saved = by_key[(method, row["intervention_id"], row["prompt_id"])]
                    episode_rows.append(
                        {
                            **base,
                            "episode_id": f"{partition}:{row['intervention_id']}:{row['prompt_id']}:{method}",
                            "method": method,
                            "predicted_effect": saved["predicted_effect"],
                            "predicted_direction": saved["predicted_direction"],
                            "uncertainty": "" if saved["uncertainty"] is None else saved["uncertainty"],
                            "model_version": "" if saved["model_version"] is None else saved["model_version"],
                        }
                    )
    with (ROOT / "reports" / "M23_G_EPISODES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)
    print(verdict, primary["class"], primary["low"], primary["high"], flush=True)


if __name__ == "__main__":
    main()
