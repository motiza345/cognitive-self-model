"""Execute the frozen project feasibility gate.

The catalog is read from the frozen module. Held-out predictions are written
before held-out outcomes. The candidate is g, with no fitted slope.
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

from scripts.feasibility_gate_catalog import PARTITIONS, catalog
from scripts.validate_m23_g_protocol import catalog as g_catalog
from scripts.validate_post_m23_single_measurement import catalog as s_catalog
from scripts.validate_project_feasibility_gate import decide
from src.cognitive_self_model.m23.m22_reuse import (
    frozen_prompts,
    logit_margin,
    make_resid_hook,
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)
from src.cognitive_self_model.m23.stats import clopper_pearson, paired_mean_ci

D1_SHA = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
D2_SHA = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
LAYERS = (0, 8, 15)
FAMILY = {
    0: "M22.1-D1-L0",
    8: "M22.1-D1-L8",
    15: "M22.1-D1-L15",
}
RAW = ROOT / "reports" / "feasibility_gate_raw"
ORDER_LOG = RAW / "order_log.jsonl"


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stamp = _stamp()
    with ORDER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": str(path.relative_to(ROOT)), "utc": stamp, "sha256": digest}) + "\n")
    return stamp


def _prompts(partition: str) -> list[dict[str, str]]:
    return sorted((row for row in catalog() if row["partition"] == partition), key=lambda row: row["prompt_id"])


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _agreement(predicted: float, observed: float) -> bool | None:
    predicted_sign = _sign(predicted)
    observed_sign = _sign(observed)
    if predicted_sign == 0 or observed_sign == 0:
        return None
    return predicted_sign == observed_sign


def _assert_catalog_frozen() -> None:
    text = (ROOT / "reports" / "PROJECT_FEASIBILITY_GATE_CATALOG.md").read_text(encoding="utf-8")
    rows = catalog()
    if len(rows) != 48:
        raise RuntimeError("catalog is not 48 prompts")
    old_texts = {record.text for record in frozen_prompts()}
    old_texts.update(row["text"] for row in g_catalog())
    old_texts.update(row["text"] for row in s_catalog())
    old_ids = {record.prompt_id for record in frozen_prompts()}
    old_ids.update(row["prompt_id"] for row in g_catalog())
    old_ids.update(row["prompt_id"] for row in s_catalog())
    by_partition: dict[str, list[dict[str, str]]] = {name: [] for name in PARTITIONS}
    for row in rows:
        if row["prompt_id"] not in text or row["text"] not in text:
            raise RuntimeError(f"catalog markdown missing {row['prompt_id']}")
        if row["text"] in old_texts or row["prompt_id"] in old_ids:
            raise RuntimeError(f"catalog reuses an old prompt: {row['prompt_id']}")
        if row["prompt_id"].startswith("g-") or row["prompt_id"].startswith("s-"):
            raise RuntimeError("contaminated id in the new catalog")
        by_partition[row["partition"]].append(row)
    for name, members in by_partition.items():
        if len(members) != 12:
            raise RuntimeError(f"{name} does not have 12 prompts")
        if {row["family"] for row in members} != {"completion", "syntax", "instruction"}:
            raise RuntimeError(f"{name} is missing a surface family")


def _derivatives(model, runner, text: str, positive_id: int, negative_id: int, d1, d2) -> dict[int, dict[str, float]]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    saved: dict[int, torch.Tensor] = {}

    def _hook(layer: int):
        def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
            if residual.requires_grad:
                residual.retain_grad()
            saved[layer] = residual
            return residual

        return hook_fn

    hooks = [(f"blocks.{layer}.hook_resid_post", _hook(layer)) for layer in LAYERS]
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=hooks))
        margin = logits[0, -1, positive_id] - logits[0, -1, negative_id]
        if not margin.requires_grad:
            raise RuntimeError("margin is not connected to the residual graph")
        margin.backward()
    vector_d1 = torch.tensor(d1, dtype=torch.float32)
    vector_d2 = torch.tensor(d2, dtype=torch.float32)
    found: dict[int, dict[str, float]] = {}
    for layer in LAYERS:
        residual = saved.get(layer)
        if residual is None or residual.grad is None:
            raise RuntimeError(f"no gradient at layer {layer}")
        gradient = residual.grad[0, -1].detach().float().cpu()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite gradient at layer {layer}")
        found[layer] = {
            "g": float(torch.dot(gradient, vector_d1)),
            "g_orth": float(torch.dot(gradient, vector_d2)),
        }
    model.zero_grad(set_to_none=True)
    return found


def _effect(model, runner, text: str, positive_id: int, negative_id: int, direction, layer: int) -> dict[str, float]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    hook_name = f"blocks.{layer}.hook_resid_post"
    with torch.no_grad():
        baseline = logit_margin(runner._as_logits(model(tokens)), positive_id, negative_id)
        recorder: dict[str, Any] = {}
        vector = torch.tensor(direction, dtype=torch.float32)
        hook_fn = make_resid_hook(1.0, vector, recorder)
        intervened = logit_margin(
            runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])),
            positive_id,
            negative_id,
        )
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("intervention hook did not fire once")
    if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not stay on the last token")
    return {
        "baseline_output": float(baseline),
        "intervened_output": float(intervened),
        "observed_effect": float(intervened - baseline),
    }


def _derivative_rows(model, runner, tokens, d1, d2, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        derivatives = _derivatives(
            model,
            runner,
            prompt["text"],
            tokens["positive_token_id"],
            tokens["negative_token_id"],
            d1,
            d2,
        )
        for layer in LAYERS:
            rows.append(
                {
                    "intervention_id": FAMILY[layer],
                    "layer": layer,
                    "direction_id": "D1",
                    "alpha": 1.0,
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt["family"],
                    "partition": prompt["partition"],
                    "g": derivatives[layer]["g"],
                    "g_orth": derivatives[layer]["g_orth"],
                }
            )
        print(f"g {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _outcome_rows(model, runner, tokens, d1, prompts, measurements) -> list[dict[str, Any]]:
    measured = {(row["intervention_id"], row["prompt_id"]): row for row in measurements}
    rows = []
    seen_prompts = []
    for prompt in prompts:
        if prompt["prompt_id"] not in seen_prompts:
            seen_prompts.append(prompt["prompt_id"])
    for prompt in prompts:
        for layer in LAYERS:
            effect = _effect(
                model,
                runner,
                prompt["text"],
                tokens["positive_token_id"],
                tokens["negative_token_id"],
                d1,
                layer,
            )
            key = (FAMILY[layer], prompt["prompt_id"])
            prior = measured[key]
            rows.append(
                {
                    "intervention_id": FAMILY[layer],
                    "layer": layer,
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt["family"],
                    "partition": prompt["partition"],
                    "alpha": 1.0,
                    "g": prior["g"],
                    "g_orth": prior["g_orth"],
                    "baseline_output": effect["baseline_output"],
                    "intervened_output": effect["intervened_output"],
                    "observed_effect": effect["observed_effect"],
                }
            )
        print(f"outcome {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _means(outcomes: list[dict[str, Any]]) -> dict[str, float]:
    found = {}
    for cell in FAMILY.values():
        values = [float(row["observed_effect"]) for row in outcomes if row["intervention_id"] == cell]
        if len(values) != 12:
            raise RuntimeError(f"{cell} does not have 12 train effects")
        found[cell] = sum(values) / len(values)
    return found


def _predictions(measurements: list[dict[str, Any]], means: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for row in measurements:
        mean = float(means[row["intervention_id"]])
        rows.append(
            {
                "intervention_id": row["intervention_id"],
                "layer": row["layer"],
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "partition": row["partition"],
                "alpha": 1.0,
                "g": row["g"],
                "g_orth": row["g_orth"],
                "m": mean,
                "baseline_prediction": mean,
                "candidate_prediction": row["g"],
                "control_prediction": row["g_orth"],
                "outcome_present": False,
            }
        )
    return rows


def _assert_no_effect(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for banned in ("observed_effect", "intervened_output", "baseline_output"):
        if banned in text:
            raise RuntimeError(f"{path.name} contains {banned}")


def _join(measurements, outcomes, predictions) -> list[dict[str, Any]]:
    outcome_by = {(row["intervention_id"], row["prompt_id"]): row for row in outcomes}
    prediction_by = {(row["intervention_id"], row["prompt_id"]): row for row in predictions}
    joined = []
    for row in measurements:
        key = (row["intervention_id"], row["prompt_id"])
        outcome = outcome_by[key]
        prediction = prediction_by[key]
        effect = float(outcome["observed_effect"])
        baseline = float(prediction["baseline_prediction"])
        candidate = float(prediction["candidate_prediction"])
        control = float(prediction["control_prediction"])
        baseline_error = abs(effect - baseline)
        candidate_error = abs(effect - candidate)
        control_error = abs(effect - control)
        joined.append(
            {
                "intervention_id": row["intervention_id"],
                "layer": row["layer"],
                "direction_id": "D1",
                "alpha": 1.0,
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "partition": row["partition"],
                "g": row["g"],
                "g_orth": row["g_orth"],
                "m": baseline,
                "baseline_prediction": baseline,
                "candidate_prediction": candidate,
                "control_prediction": control,
                "baseline_output": float(outcome["baseline_output"]),
                "intervened_output": float(outcome["intervened_output"]),
                "observed_effect": effect,
                "abs_error_baseline": baseline_error,
                "abs_error_candidate": candidate_error,
                "abs_error_control": control_error,
                "paired_difference": baseline_error - candidate_error,
                "control_paired_difference": baseline_error - control_error,
            }
        )
    return joined


def _prompt_scores(rows: list[dict[str, Any]], field: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["prompt_id"], []).append(float(row[field]))
    scores = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3:
            raise RuntimeError(f"{prompt_id} does not have three cells")
        scores.append(sum(values) / 3.0)
    if len(scores) != 12:
        raise RuntimeError("prompt aggregation is not 12 scores")
    return scores


def _interval(scores: list[float]) -> dict[str, Any]:
    found = paired_mean_ci(scores)
    return {
        "n_prompts": len(scores),
        "paired_mean": float(found["mean"]),
        "low": float(found["low"]),
        "high": float(found["high"]),
        "class": str(found["class"]),
        "baseline_mae": None,
        "candidate_mae": None,
    }


def _mae_pair(rows: list[dict[str, Any]], error_field: str) -> tuple[float, float]:
    baseline = sum(float(row["abs_error_baseline"]) for row in rows) / len(rows)
    other = sum(float(row[error_field]) for row in rows) / len(rows)
    return baseline, other


def _sign_report(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    cells = []
    passed = True
    for cell in FAMILY.values():
        counted = []
        for row in rows:
            if row["intervention_id"] != cell:
                continue
            agreement = _agreement(float(row[prediction_field]), float(row["observed_effect"]))
            if agreement is not None:
                counted.append(agreement)
        if not counted:
            passed = False
            cells.append({"intervention_id": cell, "n": 0, "successes": 0, "low": None, "high": None, "pass": False})
            continue
        successes = sum(1 for item in counted if item)
        low, high = clopper_pearson(successes, len(counted))
        cell_pass = low > 0.5
        passed = passed and cell_pass
        cells.append(
            {
                "intervention_id": cell,
                "n": len(counted),
                "successes": successes,
                "low": low,
                "high": high,
                "pass": cell_pass,
            }
        )
    return {"pass": passed, "cells": cells}


def _score_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = _interval(_prompt_scores(rows, "paired_difference"))
    control = _interval(_prompt_scores(rows, "control_paired_difference"))
    baseline_mae, candidate_mae = _mae_pair(rows, "abs_error_candidate")
    _, control_mae = _mae_pair(rows, "abs_error_control")
    candidate["baseline_mae"] = baseline_mae
    candidate["candidate_mae"] = candidate_mae
    control["baseline_mae"] = baseline_mae
    control["candidate_mae"] = control_mae
    return {
        "candidate": candidate,
        "control": control,
        "sign": _sign_report(rows, "candidate_prediction"),
    }


def _markdown(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    evaluation = payload["evaluation"]["candidate"]
    replication = payload["replication"]["candidate"]
    lines = [
        "# Project feasibility gate execution",
        "",
        "The gate specification remains `GATE_SPECIFIED`. This file records the run. It does not change the decision map.",
        "",
        f"Verdict: `{verdict}`",
        "",
        payload["interpretation"],
        "",
        "## Train means",
        "",
        "These are the only fitted numbers. The candidate prediction is `g`, with no slope.",
        "",
        "| cell | train mean m |",
        "| --- | ---: |",
    ]
    for cell, value in payload["train_means"].items():
        lines.append(f"| `{cell}` | {value} |")
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- baseline MAE: `{evaluation['baseline_mae']}`",
            f"- candidate MAE: `{evaluation['candidate_mae']}`",
            f"- paired mean: `{evaluation['paired_mean']}`",
            f"- interval: `[{evaluation['low']}, {evaluation['high']}]`",
            f"- class: `{evaluation['class']}`",
            f"- control class: `{payload['evaluation']['control']['class']}`",
            f"- control paired mean: `{payload['evaluation']['control']['paired_mean']}`",
            f"- sign co-criterion: `{payload['evaluation']['sign']['pass']}`",
            "",
            "## Replication",
            "",
            f"- baseline MAE: `{replication['baseline_mae']}`",
            f"- candidate MAE: `{replication['candidate_mae']}`",
            f"- paired mean: `{replication['paired_mean']}`",
            f"- interval: `[{replication['low']}, {replication['high']}]`",
            f"- class: `{replication['class']}`",
            f"- control class: `{payload['replication']['control']['class']}`",
            f"- control paired mean: `{payload['replication']['control']['paired_mean']}`",
            f"- sign co-criterion: `{payload['replication']['sign']['pass']}`",
            "",
            "## Validation",
            "",
            "Validation is not an input to the decision.",
            "",
            f"- candidate class: `{payload['validation']['candidate']['class']}`",
            f"- paired mean: `{payload['validation']['candidate']['paired_mean']}`",
            f"- interval: `[{payload['validation']['candidate']['low']}, {payload['validation']['candidate']['high']}]`",
            f"- control class: `{payload['validation']['control']['class']}`",
            f"- sign co-criterion: `{payload['validation']['sign']['pass']}`",
            "",
            "## Sign counts",
            "",
            "| split | cell | successes | n | low | pass |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for split_name in ("evaluation", "replication", "validation"):
        for cell in payload[split_name]["sign"]["cells"]:
            lines.append(
                f"| {split_name} | `{cell['intervention_id']}` | {cell['successes']} | {cell['n']} | {cell['low']} | `{cell['pass']}` |"
            )
    lines.extend(
        [
            "",
            "## Leakage",
            "",
            f"`leakage_ok = {str(payload['leakage_ok']).lower()}`",
            "",
            "Held-out prediction files were closed before the matching outcome files. They contain `g`, `g_orth`, and `m`. They do not contain an observed effect. No slope was fit.",
            "",
        ]
    )
    return "\n".join(lines)


def _interpretation(verdict: str) -> str:
    if verdict == "FEASIBILITY_CONTINUE":
        return (
            "The frozen rule clears both held-out partitions, the orthogonal control does not, and the sign bound holds. "
            "That is evidence that g carries held-out response information beyond the scalar mean for this family. "
            "It is not a self-model. Construction of an object whose response map is g becomes the justified next step, followed by a separate consumption test."
        )
    if verdict == "FEASIBILITY_KILL":
        return (
            "The evaluation interval lies below zero, and replication does not show the opposite win. "
            "On this catalog and this family, g increases held-out absolute error relative to the scalar mean. "
            "The recorded first-order account does not identify prompt-level response variation here. "
            "Self-models in general are not shown to be impossible. This catalog cannot be reused to shop for another observable."
        )
    return (
        "The recorded classes do not meet the continue rule or the kill rule. "
        "g is not declared useless. Feasibility stays undecided, and self-model construction stays unjustified. "
        "A second observable is not authorized."
    )


def main() -> None:
    if RAW.exists() and any(RAW.iterdir()):
        raise SystemExit("raw directory already has files; refusing to rerun")
    if (ROOT / "reports" / "PROJECT_FEASIBILITY_GATE_RESULTS.json").exists():
        raise SystemExit("results already exist; refusing to rerun")
    _assert_catalog_frozen()
    RAW.mkdir(parents=True, exist_ok=True)
    _write(RAW / "catalog_freeze.json", {"prompts": catalog(), "source": "reports/PROJECT_FEASIBILITY_GATE_CATALOG.md"})
    runner = _runner()
    model, device = runner.load_model()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("architecture check failed")
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"token ids changed: {tokens['token_ids']}")
    d1 = primary_direction(896, 22101)
    d2 = orthogonal_direction(896, 22103, d1)
    if vector_sha256(d1) != D1_SHA or vector_sha256(d2) != D2_SHA:
        raise RuntimeError("direction hash does not match the frozen vectors")

    train_prompts = _prompts("train")
    validation_prompts = _prompts("validation")
    evaluation_prompts = _prompts("evaluation")
    replication_prompts = _prompts("replication")

    train_measurements = _derivative_rows(model, runner, tokens, d1, d2, train_prompts)
    _write(RAW / "train_derivatives.json", train_measurements)
    train_outcomes = _outcome_rows(model, runner, tokens, d1, train_prompts, train_measurements)
    _write(RAW / "train_outcomes.json", train_outcomes)
    means = _means(train_outcomes)
    _write(RAW / "train_means.json", means)

    heldout = {
        "validation": validation_prompts,
        "evaluation": evaluation_prompts,
        "replication": replication_prompts,
    }
    measurements = {}
    prediction_paths = {}
    for name, prompts in heldout.items():
        measurements[name] = _derivative_rows(model, runner, tokens, d1, d2, prompts)
        _write(RAW / f"{name}_derivatives.json", measurements[name])
        predictions = _predictions(measurements[name], means)
        path = RAW / f"{name}_predictions.json"
        _write(path, predictions)
        _assert_no_effect(path)
        prediction_paths[name] = path
    for name in heldout:
        if (RAW / f"{name}_outcomes.json").exists():
            raise RuntimeError("held-out outcomes existed before predictions were checked")

    outcomes = {}
    for name, prompts in heldout.items():
        _assert_no_effect(prediction_paths[name])
        outcomes[name] = _outcome_rows(model, runner, tokens, d1, prompts, measurements[name])
        _write(RAW / f"{name}_outcomes.json", outcomes[name])

    frozen_means = json.loads((RAW / "train_means.json").read_text(encoding="utf-8"))
    if frozen_means != means:
        raise RuntimeError("train means changed after they were frozen")

    joined = {}
    for name in heldout:
        predictions = _predictions(measurements[name], means)
        joined[name] = _join(measurements[name], outcomes[name], predictions)
    train_joined = _join(train_measurements, train_outcomes, _predictions(train_measurements, means))
    scored = {name: _score_split(joined[name]) for name in heldout}
    verdict = decide(
        scored["evaluation"]["candidate"]["class"],
        scored["replication"]["candidate"]["class"],
        scored["evaluation"]["control"]["class"],
        scored["replication"]["control"]["class"],
        bool(scored["evaluation"]["sign"]["pass"]),
        bool(scored["replication"]["sign"]["pass"]),
    )
    leakage_ok = True
    for name in heldout:
        _assert_no_effect(prediction_paths[name])
        if prediction_paths[name].stat().st_mtime > (RAW / f"{name}_outcomes.json").stat().st_mtime:
            leakage_ok = False
    payload = {
        "verdict": verdict,
        "device": device,
        "alpha": 1.0,
        "family": list(FAMILY.values()),
        "observable": "g",
        "candidate": "g",
        "control": "g_orth",
        "train_means": means,
        "evaluation": scored["evaluation"],
        "replication": scored["replication"],
        "validation": scored["validation"],
        "leakage_ok": leakage_ok,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 23001,
        "direction_sha256": D1_SHA,
        "control_direction_sha256": D2_SHA,
        "interpretation": _interpretation(verdict),
    }
    _write(ROOT / "reports" / "PROJECT_FEASIBILITY_GATE_RESULTS.json", payload)
    fields = [
        "intervention_id",
        "layer",
        "direction_id",
        "alpha",
        "prompt_id",
        "family",
        "partition",
        "g",
        "g_orth",
        "m",
        "baseline_prediction",
        "candidate_prediction",
        "control_prediction",
        "baseline_output",
        "intervened_output",
        "observed_effect",
        "abs_error_baseline",
        "abs_error_candidate",
        "abs_error_control",
        "paired_difference",
        "control_paired_difference",
    ]
    episode_path = ROOT / "reports" / "PROJECT_FEASIBILITY_GATE_EPISODES.csv"
    with episode_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in train_joined + joined["validation"] + joined["evaluation"] + joined["replication"]:
            writer.writerow(row)
    (ROOT / "reports" / "PROJECT_FEASIBILITY_GATE_EXECUTION.md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "leakage_ok": leakage_ok, "evaluation": scored["evaluation"]["candidate"], "replication": scored["replication"]["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
