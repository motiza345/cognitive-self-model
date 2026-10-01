"""Execute the frozen single-measurement protocol. Does not change the protocol.

Held-out predictions are written before held-out outcomes exist.
The slope is fit on the train split only.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_post_m23_single_measurement import (
    TESTED_ALPHA,
    TESTED_CELL,
    TESTED_DIRECTION,
    TESTED_LAYER,
    TESTED_MEASUREMENT,
    catalog,
    fit_candidate,
    predict_baseline,
    predict_candidate,
)
from src.cognitive_self_model.m23.m22_reuse import (
    logit_margin,
    make_resid_hook,
    primary_direction,
    vector_sha256,
)
from src.cognitive_self_model.m23.stats import paired_mean_ci

D1_SHA = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
HOOK_NAME = "blocks.15.hook_resid_post"
RAW = ROOT / "reports" / "post_m23_s_raw"
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
        handle.write(
            json.dumps({"path": str(path.relative_to(ROOT)), "utc": stamp, "sha256": digest}) + "\n"
        )
    return stamp


def _prompts(partition: str) -> list[dict[str, str]]:
    return sorted((row for row in catalog() if row["partition"] == partition), key=lambda row: row["prompt_id"])


def _sample_sd(values: list[float]) -> float:
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _measure_pre_dot(model, runner, text: str, direction) -> float:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    recorder: dict[str, Any] = {}
    vector = torch.tensor(direction, dtype=torch.float32)

    def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
        recorder["fired"] = int(recorder.get("fired", 0)) + 1
        recorder["hook_name"] = str(getattr(hook, "name", ""))
        pre = residual[:, -1, :].detach()
        direction_on_pre = vector.to(device=pre.device, dtype=pre.dtype)
        recorder["pre_dot"] = float(torch.dot(pre[0].float(), direction_on_pre.float()).item())
        return residual

    with torch.no_grad():
        runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(HOOK_NAME, hook_fn)]))
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("measurement hook did not fire once")
    if recorder.get("hook_name") not in {HOOK_NAME, ""}:
        raise RuntimeError(f"unexpected measurement hook {recorder.get('hook_name')}")
    return float(recorder["pre_dot"])


def _outcome(model, runner, text: str, positive_id: int, negative_id: int, direction) -> dict[str, float]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        baseline = logit_margin(runner._as_logits(model(tokens)), positive_id, negative_id)
        recorder: dict[str, Any] = {}
        vector = torch.tensor(direction, dtype=torch.float32)
        hook_fn = make_resid_hook(1.0, vector, recorder)
        intervened = logit_margin(
            runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(HOOK_NAME, hook_fn)])),
            positive_id,
            negative_id,
        )
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("intervention hook did not fire once")
    if recorder.get("hook_name") not in {HOOK_NAME, ""}:
        raise RuntimeError(f"unexpected intervention hook {recorder.get('hook_name')}")
    if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not stay on the last token")
    return {
        "baseline_output": float(baseline),
        "intervened_output": float(intervened),
        "observed_effect": float(intervened - baseline),
        "audit_hook_pre_add_dot": float(recorder["pre_dot"]),
    }


def _measurement_rows(model, runner, direction, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        if prompt["prompt_id"].startswith("g-"):
            raise RuntimeError("contaminated g-* prompt entered the run")
        pre_dot = _measure_pre_dot(model, runner, prompt["text"], direction)
        rows.append(
            {
                "intervention_id": TESTED_CELL,
                "layer": TESTED_LAYER,
                "direction_id": TESTED_DIRECTION,
                "alpha": 1.0,
                "measurement": TESTED_MEASUREMENT,
                "prompt_id": prompt["prompt_id"],
                "family": prompt["family"],
                "partition": prompt["partition"],
                "pre_dot": pre_dot,
            }
        )
        print(f"measure {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _outcome_rows(model, runner, tokens, direction, prompts, measurements) -> list[dict[str, Any]]:
    by_id = {row["prompt_id"]: row for row in measurements}
    rows = []
    for prompt in prompts:
        outcome = _outcome(
            model,
            runner,
            prompt["text"],
            tokens["positive_token_id"],
            tokens["negative_token_id"],
            direction,
        )
        measured = float(by_id[prompt["prompt_id"]]["pre_dot"])
        rows.append(
            {
                "intervention_id": TESTED_CELL,
                "prompt_id": prompt["prompt_id"],
                "family": prompt["family"],
                "partition": prompt["partition"],
                "alpha": 1.0,
                "pre_dot": measured,
                "baseline_output": outcome["baseline_output"],
                "intervened_output": outcome["intervened_output"],
                "observed_effect": outcome["observed_effect"],
                "audit_hook_pre_add_dot": outcome["audit_hook_pre_add_dot"],
                "audit_abs_diff_unintervened_vs_pre_add": abs(measured - outcome["audit_hook_pre_add_dot"]),
            }
        )
        print(f"outcome {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _predictions(fit: dict[str, float], measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in measurements:
        pre_dot = float(row["pre_dot"])
        rows.append(
            {
                "intervention_id": TESTED_CELL,
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "partition": row["partition"],
                "alpha": 1.0,
                "measurement": TESTED_MEASUREMENT,
                "pre_dot": pre_dot,
                "baseline_prediction": predict_baseline(fit, pre_dot),
                "candidate_prediction": predict_candidate(fit, pre_dot),
                "outcome_present": False,
            }
        )
    return rows


def _assert_no_effect(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "observed_effect" in text or "intervened_output" in text or "baseline_output" in text:
        raise RuntimeError(f"{path.name} contains an outcome field")


def _verdict(interval_class: str) -> str:
    if interval_class == "CI_POSITIVE":
        return "PREDICTIVE_INFORMATION_SUPPORTED"
    if interval_class == "CI_NEGATIVE":
        return "MEASUREMENT_HURTS"
    if interval_class == "CI_INCLUDES_ZERO":
        return "INCONCLUSIVE"
    raise RuntimeError(f"unexpected interval class {interval_class}")


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    differences = [float(row["paired_difference"]) for row in rows]
    interval = paired_mean_ci(differences)
    baseline_errors = [float(row["abs_error_baseline"]) for row in rows]
    candidate_errors = [float(row["abs_error_candidate"]) for row in rows]
    return {
        "n": len(rows),
        "baseline_mae": sum(baseline_errors) / len(rows),
        "candidate_mae": sum(candidate_errors) / len(rows),
        "paired_mean": float(interval["mean"]),
        "low": float(interval["low"]),
        "high": float(interval["high"]),
        "class": str(interval["class"]),
        "secondary_not_used_for_verdict": True,
    }


def _join(measurements, outcomes, predictions) -> list[dict[str, Any]]:
    outcome_by = {row["prompt_id"]: row for row in outcomes}
    prediction_by = {row["prompt_id"]: row for row in predictions}
    joined = []
    for row in measurements:
        outcome = outcome_by[row["prompt_id"]]
        prediction = prediction_by[row["prompt_id"]]
        effect = float(outcome["observed_effect"])
        baseline = float(prediction["baseline_prediction"])
        candidate = float(prediction["candidate_prediction"])
        baseline_error = abs(effect - baseline)
        candidate_error = abs(effect - candidate)
        joined.append(
            {
                **{key: row[key] for key in ("intervention_id", "prompt_id", "family", "partition", "alpha", "pre_dot")},
                "layer": TESTED_LAYER,
                "direction_id": TESTED_DIRECTION,
                "measurement": TESTED_MEASUREMENT,
                "baseline_prediction": baseline,
                "candidate_prediction": candidate,
                "baseline_output": float(outcome["baseline_output"]),
                "intervened_output": float(outcome["intervened_output"]),
                "observed_effect": effect,
                "abs_error_baseline": baseline_error,
                "abs_error_candidate": candidate_error,
                "paired_difference": baseline_error - candidate_error,
                "audit_abs_diff_unintervened_vs_pre_add": float(outcome["audit_abs_diff_unintervened_vs_pre_add"]),
            }
        )
    return joined


def _run_validator() -> str:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_post_m23_single_measurement.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0 or "PROTOCOL_CHECK_PASS" not in output or "HARNESS_PASS" not in output:
        raise RuntimeError(f"protocol validator failed:\n{output}")
    design = (ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_DESIGN.md").read_text(encoding="utf-8")
    if "READY_FOR_EXECUTION" not in design:
        raise RuntimeError("design status is not READY_FOR_EXECUTION")
    return output.strip()


def _leakage(fit_stamp: str, prediction_paths: list[Path], outcome_paths: list[Path]) -> dict[str, Any]:
    for path in prediction_paths:
        _assert_no_effect(path)
    prediction_mtime = max(path.stat().st_mtime for path in prediction_paths)
    outcome_mtime = min(path.stat().st_mtime for path in outcome_paths)
    order_ok = prediction_mtime <= outcome_mtime
    return {
        "leakage_ok": bool(order_ok),
        "prediction_files_lack_observed_effect": True,
        "heldout_predictions_mtime": prediction_mtime,
        "first_heldout_outcome_mtime": outcome_mtime,
        "coefficients_stamp": fit_stamp,
        "fit_inputs": "train only",
        "alpha": TESTED_ALPHA,
        "measurement": TESTED_MEASUREMENT,
        "cell": TESTED_CELL,
    }


def _markdown(payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation"]
    validation = payload["validation"]
    train = payload["train"]
    lines = [
        "# Post-M23 single-measurement execution",
        "",
        "Protocol status remains `READY_FOR_EXECUTION`. This file does not change that status.",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        "The verdict uses the evaluation interval only. A supported result means `pre_dot` carries predictive information about the response of `M22.1-D1-L15` on this catalog. It does not establish a mechanism representation, causal understanding, a self-model, or generalization across mechanisms, layers, or interventions.",
        "",
        "## Train fit",
        "",
        f"- n: `{train['n']}`",
        f"- effect mean `m`: `{train['m']}`",
        f"- pre_dot mean: `{train['pre_dot_mean']}`",
        f"- slope `b`: `{train['b']}`",
        f"- pre_dot sample sd: `{train['pre_dot_sample_sd']}`",
        f"- pre_dot min: `{train['pre_dot_min']}`",
        f"- pre_dot max: `{train['pre_dot_max']}`",
        f"- pre_dot sum of squared deviations: `{train['pre_dot_sum_of_squares']}`",
        "",
        "The slope was fit on these twelve train pairs and was not refit.",
        "",
        "## Evaluation",
        "",
        f"- baseline MAE: `{evaluation['baseline_mae']}`",
        f"- candidate MAE: `{evaluation['candidate_mae']}`",
        f"- paired mean: `{evaluation['paired_mean']}`",
        f"- interval: `[{evaluation['low']}, {evaluation['high']}]`",
        f"- class: `{evaluation['class']}`",
        f"- draws: `5000`, seed: `23001`",
        "",
        "## Validation",
        "",
        "Validation is not the decision. The train fit was kept.",
        "",
        f"- baseline MAE: `{validation['baseline_mae']}`",
        f"- candidate MAE: `{validation['candidate_mae']}`",
        f"- paired mean: `{validation['paired_mean']}`",
        f"- interval: `[{validation['low']}, {validation['high']}]`",
        f"- class: `{validation['class']}`",
        "",
        "## Per family",
        "",
        "These four-prompt intervals are secondary. They do not change the verdict.",
        "",
        "| family | n | baseline MAE | candidate MAE | paired mean | class |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["per_family"]:
        lines.append(
            f"| {row['family']} | {row['n']} | {row['baseline_mae']} | {row['candidate_mae']} | {row['paired_mean']} | `{row['class']}` |"
        )
    audit = payload["audit_max_abs_diff_unintervened_vs_pre_add"]
    lines.extend(
        [
            "",
            "## Leakage",
            "",
            f"`leakage_ok = {str(payload['leakage']['leakage_ok']).lower()}`",
            "",
            "Validation and evaluation predictions were written after the train fit and before either held-out outcome file. Those prediction files contain `pre_dot` and the two predictions. They do not contain an observed effect. No `g-*` prompt was scored.",
            "",
            "## Execution notes",
            "",
            f"Maximum absolute difference between the unintervened-forward `pre_dot` and the pre-add dot inside the later intervention hook: `{audit}`.",
            "The predictor used the unintervened-forward value. The hook dot was an audit quantity and was not a second measurement.",
            "",
            payload["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if RAW.exists() and any(RAW.iterdir()):
        raise SystemExit("raw directory already has files; refusing to rerun")
    if (ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_RESULTS.json").exists():
        raise SystemExit("results already exist; refusing to rerun")
    validator_output = _run_validator()
    print(validator_output, flush=True)
    RAW.mkdir(parents=True, exist_ok=True)
    runner = _runner()
    model, device = runner.load_model()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("architecture check failed")
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"token ids changed: {tokens['token_ids']}")
    direction = primary_direction(896, 22101)
    if vector_sha256(direction) != D1_SHA:
        raise RuntimeError("D1 hash does not match the frozen vector")

    train_prompts = _prompts("train")
    validation_prompts = _prompts("validation")
    evaluation_prompts = _prompts("evaluation")
    if len(train_prompts) != 12 or len(validation_prompts) != 12 or len(evaluation_prompts) != 12:
        raise RuntimeError("partition sizes are not 12")

    train_measurements = _measurement_rows(model, runner, direction, train_prompts)
    _write(RAW / "train_measurements.json", train_measurements)
    train_outcomes = _outcome_rows(model, runner, tokens, direction, train_prompts, train_measurements)
    _write(RAW / "train_outcomes.json", train_outcomes)
    fit = fit_candidate(
        [float(row["pre_dot"]) for row in train_measurements],
        [float(row["observed_effect"]) for row in train_outcomes],
    )
    fit_payload = {
        "intervention_id": TESTED_CELL,
        "measurement": TESTED_MEASUREMENT,
        "alpha": 1.0,
        "n_train": 12,
        "m": fit["mean"],
        "pre_dot_mean": fit["pre_dot_mean"],
        "b": fit["slope"],
        "fit_source": "train outcomes only",
    }
    fit_stamp = _write(RAW / "coefficients.json", fit_payload)

    validation_measurements = _measurement_rows(model, runner, direction, validation_prompts)
    evaluation_measurements = _measurement_rows(model, runner, direction, evaluation_prompts)
    _write(RAW / "validation_measurements.json", validation_measurements)
    _write(RAW / "evaluation_measurements.json", evaluation_measurements)
    validation_predictions = _predictions(fit, validation_measurements)
    evaluation_predictions = _predictions(fit, evaluation_measurements)
    validation_prediction_path = RAW / "validation_predictions.json"
    evaluation_prediction_path = RAW / "evaluation_predictions.json"
    _write(validation_prediction_path, validation_predictions)
    _write(evaluation_prediction_path, evaluation_predictions)
    _assert_no_effect(validation_prediction_path)
    _assert_no_effect(evaluation_prediction_path)
    if (RAW / "validation_outcomes.json").exists() or (RAW / "evaluation_outcomes.json").exists():
        raise RuntimeError("held-out outcomes existed before predictions were checked")

    validation_outcomes = _outcome_rows(model, runner, tokens, direction, validation_prompts, validation_measurements)
    _write(RAW / "validation_outcomes.json", validation_outcomes)
    if "observed_effect" in evaluation_prediction_path.read_text(encoding="utf-8"):
        raise RuntimeError("evaluation predictions gained an outcome field")
    evaluation_outcomes = _outcome_rows(model, runner, tokens, direction, evaluation_prompts, evaluation_measurements)
    _write(RAW / "evaluation_outcomes.json", evaluation_outcomes)

    # The frozen coefficients are not rewritten.
    frozen = json.loads((RAW / "coefficients.json").read_text(encoding="utf-8"))
    if frozen["b"] != fit_payload["b"] or frozen["m"] != fit_payload["m"]:
        raise RuntimeError("coefficients changed after the train fit")

    leakage = _leakage(
        fit_stamp,
        [validation_prediction_path, evaluation_prediction_path],
        [RAW / "validation_outcomes.json", RAW / "evaluation_outcomes.json"],
    )
    train_joined = _join(train_measurements, train_outcomes, _predictions(fit, train_measurements))
    validation_joined = _join(validation_measurements, validation_outcomes, validation_predictions)
    evaluation_joined = _join(evaluation_measurements, evaluation_outcomes, evaluation_predictions)
    evaluation_score = _score(evaluation_joined)
    evaluation_score["secondary_not_used_for_verdict"] = False
    validation_score = _score(validation_joined)
    verdict = _verdict(str(evaluation_score["class"]))
    per_family = []
    for family in ("completion", "instruction", "syntax"):
        subset = [row for row in evaluation_joined if row["family"] == family]
        scored = _score(subset)
        scored["family"] = family
        per_family.append(scored)
    pre_dots = [float(row["pre_dot"]) for row in train_measurements]
    center = fit["pre_dot_mean"]
    audit_diff = max(float(row["audit_abs_diff_unintervened_vs_pre_add"]) for row in train_outcomes + validation_outcomes + evaluation_outcomes)
    if verdict == "PREDICTIVE_INFORMATION_SUPPORTED":
        interpretation = (
            "The interval lies above zero. On this catalog, the centered linear use of pre_dot reduced held-out absolute error relative to the train mean. "
            "That is predictive information for this fixed cell. It is not a mechanism representation, not causal understanding, and not a self-model."
        )
    elif verdict == "MEASUREMENT_HURTS":
        interpretation = (
            "The interval lies below zero. This predeclared measurement, in the predeclared linear form, increased held-out absolute error on this test."
        )
    else:
        interpretation = (
            "The interval includes zero. This test does not show that pre_dot reduces held-out absolute error relative to the train mean. "
            "The measurement is not declared useless. The uncertainty is the reported interval."
        )
    payload = {
        "verdict": verdict,
        "cell": TESTED_CELL,
        "hook": HOOK_NAME,
        "direction_id": TESTED_DIRECTION,
        "direction_sha256": D1_SHA,
        "alpha": 1.0,
        "measurement": TESTED_MEASUREMENT,
        "device": device,
        "train": {
            "n": 12,
            "m": fit["mean"],
            "b": fit["slope"],
            "pre_dot_mean": fit["pre_dot_mean"],
            "pre_dot_sample_sd": _sample_sd(pre_dots),
            "pre_dot_min": min(pre_dots),
            "pre_dot_max": max(pre_dots),
            "pre_dot_sum_of_squares": sum((value - center) ** 2 for value in pre_dots),
        },
        "validation": validation_score,
        "evaluation": evaluation_score,
        "per_family": per_family,
        "leakage": leakage,
        "audit_max_abs_diff_unintervened_vs_pre_add": audit_diff,
        "interpretation": interpretation,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 23001,
    }
    _write(ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_RESULTS.json", payload)
    episode_fields = [
        "intervention_id",
        "layer",
        "direction_id",
        "alpha",
        "measurement",
        "prompt_id",
        "family",
        "partition",
        "pre_dot",
        "baseline_prediction",
        "candidate_prediction",
        "baseline_output",
        "intervened_output",
        "observed_effect",
        "abs_error_baseline",
        "abs_error_candidate",
        "paired_difference",
    ]
    episode_path = ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_EPISODES.csv"
    with episode_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=episode_fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in train_joined + validation_joined + evaluation_joined:
            writer.writerow(row)
    report_path = ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_EXECUTION.md"
    report_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "leakage_ok": leakage["leakage_ok"], "evaluation": evaluation_score}, indent=2))


if __name__ == "__main__":
    main()
