"""Frozen M23-E measurement: does the current intervention response vary.

The split and the two linear predictors are fixed before the forwards.
Train coefficients are fit only after measurement, and only on the train ids.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.m22_reuse import frozen_prompts, prompt_manifest_sha256
from src.cognitive_self_model.m23.stats import paired_mean_ci

FROZEN_HASHES = {
    "reports/M23_E_PREREGISTRATION.md": "44ccd795c309b38d70b86a2fab7186707920c95589a1255335d660d3acb04cbf",
    "reports/M23_E_LEAKAGE_AUDIT.md": "b45355e40ee7dadb015e85f1d6e214d99e14779db88c171d152cb6b8cf032828",
}
PROMPT_MANIFEST = "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db"
TRAIN_IDS = (
    "completion-01",
    "completion-03",
    "completion-05",
    "instruction-01",
    "instruction-03",
    "instruction-05",
    "syntax-01",
    "syntax-03",
    "syntax-05",
)
EVAL_IDS = (
    "completion-02",
    "completion-04",
    "completion-06",
    "instruction-02",
    "instruction-04",
    "instruction-06",
    "syntax-02",
    "syntax-04",
    "syntax-06",
)
REPLICATION_IDS = {
    "completion-03",
    "completion-06",
    "instruction-03",
    "instruction-06",
    "syntax-03",
    "syntax-06",
}
DISCOVERY_IDS = {
    "completion-01",
    "completion-04",
    "instruction-01",
    "instruction-04",
    "syntax-01",
    "syntax-04",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _sample_variance(values: list[float]) -> float:
    center = _mean(values)
    return float(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _sample_sd(values: list[float]) -> float:
    return float(math.sqrt(_sample_variance(values)))


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count % 2 == 1:
        return float(ordered[count // 2])
    return float(0.5 * (ordered[count // 2 - 1] + ordered[count // 2]))


def _pearson(left: list[float], right: list[float]) -> float | None:
    left_center = _mean(left)
    right_center = _mean(right)
    numerator = sum((x - left_center) * (y - right_center) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_center) ** 2 for x in left) * sum((y - right_center) ** 2 for y in right)
    )
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _ols(feature: list[float], target: list[float]) -> tuple[float, float]:
    feature_center = _mean(feature)
    target_center = _mean(target)
    variance = sum((value - feature_center) ** 2 for value in feature)
    if variance == 0.0:
        raise RuntimeError("train feature has no variance")
    slope = sum((x - feature_center) * (y - target_center) for x, y in zip(feature, target)) / variance
    intercept = target_center - slope * feature_center
    return float(intercept), float(slope)


def _summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "sample_sd": _sample_sd(values),
        "sample_variance": _sample_variance(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
    }


def _label(interval_class: str, leakage_ok: bool) -> str:
    if not leakage_ok:
        return "INCONCLUSIVE"
    if interval_class == "CI_POSITIVE":
        return "SUPPORTED"
    if interval_class == "CI_NEGATIVE":
        return "NOT_SUPPORTED"
    return "INCONCLUSIVE"


def _check_frozen() -> None:
    for relative, expected in FROZEN_HASHES.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"frozen hash mismatch for {relative}: {actual}")
    if prompt_manifest_sha256() != PROMPT_MANIFEST:
        raise SystemExit("prompt catalog hash changed")
    prompts = frozen_prompts()
    if len(prompts) != 18:
        raise SystemExit("catalog is not the frozen 18 prompts")
    train = tuple(record.prompt_id for index, record in enumerate(prompts) if index % 2 == 0)
    evaluation = tuple(record.prompt_id for index, record in enumerate(prompts) if index % 2 == 1)
    if train != TRAIN_IDS or evaluation != EVAL_IDS:
        raise SystemExit("alternating split does not match the preregistered ids")


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measure(runner: Any) -> list[dict[str, Any]]:
    model, _device = runner.load_model()
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise SystemExit(f"token ids changed: {tokens['token_ids']}")
    direction = runner.primary_direction(int(model.cfg.d_model), runner.DIRECTION_SEED)
    prompts = {record.prompt_id: record for record in frozen_prompts()}
    rows: list[dict[str, Any]] = []
    for repeat_index in (0, 1):
        for prompt_id in [record.prompt_id for record in frozen_prompts()]:
            text = prompts[prompt_id].text
            baseline, _empty = runner.forward_margin(
                model, text, tokens["positive_token_id"], tokens["negative_token_id"], None
            )
            for alpha in (1.0, 2.0):
                intervened, recorder = runner.forward_margin(
                    model,
                    text,
                    tokens["positive_token_id"],
                    tokens["negative_token_id"],
                    {"alpha": alpha, "direction": direction},
                )
                if "pre_dot" not in recorder:
                    raise RuntimeError(f"pre_dot missing for {prompt_id}")
                pre_dot = float(recorder["pre_dot"])
                effect = float(intervened - baseline)
                partition = "train" if prompt_id in TRAIN_IDS else "evaluation"
                rows.append(
                    {
                        "episode_id": f"{prompt_id}-a{int(alpha)}-r{repeat_index}",
                        "prompt_id": prompt_id,
                        "seed": 22101,
                        "alpha": alpha,
                        "repeat_index": repeat_index,
                        "partition": partition,
                        "pre_intervention_observation": {
                            "pre_dot": pre_dot,
                            "baseline_margin": float(baseline),
                        },
                        "pre_dot": pre_dot,
                        "baseline_output": float(baseline),
                        "baseline_margin": float(baseline),
                        "intervened_output": float(intervened),
                        "observed_effect": effect,
                    }
                )
            print(f"repeat {repeat_index} {prompt_id} baseline {baseline:.6f}", flush=True)
    return rows


def _primary(rows: list[dict[str, Any]], alpha: float, partition: str) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["repeat_index"] == 0 and row["alpha"] == alpha and row["partition"] == partition
    ]
    return sorted(selected, key=lambda row: row["prompt_id"])


def _contrast(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    feature_name: str | None,
) -> dict[str, Any]:
    train_y = [float(row["observed_effect"]) for row in train_rows]
    constant = _mean(train_y)
    if feature_name is None:
        intercept, slope = constant, 0.0
        feature_key = None
    else:
        intercept, slope = _ols([float(row[feature_name]) for row in train_rows], train_y)
        feature_key = feature_name

    def predict(row: dict[str, Any]) -> float:
        if feature_key is None:
            return constant
        return intercept + slope * float(row[feature_key])

    eval_errors = []
    paired_against_constant = []
    predictions = []
    for row in eval_rows:
        predicted = predict(row)
        predictions.append(predicted)
        error = abs(float(row["observed_effect"]) - predicted)
        eval_errors.append(error)
        if feature_key is not None:
            constant_error = abs(float(row["observed_effect"]) - constant)
            paired_against_constant.append(constant_error - error)
    payload: dict[str, Any] = {
        "intercept": intercept,
        "slope": slope,
        "train_mean_abs_error": _mean(
            [abs(float(row["observed_effect"]) - predict(row)) for row in train_rows]
        ),
        "eval_mean_abs_error": _mean(eval_errors),
        "eval_predictions": predictions,
        "eval_prompt_ids": [row["prompt_id"] for row in eval_rows],
    }
    if feature_key is not None:
        interval = paired_mean_ci(paired_against_constant)
        payload["paired_differences"] = paired_against_constant
        payload["interval_mean"] = interval["mean"]
        payload["interval_low"] = interval["low"]
        payload["interval_high"] = interval["high"]
        payload["interval_class"] = interval["class"]
        payload["interval_width"] = float(interval["high"]) - float(interval["low"])
    return payload


def _alpha_block(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    train_rows = _primary(rows, alpha, "train")
    eval_rows = _primary(rows, alpha, "evaluation")
    if [row["prompt_id"] for row in train_rows] != sorted(TRAIN_IDS):
        raise RuntimeError("train ids drifted")
    if [row["prompt_id"] for row in eval_rows] != sorted(EVAL_IDS):
        raise RuntimeError("evaluation ids drifted")
    all_rows = sorted(train_rows + eval_rows, key=lambda row: row["prompt_id"])
    effects = [float(row["observed_effect"]) for row in all_rows]
    pre_dots = [float(row["pre_dot"]) for row in all_rows]
    margins = [float(row["baseline_margin"]) for row in all_rows]

    def correlations(group: list[dict[str, Any]]) -> dict[str, float | None]:
        target = [float(row["observed_effect"]) for row in group]
        return {
            "pre_dot": _pearson([float(row["pre_dot"]) for row in group], target),
            "baseline_margin": _pearson([float(row["baseline_margin"]) for row in group], target),
        }

    constant = _contrast(train_rows, eval_rows, None)
    pre_dot = _contrast(train_rows, eval_rows, "pre_dot")
    baseline = _contrast(train_rows, eval_rows, "baseline_margin")
    return {
        "effect": _summarize(effects),
        "pre_dot": _summarize(pre_dots),
        "baseline_margin": _summarize(margins),
        "correlation_train": correlations(train_rows),
        "correlation_evaluation": correlations(eval_rows),
        "correlation_all_descriptive": correlations(all_rows),
        "constant": constant,
        "pre_dot_predictor": pre_dot,
        "baseline_margin_predictor": baseline,
    }


def _scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_prompt: dict[str, dict[float, float]] = {}
    for row in rows:
        if row["repeat_index"] != 0:
            continue
        by_prompt.setdefault(row["prompt_id"], {})[float(row["alpha"])] = float(row["observed_effect"])
    ratios = []
    residuals = []
    excluded = 0
    for prompt_id in sorted(by_prompt):
        first = by_prompt[prompt_id][1.0]
        second = by_prompt[prompt_id][2.0]
        residual = second - 2.0 * first
        residuals.append({"prompt_id": prompt_id, "scaling_residual": residual, "effect_alpha1": first, "effect_alpha2": second})
        if first == 0.0:
            excluded += 1
            ratios.append({"prompt_id": prompt_id, "ratio": None})
        else:
            ratios.append({"prompt_id": prompt_id, "ratio": second / first})
    defined = [float(row["ratio"]) for row in ratios if row["ratio"] is not None]
    residual_values = [float(row["scaling_residual"]) for row in residuals]
    return {
        "n_prompts": len(residuals),
        "undefined_ratios": excluded,
        "mean_ratio": _mean(defined),
        "median_ratio": _median(defined),
        "sample_sd_ratio": _sample_sd(defined),
        "ratios": ratios,
        "residual": {
            **_summarize(residual_values),
            "values": residuals,
        },
    }


def _repeatability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = {(row["prompt_id"], row["alpha"], row["repeat_index"]): row for row in rows}
    fields = ("pre_dot", "baseline_output", "intervened_output", "observed_effect")
    max_abs = 0.0
    within_variances = []
    for prompt_id in [record.prompt_id for record in frozen_prompts()]:
        for alpha in (1.0, 2.0):
            first = index[(prompt_id, alpha, 0)]
            second = index[(prompt_id, alpha, 1)]
            if first["seed"] != second["seed"] or first["prompt_id"] != second["prompt_id"]:
                raise RuntimeError("repeat identity failed")
            for field in fields:
                max_abs = max(max_abs, abs(float(first[field]) - float(second[field])))
            pair = [float(first["observed_effect"]), float(second["observed_effect"])]
            within_variances.append(_sample_variance(pair))
    between = {}
    for alpha in (1.0, 2.0):
        effects = [
            float(index[(prompt_id, alpha, 0)]["observed_effect"])
            for prompt_id in [record.prompt_id for record in frozen_prompts()]
        ]
        between[str(alpha)] = _sample_variance(effects)
    return {
        "n_repeats": 2,
        "max_abs_repeat_difference": max_abs,
        "deterministic_identical": max_abs == 0.0,
        "within_condition_mean_variance": _mean(within_variances),
        "between_prompt_variance": between,
        "estimates_stochastic_noise": False if max_abs == 0.0 else True,
    }


def _historical_diff(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Read-only check. Historical numbers are not copied into the fit."""
    index = {
        (row["prompt_id"], float(row["alpha"])): row
        for row in rows
        if row["repeat_index"] == 0
    }
    diffs: list[float] = []

    def compare(prompt_id: str, alpha: float, field: str, historical: float) -> None:
        diffs.append(abs(float(index[(prompt_id, alpha)][field]) - float(historical)))

    validation = json.loads((ROOT / "reports" / "m23_raw" / "validation_outcomes.json").read_text(encoding="utf-8"))
    for row in validation["d1"]:
        compare(row["prompt_id"], 1.0, "observed_effect", row["observed_delta"])
        compare(row["prompt_id"], 1.0, "baseline_margin", row["baseline_margin"])
        compare(row["prompt_id"], 1.0, "pre_dot", row["pre_dot"])
    replication = json.loads((ROOT / "reports" / "m23_raw" / "replication_outcomes.json").read_text(encoding="utf-8"))
    for key, alpha in (("alpha_1", 1.0), ("alpha_2", 2.0)):
        for row in replication[key]:
            compare(row["prompt_id"], alpha, "observed_effect", row["observed_delta"])
            compare(row["prompt_id"], alpha, "baseline_margin", row["baseline_margin"])
            compare(row["prompt_id"], alpha, "pre_dot", row["pre_dot"])
    with (ROOT / "reports" / "M23_D2_EPISODES.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["prompt_id"] not in DISCOVERY_IDS:
                raise RuntimeError("D2 file is not limited to discovery prompts")
            compare(row["prompt_id"], 1.0, "observed_effect", float(row["observed_effect"]))
    return {
        "n_comparisons": len(diffs),
        "max_abs_difference": max(diffs) if diffs else None,
        "used_as_fit_input": False,
    }


def _write_episodes(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "episode_id",
        "prompt_id",
        "seed",
        "alpha",
        "repeat_index",
        "partition",
        "pre_dot",
        "baseline_output",
        "baseline_margin",
        "intervened_output",
        "observed_effect",
    ]
    path = ROOT / "reports" / "M23_E_EPISODES.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    _check_frozen()
    raw = ROOT / "reports" / "m23_e_raw"
    measurement_path = raw / "measurements.json"
    coefficient_path = raw / "train_coefficients.json"
    if measurement_path.exists() or coefficient_path.exists():
        raise SystemExit("M23-E raw outputs already exist; refusing to mix runs")
    raw.mkdir(parents=True, exist_ok=True)
    split_path = raw / "split.json"
    split_path.write_text(
        json.dumps({"train": list(TRAIN_IDS), "evaluation": list(EVAL_IDS)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if measurement_path.exists():
        raise SystemExit("measurements appeared before the forward")

    runner = _runner()
    rows = _measure(runner)
    measurement_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    split_before_outcomes = split_path.stat().st_mtime_ns <= measurement_path.stat().st_mtime_ns

    alpha_results = {}
    coefficients = {}
    for alpha in (1.0, 2.0):
        block = _alpha_block(rows, alpha)
        alpha_results[str(alpha)] = block
        coefficients[str(alpha)] = {
            "constant": block["constant"]["intercept"],
            "pre_dot": {
                "intercept": block["pre_dot_predictor"]["intercept"],
                "slope": block["pre_dot_predictor"]["slope"],
            },
            "baseline_margin": {
                "intercept": block["baseline_margin_predictor"]["intercept"],
                "slope": block["baseline_margin_predictor"]["slope"],
            },
            "train_prompt_ids": list(TRAIN_IDS),
        }
    coefficient_path.write_text(json.dumps(coefficients, indent=2, sort_keys=True), encoding="utf-8")

    contrasts = []
    for alpha in ("1.0", "2.0"):
        for feature, key in (("pre_dot", "pre_dot_predictor"), ("baseline_margin", "baseline_margin_predictor")):
            predictor = alpha_results[alpha][key]
            contrasts.append(
                {
                    "alpha": float(alpha),
                    "feature": feature,
                    "label": predictor["interval_class"],
                    "mean_paired_difference": predictor["interval_mean"],
                    "interval_low": predictor["interval_low"],
                    "interval_high": predictor["interval_high"],
                    "interval_width": predictor["interval_width"],
                    "eval_mean_abs_error_constant": alpha_results[alpha]["constant"]["eval_mean_abs_error"],
                    "eval_mean_abs_error_feature": predictor["eval_mean_abs_error"],
                }
            )
    score_path = raw / "eval_scores.json"
    score_path.write_text(json.dumps({"contrasts": contrasts}, indent=2, sort_keys=True), encoding="utf-8")
    coefficients_before_scores = coefficient_path.stat().st_mtime_ns <= score_path.stat().st_mtime_ns

    repeatability = _repeatability(rows)
    scaling = _scaling(rows)
    historical = _historical_diff(rows)
    eval_not_replication_only = any(prompt_id not in REPLICATION_IDS for prompt_id in EVAL_IDS)
    leakage = {
        "split_written_before_outcomes": split_before_outcomes,
        "fit_uses_train_ids_only": True,
        "evaluation_not_replication_only": eval_not_replication_only,
        "features_exclude_outcome_and_regime": True,
        "coefficients_written_before_eval_scores": coefficients_before_scores,
        "repeats_share_prompt_seed_alpha": repeatability["n_repeats"] == 2,
        "historical_files_not_used_as_measurements": historical["used_as_fit_input"] is False,
    }
    leakage_ok = all(bool(value) for value in leakage.values())
    for contrast in contrasts:
        contrast["result"] = _label(str(contrast["label"]), leakage_ok)
        contrast["label"] = contrast["result"]
    any_supported = any(contrast["result"] == "SUPPORTED" for contrast in contrasts)
    if leakage_ok and not any_supported:
        statement = (
            "No evidence that the tested pre-intervention measurements identify variation in intervention response."
        )
    elif leakage_ok and any_supported:
        statement = (
            "At least one frozen linear measurement reduced held-out absolute error. "
            "That is not a self-model."
        )
    else:
        statement = "LEAKAGE"
    payload = {
        "historical_m23_verdict": "INCONCLUSIVE",
        "historical_diagnostic": {
            "INTERVENTION_DESIGN_FAILURE": "INCONCLUSIVE",
            "IDENTIFIABILITY_FAILURE": "INCONCLUSIVE",
            "REPRESENTATION_BOTTLENECK": "INCONCLUSIVE",
            "BELIEF_UPDATE_INFORMATION_FAILURE": "NOT_SUPPORTED",
        },
        "historical_d2": "INCONCLUSIVE",
        "intervention_design_failure": "INCONCLUSIVE",
        "n_train": len(TRAIN_IDS),
        "n_evaluation": len(EVAL_IDS),
        "alphas": alpha_results,
        "contrasts": contrasts,
        "any_contrast_supported": any_supported,
        "identifiability_statement": statement,
        "scaling": scaling,
        "repeatability": repeatability,
        "historical_consistency": historical,
        "leakage": leakage,
        "leakage_ok": leakage_ok,
        "baseline_output_is_baseline_margin": True,
    }
    (ROOT / "reports" / "M23_E_RESULTS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_episodes(rows)
    print(statement, flush=True)
    for contrast in contrasts:
        print(
            contrast["alpha"],
            contrast["feature"],
            contrast["result"],
            contrast["interval_low"],
            contrast["interval_high"],
            flush=True,
        )


if __name__ == "__main__":
    main()
