"""Frozen M24 consumption rules. This module does not load Qwen.

The verdict map is fixed before outcomes. Validation is scored and is not
an input to the verdict. The train partition is not forwarded.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from src.cognitive_self_model.m23.score import _agreement
from src.cognitive_self_model.m23.stats import clopper_pearson, paired_mean_ci
from src.cognitive_self_model.mechanism_response import MechanismResponseModel

FAMILY = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")
EXECUTABLE_PARTITIONS = ("validation", "evaluation")
SHUFFLE_SEED = 24001
DECISION_THRESHOLD = 0.0
BANNED_PREDICTION_KEYS = ("observed_effect", "baseline_output", "intervened_output")


def baseline_prediction(model: MechanismResponseModel) -> float:
    return float(model.training_baseline_mean)


def positive_decision(value: float) -> bool:
    """True when the value is strictly above the frozen threshold of zero."""
    return float(value) > DECISION_THRESHOLD


def attach_shuffled_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Permute candidate predictions within each cell. One stream, seed 24001.

    Cells are visited in family order. Prompts within a cell are sorted by
    prompt id. Each call starts a new generator, so partitions do not share a stream.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["intervention_id"]), []).append(row)
    if set(grouped) != set(FAMILY):
        raise ValueError("shuffle requires exactly the frozen three cells")
    rng = random.Random(SHUFFLE_SEED)
    shuffled: dict[tuple[str, str], float] = {}
    for intervention_id in FAMILY:
        group = sorted(grouped[intervention_id], key=lambda row: str(row["prompt_id"]))
        values = [float(row["candidate_prediction"]) for row in group]
        rng.shuffle(values)
        for row, value in zip(group, values):
            shuffled[(intervention_id, str(row["prompt_id"]))] = value
    attached = []
    for row in rows:
        key = (str(row["intervention_id"]), str(row["prompt_id"]))
        copied = dict(row)
        copied["shuffled_prediction"] = shuffled[key]
        attached.append(copied)
    return attached


def rows_lack_outcomes(rows: list[dict[str, Any]]) -> bool:
    return all(not any(key in row for key in BANNED_PREDICTION_KEYS) for row in rows)


def prediction_file_clean(text: str) -> bool:
    return all(key not in text for key in BANNED_PREDICTION_KEYS)


def leakage_ok(
    *,
    prediction_before_outcome: bool,
    rows_clean: bool,
    model_unchanged: bool,
    baseline_from_model: bool,
) -> bool:
    return bool(prediction_before_outcome and rows_clean and model_unchanged and baseline_from_model)


def run_prediction_before_outcome(
    names: tuple[str, ...],
    prepare: Callable[[str], Any],
    finish: Callable[[str], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call prepare for every held-out partition before any finish call."""
    if tuple(names) != EXECUTABLE_PARTITIONS:
        raise ValueError("held-out order is validation then evaluation")
    prepared: dict[str, Any] = {}
    for name in names:
        prepared[name] = prepare(name)
    finished: dict[str, Any] = {}
    for name in names:
        finished[name] = finish(name)
    return prepared, finished


def paired_difference(effect: float, baseline: float, prediction: float) -> float:
    return abs(float(effect) - float(baseline)) - abs(float(effect) - float(prediction))


def prompt_level_differences(
    rows: list[dict[str, Any]],
    field: str,
    *,
    expected_prompts: int,
) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(float(row[field]))
    scores = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3:
            raise ValueError(f"{prompt_id} does not have three cells")
        scores.append(sum(values) / 3.0)
    if len(scores) != expected_prompts:
        raise ValueError("prompt aggregation does not match the predeclared partition size")
    return scores


def decision_counts(predicted: list[float], observed: list[float]) -> dict[str, float | int]:
    if len(predicted) != len(observed) or len(predicted) == 0:
        raise ValueError("decision counts require paired cases")
    tp = fp = tn = fn = 0
    for pred, obs in zip(predicted, observed):
        pred_pos = positive_decision(pred)
        obs_pos = positive_decision(obs)
        if pred_pos and obs_pos:
            tp += 1
        elif pred_pos and not obs_pos:
            fp += 1
        elif (not pred_pos) and (not obs_pos):
            tn += 1
        else:
            fn += 1
    correct = tp + tn
    n = tp + fp + tn + fn
    low, high = clopper_pearson(correct, n)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "low": low,
        "high": high,
    }


def sign_does_not_lose(model_counts: dict[str, Any], baseline_counts: dict[str, Any]) -> bool:
    if int(model_counts["n"]) != int(baseline_counts["n"]):
        raise ValueError("decision arms have different lengths")
    return int(model_counts["correct"]) >= int(baseline_counts["correct"])


def sign_agreement(predicted: list[float], observed: list[float]) -> dict[str, Any]:
    counted = []
    for pred, obs in zip(predicted, observed):
        agreement = _agreement(float(pred), float(obs))
        if agreement is not None:
            counted.append(agreement)
    if not counted:
        return {"n": 0, "successes": 0, "accuracy": None, "low": None, "high": None}
    successes = sum(1 for item in counted if item)
    low, high = clopper_pearson(successes, len(counted))
    return {
        "n": len(counted),
        "successes": successes,
        "accuracy": successes / len(counted),
        "low": low,
        "high": high,
    }


def consumption_verdict(
    evaluation_class: str,
    sign_not_lose: bool,
    shuffled_class: str,
    leakage: bool,
) -> str:
    """Frozen map. Classes are the existing ci_class labels."""
    if not leakage:
        return "INCONCLUSIVE"
    if evaluation_class == "CI_POSITIVE" and sign_not_lose and shuffled_class != "CI_POSITIVE":
        return "CONSUMPTION_SUPPORTED"
    if evaluation_class == "CI_NEGATIVE":
        return "CONSUMPTION_NOT_SUPPORTED"
    return "INCONCLUSIVE"


def join_rows(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows_lack_outcomes(predictions):
        raise ValueError("prediction rows already contain an outcome")
    outcome_by = {(str(row["intervention_id"]), str(row["prompt_id"])): row for row in outcomes}
    joined = []
    for prediction in predictions:
        key = (str(prediction["intervention_id"]), str(prediction["prompt_id"]))
        outcome = outcome_by[key]
        effect = float(outcome["observed_effect"])
        baseline = float(prediction["baseline_prediction"])
        candidate = float(prediction["candidate_prediction"])
        shuffled = float(prediction["shuffled_prediction"])
        orth = float(prediction["g_orth"])
        baseline_error = abs(effect - baseline)
        candidate_error = abs(effect - candidate)
        joined.append(
            {
                **prediction,
                "baseline_output": float(outcome["baseline_output"]),
                "intervened_output": float(outcome["intervened_output"]),
                "observed_effect": effect,
                "abs_error_baseline": baseline_error,
                "abs_error_mechanism": candidate_error,
                "paired_difference": baseline_error - candidate_error,
                "shuffled_paired_difference": paired_difference(effect, baseline, shuffled),
                "orth_paired_difference": paired_difference(effect, baseline, orth),
            }
        )
    return joined


def _mae(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        raise ValueError("MAE requires rows")
    return float(sum(float(row[field]) for row in rows) / len(rows))


def _sorted_arm(rows: list[dict[str, Any]], field: str) -> list[float]:
    ordered = sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))
    return [float(row[field]) for row in ordered]


def score_split(rows: list[dict[str, Any]], *, expected_prompts: int) -> dict[str, Any]:
    observed = _sorted_arm(rows, "observed_effect")
    order = sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))
    model_decision = decision_counts(_sorted_arm(rows, "candidate_prediction"), observed)
    baseline_decision = decision_counts(_sorted_arm(rows, "baseline_prediction"), observed)
    primary = paired_mean_ci(prompt_level_differences(rows, "paired_difference", expected_prompts=expected_prompts))
    shuffled = paired_mean_ci(
        prompt_level_differences(rows, "shuffled_paired_difference", expected_prompts=expected_prompts)
    )
    orth = paired_mean_ci(prompt_level_differences(rows, "orth_paired_difference", expected_prompts=expected_prompts))
    cells = []
    for intervention_id in FAMILY:
        cell_rows = [row for row in order if row["intervention_id"] == intervention_id]
        cell_primary = paired_mean_ci(
            [float(row["paired_difference"]) for row in sorted(cell_rows, key=lambda row: str(row["prompt_id"]))]
        )
        cells.append(
            {
                "intervention_id": intervention_id,
                "n": len(cell_rows),
                "paired_mean": float(cell_primary["mean"]),
                "low": float(cell_primary["low"]),
                "high": float(cell_primary["high"]),
                "class": str(cell_primary["class"]),
                "baseline_mae": _mae(cell_rows, "abs_error_baseline"),
                "mechanism_mae": _mae(cell_rows, "abs_error_mechanism"),
                "decision_model": decision_counts(
                    [float(row["candidate_prediction"]) for row in cell_rows],
                    [float(row["observed_effect"]) for row in cell_rows],
                ),
                "decision_baseline": decision_counts(
                    [float(row["baseline_prediction"]) for row in cell_rows],
                    [float(row["observed_effect"]) for row in cell_rows],
                ),
                "sign_agreement": sign_agreement(
                    [float(row["candidate_prediction"]) for row in cell_rows],
                    [float(row["observed_effect"]) for row in cell_rows],
                ),
            }
        )
    return {
        "n_prompts": expected_prompts,
        "n_episodes": len(rows),
        "paired_mean": float(primary["mean"]),
        "low": float(primary["low"]),
        "high": float(primary["high"]),
        "class": str(primary["class"]),
        "baseline_mae": _mae(rows, "abs_error_baseline"),
        "mechanism_mae": _mae(rows, "abs_error_mechanism"),
        "shuffled_class": str(shuffled["class"]),
        "shuffled_paired_mean": float(shuffled["mean"]),
        "shuffled_low": float(shuffled["low"]),
        "shuffled_high": float(shuffled["high"]),
        "orth_class": str(orth["class"]),
        "orth_paired_mean": float(orth["mean"]),
        "orth_low": float(orth["low"]),
        "orth_high": float(orth["high"]),
        "decision_model": model_decision,
        "decision_baseline": baseline_decision,
        "sign_does_not_lose": sign_does_not_lose(model_decision, baseline_decision),
        "sign_agreement": sign_agreement(
            [float(row["candidate_prediction"]) for row in order],
            [float(row["observed_effect"]) for row in order],
        ),
        "cells": cells,
    }
