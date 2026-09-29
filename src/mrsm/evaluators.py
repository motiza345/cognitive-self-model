"""H1–H4 scoring. Ground truth is an evaluator argument, never a self-model input."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mrsm import HEADS


def _sign(value: float, floor: float) -> int:
    if abs(value) <= floor:
        return 0
    return 1 if value > 0.0 else -1


def score_h1(
    mechanism: dict[str, Any] | None,
    holdout_effects: dict[str, float],
    *,
    ground_truth_ordered: tuple[str, str],
    sign_floor: float,
) -> dict[str, Any]:
    """Score one predicted edge against the evaluator-supplied ground-truth edge."""
    if mechanism is None or mechanism.get("status") != "IDENTIFIED":
        return {
            "status": "FAIL",
            "mechanism_f1": 0.0,
            "causal_edge_precision": 0.0,
            "false_discovery_rate": 0.0,
            "reason": "NOT_IDENTIFIABLE" if mechanism and mechanism.get("status") == "NOT_IDENTIFIABLE" else "no mechanism",
            "verifies_ground_truth": True,
        }
    predicted = tuple(mechanism["ordered"])
    identity = predicted == tuple(ground_truth_ordered)
    signs = [_sign(float(holdout_effects[name]), sign_floor) for name in predicted]
    sign_match = signs == [1, 1]
    recovered = identity and sign_match
    tp = 1 if recovered else 0
    fp = 1 - tp
    fn = 1 - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    fdr = fp / (tp + fp)
    passed = f1 >= 0.80 and precision >= 0.80 and fdr <= 0.20
    return {
        "status": "PASS" if passed else "FAIL",
        "mechanism_f1": f1,
        "causal_edge_precision": precision,
        "false_discovery_rate": fdr,
        "identity_match": identity,
        "sign_match": sign_match,
        "predicted": list(predicted),
        "verifies_ground_truth": True,
    }


def _mae(errors: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(errors, dtype=np.float64))))


def paired_bootstrap_mean_diff(diffs: list[float], *, resamples: int, seed: int) -> dict[str, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if arr.size == 0:
        raise ValueError("empty bootstrap sample")
    stats = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draw = arr[rng.integers(0, arr.size, arr.size)]
        stats[i] = float(np.mean(draw))
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "mean": float(np.mean(arr))}


def score_h2(
    self_pred: dict[str, float],
    baseline_pred: dict[str, dict[str, float]],
    actual_episodes: dict[str, list[float]],
    *,
    resamples: int,
    seed: int,
    sign_floor: float,
) -> dict[str, Any]:
    keys = list(actual_episodes.keys())
    actual = {key: float(np.mean(actual_episodes[key])) for key in keys}
    self_err = [self_pred[key] - actual[key] for key in keys]
    mae_self = _mae(self_err)
    baseline_mae = {name: _mae([preds[key] - actual[key] for key in keys]) for name, preds in baseline_pred.items()}
    strongest = min(baseline_mae, key=baseline_mae.get)
    mae_b = baseline_mae[strongest]
    if mae_b == 0.0:
        reduction = None
        reduction_pass = False
    else:
        reduction = (mae_b - mae_self) / mae_b
        reduction_pass = reduction >= 0.20
    matrix = np.stack([np.asarray(actual_episodes[key], dtype=np.float64) for key in keys], axis=0)
    self_vec = np.array([self_pred[key] for key in keys], dtype=np.float64)
    base_vec = np.array([baseline_pred[strongest][key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples, dtype=np.float64)
    n_episodes = matrix.shape[1]
    for i in range(resamples):
        draw = matrix[:, rng.integers(0, n_episodes, n_episodes)].mean(axis=1)
        stats[i] = float(np.mean(np.abs(base_vec - draw)) - np.mean(np.abs(self_vec - draw)))
    low, high = np.percentile(stats, [2.5, 97.5])
    ci = {"low": float(low), "high": float(high), "mean": float(np.mean(stats))}
    signs = [_sign(self_pred[key], sign_floor) == _sign(actual[key], sign_floor) for key in keys]
    sign_accuracy = float(np.mean(signs))
    ci_pass = ci["low"] > 0.0
    sign_pass = sign_accuracy >= 0.75
    passed = reduction_pass and ci_pass and sign_pass
    return {
        "status": "PASS" if passed else "FAIL",
        "mae_self": mae_self,
        "baseline_mae": baseline_mae,
        "strongest_baseline": strongest,
        "mae_reduction": reduction,
        "bootstrap": ci,
        "sign_accuracy": sign_accuracy,
        "gates": {
            "mae_reduction": reduction_pass,
            "bootstrap": ci_pass,
            "sign_accuracy": sign_pass,
        },
    }


def score_h3(cases: list[dict[str, Any]], *, sign_floor: float) -> dict[str, Any]:
    if not cases:
        return {"status": "NOT_EVALUATED", "reason": "no cases"}
    if any(case["mechanism_abstraction"] == "NOT_IDENTIFIABLE" for case in cases):
        return {
            "status": "NOT_EVALUATED",
            "reason": "NOT_IDENTIFIABLE",
            "not_converted_to_pass": True,
        }
    sign_hits = []
    abs_diffs = []
    abs_orig = []
    abstraction_ok = True
    scope_ok = True
    for case in cases:
        original = float(case["original_prediction"])
        transformed = float(case["transformed_prediction"])
        sign_hits.append(_sign(original, sign_floor) == _sign(transformed, sign_floor))
        abs_diffs.append(abs(transformed - original))
        abs_orig.append(abs(original))
        abstraction_ok = abstraction_ok and case["mechanism_abstraction"] == case["original_abstraction"]
        scope_ok = scope_ok and case["scope"] == case["original_scope"]
    sign_agreement = float(np.mean(sign_hits))
    denom = float(np.mean(abs_orig))
    norm = float(np.mean(abs_diffs) / denom) if denom > sign_floor else float(np.mean(abs_diffs))
    passed = sign_agreement >= 0.90 and norm <= 0.10 and abstraction_ok and scope_ok
    return {
        "status": "PASS" if passed else "FAIL",
        "sign_agreement": sign_agreement,
        "normalized_prediction_difference": norm,
        "mechanism_abstraction_consistency": abstraction_ok,
        "scope_consistency": scope_ok,
    }


def score_h4(
    full_correct: list[int],
    blind_correct: list[int],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    u_full = float(np.mean(full_correct))
    u_blind = float(np.mean(blind_correct))
    denom = max(abs(u_blind), 1e-8)
    relative = (u_full - u_blind) / denom
    diffs = [float(a) - float(b) for a, b in zip(full_correct, blind_correct)]
    ci = paired_bootstrap_mean_diff(diffs, resamples=resamples, seed=seed)
    passed = relative >= 0.10 and ci["low"] > 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "utility_full": u_full,
        "utility_consumption_ablation": u_blind,
        "relative_utility_improvement": relative,
        "bootstrap": ci,
    }


def intervention_ids() -> list[str]:
    ids = [f"single:{name}" for name in HEADS]
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            ids.append(f"pair:{left}+{right}")
    return ids
