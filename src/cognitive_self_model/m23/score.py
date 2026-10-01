"""Score M23 rows with the pre-registered decision map."""

from __future__ import annotations

from typing import Any

from src.cognitive_self_model.m23.stats import (
    clopper_pearson,
    paired_mean_ci,
    q1_status,
    q2_status,
    q4_status,
    q5_status,
    overall_verdict,
)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _mae(errors: list[float]) -> float:
    if not errors:
        raise ValueError("MAE requires cases")
    return float(sum(abs(error) for error in errors) / len(errors))


def _agreement(predicted: float, observed: float) -> bool | None:
    predicted_sign = _sign(predicted)
    observed_sign = _sign(observed)
    if predicted_sign == 0 or observed_sign == 0:
        return None
    return predicted_sign == observed_sign


def _arm_difference(observed: list[float], left: list[float], right: list[float]) -> list[float]:
    """Positive when `right` is closer to the observation than `left` is."""
    if not (len(observed) == len(left) == len(right)):
        raise ValueError("arm lengths differ")
    return [abs(obs - lval) - abs(obs - rval) for obs, lval, rval in zip(observed, left, right)]


def score_partition(
    validation: list[dict[str, Any]],
    replication_anchor: list[dict[str, Any]],
    replication_magnitude: list[dict[str, Any]],
    *,
    q3_pass: bool,
    leakage_ok: bool,
) -> dict[str, Any]:
    if len(validation) == 0 or len(replication_anchor) == 0 or len(replication_magnitude) == 0:
        raise ValueError("each scored partition needs at least one case")

    q1_observed = [float(row["observed_delta"]) for row in validation]
    q1_predicted = [float(row["predicted_effect"]) for row in validation]
    q1_zero = [0.0 for _ in validation]
    q1_ci = paired_mean_ci(_arm_difference(q1_observed, q1_zero, q1_predicted))
    agreements = [_agreement(pred, obs) for pred, obs in zip(q1_predicted, q1_observed)]
    counted = [item for item in agreements if item is not None]
    if counted:
        sign_low, sign_high = clopper_pearson(sum(1 for item in counted if item), len(counted))
    else:
        sign_low, sign_high = None, None
    q1 = q1_status(str(q1_ci["class"]), sign_low, sign_high)

    n_critical = sum(1 for row in validation if bool(row["critical"]))
    q2 = q2_status(n_critical)
    q3 = "PASS" if q3_pass else "FAIL"

    def _q4_like(rows: list[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        observed = [float(row["observed_delta"]) for row in rows]
        updated = [float(row["predicted_effect"]) for row in rows]
        base = [float(row["no_update_prediction"]) for row in rows]
        shuffled = [float(row["shuffled_prediction"]) for row in rows]
        outcome = [float(row["outcome_only_prediction"]) for row in rows]
        update_ci = paired_mean_ci(_arm_difference(observed, base, updated))
        shuffle_ci = paired_mean_ci(_arm_difference(observed, base, shuffled))
        outcome_ci = paired_mean_ci(_arm_difference(observed, outcome, updated))
        status = q4_status(str(update_ci["class"]), str(shuffle_ci["class"]))
        reason = None
        if status == "FAIL" and update_ci["class"] == "CI_POSITIVE":
            reason = "SHUFFLED_ALSO_IMPROVED"
        elif status == "FAIL":
            reason = "UPDATE_WORSE"
        return status, str(q5_status(str(outcome_ci["class"]))), {
            "update_vs_no_update": update_ci,
            "shuffled_vs_no_update": shuffle_ci,
            "specific_vs_outcome_only": outcome_ci,
            "mae_updated": _mae([obs - pred for obs, pred in zip(observed, updated)]),
            "mae_no_update": _mae([obs - pred for obs, pred in zip(observed, base)]),
            "mae_shuffled": _mae([obs - pred for obs, pred in zip(observed, shuffled)]),
            "mae_outcome_only": _mae([obs - pred for obs, pred in zip(observed, outcome)]),
            "q4_fail_reason": reason,
        }

    q4, q5, anchor_metrics = _q4_like(replication_anchor)
    q6, _q5_magnitude, magnitude_metrics = _q4_like(replication_magnitude)
    # Q5 is defined on the same-magnitude held-out context, not on the scaled arm.
    del _q5_magnitude

    questions = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5, "Q6": q6}
    verdict = overall_verdict(
        questions,
        leakage_ok,
        q4_reason=anchor_metrics["q4_fail_reason"],
        q6_reason=magnitude_metrics["q4_fail_reason"],
    )
    coverage = sum(
        1
        for row in validation
        if abs(float(row["observed_delta"]) - float(row["predicted_effect"]))
        <= 1.96 * float(row["uncertainty"])
    ) / len(validation)
    return {
        **verdict,
        "metrics": {
            "Q1": {
                "mae_belief": _mae([obs - pred for obs, pred in zip(q1_observed, q1_predicted)]),
                "mae_zero": _mae(q1_observed),
                "mae_interval": q1_ci,
                "sign_successes": sum(1 for item in counted if item),
                "sign_n": len(counted),
                "sign_low": sign_low,
                "sign_high": sign_high,
                "n_critical": n_critical,
                "uncertainty_coverage": coverage,
            },
            "eval_context": anchor_metrics,
            "eval_magnitude": magnitude_metrics,
        },
    }
