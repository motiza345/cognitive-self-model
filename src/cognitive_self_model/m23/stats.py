"""Pre-registered M23 intervals and verdict map.

The cutoffs in this file are the ones written in reports/M23_PREREGISTRATION.md.
They are not estimated from M23 outcomes.
"""

from __future__ import annotations

import math
import random
from typing import Any


BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 23001


def _binom_sf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0 if k <= 0 else 0.0
    if p >= 1.0:
        return 1.0 if k <= n else 0.0
    total = 0.0
    for index in range(k, n + 1):
        total += math.comb(n, index) * (p ** index) * ((1.0 - p) ** (n - index))
    return total


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval. Degenerate bootstrap is not used."""
    if n <= 0:
        raise ValueError("sign interval requires at least one non-zero sign")
    if successes < 0 or successes > n:
        raise ValueError("success count out of range")
    target = alpha / 2.0

    def lower() -> float:
        if successes == 0:
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if _binom_sf(successes, n, mid) > target:
                hi = mid
            else:
                lo = mid
        return lo

    def upper() -> float:
        if successes == n:
            return 1.0
        lo, hi = 0.0, 1.0
        failures_or_fewer = n - successes
        for _ in range(80):
            mid = (lo + hi) / 2.0
            # P(X <= successes) = P(Y >= n - successes) at 1-p, with Y failures.
            tail = _binom_sf(failures_or_fewer, n, 1.0 - mid)
            if tail > target:
                lo = mid
            else:
                hi = mid
        return hi

    return lower(), upper()


def ci_class(low: float, high: float) -> str:
    if low > 0.0:
        return "CI_POSITIVE"
    if high < 0.0:
        return "CI_NEGATIVE"
    return "CI_INCLUDES_ZERO"


def paired_mean_ci(
    differences: list[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | str]:
    if not differences:
        raise ValueError("bootstrap requires at least one paired difference")
    point = float(sum(differences) / len(differences))
    rng = random.Random(seed)
    n = len(differences)
    means: list[float] = []
    for _ in range(draws):
        total = 0.0
        for _index in range(n):
            total += differences[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low_index = int(math.floor(0.025 * (draws - 1)))
    high_index = int(math.ceil(0.975 * (draws - 1)))
    low = float(means[low_index])
    high = float(means[high_index])
    return {"mean": point, "low": low, "high": high, "class": ci_class(low, high)}


def q1_status(mae_ci: str, sign_low: float | None, sign_high: float | None) -> str:
    sign_fail = sign_high is not None and sign_high < 0.5
    sign_pass = sign_low is not None and sign_low > 0.5
    if mae_ci == "CI_NEGATIVE" or sign_fail:
        return "FAIL"
    if mae_ci == "CI_POSITIVE" and sign_pass:
        return "PASS"
    return "INCONCLUSIVE"


def q2_status(n_critical: int) -> str:
    if n_critical >= 1:
        return "PASS"
    return "INCONCLUSIVE"


def q4_status(update_vs_base: str, shuffled_vs_base: str) -> str:
    if update_vs_base == "CI_NEGATIVE":
        return "FAIL"
    if update_vs_base == "CI_POSITIVE" and shuffled_vs_base == "CI_POSITIVE":
        return "FAIL"
    if update_vs_base == "CI_POSITIVE":
        return "PASS"
    return "INCONCLUSIVE"


def q5_status(specific_vs_outcome: str) -> str:
    if specific_vs_outcome == "CI_POSITIVE":
        return "PASS"
    if specific_vs_outcome == "CI_NEGATIVE":
        return "FAIL"
    return "INCONCLUSIVE"


def overall_verdict(
    questions: dict[str, str],
    leakage_ok: bool,
    q4_reason: str | None = None,
    q6_reason: str | None = None,
) -> dict[str, Any]:
    required = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
    missing = [name for name in required if name not in questions]
    if missing:
        raise ValueError(f"missing questions: {missing}")
    failure_classes: list[str] = []
    if not leakage_ok:
        failure_classes.append("LEAKAGE")
    if questions["Q1"] == "FAIL":
        failure_classes.append("PREDICTION_FAILURE")
    if questions["Q3"] == "FAIL":
        failure_classes.append("BELIEF_UPDATE_FAILURE")
    if questions["Q4"] == "FAIL":
        if q4_reason == "SHUFFLED_ALSO_IMPROVED":
            failure_classes.append("BASELINE_MATCH")
        else:
            failure_classes.append("BELIEF_UPDATE_FAILURE")
    if questions["Q5"] == "FAIL":
        failure_classes.append("BASELINE_MATCH")
    if questions["Q6"] == "FAIL":
        if q6_reason == "SHUFFLED_ALSO_IMPROVED":
            failure_classes.append("BASELINE_MATCH")
        else:
            failure_classes.append("GENERALIZATION_FAILURE")
    deduped: list[str] = []
    for item in failure_classes:
        if item not in deduped:
            deduped.append(item)
    statuses = [questions[name] for name in required]
    if (not leakage_ok) or any(status == "FAIL" for status in statuses):
        verdict = "FAIL"
    elif all(status == "PASS" for status in statuses):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "verdict": verdict,
        "questions": {name: questions[name] for name in required},
        "failure_class": deduped,
        "leakage_ok": leakage_ok,
        "q4_fail_reason": q4_reason,
        "q6_fail_reason": q6_reason,
    }
