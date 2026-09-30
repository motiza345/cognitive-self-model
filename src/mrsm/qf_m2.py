"""Cross-model regime comparison. Not an MRSM gate and not a mechanism claim.

The primary profile metric is cosine similarity of null-corrected regime vectors.
It is fixed here, before any M2 result is read. Pearson correlation is exploratory
and is not an input to the decision.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

from src.mrsm import HEADS
from src.mrsm.qf_regime import FROZEN_REGIMES, heterogeneity
from src.mrsm.qwen_bridge import SIGN_FLOOR, is_context_cancellation

PAIRS = (
    ("completion", "instruction"),
    ("completion", "syntax"),
    ("instruction", "syntax"),
)
PRIMARY_METRIC = "cosine_similarity_of_r1_null_corrected_slot_vectors"
PERMUTATION_COUNT = 6


def r2_standardized(effect: float, null_mean: float, null_std: float) -> dict[str, Any]:
    """Written M2 formula. No substitution when the null standard deviation is zero."""
    if abs(float(null_std)) <= SIGN_FLOOR:
        return {
            "value": None,
            "status": "UNDEFINED_BECAUSE_NULL_STD_IS_ZERO",
            "formula": "(effect - null_mean) / null_std",
        }
    return {
        "value": (float(effect) - float(null_mean)) / float(null_std),
        "status": "DEFINED",
        "formula": "(effect - null_mean) / null_std",
    }


def cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= SIGN_FLOOR or right_norm <= SIGN_FLOOR:
        return None
    return dot / (left_norm * right_norm)


def pearson(left: list[float], right: list[float]) -> float | None:
    """Exploratory. Not a decision input."""
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_dev = [value - left_mean for value in left]
    right_dev = [value - right_mean for value in right]
    return cosine(left_dev, right_dev)


def regime_permutations() -> list[tuple[str, ...]]:
    """All 3! label orders. The count is the regime count, not a tuned draw count."""
    permutations = list(itertools.permutations(FROZEN_REGIMES))
    if len(permutations) != PERMUTATION_COUNT:
        raise RuntimeError("Regime permutation count drifted.")
    return permutations


def _profile(cells: dict[str, dict[str, float]], regime: str) -> list[float]:
    return [float(cells[slot][regime]) for slot in HEADS]


def matched_cosines(qwen: dict[str, list[float]], gpt2: dict[str, list[float]], order: tuple[str, ...]) -> list[float | None]:
    return [cosine(qwen[regime], gpt2[permuted]) for regime, permuted in zip(FROZEN_REGIMES, order)]


def _mean(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def cross_model_null(qwen: dict[str, list[float]], gpt2: dict[str, list[float]]) -> dict[str, Any]:
    rows = []
    for order in regime_permutations():
        values = matched_cosines(qwen, gpt2, order)
        rows.append({
            "gpt2_regime_order": list(order),
            "identity": list(order) == list(FROZEN_REGIMES),
            "cosines": values,
            "mean_cosine": _mean(values),
        })
    identity = next(row for row in rows if row["identity"])
    others = [row for row in rows if not row["identity"]]
    undefined = identity["mean_cosine"] is None or any(row["mean_cosine"] is None for row in others)
    strictly_best = False
    if not undefined:
        strictly_best = all(identity["mean_cosine"] > row["mean_cosine"] for row in others)
    return {
        "status": "UNDEFINED" if undefined else "COMPUTED",
        "procedure": "exhaustive permutation of the three GPT-2 regime labels; Qwen labels stay fixed",
        "permutation_count": PERMUTATION_COUNT,
        "count_rule": "3 factorial, fixed by the frozen regime count",
        "metric": PRIMARY_METRIC,
        "numeric_threshold": None,
        "rows": rows,
        "identity_mean_cosine": identity["mean_cosine"],
        "identity_strictly_best": strictly_best,
    }


def _pairwise(profiles: dict[str, list[float]]) -> dict[str, float | None]:
    return {
        f"{left}__{right}": cosine(profiles[left], profiles[right])
        for left, right in PAIRS
    }


def _strongest(pairs: dict[str, float | None]) -> str | None:
    defined = {name: value for name, value in pairs.items() if value is not None}
    if len(defined) != len(pairs):
        return None
    best = max(defined.values())
    winners = [name for name, value in defined.items() if value == best]
    if len(winners) != 1:
        return None
    return winners[0]


def sign_cell(qwen_sign: str, gpt2_sign: str, comparable: bool) -> str:
    if not comparable or qwen_sign == "NOT_COMPARABLE" or gpt2_sign == "NOT_COMPARABLE":
        return "not_comparable"
    if qwen_sign == "~" or gpt2_sign == "~":
        return "near_zero"
    if qwen_sign == gpt2_sign:
        return "sign_agreement"
    return "sign_disagreement"


def level_a(cells: list[str]) -> str:
    comparable = [cell for cell in cells if cell != "not_comparable"]
    if not comparable:
        return "INCONCLUSIVE"
    agreements = comparable.count("sign_agreement")
    disagreements = comparable.count("sign_disagreement")
    if disagreements == 0 and comparable.count("near_zero") == 0:
        return "SUPPORTED"
    if agreements > disagreements:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def level_b(null: dict[str, Any], matched: list[float | None]) -> str:
    if null["status"] != "COMPUTED" or any(value is None for value in matched):
        return "INCONCLUSIVE"
    positive = all(value > 0 for value in matched)
    if null["identity_strictly_best"] and positive:
        return "SUPPORTED"
    if null["identity_strictly_best"] or positive:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def level_c(qwen_pairs: dict[str, float | None], gpt2_pairs: dict[str, float | None]) -> str:
    qwen_values = [qwen_pairs[f"{left}__{right}"] for left, right in PAIRS]
    gpt2_values = [gpt2_pairs[f"{left}__{right}"] for left, right in PAIRS]
    if any(value is None for value in qwen_values + gpt2_values):
        return "INCONCLUSIVE"
    relation = cosine(qwen_values, gpt2_values)
    if relation is None:
        return "INCONCLUSIVE"
    same_peak = _strongest(qwen_pairs) is not None and _strongest(qwen_pairs) == _strongest(gpt2_pairs)
    if relation > 0 and same_peak:
        return "SUPPORTED"
    if relation > 0 or same_peak:
        return "PARTIAL"
    return "NOT_SUPPORTED"


def decide(level_b_label: str, level_c_label: str) -> dict[str, str]:
    if "INCONCLUSIVE" in {level_b_label, level_c_label}:
        label = "INCONCLUSIVE"
    elif level_b_label == "SUPPORTED" and level_c_label == "SUPPORTED":
        label = "CROSS_MODEL_REGIME_PATTERN_SUPPORTED"
    elif level_b_label == "NOT_SUPPORTED" and level_c_label == "NOT_SUPPORTED":
        label = "CROSS_MODEL_REGIME_PATTERN_NOT_SUPPORTED"
    else:
        label = "PARTIAL_CROSS_MODEL_CONSISTENCY"
    interpretation = {
        "CROSS_MODEL_REGIME_PATTERN_SUPPORTED": (
            "A regime-level causal response pattern appears to transfer across the two models "
            "under the tested intervention protocol."
        ),
        "PARTIAL_CROSS_MODEL_CONSISTENCY": (
            "Some higher-level regime structure appears transferable, but the evidence is "
            "insufficient for a general model-independent causal representation."
        ),
        "CROSS_MODEL_REGIME_PATTERN_NOT_SUPPORTED": (
            "The tested regime structure does not transfer across Qwen and GPT-2 under the "
            "current intervention protocol."
        ),
        "INCONCLUSIVE": (
            "The controls or the profiles do not decide whether a regime-level pattern transfers."
        ),
    }
    recommendation = {
        "CROSS_MODEL_REGIME_PATTERN_SUPPORTED": (
            "M3 is motivated by a transferred regime pattern. Do not run it automatically."
        ),
        "PARTIAL_CROSS_MODEL_CONSISTENCY": (
            "Do not run M3. Partial transfer is not sufficient to test a general "
            "model-independent representation."
        ),
        "CROSS_MODEL_REGIME_PATTERN_NOT_SUPPORTED": (
            "Do not run M3. This comparison does not warrant a model-independent abstraction test."
        ),
        "INCONCLUSIVE": (
            "Do not run M3. The controls do not show that an abstraction test is meaningful."
        ),
    }
    return {
        "m2_decision": label,
        "primary_interpretation": interpretation[label],
        "recommendation_for_m3": recommendation[label],
        "m3_executed": False,
        "mechanism_identity": "NOT_EVALUATED",
    }


def slot_heterogeneity(effects_by_regime: dict[str, list[float]]) -> dict[str, Any]:
    stats = heterogeneity(effects_by_regime)
    means = {regime: sum(effects_by_regime[regime]) / len(effects_by_regime[regime]) for regime in FROZEN_REGIMES}
    pooled = sum(sum(effects_by_regime[regime]) for regime in FROZEN_REGIMES) / sum(
        len(effects_by_regime[regime]) for regime in FROZEN_REGIMES
    )
    stats["context_cancellation"] = is_context_cancellation(means, pooled)
    stats["used_as_gate"] = False
    return stats
