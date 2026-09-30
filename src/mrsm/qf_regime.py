"""Regime-specific characterization. Not an MRSM gate and not a mechanism claim."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

from cognitive_self_model.m22_1.metrics import bootstrap_mean_interval

from src.mrsm.qf_localize import CLEAR_SIGN_CONSISTENCY, effect_floor
from src.mrsm.qwen_bridge import SIGN_FLOOR, is_context_cancellation

FROZEN_REGIMES = ("completion", "instruction", "syntax")
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 22102
CI_METHOD = (
    "percentile bootstrap of the paired margin-drop mean via "
    "cognitive_self_model.m22_1.metrics.bootstrap_mean_interval; "
    "draws and seed are the frozen m22_1_preflight values 2000 and 22102"
)


def _sign(value: float) -> int:
    if abs(value) <= SIGN_FLOOR:
        return 0
    return 1 if value > 0 else -1


def sign_symbol(sign: int) -> str:
    if sign > 0:
        return "+"
    if sign < 0:
        return "-"
    return "~"


def sign_class(effects: list[float]) -> str:
    """Reuse the existing zero floor and the existing 0.75 consistency bar.

    UNSTABLE is that existing bar, not a new threshold. NEAR_ZERO is the
    existing sign-zero floor. No other band is named.
    """
    if len(effects) < 2:
        return "UNDEFINED_UNDER_CURRENT_CONTRACT"
    signs = [_sign(value) for value in effects]
    nonzero = [sign for sign in signs if sign != 0]
    if nonzero and len(set(nonzero)) > 1:
        consistency = max(nonzero.count(1), nonzero.count(-1)) / len(effects)
        if consistency < CLEAR_SIGN_CONSISTENCY:
            return "UNSTABLE"
    code = _sign(mean(effects))
    if code > 0:
        return "POSITIVE"
    if code < 0:
        return "NEGATIVE"
    return "NEAR_ZERO"


def regime_cell(effects: list[float], nulls: list[float]) -> dict[str, Any]:
    if not effects or not nulls or len(effects) != len(nulls):
        return {"status": "INSUFFICIENT_DATA", "n": len(effects)}
    center = mean(effects)
    null_center = mean(nulls)
    spread = pstdev(effects)
    null_spread = pstdev(nulls)
    count = len(effects)
    floor = effect_floor(max(abs(value) for value in nulls))
    signs = [_sign(value) for value in effects]
    nonzero = [sign for sign in signs if sign != 0]
    if nonzero:
        consistency = max(nonzero.count(1), nonzero.count(-1)) / count
    else:
        consistency = 0.0
    separable = abs(center) > floor
    label = sign_class(effects)
    if separable and consistency >= CLEAR_SIGN_CONSISTENCY:
        status = "CAUSAL_SIGNAL_SUPPORTED"
    elif separable:
        status = "CAUSAL_SIGNAL_WEAK"
    else:
        status = "CAUSAL_SIGNAL_NOT_SEPARATED_FROM_NULL"
    interval = bootstrap_mean_interval(effects, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)
    low, high = interval["low"], interval["high"]
    standardized = None
    if spread > SIGN_FLOOR:
        standardized = (center - null_center) / spread
    return {
        "n": count,
        "mean_effect": center,
        "median_effect": sorted(effects)[count // 2] if count % 2 else (
            sorted(effects)[count // 2 - 1] + sorted(effects)[count // 2]
        ) / 2,
        "std_effect": spread,
        "standard_error": spread / math.sqrt(count),
        "null_mean": null_center,
        "null_std": null_spread,
        "effect_minus_null": center - null_center,
        "standardized_effect": standardized,
        "sign": _sign(center),
        "sign_symbol": sign_symbol(_sign(center)),
        "sign_class": label,
        "sign_consistency": consistency,
        "separable_from_regime_null": separable,
        "effect_floor": floor,
        "confidence_interval": [low, high],
        "confidence_interval_method": CI_METHOD,
        "status": status,
        "std_definition": "population standard deviation, matching m22 summarize_deltas ddof=0",
    }


def heterogeneity(regime_effects: dict[str, list[float]]) -> dict[str, Any]:
    """Exploratory variance comparison. Not a gate and not a preregistered test."""
    if set(regime_effects) != set(FROZEN_REGIMES):
        return {"label": "EXPLORATORY", "status": "INCONCLUSIVE", "used_as_gate": False}
    means = [mean(regime_effects[name]) for name in FROZEN_REGIMES]
    between = pstdev(means) ** 2
    within_parts = [pstdev(regime_effects[name]) ** 2 for name in FROZEN_REGIMES]
    within = mean(within_parts)
    ratio = None if within <= SIGN_FLOOR else between / within
    return {
        "label": "EXPLORATORY",
        "used_as_gate": False,
        "between_regime_variance": between,
        "within_regime_variance": within,
        "heterogeneity_ratio": ratio,
        "definition": (
            "between is the variance of the three regime means; within is the "
            "mean of the within-regime variances. Not an ANOVA and not a gate."
        ),
    }


def slot_context_dependent(cells: dict[str, dict[str, Any]], pooled: float) -> bool:
    means = {name: float(cells[name]["mean_effect"]) for name in FROZEN_REGIMES}
    if is_context_cancellation(means, pooled):
        return True
    signs = {_sign(means[name]) for name in FROZEN_REGIMES}
    return len(signs - {0}) > 1


def decide_regimes(slots: list[dict[str, Any]]) -> dict[str, str]:
    """One characterization label. Heterogeneity is not an input."""
    if not slots:
        label = "INCONCLUSIVE"
        return _pack(label)
    separable = 0
    total = 0
    any_missing = False
    dependent_slots = 0
    for slot in slots:
        cells = slot.get("regimes", {})
        if set(cells) != set(FROZEN_REGIMES):
            any_missing = True
            break
        if any(cells[name].get("status") == "INSUFFICIENT_DATA" for name in FROZEN_REGIMES):
            any_missing = True
            break
        total += len(FROZEN_REGIMES)
        separable += sum(1 for name in FROZEN_REGIMES if cells[name]["separable_from_regime_null"])
        if slot.get("context_dependent"):
            dependent_slots += 1
    if any_missing:
        label = "INCONCLUSIVE"
    elif separable == 0:
        label = "NO_REGIME_SEPARATION"
    elif separable < total and dependent_slots:
        label = "MIXED_SIGNAL"
    elif separable < total:
        label = "MIXED_SIGNAL"
    elif dependent_slots == len(slots):
        label = "REGIME_SPECIFIC_CAUSAL_SIGNAL"
    elif dependent_slots:
        label = "MIXED_SIGNAL"
    else:
        # Every regime is separable and no slot changes sign across regimes.
        # A magnitude-separation claim would need a threshold this contract does not have.
        label = "INCONCLUSIVE"
    return _pack(label, all_cells_separable=separable == total and total > 0)


def _pack(label: str, *, all_cells_separable: bool = False) -> dict[str, str]:
    nxt = {
        "REGIME_SEPARATION_SUPPORTED": (
            "Do not modify the Self-Model or rerun MRSM. Keep every frozen regime "
            "in the next preregistered report; do not select a representative regime."
        ),
        "REGIME_SPECIFIC_CAUSAL_SIGNAL": (
            "Do not infer multiple mechanisms and do not modify the Self-Model. "
            "The next single step is a preregistered decision to report the "
            "intervention effect at regime level, keeping every frozen regime."
        ),
        "MIXED_SIGNAL": (
            "Do not modify the Self-Model or rerun MRSM. Do not drop a regime. "
            "The next single step is a preregistered decision that keeps every "
            "frozen regime visible, because the intervention slots do not share one regime pattern."
        ),
        "NO_REGIME_SEPARATION": (
            "Do not modify the Self-Model or drop a regime. Regime-specific effects "
            "were not separated from each regime's own null. Do not rerun MRSM."
        ),
        "INCONCLUSIVE": (
            "Do not modify the Self-Model, the thresholds, or the Qwen revision. "
            "Do not start a repair from this audit."
        ),
    }
    interpretation = {
        "REGIME_SEPARATION_SUPPORTED": (
            "Regime differences are large enough to report separately. "
            "This does not identify a mechanism."
        ),
        "REGIME_SPECIFIC_CAUSAL_SIGNAL": (
            "The intervention effect depends on regime and has to be reported at "
            "regime level. This does not show that Qwen has several independent mechanisms."
        ),
        "MIXED_SIGNAL": (
            "Every measured discovery regime cell separates from that regime's own null, "
            "but the slots do not share one sign pattern. A single regime explanation is not supported. "
            "This does not identify a mechanism."
            if all_cells_separable
            else
            "Some regime cells separate from their own null and some do not, and "
            "the slots do not share one sign pattern. A single regime explanation is not supported."
        ),
        "NO_REGIME_SEPARATION": (
            "Splitting by regime did not separate the intervention effect from the regime null."
        ),
        "INCONCLUSIVE": (
            "The regime split does not decide whether the differences are a "
            "separable regime relation or measurement variation."
        ),
    }
    return {
        "regime_decision": label,
        "primary_interpretation": interpretation[label],
        "next_single_intervention": nxt[label],
        "mechanism_identity": "NOT_EVALUATED",
        "previous_diagnosis_retained": "INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED",
    }
