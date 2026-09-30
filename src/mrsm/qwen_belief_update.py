"""Qwen scope-update diagnostic. The P head table and the P floor are not used.

The decision shape is the one locked on arm P: a scope conflict revises scope,
an in-scope conflict does not receive a mechanism name, both together abstain,
and a quiet record holds. Uncertainty is computed inside each stored
intervention from discovery prompts only. Tolerance is twice that uncertainty.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mrsm.ccso import (
    LAYERS,
    NOVEL_DIRECTION_SEEDS,
    PRIMARY_ALPHAS,
    REGIMES,
    TRAIN_DIRECTION_SEEDS,
)
from src.mrsm.self_model import LeakageError

HOLD = "HOLD"
REVISE_SCOPE = "REVISE_SCOPE"
ABSTAIN = "ABSTAIN"
RULE_LABELS = (HOLD, REVISE_SCOPE, ABSTAIN)
MIXED = "MIXED"

SEPARATES = "QWEN_UPDATE_SEPARATES"
DOES_NOT_SEPARATE = "QWEN_UPDATE_DOES_NOT_SEPARATE"
INCONCLUSIVE = "INCONCLUSIVE"
DECISIONS = (SEPARATES, DOES_NOT_SEPARATE, INCONCLUSIVE)

# Structural piece of the P rule. The absolute floor of 1.0 is a P margin scale.
TOLERANCE_MULTIPLIER = 2.0

PRIMARY_EXPECTATION = {
    "same_regime": HOLD,
    "regime_change": REVISE_SCOPE,
    "two_changes": ABSTAIN,
}
CONTROL_EXPECTATION = {"scope_string_alone": HOLD}

BELIEF_FIELDS = ("scope", "predictions", "uncertainties")
RECORD_FIELDS = ("intervention_id", "actual_outcome", "scope")
FORBIDDEN_KEYS = {
    "failure_type",
    "expected",
    "expected_label",
    "ground_truth",
    "mechanism",
    "mechanism_label",
    "planted",
    "injection",
}


def preregistration_document() -> dict[str, Any]:
    return {
        "diagnostic": "qwen_belief_scope",
        "status": "diagnostic",
        "model": "frozen Qwen measurements from the CCSO run",
        "question": (
            "Using a belief stored from discovery prompts of one regime, does an "
            "unchanged-context measurement hold, does a regime change revise scope "
            "when the stored effect misses, and does an unseen direction block the update?"
        ),
        "rule_labels": list(RULE_LABELS),
        "hypothesis_output": False,
        "mechanism_identity": "NOT_EVALUATED",
        "decisions": list(DECISIONS),
        "primary_expectation": dict(PRIMARY_EXPECTATION),
        "control_expectation": dict(CONTROL_EXPECTATION),
        "tolerance": {
            "expression": "2.0 * per_intervention_discovery_std",
            "multiplier": TOLERANCE_MULTIPLIER,
            "std": "ddof=0 of Δ/α on discovery prompts, primary alphas, one intervention",
            "p_absolute_floor_1": False,
            "p_head_table": False,
        },
        "belief_fit": "discovery prompts of one regime, train directions, primary alphas, frozen layers",
        "same_regime_prompts": "validation and replication of that regime",
        "regime_change": "all ordered pairs of regimes; mixed pair labels become MIXED",
        "two_changes": (
            "Regime-changed records plus novel directions. A novel direction has no "
            "stored prediction, so the existing undefined-intervention rule abstains."
        ),
        "scope_string_alone": "Same-regime numbers with a scope string that only adds '::label-only'.",
        "excluded": ["alpha ±0.25", "novel directions inside the belief", "P tolerance 10.48", "eight-head table"],
        "decision_order": [
            "split, finiteness, or discovery self-check fails -> INCONCLUSIVE",
            "a primary label mismatches -> QWEN_UPDATE_DOES_NOT_SEPARATE",
            "scope-string control mismatches -> INCONCLUSIVE",
            "otherwise -> QWEN_UPDATE_SEPARATES",
        ],
        "does_not": [
            "modify belief_update.classify",
            "modify SelfModelCore",
            "modify the CCSO measurements",
            "load a new Qwen forward",
            "start M22",
            "claim mechanism identity",
            "choose a new multiplier after scoring",
        ],
    }


def intervention_id(layer: int, seed: int) -> str:
    return f"L{int(layer)}:d{int(seed)}"


def _reject(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS or lowered.startswith("planted"):
                raise LeakageError(f"forbidden input field: {key}")
            _reject(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _reject(item)


def _check_belief(belief: dict[str, Any]) -> None:
    _reject(belief)
    missing = [name for name in BELIEF_FIELDS if name not in belief]
    if missing:
        raise ValueError(f"belief missing fields: {missing}")
    extra = [name for name in belief if name not in BELIEF_FIELDS]
    if extra:
        raise LeakageError(f"belief has fields the rule does not read: {extra}")


def _check_record(record: dict[str, Any]) -> None:
    _reject(record)
    missing = [name for name in RECORD_FIELDS if name not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    extra = [name for name in record if name not in RECORD_FIELDS]
    if extra:
        raise LeakageError(f"record has fields the rule does not read: {extra}")


def classify_qwen(belief: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Return HOLD, REVISE_SCOPE, or ABSTAIN. No mechanism name is available."""
    _check_belief(belief)
    for record in records:
        _check_record(record)
    scope = str(belief["scope"])
    predictions = belief["predictions"]
    uncertainties = belief["uncertainties"]
    judged: list[tuple[str, float, float]] = []
    for record in records:
        key = str(record["intervention_id"])
        if key not in predictions or key not in uncertainties:
            return ABSTAIN
        residual = float(record["actual_outcome"]) - float(predictions[key])
        tolerance = TOLERANCE_MULTIPLIER * float(uncertainties[key])
        judged.append((str(record["scope"]), residual, tolerance))
    scope_hit = any(record_scope != scope and abs(residual) > tolerance for record_scope, residual, tolerance in judged)
    in_hit = any(record_scope == scope and abs(residual) > tolerance for record_scope, residual, tolerance in judged)
    if scope_hit and in_hit:
        return ABSTAIN
    if scope_hit:
        return REVISE_SCOPE
    if in_hit:
        return ABSTAIN
    return HOLD


def _primary_alpha_indices(alphas: np.ndarray) -> list[int]:
    wanted = {format(value, ".2f") for value in PRIMARY_ALPHAS}
    found = [index for index, value in enumerate(alphas) if format(float(value), ".2f") in wanted]
    if len(found) != len(PRIMARY_ALPHAS):
        raise ValueError("primary alphas are not all present")
    return found


def _values(
    delta: np.ndarray,
    alphas: np.ndarray,
    prompts: list[int],
    layer_index: int,
    direction_index: int,
    alpha_indices: list[int],
) -> np.ndarray:
    samples = []
    for prompt in prompts:
        for alpha_index in alpha_indices:
            alpha = float(alphas[alpha_index])
            if alpha == 0.0:
                raise ValueError("primary alpha is zero")
            samples.append(float(delta[prompt, layer_index, direction_index, alpha_index]) / alpha)
    return np.asarray(samples, dtype=np.float64)


def fit_belief(
    delta: np.ndarray,
    alphas: np.ndarray,
    prompts: list[int],
    direction_seeds: list[int],
    regime: str,
) -> dict[str, Any]:
    alpha_indices = _primary_alpha_indices(alphas)
    train = [index for index, seed in enumerate(direction_seeds) if int(seed) in TRAIN_DIRECTION_SEEDS]
    predictions = {}
    uncertainties = {}
    for layer_index, layer in enumerate(LAYERS):
        for direction_index in train:
            samples = _values(delta, alphas, prompts, layer_index, direction_index, alpha_indices)
            key = intervention_id(layer, direction_seeds[direction_index])
            predictions[key] = float(samples.mean())
            uncertainties[key] = float(samples.std(ddof=0))
    return {"scope": regime, "predictions": predictions, "uncertainties": uncertainties}


def records_for(
    delta: np.ndarray,
    alphas: np.ndarray,
    prompts: list[int],
    direction_seeds: list[int],
    seeds: tuple[int, ...],
    scope: str,
) -> list[dict[str, Any]]:
    alpha_indices = _primary_alpha_indices(alphas)
    chosen = [index for index, seed in enumerate(direction_seeds) if int(seed) in seeds]
    rows = []
    for layer_index, layer in enumerate(LAYERS):
        for direction_index in chosen:
            samples = _values(delta, alphas, prompts, layer_index, direction_index, alpha_indices)
            rows.append(
                {
                    "intervention_id": intervention_id(layer, direction_seeds[direction_index]),
                    "actual_outcome": float(samples.mean()),
                    "scope": scope,
                }
            )
    return rows


def combine(labels: list[str]) -> str:
    if not labels:
        return MIXED
    if len(set(labels)) == 1:
        return labels[0]
    return MIXED


def prompts_by(roles: list[str], regimes: list[str], role: str, regime: str) -> list[int]:
    return [index for index, (row_role, row_regime) in enumerate(zip(roles, regimes)) if row_role == role and row_regime == regime]


def held_out_prompts(roles: list[str], regimes: list[str], regime: str) -> list[int]:
    return prompts_by(roles, regimes, "validation", regime) + prompts_by(roles, regimes, "replication", regime)


def evaluate_cases(
    delta: np.ndarray,
    alphas: np.ndarray,
    roles: list[str],
    regimes: list[str],
    direction_seeds: list[int],
) -> dict[str, Any]:
    """Fit on discovery only. Validation and replication are records, not fit data."""
    trace: dict[str, Any] = {
        "same_regime": {},
        "regime_change": {},
        "two_changes": {},
        "scope_string_alone": {},
        "discovery_self": {},
        "decision_input": False,
    }
    finite = True
    for regime in REGIMES:
        discovery = prompts_by(roles, regimes, "discovery", regime)
        if len(discovery) == 0 or len(held_out_prompts(roles, regimes, regime)) == 0:
            return {"ok": False, "reason": "missing split", "trace": trace, "primary": {}, "controls": {}}
        belief = fit_belief(delta, alphas, discovery, direction_seeds, regime)
        if not np.isfinite(list(belief["predictions"].values())).all():
            finite = False
        if not np.isfinite(list(belief["uncertainties"].values())).all():
            finite = False
        self_rows = records_for(delta, alphas, discovery, direction_seeds, TRAIN_DIRECTION_SEEDS, regime)
        trace["discovery_self"][regime] = classify_qwen(belief, self_rows)
        same_rows = records_for(
            delta, alphas, held_out_prompts(roles, regimes, regime), direction_seeds, TRAIN_DIRECTION_SEEDS, regime
        )
        trace["same_regime"][regime] = classify_qwen(belief, same_rows)
        relabeled = [{**row, "scope": f"{regime}::label-only"} for row in same_rows]
        trace["scope_string_alone"][regime] = classify_qwen(belief, relabeled)
        for other in REGIMES:
            if other == regime:
                continue
            key = f"{regime}->{other}"
            changed = records_for(
                delta, alphas, held_out_prompts(roles, regimes, other), direction_seeds, TRAIN_DIRECTION_SEEDS, other
            )
            trace["regime_change"][key] = classify_qwen(belief, changed)
            novel = records_for(
                delta,
                alphas,
                held_out_prompts(roles, regimes, other),
                direction_seeds,
                NOVEL_DIRECTION_SEEDS,
                other,
            )
            trace["two_changes"][key] = classify_qwen(belief, changed + novel)
    primary = {
        "same_regime": combine(list(trace["same_regime"].values())),
        "regime_change": combine(list(trace["regime_change"].values())),
        "two_changes": combine(list(trace["two_changes"].values())),
    }
    controls = {"scope_string_alone": combine(list(trace["scope_string_alone"].values()))}
    self_ok = combine(list(trace["discovery_self"].values())) == HOLD
    return {"ok": True, "finite": finite, "self_ok": self_ok, "trace": trace, "primary": primary, "controls": controls}


def decide_qwen(
    primary: dict[str, str],
    controls: dict[str, str],
    *,
    split_ok: bool,
    finite_ok: bool,
    self_ok: bool,
) -> dict[str, Any]:
    if not split_ok or not finite_ok or not self_ok:
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "PRECONDITION",
            "reason": "The split, a finite effect, or the discovery self-check failed.",
        }
    if any(primary.get(name) != expected for name, expected in PRIMARY_EXPECTATION.items()):
        return {
            "decision": DOES_NOT_SEPARATE,
            "reason_code": "PRIMARY_MISMATCH",
            "reason": "The Qwen cases did not receive the three preregistered updates.",
        }
    if any(controls.get(name) != expected for name, expected in CONTROL_EXPECTATION.items()):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "CONTROL_MISMATCH",
            "reason": "The primary cases matched, and the scope-string control did not hold.",
        }
    return {
        "decision": SEPARATES,
        "reason_code": "SEPARATED",
        "reason": "Same-regime records held, regime changes revised scope, and unseen directions abstained.",
    }
