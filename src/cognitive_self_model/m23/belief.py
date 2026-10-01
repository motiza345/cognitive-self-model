"""Explicit M23 belief and the pre-registered update rule.

predict accepts a belief and an alpha. It has no outcome argument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


ANCHOR_ALPHA = 1.0
CRITICAL_Z = 1.96
INFLATION_CAP = 2.0


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty observations")
    return float(sum(values) / len(values))


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("sample sd requires at least two observations")
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


@dataclass
class SelfModelBelief:
    mechanism_id: str
    intervention_type: str
    predicted_effect: float
    uncertainty: float
    validity_scope: dict[str, Any]
    evidence_history: list[dict[str, Any]] = field(default_factory=list)
    update_history: list[dict[str, Any]] = field(default_factory=list)
    version: int = 0
    observations: list[float] = field(default_factory=list)
    inflation: float = 1.0
    anchor_alpha: float = ANCHOR_ALPHA

    def predict(self, alpha: float) -> dict[str, float | int]:
        scale = float(alpha) / float(self.anchor_alpha)
        effect = scale * float(self.predicted_effect)
        uncertainty = abs(scale) * float(self.uncertainty)
        return {
            "predicted_effect": float(effect),
            "predicted_direction": _sign(effect),
            "uncertainty": float(uncertainty),
            "model_version": int(self.version),
            "evidence_version": int(self.version),
        }


def belief_from_observations(
    mechanism_id: str,
    intervention_type: str,
    observations: list[float],
    *,
    validity_scope: dict[str, Any],
) -> SelfModelBelief:
    if len(observations) < 2:
        raise ValueError("initial belief requires at least two discovery observations")
    copied = [float(value) for value in observations]
    return SelfModelBelief(
        mechanism_id=mechanism_id,
        intervention_type=intervention_type,
        predicted_effect=_mean(copied),
        uncertainty=_sample_sd(copied),
        validity_scope=dict(validity_scope),
        version=1,
        observations=copied,
        inflation=1.0,
    )


def _snapshot(belief: SelfModelBelief) -> dict[str, Any]:
    return {
        "mechanism_id": belief.mechanism_id,
        "intervention_type": belief.intervention_type,
        "predicted_effect": belief.predicted_effect,
        "uncertainty": belief.uncertainty,
        "validity_scope": dict(belief.validity_scope),
        "version": belief.version,
        "inflation": belief.inflation,
        "n_observations": len(belief.observations),
    }


def update_belief(
    belief: SelfModelBelief,
    observed_effect: float,
    evidence_id: str,
) -> SelfModelBelief:
    """Return a new belief. Does not replace predicted_effect with the observation."""
    observed = float(observed_effect)
    if not math.isfinite(observed):
        raise ValueError("observed effect must be finite")
    prior_mean = float(belief.predicted_effect)
    prior_uncertainty = float(belief.uncertainty)
    sign_mismatch = _sign(prior_mean) != 0 and _sign(observed) != 0 and _sign(prior_mean) != _sign(observed)
    outside = abs(observed - prior_mean) > CRITICAL_Z * prior_uncertainty
    confident = abs(prior_mean) >= CRITICAL_Z * prior_uncertainty
    contradicted = bool(sign_mismatch or outside)
    critical = bool(confident and contradicted)
    if critical:
        reason = "CONFIDENT_CONTRADICTION"
        inflation = min(INFLATION_CAP, float(belief.inflation) * 2.0)
    elif contradicted:
        reason = "EVIDENCE_OUTSIDE_OR_SIGN"
        inflation = float(belief.inflation)
    else:
        reason = "PRECISION_WEIGHTED_MEAN"
        inflation = float(belief.inflation)
    observations = list(belief.observations) + [observed]
    mean = _mean(observations)
    sd = _sample_sd(observations)
    if len(belief.observations) >= 1 and any(abs(value - observed) > 1e-12 for value in belief.observations):
        if abs(mean - observed) <= 1e-12:
            raise RuntimeError("update overwrote the belief with the observation")
    scope = dict(belief.validity_scope)
    if contradicted:
        scope["status"] = "CONTRADICTED"
    elif scope.get("status") != "CONTRADICTED":
        scope["status"] = "IN_SCOPE"
    updated = SelfModelBelief(
        mechanism_id=belief.mechanism_id,
        intervention_type=belief.intervention_type,
        predicted_effect=mean,
        uncertainty=sd * inflation,
        validity_scope=scope,
        evidence_history=list(belief.evidence_history),
        update_history=list(belief.update_history),
        version=int(belief.version) + 1,
        observations=observations,
        inflation=inflation,
        anchor_alpha=belief.anchor_alpha,
    )
    record = {
        "belief_before": _snapshot(belief),
        "evidence": {
            "evidence_id": evidence_id,
            "observed_effect": observed,
            "sign_mismatch": sign_mismatch,
            "outside_interval": outside,
            "confident": confident,
            "critical": critical,
        },
        "belief_after": _snapshot(updated),
        "update_reason": reason,
        "model_version_before": belief.version,
        "model_version_after": updated.version,
    }
    updated.evidence_history.append(record["evidence"])
    updated.update_history.append(record)
    return updated


def apply_observations(
    belief: SelfModelBelief,
    observations: list[tuple[str, float]],
) -> SelfModelBelief:
    current = belief
    for evidence_id, observed in observations:
        current = update_belief(current, observed, evidence_id)
    return current
