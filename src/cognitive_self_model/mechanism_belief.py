"""Scoped belief around an immutable mechanism response.

The response rule stays prediction = g. Evidence is appended. A conflict
changes status and, on the inherited critical rule, inflation. It does not
change the response model.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from src.cognitive_self_model.m23.belief import CRITICAL_Z, INFLATION_CAP, _sample_sd
from src.cognitive_self_model.m23.score import _agreement, _sign
from src.cognitive_self_model.mechanism_response import (
    ALPHA,
    DIRECTION_ID,
    DIRECTION_SHA256,
    RESPONSE_RULE,
    MechanismResponseModel,
    PreInterventionState,
    frozen_mechanism_response_models,
)

PROTOCOL_VERSION = "M25.SCOPED_MECHANISM_BELIEF.1"
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
POSITIVE_TOKEN_ID = 9834
NEGATIVE_TOKEN_ID = 902
M24_COMMIT = "3af9edf5055d38e15fb7df4725cc22c0a1ed879f"
M24_CATALOG_SHA256 = "ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78"
SURFACE_FAMILY = "UNVERIFIED"
CALIBRATION_PARTITION = "validation"
CONSISTENT_PARTITION = "evaluation"

STATUS_SUPPORTED = "SUPPORTED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_CONTRADICTED = "CONTRADICTED"

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS = _ROOT / "reports" / "M24_CONSUMPTION_RESULTS.json"
_EPISODES = _ROOT / "reports" / "M24_CONSUMPTION_EPISODES.csv"


@dataclass(frozen=True)
class MechanismClaim:
    model_id: str
    model_revision: str
    intervention_id: str
    layer: int
    hook: str
    direction_id: str
    direction_sha256: str
    alpha: float
    response_rule: str
    positive_token_id: int
    negative_token_id: int
    catalog_sha256: str


@dataclass(frozen=True)
class ValidityScope:
    claim: MechanismClaim
    supported_catalog_sha256: str
    surface_family: str


@dataclass(frozen=True)
class UncertaintyRecord:
    calibration_partition: str
    calibration_source: str
    calibration_prompt_ids: tuple[str, ...]
    n: int
    residual_sd: float
    critical_z: float
    inflation: float

    @property
    def effective(self) -> float:
        return float(self.residual_sd) * float(self.inflation)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source_experiment: str
    artifact: str
    mechanism_id: str
    claim: MechanismClaim
    prompt_id: str
    predicted_value: float
    observed_value: float
    residual: float
    sign_predicted: int
    sign_observed: int
    held_out: bool
    protocol_version: str
    provenance: str
    timestamp: str | None


@dataclass(frozen=True)
class EvidenceApplication:
    evidence: EvidenceRecord
    in_mechanism_scope: bool
    calibration_reuse: bool
    sign_mismatch: bool
    outside_interval: bool
    confident: bool
    critical: bool
    status_before: str
    status_after: str
    reason: str


@dataclass(frozen=True)
class MechanismBelief:
    mechanism_id: str
    response_model: MechanismResponseModel
    validity_scope: ValidityScope
    uncertainty: UncertaintyRecord
    evidence_history: tuple[EvidenceApplication, ...]
    contradiction_history: tuple[EvidenceApplication, ...]
    version: int
    status: str

    def predict(self, pre_intervention_state: PreInterventionState) -> dict[str, float | int | str]:
        predicted = float(self.response_model.predict(pre_intervention_state))
        return {
            "predicted_effect": predicted,
            "uncertainty": self.uncertainty.effective,
            "status": self.status,
            "version": self.version,
            "mechanism_id": self.mechanism_id,
        }


def contradictory_observed(predicted: float, uncertainty: float) -> float:
    """In-scope observation that must trip the inherited contradiction rule.

    The step is (CRITICAL_Z + 1) times the stored uncertainty. A zero
    uncertainty uses step 1, so a nonzero departure is still outside.
    """
    magnitude = (float(CRITICAL_Z) + 1.0) * float(uncertainty)
    if magnitude == 0.0:
        magnitude = 1.0
    value = float(predicted)
    if value > 0.0:
        return value - magnitude
    if value < 0.0:
        return value + magnitude
    return magnitude


def make_evidence(
    *,
    evidence_id: str,
    source_experiment: str,
    artifact: str,
    claim: MechanismClaim,
    prompt_id: str,
    predicted_value: float,
    observed_value: float,
    held_out: bool,
    provenance: str,
    timestamp: str | None = None,
) -> EvidenceRecord:
    predicted = float(predicted_value)
    observed = float(observed_value)
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_experiment=source_experiment,
        artifact=artifact,
        mechanism_id=claim.intervention_id,
        claim=claim,
        prompt_id=prompt_id,
        predicted_value=predicted,
        observed_value=observed,
        residual=observed - predicted,
        sign_predicted=_sign(predicted),
        sign_observed=_sign(observed),
        held_out=held_out,
        protocol_version=PROTOCOL_VERSION,
        provenance=provenance,
        timestamp=timestamp,
    )


def apply_evidence(belief: MechanismBelief, evidence: EvidenceRecord) -> MechanismBelief:
    """Return a new belief. The response model object is the same object."""
    if evidence.residual != float(evidence.observed_value) - float(evidence.predicted_value):
        raise ValueError("evidence residual does not match observed minus predicted")
    if evidence.sign_predicted != _sign(evidence.predicted_value) or evidence.sign_observed != _sign(evidence.observed_value):
        raise ValueError("evidence signs do not match the recorded values")
    in_scope = _mechanism_scope_matches(belief, evidence)
    calibration_reuse = in_scope and evidence.prompt_id in belief.uncertainty.calibration_prompt_ids
    agreement = _agreement(evidence.predicted_value, evidence.observed_value)
    sign_mismatch = agreement is False
    uncertainty = belief.uncertainty.effective
    outside = abs(evidence.residual) > float(CRITICAL_Z) * uncertainty
    confident = abs(evidence.predicted_value) >= float(CRITICAL_Z) * uncertainty
    contradicted = (not calibration_reuse) and in_scope and (sign_mismatch or outside)
    critical = contradicted and confident
    catalog_supported = evidence.claim.catalog_sha256 == belief.validity_scope.supported_catalog_sha256
    status_after, reason = _next_status(
        belief.status,
        in_scope=in_scope,
        calibration_reuse=calibration_reuse,
        contradicted=contradicted,
        critical=critical,
        catalog_supported=catalog_supported,
    )
    inflation = belief.uncertainty.inflation
    if critical:
        inflation = min(float(INFLATION_CAP), float(inflation) * 2.0)
    application = EvidenceApplication(
        evidence=evidence,
        in_mechanism_scope=in_scope,
        calibration_reuse=calibration_reuse,
        sign_mismatch=sign_mismatch,
        outside_interval=outside and in_scope and not calibration_reuse,
        confident=confident,
        critical=critical,
        status_before=belief.status,
        status_after=status_after,
        reason=reason,
    )
    contradictions = belief.contradiction_history
    if contradicted:
        contradictions = contradictions + (application,)
    uncertainty_after = UncertaintyRecord(
        calibration_partition=belief.uncertainty.calibration_partition,
        calibration_source=belief.uncertainty.calibration_source,
        calibration_prompt_ids=belief.uncertainty.calibration_prompt_ids,
        n=belief.uncertainty.n,
        residual_sd=belief.uncertainty.residual_sd,
        critical_z=belief.uncertainty.critical_z,
        inflation=inflation,
    )
    return MechanismBelief(
        mechanism_id=belief.mechanism_id,
        response_model=belief.response_model,
        validity_scope=belief.validity_scope,
        uncertainty=uncertainty_after,
        evidence_history=belief.evidence_history + (application,),
        contradiction_history=contradictions,
        version=belief.version + 1,
        status=status_after,
    )


def belief_from_m24(model: MechanismResponseModel) -> MechanismBelief:
    payload = json.loads(_RESULTS.read_text(encoding="utf-8"))
    if payload.get("verdict") != "CONSUMPTION_SUPPORTED":
        raise RuntimeError("M24 artifact is not a supported consumption result")
    if payload.get("catalog_sha256") != M24_CATALOG_SHA256:
        raise RuntimeError("M24 catalog hash does not match the frozen catalog")
    if model.intervention_id not in payload.get("family", []):
        raise RuntimeError("response model is outside the M24 family")
    if model.response_rule != RESPONSE_RULE or model.alpha != ALPHA:
        raise RuntimeError("response model is not the frozen g rule")
    if model.direction_id != DIRECTION_ID or model.direction_sha256 != DIRECTION_SHA256:
        raise RuntimeError("response model direction is not frozen D1")
    prompt_ids, residuals = _validation_residuals(model.intervention_id)
    if len(residuals) != 12:
        raise RuntimeError("validation calibration does not have 12 episodes")
    claim = _claim_for(model, M24_CATALOG_SHA256)
    return MechanismBelief(
        mechanism_id=model.intervention_id,
        response_model=model,
        validity_scope=ValidityScope(
            claim=claim,
            supported_catalog_sha256=M24_CATALOG_SHA256,
            surface_family=SURFACE_FAMILY,
        ),
        uncertainty=UncertaintyRecord(
            calibration_partition=CALIBRATION_PARTITION,
            calibration_source="reports/M24_CONSUMPTION_EPISODES.csv",
            calibration_prompt_ids=tuple(prompt_ids),
            n=len(residuals),
            residual_sd=_sample_sd(residuals),
            critical_z=float(CRITICAL_Z),
            inflation=1.0,
        ),
        evidence_history=(),
        contradiction_history=(),
        version=1,
        status=STATUS_SUPPORTED,
    )


def beliefs_from_m24() -> tuple[MechanismBelief, ...]:
    return tuple(belief_from_m24(model) for model in frozen_mechanism_response_models())


def _next_status(
    status: str,
    *,
    in_scope: bool,
    calibration_reuse: bool,
    contradicted: bool,
    critical: bool,
    catalog_supported: bool,
) -> tuple[str, str]:
    if not in_scope:
        return status, "OUT_OF_SCOPE"
    if calibration_reuse:
        return status, "CALIBRATION_REUSE"
    if contradicted:
        reason = "CONFIDENT_CONTRADICTION" if critical else "EVIDENCE_OUTSIDE_OR_SIGN"
        return STATUS_CONTRADICTED, reason
    if status == STATUS_CONTRADICTED:
        return STATUS_CONTRADICTED, "STICKY_CONTRADICTION"
    if not catalog_supported:
        return STATUS_UNCERTAIN, "UNSUPPORTED_CATALOG"
    if status == STATUS_UNCERTAIN:
        return STATUS_UNCERTAIN, "CONSISTENT_IN_SCOPE"
    return STATUS_SUPPORTED, "CONSISTENT_IN_SCOPE"


def _mechanism_scope_matches(belief: MechanismBelief, evidence: EvidenceRecord) -> bool:
    claimed = evidence.claim
    scope = belief.validity_scope.claim
    return (
        evidence.mechanism_id == belief.mechanism_id
        and claimed.model_id == scope.model_id
        and claimed.model_revision == scope.model_revision
        and claimed.intervention_id == scope.intervention_id
        and claimed.layer == scope.layer
        and claimed.hook == scope.hook
        and claimed.direction_id == scope.direction_id
        and claimed.direction_sha256 == scope.direction_sha256
        and claimed.alpha == scope.alpha
        and claimed.response_rule == scope.response_rule
        and claimed.positive_token_id == scope.positive_token_id
        and claimed.negative_token_id == scope.negative_token_id
    )


def _claim_for(model: MechanismResponseModel, catalog_sha256: str) -> MechanismClaim:
    return MechanismClaim(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        intervention_id=model.intervention_id,
        layer=model.layer,
        hook=model.hook,
        direction_id=model.direction_id,
        direction_sha256=model.direction_sha256,
        alpha=model.alpha,
        response_rule=model.response_rule,
        positive_token_id=POSITIVE_TOKEN_ID,
        negative_token_id=NEGATIVE_TOKEN_ID,
        catalog_sha256=catalog_sha256,
    )


def _validation_residuals(intervention_id: str) -> tuple[list[str], list[float]]:
    prompt_ids: list[str] = []
    residuals: list[float] = []
    with _EPISODES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["intervention_id"] != intervention_id or row["partition"] != CALIBRATION_PARTITION:
                continue
            if row["response_rule"] != RESPONSE_RULE:
                raise RuntimeError("calibration row is not prediction = g")
            predicted = float(row["g"])
            if float(row["candidate_prediction"]) != predicted:
                raise RuntimeError("calibration row changed g")
            prompt_ids.append(row["prompt_id"])
            residuals.append(float(row["observed_effect"]) - predicted)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise RuntimeError("calibration prompt ids are not unique")
    return prompt_ids, residuals
