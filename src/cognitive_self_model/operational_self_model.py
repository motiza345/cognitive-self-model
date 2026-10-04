"""Smallest operational self-model.

The decision reads the claim. Evidence can replace that claim. Sealed
predictions stay immutable. The object does not estimate a coefficient
and does not depend on a language-model mechanism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PROTOCOL_VERSION = "OPERATIONAL_SELF_MODEL.1"

CLAIM_USE_LINEAR = "USE_LINEAR"
CLAIM_WITHHOLD = "WITHHOLD"
ACTION_INTERVENE = "INTERVENE"
ACTION_ABSTAIN = "ABSTAIN"

EPISODE_A = "episode-a"
EPISODE_B = "episode-b"
EPISODE_D = "episode-d"
MEASUREMENT_A = 1.0
OUTCOME_A = -1.0
MEASUREMENT_D = 1.0
OUTCOME_D = 2.0
PROVENANCE_A = "synthetic.operational_self_model.episode_a"
PROVENANCE_D = "synthetic.operational_self_model.episode_d"

_CLAIMS = (CLAIM_USE_LINEAR, CLAIM_WITHHOLD)


def sign(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("sign requires a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("sign requires a finite number")
    if number > 0.0:
        return 1
    if number < 0.0:
        return -1
    return 0


def next_claim(claim: str, predicted_sign: int, observed_sign: int) -> str:
    """Return the claim the next decision will read."""
    if claim not in _CLAIMS:
        raise ValueError("claim is not a self-model claim")
    if predicted_sign not in (-1, 0, 1) or observed_sign not in (-1, 0, 1):
        raise ValueError("signs must be -1, 0, or 1")
    if (
        claim == CLAIM_USE_LINEAR
        and predicted_sign != 0
        and observed_sign != 0
        and predicted_sign != observed_sign
    ):
        return CLAIM_WITHHOLD
    return claim


@dataclass(frozen=True)
class DecisionContext:
    episode_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or self.episode_id == "":
            raise ValueError("episode_id must be a non-empty string")


@dataclass(frozen=True)
class PredictionRecord:
    record_id: str
    episode_id: str
    model_version: int
    claim: str
    prediction: float
    order: int
    sealed_before_outcome: bool

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, str) or self.record_id == "":
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(self.episode_id, str) or self.episode_id == "":
            raise ValueError("episode_id must be a non-empty string")
        if self.claim not in _CLAIMS:
            raise ValueError("prediction claim is not a self-model claim")
        if isinstance(self.prediction, bool) or not isinstance(self.prediction, (int, float)):
            raise TypeError("prediction must be a real number")
        if not math.isfinite(float(self.prediction)):
            raise ValueError("prediction must be finite")
        object.__setattr__(self, "prediction", float(self.prediction))
        if not isinstance(self.model_version, int) or isinstance(self.model_version, bool):
            raise TypeError("model_version must be an integer")
        if self.model_version < 1:
            raise ValueError("model_version starts at 1")
        if not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 1:
            raise ValueError("order starts at 1")
        if self.sealed_before_outcome is not True:
            raise ValueError("a prediction record is sealed before the outcome")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    prediction_record_id: str
    observed_outcome: float
    evidence_sign: int
    provenance: str
    order: int

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or self.evidence_id == "":
            raise ValueError("evidence_id must be a non-empty string")
        if not isinstance(self.prediction_record_id, str) or self.prediction_record_id == "":
            raise ValueError("prediction_record_id must be a non-empty string")
        if isinstance(self.observed_outcome, bool) or not isinstance(self.observed_outcome, (int, float)):
            raise TypeError("observed_outcome must be a real number")
        if not math.isfinite(float(self.observed_outcome)):
            raise ValueError("observed_outcome must be finite")
        object.__setattr__(self, "observed_outcome", float(self.observed_outcome))
        if self.evidence_sign != sign(self.observed_outcome):
            raise ValueError("evidence_sign must be the sign of the observed outcome")
        if not isinstance(self.provenance, str) or self.provenance == "":
            raise ValueError("provenance must be a non-empty string")
        if not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 1:
            raise ValueError("order starts at 1")


@dataclass(frozen=True)
class VersionRecord:
    version: int
    claim: str
    evidence_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version starts at 1")
        if self.claim not in _CLAIMS:
            raise ValueError("version claim is not a self-model claim")
        if (
            not isinstance(self.evidence_count, int)
            or isinstance(self.evidence_count, bool)
            or self.evidence_count < 0
        ):
            raise ValueError("evidence_count must be a non-negative integer")


@dataclass(frozen=True)
class SelfModel:
    claim: str
    version: int
    evidence_history: tuple[EvidenceRecord, ...]
    predictions: tuple[PredictionRecord, ...]
    version_history: tuple[VersionRecord, ...]
    next_order: int

    def __post_init__(self) -> None:
        if self.claim not in _CLAIMS:
            raise ValueError("claim is not a self-model claim")
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("version starts at 1")
        if not isinstance(self.evidence_history, tuple):
            raise TypeError("evidence_history must be a tuple")
        if not isinstance(self.predictions, tuple):
            raise TypeError("predictions must be a tuple")
        if not isinstance(self.version_history, tuple):
            raise TypeError("version_history must be a tuple")
        if len(self.version_history) != self.version:
            raise ValueError("version history has one snapshot per version")
        if self.version_history[-1].version != self.version or self.version_history[-1].claim != self.claim:
            raise ValueError("the latest snapshot matches the current claim and version")
        if self.version_history[-1].evidence_count != len(self.evidence_history):
            raise ValueError("the latest snapshot counts the evidence history")
        if not isinstance(self.next_order, int) or isinstance(self.next_order, bool) or self.next_order < 1:
            raise ValueError("next_order starts at 1")
        seen_predictions: set[str] = set()
        for record in self.predictions:
            if not isinstance(record, PredictionRecord):
                raise TypeError("predictions contain PredictionRecord values")
            if record.record_id in seen_predictions:
                raise ValueError("prediction record ids are unique")
            seen_predictions.add(record.record_id)
        seen_evidence: set[str] = set()
        for record in self.evidence_history:
            if not isinstance(record, EvidenceRecord):
                raise TypeError("evidence_history contains EvidenceRecord values")
            if record.evidence_id in seen_evidence:
                raise ValueError("evidence ids are unique")
            seen_evidence.add(record.evidence_id)

    def seal_prediction(
        self,
        *,
        record_id: str,
        episode_id: str,
        measurement: float | None,
    ) -> tuple[SelfModel, PredictionRecord]:
        """Append a sealed prediction. The claim and the version stay."""
        if any(record.record_id == record_id for record in self.predictions):
            raise ValueError("prediction record id already exists")
        if self.claim == CLAIM_USE_LINEAR:
            if measurement is None:
                raise ValueError("USE_LINEAR seals the supplied measurement")
            prediction = float(measurement)
        else:
            if measurement is not None:
                raise ValueError("WITHHOLD does not seal a linear measurement")
            prediction = 0.0
        record = PredictionRecord(
            record_id=record_id,
            episode_id=episode_id,
            model_version=self.version,
            claim=self.claim,
            prediction=prediction,
            order=self.next_order,
            sealed_before_outcome=True,
        )
        sealed = SelfModel(
            claim=self.claim,
            version=self.version,
            evidence_history=self.evidence_history,
            predictions=self.predictions + (record,),
            version_history=self.version_history,
            next_order=self.next_order + 1,
        )
        return sealed, record

    def update(self, evidence: EvidenceRecord) -> SelfModel:
        """Return a new model. Historical predictions and snapshots stay."""
        if not isinstance(evidence, EvidenceRecord):
            raise TypeError("evidence must be an EvidenceRecord")
        if any(record.evidence_id == evidence.evidence_id for record in self.evidence_history):
            raise ValueError("evidence id already exists")
        matches = [record for record in self.predictions if record.record_id == evidence.prediction_record_id]
        if len(matches) != 1:
            raise ValueError("evidence cites a prediction that is not sealed on this model")
        prediction = matches[0]
        if evidence.order != self.next_order or evidence.order <= prediction.order:
            raise ValueError("evidence order must follow the sealed prediction")
        if prediction.claim != self.claim or prediction.model_version != self.version:
            raise ValueError("evidence cites a prediction from another claim or version")
        claim = next_claim(self.claim, sign(prediction.prediction), evidence.evidence_sign)
        snapshot = VersionRecord(
            version=self.version + 1,
            claim=claim,
            evidence_count=len(self.evidence_history) + 1,
        )
        return SelfModel(
            claim=claim,
            version=self.version + 1,
            evidence_history=self.evidence_history + (evidence,),
            predictions=self.predictions,
            version_history=self.version_history + (snapshot,),
            next_order=self.next_order + 1,
        )

    def clone(self) -> SelfModel:
        """Return an equal model with a distinct identity. Does not apply evidence."""
        return SelfModel(
            claim=self.claim,
            version=self.version,
            evidence_history=tuple(self.evidence_history),
            predictions=tuple(self.predictions),
            version_history=tuple(self.version_history),
            next_order=self.next_order,
        )


def initial_self_model() -> SelfModel:
    return SelfModel(
        claim=CLAIM_USE_LINEAR,
        version=1,
        evidence_history=(),
        predictions=(),
        version_history=(VersionRecord(version=1, claim=CLAIM_USE_LINEAR, evidence_count=0),),
        next_order=1,
    )


@dataclass(frozen=True)
class DecisionPolicy:
    def decide(self, self_model: SelfModel, context: DecisionContext) -> str:
        if not isinstance(self_model, SelfModel):
            raise TypeError("decide reads a SelfModel")
        if not isinstance(context, DecisionContext):
            raise TypeError("decide reads a DecisionContext")
        claim = self_model.claim
        if claim == CLAIM_USE_LINEAR:
            return ACTION_INTERVENE
        if claim == CLAIM_WITHHOLD:
            return ACTION_ABSTAIN
        raise ValueError("claim is not a decision claim")


def make_evidence(
    model: SelfModel,
    *,
    evidence_id: str,
    prediction_record_id: str,
    observed_outcome: float,
    provenance: str,
) -> EvidenceRecord:
    """Build evidence against a sealed prediction. Does not update the model."""
    if not isinstance(model, SelfModel):
        raise TypeError("evidence is built from a SelfModel")
    if not any(record.record_id == prediction_record_id for record in model.predictions):
        raise ValueError("evidence cites a prediction that is not sealed on this model")
    return EvidenceRecord(
        evidence_id=evidence_id,
        prediction_record_id=prediction_record_id,
        observed_outcome=float(observed_outcome),
        evidence_sign=sign(observed_outcome),
        provenance=provenance,
        order=model.next_order,
    )


def _prediction_snapshot(record: PredictionRecord) -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "episode_id": record.episode_id,
        "model_version": record.model_version,
        "claim": record.claim,
        "prediction": record.prediction,
        "order": record.order,
        "sealed_before_outcome": record.sealed_before_outcome,
    }


def _evidence_snapshot(record: EvidenceRecord) -> dict[str, object]:
    return {
        "evidence_id": record.evidence_id,
        "prediction_record_id": record.prediction_record_id,
        "observed_outcome": record.observed_outcome,
        "evidence_sign": record.evidence_sign,
        "provenance": record.provenance,
        "order": record.order,
    }


def _version_snapshot(record: VersionRecord) -> dict[str, object]:
    return {
        "version": record.version,
        "claim": record.claim,
        "evidence_count": record.evidence_count,
    }


def acceptance_verdict(report: dict[str, object]) -> str:
    """Label the closed loop. No error threshold."""
    well_formed = (
        report["initial_claim"] == CLAIM_USE_LINEAR
        and report["initial_version"] == 1
        and report["sealed_before_outcome"] is True
        and int(report["prediction_order"]) < int(report["evidence_order"])
        and report["opposite_nonzero"] is True
        and report["held_out_distinct"] is True
        and report["clone_equals_pre_update"] is True
        and report["agreeing_same_nonzero_sign"] is True
    )
    if not well_formed:
        return "INCONCLUSIVE"
    causal = (
        report["initial_decision"] == ACTION_INTERVENE
        and report["updated_decision"] == ACTION_ABSTAIN
        and report["no_update_decision"] == ACTION_INTERVENE
        and report["updated_decision"] != report["no_update_decision"]
        and report["updated_claim"] == CLAIM_WITHHOLD
        and int(report["updated_version"]) > int(report["initial_version"])
        and report["evidence_history_grew"] is True
        and report["agreeing_claim"] == CLAIM_USE_LINEAR
        and report["prediction_unchanged"] is True
        and report["evidence_unchanged"] is True
        and report["prior_version_unchanged"] is True
        and report["update_returns_new_object"] is True
        and report["changed_fields"]
        == ["claim", "version", "evidence_history", "version_history", "next_order"]
    )
    if causal:
        return "SELF_MODEL_OPERATIONAL"
    return "SELF_MODEL_NOT_OPERATIONAL"


def run_acceptance() -> dict[str, object]:
    """Execute the frozen synthetic episodes. Does not write files."""
    policy = DecisionPolicy()
    initial = initial_self_model()
    context_a = DecisionContext(EPISODE_A)
    context_b = DecisionContext(EPISODE_B)
    initial_decision = policy.decide(initial, context_a)
    sealed, prediction = initial.seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=MEASUREMENT_A,
    )
    prediction_before = _prediction_snapshot(prediction)
    version_before = _version_snapshot(sealed.version_history[0])
    evidence = make_evidence(
        sealed,
        evidence_id="evidence-a",
        prediction_record_id=prediction.record_id,
        observed_outcome=OUTCOME_A,
        provenance=PROVENANCE_A,
    )
    evidence_before = _evidence_snapshot(evidence)
    updated = sealed.update(evidence)
    control = sealed.clone()
    updated_decision = policy.decide(updated, context_b)
    no_update_decision = policy.decide(control, context_b)

    agreeing = initial_self_model()
    agreeing_sealed, agreeing_prediction = agreeing.seal_prediction(
        record_id="prediction-d",
        episode_id=EPISODE_D,
        measurement=MEASUREMENT_D,
    )
    agreeing_evidence = make_evidence(
        agreeing_sealed,
        evidence_id="evidence-d",
        prediction_record_id=agreeing_prediction.record_id,
        observed_outcome=OUTCOME_D,
        provenance=PROVENANCE_D,
    )
    agreeing_updated = agreeing_sealed.update(agreeing_evidence)

    changed = [
        name
        for name, left, right in (
            ("claim", sealed.claim, updated.claim),
            ("version", sealed.version, updated.version),
            ("evidence_history", sealed.evidence_history, updated.evidence_history),
            ("predictions", sealed.predictions, updated.predictions),
            ("version_history", sealed.version_history, updated.version_history),
            ("next_order", sealed.next_order, updated.next_order),
        )
        if left != right
    ]
    report: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "initial_claim": initial.claim,
        "initial_version": initial.version,
        "initial_decision": initial_decision,
        "sealed_before_outcome": prediction.sealed_before_outcome,
        "prediction_order": prediction.order,
        "evidence_order": evidence.order,
        "prediction_value": prediction.prediction,
        "observed_outcome": evidence.observed_outcome,
        "opposite_nonzero": (
            sign(prediction.prediction) != 0
            and evidence.evidence_sign != 0
            and sign(prediction.prediction) != evidence.evidence_sign
        ),
        "held_out_episode": EPISODE_B,
        "evidence_episode": prediction.episode_id,
        "held_out_distinct": EPISODE_B != prediction.episode_id,
        "clone_equals_pre_update": control == sealed and control is not sealed,
        "updated_claim": updated.claim,
        "updated_version": updated.version,
        "updated_decision": updated_decision,
        "updated_evidence_count": len(updated.evidence_history),
        "pre_update_evidence_count": len(sealed.evidence_history),
        "evidence_history_grew": len(updated.evidence_history) > len(sealed.evidence_history),
        "no_update_claim": control.claim,
        "no_update_version": control.version,
        "no_update_decision": no_update_decision,
        "agreeing_same_nonzero_sign": (
            sign(agreeing_prediction.prediction) != 0
            and agreeing_evidence.evidence_sign != 0
            and sign(agreeing_prediction.prediction) == agreeing_evidence.evidence_sign
        ),
        "agreeing_claim": agreeing_updated.claim,
        "agreeing_version": agreeing_updated.version,
        "prediction_unchanged": _prediction_snapshot(prediction) == prediction_before,
        "evidence_unchanged": _evidence_snapshot(evidence) == evidence_before,
        "prior_version_unchanged": (
            sealed.claim == CLAIM_USE_LINEAR
            and sealed.version == 1
            and _version_snapshot(updated.version_history[0]) == version_before
        ),
        "update_returns_new_object": updated is not sealed,
        "changed_fields": changed,
        "decisions_differ": updated_decision != no_update_decision,
    }
    report["verdict"] = acceptance_verdict(report)
    return report
