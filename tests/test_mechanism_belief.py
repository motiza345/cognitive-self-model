"""M25 belief bookkeeping. These tests do not load Qwen and do not score evaluation."""

from __future__ import annotations

import csv
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from src.cognitive_self_model.m23.belief import CRITICAL_Z, _sample_sd
from src.cognitive_self_model.mechanism_belief import (
    M24_CATALOG_SHA256,
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
    SURFACE_FAMILY,
    MechanismBelief,
    apply_evidence,
    belief_from_m24,
    contradictory_observed,
    make_evidence,
)
from src.cognitive_self_model.mechanism_response import (
    PreInterventionState,
    frozen_mechanism_response_models,
)

ROOT = Path(__file__).resolve().parents[1]


def _belief() -> MechanismBelief:
    return belief_from_m24(frozen_mechanism_response_models()[0])


def _evidence(belief: MechanismBelief, predicted: float, observed: float, *, catalog: str | None = None, prompt_id: str = "synthetic-01"):
    claim = belief.validity_scope.claim
    if catalog is not None:
        claim = replace(claim, catalog_sha256=catalog)
    return make_evidence(
        evidence_id=f"{prompt_id}:{predicted}:{observed}",
        source_experiment="M25-synthetic",
        artifact="tests",
        claim=claim,
        prompt_id=prompt_id,
        predicted_value=predicted,
        observed_value=observed,
        held_out=True,
        provenance="predeclared synthetic evidence",
    )


def test_initial_belief_uses_validation_residuals_and_stays_supported() -> None:
    model = frozen_mechanism_response_models()[1]
    belief = belief_from_m24(model)
    residuals = []
    prompt_ids = []
    with (ROOT / "reports" / "M24_CONSUMPTION_EPISODES.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["intervention_id"] == model.intervention_id and row["partition"] == "validation":
                prompt_ids.append(row["prompt_id"])
                residuals.append(float(row["observed_effect"]) - float(row["g"]))
    assert belief.status == STATUS_SUPPORTED
    assert belief.version == 1
    assert belief.evidence_history == ()
    assert belief.uncertainty.n == 12
    assert belief.uncertainty.calibration_prompt_ids == tuple(prompt_ids)
    assert belief.uncertainty.residual_sd == pytest.approx(_sample_sd(residuals))
    assert belief.uncertainty.critical_z == CRITICAL_Z
    assert belief.uncertainty.inflation == 1.0
    assert belief.validity_scope.supported_catalog_sha256 == M24_CATALOG_SHA256
    assert belief.validity_scope.surface_family == SURFACE_FAMILY
    assert belief.response_model is model


def test_prediction_has_no_outcome_and_matches_the_response_model() -> None:
    belief = _belief()
    state = PreInterventionState(margin_gradient=belief.response_model.direction)
    assert list(inspect.signature(MechanismBelief.predict).parameters) == ["self", "pre_intervention_state"]
    issued = belief.predict(state)
    assert issued["predicted_effect"] == pytest.approx(belief.response_model.predict(state))
    assert issued["status"] == STATUS_SUPPORTED
    assert issued["uncertainty"] == pytest.approx(belief.uncertainty.effective)


def test_consistent_evidence_keeps_support_and_does_not_change_the_response() -> None:
    belief = _belief()
    before = belief.response_model.predict(PreInterventionState(margin_gradient=belief.response_model.direction))
    updated = apply_evidence(belief, _evidence(belief, 0.2, 0.2))
    assert updated.status == STATUS_SUPPORTED
    assert updated.version == 2
    assert updated.response_model is belief.response_model
    assert updated.uncertainty.residual_sd == belief.uncertainty.residual_sd
    assert updated.uncertainty.inflation == 1.0
    assert len(updated.evidence_history) == 1
    assert updated.contradiction_history == ()
    assert belief.status == STATUS_SUPPORTED
    assert belief.evidence_history == ()
    after = updated.response_model.predict(PreInterventionState(margin_gradient=updated.response_model.direction))
    assert after == pytest.approx(before)


def test_predeclared_contradiction_does_not_refit() -> None:
    belief = _belief()
    narrow = replace(
        belief,
        uncertainty=replace(belief.uncertainty, residual_sd=0.01, inflation=1.0),
    )
    predicted = 0.25
    observed = contradictory_observed(predicted, narrow.uncertainty.effective)
    updated = apply_evidence(narrow, _evidence(narrow, predicted, observed))
    assert updated.status == STATUS_CONTRADICTED
    assert updated.evidence_history[-1].reason == "CONFIDENT_CONTRADICTION"
    assert updated.contradiction_history[-1].reason == "CONFIDENT_CONTRADICTION"
    assert updated.uncertainty.inflation == 2.0
    assert updated.uncertainty.residual_sd == 0.01
    assert updated.response_model is belief.response_model
    assert updated.response_model.training_baseline_mean == belief.response_model.training_baseline_mean
    assert updated.response_model.response_rule == "prediction = g"
    again = apply_evidence(updated, _evidence(updated, predicted, observed, prompt_id="synthetic-02"))
    assert again.uncertainty.inflation == 2.0
    assert again.status == STATUS_CONTRADICTED
    live = apply_evidence(
        belief,
        _evidence(belief, predicted, contradictory_observed(predicted, belief.uncertainty.effective)),
    )
    assert live.status == STATUS_CONTRADICTED
    assert live.uncertainty.residual_sd == belief.uncertainty.residual_sd
    assert live.response_model.direction == belief.response_model.direction


def test_boundary_and_catalog_transitions() -> None:
    belief = replace(
        _belief(),
        uncertainty=replace(_belief().uncertainty, residual_sd=1.0, inflation=1.0),
    )
    on_boundary = apply_evidence(belief, _evidence(belief, 0.0, CRITICAL_Z))
    assert on_boundary.status == STATUS_SUPPORTED
    assert on_boundary.evidence_history[-1].outside_interval is False
    zero_sign = apply_evidence(belief, _evidence(belief, 0.0, 0.1, prompt_id="synthetic-zero"))
    assert zero_sign.evidence_history[-1].sign_mismatch is False
    outside = apply_evidence(belief, _evidence(belief, 0.0, CRITICAL_Z + 1.0, prompt_id="synthetic-out"))
    assert outside.status == STATUS_CONTRADICTED
    assert outside.evidence_history[-1].reason == "EVIDENCE_OUTSIDE_OR_SIGN"
    base = _belief()
    other_catalog = apply_evidence(base, _evidence(base, 0.2, 0.2, catalog="0" * 64))
    assert other_catalog.status == STATUS_UNCERTAIN
    assert other_catalog.validity_scope.supported_catalog_sha256 == M24_CATALOG_SHA256
    still = apply_evidence(other_catalog, _evidence(other_catalog, 0.2, 0.2, prompt_id="synthetic-03"))
    assert still.status == STATUS_UNCERTAIN


def test_out_of_scope_and_calibration_reuse_do_not_change_status() -> None:
    belief = _belief()
    foreign = replace(belief.validity_scope.claim, alpha=2.0, catalog_sha256=M24_CATALOG_SHA256)
    evidence = make_evidence(
        evidence_id="foreign",
        source_experiment="M25-synthetic",
        artifact="tests",
        claim=foreign,
        prompt_id="synthetic-foreign",
        predicted_value=0.2,
        observed_value=-5.0,
        held_out=True,
        provenance="wrong alpha",
    )
    updated = apply_evidence(belief, evidence)
    assert updated.status == STATUS_SUPPORTED
    assert updated.evidence_history[-1].reason == "OUT_OF_SCOPE"
    assert updated.contradiction_history == ()
    reused = apply_evidence(
        belief,
        _evidence(belief, 0.2, -5.0, prompt_id=belief.uncertainty.calibration_prompt_ids[0]),
    )
    assert reused.status == STATUS_SUPPORTED
    assert reused.evidence_history[-1].reason == "CALIBRATION_REUSE"
    sticky = apply_evidence(belief, _evidence(belief, 0.25, contradictory_observed(0.25, belief.uncertainty.effective)))
    kept = apply_evidence(sticky, _evidence(sticky, 0.2, 0.2, prompt_id="synthetic-later"))
    assert kept.status == STATUS_CONTRADICTED
    assert kept.evidence_history[-1].reason == "STICKY_CONTRADICTION"
    assert len(kept.contradiction_history) == 1


def test_module_does_not_grow_a_second_response_rule() -> None:
    source = (ROOT / "src" / "cognitive_self_model" / "mechanism_belief.py").read_text(encoding="utf-8")
    for banned in ("update_belief", "pre_dot", "SelfModelBelief", "slope"):
        assert banned not in source
