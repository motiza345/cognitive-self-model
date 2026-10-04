"""Synthetic loop for the operational self-model.

The tests check that a decision reads the claim, and that evidence can
change that claim. They do not score prediction error.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.cognitive_self_model.operational_self_model import (
    ACTION_ABSTAIN,
    ACTION_INTERVENE,
    CLAIM_USE_LINEAR,
    CLAIM_WITHHOLD,
    EPISODE_A,
    EPISODE_B,
    DecisionContext,
    DecisionPolicy,
    EvidenceRecord,
    PredictionRecord,
    SelfModel,
    acceptance_verdict,
    initial_self_model,
    make_evidence,
    next_claim,
    run_acceptance,
    sign,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "cognitive_self_model" / "operational_self_model.py"
SCRIPT_PATH = ROOT / "scripts" / "run_operational_self_model.py"
FORBIDDEN = (
    "qwen",
    "torch",
    "transformer_lens",
    "mechanism_belief",
    "mechanism_response",
    "residual_update",
    "m29_pre_evidence",
    "rho",
    "kappa",
    "inflation",
)


def test_decision_reads_the_claim_and_ignores_the_episode() -> None:
    policy = DecisionPolicy()
    linear = initial_self_model()
    withheld, _ = linear.seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=1.0,
    )
    withheld = withheld.update(
        make_evidence(
            withheld,
            evidence_id="evidence-a",
            prediction_record_id="prediction-a",
            observed_outcome=-1.0,
            provenance="unit",
        )
    )
    assert withheld.claim == CLAIM_WITHHOLD
    first = policy.decide(linear, DecisionContext("one"))
    second = policy.decide(linear, DecisionContext("two"))
    assert first == second == ACTION_INTERVENE
    assert policy.decide(withheld, DecisionContext("one")) == ACTION_ABSTAIN
    assert policy.decide(withheld, DecisionContext("two")) == ACTION_ABSTAIN


def test_opposite_nonzero_evidence_changes_only_the_consumed_claim() -> None:
    model, prediction = initial_self_model().seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=-3.0,
    )
    before = (model.claim, model.version, model.predictions, model.evidence_history)
    updated = model.update(
        make_evidence(
            model,
            evidence_id="evidence-a",
            prediction_record_id=prediction.record_id,
            observed_outcome=4.0,
            provenance="unit",
        )
    )
    assert updated.claim == CLAIM_WITHHOLD
    assert updated.version == model.version + 1
    assert len(updated.evidence_history) == len(model.evidence_history) + 1
    assert len(updated.version_history) == len(model.version_history) + 1
    assert updated.predictions == model.predictions
    assert before == (CLAIM_USE_LINEAR, 1, model.predictions, ())


def test_sign_agreement_and_zero_leave_the_claim() -> None:
    fresh = initial_self_model()
    agreed, prediction = fresh.seal_prediction(
        record_id="prediction-d",
        episode_id="episode-d",
        measurement=1.0,
    )
    agreed = agreed.update(
        make_evidence(
            agreed,
            evidence_id="evidence-d",
            prediction_record_id=prediction.record_id,
            observed_outcome=2.0,
            provenance="unit",
        )
    )
    assert agreed.claim == CLAIM_USE_LINEAR
    assert agreed.version == 2

    zero_model, zero_prediction = initial_self_model().seal_prediction(
        record_id="prediction-zero",
        episode_id="episode-zero",
        measurement=1.0,
    )
    zero_updated = zero_model.update(
        make_evidence(
            zero_model,
            evidence_id="evidence-zero",
            prediction_record_id=zero_prediction.record_id,
            observed_outcome=0.0,
            provenance="unit",
        )
    )
    assert zero_updated.claim == CLAIM_USE_LINEAR


def test_withhold_does_not_return_to_the_linear_claim() -> None:
    model, prediction = initial_self_model().seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=1.0,
    )
    withheld = model.update(
        make_evidence(
            model,
            evidence_id="evidence-a",
            prediction_record_id=prediction.record_id,
            observed_outcome=-1.0,
            provenance="unit",
        )
    )
    sealed, _ = withheld.seal_prediction(
        record_id="prediction-b",
        episode_id=EPISODE_B,
        measurement=None,
    )
    assert sealed.predictions[-1].prediction == 0.0
    assert sealed.claim == CLAIM_WITHHOLD
    with pytest.raises(ValueError, match="another claim or version"):
        sealed.update(
            EvidenceRecord(
                evidence_id="evidence-b",
                prediction_record_id=prediction.record_id,
                observed_outcome=5.0,
                evidence_sign=1,
                provenance="unit",
                order=sealed.next_order,
            )
        )


def test_history_and_the_sealed_prediction_stay_immutable() -> None:
    model, prediction = initial_self_model().seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=1.0,
    )
    evidence = make_evidence(
        model,
        evidence_id="evidence-a",
        prediction_record_id=prediction.record_id,
        observed_outcome=-1.0,
        provenance="unit",
    )
    updated = model.update(evidence)
    assert prediction.prediction == 1.0
    assert prediction.sealed_before_outcome is True
    assert evidence.observed_outcome == -1.0
    assert model.version_history[0] == updated.version_history[0]
    assert model.claim == CLAIM_USE_LINEAR
    assert model.version == 1
    assert updated is not model
    assert "observed_outcome" not in PredictionRecord.__dataclass_fields__


def test_no_update_clone_keeps_the_original_decision() -> None:
    policy = DecisionPolicy()
    sealed, prediction = initial_self_model().seal_prediction(
        record_id="prediction-a",
        episode_id=EPISODE_A,
        measurement=1.0,
    )
    evidence = make_evidence(
        sealed,
        evidence_id="evidence-a",
        prediction_record_id=prediction.record_id,
        observed_outcome=-1.0,
        provenance="unit",
    )
    control = sealed.clone()
    updated = sealed.update(evidence)
    assert control == sealed
    assert control is not sealed
    assert policy.decide(control, DecisionContext(EPISODE_B)) == ACTION_INTERVENE
    assert policy.decide(updated, DecisionContext(EPISODE_B)) == ACTION_ABSTAIN
    assert evidence.evidence_id
    assert control.evidence_history == ()


def test_records_reject_an_unsealed_prediction_and_a_mismatched_sign() -> None:
    with pytest.raises(ValueError, match="sealed before the outcome"):
        PredictionRecord(
            record_id="prediction-a",
            episode_id=EPISODE_A,
            model_version=1,
            claim=CLAIM_USE_LINEAR,
            prediction=1.0,
            order=1,
            sealed_before_outcome=False,
        )
    with pytest.raises(ValueError, match="evidence_sign"):
        EvidenceRecord(
            evidence_id="evidence-a",
            prediction_record_id="prediction-a",
            observed_outcome=-1.0,
            evidence_sign=1,
            provenance="unit",
            order=2,
        )


def test_transition_table_is_deterministic() -> None:
    assert next_claim(CLAIM_USE_LINEAR, 1, -1) == CLAIM_WITHHOLD
    assert next_claim(CLAIM_USE_LINEAR, -1, 1) == CLAIM_WITHHOLD
    assert next_claim(CLAIM_USE_LINEAR, 1, 1) == CLAIM_USE_LINEAR
    assert next_claim(CLAIM_USE_LINEAR, 1, 0) == CLAIM_USE_LINEAR
    assert next_claim(CLAIM_USE_LINEAR, 0, -1) == CLAIM_USE_LINEAR
    assert next_claim(CLAIM_WITHHOLD, 1, -1) == CLAIM_WITHHOLD
    assert sign(0.0) == 0


def test_acceptance_loop_is_operational() -> None:
    report = run_acceptance()
    assert report["initial_decision"] == ACTION_INTERVENE
    assert report["updated_decision"] == ACTION_ABSTAIN
    assert report["no_update_decision"] == ACTION_INTERVENE
    assert report["decisions_differ"] is True
    assert report["verdict"] == "SELF_MODEL_OPERATIONAL"
    assert acceptance_verdict({"initial_claim": "MISSING"}) == "INCONCLUSIVE"


def test_module_and_runner_stay_independent_of_prior_mechanisms() -> None:
    module_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    script_tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    def imported(tree: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names.add(node.module)
        return names

    assert imported(module_tree) <= {"__future__", "math", "dataclasses"}
    assert "src.cognitive_self_model.operational_self_model" in imported(script_tree)
    source = (MODULE_PATH.read_text(encoding="utf-8") + SCRIPT_PATH.read_text(encoding="utf-8")).lower()
    for name in FORBIDDEN:
        assert name not in source
    assert not hasattr(SelfModel, "inflation")
