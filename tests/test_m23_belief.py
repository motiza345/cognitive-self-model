"""M23 belief, leakage of predict, and the frozen verdict map."""

from __future__ import annotations

import inspect

import pytest

from src.cognitive_self_model.m23.belief import (
    SelfModelBelief,
    belief_from_observations,
    update_belief,
)
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts, prompt_manifest_sha256
from src.cognitive_self_model.m23.report import GENERAL_FORBIDDEN, render_report
from src.cognitive_self_model.m23.score import score_partition
from src.cognitive_self_model.m23.stats import (
    clopper_pearson,
    overall_verdict,
    paired_mean_ci,
    q1_status,
    q4_status,
)


def test_prompt_manifest_is_the_m22_catalog() -> None:
    assert prompt_manifest_sha256() == (
        "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db"
    )
    roles = {record.role for record in frozen_prompts()}
    assert roles == {"discovery", "validation", "replication"}


def test_predict_signature_has_no_outcome() -> None:
    parameters = inspect.signature(SelfModelBelief.predict).parameters
    assert set(parameters) == {"self", "alpha"}


def test_update_does_not_copy_the_observation() -> None:
    belief = belief_from_observations(
        "M22.1-D1-L23",
        "additive_last_token",
        [0.02, 0.03, 0.01, 0.04],
        validity_scope={"status": "IN_SCOPE", "hook": "blocks.23.hook_resid_post"},
    )
    before = belief.predicted_effect
    updated = update_belief(belief, 0.5, "validation-1")
    assert updated.predicted_effect != pytest.approx(0.5)
    assert updated.predicted_effect != pytest.approx(before)
    assert updated.version == belief.version + 1
    assert updated.update_history[-1]["update_reason"]
    assert updated.update_history[-1]["belief_before"]["version"] == belief.version
    assert updated.predict(2.0)["predicted_effect"] == pytest.approx(2.0 * updated.predicted_effect)


def test_confident_contradiction_inflates_uncertainty_once() -> None:
    belief = belief_from_observations(
        "M22.1-D1-L23",
        "additive_last_token",
        [0.20, 0.21, 0.19, 0.22],
        validity_scope={"status": "IN_SCOPE"},
    )
    updated = update_belief(belief, -0.4, "validation-sign")
    assert updated.update_history[-1]["evidence"]["critical"] is True
    assert updated.inflation == 2.0
    again = update_belief(updated, -0.5, "validation-sign-2")
    assert again.inflation == 2.0
    assert again.validity_scope["status"] == "CONTRADICTED"


def test_clopper_pearson_six_of_six_clears_one_half() -> None:
    low, high = clopper_pearson(6, 6)
    assert low == pytest.approx(0.025 ** (1.0 / 6.0), rel=1e-6)
    assert high == 1.0
    assert q1_status("CI_POSITIVE", low, high) == "PASS"
    assert q1_status("CI_INCLUDES_ZERO", low, high) == "INCONCLUSIVE"
    assert q1_status("CI_NEGATIVE", low, high) == "FAIL"


def test_zero_differences_are_not_a_pass() -> None:
    interval = paired_mean_ci([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert interval["class"] == "CI_INCLUDES_ZERO"
    assert q4_status("CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO") == "INCONCLUSIVE"
    assert q4_status("CI_POSITIVE", "CI_POSITIVE") == "FAIL"


def test_overall_pass_requires_every_question_and_blocks_a_general_claim() -> None:
    passed = overall_verdict({name: "PASS" for name in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}, True)
    assert passed["verdict"] == "PASS"
    text = render_report({**passed, "metrics": {}, "reproducibility": {"seed": 23001}})
    assert "limited operational self-model loop" in text
    assert GENERAL_FORBIDDEN not in text

    blocked = overall_verdict(
        {"Q1": "PASS", "Q2": "INCONCLUSIVE", "Q3": "PASS", "Q4": "INCONCLUSIVE", "Q5": "PASS", "Q6": "INCONCLUSIVE"},
        True,
    )
    assert blocked["verdict"] == "INCONCLUSIVE"
    failed = overall_verdict(
        {"Q1": "FAIL", "Q2": "INCONCLUSIVE", "Q3": "PASS", "Q4": "PASS", "Q5": "FAIL", "Q6": "INCONCLUSIVE"},
        True,
        q4_reason=None,
    )
    assert failed["verdict"] == "FAIL"
    assert "PREDICTION_FAILURE" in failed["failure_class"]
    assert "BASELINE_MATCH" in failed["failure_class"]


def test_score_partition_on_a_constant_accurate_belief_is_not_an_automatic_pass() -> None:
    validation = [
        {
            "predicted_effect": 0.03,
            "observed_delta": 0.03,
            "uncertainty": 0.01,
            "critical": False,
        }
        for _ in range(6)
    ]
    replication = [
        {
            "observed_delta": 0.03,
            "predicted_effect": 0.03,
            "no_update_prediction": 0.03,
            "shuffled_prediction": 0.03,
            "outcome_only_prediction": 0.03,
        }
        for _ in range(6)
    ]
    scored = score_partition(validation, replication, replication, q3_pass=True, leakage_ok=True)
    assert scored["questions"]["Q2"] == "INCONCLUSIVE"
    assert scored["questions"]["Q4"] == "INCONCLUSIVE"
    assert scored["verdict"] == "INCONCLUSIVE"
