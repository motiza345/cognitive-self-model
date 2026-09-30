"""Belief-update rules. No Qwen download and no write into the frozen P run."""

import pytest

from src.mrsm import HEADS
from src.mrsm.belief_update import (
    ABSTAIN,
    ADD_INTERACTION,
    COLLAPSES,
    CONTROL_EXPECTATION,
    HOLD,
    INCONCLUSIVE,
    PRIMARY_EXPECTATION,
    REVISE_HYPOTHESIS,
    REVISE_SCOPE,
    SEPARATES,
    build_controls,
    build_primary_cases,
    classify,
    decide,
    preregistration_document,
    tolerance_of,
)
from src.mrsm.self_model import LeakageError, pair_name


def _belief(uncertainty: float = 5.0) -> dict:
    effects = {name: 1.0 for name in HEADS}
    joints = {}
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            name = pair_name(left, right)
            interaction = -14.0 if name == "L0H0+L1H1" else 0.0
            joints[name] = effects[left] + effects[right] + interaction
    return {
        "scope": "P:test",
        "effects": effects,
        "joints": joints,
        "mechanism_ordered": ["L0H0", "L1H1"],
        "uncertainty": uncertainty,
        "min_abs_interaction": 1.0e-4,
        "min_relative_gap": 0.05,
    }


def test_preregistration_locks_the_rule():
    document = preregistration_document()
    assert document["decisions"] == [
        "UPDATE_RULE_SEPARATES_FAILURES",
        "UPDATE_RULE_COLLAPSES_FAILURES",
        "INCONCLUSIVE",
    ]
    assert "modify SelfModelCore" in document["does_not"]
    assert "load Qwen" in document["does_not"]
    assert "start M22" in document["does_not"]
    assert "use M18.7 as evidence" in document["does_not"]
    assert document["tolerance"]["new_threshold"] is False
    assert document["construction_clearance"]["decision_input"] is False
    assert set(PRIMARY_EXPECTATION) == {"hypothesis", "scope", "interaction", "none"}
    assert tolerance_of(5.238169292541669) == pytest.approx(10.476338585083338)


def test_four_injections_separate_on_a_synthetic_belief():
    belief = _belief()
    cases = build_primary_cases(belief)
    assert cases["interaction"] is not None
    found = {name: classify(belief, records) for name, records in cases.items()}
    assert found == PRIMARY_EXPECTATION
    controls = build_controls(belief, [], cases["hypothesis"])
    controls["holdout"] = [
        {
            "intervention_id": "single:L0H0",
            "actual_outcome": belief["effects"]["L0H0"],
            "scope": belief["scope"],
        }
    ]
    assert {name: classify(belief, records) for name, records in controls.items()} == CONTROL_EXPECTATION


def test_scope_string_without_a_residual_does_not_revise():
    belief = _belief()
    assert classify(belief, build_controls(belief, [], [])["scope_string_alone"]) == HOLD


def test_two_contradictions_abstain():
    belief = _belief()
    cases = build_primary_cases(belief)
    double = build_controls(belief, [], cases["hypothesis"])["double"]
    assert classify(belief, double) == ABSTAIN


def test_poisoned_record_is_rejected():
    belief = _belief()
    with pytest.raises(LeakageError):
        classify(
            belief,
            [
                {
                    "intervention_id": "single:L0H0",
                    "actual_outcome": 1.0,
                    "scope": belief["scope"],
                    "failure_type": "hypothesis",
                }
            ],
        )


def test_clearance_is_not_searched_when_no_pair_can_move():
    cases = build_primary_cases(_belief(uncertainty=100.0))
    assert cases["interaction"] is None
    decision = decide(
        {name: REVISE_HYPOTHESIS for name in PRIMARY_EXPECTATION},
        {name: HOLD for name in CONTROL_EXPECTATION},
        reconstruction_ok=True,
        tolerance_ok=True,
        construction_defined=False,
    )
    assert decision["decision"] == INCONCLUSIVE


def test_decision_order_does_not_fall_through():
    separated = dict(PRIMARY_EXPECTATION)
    controls = dict(CONTROL_EXPECTATION)
    assert (
        decide(
            separated,
            controls,
            reconstruction_ok=True,
            tolerance_ok=True,
            construction_defined=True,
        )["decision"]
        == SEPARATES
    )
    collapsed = dict(separated)
    collapsed["scope"] = REVISE_HYPOTHESIS
    bad_controls = dict(controls)
    bad_controls["holdout"] = ABSTAIN
    assert (
        decide(
            collapsed,
            bad_controls,
            reconstruction_ok=True,
            tolerance_ok=True,
            construction_defined=True,
        )["decision"]
        == COLLAPSES
    )
    assert (
        decide(
            separated,
            bad_controls,
            reconstruction_ok=True,
            tolerance_ok=True,
            construction_defined=True,
        )["decision"]
        == INCONCLUSIVE
    )


def test_matching_table_holds_and_unknown_id_abstains():
    belief = _belief()
    matched = build_controls(belief, [], [])["scope_string_alone"]
    in_scope = [{**row, "scope": belief["scope"]} for row in matched]
    assert classify(belief, in_scope) == HOLD
    assert (
        classify(
            belief,
            [{"intervention_id": "single:NO_SUCH", "actual_outcome": 0.0, "scope": belief["scope"]}],
        )
        == ABSTAIN
    )
    assert ADD_INTERACTION in PRIMARY_EXPECTATION.values()
    assert REVISE_SCOPE in PRIMARY_EXPECTATION.values()
