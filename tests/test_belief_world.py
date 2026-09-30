"""World-change rules. No training and no checkpoint load."""

import torch

from src.mrsm.belief_update import ABSTAIN, ADD_INTERACTION, HOLD, REVISE_HYPOTHESIS, REVISE_SCOPE, classify
from src.mrsm.belief_world import (
    INCONCLUSIVE,
    PRIMARY_EXPECTATION,
    WORLD_DOES_NOT_SEPARATE,
    WORLD_SEPARATES,
    decide_world,
    declare_from_belief,
    penalty_amount,
    preregistration_document,
    score_records,
    target_penalty,
)


def _belief() -> dict:
    joints = {"L0H0+L0H1": 2.0, "L0H0+L1H1": -12.0}
    return {
        "scope": "P:test",
        "effects": {"L0H0": 1.0, "L0H1": 1.0, "L1H1": 1.0},
        "joints": joints,
        "mechanism_ordered": ["L0H0", "L1H1"],
        "uncertainty": 5.0,
        "min_abs_interaction": 1.0e-4,
        "min_relative_gap": 0.05,
    }


def test_preregistration_locks_the_worlds_and_the_rule():
    document = preregistration_document()
    assert document["rule"] == "src.mrsm.belief_update.classify"
    assert document["rule_modified"] is False
    assert document["magnitude_is_searched"] is False
    assert document["ceiling_search"] is False
    assert document["ground_truth_file_loaded"] is False
    assert "modify belief_update.classify" in document["does_not"]
    assert "load Qwen" in document["does_not"]
    assert "start M22" in document["does_not"]
    assert "finetune the frozen checkpoint" in document["does_not"]
    assert "choose a new magnitude after scoring" in document["does_not"]
    assert set(PRIMARY_EXPECTATION) == {"hypothesis", "scope", "interaction", "none"}


def test_declaration_uses_the_frozen_winner_and_does_not_search():
    declaration = declare_from_belief(_belief())
    assert declaration["winner_pair"] == "L0H0+L1H1"
    assert declaration["interaction_pair"] == "L0H0+L0H1"
    assert declaration["magnitude"] == 14.0
    assert declaration["hypothesis_heads"] == ["L0H2", "L1H2"]
    assert declaration["none_head"] == "L0H3"


def test_penalty_is_structural():
    declaration = declare_from_belief(_belief())
    assert penalty_amount("interaction", frozenset({"L0H0", "L0H1"}), declaration) == 14.0
    assert penalty_amount("interaction", frozenset({"L0H0"}), declaration) == 0.0
    assert penalty_amount("none", frozenset({"L0H3"}), declaration) == 14.0
    assert penalty_amount("none", frozenset({"L0H2", "L1H2"}), declaration) == 0.0
    assert penalty_amount("hypothesis", frozenset({"L0H2", "L1H2"}), declaration) == 0.0
    logits = torch.tensor([[5.0, 1.0], [4.0, 2.0]])
    targets = torch.tensor([0, 1])
    shifted = target_penalty(logits, targets, 2.0)
    assert torch.equal(shifted, torch.tensor([[3.0, 1.0], [4.0, 0.0]]))
    assert torch.equal(target_penalty(logits, targets, 0.0), logits)


def test_score_records_is_the_existing_rule():
    assert score_records is not None
    assert score_records.__wrapped__ if hasattr(score_records, "__wrapped__") else score_records
    belief = {
        "scope": "P:test",
        "effects": {"L0H0": 1.0},
        "joints": {},
        "mechanism_ordered": ["L0H0", "L1H1"],
        "uncertainty": 5.0,
        "min_abs_interaction": 1.0e-4,
        "min_relative_gap": 0.05,
    }
    # Unknown intervention abstains inside classify. This call must be that function.
    label = score_records(
        {
            **belief,
            "effects": {name: 0.0 for name in ["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3"]},
            "joints": {},
        },
        [{"intervention_id": "single:NO_SUCH", "actual_outcome": 0.0, "scope": "P:test"}],
    )
    direct = classify(
        {
            "scope": "P:test",
            "effects": {name: 0.0 for name in ["L0H0", "L0H1", "L0H2", "L0H3", "L1H0", "L1H1", "L1H2", "L1H3"]},
            "joints": {},
            "mechanism_ordered": ["L0H0", "L1H1"],
            "uncertainty": 5.0,
            "min_abs_interaction": 1.0e-4,
            "min_relative_gap": 0.05,
        },
        [{"intervention_id": "single:NO_SUCH", "actual_outcome": 0.0, "scope": "P:test"}],
    )
    assert label == direct == ABSTAIN


def test_decision_order():
    primary = dict(PRIMARY_EXPECTATION)
    controls = {"holdout": HOLD}
    kwargs = {
        "reconstruction_ok": True,
        "tolerance_ok": True,
        "instrument_ok": True,
        "finite_ok": True,
    }
    assert decide_world(primary, controls, **kwargs)["decision"] == WORLD_SEPARATES
    missed = dict(primary)
    missed["interaction"] = REVISE_HYPOTHESIS
    assert decide_world(missed, controls, **kwargs)["decision"] == WORLD_DOES_NOT_SEPARATE
    assert decide_world(primary, {"holdout": ABSTAIN}, **kwargs)["decision"] == INCONCLUSIVE
    assert (
        decide_world(primary, controls, reconstruction_ok=True, tolerance_ok=True, instrument_ok=False, finite_ok=True)[
            "decision"
        ]
        == INCONCLUSIVE
    )
    assert REVISE_SCOPE == PRIMARY_EXPECTATION["scope"]
    assert ADD_INTERACTION == PRIMARY_EXPECTATION["interaction"]
