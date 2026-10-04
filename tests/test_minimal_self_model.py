"""Unit checks for the sealed minimal architecture test.

These checks do not write the execution artifacts and do not load a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.cognitive_self_model.minimal_self_model import (
    ABSTAIN,
    ACTION_X,
    EXPECTED_PROTOCOL_SHA256,
    STATE_A,
    STATE_B,
    STATE_C,
    STATUS_CONTRADICTED,
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
    ClaimRegistry,
    FixedPolicy,
    LegacyTwoState,
    claim_view,
    classify_evidence,
    primary_episode_ids,
)

PROTOCOL = Path("reports/MINIMAL_SELF_MODEL_ARCHITECTURE_TEST_PROTOCOL.md")
MODULE = Path("src/cognitive_self_model/minimal_self_model.py")


def _packet(evidence_id: str, state: str, outcome: float) -> dict:
    return {
        "action": ACTION_X,
        "evidence_id": evidence_id,
        "measured_state": state,
        "outcome": outcome,
    }


def test_protocol_hash_is_the_sealed_value() -> None:
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == EXPECTED_PROTOCOL_SHA256


def test_module_does_not_import_a_language_model() -> None:
    text = MODULE.read_text(encoding="utf-8")
    assert "torch" not in text
    assert "transformer_lens" not in text
    assert "qwen" not in text.casefold()


def test_classifier_matches_the_sealed_cuts() -> None:
    assert classify_evidence(1.0) == ("CONFIRMING", "strong")
    assert classify_evidence(0.5) == ("CONFIRMING", "weak")
    assert classify_evidence(0.25) == ("CONFIRMING", "weak")
    assert classify_evidence(0.0) == ("AMBIGUOUS", "none")
    assert classify_evidence(-1.0) == ("CONTRADICTORY", "strong")


def test_primary_rule_selects_state_transitions() -> None:
    assert primary_episode_ids() == ["P2", "P6", "P7"]


def test_b0_ignores_measurements() -> None:
    agent = FixedPolicy()
    agent.observe(_packet("U1", STATE_A, 1.0))
    assert agent.decide(STATE_A) == ACTION_X
    assert agent.decide(STATE_C) == ACTION_X


def test_b1_withhold_is_absorbing_and_has_no_scope() -> None:
    agent = LegacyTwoState()
    agent.observe(_packet("U1", STATE_A, 1.0))
    assert agent.decide(STATE_A) == ACTION_X
    assert agent.decide(STATE_C) == ACTION_X
    agent.observe(_packet("U2", STATE_A, 0.0))
    assert agent.claim == "WITHHOLD"
    assert agent.decide(STATE_A) == ABSTAIN
    assert agent.decide(STATE_B) == ABSTAIN
    agent.observe(_packet("U4", STATE_B, 0.25))
    assert agent.claim == "WITHHOLD"


def test_extra_packet_field_is_rejected() -> None:
    packet = _packet("U1", STATE_A, 1.0)
    packet["ground_truth_effect"] = 1.0
    with pytest.raises(ValueError):
        ClaimRegistry(use_scope=True).observe(packet)


def test_ambiguous_evidence_does_not_contradict() -> None:
    model = ClaimRegistry(use_scope=True)
    model.observe(_packet("U1", STATE_A, 1.0))
    model.observe(_packet("U2", STATE_A, 0.0))
    claim = claim_view(model)[0]
    assert claim["status"] == STATUS_UNCERTAIN
    assert claim["status"] != STATUS_CONTRADICTED
    assert claim["uncertainty"] == 0.70
    assert claim["revision_history"][0]["status"] == STATUS_SUPPORTED
    assert model.decide(STATE_A) == ABSTAIN


def test_scope_blocks_transfer_and_preserves_the_other_claim() -> None:
    scoped = ClaimRegistry(use_scope=True)
    unscoped = ClaimRegistry(use_scope=False)
    for model in (scoped, unscoped):
        model.observe(_packet("U1", STATE_A, 1.0))
    assert scoped.decide(STATE_A) == ACTION_X
    assert scoped.decide(STATE_C) == ABSTAIN
    assert unscoped.decide(STATE_C) == ACTION_X
    scoped.observe(_packet("U2", STATE_A, 0.0))
    scoped.observe(_packet("U3", STATE_A, -1.0))
    before = claim_view(scoped)
    scoped.observe(_packet("U4", STATE_B, 0.25))
    after = claim_view(scoped)
    state_a_before = next(row for row in before if row["scope"] == STATE_A)
    state_a_after = next(row for row in after if row["scope"] == STATE_A)
    state_b = next(row for row in after if row["scope"] == STATE_B)
    assert state_a_before == state_a_after
    assert state_a_after["status"] == STATUS_CONTRADICTED
    assert state_b["status"] == STATUS_SUPPORTED
    assert scoped.decide(STATE_B) == ACTION_X
    assert scoped.decide(STATE_A) == ABSTAIN
    assert scoped.decide(STATE_C) == ABSTAIN
    unscoped.observe(_packet("U2", STATE_A, 0.0))
    unscoped.observe(_packet("U3", STATE_A, -1.0))
    unscoped.observe(_packet("U4", STATE_B, 0.25))
    assert unscoped.decide(STATE_B) == ABSTAIN
    assert claim_view(unscoped)[0]["status"] == STATUS_CONTRADICTED
    assert claim_view(unscoped)[0]["scope"] == "GLOBAL"


def test_evidence_history_prefix_is_stable() -> None:
    model = ClaimRegistry(use_scope=True)
    model.observe(_packet("U1", STATE_A, 1.0))
    first = model.evidence_history
    model.observe(_packet("U3", STATE_A, -1.0))
    assert model.evidence_history[0] == first[0]
    assert model.predictions == ()


def test_bypass_probe_follows_the_registry() -> None:
    model = ClaimRegistry(use_scope=True)
    assert model.decide(STATE_A) == ABSTAIN
    assert model.decide(STATE_B) == ABSTAIN
    assert model.decide(STATE_C) == ABSTAIN
    model.add_claim(STATE_C, STATUS_SUPPORTED, 0.15)
    assert model.decide(STATE_C) == ACTION_X
    assert model.decide(STATE_A) == ABSTAIN
    assert model.decide(STATE_B) == ABSTAIN
