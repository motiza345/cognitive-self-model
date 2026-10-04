"""Rule checks for the real Qwen claim test.

These checks do not load Qwen and do not write the execution artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.cognitive_self_model.real_claim_self_model import (
    ACTION_ABSTAIN,
    ACTION_INTERVENE,
    EXPECTED_CATALOG_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    SCOPE,
    STATUS_CONTRADICTED,
    STATUS_NONE,
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
    applicable_status,
    build_catalog,
    canonical_sha256,
    clopper_pearson,
    counterfactual_changes_decision,
    decide,
    holdout_decisions,
    initial_claim,
    status_from_effects,
    update_claim,
    verdict,
)

PROTOCOL = Path("reports/REAL_QWEN_CLAIM_SELF_MODEL_PROTOCOL.md")
CATALOG = Path("reports/REAL_QWEN_CLAIM_SELF_MODEL_CATALOG.json")


def test_protocol_and_catalog_hashes_are_sealed() -> None:
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SHA256
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert canonical_sha256(document) == EXPECTED_CATALOG_SHA256
    assert canonical_sha256(build_catalog()) == EXPECTED_CATALOG_SHA256


def test_catalog_is_balanced_and_disjoint() -> None:
    document = build_catalog()
    labels = {(row["partition"], row["label"]) for row in document["prompts"]}
    assert ("EVIDENCE", "YES") in labels
    assert ("EVIDENCE", "NO") in labels
    assert ("HOLDOUT", "YES") in labels
    assert ("HOLDOUT", "NO") in labels
    assert sum(row["partition"] == "VALIDATION" for row in document["prompts"]) == 12


def test_clopper_pearson_edges() -> None:
    lower, upper = clopper_pearson(12, 12)
    assert abs(lower - (0.025 ** (1.0 / 12.0))) < 1e-12
    assert upper == 1.0
    lower, upper = clopper_pearson(0, 12)
    assert lower == 0.0
    assert abs(upper - (1.0 - (0.025 ** (1.0 / 12.0)))) < 1e-12
    summary = status_from_effects([1.0] * 12)
    assert summary["status"] == STATUS_SUPPORTED
    summary = status_from_effects([-1.0] * 12)
    assert summary["status"] == STATUS_CONTRADICTED
    summary = status_from_effects([1.0] * 6 + [-1.0] * 6)
    assert summary["status"] == STATUS_UNCERTAIN
    summary = status_from_effects([0.0] * 12)
    assert summary["positive"] == 0
    assert summary["zeros"] == 12
    assert summary["status"] == STATUS_CONTRADICTED


def test_policy_reads_only_status() -> None:
    assert decide(STATUS_SUPPORTED) == ACTION_INTERVENE
    assert decide(STATUS_UNCERTAIN) == ACTION_ABSTAIN
    assert decide(STATUS_CONTRADICTED) == ACTION_ABSTAIN
    assert decide(STATUS_NONE) == ACTION_ABSTAIN
    assert counterfactual_changes_decision(STATUS_SUPPORTED)["changed"] is True
    assert counterfactual_changes_decision(STATUS_UNCERTAIN)["changed"] is True
    assert counterfactual_changes_decision(STATUS_CONTRADICTED)["changed"] is True


def test_scope_mismatch_abstains() -> None:
    claim = initial_claim()
    assert applicable_status(claim, SCOPE) == STATUS_SUPPORTED
    other = dict(SCOPE)
    other["layer"] = 8
    assert applicable_status(claim, other) == STATUS_NONE
    assert decide(applicable_status(claim, other)) == ACTION_ABSTAIN


def test_update_keeps_the_historical_prior() -> None:
    claim = initial_claim()
    rows = [
        {"partition": "EVIDENCE", "prompt_id": f"RC1-{index:04d}", "observed_effect": -1.0}
        for index in range(1, 13)
    ]
    updated = update_claim(claim, rows)
    assert claim["status"] == STATUS_SUPPORTED
    assert updated["evidence_history"][0]["source_is_current_test_evidence"] is False
    assert updated["status"] == STATUS_CONTRADICTED
    assert updated["version"] == 2
    assert updated["revision_history"][0]["status"] == STATUS_SUPPORTED
    decisions = holdout_decisions(updated, ["RC1-0025"])
    assert set(decisions[0]) == {"claim_id", "claim_version", "decision", "prompt_id", "status"}
    assert decisions[0]["decision"] == ACTION_ABSTAIN


def test_verdict_branches() -> None:
    common = dict(
        qwen_executed=True,
        claim_exact=True,
        evidence_before_holdout=True,
        scope_pass=True,
        history_pass=True,
        decision_from_claim=True,
        causality_pass=True,
        leakage_pass=True,
        immutable_pass=True,
        utility_better=True,
        revision_from_qwen=True,
        stores_evidence=True,
        decisions_equal_always_intervene=False,
    )
    assert verdict(**common) == "REAL_CLAIM_SELF_MODEL_SUPPORTED"
    tied = dict(common)
    tied["utility_better"] = False
    tied["decisions_equal_always_intervene"] = True
    assert verdict(**tied) == "REAL_CLAIM_SELF_MODEL_NOT_SUPPORTED"
    different = dict(common)
    different["utility_better"] = False
    different["decisions_equal_always_intervene"] = False
    assert verdict(**different) == "INCONCLUSIVE"
    leaked = dict(common)
    leaked["leakage_pass"] = False
    assert verdict(**leaked) == "REAL_CLAIM_SELF_MODEL_NOT_SUPPORTED"
