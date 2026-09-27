"""Exhaustive checks for the frozen DEC-010 terminal decision table."""

from __future__ import annotations

import re
from collections import Counter
from itertools import product
from pathlib import Path

import yaml

from src.cognitive_self_model.control_plane.decision_table import GATE_KEYS, terminal_decision

_ROOT = Path(__file__).resolve().parents[1]
_DEC010 = _ROOT / "control_plane" / "DEC-010_decision_table.md"
_FROZEN_RULES = (
    ("P.H1 is FAIL, or P.H1 is PASS and P.H2 is FAIL, or P.H1 is NOT_EVALUATED and P.H2 is FAIL", "STOP"),
    ("P.H1 and P.H2 are PASS and P.H3 is FAIL", "REDEFINE_CAUSAL_AUDIT"),
    ("P.H1-H3 are PASS and P.H4 is FAIL", "REDEFINE_CAUSAL_AUDIT"),
    ("P.H1 and P.H2 are PASS, P.H3 is NOT_EVALUATED, and P.H4 is FAIL", "REDEFINE_CAUSAL_AUDIT"),
    ("P.H1-H4 are PASS and any Q.H1-H4 is NOT_EVALUATED", "INCOMPLETE_BUDGET"),
    ("P.H1-H4 are PASS and all Q.H1-H4 are evaluated and any is FAIL", "REDEFINE_SCALE"),
    ("P.H1-H4 and Q.H1-H4 are PASS", "GO"),
    ("any of P.H1-H4 is NOT_EVALUATED", "INCOMPLETE_BUDGET"),
)
_RULE_WITNESSES = (
    {"P.H1": "NOT_EVALUATED", "P.H2": "FAIL", "P.H3": "PASS", "P.H4": "PASS", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "FAIL", "P.H4": "PASS", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "PASS", "P.H4": "FAIL", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "NOT_EVALUATED", "P.H4": "FAIL", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "PASS", "P.H4": "PASS", "Q.H1": "FAIL", "Q.H2": "NOT_EVALUATED", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "PASS", "P.H4": "PASS", "Q.H1": "FAIL", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "PASS", "P.H2": "PASS", "P.H3": "PASS", "P.H4": "PASS", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
    {"P.H1": "NOT_EVALUATED", "P.H2": "PASS", "P.H3": "PASS", "P.H4": "PASS", "Q.H1": "PASS", "Q.H2": "PASS", "Q.H3": "PASS", "Q.H4": "PASS"},
)

_LEGAL = ("PASS", "FAIL", "NOT_EVALUATED")
_P_KEYS = ("P.H1", "P.H2", "P.H3", "P.H4")
_Q_KEYS = ("Q.H1", "Q.H2", "Q.H3", "Q.H4")


def _record(values: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(GATE_KEYS, values))


def test_exhaustive_all_6561_records():
    counts: Counter[str] = Counter()
    for values in product(_LEGAL, repeat=8):
        counts[terminal_decision(_record(values))] += 1
    assert sum(counts.values()) == 6561
    assert counts["STOP"] == 3645
    assert counts["REDEFINE_CAUSAL_AUDIT"] == 405
    assert counts["INCOMPLETE_BUDGET"] == 2495
    assert counts["REDEFINE_SCALE"] == 15
    assert counts["GO"] == 1
    assert counts["CONTRACT_ERROR"] == 0


def test_fully_evaluated_256_records():
    counts: Counter[str] = Counter()
    for values in product(("PASS", "FAIL"), repeat=8):
        counts[terminal_decision(_record(values))] += 1
    assert sum(counts.values()) == 256
    assert counts["STOP"] == 192
    assert counts["REDEFINE_CAUSAL_AUDIT"] == 48
    assert counts["REDEFINE_SCALE"] == 15
    assert counts["GO"] == 1


def test_schema_contract_error_missing_key():
    for missing in GATE_KEYS:
        record = {key: "PASS" for key in GATE_KEYS if key != missing}
        assert terminal_decision(record) == "CONTRACT_ERROR"


def test_schema_contract_error_invalid_status():
    for gate in GATE_KEYS:
        record = {key: "PASS" for key in GATE_KEYS}
        record[gate] = "SKIPPED"
        assert terminal_decision(record) == "CONTRACT_ERROR"


def _breakdown(text: str, name: str) -> dict[str, int]:
    match = re.search(rf"{name}: \{{([^}}]+)\}}", text)
    assert match is not None
    return {key: int(value) for key, value in re.findall(r"(\w+): (\d+)", match.group(1))}


def test_dec010_document_matches_decision_table_and_state_invariants():
    text = _DEC010.read_text(encoding="utf-8")
    rules = re.findall(r'^\s*- if: "(.*)"\n\s*then: (\S+)\s*$', text, re.M)
    assert tuple(rules) == _FROZEN_RULES
    assert "evaluation: FIRST_MATCH" in text
    assert "precedence: BEFORE_TERMINAL_DECISION" in text
    assert "else: CONTRACT_ERROR" in text
    assert "status: FROZEN" in text
    assert "decision_table_version: 1" in text
    for record, (_condition, outcome) in zip(_RULE_WITNESSES, rules):
        assert terminal_decision(record) == outcome

    legal = Counter(terminal_decision(_record(values)) for values in product(_LEGAL, repeat=8))
    full = Counter(terminal_decision(_record(values)) for values in product(("PASS", "FAIL"), repeat=8))
    assert _breakdown(text, "legal_breakdown") == {key: legal[key] for key in ("STOP", "REDEFINE_CAUSAL_AUDIT", "INCOMPLETE_BUDGET", "REDEFINE_SCALE", "GO")}
    assert _breakdown(text, "fully_evaluated_breakdown") == {key: full[key] for key in ("STOP", "REDEFINE_CAUSAL_AUDIT", "REDEFINE_SCALE", "GO")}

    state = yaml.safe_load((_ROOT / "control_plane" / "STATE.yaml").read_text(encoding="utf-8"))
    assert state["milestones"]["M22.1"]["status"] == "CANDIDATE"
    assert state["gates"]["M22.2"]["authorized"] is False


def test_incomplete_budget_50_record_case():
    family = [
        q_values
        for q_values in product(_LEGAL, repeat=4)
        if "FAIL" in q_values and "NOT_EVALUATED" in q_values
    ]
    assert len(family) == 50
    for q_values in family:
        record = {key: "PASS" for key in _P_KEYS}
        record.update(zip(_Q_KEYS, q_values))
        assert terminal_decision(record) == "INCOMPLETE_BUDGET"
