"""Exhaustive checks for the frozen DEC-010 terminal decision table."""

from __future__ import annotations

from collections import Counter
from itertools import product

from src.cognitive_self_model.control_plane.decision_table import GATE_KEYS, terminal_decision

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
