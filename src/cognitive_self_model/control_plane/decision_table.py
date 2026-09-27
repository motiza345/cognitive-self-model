"""Frozen DEC-010 terminal decision table."""

from __future__ import annotations

from typing import Mapping

GATE_KEYS = ["P.H1", "P.H2", "P.H3", "P.H4", "Q.H1", "Q.H2", "Q.H3", "Q.H4"]
ALLOWED_STATUSES = {"PASS", "FAIL", "NOT_EVALUATED"}
P_KEYS = ("P.H1", "P.H2", "P.H3", "P.H4")
Q_KEYS = ("Q.H1", "Q.H2", "Q.H3", "Q.H4")

OUTCOME_ACTIONS = {
    "INCOMPLETE_BUDGET": "Archive as inconclusive; no GO; continuation requires a new DEC",
    "STOP": "Archive with final report; no further milestones",
    "REDEFINE_CAUSAL_AUDIT": "Rename/rescope as causal-audit framework; retire Self-Model claims",
    "REDEFINE_SCALE": "Report planted-only result; any Q follow-up requires a new bounded DEC",
    "GO": "Continue only within the scope of the gates that passed",
}


def terminal_decision(record: Mapping[str, str]) -> str:
    """Return the FIRST_MATCH outcome. Schema failures return CONTRACT_ERROR."""
    for key in GATE_KEYS:
        if key not in record:
            return "CONTRACT_ERROR"
    for key in GATE_KEYS:
        if record[key] not in ALLOWED_STATUSES:
            return "CONTRACT_ERROR"

    p1 = record["P.H1"]
    p2 = record["P.H2"]
    p3 = record["P.H3"]
    p4 = record["P.H4"]
    q_values = [record[key] for key in Q_KEYS]

    if p1 == "FAIL" or (p1 == "PASS" and p2 == "FAIL") or (p1 == "NOT_EVALUATED" and p2 == "FAIL"):
        return "STOP"
    if p1 == "PASS" and p2 == "PASS" and p3 == "FAIL":
        return "REDEFINE_CAUSAL_AUDIT"
    if p1 == "PASS" and p2 == "PASS" and p3 == "PASS" and p4 == "FAIL":
        return "REDEFINE_CAUSAL_AUDIT"
    if p1 == "PASS" and p2 == "PASS" and p3 == "NOT_EVALUATED" and p4 == "FAIL":
        return "REDEFINE_CAUSAL_AUDIT"
    if p1 == p2 == p3 == p4 == "PASS" and any(value == "NOT_EVALUATED" for value in q_values):
        return "INCOMPLETE_BUDGET"
    if (
        p1 == p2 == p3 == p4 == "PASS"
        and all(value != "NOT_EVALUATED" for value in q_values)
        and any(value == "FAIL" for value in q_values)
    ):
        return "REDEFINE_SCALE"
    if p1 == p2 == p3 == p4 == "PASS" and all(value == "PASS" for value in q_values):
        return "GO"
    if any(record[key] == "NOT_EVALUATED" for key in P_KEYS):
        return "INCOMPLETE_BUDGET"
    return "CONTRACT_ERROR"
