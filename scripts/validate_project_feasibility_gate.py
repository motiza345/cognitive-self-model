"""Check the project feasibility gate. Does not load Qwen or score outcomes."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILY = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")
DOC = "reports/PROJECT_FEASIBILITY_GATE.md"
MATRIX = "reports/PROJECT_FEASIBILITY_GATE_MATRIX.csv"


def decide(
    evaluation_class: str,
    replication_class: str,
    evaluation_control_class: str,
    replication_control_class: str,
    evaluation_sign_pass: bool,
    replication_sign_pass: bool,
) -> str:
    """Frozen map. Interval classes are the M23 ci_class labels."""
    if evaluation_class == "CI_NEGATIVE" and replication_class != "CI_POSITIVE":
        return "FEASIBILITY_KILL"
    specific = evaluation_control_class != "CI_POSITIVE" and replication_control_class != "CI_POSITIVE"
    if (
        evaluation_class == "CI_POSITIVE"
        and replication_class == "CI_POSITIVE"
        and specific
        and evaluation_sign_pass
        and replication_sign_pass
    ):
        return "FEASIBILITY_CONTINUE"
    return "INCONCLUSIVE"


def expected_matrix() -> list[dict[str, str]]:
    return [
        {
            "item": "primary_hypothesis",
            "role": "hypothesis",
            "category": "C",
            "statement": "first-order directional derivative along D1 identifies held-out response variation beyond the scalar mean",
        },
        {
            "item": "representation",
            "role": "representation",
            "category": "C",
            "statement": "g is the margin gradient at the last-token residual dotted with D1, and the prediction is g",
        },
        {
            "item": "observable",
            "role": "observable",
            "category": "C",
            "statement": "one predeclared observable, g, with no epsilon search and no fitted slope",
        },
        {
            "item": "family",
            "role": "intervention_family",
            "category": "design",
            "statement": "M22.1-D1-L0, M22.1-D1-L8, M22.1-D1-L15",
        },
        {
            "item": "scalar_baseline",
            "role": "control",
            "category": "A",
            "statement": "per-cell train mean of alpha-+1 effects",
        },
        {
            "item": "specificity_control",
            "role": "control",
            "category": "C",
            "statement": "g_orth from the same gradient dotted with D2, scored against the same mean",
        },
        {
            "item": "leakage",
            "role": "leakage",
            "category": "design",
            "statement": "predictions for evaluation and replication are written before their outcomes",
        },
        {
            "item": "metric",
            "role": "metric",
            "category": "B",
            "statement": "prompt-averaged paired_mean_ci of error(baseline) minus error(g)",
        },
        {
            "item": "pass",
            "role": "decision",
            "category": "decision",
            "statement": "FEASIBILITY_CONTINUE only if both splits are CI_POSITIVE, both controls are not, and both sign co-criteria pass",
        },
        {
            "item": "kill",
            "role": "decision",
            "category": "decision",
            "statement": "FEASIBILITY_KILL if evaluation is CI_NEGATIVE and replication is not CI_POSITIVE",
        },
        {
            "item": "inconclusive",
            "role": "decision",
            "category": "decision",
            "statement": "INCONCLUSIVE for every other combination, including a single-split win",
        },
        {
            "item": "replication",
            "role": "replication",
            "category": "design",
            "statement": "fourth partition, absent from the fit, required for FEASIBILITY_CONTINUE",
        },
        {
            "item": "not_a_self_model",
            "role": "boundary",
            "category": "D",
            "statement": "a pass does not establish a consumed self-model",
        },
        {
            "item": "excluded_searches",
            "role": "exclusion",
            "category": "design",
            "statement": "no feature fishing, no second observable, no layer-23 cell, pre_dot is not the candidate",
        },
    ]


def _fail(message: str) -> None:
    raise SystemExit(message)


def check_decision_map() -> None:
    cases = [
        (("CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_NEGATIVE", True, True), "FEASIBILITY_CONTINUE"),
        (("CI_POSITIVE", "CI_POSITIVE", "CI_NEGATIVE", "CI_INCLUDES_ZERO", True, True), "FEASIBILITY_CONTINUE"),
        (("CI_NEGATIVE", "CI_NEGATIVE", "CI_POSITIVE", "CI_POSITIVE", False, False), "FEASIBILITY_KILL"),
        (("CI_NEGATIVE", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", False, False), "FEASIBILITY_KILL"),
        (("CI_NEGATIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", False, True), "INCONCLUSIVE"),
        (("CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", True, False), "INCONCLUSIVE"),
        (("CI_POSITIVE", "CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", True, True), "INCONCLUSIVE"),
        (("CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_POSITIVE", True, True), "INCONCLUSIVE"),
        (("CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", False, True), "INCONCLUSIVE"),
        (("CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", True, False), "INCONCLUSIVE"),
        (("CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", "CI_INCLUDES_ZERO", False, False), "INCONCLUSIVE"),
    ]
    for args, expected in cases:
        found = decide(*args)
        if found != expected:
            _fail(f"decision map returned {found} for {args}, expected {expected}")


def check_document() -> None:
    text = (ROOT / DOC).read_text(encoding="utf-8")
    if text.count("primary hypothesis:") != 1:
        _fail("document must contain exactly one primary hypothesis declaration")
    if text.count("primary representation hypothesis:") != 1:
        _fail("document must contain exactly one representation hypothesis declaration")
    if text.count("predeclared observable:") != 1:
        _fail("document must declare exactly one observable")
    if text.count("intervention family:") != 1:
        _fail("document must declare exactly one intervention family")
    for cell in FAMILY:
        if cell not in text:
            _fail(f"family cell missing: {cell}")
    required = (
        "GATE_SPECIFIED",
        "No feature fishing is performed.",
        "No threshold is estimated from outcomes.",
        "FEASIBILITY_CONTINUE",
        "FEASIBILITY_KILL",
        "INCONCLUSIVE",
        "paired_mean_ci",
        "clopper_pearson",
        "Predictions for evaluation are written before evaluation outcomes exist.",
        "Predictions for replication are written before replication outcomes exist.",
        "g_orth",
        "pre_dot",
        "CURRENT_INTERVENTION_INSUFFICIENT",
        "category D",
    )
    for phrase in required:
        if phrase not in text:
            _fail(f"document missing {phrase}")
    for banned in ("threshold tuned", "winning cell", "feature ranking", "try several observables"):
        if banned in text.lower():
            _fail(f"document contains banned phrase: {banned}")
    if "M22.1-D1-L23" not in text:
        _fail("document must record why the consumed layer-23 cell is outside the family")
    # The family declaration is one line. Layer 23 must not be added to that line.
    for line in text.splitlines():
        if line.startswith("intervention family:") and "L23" in line:
            _fail("layer 23 was added to the intervention family")


def check_matrix() -> None:
    with (ROOT / MATRIX).open(encoding="utf-8", newline="") as handle:
        found = list(csv.DictReader(handle))
    expected = expected_matrix()
    if found != expected:
        _fail("feasibility matrix does not match the frozen rows")
    roles = {row["role"] for row in found}
    for role in ("hypothesis", "representation", "observable", "intervention_family", "control", "decision", "replication"):
        if role not in roles:
            _fail(f"matrix missing role {role}")
    if sum(row["role"] == "observable" for row in found) != 1:
        _fail("matrix must list one observable")
    if sum(row["role"] == "hypothesis" for row in found) != 1:
        _fail("matrix must list one hypothesis")
    if sum(row["role"] == "representation" for row in found) != 1:
        _fail("matrix must list one representation")


def main() -> None:
    check_decision_map()
    check_document()
    check_matrix()
    print("FEASIBILITY_GATE_CHECK_PASS")
    print("GATE_SPECIFIED")


if __name__ == "__main__":
    main()
