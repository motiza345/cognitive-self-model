"""Score the preregistered belief-update rule on frozen arm P.

Reads artifacts/mrsm/p_run_001. Does not write there.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.mrsm.belief_update import (  # noqa: E402
    CONTROL_EXPECTATION,
    PRIMARY_EXPECTATION,
    build_controls,
    build_primary_cases,
    classify,
    decide,
    preregistration_document,
    tolerance_of,
)

P_RUN = ROOT / "artifacts" / "mrsm" / "p_run_001"
STATE_PATH = P_RUN / "self_model_state.json"
EVIDENCE_PATH = P_RUN / "evidence.json"
FALSIFICATION_PATH = P_RUN / "falsification.json"
PREREG_PATH = ROOT / "configs" / "mrsm_prereg.yaml"
OUT = ROOT / "artifacts" / "belief_update"
REPORT = ROOT / "reports" / "BELIEF_UPDATE.md"
RECONSTRUCTION_ATOL = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_belief() -> tuple[dict[str, Any], dict[str, Any]]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["records"]
    prereg = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    ident = prereg["identifiability"]
    effects: dict[str, float] = {}
    joints: dict[str, float] = {}
    for record in evidence:
        if record["evidence_type"] != "discovery_observation":
            continue
        kind, name = str(record["intervention_id"]).split(":", 1)
        if kind == "single":
            effects[name] = float(record["actual_outcome"])
        elif kind == "pair":
            joints[name] = float(record["actual_outcome"])
    belief = {
        "scope": str(state["scope"]),
        "effects": {name: float(state["effects"][name]) for name in state["effects"]},
        "joints": joints,
        "mechanism_ordered": [str(name) for name in state["mechanism"]["ordered"]],
        "uncertainty": float(state["uncertainty"]),
        "min_abs_interaction": float(ident["min_abs_interaction"]),
        "min_relative_gap": float(ident["min_relative_gap_vs_second"]),
    }
    winner = "+".join(belief["mechanism_ordered"])
    left, right = belief["mechanism_ordered"]
    reconstructed = float(joints[winner]) - float(effects[left]) - float(effects[right])
    effects_match = all(
        abs(float(state["effects"][name]) - effects[name]) <= RECONSTRUCTION_ATOL for name in effects
    )
    interaction_match = abs(reconstructed - float(state["mechanism"]["interaction"])) <= 1.0e-4
    audit = {
        "effects_match": effects_match,
        "interaction_match": interaction_match,
        "reconstructed_winner_interaction": reconstructed,
        "recorded_winner_interaction": float(state["mechanism"]["interaction"]),
        "n_discovery_singles": len(effects),
        "n_discovery_pairs": len(joints),
    }
    return belief, audit


def _recorded_tolerance() -> float | None:
    payload = json.loads(FALSIFICATION_PATH.read_text(encoding="utf-8"))
    if "tolerance" in payload:
        return float(payload["tolerance"])
    values = {float(row["tolerance"]) for row in payload["records"]}
    if len(values) != 1:
        return None
    return values.pop()


def _holdout_records() -> list[dict[str, Any]]:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))["records"]
    return [
        {
            "intervention_id": str(record["intervention_id"]),
            "actual_outcome": float(record["actual_outcome"]),
            "scope": str(record["scope"]),
        }
        for record in evidence
        if record["evidence_type"] == "holdout_scored"
    ]


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Belief update on frozen arm P",
        "",
        "Diagnostic. Not an MRSM rescore, not a Self-Model, and not M22.",
        "",
        "## 1. Question",
        "",
        "Can one rule, reading only the frozen P belief and evidence records, tell four",
        "single contradictions apart?",
        "",
        f"- Decision: `{result['decision']['decision']}`.",
        f"- {result['decision']['reason']} This result is local to the frozen P belief and these injections. It does not identify a mechanism. A Self-Model was not modified. Qwen was not loaded.",
        "",
        "## 2. What was fixed before scoring",
        "",
        "- The rule reads scope, effects, joints, the current ordered pair, uncertainty, and the frozen identifiability constants.",
        "- A record supplies an intervention id, an actual outcome, and a scope string.",
        "- Tolerance is `max(1.0, 2.0 * uncertainty)`, the expression already used by the P falsification record.",
        "- The construction clearance is not a decision input. If the interaction injection has no legal pair, the decision is `INCONCLUSIVE`.",
        "- Injection names and planted ground truth are not inputs.",
        "",
        "## 3. Primary injections",
        "",
        "| Case | Rule output | Expected |",
        "| --- | --- | --- |",
    ]
    for name, expected in PRIMARY_EXPECTATION.items():
        lines.append(f"| {name} | `{result['primary'][name]}` | `{expected}` |")
    lines.extend(
        [
            "",
            "## 4. Controls",
            "",
            "| Case | Rule output | Expected |",
            "| --- | --- | --- |",
        ]
    )
    for name, expected in CONTROL_EXPECTATION.items():
        lines.append(f"| {name} | `{result['controls'][name]}` | `{expected}` |")
    lines.extend(
        [
            "",
            "## 5. Preconditions",
            "",
            f"- Reconstruction matches the frozen state: `{result['audit']['effects_match']}` and `{result['audit']['interaction_match']}`.",
            f"- Recorded tolerance `{result['recorded_tolerance']}` against formula `{result['formula_tolerance']}`.",
            f"- Interaction construction defined: `{result['construction_defined']}`.",
            "",
            "## 6. What this does not say",
            "",
            "Section 3 is the decision trace. It does not reopen the label.",
            "The four injections are built so that each one supports a different predicate.",
            "On this belief, a legal interaction move exists and the frozen holdout stays inside tolerance.",
            "Separation means those constructed records were distinguishable.",
            "It does not say the rule recovered a failure type from unstructured history.",
            "It does not say the same rule separates failures on Qwen.",
            "A collapse would mean this rule did not keep the four contradictions apart.",
            "It would not say a belief object cannot be defined.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "preregistration.json", preregistration_document())
    input_hashes = {
        "self_model_state.json": _sha256(STATE_PATH),
        "evidence.json": _sha256(EVIDENCE_PATH),
        "falsification.json": _sha256(FALSIFICATION_PATH),
        "mrsm_prereg.yaml": _sha256(PREREG_PATH),
    }
    belief, audit = _load_belief()
    recorded = _recorded_tolerance()
    formula = tolerance_of(float(belief["uncertainty"]))
    tolerance_ok = recorded is not None and abs(recorded - formula) <= RECONSTRUCTION_ATOL
    cases = build_primary_cases(belief)
    construction_defined = cases["interaction"] is not None
    primary_records = {name: rows for name, rows in cases.items() if rows is not None}
    primary = {
        name: classify(belief, rows) if rows is not None else None
        for name, rows in cases.items()
    }
    hypothesis_rows = cases["hypothesis"] or []
    controls_in = build_controls(belief, _holdout_records(), hypothesis_rows)
    controls = {name: classify(belief, rows) for name, rows in controls_in.items()}
    decision = decide(
        primary,
        controls,
        reconstruction_ok=bool(audit["effects_match"] and audit["interaction_match"]),
        tolerance_ok=tolerance_ok,
        construction_defined=construction_defined,
    )
    result = {
        "arm": "P",
        "audit": audit,
        "construction_defined": construction_defined,
        "controls": controls,
        "decision": decision,
        "formula_tolerance": formula,
        "input_sha256_before_scoring": input_hashes,
        "input_sha256_after_scoring": {
            "self_model_state.json": _sha256(STATE_PATH),
            "evidence.json": _sha256(EVIDENCE_PATH),
            "falsification.json": _sha256(FALSIFICATION_PATH),
        },
        "mechanism_identity": "NOT_EVALUATED",
        "m22_started": False,
        "mrsm_modified": False,
        "primary": primary,
        "qwen_loaded": False,
        "recorded_tolerance": recorded,
        "self_model_modified": False,
    }
    _write(OUT / "cases.json", {"controls": controls, "primary": primary, "primary_defined": sorted(primary_records)})
    _write(OUT / "decision.json", result)
    manifest_lines = []
    for path in sorted(OUT.glob("*.json")):
        if path.name == "manifest.sha256":
            continue
        manifest_lines.append(f"{_sha256(path)}  {path.name}")
    (OUT / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    REPORT.write_text(_markdown(result), encoding="utf-8")
    unchanged = result["input_sha256_before_scoring"]["self_model_state.json"] == result[
        "input_sha256_after_scoring"
    ]["self_model_state.json"]
    print(decision["decision"])
    print(decision["reason_code"])
    print("inputs_unchanged", unchanged)


if __name__ == "__main__":
    main()
