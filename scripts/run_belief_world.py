"""Measure four planted-world changes and score them with the unchanged rule.

Does not write into artifacts/mrsm and does not finetune the frozen checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mrsm_p_v1  # noqa: E402
from src.mrsm.belief_update import select_mechanism, tolerance_of  # noqa: E402
from src.mrsm.belief_world import (  # noqa: E402
    CONTROL_EXPECTATION,
    DISCOVERY_N,
    DISCOVERY_SEED,
    INSTRUMENT_ATOL,
    MODEL_SEED,
    PRIMARY_EXPECTATION,
    TRAIN_N,
    TRAIN_SEED,
    decide_world,
    declare_from_belief,
    measure_effects,
    penalty_amount,
    preregistration_document,
    records_from_effects,
    score_records,
    finite_effects,
)
from src.mrsm.interventions import discovery_effects  # noqa: E402
from src.mrsm.p_data import dataset  # noqa: E402

P_RUN = ROOT / "artifacts" / "mrsm" / "p_run_001"
STATE_PATH = P_RUN / "self_model_state.json"
EVIDENCE_PATH = P_RUN / "evidence.json"
FALSIFICATION_PATH = P_RUN / "falsification.json"
PREREG_PATH = ROOT / "configs" / "mrsm_prereg.yaml"
CHECKPOINT = ROOT / "artifacts" / "mrsm" / "construction" / "p_model_state.pt"
RULE_PATH = ROOT / "src" / "mrsm" / "belief_update.py"
OUT = ROOT / "artifacts" / "belief_world"
REPORT = ROOT / "reports" / "BELIEF_WORLD.md"
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
    audit = {
        "effects_match": all(
            abs(float(state["effects"][name]) - effects[name]) <= RECONSTRUCTION_ATOL for name in effects
        ),
        "interaction_match": abs(reconstructed - float(state["mechanism"]["interaction"])) <= 1.0e-4,
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


def _load_frozen_model():
    prereg = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    expected = str(prereg["model"]["arm_p_checkpoint_sha256"])
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError("frozen checkpoint hash does not match the preregistration")
    cfg = mrsm_p_v1.Config()
    mrsm_p_v1.seed_all(int(prereg["seeds"]["construction_model_seed"]))
    model = mrsm_p_v1.build_model(cfg)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _train_hypothesis_copy():
    cfg = mrsm_p_v1.Config()
    cfg.planted_a = (0, 2)
    cfg.planted_b = (1, 2)
    cfg.model_seed = MODEL_SEED
    mrsm_p_v1.seed_all(MODEL_SEED)
    model = mrsm_p_v1.build_model(cfg)
    train = mrsm_p_v1.make_dataset(cfg, TRAIN_N, TRAIN_SEED)
    log = mrsm_p_v1.train_planted(model, cfg, train)
    model.eval()
    return model, log


def _shuffle_targets(ds, seed: int):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(ds.targets.shape[0], generator=generator)
    return mrsm_p_v1.Dataset(
        ds.tokens.clone(),
        ds.targets[perm].clone(),
        ds.query_key.clone(),
        ds.pair_keys.clone(),
        ds.pair_values.clone(),
        ds.query_pos,
    )


def _instrument_ok(model, ds) -> bool:
    reference_singles, reference_joints = discovery_effects(model, ds)
    measured_singles, measured_joints = measure_effects(model, ds, lambda _ablated: 0.0)
    gaps = [
        abs(reference_singles[name] - measured_singles[name]) for name in reference_singles
    ] + [abs(reference_joints[name] - measured_joints[name]) for name in reference_joints]
    return max(gaps) <= INSTRUMENT_ATOL


def _selected(belief: dict[str, Any], singles: dict[str, float], joints: dict[str, float]) -> dict[str, Any]:
    mechanism = select_mechanism(
        singles,
        joints,
        float(belief["min_abs_interaction"]),
        float(belief["min_relative_gap"]),
    )
    return {
        "status": mechanism["status"],
        "ordered": mechanism["ordered"],
        "decision_input": False,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Belief update on changed P worlds",
        "",
        "Diagnostic. The update rule is the one already locked in `belief_update.classify`.",
        "Not an MRSM rescore, not a Self-Model, and not M22.",
        "",
        "## 1. Question",
        "",
        "Do measurements from four changed planted worlds receive four different updates?",
        "",
        f"- Decision: `{result['decision']['decision']}`.",
        f"- {result['decision']['reason']} This result is local to these worlds and this frozen belief. Qwen was not loaded. The frozen checkpoint was not finetuned.",
        "",
        "## 2. What was fixed before measurement",
        "",
        f"- Hypothesis construction mask: `{result['declaration']['hypothesis_heads']}`.",
        f"- Scope name: `{result['declaration']['scope_name']}`.",
        f"- Interaction pair: `{result['declaration']['interaction_pair']}`.",
        f"- Shared magnitude, from the frozen winner interaction: `{result['declaration']['magnitude']}`.",
        f"- None lesion: `{result['declaration']['none_head']}`.",
        "- The magnitude was not searched against a gap ceiling, and it was not changed after scoring.",
        "",
        "## 3. Measured outputs",
        "",
        "| World | Rule output | Expected |",
        "| --- | --- | --- |",
    ]
    for name, expected in PRIMARY_EXPECTATION.items():
        lines.append(f"| {name} | `{result['primary'][name]}` | `{expected}` |")
    lines.extend(["", "## 4. Control", "", "| Case | Rule output | Expected |", "| --- | --- | --- |"])
    for name, expected in CONTROL_EXPECTATION.items():
        lines.append(f"| {name} | `{result['controls'][name]}` | `{expected}` |")
    lines.extend(
        [
            "",
            "## 5. Descriptive selections",
            "",
            "These selections are not decision inputs.",
            "",
        ]
    )
    for name, row in result["descriptive_selection"].items():
        lines.append(f"- {name}: status `{row['status']}`, ordered `{row['ordered']}`.")
    lines.extend(
        [
            "",
            "## 6. What this does not say",
            "",
            "Section 3 is the decision trace. It does not reopen the label.",
            "A separation means these measured worlds were distinguishable by the unchanged rule.",
            "A mismatch means that, for these worlds, the rule did not assign the world type.",
            "Neither result transfers to Qwen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    belief, audit = _load_belief()
    declaration = declare_from_belief(belief)
    _write(OUT / "preregistration.json", preregistration_document(declaration))
    recorded = _recorded_tolerance()
    formula = tolerance_of(float(belief["uncertainty"]))
    tolerance_ok = recorded is not None and abs(recorded - formula) <= RECONSTRUCTION_ATOL
    reconstruction_ok = bool(audit["effects_match"] and audit["interaction_match"])
    rule_before = _sha256(RULE_PATH)
    checkpoint_before = _sha256(CHECKPOINT)
    discovery = dataset(DISCOVERY_N, DISCOVERY_SEED)
    frozen = _load_frozen_model()
    instrument_ok = _instrument_ok(frozen, discovery)
    trained, train_log = _train_hypothesis_copy()
    worlds = {
        "hypothesis": measure_effects(trained, discovery, lambda _ablated: 0.0),
        "scope": measure_effects(
            frozen,
            _shuffle_targets(discovery, int(declaration["scope_permutation_seed"])),
            lambda _ablated: 0.0,
        ),
        "interaction": measure_effects(
            frozen,
            discovery,
            lambda ablated: penalty_amount("interaction", ablated, declaration),
        ),
        "none": measure_effects(
            frozen,
            discovery,
            lambda ablated: penalty_amount("none", ablated, declaration),
        ),
    }
    finite_ok = all(finite_effects(singles, joints) for singles, joints in worlds.values())
    scopes = {
        "hypothesis": str(belief["scope"]),
        "scope": str(declaration["scope_name"]),
        "interaction": str(belief["scope"]),
        "none": str(belief["scope"]),
    }
    primary = {}
    descriptive = {}
    for name, (singles, joints) in worlds.items():
        primary[name] = score_records(belief, records_from_effects(singles, joints, scopes[name]))
        descriptive[name] = _selected(belief, singles, joints)
    controls = {"holdout": score_records(belief, _holdout_records())}
    decision = decide_world(
        primary,
        controls,
        reconstruction_ok=reconstruction_ok,
        tolerance_ok=tolerance_ok,
        instrument_ok=instrument_ok,
        finite_ok=finite_ok,
    )
    result = {
        "arm": "P",
        "audit": audit,
        "checkpoint_sha256_after": _sha256(CHECKPOINT),
        "checkpoint_sha256_before": checkpoint_before,
        "controls": controls,
        "decision": decision,
        "declaration": declaration,
        "descriptive_selection": descriptive,
        "finite_ok": finite_ok,
        "formula_tolerance": formula,
        "instrument_ok": instrument_ok,
        "m22_started": False,
        "mrsm_modified": False,
        "primary": primary,
        "qwen_loaded": False,
        "recorded_tolerance": recorded,
        "rule_modified": False,
        "rule_sha256_after": _sha256(RULE_PATH),
        "rule_sha256_before": rule_before,
        "self_model_modified": False,
        "train_log": {
            "final_loss": train_log["final_loss"],
            "seconds": train_log["seconds"],
            "steps": train_log["steps"],
            "decision_input": False,
        },
    }
    _write(OUT / "decision.json", result)
    manifest = [f"{_sha256(path)}  {path.name}" for path in sorted(OUT.glob("*.json"))]
    (OUT / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    REPORT.write_text(_markdown(result), encoding="utf-8")
    print(decision["decision"])
    print(decision["reason_code"])
    print("instrument", instrument_ok, "finite", finite_ok)
    print(primary)
    print("rule_unchanged", result["rule_sha256_before"] == result["rule_sha256_after"])
    print("checkpoint_unchanged", checkpoint_before == result["checkpoint_sha256_after"])


if __name__ == "__main__":
    main()
