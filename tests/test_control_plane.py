"""Control-plane validator: real ledger plus negative copies."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from src.cognitive_self_model.control_plane.validate import _check_m22_1_r_replay_config, validate

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "control_plane"


def _copy_control(tmp_path: Path) -> Path:
    dest = tmp_path / "control_plane"
    shutil.copytree(CONTROL, dest)
    return dest


def _rewrite(path: Path, mutator):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_real_control_plane_passes():
    errors = validate(CONTROL, ROOT)
    assert errors == []


def test_tampered_hash_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        for entry in payload["files"]:
            if entry.get("immutable") and entry.get("sha256"):
                entry["sha256"] = "0" * 64
                break

    _rewrite(dest / "FILE_MAP.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("sha256 mismatch" in item for item in errors)


def test_missing_path_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["files"][0]["path"] = "does/not/exist.txt"

    _rewrite(dest / "FILE_MAP.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("does not exist" in item for item in errors)


def test_dangling_reinterpretation_link_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        for entry in payload["entries"]:
            if entry["id"] == "F-M22.1.2-RESPONSE":
                entry["reinterpreted_by"] = "F-DOES-NOT-EXIST"
                break

    _rewrite(dest / "CLAIMS.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("reinterpreted_by missing" in item for item in errors)


def test_observation_cited_by_gate_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["gates"]["M22.2"]["unblock_requires"].append("OBS-M22.1.3-CURVATURE")

    _rewrite(dest / "STATE.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("observation" in item for item in errors)


def test_m22_2_authorized_without_decision_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["gates"]["M22.2"]["authorized"] = True

    _rewrite(dest / "STATE.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("M22.2.authorized" in item for item in errors)


def test_open_integrity_flag_missing_from_state_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate_claims(payload):
        for entry in payload["entries"]:
            if entry["id"] == "F-M22.1.3-READOUT":
                entry["integrity_flags"] = [
                    {
                        "id": "INT-OPEN-TEST",
                        "description": "synthetic open flag",
                        "source": "test",
                        "resolution_status": "open",
                        "path": "artifacts/m22_1_3/predictions.json",
                    }
                ]
                break

    def mutate_state(payload):
        payload["known_integrity_issues"] = {"open": [], "resolved": []}

    _rewrite(dest / "CLAIMS.yaml", mutate_claims)
    _rewrite(dest / "STATE.yaml", mutate_state)
    errors = validate(dest, ROOT)
    assert any("integrity flag" in item for item in errors)


def test_exact_with_missing_items_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M22.1.1"]["missing"] = ["synthetic missing item"]

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("EXACT must have no missing" in item for item in errors)


def test_partial_with_empty_missing_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M22.1"]["missing"] = []

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("PARTIAL must have a non-empty missing list" in item for item in errors)


def test_chain_grade_inconsistent_with_depends_on_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M22.1.1"]["recovery_grade_chain"] = "EXACT"

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("recovery_grade_chain" in item and "computed" in item for item in errors)


def test_unknown_depends_on_complete_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M18"]["chain_status"] = "COMPLETE"

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("UNKNOWN requires chain_status INCOMPLETE" in item for item in errors)


def test_unknown_depends_on_chain_better_than_local_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M18"]["recovery_grade_chain"] = "PARTIAL"

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("must not be better than recovery_grade_local" in item for item in errors)


def test_empty_depends_on_without_verified_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        payload["milestones"]["M22.1"]["depends_on_verified"] = False

    _rewrite(dest / "RECOVERY.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("empty depends_on requires depends_on_verified true" in item for item in errors)


def test_environment_python_and_threads_types(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def accept(payload):
        payload["milestones"]["M22.1"]["environment"]["python"] = "3.12.3"
        payload["milestones"]["M22.1"]["environment"]["threads"] = 4

    _rewrite(dest / "RECOVERY.yaml", accept)
    assert validate(dest, ROOT) == []

    def unknown(payload):
        payload["milestones"]["M22.1"]["environment"]["python"] = "UNKNOWN"
        payload["milestones"]["M22.1"]["environment"]["threads"] = "UNKNOWN"

    _rewrite(dest / "RECOVERY.yaml", unknown)
    assert validate(dest, ROOT) == []

    def bad_python(payload):
        payload["milestones"]["M22.1"]["environment"]["python"] = "cpython"

    _rewrite(dest / "RECOVERY.yaml", bad_python)
    errors = validate(dest, ROOT)
    assert any("environment.python invalid" in item for item in errors)

    def bad_threads(payload):
        payload["milestones"]["M22.1"]["environment"]["python"] = "UNKNOWN"
        payload["milestones"]["M22.1"]["environment"]["threads"] = 0

    _rewrite(dest / "RECOVERY.yaml", bad_threads)
    errors = validate(dest, ROOT)
    assert any("environment.threads invalid" in item for item in errors)


def test_claim_worse_than_partial_chain_without_scope_limitation_fails(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def mutate(payload):
        for entry in payload["entries"]:
            if entry["id"] == "F-M21.2.4.3.3-QINVALID":
                entry["scope_limitation"] = ""
                break

    _rewrite(dest / "CLAIMS.yaml", mutate)
    errors = validate(dest, ROOT)
    assert any("scope_limitation referring to the upstream gap" in item for item in errors)


@pytest.mark.parametrize("key", ["torch", "transformers", "transformer_lens", "numpy"])
def test_environment_version_keys_are_required_and_typed(tmp_path: Path, key: str):
    dest = _copy_control(tmp_path)

    def missing(payload):
        del payload["milestones"]["M22.1"]["environment"][key]

    _rewrite(dest / "RECOVERY.yaml", missing)
    errors = validate(dest, ROOT)
    assert any(
        f"control_plane/RECOVERY.yaml: RECOVERY M22.1 environment missing required key: {key}" in item
        for item in errors
    )

    def unknown(payload):
        payload["milestones"]["M22.1"]["environment"][key] = "UNKNOWN"

    _rewrite(dest / "RECOVERY.yaml", unknown)
    assert validate(dest, ROOT) == []

    def plus_local(payload):
        payload["milestones"]["M22.1"]["environment"][key] = "2.14.0+cpu"

    _rewrite(dest / "RECOVERY.yaml", plus_local)
    assert validate(dest, ROOT) == []

    def malformed(payload):
        payload["milestones"]["M22.1"]["environment"][key] = "cpu-2.14"

    _rewrite(dest / "RECOVERY.yaml", malformed)
    errors = validate(dest, ROOT)
    assert any(
        f"control_plane/RECOVERY.yaml: RECOVERY M22.1 environment.{key} invalid: cpu-2.14" in item
        for item in errors
    )


def test_real_m22_1_r_replay_config_passes():
    errors: list[str] = []
    _check_m22_1_r_replay_config(ROOT, errors)
    assert errors == []


def _replay_copy(tmp_path: Path) -> Path:
    dest_dir = tmp_path / "configs"
    dest_dir.mkdir()
    target = dest_dir / "m22_1_r_replay.yaml"
    shutil.copy(ROOT / "configs" / "m22_1_r_replay.yaml", target)
    return target


def test_replay_config_missing_mitigation_required(tmp_path: Path):
    target = _replay_copy(tmp_path)

    def mutate(payload):
        del payload["revision_resolution"]["mitigation_required"]

    _rewrite(target, mutate)
    errors: list[str] = []
    _check_m22_1_r_replay_config(tmp_path, errors)
    assert any(
        "configs/m22_1_r_replay.yaml: revision_resolution missing required key: mitigation_required" in item
        for item in errors
    )


def test_replay_config_rejects_behavioral_equivalent_for_dec007(tmp_path: Path):
    target = _replay_copy(tmp_path)

    def mutate(payload):
        payload["dec007_success_tiers"]["tiers"].append("REPLAY_BEHAVIORAL_EQUIVALENT")

    _rewrite(target, mutate)
    errors: list[str] = []
    _check_m22_1_r_replay_config(tmp_path, errors)
    assert any(
        "dec007_success_tiers.tiers must not contain REPLAY_BEHAVIORAL_EQUIVALENT" in item
        for item in errors
    )


def _clean_execution_record() -> dict:
    return {
        "entrypoint": "python3 -m cognitive_self_model.m22_1.preflight",
        "invocation": "PYTHONPATH=src python3 -m cognitive_self_model.m22_1.preflight",
        "git_dirty": False,
    }


def test_execution_records_must_be_top_level_list(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def remove_list(payload):
        del payload["execution_records"]

    _rewrite(dest / "RECOVERY.yaml", remove_list)
    errors = validate(dest, ROOT)
    assert any("execution_records must be a top-level list" in item for item in errors)

    def nest(payload):
        payload["execution_records"] = []
        payload["milestones"]["M22.1"]["execution_records"] = [_clean_execution_record()]

    _rewrite(dest / "RECOVERY.yaml", nest)
    errors = validate(dest, ROOT)
    assert any("must not nest execution_records" in item for item in errors)


def test_execution_record_rejects_sentinels_and_requires_dirty_reason(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def unknown_invocation(payload):
        record = _clean_execution_record()
        record["invocation"] = "UNKNOWN"
        payload["execution_records"] = [record]

    _rewrite(dest / "RECOVERY.yaml", unknown_invocation)
    errors = validate(dest, ROOT)
    assert any("execution_records[0] invocation is required and must not be a sentinel" in item for item in errors)

    def dirty_without_reason(payload):
        record = _clean_execution_record()
        record["git_dirty"] = True
        payload["execution_records"] = [record]

    _rewrite(dest / "RECOVERY.yaml", dirty_without_reason)
    errors = validate(dest, ROOT)
    assert any("dirty_reason is required when git_dirty is true" in item for item in errors)

    def dirty_sentinel_reason(payload):
        record = _clean_execution_record()
        record["git_dirty"] = True
        record["dirty_reason"] = "NOT_RECORDED"
        payload["execution_records"] = [record]

    _rewrite(dest / "RECOVERY.yaml", dirty_sentinel_reason)
    errors = validate(dest, ROOT)
    assert any("dirty_reason is required when git_dirty is true" in item for item in errors)


def test_execution_record_accepts_clean_and_explained_dirty(tmp_path: Path):
    dest = _copy_control(tmp_path)

    def clean(payload):
        payload["execution_records"] = [_clean_execution_record()]

    _rewrite(dest / "RECOVERY.yaml", clean)
    assert validate(dest, ROOT) == []

    def dirty(payload):
        record = _clean_execution_record()
        record["git_dirty"] = True
        record["dirty_reason"] = "uncommitted local edits in src/cognitive_self_model/m22_1"
        payload["execution_records"] = [record]

    _rewrite(dest / "RECOVERY.yaml", dirty)
    assert validate(dest, ROOT) == []


def test_replay_config_rejects_renamed_tolerance_keys(tmp_path: Path):
    target = _replay_copy(tmp_path)

    def mutate(payload):
        payload["tolerances"]["s_abs"] = payload["tolerances"]["score_atol"]

    _rewrite(target, mutate)
    errors: list[str] = []
    _check_m22_1_r_replay_config(tmp_path, errors)
    assert any("tolerances.s_abs is not a frozen tolerance key" in item for item in errors)


# --- Commit 4: independent gate overrides -----------------------------------

import copy as _copy_mod
import shutil as _shutil

import yaml as _yaml

from cognitive_self_model.control_plane.validate import _decision_grants
from cognitive_self_model.control_plane.validate import validate as _validate_cp

_REPO_ROOT = Path(__file__).resolve().parents[1]
_M22_2_ERR = "STATE.gates.M22.2.authorized must be false"


def _gate_variant(tmp_path: Path, extra_decision: dict | None, m22_2_authorized) -> list[str]:
    cp = tmp_path / "control_plane"
    _shutil.copytree(_REPO_ROOT / "control_plane", cp)
    state_path = cp / "STATE.yaml"
    state = _yaml.safe_load(state_path.read_text())
    if extra_decision is not None:
        template = _copy_mod.deepcopy(next(d for d in state["decisions"] if d.get("id") == "DEC-009"))
        template.update(extra_decision)
        state["decisions"].append(template)
    state["gates"]["M22.2"]["authorized"] = m22_2_authorized
    state_path.write_text(_yaml.safe_dump(state, sort_keys=False, allow_unicode=True))
    return _validate_cp(cp, _REPO_ROOT)


def _has_m22_2_error(errors: list[str]) -> bool:
    return any(_M22_2_ERR in e for e in errors)


def test_m22_1_override_does_not_disarm_m22_2(tmp_path: Path):
    decision = {
        "id": "DEC-TEST-M22-1",
        "status": "ACCEPTED",
        "approved_by": "user",
        "decision": "M22.1 validated for test purposes",
    }
    assert _decision_grants(decision, "M22.1") is True
    assert _decision_grants(decision, "M22.2") is False
    errors = _gate_variant(tmp_path, decision, True)
    assert _has_m22_2_error(errors), errors


def test_gate_override_requires_accepted_status():
    for status in ("PROPOSED", "REJECTED", "SUPERSEDED", None):
        decision = {"approved_by": "user", "decision": "M22.2 authorized"}
        if status is not None:
            decision["status"] = status
        assert _decision_grants(decision, "M22.2") is False, status


def test_gate_override_requires_user_approval():
    decision = {"approved_by": "assistant", "status": "ACCEPTED", "decision": "M22.2 authorized"}
    assert _decision_grants(decision, "M22.2") is False


def test_m22_2_override_true_path_is_reachable(tmp_path: Path):
    decision = {
        "id": "DEC-TEST-M22-2",
        "status": "ACCEPTED",
        "approved_by": "user",
        "decision": "M22.2 authorized for test purposes",
    }
    assert _decision_grants(decision, "M22.2") is True
    assert not _has_m22_2_error(_gate_variant(tmp_path, decision, True))


def test_real_state_has_no_active_gate_override():
    state = _yaml.safe_load((_REPO_ROOT / "control_plane" / "STATE.yaml").read_text())
    decisions = state.get("decisions", [])
    assert not any(_decision_grants(d, "M22.1") for d in decisions)
    assert not any(_decision_grants(d, "M22.2") for d in decisions)
    assert state["gates"]["M22.2"]["authorized"] is False
