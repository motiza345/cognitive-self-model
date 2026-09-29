"""Engineering tests for the M22.1-R harness. No Qwen load and no MRSM run."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from src.cognitive_self_model.m22_1_r.comparator import compare_bundles
from src.cognitive_self_model.m22_1_r.config import REQUIRED_REVISION, load_replay_config
from src.cognitive_self_model.m22_1_r.harness import run_harness, splice_execution_records
from src.cognitive_self_model.m22_1_r.hashing import file_sha256
from src.cognitive_self_model.m22_1_r.record import SENTINELS, assert_no_sentinels
from src.cognitive_self_model.m22_1_r.strict_loader import ReplayBlockedError, load_pinned_revision

ROOT = Path(__file__).resolve().parents[1]
REPLAY_YAML = ROOT / "configs" / "m22_1_r_replay.yaml"
FROZEN_CONFIG_SHA256 = "d9e1cf5dc3b88535cda0516b9b24ae3510082b7fed452f2e8adb86c574e8039a"
FROZEN_TOLERANCES = {
    "score_atol": 1.0e-5,
    "score_rtol": 1.0e-5,
    "delta_atol": 1.0e-5,
    "mean_atol": 1.0e-6,
    "null_atol": 1.0e-6,
}
CLEAN = {
    "commit": "d96b3eeed1a3a4a8f94b1ac372e35b9a8489cd60",
    "branch": "cursor/m22-1-r-harness-4e19",
    "dirty": False,
    "origin": "https://example.invalid/cognitive-self-model",
}


def _clock() -> str:
    return "2026-09-29T00:00:00+00:00"


def _snapshot(root: Path, revision: str = REQUIRED_REVISION) -> Path:
    snapshot = root / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"not-a-real-weight")
    return snapshot


def _walk_sentinels(value, prefix="record"):
    if value is None:
        raise AssertionError(f"{prefix} is null")
    if isinstance(value, str):
        assert value.strip() not in SENTINELS, prefix
        assert value != "UNKNOWN", prefix
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_sentinels(item, f"{prefix}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _walk_sentinels(item, f"{prefix}[{index}]")


def test_a_different_revision_is_not_accepted():
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("from_pretrained must not be called")

    with pytest.raises(ReplayBlockedError, match="does not match the recorded pin"):
        load_pinned_revision(
            "Qwen/Qwen2.5-0.5B",
            "ffffffffffffffffffffffffffffffffffffffff",
            from_pretrained=spy,
            dtype="float32",
        )
    assert calls == []


def test_b_pinned_load_failure_does_not_retry_unpinned(tmp_path: Path):
    _snapshot(tmp_path)
    calls = []

    def spy(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        raise RuntimeError("offline revision unavailable")

    record = run_harness(
        ROOT,
        output_path=tmp_path / "execution_record.json",
        cache_root=tmp_path,
        identity=CLEAN,
        execute_forward=True,
        from_pretrained=spy,
        clock=_clock,
    )
    assert record["replay_tier"] == "REPLAY_BLOCKED"
    assert record["scientific_failure"] is False
    assert len(calls) == 1
    assert calls[0]["revision"] == REQUIRED_REVISION
    assert calls[0]["local_files_only"] is True
    assert record["forward_executed"] is False


def test_c_weight_hash_mismatch_rejects_before_load(tmp_path: Path):
    _snapshot(tmp_path)
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        raise AssertionError("from_pretrained must not be called")

    record = run_harness(
        ROOT,
        output_path=tmp_path / "execution_record.json",
        cache_root=tmp_path,
        identity=CLEAN,
        execute_forward=True,
        from_pretrained=spy,
        expected_weight_hashes={"model.safetensors": "0" * 64},
        clock=_clock,
    )
    assert record["replay_tier"] == "REPLAY_DIVERGENT"
    assert record["failure_class"] == "weight_identity"
    assert record["scientific_failure"] is False
    assert calls == []
    assert record["weight_mismatch"] != "none"


def test_d_reference_copy_is_exact_and_tiny_delta_is_numeric():
    config = load_replay_config(ROOT)
    from src.cognitive_self_model.m22_1_r.harness import load_reference_bundle

    reference = load_reference_bundle(ROOT)
    exact = compare_bundles(reference, copy.deepcopy(reference), config)
    assert exact["replay_tier"] == "REPLAY_EXACT"
    assert exact["dec007_success"] is True
    assert exact["invalid_reasons"] == []

    perturbed = copy.deepcopy(reference)
    row = perturbed["intervention_preflight_results.json"]["discovery_rows"][0]
    row["s_null"] = float(row["s_null"]) + 1.0e-8
    numeric = compare_bundles(reference, perturbed, config)
    assert numeric["replay_tier"] == "REPLAY_NUMERIC_EQUIVALENT"
    assert numeric["dec007_success"] is True
    assert numeric["failure_class"] == "none"


def test_e_tolerance_violation_is_not_a_success_or_a_block():
    config = load_replay_config(ROOT)
    from src.cognitive_self_model.m22_1_r.harness import load_reference_bundle

    reference = load_reference_bundle(ROOT)
    perturbed = copy.deepcopy(reference)
    row = perturbed["intervention_preflight_results.json"]["discovery_rows"][0]
    row["s_null"] = float(row["s_null"]) + 1.0
    result = compare_bundles(reference, perturbed, config)
    assert result["replay_tier"] == "REPLAY_BEHAVIORAL_EQUIVALENT"
    assert result["failure_class"] == "tolerance"
    assert result["dec007_success"] is False
    assert result["scientific_failure"] is False
    assert result["replay_tier"] != "REPLAY_BLOCKED"
    assert result["replay_tier"] != "REPLAY_EXACT"

    broken = copy.deepcopy(reference)
    broken["intervention_preflight_results.json"]["status"] = "NOT_THE_RECORDED_STATUS"
    divergent = compare_bundles(reference, broken, config)
    assert divergent["replay_tier"] == "REPLAY_DIVERGENT"
    assert divergent["failure_class"] == "identity"

    nonfinite = copy.deepcopy(reference)
    nonfinite["intervention_preflight_results.json"]["discovery_rows"][0]["s_intervened"] = math.nan
    invalid = compare_bundles(reference, nonfinite, config)
    assert invalid["replay_tier"] == "REPLAY_INVALID"
    assert invalid["failure_class"] == "incomplete_or_nonfinite"


def test_f_missing_snapshot_is_blocked_not_a_scientific_failure(tmp_path: Path):
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)

    record = run_harness(
        ROOT,
        output_path=tmp_path / "execution_record.json",
        cache_root=tmp_path,
        identity=CLEAN,
        execute_forward=True,
        from_pretrained=spy,
        clock=_clock,
    )
    assert record["replay_tier"] == "REPLAY_BLOCKED"
    assert record["failure_class"] == "prerequisite"
    assert record["scientific_failure"] is False
    assert record["local_snapshot_present"] is False
    assert calls == []
    assert REQUIRED_REVISION in record["block_reason"]


def test_g_execution_record_has_no_unknown_sentinels(tmp_path: Path):
    record = run_harness(
        ROOT,
        output_path=tmp_path / "execution_record.json",
        cache_root=tmp_path,
        identity=CLEAN,
        clock=_clock,
    )
    required = [
        "record_id",
        "milestone",
        "entrypoint",
        "invocation",
        "git_commit",
        "git_dirty",
        "git_branch",
        "python",
        "os",
        "numpy",
        "torch",
        "transformers",
        "transformer_lens",
        "cuda_available",
        "cuda_version",
        "gpu_identity",
        "model_id",
        "model_revision",
        "local_snapshot_identity",
        "weight_files",
        "expected_weight_hashes",
        "reference_artifacts",
        "reference_manifest_sha256",
        "holdout_identity",
        "config_path",
        "config_sha256",
        "tolerance_values",
        "timestamp",
        "replay_tier",
        "warnings",
        "block_reason",
        "seed",
    ]
    for key in required:
        assert key in record
    _walk_sentinels(record)
    assert_no_sentinels(record)
    assert record["model_revision"] == REQUIRED_REVISION
    assert record["cuda_available"] is False
    assert record["cuda_version"] == "N/A"
    assert "false" in record["cuda_version_reason"]
    assert record["gpu_identity"] == "N/A"
    assert record["expected_weight_hashes"] == "ABSENT_NO_REFERENCE_RECORDED"
    assert record["git_dirty"] is False
    assert "dirty_reason" not in record


def test_h_harness_does_not_modify_tolerance_configuration(tmp_path: Path):
    before = file_sha256(REPLAY_YAML)
    config = load_replay_config(ROOT)
    from src.cognitive_self_model.m22_1_r.harness import load_reference_bundle

    reference = load_reference_bundle(ROOT)
    compare_bundles(reference, copy.deepcopy(reference), config)
    run_harness(
        ROOT,
        output_path=tmp_path / "execution_record.json",
        cache_root=tmp_path,
        identity=CLEAN,
        clock=_clock,
    )
    after = file_sha256(REPLAY_YAML)
    assert before == after == FROZEN_CONFIG_SHA256
    assert config.tolerances == FROZEN_TOLERANCES
    reloaded = load_replay_config(ROOT)
    assert reloaded.tolerances == FROZEN_TOLERANCES
    assert reloaded.recorded_revision == REQUIRED_REVISION
    assert "REPLAY_BEHAVIORAL_EQUIVALENT" not in reloaded.document["dec007_success_tiers"]["tiers"]
    assert list(reloaded.document["dec007_success_tiers"]["tiers"]) == [
        "REPLAY_EXACT",
        "REPLAY_NUMERIC_EQUIVALENT",
    ]


def test_strict_loader_does_not_call_live_revision_resolution():
    source = (ROOT / "src" / "cognitive_self_model" / "m22_1_r" / "strict_loader.py").read_text()
    assert "resolve_hf_revision" not in source
    assert "model_info" not in source


def test_recovery_splice_keeps_milestone_text():
    original = "header:\n  note: keep\nexecution_records: []\nmilestones:\n  M18:\n    commit: UNKNOWN\n"
    record = {
        "entrypoint": "scripts/run_m22_1_r_harness.py",
        "invocation": "PYTHONPATH=src python3 scripts/run_m22_1_r_harness.py",
        "git_dirty": False,
        "replay_tier": "REPLAY_BLOCKED",
    }
    updated = splice_execution_records(original, record)
    assert "commit: UNKNOWN" in updated
    assert "REPLAY_BLOCKED" in updated
    assert updated.startswith("header:\n  note: keep\n")
