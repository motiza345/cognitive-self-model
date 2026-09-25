"""M22.1.2 response-space audit tests. The real matrices are not required here."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from src.cognitive_self_model.m22_1.prompts import frozen_prompts, prompt_manifest_sha256
from src.cognitive_self_model.m22_1_2.data import load_config, prompts_by_split
from src.cognitive_self_model.m22_1_2.decision import assign_label
from src.cognitive_self_model.m22_1_2.masking import held_out_mask
from src.cognitive_self_model.m22_1_2.metrics import advantage_destroyed, beats
from src.cognitive_self_model.m22_1_2.models import PREDICTORS
from src.cognitive_self_model.m22_1_2.protocol import (
    METHOD_NAMES,
    _fit_kwargs,
    _package_leakage,
    build_manifest,
    score_held_out,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "m22_1_2_response_space_audit.json"
PACKAGE = ROOT / "src" / "cognitive_self_model" / "m22_1_2"
LABELS = {
    "PREDICTIVE_RESPONSE_STRUCTURE",
    "CONTEXT_DEPENDENT_STRUCTURE",
    "CALIBRATED_INTERVENTION_RESPONSE",
    "INSUFFICIENT_EVIDENCE",
}


def _config():
    return load_config(CONFIG_PATH)


def _summary(
    *,
    rank=0.01,
    global_mae=0.05,
    direction_mae=0.04,
    seeds=5,
    perm_rank=0.05,
    perm_global=0.05,
    perm_direction=0.05,
    regime=0.012,
    cal_rank=0.04,
    cal_constant=0.04,
    cal_global=0.05,
    cal_directions=8,
):
    return {
        "blocked_mae_mean": {
            "B0_global_mean": global_mae,
            "B2_direction_mean": direction_mae,
            "B4_rank1": rank,
        },
        "seeds_beating_both_baselines": seeds,
        "control_mae_mean": {
            "cell_permutation": {
                "B0_global_mean": perm_global,
                "B2_direction_mean": perm_direction,
                "B4_rank1": perm_rank,
            },
            "global_permutation": {
                "B0_global_mean": perm_global,
                "B2_direction_mean": perm_direction,
                "B4_rank1": perm_rank,
            },
        },
        "leave_one_regime_mae": regime,
        "calibration": {
            "rank1_mae": cal_rank,
            "constant_mae": cal_constant,
            "global_mae": cal_global,
            "directions_not_worse_than_constant": cal_directions,
            "permuted_rank1_mae": 0.05,
            "permuted_constant_mae": 0.05,
            "permuted_global_mae": 0.05,
        },
    }


def test_config_freezes_estimator_masks_and_status_before_fitting():
    config = _config()
    assert config["frozen_before_benchmark"] is True
    assert config["model_forward_passes"] == 0
    assert config["model_revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert config["hook_name"] == "blocks.23.hook_resid_post"
    assert config["prompt_manifest_sha256"] == "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db"
    assert config["primary_alpha"] == 1.0
    assert config["symmetry_alpha"] == -1.0
    assert config["primary_estimator"] == "ridge_als"
    assert config["decision_model"] == "B4_rank1"
    assert config["decision_rank"] == 1
    assert config["secondary_estimator_is_not_used_for_the_label"] is True
    assert config["mask_fraction"] == 0.25
    assert config["masks_per_prompt"] == 2
    assert config["mask_seeds"] == [221201, 221202, 221203, 221204, 221205]
    assert config["ridge_lambda"] == 0.01
    assert config["min_relative_mae_reduction"] == 0.1
    assert config["min_seeds_beating_both_baselines"] == 4
    assert config["permutation_advantage_remaining_max_fraction"] == 0.25
    assert config["calibration_option"] == "A"
    assert config["calibration_prompt_rule"] == "lexicographically_first_prompt_id_within_split"
    assert config["m22_1_status_preserved"] == "CANDIDATE"
    assert config["m22_2_authorized"] is False
    assert "PREDICTIVE if" in config["decision_rule"]
    assert prompt_manifest_sha256() == config["prompt_manifest_sha256"]
    assert len(frozen_prompts()) == 18


def test_masks_are_deterministic_stratified_and_outcome_free():
    config = _config()
    source = (PACKAGE / "masking.py").read_text(encoding="utf-8")
    assert "actual_delta" not in source
    signature = ast.parse(source)
    names = [node.name for node in signature.body if isinstance(node, ast.FunctionDef)]
    assert "held_out_mask" in names
    for seed in config["mask_seeds"]:
        first = held_out_mask(6, 8, seed, config["base_mask_pairs"])
        second = held_out_mask(6, 8, seed, config["base_mask_pairs"])
        assert np.array_equal(first, second)
        assert first.sum() == 12
        assert np.all(first.sum(axis=1) == 2)
        assert np.all((~first).sum(axis=0) >= 1)
        assert np.all(first.sum(axis=0) >= 1)
    assert not np.array_equal(
        held_out_mask(6, 8, 221201, config["base_mask_pairs"]),
        held_out_mask(6, 8, 221202, config["base_mask_pairs"]),
    )


def test_manifest_is_stable_and_selects_calibration_without_deltas():
    config = _config()
    first = build_manifest(config)
    second = build_manifest(config)
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["model_forward_passes"] == 0
    grouped, _ = prompts_by_split()
    for split, prompts in grouped.items():
        assert first["calibration_prompt_id"][split] == prompts[0].prompt_id
        assert len(first["mask_schedule"][split]) == 5
        assert len(first["mask_schedule"][split][0]["held_out_cells"]) == 12


def test_held_out_values_do_not_change_fits():
    config = _config()
    generator = np.random.default_rng(11)
    matrix = generator.normal(size=(6, 8))
    held = held_out_mask(6, 8, 221201, config["base_mask_pairs"])
    contaminated = np.array(matrix, copy=True)
    contaminated[held] = 1e9
    for method in METHOD_NAMES:
        clean = PREDICTORS[method](matrix, ~held, **_fit_kwargs(config, method))
        dirty = PREDICTORS[method](contaminated, ~held, **_fit_kwargs(config, method))
        assert np.allclose(clean, dirty, atol=1e-8, rtol=1e-7)


def test_rank1_recovers_a_separable_matrix_better_than_baselines():
    config = _config()
    prompts = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.7])
    directions = np.array([-1.5, -0.4, 0.2, 0.5, 0.9, 1.3, -0.8, 0.3])
    matrix = np.outer(prompts, directions)
    held = held_out_mask(6, 8, 221201, config["base_mask_pairs"])
    scored = score_held_out(matrix, held, config)
    assert scored["B4_rank1"]["mae"] < 0.5 * scored["B0_global_mean"]["mae"]
    assert scored["B4_rank1"]["mae"] < 0.5 * scored["B2_direction_mean"]["mae"]


def test_decision_rule_covers_each_frozen_label():
    config = _config()
    predictive = _summary()
    context = _summary(regime=0.02)
    calibrated_split = _summary(
        rank=0.04,
        direction_mae=0.041,
        seeds=5,
        cal_rank=0.01,
        cal_constant=0.04,
        cal_global=0.05,
    )
    insufficient = _summary(rank=0.05, global_mae=0.05, direction_mae=0.05, seeds=0)
    assert assign_label(predictive, predictive, True, config)["label"] == "PREDICTIVE_RESPONSE_STRUCTURE"
    assert assign_label(context, context, True, config)["label"] == "CONTEXT_DEPENDENT_STRUCTURE"
    assert assign_label(calibrated_split, calibrated_split, True, config)["label"] == "CALIBRATED_INTERVENTION_RESPONSE"
    assert assign_label(insufficient, insufficient, True, config)["label"] == "INSUFFICIENT_EVIDENCE"
    one_split_fails = _summary(rank=0.05, global_mae=0.05, direction_mae=0.05, seeds=0)
    assert assign_label(predictive, context, True, config)["label"] == "PREDICTIVE_RESPONSE_STRUCTURE"
    assert assign_label(predictive, one_split_fails, True, config)["label"] == "INSUFFICIENT_EVIDENCE"
    assert assign_label(predictive, predictive, False, config)["label"] == "INSUFFICIENT_EVIDENCE"
    assert assign_label(predictive, predictive, True, config)["label"] in LABELS
    assert beats(0.01, 0.02, 0.1, 1e-8) is True
    assert beats(0.0, 0.0, 0.1, 1e-8) is False
    assert advantage_destroyed(0.05, 0.01, 0.05, 0.048, 0.25) is True
    assert advantage_destroyed(0.05, 0.01, 0.05, 0.02, 0.25) is False


def test_package_has_no_m20_m21_or_outcome_features():
    offenders = _package_leakage(PACKAGE)
    assert offenders == []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any("torch" in name for name in imported)


def test_saved_audit_artifacts_match_schema():
    report_dir = ROOT / "reports"
    manifest = json.loads((report_dir / "m22_1_2_response_space_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((report_dir / "m22_1_2_response_space_results.json").read_text(encoding="utf-8"))
    report = (report_dir / "m22_1_2_response_space_report.md").read_text(encoding="utf-8")
    readme = (report_dir / "m22_1_2_response_space_README.md").read_text(encoding="utf-8")
    assert manifest["manifest_sha256"] == results["manifest_sha256"]
    assert manifest["manifest_sha256"] == build_manifest(_config())["manifest_sha256"]
    assert results["model_forward_passes"] == 0
    assert results["m22_1_status"] == "CANDIDATE"
    assert results["m22_2_authorized"] is False
    assert results["discovery_used_for_decision"] is False
    assert results["label"]["label"] in LABELS
    assert results["label"]["secondary_svd_not_used"] is True
    assert set(results["evaluation_splits"]) == {"validation", "replication"}
    for split in ("validation", "replication"):
        assert "B4_rank1" in results["splits"][split]["blocked_mae_mean"]
        assert "secondary_imputed_svd_mae_mean" in results["splits"][split]
    assert results["label"]["label"] in report
    assert "not authorized" in readme
    assert not (report_dir / "validated_intervention.json").exists()
