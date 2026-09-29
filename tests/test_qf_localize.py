"""Rules for the QF diagnostic. These tests do not load Qwen and do not rescore MRSM."""

import json
from pathlib import Path

from src.mrsm.qf_localize import (
    FROZEN_Q_B2_MAE,
    FROZEN_Q_SELF_MAE,
    assign_partitions,
    classify_signal,
    diagnose,
    discovery_stability,
    partitions_are_regime_confounded,
    render_report,
    scores_from_saved,
)

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_partitions_follow_sorted_ids_and_are_confounded():
    prompt_ids = [
        "completion-01",
        "completion-04",
        "instruction-01",
        "instruction-04",
        "syntax-01",
        "syntax-04",
    ]
    bins = assign_partitions(prompt_ids)
    assert bins["D1"] == ["completion-01", "syntax-01"]
    assert bins["D2"] == ["completion-04", "syntax-04"]
    assert bins["D3"] == ["instruction-01"]
    assert bins["D4"] == ["instruction-04"]
    records = [
        {"prompt_id": "completion-01", "regime_id": "completion"},
        {"prompt_id": "completion-04", "regime_id": "completion"},
        {"prompt_id": "instruction-01", "regime_id": "instruction"},
        {"prompt_id": "instruction-04", "regime_id": "instruction"},
        {"prompt_id": "syntax-01", "regime_id": "syntax"},
        {"prompt_id": "syntax-04", "regime_id": "syntax"},
    ]
    assert partitions_are_regime_confounded(records, bins)


def test_stability_labels():
    identified = lambda name: {"status": "IDENTIFIED", "candidate": name}
    four = [identified("L0H0->L0H1") for _ in range(4)]
    assert discovery_stability(four) == "STABLE"
    mixed = [identified("L0H0->L0H1"), identified("L0H2->L0H3"), identified("L0H0->L0H1"), identified("L1H0->L1H1")]
    assert discovery_stability(mixed) == "UNSTABLE"
    assert discovery_stability([identified("L0H0->L0H1"), {"status": "NOT_IDENTIFIABLE", "candidate": None}]) == "UNIDENTIFIABLE"


def test_signal_and_diagnosis_rules():
    clear = [{"separable": True, "sign_consistency": 1.0, "regime_conflict": False} for _ in range(4)]
    assert classify_signal(clear) == "SIGNAL_CLEAR"
    weak = [{"separable": True, "sign_consistency": 0.5, "regime_conflict": False}]
    assert classify_signal(weak) == "SIGNAL_WEAK"
    assert classify_signal([{"separable": False, "sign_consistency": 1.0, "regime_conflict": False}]) == "SIGNAL_NOT_IDENTIFIABLE"
    label, _text = diagnose(
        discovery="UNSTABLE",
        signal="SIGNAL_WEAK",
        representation="INCONCLUSIVE",
        oracle="INCONCLUSIVE",
        ceiling_near=True,
        partitions_confounded=True,
    )
    assert label == "INTERVENTION_BOTTLENECK"
    label, _text = diagnose(
        discovery="UNSTABLE",
        signal="SIGNAL_CLEAR",
        representation="INCONCLUSIVE",
        oracle="INCONCLUSIVE",
        ceiling_near=True,
        partitions_confounded=True,
    )
    assert label == "INCONCLUSIVE"
    label, text = diagnose(
        discovery="STABLE",
        signal="SIGNAL_CLEAR",
        representation="INCONCLUSIVE",
        oracle="INCONCLUSIVE",
        ceiling_near=True,
        partitions_confounded=True,
    )
    assert label == "PREDICTION_BOTTLENECK"
    assert "predictor" in text


def test_saved_q_scores_match_the_frozen_run():
    evidence = json.loads((ROOT / "artifacts/mrsm/q_run_001/evidence.json").read_text(encoding="utf-8"))
    predictions = json.loads((ROOT / "artifacts/mrsm/q_run_001/predictions.json").read_text(encoding="utf-8"))
    saved = scores_from_saved(evidence["records"], predictions["baselines"]["B2"])
    assert abs(saved["self_mae"] - FROZEN_Q_SELF_MAE) < 1e-9
    assert abs(saved["b2_mae"] - FROZEN_Q_B2_MAE) < 1e-9


def test_report_stays_diagnostic():
    payload = {
        "qf1": {
            "label": "UNSTABLE",
            "partition_regime_confounded": True,
            "rows": [{
                "partition": "D1",
                "candidate": None,
                "effect": None,
                "sign": None,
                "support": 0,
                "contradiction": 0,
                "status": "NOT_IDENTIFIABLE",
            }],
            "tracked_edge": [{
                "partition": "D1",
                "effect": 0.0,
                "member_effects": {"L0H0": 0.0, "L0H1": 0.0},
                "sign": 0,
                "support": 0,
                "contradiction": 2,
                "selected": False,
            }],
        },
        "qf2": {"label": "SIGNAL_WEAK", "floor": 0.0001, "floor_rule": "test", "rows": []},
        "qf3": {
            "status": "INCONCLUSIVE",
            "reason": "missing",
            "families": {"R1": {
                "self_model_mae": 0.1,
                "b2_linear_probe_mae": 0.1,
                "sign_accuracy": 0.5,
                "feature_dimensionality": 36,
                "sample_count_discovery_prompts": 6,
                "train_test_isolation": "loo",
                "saved_validation_self_mae": 0.027,
                "saved_validation_b2_mae": 0.028,
            }},
        },
        "qf4": {"status": "ORACLE_INVALID", "interpretation": "INCONCLUSIVE", "reason": "undefined", "o0_mae": 0.027, "b2_mae": 0.028},
        "qf5": {
            "status": "global",
            "regime": {"status": "NOT_AVAILABLE", "reason": "absent"},
            "in_distribution_vs_ood": {"status": "NOT_AVAILABLE", "reason": "absent"},
            "buckets": [],
        },
        "qf6": {
            "status": "NEAR_CEILING",
            "feature_dimensionality_b2": 8,
            "feature_dimensionality_self": 36,
            "effective_sample_count_discovery_prompts": 6,
            "train_split": "discovery",
            "test_split_for_frozen_mae": "validation",
            "b2_sees_information_unavailable_to_self_model": False,
            "b2_mae_min": 0.1,
            "b2_mae_max": 0.2,
            "approximately_linear": "within band",
            "frozen_validation": {"self_mae": 0.027, "b2_mae": 0.028, "relative_reduction": 0.03},
        },
        "matrix": [{"layer": "Discovery", "evidence": "QF-1", "status": "UNSTABLE"}],
        "primary_bottleneck": "INCONCLUSIVE",
        "next_intervention": "Do not modify the Self-Model.",
        "reproducibility": {"git_commit": "abc", "git_branch": "qf", "python": "3.12", "torch": "cpu", "transformer_lens": "x", "numpy": "y", "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987", "device": "cpu", "weight_sha256": "0", "prereg_sha256": "1", "final_result_sha256": "2"},
    }
    text = render_report(payload)
    assert "DIAGNOSTIC_ONLY" in text
    assert "H1 = FAIL" in text
    assert "does not claim that MRSM succeeds on Qwen" in text
    assert "REDEFINE_SCALE" in text
