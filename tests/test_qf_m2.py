"""M2 comparison rules. No model and no MRSM gate."""

from src.mrsm.qf_m2 import (
    PERMUTATION_COUNT,
    cosine,
    decide,
    level_a,
    level_b,
    level_c,
    cross_model_null,
    r2_standardized,
    regime_permutations,
    sign_cell,
)


def _profiles():
    qwen = {
        "completion": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "instruction": [1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "syntax": [-1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    return qwen, {regime: list(values) for regime, values in qwen.items()}


def test_r2_is_undefined_when_the_null_has_no_spread():
    result = r2_standardized(0.2, 0.0, 0.0)
    assert result["value"] is None
    assert result["status"] == "UNDEFINED_BECAUSE_NULL_STD_IS_ZERO"


def test_permutation_count_is_fixed_by_three_regimes():
    assert len(regime_permutations()) == PERMUTATION_COUNT == 6


def test_matching_profiles_support_transfer_without_a_mechanism_claim():
    qwen, gpt2 = _profiles()
    null = cross_model_null(qwen, gpt2)
    matched = [cosine(qwen[regime], gpt2[regime]) for regime in ("completion", "instruction", "syntax")]
    assert level_b(null, matched) == "SUPPORTED"
    assert level_c(
        {"completion__instruction": 0.8, "completion__syntax": -0.4, "instruction__syntax": -0.2},
        {"completion__instruction": 0.7, "completion__syntax": -0.5, "instruction__syntax": -0.1},
    ) == "SUPPORTED"
    decision = decide("SUPPORTED", "SUPPORTED")
    assert decision["m2_decision"] == "CROSS_MODEL_REGIME_PATTERN_SUPPORTED"
    assert decision["m3_executed"] is False
    assert "mechanism identity" not in decision["primary_interpretation"]
    assert "better" not in decision["primary_interpretation"]


def test_reversed_profiles_do_not_support_transfer():
    qwen, gpt2 = _profiles()
    flipped = {regime: [-value for value in values] for regime, values in gpt2.items()}
    null = cross_model_null(qwen, flipped)
    matched = [cosine(qwen[regime], flipped[regime]) for regime in ("completion", "instruction", "syntax")]
    assert level_b(null, matched) == "NOT_SUPPORTED"
    decision = decide("NOT_SUPPORTED", "NOT_SUPPORTED")
    assert decision["m2_decision"] == "CROSS_MODEL_REGIME_PATTERN_NOT_SUPPORTED"
    assert "model-specific in general" not in decision["primary_interpretation"]


def test_sign_categories_and_partial_decision():
    assert sign_cell("+", "+", True) == "sign_agreement"
    assert sign_cell("+", "-", True) == "sign_disagreement"
    assert sign_cell("~", "+", True) == "near_zero"
    assert sign_cell("+", "+", False) == "not_comparable"
    assert level_a(["sign_agreement"] * 9 + ["sign_disagreement"] * 15) == "NOT_SUPPORTED"
    assert decide("PARTIAL", "NOT_SUPPORTED")["m2_decision"] == "PARTIAL_CROSS_MODEL_CONSISTENCY"
    assert decide("INCONCLUSIVE", "SUPPORTED")["m2_decision"] == "INCONCLUSIVE"
