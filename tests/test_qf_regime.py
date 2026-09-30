"""Regime characterization rules. No model and no MRSM gate."""

from src.mrsm.qf_regime import (
    decide_regimes,
    heterogeneity,
    regime_cell,
    sign_class,
    slot_context_dependent,
)


def _cell(effects, nulls=None):
    return regime_cell(effects, [0.0, 0.0] if nulls is None else nulls)


def _slot(patterns, pooled, dependent=None):
    regimes = {}
    for name, effects in patterns.items():
        regimes[name] = _cell(effects)
    cells_for_flag = regimes
    flag = slot_context_dependent(cells_for_flag, pooled) if dependent is None else dependent
    return {"regimes": regimes, "context_dependent": flag, "pooled_mean": pooled}


def test_sign_classes_use_existing_bars():
    assert sign_class([0.2, 0.3]) == "POSITIVE"
    assert sign_class([-0.2, -0.3]) == "NEGATIVE"
    assert sign_class([1e-12, -1e-12]) == "NEAR_ZERO"
    assert sign_class([0.2, -0.3]) == "UNSTABLE"
    assert sign_class([0.2]) == "UNDEFINED_UNDER_CURRENT_CONTRACT"


def test_separable_cell_uses_regime_null_floor():
    supported = _cell([0.2, 0.3], [0.0, 0.0])
    assert supported["status"] == "CAUSAL_SIGNAL_SUPPORTED"
    buried = _cell([1e-6, 1e-6], [0.0, 0.0])
    assert buried["status"] == "CAUSAL_SIGNAL_NOT_SEPARATED_FROM_NULL"
    weak = _cell([0.2, -0.3], [0.0, 0.0])
    assert weak["status"] == "CAUSAL_SIGNAL_WEAK"
    assert weak["sign_class"] == "UNSTABLE"


def test_cancellation_is_not_a_mechanism_claim():
    cells = {
        "completion": _cell([0.2, 0.2]),
        "instruction": _cell([-0.2, -0.2]),
        "syntax": _cell([1e-8, 1e-8]),
    }
    assert slot_context_dependent(cells, 0.0) is True
    decision = decide_regimes([
        {"regimes": cells, "context_dependent": True},
    ])
    assert decision["regime_decision"] == "MIXED_SIGNAL"
    assert decision["mechanism_identity"] == "NOT_EVALUATED"


def test_all_context_dependent_separable_slots():
    left = _slot(
        {"completion": [0.2, 0.2], "instruction": [-0.2, -0.2], "syntax": [0.3, 0.3]},
        0.1,
    )
    right = _slot(
        {"completion": [-0.4, -0.4], "instruction": [0.4, 0.4], "syntax": [-0.2, -0.2]},
        0.0,
    )
    decision = decide_regimes([left, right])
    assert decision["regime_decision"] == "REGIME_SPECIFIC_CAUSAL_SIGNAL"
    assert decision["mechanism_identity"] == "NOT_EVALUATED"
    assert "does not show" in decision["primary_interpretation"]


def test_mixed_when_some_slots_change_sign_and_some_do_not():
    changing = _slot(
        {"completion": [0.2, 0.2], "instruction": [-0.2, -0.2], "syntax": [0.3, 0.3]},
        0.1,
    )
    stable = _slot(
        {"completion": [-0.2, -0.2], "instruction": [-0.3, -0.3], "syntax": [-0.25, -0.25]},
        -0.25,
    )
    decision = decide_regimes([changing, stable])
    assert decision["regime_decision"] == "MIXED_SIGNAL"
    assert "Every measured discovery regime cell" in decision["primary_interpretation"]
    assert decision["mechanism_identity"] == "NOT_EVALUATED"


def test_no_cell_separates_from_its_null():
    slot = _slot(
        {"completion": [1e-7, 1e-7], "instruction": [1e-8, -1e-8], "syntax": [0.0, 0.0]},
        0.0,
    )
    assert decide_regimes([slot])["regime_decision"] == "NO_REGIME_SEPARATION"


def test_heterogeneity_is_not_a_decision_input():
    stats = heterogeneity(
        {"completion": [0.2, 0.25], "instruction": [-0.2, -0.15], "syntax": [0.0, 0.05]}
    )
    assert stats["label"] == "EXPLORATORY"
    assert stats["used_as_gate"] is False
    assert stats["heterogeneity_ratio"] is not None
    missing = decide_regimes([{"regimes": {"completion": _cell([0.2, 0.2])}, "context_dependent": False}])
    assert missing["regime_decision"] == "INCONCLUSIVE"
