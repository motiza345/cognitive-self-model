"""M1 comparison rules. No model download and no MRSM gate."""

from src.mrsm import HEADS
from src.mrsm.qf_cross_model import (
    FROZEN_QWEN_MATRIX,
    decide_m1,
    functional_layers,
    qwen_coordinate_layers,
    slot_catalog,
)
from src.mrsm.qf_regime import FROZEN_REGIMES, regime_cell


def _signed(symbol: str):
    effects = [0.2, 0.25] if symbol == "+" else [-0.2, -0.25]
    return regime_cell(effects, [0.0, 0.0])


def _matrix_cells(matrix):
    return {
        slot: {regime: _signed(matrix[slot][regime]) for regime in FROZEN_REGIMES}
        for slot in HEADS
    }


def test_counterpart_grid_is_fixed_before_effects():
    assert qwen_coordinate_layers() == [0, 8, 15, 23]
    assert functional_layers(12) == [0, 4, 7, 11]
    assert functional_layers(3) is None
    functional = slot_catalog([0, 4, 7, 11], 768, n_layers=12)
    coordinate = slot_catalog(qwen_coordinate_layers(), 768, n_layers=12)
    assert [row["slot"] for row in functional] == HEADS
    assert [row["layer"] for row in functional] == [0, 0, 4, 4, 7, 7, 11, 11]
    assert [row["comparable"] for row in coordinate] == [
        True, True, True, True, False, False, False, False
    ]
    assert functional[0]["vector_sha256"] == coordinate[0]["vector_sha256"]
    assert functional[2]["layer"] != coordinate[2]["layer"]


def test_reproduced_requires_the_full_sign_matrix():
    decision = decide_m1(comparable=True, cells=_matrix_cells(FROZEN_QWEN_MATRIX))
    assert decision["m1_decision"] == "MODEL_B_SIGNAL_REPRODUCED"
    assert decision["mechanism_identity"] == "NOT_EVALUATED"
    assert decision["m2_executed"] is False
    assert "better" not in decision["key_difference"]


def test_one_sign_difference_is_not_reproduction():
    matrix = {slot: dict(signs) for slot, signs in FROZEN_QWEN_MATRIX.items()}
    matrix["L1H3"] = dict(matrix["L1H3"])
    matrix["L1H3"]["syntax"] = "+"
    decision = decide_m1(comparable=True, cells=_matrix_cells(matrix))
    assert decision["m1_decision"] == "MODEL_B_SIGNAL_PRESENT_BUT_DIFFERENT"
    assert "not a claim that Model B is better" in decision["inferred"]
    assert "L1H3 syntax" in decision["key_difference"]


def test_null_separation_failure_is_not_a_method_verdict():
    cells = {
        slot: {regime: regime_cell([1e-8, 1e-8], [0.0, 0.0]) for regime in FROZEN_REGIMES}
        for slot in HEADS
    }
    decision = decide_m1(comparable=True, cells=cells)
    assert decision["m1_decision"] == "MODEL_B_SIGNAL_NOT_SEPARATED_FROM_NULL"
    assert "invalid" in decision["inferred"]


def test_incomparable_and_incomplete_stop():
    refused = decide_m1(comparable=False, cells=None, incomparability_reason="outcome tokens are not single pieces")
    assert refused["m1_decision"] == "MODEL_B_NOT_COMPARABLE"
    missing = decide_m1(comparable=True, cells={"L0H0": {}})
    assert missing["m1_decision"] == "M1_INCONCLUSIVE"
