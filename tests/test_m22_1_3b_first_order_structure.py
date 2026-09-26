"""Synthetic unit tests for M22.1.3b gates. No model, no stored-cell metrics."""

from __future__ import annotations

import numpy as np

from src.cognitive_self_model.m22_1_3b.compute import (
    degenerate,
    leading_energy,
    least_squares_scale,
)


def test_g3_least_squares_recovers_known_scale():
    predictor = np.array([0.5, -1.0, 2.0, 0.25, 1.5, -0.75, 0.1, 3.0], dtype=np.float64)
    known = -0.37
    target = known * predictor
    coef, rel = least_squares_scale(target, predictor)
    assert abs(coef - known) < 1e-12
    assert rel < 1e-12


def test_g2_rank1_t1_has_unit_leading_energy():
    left = np.array([1.0, 2.0, 0.5, -1.0, 0.25, 3.0], dtype=np.float64)
    right = np.array([0.4, -0.2, 1.0, 0.5, 0.1, -0.3, 0.8, 0.05], dtype=np.float64)
    t1 = np.outer(left, right)
    assert abs(leading_energy(t1) - 1.0) < 1e-9


def test_degenerate_guard_triggers_on_tiny_t2():
    t1 = np.ones(8, dtype=np.float64)
    t2 = 1e-13 * t1
    assert degenerate(t2, t1) is True
    assert degenerate(0.1 * t1, t1) is False
