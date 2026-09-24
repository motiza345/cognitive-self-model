"""
Frozen reference self-model.

The reference self-model is intentionally frozen at ``y_pred = 0.6 * x + 0.5 * u``.
It is the single fixed predictor whose contextual validity the benchmark probes;
the true causal environment may deviate from it per track and per timestep.
"""

from __future__ import annotations

REFERENCE_H1: float = 0.6
REFERENCE_H4: float = 0.5


def frozen_reference_prediction(x: float, u: float) -> float:
    """Return the frozen reference self-model prediction ``0.6*x + 0.5*u``."""
    return REFERENCE_H1 * x + REFERENCE_H4 * u
