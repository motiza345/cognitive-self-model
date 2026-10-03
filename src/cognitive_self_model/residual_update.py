"""Explicit residual correction for M26.

g stays the output of the frozen response model. The correction is a
Normal-Normal posterior mean of the residual, with a prior fixed before
any M26 outcome.
"""

from __future__ import annotations

import math
import random

from src.cognitive_self_model.m23.belief import CRITICAL_Z

PROTOCOL_VERSION = "M26.INDEPENDENT_EVIDENCE_UPDATE.1"
FAMILY = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")
PRIOR_MEAN = 0.0
PRIOR_COUNT = 1.0
SHUFFLE_SEED = 26001

# Published M25 calibration scales at inflation 1. Not re-estimated here.
FROZEN_SIGMA = {
    "M22.1-D1-L0": 0.02029281160603693,
    "M22.1-D1-L8": 0.009377970182012517,
    "M22.1-D1-L15": 0.004044481362351381,
}


def posterior_delta(
    residuals: list[float],
    *,
    prior_mean: float = PRIOR_MEAN,
    prior_count: float = PRIOR_COUNT,
) -> float:
    """δ = (n0 μ0 + n r_bar) / (n0 + n). Empty evidence returns the prior mean."""
    if prior_count <= 0.0 or not math.isfinite(prior_count):
        raise ValueError("prior count must be positive")
    if not math.isfinite(prior_mean):
        raise ValueError("prior mean must be finite")
    values = [float(value) for value in residuals]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("residuals must be finite")
    n = len(values)
    if n == 0:
        return float(prior_mean)
    residual_mean = sum(values) / n
    return (prior_count * float(prior_mean) + n * residual_mean) / (prior_count + n)


def posterior_sd(sigma: float, n: int, *, prior_count: float = PRIOR_COUNT) -> float:
    """τ = σ / sqrt(n0 + n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if prior_count <= 0.0:
        raise ValueError("prior count must be positive")
    return float(sigma) / math.sqrt(prior_count + n)


def predictive_sd(sigma: float, n: int, *, prior_count: float = PRIOR_COUNT) -> float:
    """Scale of a new residual around g + δ: sqrt(τ² + σ²)."""
    tau = posterior_sd(sigma, n, prior_count=prior_count)
    return math.sqrt(tau * tau + float(sigma) * float(sigma))


def prediction_after(g: float, delta: float) -> float:
    return float(g) + float(delta)


def prediction_before(g: float) -> float:
    return float(g)


def cell_deltas(residuals_by_cell: dict[str, list[float]]) -> dict[str, float]:
    if set(residuals_by_cell) != set(FAMILY):
        raise ValueError("cell deltas require exactly the frozen three cells")
    return {cell: posterior_delta(residuals_by_cell[cell]) for cell in FAMILY}


def global_delta(residuals_by_cell: dict[str, list[float]]) -> float:
    if set(residuals_by_cell) != set(FAMILY):
        raise ValueError("global delta requires exactly the frozen three cells")
    pooled = [value for cell in FAMILY for value in residuals_by_cell[cell]]
    return posterior_delta(pooled)


def update_residuals(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    """Residuals from the update partition only. Holdout rows are rejected."""
    grouped: dict[str, list[float]] = {cell: [] for cell in FAMILY}
    if not rows:
        raise ValueError("update residuals require rows")
    for row in rows:
        if row.get("partition") != "update":
            raise ValueError("residual update accepts the update partition only")
        intervention_id = str(row["intervention_id"])
        if intervention_id not in grouped:
            raise ValueError("residual update received a cell outside the frozen family")
        grouped[intervention_id].append(float(row["observed_effect"]) - float(row["g"]))
    if any(len(values) == 0 for values in grouped.values()):
        raise ValueError("each cell needs at least one update residual")
    return grouped


def shuffled_residuals(
    rows: list[dict[str, object]],
    *,
    seed: int = SHUFFLE_SEED,
) -> dict[str, list[float]]:
    """Permute update residuals across cells. One stream, seed 26001."""
    ordered = sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))
    if any(row.get("partition") != "update" for row in ordered):
        raise ValueError("shuffled residuals accept the update partition only")
    residuals = [float(row["observed_effect"]) - float(row["g"]) for row in ordered]
    random.Random(seed).shuffle(residuals)
    grouped: dict[str, list[float]] = {cell: [] for cell in FAMILY}
    for row, residual in zip(ordered, residuals):
        grouped[str(row["intervention_id"])].append(residual)
    if any(len(values) == 0 for values in grouped.values()):
        raise ValueError("shuffled residuals lost a cell")
    return grouped


def update_verdict(
    primary_class: str,
    versus_global_class: str,
    shuffled_class: str,
    leakage_ok: bool,
) -> str:
    """Frozen map. Classes are the existing ci_class labels.

    primary: MAE(g) - MAE(g + δ_cell) on the holdout.
    versus_global: MAE(g + δ_global) - MAE(g + δ_cell).
    shuffled: MAE(g) - MAE(g + δ_shuffled).
    """
    if not leakage_ok:
        return "INCONCLUSIVE"
    if (
        primary_class == "CI_POSITIVE"
        and versus_global_class == "CI_POSITIVE"
        and shuffled_class != "CI_POSITIVE"
    ):
        return "UPDATE_SUPPORTED"
    if primary_class == "CI_NEGATIVE":
        return "UPDATE_NOT_SUPPORTED"
    return "INCONCLUSIVE"
