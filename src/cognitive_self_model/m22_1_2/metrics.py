"""Held-out scores. Correlation is descriptive and is not a decision metric."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def _correlation(left: np.ndarray, right: np.ndarray, method) -> float | None:
    if left.size < 2 or float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return None
    statistic = float(method(left, right).statistic)
    if not np.isfinite(statistic):
        return None
    return statistic


def cell_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    truth = np.asarray(actual, dtype=np.float64).ravel()
    estimate = np.asarray(predicted, dtype=np.float64).ravel()
    if truth.shape != estimate.shape or truth.size == 0:
        raise ValueError("Metric inputs must be non-empty and aligned.")
    error = estimate - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "pearson": _correlation(truth, estimate, pearsonr),
        "spearman": _correlation(truth, estimate, spearmanr),
        "n_cells": int(truth.size),
    }


def beats(model_mae: float, baseline_mae: float, reduction: float, floor: float) -> bool:
    if not np.isfinite(model_mae) or not np.isfinite(baseline_mae):
        return False
    if float(baseline_mae) <= float(floor):
        return False
    return float(model_mae) <= (1.0 - float(reduction)) * float(baseline_mae)


def advantage_destroyed(
    real_baseline: float,
    real_model: float,
    permuted_baseline: float,
    permuted_model: float,
    remaining_fraction: float,
) -> bool:
    real_advantage = float(real_baseline) - float(real_model)
    permuted_advantage = float(permuted_baseline) - float(permuted_model)
    if real_advantage <= 0.0:
        return False
    return permuted_advantage <= float(remaining_fraction) * real_advantage


def bootstrap_mean_interval(values: np.ndarray, seed: int, n_bootstrap: int) -> dict:
    sample = np.asarray(values, dtype=np.float64).ravel()
    if sample.size == 0:
        raise ValueError("Bootstrap sample is empty.")
    generator = np.random.default_rng(int(seed))
    draws = generator.choice(sample, size=(int(n_bootstrap), sample.size), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(sample.mean()), "low": float(low), "high": float(high), "n": int(sample.size)}
