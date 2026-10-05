"""Paired percentile bootstrap. The resample count is an implementation choice."""

from __future__ import annotations

import numpy as np


def bootstrap_indices(n_seeds: int, n_replicates: int, seed: int) -> np.ndarray:
    if n_seeds < 2:
        raise ValueError("paired bootstrap needs at least two seeds")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n_seeds), size=(int(n_replicates), int(n_seeds)))


def summarize_paired(values, indices) -> dict:
    array = np.asarray(values, dtype=float)
    if array.shape != (indices.shape[1],):
        raise ValueError("paired values do not match the bootstrap index width")
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "low": float(low),
        "high": float(high),
        "n": int(array.size),
    }


def strictly_positive(summary: dict) -> bool:
    return summary["mean"] > 0.0 and summary["low"] > 0.0
