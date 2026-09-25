"""Paired-effect summaries. These functions do not choose an intervention."""

from __future__ import annotations

from typing import Any

import numpy as np


def effect_floor(max_null_abs: float, floor_abs: float, floor_mult: float) -> float:
    if max_null_abs < 0.0:
        raise ValueError("Null absolute difference cannot be negative.")
    return float(max(float(floor_abs), float(floor_mult) * float(max_null_abs)))


def summarize_deltas(deltas: list[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "mean_abs": None,
        }
    if not np.all(np.isfinite(values)):
        raise ValueError("Deltas must be finite.")
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "mean_abs": float(np.mean(np.abs(values))),
    }


def bootstrap_mean_interval(
    deltas: list[float],
    draws: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    summary = summarize_deltas(deltas)
    if values.size == 0:
        summary.update({"low": None, "high": None, "draws": int(draws), "seed": int(seed)})
        return summary
    generator = np.random.default_rng(int(seed))
    means = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        sample = generator.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(sample))
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    summary.update(
        {
            "low": float(low),
            "high": float(high),
            "draws": int(draws),
            "seed": int(seed),
            "interval_method": "percentile bootstrap of the paired-delta mean",
        }
    )
    return summary


def fraction_beyond_floor(deltas: list[float], floor: float) -> float:
    if len(deltas) == 0:
        return 0.0
    values = np.asarray(deltas, dtype=np.float64)
    return float(np.mean(np.abs(values) > float(floor)))


def by_regime(rows: list[dict[str, Any]], value_key: str = "actual_delta") -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["regime_id"]), []).append(float(row[value_key]))
    return {regime: summarize_deltas(values) for regime, values in sorted(grouped.items())}


def spearman_alpha_response(alpha_to_mean: list[tuple[float, float]]) -> dict[str, Any]:
    if len(alpha_to_mean) < 3:
        return {"spearman": None, "monotonic": None, "n_alphas": len(alpha_to_mean)}
    ordered = sorted(alpha_to_mean, key=lambda item: item[0])
    alphas = np.asarray([item[0] for item in ordered], dtype=np.float64)
    means = np.asarray([item[1] for item in ordered], dtype=np.float64)
    differences = np.diff(means)
    monotonic = bool(np.all(differences >= -1e-12) or np.all(differences <= 1e-12))
    if float(np.std(alphas)) < 1e-12 or float(np.std(means)) < 1e-12:
        coefficient = None
    else:
        from scipy.stats import spearmanr

        coefficient = spearmanr(alphas, means).statistic
        coefficient = None if coefficient is None else float(coefficient)
    return {
        "spearman": coefficient,
        "monotonic": monotonic,
        "n_alphas": int(len(ordered)),
        "alpha_to_mean": [{"alpha": float(alpha), "mean_delta": float(mean)} for alpha, mean in ordered],
    }


def classify_directionality(
    positive_deltas: list[float],
    negative_deltas: list[float],
    floor: float,
    clear_fraction: float,
    partial_fraction: float,
) -> dict[str, Any]:
    if len(positive_deltas) != len(negative_deltas) or len(positive_deltas) == 0:
        raise ValueError("Directionality requires paired non-empty delta lists.")
    positive = np.asarray(positive_deltas, dtype=np.float64)
    negative = np.asarray(negative_deltas, dtype=np.float64)
    mean_positive = float(np.mean(positive))
    mean_negative = float(np.mean(negative))
    comparable = (np.abs(positive) > floor) & (np.abs(negative) > floor)
    if int(np.sum(comparable)) == 0:
        opposite_fraction = None
    else:
        opposite_fraction = float(np.mean(np.sign(positive[comparable]) != np.sign(negative[comparable])))
    means_identifiable = abs(mean_positive) > floor and abs(mean_negative) > floor
    means_opposite = means_identifiable and (np.sign(mean_positive) != np.sign(mean_negative))
    if not means_identifiable:
        label = "UNIDENTIFIABLE"
    elif means_opposite and opposite_fraction is not None and opposite_fraction >= clear_fraction:
        label = "CLEAR_DIRECTIONAL"
    elif means_opposite and opposite_fraction is not None and opposite_fraction >= partial_fraction:
        label = "PARTIAL_DIRECTIONAL"
    else:
        label = "NON_DIRECTIONAL"
    return {
        "label": label,
        "mean_delta_positive_alpha": mean_positive,
        "mean_delta_negative_alpha": mean_negative,
        "opposite_sign_fraction": opposite_fraction,
        "n": int(len(positive)),
        "floor": float(floor),
        "gate": False,
    }


def wilcoxon_versus_zero(deltas: list[float]) -> dict[str, Any]:
    """Descriptive paired test. Not a validation gate."""
    values = np.asarray(deltas, dtype=np.float64)
    result: dict[str, Any] = {
        "test": "wilcoxon signed-rank versus zero, two-sided",
        "gate": False,
        "statistic": None,
        "pvalue": None,
        "n": int(values.size),
    }
    if values.size < 1 or np.allclose(values, 0.0):
        result["note"] = "undefined because the sample is empty or all zeros"
        return result
    from scipy.stats import wilcoxon

    try:
        statistic, pvalue = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
    except ValueError as exc:
        result["note"] = str(exc)
        return result
    result["statistic"] = float(statistic)
    result["pvalue"] = float(pvalue)
    return result
