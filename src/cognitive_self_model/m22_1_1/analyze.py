"""Descriptive characterization of a frozen direction panel.

The rules below were written into the audit config before measurement.
They do not select a new direction and they do not change M22.1.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _rows(records: list[dict], **equals) -> list[dict]:
    chosen = []
    for row in records:
        if all(row[key] == value for key, value in equals.items()):
            chosen.append(row)
    return chosen


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def sign_consistency(values: list[float], floor: float) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    center = float(np.mean(array))
    if abs(center) <= float(floor):
        return None
    return float(np.mean(np.sign(array) == np.sign(center)))


def effect_profile(records: list[dict], config: dict) -> dict:
    rules = config["classification"]
    alpha = float(rules["alpha_for_effect_profile"])
    profiles = []
    for direction_id in sorted({row["direction_id"] for row in records}):
        by_alpha = {}
        for value in config["magnitude_grid"]:
            subset = _rows(records, direction_id=direction_id, alpha=float(value))
            by_alpha[str(float(value))] = _mean([row["actual_delta"] for row in subset])
        plus = _rows(records, direction_id=direction_id, alpha=alpha)
        minus = _rows(records, direction_id=direction_id, alpha=-alpha)
        plus_values = [row["actual_delta"] for row in plus]
        minus_values = [row["actual_delta"] for row in minus]
        plus_mean = _mean(plus_values)
        norms = {row["direction_norm"] for row in plus}
        norm = float(next(iter(norms))) if len(norms) == 1 else None
        regimes = {}
        for regime in ("completion", "instruction", "syntax"):
            regimes[regime] = _mean(
                [row["actual_delta"] for row in plus if row["regime"] == regime]
            )
        regime_values = [value for value in regimes.values() if value is not None]
        alphas = np.asarray(config["magnitude_grid"], dtype=np.float64)
        dose_values = [by_alpha[str(float(value))] for value in alphas]
        if any(value is None for value in dose_values):
            dose_score = None
        else:
            dose_array = np.asarray(dose_values, dtype=np.float64)
            if float(np.std(dose_array)) <= 1e-15:
                dose_score = None
            else:
                statistic = float(spearmanr(alphas, dose_array).statistic)
                dose_score = statistic if np.isfinite(statistic) else None
        symmetry = None
        if plus_mean is not None and minus_values:
            symmetry = float(plus_mean - (-float(np.mean(minus_values))))
        profiles.append(
            {
                "direction_id": direction_id,
                "role": plus[0]["role"] if plus else None,
                "mean_delta_plus1": plus_mean,
                "mean_delta_minus1": _mean(minus_values),
                "abs_mean_delta": None if plus_mean is None else abs(plus_mean),
                "std_delta": _std(plus_values),
                "effect_per_unit_norm": None if plus_mean is None or norm in (None, 0.0) else plus_mean / norm,
                "sign_consistency": sign_consistency(plus_values, float(rules["substantial_abs_floor"])),
                "directional_symmetry_plus_plus_minus": symmetry,
                "dose_response": by_alpha,
                "dose_response_score": dose_score,
                "regime_means": regimes,
                "regime_variance": None if len(regime_values) < 2 else float(np.var(regime_values)),
            }
        )
    return {"alpha": alpha, "directions": profiles}


def null_abs_max(records: list[dict]) -> float:
    zeros = [abs(float(row["actual_delta"])) for row in records if float(row["alpha"]) == 0.0]
    return float(max(zeros)) if zeros else 0.0


def _singular_energy(matrix: np.ndarray) -> dict:
    if matrix.size == 0 or float(np.sum(matrix ** 2)) <= 1e-18:
        return {"singular_values": [], "energy_fractions": [], "leading_energy": None}
    _, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    energy = singular ** 2
    total = float(np.sum(energy))
    fractions = [float(value / total) for value in energy]
    return {
        "singular_values": [float(value) for value in singular],
        "energy_fractions": fractions,
        "leading_energy": fractions[0],
    }


def functional_correlations(records: list[dict], alpha: float = 1.0) -> dict:
    ids = sorted({row["direction_id"] for row in records})
    prompts = sorted({row["prompt_id"] for row in records})
    columns = {}
    for direction_id in ids:
        values = []
        usable = True
        for prompt_id in prompts:
            matched = _rows(records, direction_id=direction_id, prompt_id=prompt_id, alpha=float(alpha))
            if len(matched) != 1:
                usable = False
                break
            values.append(float(matched[0]["actual_delta"]))
        columns[direction_id] = np.asarray(values, dtype=np.float64) if usable else None
    output = {}
    for left in ids:
        output[left] = {}
        for right in ids:
            a = columns[left]
            b = columns[right]
            if a is None or b is None or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
                output[left][right] = None
            else:
                output[left][right] = float(np.corrcoef(a, b)[0, 1])
    return output


def classify_split(records: list[dict], config: dict) -> dict:
    rules = config["classification"]
    profile = effect_profile(records, config)
    rows = profile["directions"]
    floor = max(
        float(rules["substantial_abs_floor"]),
        float(rules["substantial_floor_mult_of_null"]) * null_abs_max(records),
    )
    abs_means = np.asarray([row["abs_mean_delta"] for row in rows], dtype=np.float64)
    substantial = [row["direction_id"] for row, magnitude in zip(rows, abs_means) if magnitude > floor]
    positive = abs_means[abs_means > 0]
    max_over_min = None if positive.size == 0 or float(np.min(positive)) == 0.0 else float(np.max(abs_means) / np.min(positive))
    median = float(np.median(abs_means)) if abs_means.size else None
    max_over_median = None if median in (None, 0.0) else float(np.max(abs_means) / median)
    ids = [row["direction_id"] for row in rows]
    prompts = sorted({row["prompt_id"] for row in records if float(row["alpha"]) == 1.0})
    matrix = []
    for prompt_id in prompts:
        matrix.append([
            float(_rows(records, direction_id=direction_id, prompt_id=prompt_id, alpha=1.0)[0]["actual_delta"])
            for direction_id in ids
        ])
    energy = _singular_energy(np.asarray(matrix, dtype=np.float64) if matrix else np.zeros((0, 0)))
    regime_hits = []
    for row in rows:
        values = [row["regime_means"][name] for name in ("completion", "instruction", "syntax")]
        if any(value is None for value in values) or row["abs_mean_delta"] is None:
            continue
        spread = max(values) - min(values)
        sign_change = min(values) < 0.0 < max(values)
        if (
            row["abs_mean_delta"] > floor
            and sign_change
            and spread > float(rules["regime_range_over_abs_mean"]) * row["abs_mean_delta"]
        ):
            regime_hits.append(row["direction_id"])
    labels = []
    leading = energy["leading_energy"]
    if len(substantial) <= int(rules["low_dimensional_max_substantial"]) and leading is not None and leading >= float(rules["low_dimensional_min_leading_energy"]):
        labels.append("LOW_DIMENSIONAL_SENSITIVITY")
    if len(substantial) >= int(rules["broad_min_directions"]) and max_over_min is not None and max_over_min < float(rules["broad_max_over_min_ratio_exclusive"]):
        labels.append("BROAD_RESIDUAL_SENSITIVITY")
    if len(regime_hits) >= int(rules["regime_min_directions"]):
        labels.append("REGIME_DEPENDENT_SENSITIVITY")
    if len(substantial) >= int(rules["structured_min_directions"]) and max_over_median is not None and max_over_median >= float(rules["structured_min_max_over_median_ratio"]):
        labels.append("STRUCTURED_INTERVENTION_SPACE")
    label = labels[0] if len(labels) == 1 else "INSUFFICIENT_EVIDENCE"
    return {
        "label": label,
        "matching_labels": labels,
        "substantial_floor": floor,
        "substantial_direction_ids": substantial,
        "max_over_min_abs_mean": max_over_min,
        "max_over_median_abs_mean": max_over_median,
        "regime_sign_change_direction_ids": regime_hits,
        "svd_uncentered": energy,
        "profile": profile,
    }


def classify_panel(split_records: dict[str, list[dict]], config: dict) -> dict:
    per_split = {
        name: classify_split(rows, config)
        for name, rows in split_records.items()
    }
    validation = per_split["validation"]["label"]
    replication = per_split["replication"]["label"]
    if validation == replication:
        final = validation
    else:
        final = "INSUFFICIENT_EVIDENCE"
    return {"per_split": per_split, "label": final, "discovery_not_used": True}
