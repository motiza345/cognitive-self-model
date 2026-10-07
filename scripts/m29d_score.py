"""Frozen M29-D scoring. This module does not load Qwen.

The comparison is the one locked in docs/M29_D_PROTOCOL.md:

    residual = observed_effect - g
    Delta = MAE(B1) - MAE(M29)

B0 is the UPDATE residual mean. B1 is the UPDATE residual mean of the
same mechanism cell. M29 predicts the residual with the frozen
pre-outcome kappa. HOLDOUT is the only partition that enters the verdict.

The protocol requires a permutation seed and a paired 95% bootstrap,
and it does not record a separate integer. Both use the inherited
paired-bootstrap generator in src.cognitive_self_model.m23.stats:
5000 draws, seed 23001, percentiles 2.5 and 97.5. B2 shuffles kappa
inside the holdout catalog only, in (intervention_id, prompt_id) order.
"""

from __future__ import annotations

import random
from typing import Any

from src.cognitive_self_model.m23.stats import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    ci_class,
    paired_mean_ci,
)

FAMILY = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")
EXPECTED_PREDICTION_SHA256 = "9ff6a11e6ad28128d9446db66e66f67057a4ae7ba6a4cfbf886297af6aeda33a"
FORBIDDEN_PREDICTION_KEYS = (
    "observed_effect",
    "intervened_output",
    "baseline_output",
    "residual",
    "outcome",
    "actual_outcome",
)
PERMUTATION_SEED = BOOTSTRAP_SEED
PERMUTATION_PROCEDURE = "shuffle_kappa_within_holdout_catalog"


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["intervention_id"]), str(row["prompt_id"]))


def _residual(row: dict[str, Any]) -> float:
    observed = float(row["observed_effect"])
    predicted = float(row["g"])
    if observed != observed or predicted != predicted:
        raise ValueError("non-finite residual input")
    return observed - predicted


def assert_frozen_predictions(rows: list[dict[str, Any]], *, digest: str) -> None:
    """Reject a prediction document that is not the frozen 108-row artifact."""
    if digest != EXPECTED_PREDICTION_SHA256:
        raise ValueError(f"prediction sha256 mismatch: {digest}")
    if len(rows) != 108:
        raise ValueError(f"expected 108 prediction rows, got {len(rows)}")
    counts = {name: 0 for name in ("update", "validation", "holdout")}
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"prediction row {index} is not an object")
        bad = set(FORBIDDEN_PREDICTION_KEYS).intersection(row)
        if bad:
            raise ValueError(f"prediction row {index} contains {sorted(bad)}")
        if row.get("outcome_present") is not False:
            raise ValueError("prediction row claims an outcome is present")
        intervention_id = str(row["intervention_id"])
        if intervention_id not in FAMILY:
            raise ValueError(f"unexpected mechanism {intervention_id}")
        partition = str(row["partition"])
        if partition not in counts:
            raise ValueError(f"unexpected partition {partition}")
        if float(row["alpha"]) != 1.0:
            raise ValueError("alpha is not the frozen value 1.0")
        if float(row["r_hat_m29"]) != float(row["kappa"]):
            raise ValueError("r_hat_m29 is not the stored kappa")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate prediction key {key}")
        seen.add(key)
        counts[partition] += 1
    if counts != {"update": 36, "validation": 36, "holdout": 36}:
        raise ValueError(f"partition counts are not 36/36/36: {counts}")


def fit_baselines(update_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """B0 and B1 from UPDATE residuals only."""
    if not update_rows:
        raise ValueError("baselines require update rows")
    grouped: dict[str, list[float]] = {cell: [] for cell in FAMILY}
    for row in update_rows:
        if row.get("partition") != "update":
            raise ValueError("baselines accept the update partition only")
        intervention_id = str(row["intervention_id"])
        if intervention_id not in grouped:
            raise ValueError("baseline row is outside the frozen family")
        grouped[intervention_id].append(_residual(row))
    if any(len(values) == 0 for values in grouped.values()):
        raise ValueError("each cell needs at least one update residual")
    cell_mean = {cell: sum(values) / len(values) for cell, values in grouped.items()}
    pooled = [value for cell in FAMILY for value in grouped[cell]]
    return {
        "b0": sum(pooled) / len(pooled),
        "b1": cell_mean,
        "n": {cell: len(grouped[cell]) for cell in FAMILY},
    }


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))


def permuted_kappa_map(rows: list[dict[str, Any]], *, seed: int = PERMUTATION_SEED) -> dict[tuple[str, str], float]:
    """Shuffle kappa inside this catalog. One stream, inherited seed 23001."""
    ordered = _ordered(rows)
    keys = [_key(row) for row in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("permutation catalog has a duplicate key")
    values = [float(row["kappa"]) for row in ordered]
    random.Random(seed).shuffle(values)
    return dict(zip(keys, values))


def evaluate_rows(rows: list[dict[str, Any]], baselines: dict[str, Any]) -> dict[str, Any]:
    """MAE of B0, B1, and M29 on one already-selected partition."""
    if not rows:
        raise ValueError("evaluation requires rows")
    ordered = _ordered(rows)
    b0_errors: list[float] = []
    b1_errors: list[float] = []
    m29_errors: list[float] = []
    paired: list[float] = []
    detailed: list[dict[str, Any]] = []
    b0 = float(baselines["b0"])
    for row in ordered:
        residual = _residual(row)
        cell = str(row["intervention_id"])
        b1 = float(baselines["b1"][cell])
        kappa = float(row["kappa"])
        b0_error = abs(residual - b0)
        b1_error = abs(residual - b1)
        m29_error = abs(residual - kappa)
        difference = b1_error - m29_error
        b0_errors.append(b0_error)
        b1_errors.append(b1_error)
        m29_errors.append(m29_error)
        paired.append(difference)
        detailed.append(
            {
                "intervention_id": cell,
                "prompt_id": str(row["prompt_id"]),
                "partition": row.get("partition"),
                "g": float(row["g"]),
                "kappa": kappa,
                "observed_effect": float(row["observed_effect"]),
                "residual": residual,
                "b0": b0,
                "b1": b1,
                "abs_error_b0": b0_error,
                "abs_error_b1": b1_error,
                "abs_error_m29": m29_error,
                "paired_difference": difference,
            }
        )
    mae_b0 = sum(b0_errors) / len(b0_errors)
    mae_b1 = sum(b1_errors) / len(b1_errors)
    mae_m29 = sum(m29_errors) / len(m29_errors)
    delta = mae_b1 - mae_m29
    interval = paired_mean_ci(paired)
    if abs(float(interval["mean"]) - delta) > 1e-12:
        raise RuntimeError("paired bootstrap mean does not equal Delta")
    return {
        "rows": len(ordered),
        "mae_b0": mae_b0,
        "mae_b1": mae_b1,
        "mae_m29": mae_m29,
        "delta": delta,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "mean": float(interval["mean"]),
            "low": float(interval["low"]),
            "high": float(interval["high"]),
            "class": str(interval["class"]),
        },
        "paired_rows": detailed,
    }


def verdict_from_criteria(
    *,
    delta: float,
    ci_low: float,
    ci_high: float,
    b2_class: str,
    leakage_ok: bool,
    same_rows: bool,
) -> str:
    """Map the preregistered holdout checks to one verdict.

    SUPPORTED requires Delta > 0, a paired 95% interval whose lower bound
    is above 0, the same holdout rows for both arms, a permutation control
    that is not itself CI_POSITIVE, and a passing leakage audit.
    A positive point estimate whose interval includes 0 is INCONCLUSIVE.
    """
    if not leakage_ok or not same_rows:
        return "INCONCLUSIVE"
    if not (ci_low <= ci_high):
        raise ValueError("bootstrap interval is reversed")
    primary = delta > 0.0 and ci_low > 0.0
    if primary and b2_class != "CI_POSITIVE":
        return "SUPPORTED"
    if delta > 0.0 and ci_low <= 0.0:
        return "INCONCLUSIVE"
    return "NOT_SUPPORTED"


def score_m29(update_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]], *, leakage_ok: bool) -> dict[str, Any]:
    """Score HOLDOUT. Validation rows must not be passed here."""
    if any(row.get("partition") != "holdout" for row in holdout_rows):
        raise ValueError("primary score accepts the holdout partition only")
    baselines = fit_baselines(update_rows)
    primary = evaluate_rows(holdout_rows, baselines)
    perm = permuted_kappa_map(holdout_rows)
    permuted_rows = []
    for row in holdout_rows:
        copied = dict(row)
        copied["kappa"] = perm[_key(row)]
        permuted_rows.append(copied)
    control = evaluate_rows(permuted_rows, baselines)
    update_keys = {_key(row) for row in update_rows}
    holdout_keys = {_key(row) for row in holdout_rows}
    same_rows = holdout_keys.isdisjoint(update_keys) and len(holdout_keys) == len(holdout_rows)
    b2_class = str(control["bootstrap"]["class"])
    if b2_class != ci_class(float(control["bootstrap"]["low"]), float(control["bootstrap"]["high"])):
        raise RuntimeError("B2 class does not match its interval")
    verdict = verdict_from_criteria(
        delta=float(primary["delta"]),
        ci_low=float(primary["bootstrap"]["low"]),
        ci_high=float(primary["bootstrap"]["high"]),
        b2_class=b2_class,
        leakage_ok=leakage_ok,
        same_rows=same_rows,
    )
    return {
        "verdict": verdict,
        "baselines": baselines,
        "holdout": primary,
        "b2": {
            "procedure": PERMUTATION_PROCEDURE,
            "seed": PERMUTATION_SEED,
            "class": b2_class,
            "mae": control["mae_m29"],
            "delta": control["delta"],
            "low": control["bootstrap"]["low"],
            "high": control["bootstrap"]["high"],
            "kappa_by_key": {f"{cell}|{prompt_id}": value for (cell, prompt_id), value in sorted(perm.items())},
        },
        "same_rows": same_rows,
        "leakage_ok": leakage_ok,
    }
