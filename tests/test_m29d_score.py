"""M29-D scoring uses UPDATE baselines and the frozen verdict map."""

from __future__ import annotations

import pytest

from scripts.m29d_score import (
    FAMILY,
    PERMUTATION_SEED,
    evaluate_rows,
    fit_baselines,
    permuted_kappa_map,
    score_m29,
    verdict_from_criteria,
)
from src.cognitive_self_model.m23.stats import BOOTSTRAP_SEED


def _row(partition: str, cell: str, prompt: str, observed: float, kappa: float, g: float = 0.0) -> dict:
    return {
        "partition": partition,
        "intervention_id": cell,
        "prompt_id": prompt,
        "g": g,
        "kappa": kappa,
        "observed_effect": observed,
        "alpha": 1.0,
    }


def test_verdict_map_matches_the_preregistered_cases() -> None:
    assert verdict_from_criteria(
        delta=1.0,
        ci_low=0.1,
        ci_high=2.0,
        b2_class="CI_INCLUDES_ZERO",
        leakage_ok=True,
        same_rows=True,
    ) == "SUPPORTED"
    assert verdict_from_criteria(
        delta=1.0,
        ci_low=-0.1,
        ci_high=2.0,
        b2_class="CI_INCLUDES_ZERO",
        leakage_ok=True,
        same_rows=True,
    ) == "INCONCLUSIVE"
    assert verdict_from_criteria(
        delta=1.0,
        ci_low=0.0,
        ci_high=2.0,
        b2_class="CI_NEGATIVE",
        leakage_ok=True,
        same_rows=True,
    ) == "INCONCLUSIVE"
    assert verdict_from_criteria(
        delta=-0.2,
        ci_low=-1.0,
        ci_high=-0.05,
        b2_class="CI_NEGATIVE",
        leakage_ok=True,
        same_rows=True,
    ) == "NOT_SUPPORTED"
    assert verdict_from_criteria(
        delta=1.0,
        ci_low=0.2,
        ci_high=2.0,
        b2_class="CI_POSITIVE",
        leakage_ok=True,
        same_rows=True,
    ) == "NOT_SUPPORTED"
    assert verdict_from_criteria(
        delta=1.0,
        ci_low=0.2,
        ci_high=2.0,
        b2_class="CI_INCLUDES_ZERO",
        leakage_ok=False,
        same_rows=True,
    ) == "INCONCLUSIVE"
    assert verdict_from_criteria(
        delta=0.0,
        ci_low=-0.1,
        ci_high=0.1,
        b2_class="CI_INCLUDES_ZERO",
        leakage_ok=True,
        same_rows=True,
    ) == "NOT_SUPPORTED"


def test_baselines_use_update_cell_means_only() -> None:
    rows = []
    values = {
        FAMILY[0]: (1.0, 3.0),
        FAMILY[1]: (0.0, 0.0),
        FAMILY[2]: (2.0, 4.0),
    }
    for cell, pair in values.items():
        for index, observed in enumerate(pair):
            rows.append(_row("update", cell, f"{cell}-{index}", observed, 0.0))
    fitted = fit_baselines(rows)
    assert fitted["b1"][FAMILY[0]] == pytest.approx(2.0)
    assert fitted["b1"][FAMILY[1]] == pytest.approx(0.0)
    assert fitted["b1"][FAMILY[2]] == pytest.approx(3.0)
    assert fitted["b0"] == pytest.approx((1.0 + 3.0 + 2.0 + 4.0) / 6.0)
    with pytest.raises(ValueError):
        fit_baselines([_row("holdout", FAMILY[0], "h", 100.0, 0.0)])


def test_holdout_does_not_change_the_update_baseline() -> None:
    update = [_row("update", cell, f"u-{cell}", 0.0, 0.0) for cell in FAMILY]
    holdout = [_row("holdout", cell, f"h-{cell}", 100.0, 100.0) for cell in FAMILY]
    fitted = fit_baselines(update)
    scored = evaluate_rows(holdout, fitted)
    assert fitted["b1"][FAMILY[0]] == pytest.approx(0.0)
    assert scored["mae_b1"] == pytest.approx(100.0)
    assert scored["mae_m29"] == pytest.approx(0.0)
    assert scored["delta"] == pytest.approx(100.0)


def test_constant_kappa_is_not_supported_when_permutation_matches() -> None:
    update = []
    holdout = []
    for cell in FAMILY:
        for index in range(4):
            update.append(_row("update", cell, f"u-{cell}-{index}", 0.0, 0.0))
            holdout.append(_row("holdout", cell, f"h-{cell}-{index}", 5.0, 5.0))
    result = score_m29(update, holdout, leakage_ok=True)
    assert result["holdout"]["mae_b1"] == pytest.approx(5.0)
    assert result["holdout"]["mae_m29"] == pytest.approx(0.0)
    assert result["holdout"]["delta"] == pytest.approx(5.0)
    assert float(result["holdout"]["bootstrap"]["low"]) > 0.0
    assert result["b2"]["class"] == "CI_POSITIVE"
    assert result["verdict"] == "NOT_SUPPORTED"
    assert PERMUTATION_SEED == BOOTSTRAP_SEED == 23001


def test_permutation_is_deterministic_and_stays_inside_the_catalog() -> None:
    rows = []
    for cell in FAMILY:
        for index, kappa in enumerate((1.0, 2.0, 3.0, 4.0)):
            rows.append(_row("holdout", cell, f"h-{index}", kappa, kappa))
    first = permuted_kappa_map(rows)
    second = permuted_kappa_map(rows)
    assert first == second
    assert set(first) == {(row["intervention_id"], row["prompt_id"]) for row in rows}
    assert sorted(first.values()) == sorted(float(row["kappa"]) for row in rows)
