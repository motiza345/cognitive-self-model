"""M26 update algebra and catalog freeze. These tests do not read outcome files."""

from __future__ import annotations

import inspect
import math
import random

import pytest

from scripts.feasibility_gate_catalog import catalog as f_catalog
from scripts.m24_consumption_catalog import catalog as c_catalog
from scripts.m26_update_catalog import CATALOG_SHA256, PARTITIONS, catalog, catalog_sha256
from scripts.validate_m23_g_protocol import catalog as g_catalog
from scripts.validate_post_m23_single_measurement import catalog as s_catalog
from src.cognitive_self_model.m23.belief import CRITICAL_Z
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts
from src.cognitive_self_model.residual_update import (
    FAMILY,
    FROZEN_SIGMA,
    PRIOR_COUNT,
    PRIOR_MEAN,
    SHUFFLE_SEED,
    cell_deltas,
    global_delta,
    posterior_delta,
    posterior_sd,
    prediction_after,
    prediction_before,
    predictive_sd,
    shuffled_residuals,
    update_residuals,
    update_verdict,
)

ROOT_PROTOCOL = "reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md"


def test_catalog_is_disjoint_and_balanced() -> None:
    from pathlib import Path

    rows = catalog()
    assert catalog_sha256() == CATALOG_SHA256
    protocol = Path(ROOT_PROTOCOL).read_text(encoding="utf-8")
    assert CATALOG_SHA256 in protocol
    old_texts = {record.text for record in frozen_prompts()}
    old_ids = {record.prompt_id for record in frozen_prompts()}
    for previous in (g_catalog(), s_catalog(), f_catalog(), c_catalog()):
        old_texts.update(row["text"] for row in previous)
        old_ids.update(row["prompt_id"] for row in previous)
    counts: dict[tuple[str, str], int] = {}
    update_ids = set()
    holdout_ids = set()
    for row in rows:
        assert row["text"] in protocol
        assert row["prompt_id"] in protocol
        assert row["text"] not in old_texts
        assert row["prompt_id"] not in old_ids
        counts[(row["partition"], row["family"])] = counts.get((row["partition"], row["family"]), 0) + 1
        if row["partition"] == "update":
            update_ids.add(row["prompt_id"])
        else:
            holdout_ids.add(row["prompt_id"])
    assert set(counts) == {(partition, family) for partition in PARTITIONS for family in ("completion", "syntax", "instruction")}
    assert set(counts.values()) == {4}
    assert update_ids.isdisjoint(holdout_ids)
    assert len(rows) == 24


def test_posterior_matches_the_frozen_equation() -> None:
    assert PRIOR_MEAN == 0.0
    assert PRIOR_COUNT == 1.0
    assert posterior_delta([]) == 0.0
    assert posterior_delta([1.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert posterior_delta([1.0, 3.0]) == pytest.approx((0.0 + 2.0 * 2.0) / 3.0)
    sigma = 0.2
    assert posterior_sd(sigma, 3) == pytest.approx(sigma / 2.0)
    assert predictive_sd(sigma, 3) == pytest.approx(math.sqrt((sigma / 2.0) ** 2 + sigma ** 2))
    assert prediction_before(1.5) == pytest.approx(1.5)
    assert prediction_after(1.5, 0.75) == pytest.approx(2.25)
    assert list(inspect.signature(prediction_after).parameters) == ["g", "delta"]
    assert abs(1.0 - 1.0) <= CRITICAL_Z * predictive_sd(1.0, 0)


def test_cell_correction_is_not_the_global_mean() -> None:
    residuals = {
        "M22.1-D1-L0": [1.0, 1.0],
        "M22.1-D1-L8": [-1.0, -1.0],
        "M22.1-D1-L15": [0.0, 0.0],
    }
    by_cell = cell_deltas(residuals)
    shared = global_delta(residuals)
    assert by_cell["M22.1-D1-L0"] == pytest.approx(2.0 / 3.0)
    assert by_cell["M22.1-D1-L8"] == pytest.approx(-2.0 / 3.0)
    assert by_cell["M22.1-D1-L15"] == pytest.approx(0.0)
    assert shared == pytest.approx(0.0)
    assert by_cell["M22.1-D1-L0"] != pytest.approx(shared)


def test_shuffle_is_pinned_and_holdout_is_rejected() -> None:
    rows = []
    value = 0
    for intervention_id in sorted(FAMILY):
        for prompt_index in (1, 2):
            rows.append(
                {
                    "intervention_id": intervention_id,
                    "prompt_id": f"p-{prompt_index}",
                    "partition": "update",
                    "g": 0.0,
                    "observed_effect": float(value),
                }
            )
            value += 1
    first = shuffled_residuals(rows)
    second = shuffled_residuals(rows)
    assert first == second
    flat = [item for intervention_id in sorted(FAMILY) for item in first[intervention_id]]
    assert flat == [1, 3, 2, 0, 4, 5]
    probe = list(range(6))
    random.Random(SHUFFLE_SEED).shuffle(probe)
    assert probe == [1, 3, 2, 0, 4, 5]
    with pytest.raises(ValueError, match="update partition"):
        update_residuals(
            [
                {
                    "intervention_id": "M22.1-D1-L0",
                    "partition": "holdout",
                    "g": 0.0,
                    "observed_effect": 1.0,
                }
            ]
        )


def test_verdict_map_is_frozen() -> None:
    assert update_verdict("CI_POSITIVE", "CI_POSITIVE", "CI_NEGATIVE", True) == "UPDATE_SUPPORTED"
    assert update_verdict("CI_POSITIVE", "CI_POSITIVE", "CI_INCLUDES_ZERO", True) == "UPDATE_SUPPORTED"
    assert update_verdict("CI_POSITIVE", "CI_INCLUDES_ZERO", "CI_NEGATIVE", True) == "INCONCLUSIVE"
    assert update_verdict("CI_POSITIVE", "CI_POSITIVE", "CI_POSITIVE", True) == "INCONCLUSIVE"
    assert update_verdict("CI_NEGATIVE", "CI_POSITIVE", "CI_NEGATIVE", True) == "UPDATE_NOT_SUPPORTED"
    assert update_verdict("CI_NEGATIVE", "CI_NEGATIVE", "CI_NEGATIVE", False) == "INCONCLUSIVE"
    assert update_verdict("CI_INCLUDES_ZERO", "CI_POSITIVE", "CI_NEGATIVE", True) == "INCONCLUSIVE"
    assert update_verdict("CI_POSITIVE", "CI_NEGATIVE", "CI_NEGATIVE", True) == "INCONCLUSIVE"
    assert set(FROZEN_SIGMA) == set(FAMILY)
    assert all(value > 0.0 for value in FROZEN_SIGMA.values())
