"""Frozen M24 protocol checks. These tests do not load Qwen weights."""

from __future__ import annotations

import inspect
import random
from dataclasses import asdict
from pathlib import Path

import pytest

from scripts.feasibility_gate_catalog import catalog as f_catalog
from scripts.m24_consumption_catalog import CATALOG_SHA256, PARTITIONS, catalog, catalog_sha256
from scripts.m24_consumption_protocol import (
    BANNED_PREDICTION_KEYS,
    DECISION_THRESHOLD,
    EXECUTABLE_PARTITIONS,
    FAMILY,
    SHUFFLE_SEED,
    attach_shuffled_predictions,
    baseline_prediction,
    consumption_verdict,
    decision_counts,
    join_rows,
    leakage_ok,
    positive_decision,
    prediction_file_clean,
    prompt_level_differences,
    rows_lack_outcomes,
    run_prediction_before_outcome,
    sign_does_not_lose,
)
from scripts.run_m24_consumption_audit import _prompts
from scripts.validate_m23_g_protocol import catalog as g_catalog
from scripts.validate_post_m23_single_measurement import catalog as s_catalog
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts
from src.cognitive_self_model.mechanism_response import (
    MechanismResponseModel,
    PreInterventionState,
    frozen_mechanism_response_models,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports" / "M24_CONSUMPTION_PROTOCOL.md"


def test_catalog_is_frozen_and_disjoint() -> None:
    rows = catalog()
    assert catalog_sha256() == CATALOG_SHA256
    assert CATALOG_SHA256 in PROTOCOL.read_text(encoding="utf-8")
    assert len(rows) == 36
    assert len({row["text"] for row in rows}) == 36
    counts: dict[tuple[str, str], int] = {}
    old_texts = {record.text for record in frozen_prompts()}
    old_ids = {record.prompt_id for record in frozen_prompts()}
    for previous in (g_catalog(), s_catalog(), f_catalog()):
        old_texts.update(row["text"] for row in previous)
        old_ids.update(row["prompt_id"] for row in previous)
    protocol = PROTOCOL.read_text(encoding="utf-8")
    for row in rows:
        assert row["partition"] in PARTITIONS
        assert row["text"] in protocol
        assert row["prompt_id"] in protocol
        assert row["text"] not in old_texts
        assert row["prompt_id"] not in old_ids
        counts[(row["partition"], row["family"])] = counts.get((row["partition"], row["family"]), 0) + 1
    assert set(counts.values()) == {4}
    assert EXECUTABLE_PARTITIONS == ("validation", "evaluation")
    assert "train" not in EXECUTABLE_PARTITIONS
    with pytest.raises(RuntimeError, match="not forwarded"):
        _prompts("train")


def test_predictions_are_written_before_outcomes_and_do_not_mutate_the_model() -> None:
    mechanisms = frozen_mechanism_response_models()
    before = [asdict(model) for model in mechanisms]
    calls: list[str] = []

    def prepare(name: str) -> list[dict[str, object]]:
        calls.append(f"prepare:{name}")
        rows = []
        for model in mechanisms:
            state = PreInterventionState(margin_gradient=model.direction)
            predicted = model.predict(state)
            rows.append(
                {
                    "intervention_id": model.intervention_id,
                    "prompt_id": f"{name}-01",
                    "candidate_prediction": predicted,
                    "baseline_prediction": baseline_prediction(model),
                    "g": predicted,
                    "g_orth": 0.0,
                    "partition": name,
                }
            )
        assert rows_lack_outcomes(rows)
        return rows

    def finish(name: str) -> str:
        calls.append(f"finish:{name}")
        return name

    prepared, finished = run_prediction_before_outcome(EXECUTABLE_PARTITIONS, prepare, finish)
    assert calls == [
        "prepare:validation",
        "prepare:evaluation",
        "finish:validation",
        "finish:evaluation",
    ]
    assert set(prepared) == {"validation", "evaluation"}
    assert finished == {"validation": "validation", "evaluation": "evaluation"}
    assert [asdict(model) for model in mechanisms] == before
    parameters = inspect.signature(MechanismResponseModel.predict).parameters
    assert list(parameters) == ["self", "pre_intervention_state"]
    assert list(inspect.signature(baseline_prediction).parameters) == ["model"]


def test_baseline_ignores_outcomes_and_shuffle_is_pinned() -> None:
    model = frozen_mechanism_response_models()[0]
    assert baseline_prediction(model) == pytest.approx(model.training_baseline_mean)
    assert baseline_prediction(model) == pytest.approx(0.02711375554402669)
    rows = []
    for cell_index, intervention_id in enumerate(FAMILY):
        for prompt_index in range(12):
            rows.append(
                {
                    "intervention_id": intervention_id,
                    "prompt_id": f"p-{prompt_index:02d}",
                    "candidate_prediction": float(prompt_index + 100 * cell_index),
                }
            )
    first = attach_shuffled_predictions(rows)
    second = attach_shuffled_predictions(rows)
    assert [(row["intervention_id"], row["prompt_id"], row["shuffled_prediction"]) for row in first] == [
        (row["intervention_id"], row["prompt_id"], row["shuffled_prediction"]) for row in second
    ]
    pinned = {
        "M22.1-D1-L0": [4, 3, 8, 5, 11, 6, 1, 9, 10, 0, 7, 2],
        "M22.1-D1-L8": [4, 11, 6, 10, 8, 0, 1, 3, 2, 5, 7, 9],
        "M22.1-D1-L15": [7, 3, 0, 1, 10, 4, 11, 6, 8, 9, 2, 5],
    }
    for cell_index, intervention_id in enumerate(FAMILY):
        found = [
            int(row["shuffled_prediction"]) - 100 * cell_index
            for row in first
            if row["intervention_id"] == intervention_id
        ]
        assert found == pinned[intervention_id]
    assert SHUFFLE_SEED == 24001
    rng = random.Random(SHUFFLE_SEED)
    probe = list(range(12))
    rng.shuffle(probe)
    assert probe == pinned["M22.1-D1-L0"]


def test_prompt_aggregation_and_decision_threshold() -> None:
    rows = []
    for prompt_id, diffs in (("b", (3.0, 0.0, 0.0)), ("a", (1.0, 1.0, 1.0))):
        for intervention_id, diff in zip(FAMILY, diffs):
            rows.append({"prompt_id": prompt_id, "paired_difference": diff, "intervention_id": intervention_id})
    assert prompt_level_differences(rows, "paired_difference", expected_prompts=2) == [1.0, 1.0]
    assert DECISION_THRESHOLD == 0.0
    assert positive_decision(0.1) is True
    assert positive_decision(0.0) is False
    assert positive_decision(-1.0) is False
    model_counts = decision_counts([1.0, 0.0, -1.0], [1.0, -1.0, 0.0])
    baseline_counts = decision_counts([1.0, 1.0, 1.0], [1.0, -1.0, 0.0])
    assert model_counts["tp"] == 1
    assert model_counts["tn"] == 2
    assert model_counts["fp"] == 0
    assert model_counts["fn"] == 0
    assert sign_does_not_lose(model_counts, baseline_counts) is True
    assert sign_does_not_lose(baseline_counts, model_counts) is False


def test_verdict_map_is_frozen() -> None:
    assert consumption_verdict("CI_POSITIVE", True, "CI_NEGATIVE", True) == "CONSUMPTION_SUPPORTED"
    assert consumption_verdict("CI_POSITIVE", True, "CI_INCLUDES_ZERO", True) == "CONSUMPTION_SUPPORTED"
    assert consumption_verdict("CI_POSITIVE", False, "CI_NEGATIVE", True) == "INCONCLUSIVE"
    assert consumption_verdict("CI_POSITIVE", True, "CI_POSITIVE", True) == "INCONCLUSIVE"
    assert consumption_verdict("CI_NEGATIVE", True, "CI_NEGATIVE", True) == "CONSUMPTION_NOT_SUPPORTED"
    assert consumption_verdict("CI_NEGATIVE", False, "CI_POSITIVE", True) == "CONSUMPTION_NOT_SUPPORTED"
    assert consumption_verdict("CI_NEGATIVE", True, "CI_NEGATIVE", False) == "INCONCLUSIVE"
    assert consumption_verdict("CI_INCLUDES_ZERO", True, "CI_NEGATIVE", True) == "INCONCLUSIVE"
    assert consumption_verdict("CI_POSITIVE", True, "CI_NEGATIVE", False) == "INCONCLUSIVE"
    parameters = set(inspect.signature(consumption_verdict).parameters)
    assert "orth" not in "".join(parameters)


def test_leakage_audit_rejects_late_predictions_and_outcome_keys() -> None:
    assert leakage_ok(
        prediction_before_outcome=True,
        rows_clean=True,
        model_unchanged=True,
        baseline_from_model=True,
    )
    assert not leakage_ok(
        prediction_before_outcome=False,
        rows_clean=True,
        model_unchanged=True,
        baseline_from_model=True,
    )
    clean = [{"candidate_prediction": 0.1, "baseline_prediction": 0.2}]
    dirty = [{"candidate_prediction": 0.1, "observed_effect": 0.3}]
    assert rows_lack_outcomes(clean)
    assert not rows_lack_outcomes(dirty)
    assert prediction_file_clean('{"candidate_prediction": 0.1}')
    for key in BANNED_PREDICTION_KEYS:
        assert not prediction_file_clean('{"' + key + '": 1}')
    joined = join_rows(
        [
            {
                "intervention_id": "M22.1-D1-L0",
                "prompt_id": "p",
                "candidate_prediction": 1.0,
                "baseline_prediction": 0.0,
                "shuffled_prediction": 0.0,
                "g_orth": 0.0,
                "g": 1.0,
            }
        ],
        [
            {
                "intervention_id": "M22.1-D1-L0",
                "prompt_id": "p",
                "baseline_output": 0.2,
                "intervened_output": 1.2,
                "observed_effect": 1.0,
            }
        ],
    )
    assert joined[0]["paired_difference"] == pytest.approx(1.0)
    assert "observed_effect" not in {
        "intervention_id": "M22.1-D1-L0",
        "prompt_id": "p",
        "candidate_prediction": 1.0,
        "baseline_prediction": 0.0,
        "shuffled_prediction": 0.0,
        "g_orth": 0.0,
        "g": 1.0,
    }


def test_runner_stays_inside_the_consumption_boundary() -> None:
    source = (ROOT / "scripts" / "run_m24_consumption_audit.py").read_text(encoding="utf-8")
    protocol = (ROOT / "scripts" / "m24_consumption_protocol.py").read_text(encoding="utf-8")
    for banned in ("SelfModelBelief", "update_belief", "pre_dot", "M22.1-D1-L23"):
        assert banned not in source
        assert banned not in protocol
    assert "run_prediction_before_outcome" in source
    assert "MechanismResponseModel" not in source or "frozen_mechanism_response_models" in source
    assert FAMILY == ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")
