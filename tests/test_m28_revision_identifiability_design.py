"""Design lock for the M28 identifiability gate.

The lock checks the frozen inventory and the refusal to score. It does not
read M26 or M27 outcome rows and it does not load Qwen.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.cognitive_self_model.m23.stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED
from src.cognitive_self_model.m28_identifiability import (
    CATALOG_SHA256,
    ELIGIBLE,
    GATE_STATUS,
    GateBlocked,
    HOLDOUT_N,
    INVENTORY,
    PRIMARY_CANDIDATE,
    PROTOCOL_VERSION,
    SHUFFLE_SEED,
    UPDATE_N,
    VALIDATION_N,
    absolute_error_reduction,
    catalog,
    classify,
    eligible_candidates,
    execution_gate,
    score_holdout,
    shuffle_update_assignment,
)
from src.cognitive_self_model.mechanism_belief import MODEL_ID, MODEL_REVISION
from src.cognitive_self_model.mechanism_response import DIRECTION_SEED, DIRECTION_SHA256
from src.cognitive_self_model.residual_update import FAMILY

PROTOCOL_PATH = Path("reports/M28_REVISION_IDENTIFIABILITY_PROTOCOL.md")
RATIONALE_PATH = Path("reports/M28_REVISION_IDENTIFIABILITY_RATIONALE.md")
GATE_SCRIPT = Path("scripts/run_m28_revision_identifiability_gate.py")


def test_gate_status_is_no_suitable_pre_evidence() -> None:
    assert execution_gate() == "NO_SUITABLE_PRE_EVIDENCE"
    assert eligible_candidates() == ()
    assert PRIMARY_CANDIDATE is None
    assert CATALOG_SHA256 is None
    assert UPDATE_N is None and VALIDATION_N is None and HOLDOUT_N is None


def test_stored_fields_are_ineligible_and_invented_names_are_rejected() -> None:
    assert classify("g") == "MECHANISM_OUTPUT"
    assert classify("prediction_before") == "IDENTICAL_TO_MECHANISM_OUTPUT"
    assert classify("family") == "ALREADY_SCORED_ON_M26_HOLDOUT"
    assert classify("baseline_output") == "ALREADY_SCORED_ON_M26_HOLDOUT"
    assert classify("layer") == "CELL_MEAN_BASELINE"
    assert classify("observed_effect") == "FORBIDDEN_OUTCOME"
    assert classify("residual") == "FORBIDDEN_OUTCOME"
    assert classify("revised_gain") == "FORBIDDEN_POST_EVIDENCE"
    assert classify("pre_dot") == "ABSENT_FROM_SEALED_SCHEMA"
    assert classify("margin_gradient") == "ABSENT_FROM_SEALED_SCHEMA"
    assert classify("prompt_length") == "ABSENT_FROM_SEALED_SCHEMA"
    assert classify("attention_entropy") == "INVENTED_MEASUREMENT"
    assert all(item.eligibility != ELIGIBLE for item in INVENTORY)
    assert len({item.name for item in INVENTORY}) == len(INVENTORY)


def test_scoring_paths_refuse_before_reading_rows() -> None:
    with pytest.raises(GateBlocked, match="NO_SUITABLE_PRE_EVIDENCE"):
        catalog()
    with pytest.raises(GateBlocked, match="NO_SUITABLE_PRE_EVIDENCE"):
        score_holdout([{"observed_effect": 1.0, "g": 1.0, "partition": "holdout"}])
    with pytest.raises(GateBlocked, match="NO_SUITABLE_PRE_EVIDENCE"):
        shuffle_update_assignment([{"g": 1.0}], seed=SHUFFLE_SEED)
    for function in (execution_gate, catalog, score_holdout, shuffle_update_assignment):
        assert "open(" not in inspect.getsource(function)


def test_metric_definition_on_synthetic_scalars() -> None:
    assert absolute_error_reduction(1.0, 0.0, 1.0) == pytest.approx(1.0)
    assert absolute_error_reduction(1.0, 0.0, 0.0) == pytest.approx(0.0)
    assert absolute_error_reduction(1.0, 0.0, -1.0) == pytest.approx(-1.0)
    assert SHUFFLE_SEED == 28001
    assert BOOTSTRAP_DRAWS == 5000
    assert BOOTSTRAP_SEED == 23001


def test_protocol_freezes_the_gate_and_the_existing_apparatus() -> None:
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    rationale = RATIONALE_PATH.read_text(encoding="utf-8")
    assert "READY_FOR_EXECUTION" not in protocol
    assert GATE_STATUS in protocol
    assert GATE_STATUS in rationale
    assert "PRIMARY_CANDIDATE: NONE" in protocol
    assert "CATALOG_SHA256: NONE" in protocol
    assert "UPDATE_N: NONE" in protocol
    assert "VALIDATION_N: NONE" in protocol
    assert "HOLDOUT_N: NONE" in protocol
    assert "SHUFFLE_SEED: 28001" in protocol
    assert "BOOTSTRAP_DRAWS: 5000" in protocol
    assert "BOOTSTRAP_SEED: 23001" in protocol
    assert PROTOCOL_VERSION in protocol
    assert MODEL_ID in protocol
    assert MODEL_REVISION in protocol
    assert DIRECTION_SHA256 in protocol
    assert str(DIRECTION_SEED) in protocol
    assert "alpha | `+1`" in protocol
    for cell in FAMILY:
        assert cell in protocol
    for item in INVENTORY:
        assert f"`{item.name}`" in protocol
        assert item.eligibility in protocol
    for label in (
        "M28_IDENTIFIABILITY_SUPPORTED",
        "M28_IDENTIFIABILITY_NOT_SUPPORTED",
        "M28_INCONCLUSIVE",
        "CI_POSITIVE",
        "CI_NEGATIVE",
        "prediction = g",
        "residual = observed_effect - g",
    ):
        assert label in protocol
    assert "The capital of Chile is" not in protocol
    assert "The capital of France is" not in protocol


def test_gate_script_prints_the_status_and_does_not_name_a_model_loader() -> None:
    source = GATE_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported == {"__future__", "pathlib", "sys", "src.cognitive_self_model.m28_identifiability"}
    forbidden = ("load_model", "HookedTransformer", "m27_raw", "EPISODES.csv", "torch")
    assert all(token not in source for token in forbidden)
    assert execution_gate() == GATE_STATUS
