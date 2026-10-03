"""MechanismResponseModel is the frozen rule prediction = g."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.cognitive_self_model.mechanism_response import (
    ALPHA,
    DIRECTION_SHA256,
    RESPONSE_RULE,
    ExperimentReference,
    MechanismResponseModel,
    PreInterventionState,
    directional_derivative,
    frozen_mechanism_response_models,
)
from src.cognitive_self_model.m23.m22_reuse import primary_direction, vector_sha256


def _models() -> tuple[MechanismResponseModel, ...]:
    return frozen_mechanism_response_models()


def _state(gradient: np.ndarray) -> PreInterventionState:
    return PreInterventionState(margin_gradient=tuple(float(value) for value in gradient))


def test_factory_returns_the_three_recorded_cells() -> None:
    models = _models()
    assert [model.intervention_id for model in models] == [
        "M22.1-D1-L0",
        "M22.1-D1-L8",
        "M22.1-D1-L15",
    ]
    assert [model.layer for model in models] == [0, 8, 15]
    assert [model.hook for model in models] == [
        "blocks.0.hook_resid_post",
        "blocks.8.hook_resid_post",
        "blocks.15.hook_resid_post",
    ]
    by_id = {model.intervention_id: model for model in models}
    assert by_id["M22.1-D1-L0"].training_baseline_mean == pytest.approx(0.02711375554402669)
    assert by_id["M22.1-D1-L8"].training_baseline_mean == pytest.approx(0.02379457155863444)
    assert by_id["M22.1-D1-L15"].training_baseline_mean == pytest.approx(-0.0065801143646240234)


def test_prediction_equals_the_directional_derivative() -> None:
    direction = primary_direction(896, 22101)
    gradient = np.linspace(-0.25, 0.5, 896, dtype=np.float64)
    expected = directional_derivative(gradient, direction)
    assert expected == pytest.approx(float(np.dot(gradient, direction)))
    state = _state(gradient)
    for model in _models():
        assert model.predict(state) == pytest.approx(expected)
        axis = int(model.layer)
        basis = np.zeros(896, dtype=np.float64)
        basis[axis] = 1.0
        assert model.predict(_state(basis)) == pytest.approx(float(direction[axis]))


def test_predict_needs_no_outcome() -> None:
    parameters = inspect.signature(MechanismResponseModel.predict).parameters
    assert list(parameters) == ["self", "pre_intervention_state"]
    state_fields = set(PreInterventionState.__dataclass_fields__)
    assert state_fields == {"margin_gradient"}
    source = inspect.getsource(MechanismResponseModel.predict)
    for banned in ("observed_effect", "outcome", "pre_dot", "slope", "fit(", "update"):
        assert banned not in source
    gradient = np.ones(896, dtype=np.float64)
    assert np.isfinite(_models()[0].predict(_state(gradient)))


def test_predict_does_not_change_stored_parameters() -> None:
    model = _models()[1]
    before = (
        model.training_baseline_mean,
        model.direction,
        model.alpha,
        model.response_rule,
        model.provenance,
        model.direction_sha256,
    )
    first = model.predict(_state(np.full(896, 0.1)))
    second = model.predict(_state(np.full(896, -0.3)))
    assert first != pytest.approx(second)
    assert first != pytest.approx(model.training_baseline_mean)
    after = (
        model.training_baseline_mean,
        model.direction,
        model.alpha,
        model.response_rule,
        model.provenance,
        model.direction_sha256,
    )
    assert after == before
    with pytest.raises(AttributeError):
        model.training_baseline_mean = 0.0  # type: ignore[misc]


def test_provenance_is_the_feasibility_gate_execution() -> None:
    reference = ExperimentReference(
        experiment="PROJECT_FEASIBILITY_GATE",
        commit="7aa64543fa45b0e09bb7e247641ce8c926f8a66d",
        results_artifact="reports/PROJECT_FEASIBILITY_GATE_RESULTS.json",
        verdict="FEASIBILITY_CONTINUE",
    )
    for model in _models():
        assert model.provenance == reference
        assert model.response_rule == RESPONSE_RULE
        assert model.alpha == ALPHA
        assert model.direction_id == "D1"
        assert model.direction_seed == 22101
        assert model.direction_sha256 == DIRECTION_SHA256
        assert vector_sha256(np.asarray(model.direction, dtype=np.float64)) == DIRECTION_SHA256


def test_object_has_no_fit_or_update_method() -> None:
    names = set(dir(MechanismResponseModel))
    for banned in ("fit", "update", "update_belief", "search", "calibrate"):
        assert banned not in names


def test_layer_23_and_other_rules_are_rejected() -> None:
    model = _models()[0]
    with pytest.raises(ValueError, match="recorded feasibility-gate family"):
        MechanismResponseModel(
            intervention_id="M22.1-D1-L23",
            layer=23,
            hook="blocks.23.hook_resid_post",
            direction_id=model.direction_id,
            direction_seed=model.direction_seed,
            direction_sha256=model.direction_sha256,
            direction=model.direction,
            alpha=model.alpha,
            response_rule=model.response_rule,
            training_baseline_mean=model.training_baseline_mean,
            provenance=model.provenance,
        )
    with pytest.raises(ValueError, match="prediction = g"):
        MechanismResponseModel(
            intervention_id=model.intervention_id,
            layer=model.layer,
            hook=model.hook,
            direction_id=model.direction_id,
            direction_seed=model.direction_seed,
            direction_sha256=model.direction_sha256,
            direction=model.direction,
            alpha=model.alpha,
            response_rule="prediction = m + b * pre_dot",
            training_baseline_mean=model.training_baseline_mean,
            provenance=model.provenance,
        )
    with pytest.raises(TypeError):
        model.predict({"margin_gradient": (0.0,) * 896, "observed_effect": 1.0})  # type: ignore[arg-type]
