"""Explicit mechanism response rule for the frozen D1 family.

Category C only. The issued prediction is the directional derivative g.
This object is not a self-model and has no update rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.cognitive_self_model.m23.m22_reuse import primary_direction, vector_sha256

RESPONSE_RULE = "prediction = g"
DIRECTION_ID = "D1"
DIRECTION_SEED = 22101
DIRECTION_DIMENSION = 896
DIRECTION_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
ALPHA = 1.0

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ARTIFACT = "reports/PROJECT_FEASIBILITY_GATE_RESULTS.json"
_GATE_COMMIT = "7aa64543fa45b0e09bb7e247641ce8c926f8a66d"

# The feasibility-gate family. Layer 23 is absent on purpose.
_CELLS: tuple[tuple[str, int, str], ...] = (
    ("M22.1-D1-L0", 0, "blocks.0.hook_resid_post"),
    ("M22.1-D1-L8", 8, "blocks.8.hook_resid_post"),
    ("M22.1-D1-L15", 15, "blocks.15.hook_resid_post"),
)


@dataclass(frozen=True)
class ExperimentReference:
    experiment: str
    commit: str
    results_artifact: str
    verdict: str


@dataclass(frozen=True)
class PreInterventionState:
    """Gradient of the yes/no margin with respect to the last-token residual.

    The gradient is evaluated at the pre-intervention residual. The margin is
    the last-position logit of token 9834 minus the logit of token 902.
    The state carries that gradient only.
    """

    margin_gradient: tuple[float, ...]


@dataclass(frozen=True)
class MechanismResponseModel:
    intervention_id: str
    layer: int
    hook: str
    direction_id: str
    direction_seed: int
    direction_sha256: str
    direction: tuple[float, ...]
    alpha: float
    response_rule: str
    training_baseline_mean: float
    provenance: ExperimentReference

    def __post_init__(self) -> None:
        try:
            expected = _cell_table()[self.intervention_id]
        except KeyError as exc:
            raise ValueError("cell fields do not match the recorded feasibility-gate family") from exc
        if (self.layer, self.hook, self.training_baseline_mean) != expected:
            raise ValueError("cell fields do not match the recorded feasibility-gate family")
        if self.direction_id != DIRECTION_ID or self.direction_seed != DIRECTION_SEED:
            raise ValueError("direction provenance is not the frozen D1 direction")
        if self.alpha != ALPHA:
            raise ValueError("alpha is fixed at +1 for this response rule")
        if self.response_rule != RESPONSE_RULE:
            raise ValueError('response_rule must be "prediction = g"')
        if self.provenance != _gate_reference():
            raise ValueError("provenance does not match the feasibility-gate execution")
        vector = np.asarray(self.direction, dtype=np.float64)
        if vector.shape != (DIRECTION_DIMENSION,):
            raise ValueError("direction dimension is not 896")
        if vector_sha256(vector) != DIRECTION_SHA256 or self.direction_sha256 != DIRECTION_SHA256:
            raise ValueError("direction sha256 does not match frozen D1")
        if not np.isfinite(self.training_baseline_mean):
            raise ValueError("training baseline mean must be finite")

    def predict(self, pre_intervention_state: PreInterventionState) -> float:
        if not isinstance(pre_intervention_state, PreInterventionState):
            raise TypeError("pre_intervention_state must be a PreInterventionState")
        gradient = np.asarray(pre_intervention_state.margin_gradient, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        return directional_derivative(gradient, direction)


def directional_derivative(margin_gradient: np.ndarray, direction: np.ndarray) -> float:
    """g = dot(margin gradient at the last token, direction). No coefficient."""
    gradient = np.asarray(margin_gradient, dtype=np.float64)
    vector = np.asarray(direction, dtype=np.float64)
    if gradient.shape != vector.shape or gradient.ndim != 1:
        raise ValueError("margin gradient shape does not match the direction")
    if not np.isfinite(gradient).all() or not np.isfinite(vector).all():
        raise ValueError("margin gradient and direction must be finite")
    return float(np.dot(gradient, vector))


def frozen_mechanism_response_models() -> tuple[MechanismResponseModel, ...]:
    direction = tuple(float(value) for value in _d1())
    reference = _gate_reference()
    table = _cell_table()
    return tuple(
        MechanismResponseModel(
            intervention_id=intervention_id,
            layer=layer,
            hook=hook,
            direction_id=DIRECTION_ID,
            direction_seed=DIRECTION_SEED,
            direction_sha256=DIRECTION_SHA256,
            direction=direction,
            alpha=ALPHA,
            response_rule=RESPONSE_RULE,
            training_baseline_mean=table[intervention_id][2],
            provenance=reference,
        )
        for intervention_id, layer, hook in _CELLS
    )


def _gate_reference() -> ExperimentReference:
    return ExperimentReference(
        experiment="PROJECT_FEASIBILITY_GATE",
        commit=_GATE_COMMIT,
        results_artifact=_RESULTS_ARTIFACT,
        verdict="FEASIBILITY_CONTINUE",
    )


def _d1() -> np.ndarray:
    vector = primary_direction(DIRECTION_DIMENSION, DIRECTION_SEED)
    if vector_sha256(vector) != DIRECTION_SHA256:
        raise RuntimeError("resampled D1 does not match the recorded sha256")
    return vector


def _cell_table() -> dict[str, tuple[int, str, float]]:
    payload = json.loads((_ROOT / _RESULTS_ARTIFACT).read_text(encoding="utf-8"))
    if payload.get("verdict") != "FEASIBILITY_CONTINUE":
        raise RuntimeError("feasibility-gate artifact is not a continue result")
    if payload.get("candidate") != "g" or payload.get("observable") != "g":
        raise RuntimeError("feasibility-gate candidate is not g")
    if float(payload.get("alpha")) != ALPHA:
        raise RuntimeError("feasibility-gate alpha is not +1")
    if payload.get("direction_sha256") != DIRECTION_SHA256:
        raise RuntimeError("feasibility-gate direction sha256 does not match D1")
    if list(payload.get("family", [])) != [intervention_id for intervention_id, _, _ in _CELLS]:
        raise RuntimeError("feasibility-gate family is not the frozen three cells")
    means = payload.get("train_means")
    if not isinstance(means, dict) or set(means) != {intervention_id for intervention_id, _, _ in _CELLS}:
        raise RuntimeError("feasibility-gate train means do not match the frozen family")
    table: dict[str, tuple[int, str, float]] = {}
    for intervention_id, layer, hook in _CELLS:
        table[intervention_id] = (layer, hook, float(means[intervention_id]))
    return table
