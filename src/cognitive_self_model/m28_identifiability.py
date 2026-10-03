"""Frozen M28 gate. The eligible pre-evidence set is empty.

This module does not read outcome files and does not load a model.
"""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_VERSION = "M28.REVISION_IDENTIFIABILITY.1"
GATE_STATUS = "NO_SUITABLE_PRE_EVIDENCE"
PRIMARY_CANDIDATE = None
CATALOG_SHA256 = None
UPDATE_N = None
VALIDATION_N = None
HOLDOUT_N = None
SHUFFLE_SEED = 28001
ELIGIBLE = "ELIGIBLE"


class GateBlocked(RuntimeError):
    """Raised when a scoring path is called on this protocol."""


@dataclass(frozen=True)
class Measurement:
    name: str
    storage: str
    eligibility: str


INVENTORY: tuple[Measurement, ...] = (
    Measurement("g", "sealed prediction file", "MECHANISM_OUTPUT"),
    Measurement("prediction_before", "sealed prediction file", "IDENTICAL_TO_MECHANISM_OUTPUT"),
    Measurement("prediction", "sealed M27 prediction file", "IDENTICAL_TO_MECHANISM_OUTPUT"),
    Measurement("family", "sealed prediction file", "ALREADY_SCORED_ON_M26_HOLDOUT"),
    Measurement("baseline_prediction", "sealed prediction file", "CELL_CONSTANT"),
    Measurement("layer", "sealed prediction file", "CELL_MEAN_BASELINE"),
    Measurement("hook", "sealed prediction file", "CELL_MEAN_BASELINE"),
    Measurement("intervention_id", "sealed prediction file", "CELL_MEAN_BASELINE"),
    Measurement("cell", "sealed M27 prediction file", "CELL_MEAN_BASELINE"),
    Measurement("mechanism_id", "sealed M27 prediction file", "CELL_MEAN_BASELINE"),
    Measurement("alpha", "sealed prediction file", "CONSTANT"),
    Measurement("direction_id", "sealed prediction file", "CONSTANT"),
    Measurement("response_rule", "sealed prediction file", "CONSTANT"),
    Measurement("response_form", "sealed M27 prediction file", "CONSTANT"),
    Measurement("initial_gain", "sealed M27 prediction file", "CONSTANT"),
    Measurement("mechanism_version", "sealed prediction file", "CONSTANT"),
    Measurement("outcome_present", "sealed prediction file", "CONSTANT"),
    Measurement("partition", "sealed prediction file", "DESIGN_LABEL"),
    Measurement("prompt_id", "sealed prediction file", "IDENTIFIER"),
    Measurement("timestamp", "sealed prediction file", "CLOSURE_METADATA"),
    Measurement("baseline_output", "outcome file", "ALREADY_SCORED_ON_M26_HOLDOUT"),
    Measurement("observed_effect", "outcome file", "FORBIDDEN_OUTCOME"),
    Measurement("intervened_output", "outcome file", "FORBIDDEN_OUTCOME"),
    Measurement("residual", "episode file", "FORBIDDEN_OUTCOME"),
    Measurement("residual_initial", "M27 episode file", "FORBIDDEN_OUTCOME"),
    Measurement("revised_gain", "post-update artifact", "FORBIDDEN_POST_EVIDENCE"),
    Measurement("delta_cell", "post-update artifact", "FORBIDDEN_POST_EVIDENCE"),
    Measurement("pre_dot", "absent from the episode files", "ABSENT_FROM_SEALED_SCHEMA"),
    Measurement("margin_gradient", "in memory during prediction; absent from the artifacts", "ABSENT_FROM_SEALED_SCHEMA"),
    Measurement("prompt_length", "absent from the sealed schema", "ABSENT_FROM_SEALED_SCHEMA"),
    Measurement("gradient_norm", "absent from the sealed schema", "ABSENT_FROM_SEALED_SCHEMA"),
    Measurement("g_orth", "absent from the sealed schema", "ABSENT_FROM_SEALED_SCHEMA"),
)


def classify(name: str) -> str:
    """Eligibility of one measurement name. Unknown names are invented."""
    for item in INVENTORY:
        if item.name == name:
            return item.eligibility
    return "INVENTED_MEASUREMENT"


def eligible_candidates() -> tuple[Measurement, ...]:
    return tuple(item for item in INVENTORY if item.eligibility == ELIGIBLE)


def execution_gate() -> str:
    """Return the frozen status. Scoring is unreachable from this function."""
    if eligible_candidates():
        raise RuntimeError("an eligible candidate appeared after the freeze")
    if PRIMARY_CANDIDATE is not None or CATALOG_SHA256 is not None:
        raise RuntimeError("catalog or candidate was filled after the freeze")
    return GATE_STATUS


def absolute_error_reduction(residual: float, reference: float, candidate: float) -> float:
    """Episode contribution. Positive when the candidate is closer."""
    return abs(float(residual) - float(reference)) - abs(float(residual) - float(candidate))


def catalog() -> list[dict[str, str]]:
    """This protocol generates no prompts."""
    raise GateBlocked(GATE_STATUS)


def score_holdout(*_args: object, **_kwargs: object) -> str:
    """Refuse before reading rows."""
    raise GateBlocked(GATE_STATUS)


def shuffle_update_assignment(*_args: object, **_kwargs: object) -> str:
    """The reserved seed has no rows to permute."""
    raise GateBlocked(GATE_STATUS)
