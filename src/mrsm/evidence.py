"""Structured evidence. The self-model may store these. Ground truth is not a field."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Evidence:
    observation_id: str
    intervention_id: str
    prediction: float | None
    actual_outcome: float | None
    residual: float | None
    context: dict[str, Any]
    scope: str
    evidence_type: str
    timestamp: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def residual_of(prediction: float | None, actual: float | None) -> float | None:
    if prediction is None or actual is None:
        return None
    return float(actual) - float(prediction)
