"""Core immutable records (v0.2).

Origins in cognitive-self-model:
- canonical_sha256        <- real_claim_self_model.canonical_sha256
- Status / Action          <- STATUS_* / ACTION_* constants
- Scope matching           <- real_claim_self_model.applicable_status (generalized, with backoff)
- Prediction (sealed)      <- operational_self_model.PredictionRecord
- Evidence (cites seal)    <- operational_self_model.EvidenceRecord + mrsm.evidence.Evidence
- reject_forbidden         <- mrsm.self_model._reject_forbidden

v0.2 change: a prediction now seals the raw confidence, the calibrated
probability, the expected utility of every action, and the exploration
propensity. This is what makes later off-policy evaluation possible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Status(str, Enum):
    # The confidence signal is informative in this scope: use the calibrated value per instance.
    SUPPORTED = "SUPPORTED"
    # Not enough labelled outcomes of both kinds, or the test is undecided.
    UNCERTAIN = "UNCERTAIN"
    # The confidence signal is not informative in this scope: fall back to the base rate.
    CONTRADICTED = "CONTRADICTED"
    NO_APPLICABLE_CLAIM = "NO_APPLICABLE_CLAIM"


class Action(str, Enum):
    ACT = "ACT"
    VERIFY = "VERIFY"
    ABSTAIN = "ABSTAIN"
    DELEGATE = "DELEGATE"


# Actions after which the correctness of the agent's own answer becomes known.
OBSERVING_ACTIONS = frozenset({Action.ACT, Action.VERIFY})

FORBIDDEN_KEYS = {"ground_truth", "label", "gold", "holdout_outcome", "answer_key", "correct", "is_correct"}


class LeakageError(RuntimeError):
    pass


def reject_forbidden(obj: Any) -> None:
    """Refuse inputs that carry the answer. Generalized from mrsm._reject_forbidden."""
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS or lowered.startswith("planted"):
                raise LeakageError(f"forbidden input field: {key}")
            reject_forbidden(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            reject_forbidden(item)


@dataclass(frozen=True)
class Scope:
    """Key/value conditions. A claim applies when every key matches the context.
    The empty scope is GLOBAL and matches everything (used for backoff)."""

    conditions: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def of(**conditions: Any) -> "Scope":
        return Scope(tuple(sorted((k, str(v)) for k, v in conditions.items())))

    def matches(self, context: Mapping[str, Any]) -> bool:
        return all(str(context.get(k)) == v for k, v in self.conditions)

    @property
    def specificity(self) -> int:
        return len(self.conditions)

    def as_dict(self) -> dict[str, str]:
        return dict(self.conditions)


GLOBAL = Scope()


def _finite_prob(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite real number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class Prediction:
    """Sealed before the outcome exists. Never edited."""

    prediction_id: str
    context_sha256: str
    raw_confidence: float | None
    matching_claim_ids: tuple[str, ...]
    used_claim_id: str | None
    used_claim_version: int | None
    used_status: Status
    learning: bool                      # most specific claim still UNCERTAIN
    p_hat: float                        # probability the agent's own answer is correct
    p_source: str                       # "calibrated" | "base_rate" | "raw" | "prior"
    expected_utility: tuple[tuple[str, float], ...]
    greedy_action: Action
    action: Action
    explored: bool
    propensity: float                   # probability of taking `action` under the policy
    seal_sha256: str = ""

    def __post_init__(self) -> None:
        _finite_prob("raw_confidence", self.raw_confidence)
        _finite_prob("p_hat", self.p_hat)
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError("propensity must be in (0, 1]")
        body = {k: v for k, v in asdict(self).items() if k != "seal_sha256"}
        object.__setattr__(self, "seal_sha256", canonical_sha256(body))


@dataclass(frozen=True)
class Evidence:
    """An outcome that cites exactly one sealed prediction.

    success is the correctness of the agent's own (unverified) answer.
    It must be None after a non-observing action (ABSTAIN, DELEGATE).
    """

    evidence_id: str
    prediction_id: str
    prediction_seal_sha256: str
    success: bool | None
    raw_confidence: float | None
    weight: float                       # 1 / propensity, for selection-corrected fitting
    applied_to: tuple[str, ...]
    provenance: str
    source_is_current: bool = True

    def __post_init__(self) -> None:
        if not self.provenance:
            raise ValueError("evidence needs provenance")
        if not (self.weight >= 1.0 and math.isfinite(self.weight)):
            raise ValueError("weight is 1/propensity and must be >= 1")

    def payload(self) -> dict[str, Any]:
        return asdict(self)
