"""Evidence-grounded causal self-model for the planted two-hop benchmark.

The model never receives a ground-truth mechanism label. Discovery statistics
are behavioral effects. Holdout outcomes are refused once the candidate is frozen.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mrsm import HEADS
from src.mrsm.evidence import Evidence
from src.mrsm.transforms import apply_transform, invert_transform

_FORBIDDEN_EXACT = {
    "ground_truth",
    "mechanism_label",
    "holdout_outcome",
    "holdout_accuracy",
    "holdout_margin",
}


class LeakageError(RuntimeError):
    pass


def _reject_forbidden(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_EXACT or lowered.startswith("planted"):
                raise LeakageError(f"forbidden input field: {key}")
            _reject_forbidden(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _reject_forbidden(item)


def pair_name(left: str, right: str) -> str:
    if HEADS.index(left) > HEADS.index(right):
        left, right = right, left
    return f"{left}+{right}"


def _split_pair(name: str) -> tuple[str, str]:
    left, right = name.split("+")
    return left, right


class SelfModelCore:
    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.effects: dict[str, float] = {}
        self.joints: dict[str, float] = {}
        self.interactions: dict[str, float] = {}
        self.mechanism: dict[str, Any] | None = None
        self.uncertainty: float | None = None
        self.evidence_ids: list[str] = []
        self.frozen = False
        self.identifiable = False

    def update(
        self,
        head_effects: dict[str, float],
        joint_effects: dict[str, float],
        evidence_ids: list[str],
        *,
        min_abs_interaction: float,
        min_relative_gap: float,
    ) -> dict[str, Any]:
        if self.frozen:
            raise RuntimeError("self-model is frozen")
        _reject_forbidden(head_effects)
        _reject_forbidden(joint_effects)
        missing = [name for name in HEADS if name not in head_effects]
        if missing:
            raise ValueError(f"missing head effects: {missing}")
        self.effects = {name: float(head_effects[name]) for name in HEADS}
        self.joints = {}
        self.interactions = {}
        for left_i, left in enumerate(HEADS):
            for right in HEADS[left_i + 1 :]:
                key = pair_name(left, right)
                if key not in joint_effects:
                    raise ValueError(f"missing joint effect: {key}")
                joint = float(joint_effects[key])
                interaction = joint - self.effects[left] - self.effects[right]
                self.joints[key] = joint
                self.interactions[key] = interaction
        self.evidence_ids = list(evidence_ids)
        values = np.array(list(self.effects.values()), dtype=np.float64)
        self.uncertainty = float(np.std(values))
        self.mechanism = self._select(min_abs_interaction, min_relative_gap)
        self.identifiable = self.mechanism is not None and self.mechanism["status"] == "IDENTIFIED"
        return self.explain()

    def _select(self, min_abs: float, min_gap: float) -> dict[str, Any]:
        ranked = sorted(self.interactions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_name, top_value = ranked[0]
        second = abs(ranked[1][1]) if len(ranked) > 1 else 0.0
        top_abs = abs(top_value)
        gap = (top_abs - second) / top_abs if top_abs > 0.0 else 0.0
        if top_abs < min_abs or gap < min_gap:
            return {
                "status": "NOT_IDENTIFIABLE",
                "pair": None,
                "ordered": None,
                "interaction": top_value,
                "gap": gap,
            }
        left, right = _split_pair(top_name)
        upstream, downstream = _direction(left, right)
        return {
            "status": "IDENTIFIED",
            "pair": [upstream, downstream],
            "ordered": [upstream, downstream],
            "interaction": top_value,
            "gap": gap,
            "abstraction": f"{upstream}->{downstream}",
        }

    def freeze(self) -> dict[str, Any]:
        self.frozen = True
        return self.explain()

    def predict(self, intervention_id: str) -> float:
        kind, name = intervention_id.split(":", 1)
        if kind == "single":
            return float(self.effects[name])
        if kind == "pair":
            return float(self.joints[name])
        raise KeyError(intervention_id)

    def explain(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "scope": self.scope,
            "uncertainty": self.uncertainty,
            "evidence_ids": list(self.evidence_ids),
            "effects": dict(self.effects),
            "identifiable": self.identifiable,
            "frozen": self.frozen,
        }

    def intervene(self, intervention_id: str) -> dict[str, Any]:
        return {
            "intervention_id": intervention_id,
            "predicted_margin_drop": self.predict(intervention_id),
            "mechanism": None if self.mechanism is None else self.mechanism.get("abstraction"),
            "scope": self.scope,
            "uncertainty": self.uncertainty,
            "executed": False,
        }

    def compare(self, other: dict[str, float]) -> dict[str, float]:
        out = {}
        for key, value in other.items():
            out[key] = float(self.predict(key) - float(value))
        return out

    def falsify(self, evidence: Evidence, *, tolerance: float) -> dict[str, Any]:
        if evidence.residual is None:
            return {"falsified": False, "reason": "no residual"}
        challenged = abs(float(evidence.residual)) > float(tolerance)
        return {
            "falsified": challenged,
            "observation_id": evidence.observation_id,
            "intervention_id": evidence.intervention_id,
            "residual": evidence.residual,
            "tolerance": tolerance,
            "mechanism_changed": False,
        }

    def effect_vector(self) -> list[float]:
        return [float(self.effects[name]) for name in HEADS]

    def read_transformed(self, transform_id: str) -> dict[str, Any]:
        """Read the same causal predictions through a frozen invertible map."""
        original = np.array(self.effect_vector(), dtype=np.float64)
        transformed = apply_transform(transform_id, original)
        recovered = invert_transform(transform_id, transformed)
        recovered_effects = {name: float(recovered[i]) for i, name in enumerate(HEADS)}
        abstraction = None if not self.mechanism else self.mechanism.get("abstraction")
        if self.mechanism is not None and self.mechanism.get("status") == "NOT_IDENTIFIABLE":
            abstraction = "NOT_IDENTIFIABLE"
        return {
            "transformation_id": transform_id,
            "original_vector": original.tolist(),
            "transformed_coordinate": transformed.tolist(),
            "recovered_vector": recovered.tolist(),
            "recovered_effects": recovered_effects,
            "mechanism_abstraction": abstraction if abstraction is not None else "NOT_IDENTIFIABLE",
            "scope": self.scope,
            "uncertainty": self.uncertainty,
        }

    def choose_ablation(self) -> str:
        best = min(HEADS, key=lambda name: (self.effects[name], HEADS.index(name)))
        return best


def _direction(left: str, right: str) -> tuple[str, str]:
    def key(name: str) -> tuple[int, int]:
        layer = int(name[1])
        head = int(name[3])
        return layer, head

    ordered = tuple(sorted((left, right), key=key))
    return ordered[0], ordered[1]
