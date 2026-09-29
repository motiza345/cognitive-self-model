"""Non-self-model baselines required by the preregistration."""

from __future__ import annotations

from src.mrsm import HEADS
from src.mrsm.self_model import pair_name


def predictions(head_effects: dict[str, float]) -> dict[str, dict[str, float]]:
    singles = [float(head_effects[name]) for name in HEADS]
    mean = sum(singles) / len(singles)
    b0: dict[str, float] = {}
    b1: dict[str, float] = {}
    b2: dict[str, float] = {}
    b3: dict[str, float] = {}
    for name in HEADS:
        key = f"single:{name}"
        b0[key] = 0.0
        b1[key] = mean
        b2[key] = float(head_effects[name])
        b3[key] = 0.0
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            key = f"pair:{pair_name(left, right)}"
            b0[key] = 0.0
            b1[key] = 2.0 * mean
            b2[key] = float(head_effects[left]) + float(head_effects[right])
            b3[key] = 0.0
    return {"B0": b0, "B1": b1, "B2": b2, "B3": b3}
