"""Arm runner and paired bootstrap for pre-registered comparisons (M34b style).

An episode is {"context": {...}, "confidence": float|None, "correct": bool}.
`correct` is visible only to the harness. The layer receives it only through
`after`, and only when the chosen action reveals it.

Protocol: learning phase (online, with exploration) -> freeze -> holdout
scored once. Utility is computed with the same UtilityTable for every arm.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from .core import OBSERVING_ACTIONS, Action
from .layer import CognitiveLayer
from .policy import UtilityTable


def run_arm(make_layer: Callable[[], CognitiveLayer], learn: Sequence[dict], holdout: Sequence[dict],
            utility: UtilityTable | None = None) -> dict:
    layer = make_layer()
    u = utility or layer.policy.utility
    learn_util = []
    for ep in learn:
        pred = layer.before(ep["context"], ep.get("confidence"))
        observed = ep["correct"] if pred.action in OBSERVING_ACTIONS else None
        layer.after(pred.prediction_id, observed, provenance="harness:learn")
        learn_util.append(u.realized(pred.action, ep["correct"]))
    layer.freeze()
    hold_util, actions = [], []
    for ep in holdout:
        pred = layer.before(ep["context"], ep.get("confidence"))
        observed = ep["correct"] if pred.action in OBSERVING_ACTIONS else None
        layer.after(pred.prediction_id, observed, provenance="harness:holdout")
        hold_util.append(u.realized(pred.action, ep["correct"]))
        actions.append(pred.action.value)
    return {"layer": layer, "learn_utility": np.array(learn_util), "holdout_utility": np.array(hold_util),
            "actions": actions}


def oracle_instance(holdout: Sequence[dict], utility: UtilityTable | None = None) -> np.ndarray:
    """Knows correctness per instance: best action given the true outcome."""
    u = utility or UtilityTable()
    out = []
    for ep in holdout:
        options = [u.realized(a, ep["correct"]) for a in (Action.ACT, Action.VERIFY, Action.ABSTAIN)]
        out.append(max(options))
    return np.array(out)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 5000, seed: int = 23001,
                     alpha: float = 0.05) -> dict:
    """Mean of a - b with a percentile interval over holdout episodes (unit = episode)."""
    if a.shape != b.shape:
        raise ValueError("paired arms must score the same holdout")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(draws, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(d.mean()), "lower": float(lo), "upper": float(hi), "n": int(d.size)}
