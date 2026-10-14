"""Synthetic agent with planted, scope-specific properties. For tests and for
power analysis before a real experiment. Not evidence about real models.

Scopes:
  task=arith,  wrapper=off   confidence is informative and roughly calibrated
  task=arith,  wrapper=on    planted execution bias: the wrapper corrupts some
                             outputs after the model is done, so the true success
                             rate is lower than confidence says; confidence is
                             still informative about the rest
  task=trivia, wrapper=off   confidence is uninformative (pure noise)
"""

from __future__ import annotations

import math
import random

from .core import Scope

SCOPES = {
    "arith_clean": {"task": "arith", "wrapper": "off"},
    "arith_wrapped": {"task": "arith", "wrapper": "on"},
    "trivia": {"task": "trivia", "wrapper": "off"},
}


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def episode(rng: random.Random, scope: str, wrapper_bias: float = 0.35) -> dict:
    z = rng.gauss(0.8, 1.6)                       # latent ease of the item for this agent
    if scope == "trivia":
        p_true = 0.6
        conf = _sig(rng.gauss(1.0, 1.0))          # unrelated to correctness
    else:
        p_true = _sig(z)
        conf = _sig(z + rng.gauss(0.0, 0.4))      # informative, slightly noisy
        if scope == "arith_wrapped":
            p_true *= (1.0 - wrapper_bias)        # invisible to the model's confidence
    return {"context": dict(SCOPES[scope]), "confidence": conf, "correct": rng.random() < p_true,
            "scope": scope}


def stream(n: int, seed: int, weights=(0.4, 0.3, 0.3), wrapper_bias: float = 0.35) -> list[dict]:
    rng = random.Random(seed)
    names = list(SCOPES)
    return [episode(rng, rng.choices(names, weights)[0], wrapper_bias) for _ in range(n)]


def declare_all(registry) -> None:
    registry.declare("arith_clean", "My confidence is informative on clean arithmetic", Scope.of(**SCOPES["arith_clean"]))
    registry.declare("arith_wrapped", "My confidence is informative on wrapped arithmetic", Scope.of(**SCOPES["arith_wrapped"]))
    registry.declare("trivia", "My confidence is informative on trivia", Scope.of(**SCOPES["trivia"]))
