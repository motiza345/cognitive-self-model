"""Leakage guard: the encoder must be agnostic to ground-truth labels (L1/L2)."""

import numpy as np

from src.cognitive_self_model.benchmark import (
    EnvironmentTrack,
    ScientificFrozenEncoder,
    ScientificFrozenEnvironment,
)


def test_encoder_is_label_agnostic():
    # Same observation, two independent encoders -> identical evidence, regardless
    # of the ground-truth labels attached to that observation.
    env = ScientificFrozenEnvironment(EnvironmentTrack.CAUSAL_BREAK, seed=123).reset()
    obs = None
    gt = None
    for _ in range(25):  # past onset so the label is 1
        obs, gt = env.step(0.5, 0.5, [0.6, 0.7])

    assert gt.self_model_invalid == 1  # label present but must not touch encoding

    phi_a = ScientificFrozenEncoder().encode(obs)
    phi_b = ScientificFrozenEncoder().encode(obs)
    np.testing.assert_array_equal(phi_a, phi_b)


def test_observation_excludes_forbidden_fields():
    env = ScientificFrozenEnvironment(EnvironmentTrack.NOVEL_AND_INVALID, seed=1).reset()
    obs, _ = env.step(0.1, 0.2, [0.3])
    fields = set(vars(obs).keys())
    for forbidden in ("track", "true_active_mechanism", "is_contextually_invalid", "self_model_invalid"):
        assert forbidden not in fields
