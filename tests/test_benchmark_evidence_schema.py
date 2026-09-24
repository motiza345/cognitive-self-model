"""Schema tests for the frozen 16-D evidence encoder."""

import numpy as np

from src.cognitive_self_model.benchmark import (
    FEATURE_DIM,
    FEATURE_NAMES,
    EnvironmentTrack,
    ScientificFrozenEncoder,
    ScientificFrozenEnvironment,
)


def test_feature_dim_is_sixteen():
    assert FEATURE_DIM == 16
    assert len(FEATURE_NAMES) == 16
    assert len(set(FEATURE_NAMES)) == 16


def test_encode_returns_16d_vector():
    env = ScientificFrozenEnvironment(EnvironmentTrack.KNOWN_VALID, seed=7).reset()
    encoder = ScientificFrozenEncoder()
    obs, _ = env.step(0.3, 0.4, [0.6])
    phi = encoder.encode(obs)
    assert phi.shape == (16,)
    assert np.all(np.isfinite(phi))
