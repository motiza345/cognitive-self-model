"""Ground-truth label semantics and identifiability boundaries."""

import numpy as np

from src.cognitive_self_model.benchmark import (
    EnvironmentTrack,
    ScientificFrozenEncoder,
    ScientificFrozenEnvironment,
)


def _run(track, seed=42, steps=40):
    env = ScientificFrozenEnvironment(track, seed=seed).reset()
    encoder = ScientificFrozenEncoder()
    phis, labels = [], []
    for step in range(steps):
        x = np.sin(step * 0.1)
        u = np.cos(step * 0.1)
        obs, gt = env.step(x, u, [u + 0.2])
        phis.append(encoder.encode(obs))
        labels.append(gt.self_model_invalid)
    return np.asarray(phis), np.asarray(labels)


def test_valid_tracks_label_zero_after_onset():
    for track in (
        EnvironmentTrack.KNOWN_VALID,
        EnvironmentTrack.REGIME_CHANGE_IN_SCOPE,
        EnvironmentTrack.NOVEL_BUT_VALID,
        EnvironmentTrack.FALSE_ALARM,
    ):
        _, labels = _run(track)
        assert labels.max() == 0, f"{track} must be labeled valid (0)"


def test_invalid_tracks_label_one_after_onset():
    for track in (
        EnvironmentTrack.CAUSAL_BREAK,
        EnvironmentTrack.REGIME_CHANGE_OUT_OF_SCOPE,
        EnvironmentTrack.NOVEL_AND_INVALID,
    ):
        _, labels = _run(track)
        assert labels[25:].max() == 1, f"{track} must become invalid (1) after onset"
        assert labels[:20].max() == 0, f"{track} must be valid before onset"


def test_in_scope_regime_change_is_observationally_equivalent_to_known_valid():
    # Same seed -> identical RNG draws; in-scope regime change does not alter the
    # true mechanism, so evidence must be indistinguishable from KNOWN_VALID.
    phi_known, _ = _run(EnvironmentTrack.KNOWN_VALID, seed=99)
    phi_regime, _ = _run(EnvironmentTrack.REGIME_CHANGE_IN_SCOPE, seed=99)
    np.testing.assert_array_equal(phi_known, phi_regime)


def test_novel_but_valid_is_observationally_equivalent_to_known_valid():
    phi_known, _ = _run(EnvironmentTrack.KNOWN_VALID, seed=99)
    phi_novel, _ = _run(EnvironmentTrack.NOVEL_BUT_VALID, seed=99)
    np.testing.assert_array_equal(phi_known, phi_novel)


def test_causal_break_is_separable_from_known_valid():
    phi_known, _ = _run(EnvironmentTrack.KNOWN_VALID, seed=99)
    phi_break, _ = _run(EnvironmentTrack.CAUSAL_BREAK, seed=99)
    # After onset the evidence must diverge (invalid tracks are separable).
    assert not np.allclose(phi_known[25:], phi_break[25:])
