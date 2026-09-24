"""Eight-key NPZ bundle contract."""

import numpy as np
import pytest

from src.cognitive_self_model.benchmark.bundle import (
    BUNDLE_KEYS,
    verify_bundle_keys,
    write_bundle,
)


def _payload(n_cal=5, n_test=7):
    return dict(
        raw_probability_calibration=np.linspace(0, 1, n_cal),
        y_calibration=np.array([0, 1, 0, 1, 0])[:n_cal],
        track_calibration=np.array(["KNOWN_VALID"] * n_cal),
        episode_id_calibration=np.array([f"cal__e{i}" for i in range(n_cal)]),
        raw_probability_test=np.linspace(0, 1, n_test),
        y_test=np.array([0, 1, 0, 1, 0, 1, 0])[:n_test],
        track_test=np.array(["CAUSAL_BREAK"] * n_test),
        episode_id_test=np.array([f"test__e{i}" for i in range(n_test)]),
    )


def test_contract_has_exactly_eight_keys():
    assert len(BUNDLE_KEYS) == 8
    assert "text" not in BUNDLE_KEYS


def test_write_bundle_produces_exactly_eight_keys(tmp_path):
    path = write_bundle(tmp_path / "M21_2_4_3_3_input.npz", **_payload())
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(BUNDLE_KEYS)


def test_verify_rejects_missing_and_unexpected_keys():
    with pytest.raises(ValueError):
        verify_bundle_keys(list(BUNDLE_KEYS)[:-1])
    with pytest.raises(ValueError):
        verify_bundle_keys(list(BUNDLE_KEYS) + ["text"])


def test_probabilities_must_be_in_unit_interval(tmp_path):
    bad = _payload()
    bad["raw_probability_test"] = np.array([0.5, 1.5, -0.1, 0.2, 0.3, 0.4, 0.6])
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "bad.npz", **bad)
