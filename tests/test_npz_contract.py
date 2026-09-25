from pathlib import Path

import numpy as np
import pytest

from src.common.io_contracts import (
    REQUIRED_M21_2_4_3_NPZ_KEYS,
    validate_npz_required_keys,
)


def make_valid_npz_payload():
    return {
        "raw_probability_calibration": np.array([0.1, 0.2]),
        "y_calibration": np.array([0, 1]),
        "track_calibration": np.array(["KNOWN_VALID", "CAUSAL_BREAK"]),
        "episode_id_calibration": np.array(["cal_ep_000", "cal_ep_001"]),
        "raw_probability_test": np.array([0.3, 0.4]),
        "y_test": np.array([0, 1]),
        "track_test": np.array(["KNOWN_VALID", "CAUSAL_BREAK"]),
        "episode_id_test": np.array(["test_ep_000", "test_ep_001"]),
    }


def test_valid_npz_contract_passes(tmp_path: Path):
    npz_path = tmp_path / "valid_input.npz"

    np.savez(npz_path, **make_valid_npz_payload())

    validate_npz_required_keys(npz_path)


def test_missing_key_raises_error(tmp_path: Path):
    npz_path = tmp_path / "invalid_input.npz"

    payload = make_valid_npz_payload()
    del payload["y_test"]

    np.savez(npz_path, **payload)

    with pytest.raises(ValueError, match="Missing keys"):
        validate_npz_required_keys(npz_path)


def test_unexpected_key_raises_error(tmp_path: Path):
    npz_path = tmp_path / "invalid_input.npz"

    payload = make_valid_npz_payload()
    payload["unexpected_key"] = np.array([123])

    np.savez(npz_path, **payload)

    with pytest.raises(ValueError, match="Unexpected keys"):
        validate_npz_required_keys(npz_path)


def test_required_contract_has_eight_keys():
    assert len(REQUIRED_M21_2_4_3_NPZ_KEYS) == 8
    assert "text" not in REQUIRED_M21_2_4_3_NPZ_KEYS
