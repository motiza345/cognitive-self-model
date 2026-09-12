from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_M21_2_4_3_NPZ_KEYS = (
    "text",
    "raw_probability_calibration",
    "y_calibration",
    "track_calibration",
    "episode_id_calibration",
    "raw_probability_test",
    "y_test",
    "track_test",
    "episode_id_test",
)


def file_sha256(
    file_path: str | Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Compute a SHA256 fingerprint for a local file.

    This identifies the precise data artifact used by an experiment
    without committing private or large NPZ data into Git.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.sha256()

    with open(path, "rb") as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def validate_npz_required_keys(
    npz_path: str | Path,
    required_keys: Iterable[str] = REQUIRED_M21_2_4_3_NPZ_KEYS,
) -> None:
    """
    Validate exact required keys for the M21.2.4.3 input NPZ.

    Raises:
        ValueError: if the key set differs from the expected contract.
    """
    npz_path = Path(npz_path)
    required_key_set = set(required_keys)

    with np.load(npz_path, allow_pickle=False) as data:
        actual_key_set = set(data.files)

    missing_keys = sorted(required_key_set - actual_key_set)
    unexpected_keys = sorted(actual_key_set - required_key_set)

    if missing_keys or unexpected_keys:
        raise ValueError(
            "NPZ key contract violation. "
            f"Missing keys: {missing_keys}. "
            f"Unexpected keys: {unexpected_keys}. "
            f"Expected exact keys: {sorted(required_key_set)}."
        )
