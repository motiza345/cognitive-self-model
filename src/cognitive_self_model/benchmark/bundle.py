"""
The eight-key M21.2.4.3.3 input bundle contract.

This is the authoritative producer-side definition of the prediction bundle that
the M21.2.4.3.3 calibration audit consumes. It is **exactly eight keys** (no
``text`` key): the raw uncalibrated ``q_invalid`` probabilities, binary
``self_model_invalid`` labels, track names, and stable episode ids for the
calibration and test splits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

BUNDLE_KEYS = (
    "raw_probability_calibration",
    "y_calibration",
    "track_calibration",
    "episode_id_calibration",
    "raw_probability_test",
    "y_test",
    "track_test",
    "episode_id_test",
)


def verify_bundle_keys(keys: Iterable[str]) -> None:
    """Raise ``ValueError`` unless ``keys`` is exactly the eight-key contract."""
    found = set(keys)
    expected = set(BUNDLE_KEYS)
    if found != expected:
        missing = sorted(expected - found)
        unexpected = sorted(found - expected)
        raise ValueError(
            "NPZ key contract violation (expected exactly 8 keys). "
            f"Missing: {missing}. Unexpected: {unexpected}. "
            f"Expected: {sorted(expected)}."
        )


def write_bundle(
    path: str | Path,
    *,
    raw_probability_calibration: np.ndarray,
    y_calibration: np.ndarray,
    track_calibration: np.ndarray,
    episode_id_calibration: np.ndarray,
    raw_probability_test: np.ndarray,
    y_test: np.ndarray,
    track_test: np.ndarray,
    episode_id_test: np.ndarray,
) -> Path:
    """Write the eight-key NPZ bundle and verify the on-disk key contract."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for name, arr in (
        ("raw_probability_calibration", raw_probability_calibration),
        ("raw_probability_test", raw_probability_test),
    ):
        arr = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
        if not np.all((arr >= 0.0) & (arr <= 1.0)):
            raise ValueError(f"{name} must lie in [0, 1].")

    np.savez_compressed(
        path,
        raw_probability_calibration=np.asarray(raw_probability_calibration, dtype=float),
        y_calibration=np.asarray(y_calibration, dtype=int),
        track_calibration=np.asarray(track_calibration, dtype=str),
        episode_id_calibration=np.asarray(episode_id_calibration, dtype=str),
        raw_probability_test=np.asarray(raw_probability_test, dtype=float),
        y_test=np.asarray(y_test, dtype=int),
        track_test=np.asarray(track_test, dtype=str),
        episode_id_test=np.asarray(episode_id_test, dtype=str),
    )

    with np.load(path, allow_pickle=False) as saved:
        verify_bundle_keys(saved.files)

    return path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest of a file (for reproducibility manifests)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def binary_logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
