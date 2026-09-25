"""Intervention directions that do not depend on outcomes."""

from __future__ import annotations

import hashlib

import numpy as np


def _unit_gaussian(dimension: int, seed: int) -> np.ndarray:
    if dimension < 2:
        raise ValueError("Direction dimension must be at least 2.")
    vector = np.random.default_rng(int(seed)).normal(size=int(dimension)).astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Failed to sample a finite direction.")
    return vector / norm


def vector_sha256(vector: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def primary_direction(dimension: int, seed: int) -> np.ndarray:
    return _unit_gaussian(dimension, seed)


def orthogonal_direction(dimension: int, seed: int, primary: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    if primary.shape != (dimension,):
        raise ValueError("Primary direction shape does not match dimension.")
    primary_norm = float(np.linalg.norm(primary))
    if abs(primary_norm - 1.0) > 1e-6:
        raise ValueError("Primary direction must be unit length.")
    raw = _unit_gaussian(dimension, seed)
    projected = raw - float(np.dot(raw, primary)) * primary
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("Orthogonal direction collapsed.")
    orthogonal = projected / norm
    if abs(float(np.dot(orthogonal, primary))) > 1e-6:
        raise RuntimeError("Orthogonalization failed.")
    return orthogonal
