"""Frozen H3 coordinate transforms. Parameters come from the preregistration."""

from __future__ import annotations

import hashlib

import numpy as np

from src.mrsm import HEADS

_A_DIAG = np.array([1.5, 0.5, 2.0, 0.25, 1.25, 0.75, 1.1, 0.9], dtype=np.float64)
_B = np.full(8, 0.01, dtype=np.float64)
_PERM = np.array([2, 0, 3, 1, 6, 4, 7, 5], dtype=np.int64)
_T3_SHA = "0194bd288a3c6fd1ac054158cf958d507d141a83c973ef50c565a040648bac9a"


def orthogonal_matrix(seed: int = 17291) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(8, 8))
    q, r = np.linalg.qr(raw)
    sign = np.sign(np.diag(r))
    sign[sign == 0.0] = 1.0
    q = q * sign
    digest = hashlib.sha256(np.ascontiguousarray(q, dtype=np.float64).tobytes()).hexdigest()
    if digest != _T3_SHA:
        raise RuntimeError("T3 matrix does not match the preregistered hash")
    return q


def _as_vec(values: np.ndarray | list[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.shape != (len(HEADS),):
        raise ValueError(f"expected {len(HEADS)} coordinates")
    return vec


def apply_transform(transform_id: str, values: np.ndarray | list[float]) -> np.ndarray:
    vec = _as_vec(values)
    if transform_id == "T1":
        return _A_DIAG * vec + _B
    if transform_id == "T2":
        return vec[_PERM]
    if transform_id == "T3":
        return orthogonal_matrix() @ vec
    raise KeyError(transform_id)


def invert_transform(transform_id: str, values: np.ndarray | list[float]) -> np.ndarray:
    vec = _as_vec(values)
    if transform_id == "T1":
        return (vec - _B) / _A_DIAG
    if transform_id == "T2":
        out = np.empty_like(vec)
        out[_PERM] = vec
        return out
    if transform_id == "T3":
        q = orthogonal_matrix()
        return q.T @ vec
    raise KeyError(transform_id)


def frozen_parameters() -> dict[str, object]:
    q = orthogonal_matrix()
    return {
        "T1": {"A_diagonal": _A_DIAG.tolist(), "b": _B.tolist()},
        "T2": {"perm": _PERM.tolist()},
        "T3": {
            "seed": 17291,
            "matrix_sha256": _T3_SHA,
            "matrix": q.tolist(),
        },
    }
