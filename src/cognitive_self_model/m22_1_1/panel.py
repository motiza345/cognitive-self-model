"""Deterministic residual-direction panel.

This module can run before any model forward. It does not read margins,
intervention outputs, or held-out behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..m22_1.direction import (
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)


def load_audit_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _project_onto(raw: np.ndarray, accepted: list[np.ndarray]) -> tuple[np.ndarray, float]:
    projected = np.array(raw, dtype=np.float64, copy=True)
    for previous in accepted:
        projected = projected - float(np.dot(projected, previous)) * previous
    return projected, float(np.linalg.norm(projected))


def _new_direction(
    dimension: int,
    seed: int,
    accepted: list[np.ndarray],
    floor: float,
) -> dict:
    for attempt in range(3):
        used_seed = int(seed) + attempt * 10000
        raw = primary_direction(dimension, used_seed)
        projected, residual_norm = _project_onto(raw, accepted)
        if residual_norm >= float(floor):
            final = projected / residual_norm
            return {
                "seed": int(seed),
                "seed_used": used_seed,
                "fallback_attempt": attempt,
                "raw_sha256": vector_sha256(raw),
                "final_sha256": vector_sha256(final),
                "residual_norm_before_normalization": residual_norm,
                "norm": float(np.linalg.norm(final)),
                "vector": final,
            }
    raise RuntimeError(
        f"direction seed {seed} remained numerically degenerate after the pre-declared fallback"
    )


def build_direction_panel(dimension: int, config: dict) -> dict:
    """Build the frozen 8-direction panel. No outcome argument is accepted."""
    floor = float(config["orthogonalization"]["degeneracy_residual_norm_floor"])
    d1 = primary_direction(dimension, int(config["m22_1_primary_seed"]))
    if vector_sha256(d1) != config["m22_1_primary_sha256"]:
        raise RuntimeError("M22.1 primary direction was not reproduced exactly")
    d2 = orthogonal_direction(dimension, int(config["m22_1_control_seed"]), d1)
    if vector_sha256(d2) != config["m22_1_control_sha256"]:
        raise RuntimeError("M22.1 control direction was not reproduced exactly")
    raw_control = primary_direction(dimension, int(config["m22_1_control_seed"]))
    control_projected, control_residual = _project_onto(raw_control, [d1])
    records = [
        {
            "direction_id": "D1",
            "role": "m22_1_primary",
            "seed": int(config["m22_1_primary_seed"]),
            "seed_used": int(config["m22_1_primary_seed"]),
            "fallback_attempt": 0,
            "generation_method": "unit_gaussian",
            "raw_sha256": vector_sha256(d1),
            "final_sha256": vector_sha256(d1),
            "residual_norm_before_normalization": None,
            "norm": float(np.linalg.norm(d1)),
            "vector": d1,
        },
        {
            "direction_id": "D2",
            "role": "m22_1_control",
            "seed": int(config["m22_1_control_seed"]),
            "seed_used": int(config["m22_1_control_seed"]),
            "fallback_attempt": 0,
            "generation_method": "gram_schmidt_against_d1",
            "raw_sha256": vector_sha256(raw_control),
            "final_sha256": vector_sha256(d2),
            "residual_norm_before_normalization": control_residual,
            "norm": float(np.linalg.norm(d2)),
            "vector": d2,
        },
    ]
    if not np.allclose(d2, control_projected / control_residual):
        raise RuntimeError("control reproduction diverged from the recorded projection")
    accepted = [d1, d2]
    for index, seed in enumerate(config["new_direction_seeds"], start=3):
        built = _new_direction(dimension, int(seed), accepted, floor)
        built["direction_id"] = f"D{index}"
        built["role"] = "pre_registered_new"
        built["generation_method"] = "gram_schmidt_against_all_previous"
        records.append(built)
        accepted.append(built["vector"])
    ids = [row["direction_id"] for row in records]
    dots = {
        left["direction_id"]: {
            right["direction_id"]: float(np.dot(left["vector"], right["vector"]))
            for right in records
        }
        for left in records
    }
    return {
        "dimension": int(dimension),
        "direction_ids": ids,
        "directions": records,
        "pairwise_dot": dots,
        "geometric_note": "Geometric orthogonality does not imply causal or functional independence.",
    }


def panel_manifest(panel: dict) -> dict:
    """JSON-safe panel record. Vectors themselves are omitted; hashes identify them."""
    return {
        "dimension": panel["dimension"],
        "direction_ids": panel["direction_ids"],
        "geometric_note": panel["geometric_note"],
        "directions": [
            {key: value for key, value in row.items() if key != "vector"}
            for row in panel["directions"]
        ],
        "pairwise_dot": panel["pairwise_dot"],
    }
