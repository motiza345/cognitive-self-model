"""Planted P data helpers. Holdout scoring does not live here."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mrsm_p_v1  # noqa: E402


def config() -> mrsm_p_v1.Config:
    return mrsm_p_v1.Config()


def dataset(n: int, seed: int):
    return mrsm_p_v1.make_dataset(config(), n, seed)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holdout_manifest(seed: int, n: int, created_at: str) -> dict[str, Any]:
    ds = dataset(n, seed)
    raw = ds.tokens.numpy().tobytes() + ds.targets.numpy().tobytes()
    ids = [f"holdout-{seed}-{index}" for index in range(n)]
    return {
        "dataset_version": "MRSM_P_v1_layout_v1",
        "seed": seed,
        "n": n,
        "sample_ids": ids,
        "hash": hashlib.sha256(raw).hexdigest(),
        "creation_timestamp": created_at,
        "outcomes_included": False,
        "scored": False,
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
