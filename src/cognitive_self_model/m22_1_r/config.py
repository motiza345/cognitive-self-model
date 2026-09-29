"""Read the frozen M22.1-R replay configuration.

This module does not write the configuration and does not change tolerance
values. The required revision is the value recorded for this phase. If the
yaml disagrees with that value, callers must stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .hashing import file_sha256

REPLAY_CONFIG_PATH = "configs/m22_1_r_replay.yaml"
REQUIRED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TOLERANCE_KEYS = ("score_atol", "score_rtol", "delta_atol", "mean_atol", "null_atol")
DEC007_SUCCESS = ("REPLAY_EXACT", "REPLAY_NUMERIC_EQUIVALENT")


@dataclass(frozen=True)
class ReplayConfig:
    document: dict[str, Any]
    path: str
    file_sha256: str

    @property
    def recorded_revision(self) -> str:
        pin = self.document.get("revision_pin") or {}
        return str(pin.get("recorded_revision") or "")

    @property
    def tolerances(self) -> dict[str, float]:
        raw = self.document.get("tolerances") or {}
        return {key: raw[key] for key in TOLERANCE_KEYS}

    @property
    def model_id(self) -> str:
        return "Qwen/Qwen2.5-0.5B"

    def revision_matches_required_pin(self) -> bool:
        return self.recorded_revision == REQUIRED_REVISION


def load_replay_config(repo: Path) -> ReplayConfig:
    path = repo / REPLAY_CONFIG_PATH
    return ReplayConfig(
        document=yaml.safe_load(path.read_text()),
        path=REPLAY_CONFIG_PATH,
        file_sha256=file_sha256(path),
    )
