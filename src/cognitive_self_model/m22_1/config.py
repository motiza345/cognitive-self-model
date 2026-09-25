"""Load the frozen M22.1 preflight contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "m22_1_preflight.json"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path if path is not None else CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    _validate_config(config)
    return config


def config_sha256(config: dict[str, Any] | None = None) -> str:
    payload = config if config is not None else load_config()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hook_name(layer: int, config: dict[str, Any] | None = None) -> str:
    cfg = config if config is not None else load_config()
    return str(cfg["hook_template"]).format(layer=int(layer))


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("milestone") != "M22.1":
        raise ValueError("M22.1 config milestone mismatch.")
    if config.get("model_id") != "Qwen/Qwen2.5-0.5B":
        raise ValueError("M22.1 model id is frozen.")
    if config.get("dtype") != "float32":
        raise ValueError("M22.1 dtype is frozen to float32.")
    if config["outcome"]["type"] != "logit_margin":
        raise ValueError("M22.1 outcome type is frozen.")
    grid = [float(value) for value in config["magnitude_grid"]]
    if 0.0 not in grid:
        raise ValueError("Magnitude grid must include 0.")
    primary = float(config["primary_alpha"])
    if primary not in grid or -primary not in grid:
        raise ValueError("Primary alpha and its negation must be on the grid.")
    if int(config["n_candidate_layers"]) < 1:
        raise ValueError("Candidate layer count must be positive.")
    if float(config["null_bug_threshold"]) <= 0.0:
        raise ValueError("Null bug threshold must be positive.")
