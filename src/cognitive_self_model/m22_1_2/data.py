"""Load frozen measurements and build response matrices.

The matrix builder reads actual_delta only. Null margins, intervened scores,
and token text are not features.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..m22_1.prompts import frozen_prompts, prompt_manifest_sha256


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prompts_by_split() -> dict[str, list]:
    grouped = {role: [] for role in ("discovery", "validation", "replication")}
    for prompt in frozen_prompts():
        grouped[prompt.role].append(prompt)
    for role, records in grouped.items():
        grouped[role] = sorted(records, key=lambda record: record.prompt_id)
        if len(grouped[role]) != 6:
            raise RuntimeError(f"{role} does not contain the frozen six prompts")
    manifest = prompt_manifest_sha256()
    return grouped, manifest


def response_matrix(
    records: list[dict],
    *,
    split: str,
    alpha: float,
    prompt_ids: list[str],
    direction_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    matrix = np.full((len(prompt_ids), len(direction_ids)), np.nan, dtype=np.float64)
    regimes = [None for _ in prompt_ids]
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    direction_index = {direction_id: index for index, direction_id in enumerate(direction_ids)}
    for record in records:
        if record["role_split"] != split or float(record["alpha"]) != float(alpha):
            continue
        row = prompt_index[record["prompt_id"]]
        column = direction_index[record["direction_id"]]
        if np.isfinite(matrix[row, column]):
            raise RuntimeError("Duplicate response cell")
        matrix[row, column] = float(record["actual_delta"])
        regimes[row] = record["regime"]
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Response matrix is missing a frozen cell")
    if any(regime is None for regime in regimes):
        raise RuntimeError("A prompt is missing its regime")
    return matrix, regimes


def load_measurement_payload(path: str | Path, config: dict) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    environment = payload["environment"]
    expected = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "prompt_manifest_sha256": config["prompt_manifest_sha256"],
    }
    for key, value in expected.items():
        if environment[key] != value:
            raise RuntimeError(f"Frozen measurement field {key} does not match the audit config")
    hooks = {record["hook_name_seen"] for record in payload["records"]}
    if hooks != {config["hook_name"]}:
        raise RuntimeError("Measurements were not taken at the frozen hook")
    return payload
