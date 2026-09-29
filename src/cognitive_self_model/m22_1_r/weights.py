"""Local weight-file identity for one pinned Hugging Face revision.

No expected hash is invented. A missing snapshot is reported as absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from .hashing import file_sha256

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
EXPECTED_WEIGHT_HASHES_ABSENT = "ABSENT_NO_REFERENCE_RECORDED"


def default_hub_cache() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(key)
        if value:
            return Path(value)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def snapshot_directory(cache_root: Path, model_id: str, revision: str) -> Path:
    folder = "models--" + model_id.replace("/", "--")
    return cache_root / folder / "snapshots" / revision


def list_weight_files(snapshot: Path) -> list[Path]:
    if not snapshot.is_dir():
        return []
    found: list[Path] = []
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        if not path.name.endswith(WEIGHT_SUFFIXES):
            continue
        if "optimizer" in path.name:
            continue
        found.append(path)
    return sorted(found, key=lambda item: item.relative_to(snapshot).as_posix())


def hash_weight_files(snapshot: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in list_weight_files(snapshot):
        relative = path.relative_to(snapshot).as_posix()
        rows.append({"relative_path": relative, "sha256": file_sha256(path)})
    return rows


def weight_identity_mismatch(
    found: list[dict[str, str]],
    expected: dict[str, str] | None,
) -> str | None:
    """Return a mismatch reason, or None when there is nothing to reject.

    ``expected is None`` means no reference weight hash is on record. That
    fact must be recorded by the caller. It is not a match and not a mismatch.
    """
    if expected is None:
        return None
    found_map = {row["relative_path"]: row["sha256"] for row in found}
    if set(found_map) != set(expected):
        return "weight file set does not match the supplied expected hashes"
    for relative, digest in sorted(expected.items()):
        if found_map[relative] != digest:
            return f"weight sha256 mismatch for {relative}"
    return None
