"""SHA256 helper.

Reuses ``common.io_contracts.file_sha256`` when that module is importable.
The algorithm is the file-byte SHA256 already used by the control plane.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    for module_name in ("common.io_contracts", "src.common.io_contracts"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        return module.file_sha256(path)
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
