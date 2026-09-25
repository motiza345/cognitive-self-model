"""Runtime identity for the preflight manifest. Does not record credentials."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(args: list[str], root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def sanitize_origin(url: str) -> str:
    if "://" not in url or "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    return f"{scheme}://{rest.split('@', 1)[1]}"


def git_identity(root: Path | None = None) -> dict[str, Any]:
    repository = root if root is not None else Path(__file__).resolve().parents[3]
    origin = _git(["remote", "get-url", "origin"], repository)
    status = _git(["status", "--porcelain"], repository)
    return {
        "branch": _git(["branch", "--show-current"], repository),
        "commit": _git(["rev-parse", "HEAD"], repository),
        "dirty": None if status is None else status != "",
        "origin": None if origin is None else sanitize_origin(origin),
    }
