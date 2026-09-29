"""Execution-record builder for one M22.1-R harness attempt.

DEC-009 requires ``entrypoint``, ``invocation``, and boolean ``git_dirty``
on records stored in ``control_plane/RECOVERY.yaml``. Sentinel strings
UNKNOWN, NOT_APPLICABLE, NOT_RECORDED, and empty string are not written.
"""

from __future__ import annotations

SENTINELS = {"UNKNOWN", "NOT_APPLICABLE", "NOT_RECORDED", ""}


def assert_no_sentinels(value, prefix="record") -> None:
    if value is None:
        raise ValueError(f"{prefix} is null")
    if isinstance(value, str):
        if value.strip() in SENTINELS:
            raise ValueError(f"{prefix} contains sentinel {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_no_sentinels(item, f"{prefix}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_sentinels(item, f"{prefix}[{index}]")
        return
    if isinstance(value, (bool, int, float)):
        return
    raise ValueError(f"{prefix} has unsupported type {type(value).__name__}")


def build_record(fields: dict) -> dict:
    required = ("entrypoint", "invocation", "git_dirty", "replay_tier")
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"execution record missing {missing}")
    if not isinstance(fields["git_dirty"], bool):
        raise ValueError("git_dirty must be a boolean")
    if fields["git_dirty"] and not str(fields.get("dirty_reason") or "").strip():
        raise ValueError("dirty_reason is required when git_dirty is true")
    assert_no_sentinels(fields)
    return fields
