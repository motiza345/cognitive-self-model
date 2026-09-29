"""Verify the frozen M22.1 reference manifest before any comparison."""

from __future__ import annotations

from pathlib import Path

from .hashing import file_sha256

MANIFEST_PATH = "artifacts/m22_1_r/manifest.sha256"


def verify_reference_manifest(repo: Path) -> tuple[list[dict[str, str]], list[str]]:
    manifest = repo / MANIFEST_PATH
    if not manifest.is_file():
        return [], [f"missing reference manifest {MANIFEST_PATH}"]
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            errors.append(f"manifest line {line_number} is not a sha256 and path")
            continue
        expected, relative = parts
        path = repo / relative
        if not path.is_file():
            errors.append(f"missing reference artifact {relative}")
            rows.append(
                {
                    "path": relative,
                    "expected_sha256": expected,
                    "sha256": "not_computed_file_absent",
                    "match": False,
                }
            )
            continue
        actual = file_sha256(path)
        rows.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "sha256": actual,
                "match": actual == expected,
            }
        )
        if actual != expected:
            errors.append(f"reference sha256 changed for {relative}")
    if not rows and not errors:
        errors.append("reference manifest listed no artifacts")
    return rows, errors
