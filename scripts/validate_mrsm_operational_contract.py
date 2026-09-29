#!/usr/bin/env python3
"""Validate the draft MRSM operational contract.

This script checks epistemic keys and rejects numeric threshold fields.
It does not load a model and it does not score a run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

Q_EVALUATES = (
    "prediction_quality",
    "counterfactual_consistency",
    "falsification_behavior",
    "abstention_quality",
    "scope_consistency",
)
H3_FAMILIES = (
    "invertible_affine",
    "coordinate_permutation",
    "orthogonal_basis_change",
)
H3_INVARIANTS = (
    "causal_prediction",
    "mechanism_identity_at_declared_abstraction",
    "scope",
    "falsification_behavior",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_paths(value: Any, prefix: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key != "thresholds_status" and "threshold" in str(key).lower():
                found.append(f"threshold field is not allowed: {path}")
            found.extend(_numeric_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_numeric_paths(item, f"{prefix}[{index}]"))
    elif _is_number(value):
        found.append(f"numeric value is not allowed: {prefix}={value!r}")
    return found


def _require_mapping(document: Any, key: str, errors: list[str]) -> dict[str, Any] | None:
    value = document.get(key) if isinstance(document, dict) else None
    if not isinstance(value, dict):
        errors.append(f"missing mapping: {key}")
        return None
    return value


def _require_string_list(mapping: dict[str, Any], key: str, expected: tuple[str, ...], errors: list[str]) -> None:
    value = mapping.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{key} must be a list of strings")
        return
    missing = [item for item in expected if item not in value]
    extra = [item for item in value if item not in expected]
    if missing:
        errors.append(f"{key} missing: {', '.join(missing)}")
    if extra:
        errors.append(f"{key} has undeclared entries: {', '.join(extra)}")


def validate_document(document: Any) -> list[str]:
    """Return a list of contract violations. An empty list means the draft is valid."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["operational contract must be a mapping"]
    for key in ("version", "status", "q_h1", "h3", "p_q_boundary"):
        if key not in document:
            errors.append(f"missing key: {key}")
    if document.get("version") != "0.1":
        errors.append("version must be 0.1")
    if document.get("status") != "DRAFT":
        errors.append("status must be DRAFT")

    q_h1 = _require_mapping(document, "q_h1", errors)
    if q_h1 is not None:
        if q_h1.get("role") != "DIAGNOSTIC_TRANSFER":
            errors.append(
                f"q_h1.role must be DIAGNOSTIC_TRANSFER, found {q_h1.get('role')!r}"
            )
        if q_h1.get("ground_truth_available") is not False:
            errors.append("q_h1.ground_truth_available must be false")
        if q_h1.get("verifies_mechanism_identity") is not False:
            errors.append("q_h1.verifies_mechanism_identity must be false")
        _require_string_list(q_h1, "evaluates", Q_EVALUATES, errors)
        if q_h1.get("thresholds_status") != "DEFERRED":
            errors.append("q_h1.thresholds_status must be DEFERRED")
        errors.extend(_numeric_paths(q_h1, "q_h1"))

    h3 = _require_mapping(document, "h3", errors)
    if h3 is not None:
        if h3.get("role") != "REPRESENTATION_INVARIANCE":
            errors.append(
                f"h3.role must be REPRESENTATION_INVARIANCE, found {h3.get('role')!r}"
            )
        _require_string_list(h3, "transformation_family", H3_FAMILIES, errors)
        _require_string_list(h3, "primary_invariant", H3_INVARIANTS, errors)
        if h3.get("thresholds_status") != "DEFERRED":
            errors.append("h3.thresholds_status must be DEFERRED")
        errors.extend(_numeric_paths(h3, "h3"))

    boundary = _require_mapping(document, "p_q_boundary", errors)
    if boundary is not None:
        if boundary.get("P_ground_truth") is not True:
            errors.append("p_q_boundary.P_ground_truth must be true")
        if boundary.get("Q_ground_truth") is not False:
            errors.append("p_q_boundary.Q_ground_truth must be false")
        if boundary.get("q_h1_is_mechanism_verification") is not False:
            errors.append("p_q_boundary.q_h1_is_mechanism_verification must be false")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    default = Path(__file__).resolve().parents[1] / "configs" / "mrsm_operational_contract_v0.1.yaml"
    path = Path(args[0]) if args else default
    if not path.is_file():
        print(f"operational contract file is missing: {path}", file=sys.stderr)
        return 1
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = validate_document(document)
    if errors:
        for item in errors:
            print(item, file=sys.stderr)
        return 1
    print("operational contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
