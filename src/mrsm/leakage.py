"""Minimum leakage audit. A failure stops scientific scoring."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

_RUNTIME_FILES = (
    "self_model.py",
    "baselines.py",
    "transforms.py",
    "interventions.py",
)
_FORBIDDEN_SOURCE = (
    "planted_ground_truth",
    "planted_a",
    "planted_b",
)


def audit_sources(package_dir: Path | None = None) -> dict[str, Any]:
    root = package_dir if package_dir is not None else Path(__file__).resolve().parent
    offenders: list[str] = []
    for name in _RUNTIME_FILES:
        path = root / name
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for token in _FORBIDDEN_SOURCE:
                    if token in node.value:
                        offenders.append(f"{name}: {token}")
    return {"pass": len(offenders) == 0, "offenders": offenders}


def audit_state(state: dict[str, Any], fit_inputs: dict[str, Any], manifest_ids: list[str]) -> dict[str, Any]:
    failures: list[str] = []
    blob = repr(state)
    for token in ("ground_truth", "planted_a", "planted_b", "mechanism_label"):
        if token in blob:
            failures.append(f"self-model state contains {token}")
    for token in ("ground_truth", "planted_a", "planted_b", "holdout_outcome"):
        if token in fit_inputs:
            failures.append(f"fit input contains {token}")
    effects = fit_inputs.get("head_effects", {})
    if set(effects) != set(fit_inputs.get("expected_heads", effects)):
        failures.append("discovery head set mismatch")
    discovery_ids = set(fit_inputs.get("discovery_ids", []))
    overlap = discovery_ids.intersection(manifest_ids)
    if overlap:
        failures.append(f"holdout ids present in discovery data: {sorted(overlap)[:5]}")
    if fit_inputs.get("holdout_outcomes_present"):
        failures.append("holdout outcomes were present at fit time")
    return {"pass": len(failures) == 0, "failures": failures}


def audit_predictions(predictions: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if predictions.get("phase") != "before_outcome":
        failures.append("predictions were not marked before_outcome")
    for row in predictions.get("rows", []):
        if row.get("actual_outcome") is not None:
            failures.append(f"actual outcome present in prediction row {row.get('intervention_id')}")
            break
    return {"pass": len(failures) == 0, "failures": failures}
