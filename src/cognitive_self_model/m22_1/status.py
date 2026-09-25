"""Status rules frozen in the M22.1 config. Directionality is not a gate."""

from __future__ import annotations

from typing import Any


def same_sign_and_identifiable(left: float, right: float, floor: float) -> bool:
    if abs(float(left)) <= float(floor) or abs(float(right)) <= float(floor):
        return False
    return (float(left) > 0.0) == (float(right) > 0.0)


def control_separated(target_mean: float, control_mean: float, paired_difference_mean: float, floor: float) -> bool:
    return abs(float(target_mean)) > abs(float(control_mean)) and abs(float(paired_difference_mean)) > float(floor)


def classify_status(evidence: dict[str, Any]) -> str:
    """Return exactly one of REJECTED, CANDIDATE, VALIDATED_FOR_M22."""
    fundamental = [
        not bool(evidence["null_pass"]),
        not bool(evidence["hook_integrity_pass"]),
        not bool(evidence["split_integrity_pass"]),
        not bool(evidence["leakage_pass"]),
        not bool(evidence["definition_reproducible"]),
        not bool(evidence["scores_finite"]),
    ]
    if any(fundamental):
        return "REJECTED"
    floor = float(evidence["floor"])
    discovery_ok = abs(float(evidence["discovery_mean"])) > floor
    validation_ok = same_sign_and_identifiable(
        float(evidence["discovery_mean"]),
        float(evidence["validation_mean"]),
        floor,
    )
    if not discovery_ok or not validation_ok:
        return "REJECTED"
    validation_control = control_separated(
        float(evidence["validation_mean"]),
        float(evidence["validation_control_mean"]),
        float(evidence["validation_paired_difference_mean"]),
        floor,
    )
    replication_ok = same_sign_and_identifiable(
        float(evidence["validation_mean"]),
        float(evidence["replication_mean"]),
        floor,
    )
    replication_control = control_separated(
        float(evidence["replication_mean"]),
        float(evidence["replication_control_mean"]),
        float(evidence["replication_paired_difference_mean"]),
        floor,
    )
    if not bool(evidence["revision_pinned"]) or not validation_control or not replication_ok or not replication_control:
        return "CANDIDATE"
    return "VALIDATED_FOR_M22"
