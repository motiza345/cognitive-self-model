"""Qwen bridge diagnostic. Not an MRSM gate and not a mechanism discovery."""

from __future__ import annotations

import hashlib
from typing import Any

FLOOR = 1e-4
SIGN_FLOOR = 1e-8
MATCH_TOLERANCE = 1e-4
PINNED_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
RECORDED_LAYER = 23
RECORDED_DIRECTION_SHA = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"

NEXT = {
    "INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED": (
        "Do not modify the Self-Model or rerun MRSM. The next single step is a new "
        "preregistered decision to report each frozen regime separately, keeping every "
        "regime, before any architectural change."
    ),
    "SETUP_MISMATCH_BOTTLENECK": (
        "Do not modify the Self-Model. The next single step is a preregistered record of "
        "the setup difference that removed the historical signal, before any new MRSM run."
    ),
    "INTERVENTION_BOTTLENECK_CONFIRMED": (
        "Do not modify the Self-Model. The historical contrast was not recovered as a "
        "separable MRSM-metric signal. A new decision is required before any intervention change."
    ),
    "REPRESENTATION_BOTTLENECK_MORE_LIKELY": (
        "Do not add a representation in this diagnostic. This label is unused unless the "
        "margin observation loses a signal that the recorded residual contrast still has."
    ),
    "MULTIPLE_BOTTLENECKS": (
        "Do not modify the Self-Model. Separate the historical site from the cancelling "
        "slots in a new preregistered diagnostic before any repair."
    ),
    "INCONCLUSIVE": (
        "Do not modify the Self-Model, the thresholds, or the Qwen revision. "
        "Do not start a repair from this audit."
    ),
}


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_identity(checks: dict[str, str]) -> str:
    """checks values are yes, no, or unknown. No ranking."""
    revision = checks.get("model_revision", "unknown")
    family = checks.get("model_family", "unknown")
    if revision == "no":
        if family == "yes":
            return "SAME_FAMILY_DIFFERENT_REVISION"
        return "NOT_COMPARABLE"
    keys = (
        "model_revision",
        "tokenizer",
        "dtype",
        "activation_extraction",
        "indexing",
        "prompt_distribution",
        "intervention_location",
        "intervention_type",
        "intervention_magnitude",
    )
    if revision != "yes":
        return "NOT_COMPARABLE"
    if any(checks.get(key) == "no" for key in keys):
        return "SAME_MODEL_DIFFERENT_SETUP"
    if all(checks.get(key) == "yes" for key in keys):
        return "IDENTICAL_SETUP"
    return "NOT_COMPARABLE"


def select_bridge(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Ordered criteria from the phase statement. Stable, no effect-size ranking."""
    eligible = []
    for record in records:
        if not record.get("direct_causal_intervention"):
            continue
        if not record.get("negative_control_present"):
            continue
        if record.get("effect_size") in (None, "UNKNOWN"):
            continue
        if not record.get("reproducible_code"):
            continue
        eligible.append(record)
    if not eligible:
        raise RuntimeError("no historical experiment met the bridge criteria")
    eligible.sort(key=lambda record: (int(record.get("external_dependency_count", 99)), record["experiment_id"]))
    chosen = eligible[0]
    return {
        "experiment_id": chosen["experiment_id"],
        "rule": [
            "direct causal intervention",
            "negative or null control present",
            "reported effect size",
            "reproducible code",
            "least external dependency, then experiment id",
        ],
        "eligible_ids": [record["experiment_id"] for record in eligible],
    }


def is_context_cancellation(regime_means: dict[str, float], pooled: float) -> bool:
    opposing = []
    for mean in regime_means.values():
        if abs(float(mean)) <= FLOOR:
            continue
        if abs(float(mean)) <= SIGN_FLOOR:
            continue
        opposing.append(float(mean))
    signs = {1 if value > 0 else -1 for value in opposing}
    if len(signs) < 2:
        return False
    return abs(float(pooled)) < min(abs(value) for value in opposing)


def decide(
    *,
    historical_reproduced: bool,
    failure_locus: str | None,
    mrsm_signal_present: bool,
    context_cancellation: bool,
) -> dict[str, str]:
    if not historical_reproduced:
        return {
            "bridge_result": "HISTORICAL_RESULT_NOT_REPRODUCIBLE",
            "failure_locus": failure_locus or "UNKNOWN",
            "primary_diagnosis": "INCONCLUSIVE",
            "next_intervention": NEXT["INCONCLUSIVE"],
        }
    if not mrsm_signal_present:
        label = "SETUP_MISMATCH_BOTTLENECK"
        return {
            "bridge_result": "BRIDGE_SIGNAL_LOST_IN_MRSM_SETUP",
            "failure_locus": None,
            "primary_diagnosis": label,
            "next_intervention": NEXT[label],
        }
    if context_cancellation:
        label = "INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED"
        return {
            "bridge_result": "BRIDGE_SIGNAL_PRESENT_BUT_CONTEXT_DEPENDENT",
            "failure_locus": None,
            "primary_diagnosis": label,
            "next_intervention": NEXT[label],
        }
    return {
        "bridge_result": "BRIDGE_SIGNAL_PRESERVED",
        "failure_locus": None,
        "primary_diagnosis": "INCONCLUSIVE",
        "next_intervention": NEXT["INCONCLUSIVE"],
        "note": (
            "The historical contrast is present in the MRSM metric and no context cancellation "
            "was found. INTERVENTION_BOTTLENECK is not confirmed, and none of the other bottleneck "
            "labels is identified."
        ),
    }
