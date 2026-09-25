"""Frozen descriptive label. Thresholds come from the pre-registered config."""

from __future__ import annotations

from .metrics import advantage_destroyed, beats


def _mean(summary: dict, method: str) -> float:
    return float(summary["blocked_mae_mean"][method])


def predictive_on_split(summary: dict, config: dict) -> bool:
    reduction = float(config["min_relative_mae_reduction"])
    floor = float(config["baseline_mae_floor"])
    fraction = float(config["permutation_advantage_remaining_max_fraction"])
    rank = _mean(summary, "B4_rank1")
    global_mean = _mean(summary, "B0_global_mean")
    direction_mean = _mean(summary, "B2_direction_mean")
    if summary["seeds_beating_both_baselines"] < int(config["min_seeds_beating_both_baselines"]):
        return False
    if not beats(rank, global_mean, reduction, floor) or not beats(rank, direction_mean, reduction, floor):
        return False
    for control in ("cell_permutation", "global_permutation"):
        permuted = summary["control_mae_mean"][control]
        if not advantage_destroyed(global_mean, rank, permuted["B0_global_mean"], permuted["B4_rank1"], fraction):
            return False
        if not advantage_destroyed(direction_mean, rank, permuted["B2_direction_mean"], permuted["B4_rank1"], fraction):
            return False
    return True


def calibrated_on_split(summary: dict, config: dict) -> bool:
    reduction = float(config["min_relative_mae_reduction"])
    floor = float(config["baseline_mae_floor"])
    fraction = float(config["permutation_advantage_remaining_max_fraction"])
    calibration = summary["calibration"]
    rank = float(calibration["rank1_mae"])
    constant = float(calibration["constant_mae"])
    global_mean = float(calibration["global_mae"])
    if calibration["directions_not_worse_than_constant"] < int(config["calibration_min_directions_not_worse_than_constant"]):
        return False
    if not beats(rank, constant, reduction, floor) or not beats(rank, global_mean, reduction, floor):
        return False
    return advantage_destroyed(
        constant, rank, calibration["permuted_constant_mae"], calibration["permuted_rank1_mae"], fraction
    ) and advantage_destroyed(
        global_mean, rank, calibration["permuted_global_mae"], calibration["permuted_rank1_mae"], fraction
    )


def regime_degrades(summary: dict, config: dict) -> bool:
    blocked = _mean(summary, "B4_rank1")
    return float(summary["leave_one_regime_mae"]) > float(config["regime_degradation_mae_ratio"]) * blocked


def assign_label(validation: dict, replication: dict, leakage_pass: bool, config: dict) -> dict:
    predictive = leakage_pass and predictive_on_split(validation, config) and predictive_on_split(replication, config)
    context = predictive and regime_degrades(validation, config) and regime_degrades(replication, config)
    calibrated = (
        leakage_pass
        and not predictive
        and calibrated_on_split(validation, config)
        and calibrated_on_split(replication, config)
    )
    if not leakage_pass or not (context or predictive or calibrated):
        label = "INSUFFICIENT_EVIDENCE"
    elif context:
        label = "CONTEXT_DEPENDENT_STRUCTURE"
    elif predictive:
        label = "PREDICTIVE_RESPONSE_STRUCTURE"
    else:
        label = "CALIBRATED_INTERVENTION_RESPONSE"
    return {
        "label": label,
        "leakage_pass": bool(leakage_pass),
        "predictive_on_both_splits": bool(predictive),
        "regime_degrades_on_both_splits": bool(context),
        "calibrated_on_both_splits": bool(calibrated),
        "discovery_not_used": True,
        "secondary_svd_not_used": True,
    }
