"""Diagnostic localization of the frozen Q failure.

Labels in this module are not MRSM gates. The classification rules are fixed
in this file before the Qwen forwards for this phase. They are not a license
to edit thresholds, the Q revision, or the frozen run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.mrsm import HEADS
from src.mrsm.baselines import predictions as baseline_predictions
from src.mrsm.evaluators import _sign
from src.mrsm.self_model import SelfModelCore, pair_name

TRACKED_EDGE = ("L0H0", "L0H1")
TRACKED_ABSTRACTION = "L0H0->L0H1"
SIGN_FLOOR = 1e-8
IDENTIFIABILITY_FLOOR = 1e-4
M22_FLOOR_ABS = 1e-4
M22_FLOOR_MULT = 10.0
CLEAR_SIGN_CONSISTENCY = 0.75
CLEAR_MIN_SEPARABLE = 4
H2_RELATIVE_BAR = 0.20
MIN_BUCKET_N = 4
PARTITION_NAMES = ("D1", "D2", "D3", "D4")
SCOPE = "Q:m22_1_residual:diagnostic"

FROZEN_Q_SELF_MAE = 0.027457769270296452
FROZEN_Q_B2_MAE = 0.02828113017258821
FROZEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
FROZEN_P_FREEZE = "2cccafd047044332828eaf602b69f4267852cba2"
FROZEN_HOLDOUT_HASH = "c6b3f1b4b144a63d0f9665a6d9a06e6e0e27423b1a29533b8faab36da43fd8de"
FROZEN_HOLDOUT_SEED = 424242
FROZEN_HOLDOUT_N = 512

NEXT_INTERVENTION = {
    "DISCOVERY_BOTTLENECK": (
        "improve discovery stability while preserving the existing Self-Model and MRSM contract."
    ),
    "INTERVENTION_BOTTLENECK": (
        "Do not modify the Self-Model or the MRSM contract. "
        "The next intervention is a new preregistered decision on whether these Q residual slots "
        "produce a separable causal signal. Do not rerun the frozen MRSM gates to chase a pass."
    ),
    "REPRESENTATION_BOTTLENECK": "add a minimal causal representation layer.",
    "PREDICTION_BOTTLENECK": "modify the Self-Model predictor.",
    "ABSTRACTION_BOTTLENECK": (
        "connect the mechanism abstraction to prediction. The current predict() path does not read it."
    ),
    "MULTIPLE_BOTTLENECKS": (
        "Do not modify the Self-Model. Separate the failed discovery partitions from the weak "
        "intervention signal in a new preregistered diagnostic before any repair."
    ),
    "INCONCLUSIVE": (
        "Do not modify the Self-Model, the thresholds, or the Q artifact. Do not start a repair from this audit."
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_partitions(prompt_ids: list[str]) -> dict[str, list[str]]:
    """Disjoint round-robin on sorted prompt ids. Not chosen from effects."""
    ordered = sorted(prompt_ids)
    bins = {name: [] for name in PARTITION_NAMES}
    for index, prompt_id in enumerate(ordered):
        bins[PARTITION_NAMES[index % 4]].append(prompt_id)
    return bins


def partitions_are_regime_confounded(records: list[dict[str, str]], bins: dict[str, list[str]]) -> bool:
    """True when a partition is a single regime or regimes are not shared across bins."""
    regime = {row["prompt_id"]: row["regime_id"] for row in records}
    signatures = []
    for name in PARTITION_NAMES:
        signatures.append(tuple(sorted({regime[prompt_id] for prompt_id in bins[name]})))
    if any(len(signature) <= 1 for signature in signatures):
        return True
    return len(set(signatures)) > 1


def _median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def discovery_stability(rows: list[dict[str, Any]]) -> str:
    identified = [row for row in rows if row.get("status") == "IDENTIFIED"]
    if len(identified) < 2:
        return "UNIDENTIFIABLE"
    names = [str(row["candidate"]) for row in identified]
    if len(set(names)) == 1:
        only = names[0]
        if len(identified) == len(rows) and only == TRACKED_ABSTRACTION:
            return "STABLE"
        return "PARTIALLY_STABLE"
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    top = max(counts.values())
    if top > len(identified) / 2.0:
        return "PARTIALLY_STABLE"
    return "UNSTABLE"


def classify_signal(rows: list[dict[str, Any]]) -> str:
    separable = [row for row in rows if row.get("separable")]
    if not separable:
        return "SIGNAL_NOT_IDENTIFIABLE"
    conflicts = sum(1 for row in separable if row.get("regime_conflict"))
    if conflicts / len(separable) >= 0.5:
        return "SIGNAL_CONTEXT_DEPENDENT"
    median_consistency = _median([float(row["sign_consistency"]) for row in separable])
    if len(separable) >= CLEAR_MIN_SEPARABLE and median_consistency >= CLEAR_SIGN_CONSISTENCY:
        return "SIGNAL_CLEAR"
    return "SIGNAL_WEAK"


def diagnose(
    *,
    discovery: str,
    signal: str,
    representation: str,
    oracle: str,
    ceiling_near: bool,
    partitions_confounded: bool,
) -> tuple[str, str]:
    discovery_bad = discovery in {"UNSTABLE", "PARTIALLY_STABLE", "UNIDENTIFIABLE"}
    signal_bad = signal in {"SIGNAL_WEAK", "SIGNAL_NOT_IDENTIFIABLE", "SIGNAL_CONTEXT_DEPENDENT"}
    discovery_isolated = discovery_bad and not partitions_confounded
    if signal_bad and discovery_isolated:
        label = "MULTIPLE_BOTTLENECKS"
    elif signal_bad:
        label = "INTERVENTION_BOTTLENECK"
    elif discovery_isolated:
        label = "DISCOVERY_BOTTLENECK"
    elif discovery_bad and partitions_confounded:
        label = "INCONCLUSIVE"
    elif representation == "REPRESENTATION_BOTTLENECK":
        label = "REPRESENTATION_BOTTLENECK"
    elif oracle == "PREDICTION_OR_ABSTRACTION_BOTTLENECK":
        label = "ABSTRACTION_BOTTLENECK"
    elif ceiling_near:
        label = "PREDICTION_BOTTLENECK"
    else:
        label = "INCONCLUSIVE"
    return label, NEXT_INTERVENTION[label]


def subset_means(per_example: dict[str, list[float]], indices: list[int]) -> tuple[dict[str, float], dict[str, float]]:
    if not indices:
        raise ValueError("empty partition")
    singles: dict[str, float] = {}
    joints: dict[str, float] = {}
    for key, values in per_example.items():
        chosen = [float(values[index]) for index in indices]
        mean = float(np.mean(chosen))
        kind, name = key.split(":", 1)
        if kind == "single":
            singles[name] = mean
        elif kind == "pair":
            joints[name] = mean
        else:
            raise KeyError(key)
    return singles, joints


def fit_core(singles: dict[str, float], joints: dict[str, float], *, min_abs: float, min_gap: float) -> SelfModelCore:
    core = SelfModelCore(SCOPE)
    core.update(singles, joints, ["qf-discovery"], min_abs_interaction=min_abs, min_relative_gap=min_gap)
    core.freeze()
    return core


def _member_support(effects: dict[str, float], members: tuple[str, str]) -> tuple[int, int]:
    support = 0
    contradiction = 0
    for name in members:
        sign = _sign(float(effects[name]), SIGN_FLOOR)
        if sign > 0:
            support += 1
        else:
            contradiction += 1
    return support, contradiction


def partition_row(name: str, core: SelfModelCore, prompt_ids: list[str]) -> dict[str, Any]:
    mechanism = core.mechanism or {}
    identified = bool(core.identifiable) and mechanism.get("status") == "IDENTIFIED"
    if not identified:
        return {
            "partition": name,
            "prompt_ids": list(prompt_ids),
            "n": len(prompt_ids),
            "candidate": None,
            "source": None,
            "target": None,
            "effect": None,
            "sign": None,
            "support": 0,
            "contradiction": 0,
            "confidence": None if mechanism is None else mechanism.get("gap"),
            "scope": SCOPE,
            "status": "NOT_IDENTIFIABLE",
        }
    ordered = tuple(mechanism["ordered"])
    support, contradiction = _member_support(core.effects, ordered)
    interaction = float(mechanism["interaction"])
    return {
        "partition": name,
        "prompt_ids": list(prompt_ids),
        "n": len(prompt_ids),
        "candidate": mechanism["abstraction"],
        "source": ordered[0],
        "target": ordered[1],
        "effect": interaction,
        "sign": _sign(interaction, SIGN_FLOOR),
        "support": support,
        "contradiction": contradiction,
        "confidence": float(mechanism["gap"]),
        "scope": SCOPE,
        "status": "IDENTIFIED",
    }


def tracked_edge_row(name: str, core: SelfModelCore) -> dict[str, Any]:
    left, right = TRACKED_EDGE
    key = pair_name(left, right)
    interaction = float(core.interactions[key])
    support, contradiction = _member_support(core.effects, TRACKED_EDGE)
    competing = sum(1 for other, value in core.interactions.items() if other != key and abs(value) > abs(interaction))
    return {
        "partition": name,
        "candidate": TRACKED_ABSTRACTION,
        "source": left,
        "target": right,
        "effect": interaction,
        "member_effects": {left: float(core.effects[left]), right: float(core.effects[right])},
        "sign": _sign(interaction, SIGN_FLOOR),
        "support": support,
        "contradiction": contradiction,
        "competing_pairs_with_larger_abs_interaction": competing,
        "selected": bool(core.mechanism and core.mechanism.get("abstraction") == TRACKED_ABSTRACTION),
    }


def effect_floor(null_abs_max: float) -> float:
    return float(max(M22_FLOOR_ABS, M22_FLOOR_MULT * float(null_abs_max)))


def intervention_signal_row(
    name: str,
    drops: list[float],
    null_drops: list[float],
    regimes: list[str],
    floor: float,
) -> dict[str, Any]:
    values = np.asarray(drops, dtype=np.float64)
    nulls = np.asarray(null_drops, dtype=np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    sign = _sign(mean, SIGN_FLOOR)
    active = [float(value) for value in values if abs(float(value)) > SIGN_FLOOR]
    if active and sign != 0:
        sign_consistency = float(np.mean([_sign(value, SIGN_FLOOR) == sign for value in active]))
    elif sign == 0:
        sign_consistency = 0.0
    else:
        sign_consistency = 0.0
    regime_means: dict[str, list[float]] = {}
    for regime, value in zip(regimes, values.tolist()):
        regime_means.setdefault(regime, []).append(float(value))
    regime_summary = {regime: float(np.mean(items)) for regime, items in sorted(regime_means.items())}
    strong = []
    for regime, regime_mean in regime_summary.items():
        if abs(regime_mean) > floor and _sign(regime_mean, SIGN_FLOOR) != 0:
            if len(regime_means[regime]) >= 2:
                strong.append(_sign(regime_mean, SIGN_FLOOR))
    regime_conflict = len(set(strong)) > 1
    snr = None if std <= SIGN_FLOOR else abs(mean) / std
    return {
        "intervention": name,
        "raw_effect": mean,
        "normalized_effect": snr,
        "variance": float(np.var(values)),
        "std": std,
        "signal_to_noise": snr,
        "sign": sign,
        "sign_consistency": sign_consistency,
        "effect_distribution": {
            "n": int(values.size),
            "mean": mean,
            "std": std,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        },
        "null_distribution": {
            "n": int(nulls.size),
            "mean": float(np.mean(nulls)),
            "std": float(np.std(nulls)),
            "max_abs": float(np.max(np.abs(nulls))),
        },
        "regime_means": regime_summary,
        "regime_conflict": regime_conflict,
        "separable": abs(mean) > floor,
        "context_dependence": regime_summary,
    }


def mae_maps(predicted: dict[str, float], actual: dict[str, float]) -> float:
    keys = sorted(actual)
    errors = [float(predicted[key]) - float(actual[key]) for key in keys]
    return float(np.mean(np.abs(errors)))


def sign_accuracy(predicted: dict[str, float], actual: dict[str, float]) -> float:
    keys = sorted(actual)
    hits = [_sign(float(predicted[key]), SIGN_FLOOR) == _sign(float(actual[key]), SIGN_FLOOR) for key in keys]
    return float(np.mean(hits))


def relative_reduction(baseline_mae: float, self_mae: float) -> float | None:
    if baseline_mae == 0.0:
        return None
    return (baseline_mae - self_mae) / baseline_mae


def scores_from_saved(records: list[dict[str, Any]], b2: dict[str, float]) -> dict[str, Any]:
    actual = {row["intervention_id"]: float(row["actual_outcome"]) for row in records}
    self_pred = {row["intervention_id"]: float(row["prediction"]) for row in records}
    self_mae = mae_maps(self_pred, actual)
    b2_mae = mae_maps(b2, actual)
    if abs(self_mae - FROZEN_Q_SELF_MAE) > 1e-9 or abs(b2_mae - FROZEN_Q_B2_MAE) > 1e-9:
        raise RuntimeError("saved Q scores do not match the frozen MRSM result")
    return {
        "self_mae": self_mae,
        "b2_mae": b2_mae,
        "relative_reduction": relative_reduction(b2_mae, self_mae),
        "sign_accuracy_self": sign_accuracy(self_pred, actual),
        "sign_accuracy_b2": sign_accuracy(b2, actual),
        "n_interventions": len(actual),
        "diagnostic_reuse": True,
        "source": "artifacts/mrsm/q_run_001",
    }


def loo_scores(
    per_example: dict[str, list[float]],
    *,
    min_abs: float,
    min_gap: float,
) -> dict[str, Any]:
    width = len(next(iter(per_example.values())))
    self_maes = []
    b2_maes = []
    self_signs = []
    for held in range(width):
        train = [index for index in range(width) if index != held]
        singles, joints = subset_means(per_example, train)
        core = fit_core(singles, joints, min_abs=min_abs, min_gap=min_gap)
        actual_singles, actual_joints = subset_means(per_example, [held])
        actual = {f"single:{name}": value for name, value in actual_singles.items()}
        actual.update({f"pair:{name}": value for name, value in actual_joints.items()})
        self_pred = {key: float(core.predict(key)) for key in actual}
        b2_pred = baseline_predictions(singles)["B2"]
        self_maes.append(mae_maps(self_pred, actual))
        b2_maes.append(mae_maps(b2_pred, actual))
        self_signs.append(sign_accuracy(self_pred, actual))
    return {
        "protocol": "leave_one_discovery_prompt_out",
        "holdout_used": False,
        "folds": width,
        "self_mae_mean": float(np.mean(self_maes)),
        "b2_mae_mean": float(np.mean(b2_maes)),
        "self_sign_accuracy_mean": float(np.mean(self_signs)),
        "feature_dimensionality_self": 36,
        "feature_dimensionality_b2": 8,
        "train_prompts_per_fold": width - 1,
        "test_prompts_per_fold": 1,
    }


def h2_buckets(records: list[dict[str, Any]], b2: dict[str, float]) -> dict[str, Any]:
    """Decompose the saved validation residuals. Regime and OOD are absent there."""
    rows = []
    for record in records:
        intervention_id = record["intervention_id"]
        actual = float(record["actual_outcome"])
        predicted = float(record["prediction"])
        rows.append({
            "intervention_id": intervention_id,
            "actual": actual,
            "self": predicted,
            "b2": float(b2[intervention_id]),
            "kind": intervention_id.split(":", 1)[0],
            "sign": _sign(actual, SIGN_FLOOR),
            "above_floor": abs(actual) > IDENTIFIABILITY_FLOOR,
            "heads": intervention_id.split(":", 1)[1].split("+"),
        })

    def pack(title: str, chosen: list[dict[str, Any]]) -> dict[str, Any]:
        if not chosen:
            return {"bucket": title, "n": 0, "informative": False}
        actual = {row["intervention_id"]: row["actual"] for row in chosen}
        self_pred = {row["intervention_id"]: row["self"] for row in chosen}
        b2_pred = {row["intervention_id"]: row["b2"] for row in chosen}
        self_mae = mae_maps(self_pred, actual)
        b2_mae = mae_maps(b2_pred, actual)
        return {
            "bucket": title,
            "n": len(chosen),
            "informative": len(chosen) >= MIN_BUCKET_N,
            "self_mae": self_mae,
            "b2_mae": b2_mae,
            "relative_reduction": relative_reduction(b2_mae, self_mae),
            "sign_accuracy": sign_accuracy(self_pred, actual),
        }

    buckets = [
        pack("type:single", [row for row in rows if row["kind"] == "single"]),
        pack("type:pair", [row for row in rows if row["kind"] == "pair"]),
        pack("sign:positive", [row for row in rows if row["sign"] > 0]),
        pack("sign:negative", [row for row in rows if row["sign"] < 0]),
        pack("sign:zero", [row for row in rows if row["sign"] == 0]),
        pack("magnitude:at_or_below_1e-4", [row for row in rows if not row["above_floor"]]),
        pack("magnitude:above_1e-4", [row for row in rows if row["above_floor"]]),
    ]
    for head in HEADS:
        buckets.append(pack(f"target:{head}", [row for row in rows if head in row["heads"]]))
    informative = [
        bucket for bucket in buckets
        if bucket["informative"] and not bucket["bucket"].startswith("target:") and bucket.get("relative_reduction") is not None
    ]
    reductions = [float(bucket["relative_reduction"]) for bucket in informative]
    if len(informative) < 2:
        label = "INCONCLUSIVE"
    elif any(value >= H2_RELATIVE_BAR for value in reductions) and any(value < H2_RELATIVE_BAR for value in reductions):
        label = "localized"
    elif all(value < H2_RELATIVE_BAR for value in reductions):
        label = "global"
    else:
        label = "global"
    return {
        "status": label,
        "diagnostic_reuse": True,
        "regime": {"status": "NOT_AVAILABLE", "reason": "q_run_001 stores validation means, not per-regime residuals"},
        "in_distribution_vs_ood": {"status": "NOT_AVAILABLE", "reason": "the frozen Q split has no OOD bucket"},
        "buckets": buckets,
        "decision_buckets": [bucket["bucket"] for bucket in informative],
    }


def partition_b2_stability(
    per_example: dict[str, list[float]],
    prompt_ids: list[str],
    bins: dict[str, list[str]],
) -> list[dict[str, Any]]:
    index = {prompt_id: position for position, prompt_id in enumerate(prompt_ids)}
    rows = []
    for name in PARTITION_NAMES:
        train_ids = bins[name]
        test_ids = [prompt_id for prompt_id in prompt_ids if prompt_id not in train_ids]
        train_index = [index[prompt_id] for prompt_id in train_ids]
        test_index = [index[prompt_id] for prompt_id in test_ids]
        singles, _joints = subset_means(per_example, train_index)
        actual_singles, actual_joints = subset_means(per_example, test_index)
        actual = {f"single:{head}": value for head, value in actual_singles.items()}
        actual.update({f"pair:{pair}": value for pair, value in actual_joints.items()})
        b2 = baseline_predictions(singles)["B2"]
        rows.append({
            "partition": name,
            "train_n": len(train_ids),
            "test_n": len(test_ids),
            "test": "complement_discovery_prompts",
            "holdout_used": False,
            "b2_mae": mae_maps(b2, actual),
        })
    return rows


def representation_audit(loo: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "INCONCLUSIVE",
        "reason": (
            "R2 is the same intervention-effect table already stored by SelfModelCore. "
            "R3 activation trajectories are not in the Q observation pipeline. "
            "R4 is not formed, because R3 is absent. No synthetic features were added."
        ),
        "families": {
            "R1": {
                "name": "current_mrsm_representation",
                "features": "discovery mean margin-drop for 8 singles and 28 pairs",
                "feature_dimensionality": 36,
                "sample_count_discovery_prompts": loo["folds"],
                "train_test_isolation": "leave-one-discovery-prompt-out; validation not used",
                "self_model_mae": loo["self_mae_mean"],
                "b2_linear_probe_mae": loo["b2_mae_mean"],
                "sign_accuracy": loo["self_sign_accuracy_mean"],
                "saved_validation_self_mae": saved["self_mae"],
                "saved_validation_b2_mae": saved["b2_mae"],
                "saved_validation_note": "diagnostic_reuse of q_run_001 means; not a new score",
            },
            "R2": {
                "status": "SAME_AS_R1",
                "reason": "SelfModelCore.predict is already the intervention-effect lookup. B2 is the additive probe on the 8 single effects.",
                "feature_dimensionality": 8,
                "b2_linear_probe_mae": loo["b2_mae_mean"],
                "self_model_mae": loo["self_mae_mean"],
            },
            "R3": {
                "status": "NOT_AVAILABLE",
                "reason": "run_q records logit-margin drops only. Residual trajectories were not part of the frozen Q observation and were not collected.",
            },
            "R4": {
                "status": "NOT_AVAILABLE",
                "reason": "R3 is absent, so a combined representation was not built.",
            },
        },
    }


def oracle_audit(saved: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ORACLE_INVALID",
        "interpretation": "INCONCLUSIVE",
        "candidate": TRACKED_ABSTRACTION,
        "reason": (
            "SelfModelCore.predict returns the stored margin-drop and does not read mechanism.abstraction. "
            "P_SPEC defines the planted P edge as an attention restriction on L0H0 and L1H1, not a numeric map "
            "from the Q slot label L0H0->L0H1 onto logit-margin drops. No repository function supplies that map. "
            "An oracle predictor was not invented, and holdout outcomes were not used to build one."
        ),
        "o0_mae": saved["self_mae"],
        "o1_mae": None,
        "o2_mae": None,
        "b2_mae": saved["b2_mae"],
        "o0_source": "diagnostic_reuse of q_run_001 self predictions",
        "holdout_used_to_build_oracle": False,
        "tuned_after_predictions": False,
    }


def ceiling_near(saved: dict[str, Any]) -> bool:
    reduction = saved["relative_reduction"]
    return reduction is not None and reduction < H2_RELATIVE_BAR


def build_payload(
    *,
    verification: dict[str, Any],
    reproducibility: dict[str, Any],
    qf1_rows: list[dict[str, Any]],
    tracked_rows: list[dict[str, Any]],
    confounded: bool,
    signal_rows: list[dict[str, Any]],
    signal_floor: float,
    loo: dict[str, Any],
    saved: dict[str, Any],
    buckets: dict[str, Any],
    b2_partitions: list[dict[str, Any]],
) -> dict[str, Any]:
    discovery = discovery_stability(qf1_rows)
    signal = classify_signal(signal_rows)
    representation = representation_audit(loo, saved)
    oracle = oracle_audit(saved)
    near = ceiling_near(saved)
    primary, nxt = diagnose(
        discovery=discovery,
        signal=signal,
        representation=representation["status"],
        oracle=oracle["interpretation"],
        ceiling_near=near,
        partitions_confounded=confounded,
    )
    qf6 = {
        "status": "NEAR_CEILING" if near else "BELOW_CEILING",
        "feature_dimensionality_b2": 8,
        "feature_dimensionality_self": 36,
        "effective_sample_count_discovery_prompts": loo["folds"],
        "train_split": "m22 discovery role, 6 prompts",
        "test_split_for_frozen_mae": "m22 validation role, diagnostic_reuse, 6 prompts aggregated to 36 intervention means",
        "b2_sees_information_unavailable_to_self_model": False,
        "b2_information": "8 discovery single-head mean effects; pair prediction is their sum",
        "self_information": "those 8 means plus 28 discovery joint means",
        "frozen_validation": saved,
        "discovery_complement_b2_mae": b2_partitions,
        "b2_mae_min": min(row["b2_mae"] for row in b2_partitions),
        "b2_mae_max": max(row["b2_mae"] for row in b2_partitions),
        "approximately_linear": (
            "On the saved validation means, B2 is within the frozen 20% band of the self-model. "
            "That is the preregistered additive probe, not a new fit."
        ),
    }
    matrix = [
        {"layer": "Discovery", "evidence": "QF-1", "status": discovery},
        {"layer": "Intervention", "evidence": "QF-2", "status": signal},
        {"layer": "Representation", "evidence": "QF-3", "status": representation["status"]},
        {"layer": "Prediction", "evidence": "QF-4", "status": oracle["status"]},
        {"layer": "H2 localization", "evidence": "QF-5", "status": buckets["status"]},
        {"layer": "Baseline ceiling", "evidence": "QF-6", "status": qf6["status"]},
    ]
    return {
        "scientific_status": "DIAGNOSTIC_ONLY",
        "phase_status": "COMPLETE",
        "mrsm_rerun": False,
        "frozen_result_modified": False,
        "verification": verification,
        "reproducibility": reproducibility,
        "qf1": {
            "label": discovery,
            "partition_regime_confounded": confounded,
            "support_definition": (
                "For an identified candidate, support counts member singles whose discovery-partition "
                "margin-drop is positive above 1e-8. Contradiction counts the other member singles. "
                "This follows the existing Q.H1 member-sign rule. It is not a new selector."
            ),
            "rows": qf1_rows,
            "tracked_edge": tracked_rows,
            "holdout_used": False,
        },
        "qf2": {
            "label": signal,
            "floor": signal_floor,
            "floor_rule": "max(1e-4, 10 * max abs alpha=0 drop), from configs/m22_1_preflight.json",
            "null": "same catalog slots at alpha 0, which is already on the M22.1 magnitude grid",
            "rows": signal_rows,
            "holdout_used": False,
            "split": "discovery",
        },
        "qf3": representation,
        "qf4": oracle,
        "qf5": buckets,
        "qf6": qf6,
        "matrix": matrix,
        "primary_bottleneck": primary,
        "next_intervention": nxt,
        "leakage": {
            "p_holdout_accessed": False,
            "planted_ground_truth_loaded": False,
            "q_validation_reexecuted": False,
            "diagnostic_reuse": ["QF-3 saved validation MAE", "QF-4 O0/B2 reference", "QF-5", "QF-6 frozen MAE"],
            "discovery_only_forwards": ["QF-1", "QF-2", "QF-3 leave-one-out", "QF-6 complement MAE"],
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(payload: dict[str, Any]) -> str:
    qf1 = payload["qf1"]
    qf2 = payload["qf2"]
    lines = [
        "# Q Failure Analysis",
        "",
        "## Frozen MRSM Result",
        "",
        "This file is a diagnostic follow-up. It does not replace `reports/MRSM_FINAL_RESULT.md`.",
        "",
        "Q gates, unchanged:",
        "",
        "H1 = FAIL",
        "H2 = FAIL",
        "H3 = PASS",
        "H4 = FAIL",
        "",
        f"P gates remain PASS. Leakage on P and Q remains PASS. DEC-010 remains `REDEFINE_SCALE`.",
        f"`p_freeze` remains `{FROZEN_P_FREEZE}`.",
        f"Q revision remains `{FROZEN_REVISION}`.",
        f"P holdout remains seed {FROZEN_HOLDOUT_SEED}, n {FROZEN_HOLDOUT_N}, hash `{FROZEN_HOLDOUT_HASH}`.",
        "No threshold, H3 transform, holdout id, or frozen artifact was edited.",
        "",
        "## QF-1 Discovery Stability",
        "",
        "Discovery prompts were split by sorted `prompt_id` round-robin into D1–D4.",
        "The official P holdout was not read. The Q validation split was not used to choose a candidate.",
        f"Partition design is regime-confounded: `{qf1['partition_regime_confounded']}`.",
        "D3 and D4 are single instruction prompts. D1 and D2 contain completion and syntax only.",
        "That imbalance is a property of the frozen ids. It was not rebalanced after measurement.",
        "",
        "partition | candidate | effect | sign | support | contradiction | status",
        "---|---|---|---|---|---|---",
    ]
    for row in qf1["rows"]:
        lines.append(
            " | ".join([
                str(row["partition"]),
                _fmt(row["candidate"]),
                _fmt(row["effect"]),
                _fmt(row["sign"]),
                _fmt(row["support"]),
                _fmt(row["contradiction"]),
                str(row["status"]),
            ])
        )
    lines.extend([
        "",
        "Tracked edge `L0H0 -> L0H1` on the same partitions:",
        "",
        "partition | effect | member L0H0 | member L0H1 | sign | support | contradiction | selected",
        "---|---|---|---|---|---|---|---",
    ])
    for row in qf1["tracked_edge"]:
        members = row["member_effects"]
        lines.append(
            " | ".join([
                str(row["partition"]),
                _fmt(row["effect"]),
                _fmt(members["L0H0"]),
                _fmt(members["L0H1"]),
                _fmt(row["sign"]),
                _fmt(row["support"]),
                _fmt(row["contradiction"]),
                str(row["selected"]),
            ])
        )
    lines.extend([
        "",
        "```text",
        "DISCOVERY_STABILITY:",
        qf1["label"],
        "```",
        "",
        "This label is diagnostic. It is not a new H1 result.",
        "",
        "## QF-2 Intervention Signal",
        "",
        "Real interventions are the eight frozen Q catalog slots at alpha 1.",
        "The null is the same slot at alpha 0, already present on the M22.1 magnitude grid.",
        "No new direction was sampled.",
        f"Floor = {_fmt(qf2['floor'])} by `{qf2['floor_rule']}`.",
        "Measurement split: discovery prompts only.",
        "",
        "intervention | raw effect | snr | sign consistency | separable | regime conflict",
        "---|---|---|---|---|---",
    ])
    for row in qf2["rows"]:
        lines.append(
            " | ".join([
                str(row["intervention"]),
                _fmt(row["raw_effect"]),
                _fmt(row["signal_to_noise"]),
                _fmt(row["sign_consistency"]),
                str(row["separable"]),
                str(row["regime_conflict"]),
            ])
        )
    lines.extend([
        "",
        "```text",
        qf2["label"],
        "```",
        "",
        "This classification is diagnostic. Separable means larger than the M22.1 null floor, not large enough to pass H2.",
        "",
        "## QF-3 Representation Sufficiency",
        "",
        f"Status: `{payload['qf3']['status']}`.",
        "",
        payload["qf3"]["reason"],
        "",
    ])
    r1 = payload["qf3"]["families"]["R1"]
    lines.extend([
        "R1 leave-one-discovery-prompt-out:",
        "",
        f"- Self-Model MAE: {_fmt(r1['self_model_mae'])}",
        f"- B2 linear probe MAE: {_fmt(r1['b2_linear_probe_mae'])}",
        f"- Self-Model sign accuracy: {_fmt(r1['sign_accuracy'])}",
        f"- Feature dimensionality: {r1['feature_dimensionality']}",
        f"- Discovery prompts: {r1['sample_count_discovery_prompts']}",
        f"- Train/test: {r1['train_test_isolation']}",
        "",
        "Saved validation means, marked diagnostic_reuse, are the frozen H2 numbers:",
        "",
        f"- Self-Model MAE: {_fmt(r1['saved_validation_self_mae'])}",
        f"- B2 MAE: {_fmt(r1['saved_validation_b2_mae'])}",
        "",
        "R2 status: `SAME_AS_R1`. R3 status: `NOT_AVAILABLE`. R4 status: `NOT_AVAILABLE`.",
        "",
        "## QF-4 Oracle Prediction",
        "",
        f"Status: `{payload['qf4']['status']}`.",
        f"Interpretation: `{payload['qf4']['interpretation']}`.",
        "",
        payload["qf4"]["reason"],
        "",
        "```text",
        f"O0 MAE = {_fmt(payload['qf4']['o0_mae'])}",
        "O1 MAE = null",
        "O2 MAE = null",
        f"B2 MAE = {_fmt(payload['qf4']['b2_mae'])}",
        "```",
        "",
        "O0 and B2 are the frozen validation scores, read back from `q_run_001`. They are not an oracle fit.",
        "",
        "## QF-5 H2 Decomposition",
        "",
        f"Status: `{payload['qf5']['status']}`.",
        "Source: saved `q_run_001` intervention means. Diagnostic reuse. The validation split was not executed again.",
        f"Regime: `{payload['qf5']['regime']['status']}` — {payload['qf5']['regime']['reason']}",
        f"In-distribution vs OOD: `{payload['qf5']['in_distribution_vs_ood']['status']}` — {payload['qf5']['in_distribution_vs_ood']['reason']}",
        "",
        "bucket | n | self MAE | B2 MAE | relative reduction | sign accuracy",
        "---|---|---|---|---|---",
    ])
    for bucket in payload["qf5"]["buckets"]:
        if bucket["n"] == 0:
            continue
        lines.append(
            " | ".join([
                str(bucket["bucket"]),
                str(bucket["n"]),
                _fmt(bucket.get("self_mae")),
                _fmt(bucket.get("b2_mae")),
                _fmt(bucket.get("relative_reduction")),
                _fmt(bucket.get("sign_accuracy")),
            ])
        )
    qf6 = payload["qf6"]
    saved = qf6["frozen_validation"]
    lines.extend([
        "",
        "The global/local label uses disjoint buckets with n >= 4: intervention type, actual sign, and the 1e-4 magnitude split.",
        "Overlapping target-head rows are descriptive and are not the decision.",
        "",
        "## QF-6 Baseline Ceiling",
        "",
        f"Status: `{qf6['status']}`.",
        "",
        f"- B2 feature dimensionality: {qf6['feature_dimensionality_b2']}",
        f"- Self-Model feature dimensionality: {qf6['feature_dimensionality_self']}",
        f"- Discovery prompts used to estimate effects: {qf6['effective_sample_count_discovery_prompts']}",
        f"- Train split: {qf6['train_split']}",
        f"- Frozen test split: {qf6['test_split_for_frozen_mae']}",
        f"- B2 sees information unavailable to the Self-Model: `{qf6['b2_sees_information_unavailable_to_self_model']}`",
        f"- Frozen self MAE: {_fmt(saved['self_mae'])}",
        f"- Frozen B2 MAE: {_fmt(saved['b2_mae'])}",
        f"- Relative reduction: {_fmt(saved['relative_reduction'])}",
        f"- B2 MAE on discovery complements: min {_fmt(qf6['b2_mae_min'])}, max {_fmt(qf6['b2_mae_max'])}",
        "",
        qf6["approximately_linear"],
        "B2 was not weakened and was not removed.",
        "",
        "## Failure Localization Matrix",
        "",
        "| Layer | Evidence | Status |",
        "| --- | --- | --- |",
    ])
    for row in payload["matrix"]:
        lines.append(f"| {row['layer']} | {row['evidence']} | {row['status']} |")
    lines.extend([
        "",
        "## Primary Bottleneck",
        "",
        payload["primary_bottleneck"],
        "",
        "## Evidence Supporting Diagnosis",
        "",
    ])
    lines.extend(_support_lines(payload))
    lines.extend(["", "## Evidence Against Diagnosis", ""])
    lines.extend(_against_lines(payload))
    lines.extend([
        "",
        "## What This Does NOT Prove",
        "",
        "This phase does not claim that MRSM succeeds on Qwen.",
        "It does not convert Q.H3 PASS into a recovered Qwen mechanism.",
        "It does not convert Q.H1 FAIL into a proof that mechanism discovery is impossible.",
        "It does not authorize a change to thresholds, the holdout, the pinned revision, or the frozen scores.",
        "A diagnostic label is not an MRSM gate result.",
        "",
        "## Single Recommended Next Intervention",
        "",
        f"Primary bottleneck: {payload['primary_bottleneck']}",
        "",
        "Next intervention:",
        payload["next_intervention"],
        "",
        "No repair and no architectural redesign was executed in this phase.",
        "",
        "## Reproducibility",
        "",
    ])
    env = payload["reproducibility"]
    for key in (
        "git_commit",
        "git_branch",
        "python",
        "torch",
        "transformer_lens",
        "numpy",
        "model_revision",
        "device",
        "weight_sha256",
        "prereg_sha256",
        "final_result_sha256",
    ):
        lines.append(f"- {key}: `{env.get(key)}`")
    lines.extend([
        "",
        "## Scientific Status",
        "",
        "DIAGNOSTIC_ONLY",
        "",
    ])
    return "\n".join(lines)


def _support_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"- QF-1 label is `{payload['qf1']['label']}`.",
        f"- QF-2 label is `{payload['qf2']['label']}` under the precommitted M22.1 null floor.",
        f"- QF-3 is `{payload['qf3']['status']}` because R3 and R4 were not available and R2 is not a new feature family.",
        "- QF-4 did not invent a numeric oracle. `predict()` does not read the mechanism label, so the H1 candidate is not an input to H2.",
        f"- QF-5 localization is `{payload['qf5']['status']}` on the saved validation means.",
        f"- QF-6 is `{payload['qf6']['status']}`: the frozen self MAE does not clear the 20% bar against B2, and B2 uses a subset of the self-model's numbers.",
    ]
    return lines


def _against_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        "- P.H1–H4 remain PASS. This audit does not reopen that result.",
        "- Q.H3 remains PASS and is the frozen inverse readout, not evidence for a Qwen circuit.",
        "- D1–D4 are regime-confounded by the frozen prompt ids, so a discovery label from those bins is not a clean estimate of sampler noise.",
        "- The signal floor is the M22.1 numerical floor. Clearing it does not mean the effect is large enough for H2.",
        "- Six discovery prompts and six validation prompts bound every comparison. Regime means use two prompts.",
        "- QF-5 cannot see regime or OOD structure, because those fields were not stored in `q_run_001`.",
    ]
    if payload["qf2"]["label"] == "SIGNAL_CLEAR":
        lines.append(
            "- SIGNAL_CLEAR means separable from alpha 0 at the M22.1 floor. It is not the H2 20% bar."
        )
    return lines


def write_manifest(directory: Path) -> None:
    lines = []
    for path in sorted(directory.glob("*.json")):
        lines.append(f"{file_sha256(path)}  {path.name}")
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
