"""Diagnostic readout of the frozen M23 artifacts.

Does not rerun Qwen, does not rewrite M23 results, and does not add a threshold.
Paired intervals reuse the M23 bootstrap: 5000 draws, seed 23001.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.belief import belief_from_observations, update_belief
from src.cognitive_self_model.m23.stats import paired_mean_ci

DISCOVERY_MEAN = 0.028899987538655598
DISCOVERY_SD = 0.003592417625678755
UPDATED_MEAN = 0.02854486306508382
N_DISCOVERY = 6


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "reports" / "m23_raw" / name).read_text(encoding="utf-8"))


def _rows(block: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(block, key=lambda row: row["prompt_id"])


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    center = _mean(values)
    return float(math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)))


def _distribution(values: list[float]) -> dict[str, Any]:
    sd = _sample_sd(values)
    center = _mean(values)
    return {
        "n": len(values),
        "mean": center,
        "sample_sd": sd,
        "coefficient_of_variation": None if sd is None or center == 0.0 else sd / abs(center),
        "min": float(min(values)),
        "max": float(max(values)),
        "range": float(max(values) - min(values)),
        "sign_positive": sum(1 for value in values if value > 0.0),
        "sign_negative": sum(1 for value in values if value < 0.0),
        "sign_zero": sum(1 for value in values if value == 0.0),
    }


def _grouped(rows: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(float(row["observed_delta"]))
    return groups


def _variance_split(groups: dict[str, list[float]]) -> dict[str, Any]:
    flat = [value for values in groups.values() for value in values]
    overall = _mean(flat)
    ss_between = 0.0
    ss_within = 0.0
    for values in groups.values():
        group_mean = _mean(values)
        ss_between += len(values) * (group_mean - overall) ** 2
        ss_within += sum((value - group_mean) ** 2 for value in values)
    total = ss_between + ss_within
    return {
        "n": len(flat),
        "n_groups": len(groups),
        "ss_between": ss_between,
        "ss_within": ss_within,
        "between_share_of_ss": None if total == 0.0 else ss_between / total,
        "group_means": {name: _mean(values) for name, values in sorted(groups.items())},
        "group_n": {name: len(values) for name, values in sorted(groups.items())},
    }


def _mae(observed: list[float], predicted: list[float]) -> float:
    return _mean([abs(obs - pred) for obs, pred in zip(observed, predicted)])


def _contrast(observed: list[float], baseline: list[float], challenger: list[float]) -> dict[str, Any]:
    """Positive mean means the challenger has smaller absolute error than the baseline."""
    differences = [
        abs(obs - base) - abs(obs - other)
        for obs, base, other in zip(observed, baseline, challenger)
    ]
    interval = paired_mean_ci(differences)
    return {
        "mae_baseline": _mae(observed, baseline),
        "mae_challenger": _mae(observed, challenger),
        "paired_mean_ae_reduction": interval["mean"],
        "low": interval["low"],
        "high": interval["high"],
        "class": interval["class"],
        "n": len(observed),
    }


def _design(train: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(train)), train])


def _ols(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> list[float]:
    beta, _, _, _ = np.linalg.lstsq(train_x, train_y, rcond=None)
    return [float(value) for value in (test_x @ beta)]


def _regime_prediction(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> list[float]:
    groups = _grouped(train, "regime_id")
    fallback = _mean([float(row["observed_delta"]) for row in train])
    return [_mean(groups[row["regime_id"]]) if row["regime_id"] in groups else fallback for row in test]


def _mean_only_update(prior_mean: float, n: int, evidence: list[float]) -> float:
    """Same scalar recursion as update_belief when the stored values sum to n * prior_mean."""
    current = prior_mean
    count = n
    for value in evidence:
        current = (count * current + float(value)) / (count + 1)
        count += 1
    return float(current)


def _belief_update(prior_mean: float, evidence: list[tuple[str, float]]) -> dict[str, Any]:
    belief = belief_from_observations(
        "M22.1-D1-L23",
        "additive_last_token",
        [prior_mean] * N_DISCOVERY,
        validity_scope={"status": "IN_SCOPE"},
    )
    for evidence_id, value in evidence:
        belief = update_belief(belief, value, evidence_id)
    return {
        "predicted_effect": belief.predicted_effect,
        "version": belief.version,
        "delta_from_prior": belief.predicted_effect - prior_mean,
        "matches_scalar_recursion": abs(
            belief.predicted_effect - _mean_only_update(prior_mean, N_DISCOVERY, [value for _, value in evidence])
        )
        < 1e-12,
    }


def _synthetic() -> dict[str, Any]:
    stable_train = [1.0, 1.0, 1.0, 1.0]
    stable = belief_from_observations(
        "synthetic-stable",
        "diagnostic",
        stable_train,
        validity_scope={"status": "IN_SCOPE"},
    )
    stable_after = update_belief(stable, 1.0, "more-of-the-same")
    context_train = [("c0", 0.0), ("c0", 0.0), ("c1", 2.0), ("c1", 2.0)]
    context_belief = belief_from_observations(
        "synthetic-context",
        "diagnostic",
        [value for _, value in context_train],
        validity_scope={"status": "IN_SCOPE"},
    )
    context_means = {"c0": 0.0, "c1": 2.0}
    context_test = [("c0", 0.0), ("c1", 2.0)]
    constant_errors = [abs(value - context_belief.predicted_effect) for _, value in context_test]
    context_errors = [abs(value - context_means[name]) for name, value in context_test]
    mech_a = belief_from_observations(
        "synthetic-mech-a",
        "diagnostic",
        [1.0, 1.0, 1.0, 1.0],
        validity_scope={"status": "IN_SCOPE"},
    )
    mech_b = belief_from_observations(
        "synthetic-mech-b",
        "diagnostic",
        [1.0, 1.0, 1.0, 1.0],
        validity_scope={"status": "IN_SCOPE"},
    )
    return {
        "case_a_stable": {
            "train_mean": stable.predicted_effect,
            "after_identical_evidence": stable_after.predicted_effect,
            "held_out_constant_error": 0.0,
            "belief_represents_stable_effect": stable.predicted_effect == 1.0,
        },
        "case_b_context_dependent": {
            "m23_style_prediction": context_belief.predicted_effect,
            "held_out_mae_constant_belief": _mean(constant_errors),
            "held_out_mae_context_means": _mean(context_errors),
            "predict_accepts_context": False,
        },
        "case_c_similar_means": {
            "mechanism_a_prediction": mech_a.predict(1.0)["predicted_effect"],
            "mechanism_b_prediction": mech_b.predict(1.0)["predicted_effect"],
            "predictions_equal": mech_a.predict(1.0)["predicted_effect"]
            == mech_b.predict(1.0)["predicted_effect"],
            "mechanism_ids_differ": mech_a.mechanism_id != mech_b.mechanism_id,
        },
    }


def main() -> None:
    validation = _load("validation_outcomes.json")
    replication = _load("replication_outcomes.json")
    d1_val = _rows(validation["d1"])
    d2_val = _rows(validation["d2"])
    d1_rep = _rows(replication["alpha_1"])
    d1_rep_2 = _rows(replication["alpha_2"])
    if [row["prompt_id"] for row in d1_rep] != [row["prompt_id"] for row in d1_rep_2]:
        raise RuntimeError("alpha +1 and +2 replication prompts differ")
    if any(a["pre_dot"] != b["pre_dot"] for a, b in zip(d1_rep, d1_rep_2)):
        raise RuntimeError("pre_dot changed between alpha +1 and alpha +2")

    d1_val_y = [float(row["observed_delta"]) for row in d1_val]
    d2_val_y = [float(row["observed_delta"]) for row in d2_val]
    d1_rep_y = [float(row["observed_delta"]) for row in d1_rep]
    d1_rep2_y = [float(row["observed_delta"]) for row in d1_rep_2]

    train_mean = _mean(d1_val_y)
    constant = [train_mean for _ in d1_rep_y]
    m23_discovery = [DISCOVERY_MEAN for _ in d1_rep_y]
    m23_updated = [UPDATED_MEAN for _ in d1_rep_y]
    regime = _regime_prediction(d1_val, d1_rep)

    pre_train = np.array([float(row["pre_dot"]) for row in d1_val], dtype=float)
    pre_test = np.array([float(row["pre_dot"]) for row in d1_rep], dtype=float)
    base_train = np.array([float(row["baseline_margin"]) for row in d1_val], dtype=float)
    base_test = np.array([float(row["baseline_margin"]) for row in d1_rep], dtype=float)
    y_train = np.array(d1_val_y, dtype=float)
    pred_pre = _ols(_design(pre_train), y_train, _design(pre_test))
    pred_base = _ols(_design(base_train), y_train, _design(base_test))
    both_train = np.column_stack([np.ones(len(y_train)), pre_train, base_train])
    both_test = np.column_stack([np.ones(len(d1_rep_y)), pre_test, base_test])
    pred_both = _ols(both_train, y_train, both_test)

    def scaled(values: list[float]) -> list[float]:
        return [2.0 * value for value in values]

    predictors = {
        "constant_validation_mean": constant,
        "m23_discovery_belief": m23_discovery,
        "m23_updated_belief": m23_updated,
        "regime_means": regime,
        "ols_pre_dot": pred_pre,
        "ols_baseline_margin": pred_base,
        "ols_pre_dot_and_baseline": pred_both,
    }
    heldout = {
        name: _contrast(d1_rep_y, constant, values) for name, values in predictors.items() if name != "constant_validation_mean"
    }
    heldout_magnitude = {
        name: _contrast(d1_rep2_y, scaled(constant), scaled(values))
        for name, values in predictors.items()
        if name != "constant_validation_mean"
    }
    versus_m23 = {
        name: _contrast(d1_rep_y, m23_discovery, values)
        for name, values in predictors.items()
        if name not in {"m23_discovery_belief"}
    }

    def _corr(x: list[float], y: list[float]) -> float | None:
        if len(x) < 3 or _sample_sd(x) in (None, 0.0) or _sample_sd(y) in (None, 0.0):
            return None
        return float(np.corrcoef(np.array(x), np.array(y))[0, 1])

    evidence_sets = {
        "correct_d1": [(row["prompt_id"], float(row["observed_delta"])) for row in d1_val],
        "shuffled_d1_same_multiset": [
            (row["prompt_id"], float(other["observed_delta"]))
            for row, other in zip(d1_val, list(reversed(d1_val)))
        ],
        "control_d2": [(f"d2-{row['prompt_id']}", float(row["observed_delta"])) for row in d2_val],
        "repeated_first_d1": [(f"repeat-{index}", d1_val_y[0]) for index in range(len(d1_val_y))],
        "no_evidence": [],
    }
    updates = {
        name: _belief_update(DISCOVERY_MEAN, pairs) if pairs else {
            "predicted_effect": DISCOVERY_MEAN,
            "version": 1,
            "delta_from_prior": 0.0,
            "matches_scalar_recursion": True,
        }
        for name, pairs in evidence_sets.items()
    }

    episodes = []
    for split_name, rows in (("validation", d1_val), ("replication_alpha_1", d1_rep), ("replication_alpha_2", d1_rep_2)):
        for row in rows:
            episodes.append(
                {
                    "split": split_name,
                    "prompt_id": row["prompt_id"],
                    "regime_id": row["regime_id"],
                    "alpha": row["alpha"],
                    "baseline_margin": row["baseline_margin"],
                    "pre_dot": row["pre_dot"],
                    "observed_delta": row["observed_delta"],
                }
            )

    payload = {
        "historical_m23_verdict": "INCONCLUSIVE",
        "historical_verdict_unchanged": True,
        "discovery_per_prompt_rows_saved": False,
        "inherited_interval_rule": "M23 paired percentile bootstrap, 5000 draws, seed 23001",
        "intervention_distribution": {
            "d1_validation_alpha_1": _distribution(d1_val_y),
            "d1_replication_alpha_1": _distribution(d1_rep_y),
            "d1_replication_alpha_2": _distribution(d1_rep2_y),
            "d2_validation_alpha_1": _distribution(d2_val_y),
            "d1_validation_by_regime": _variance_split(_grouped(d1_val, "regime_id")),
            "d1_replication_alpha_1_by_regime": _variance_split(_grouped(d1_rep, "regime_id")),
            "between_seed_variance": "not_estimable",
            "between_seed_reason": "M23 stored one primary direction seed and one orthogonal control, not repeated seeds of the same intervention.",
            "alpha_2_over_alpha_1_mean_ratio": _mean(d1_rep2_y) / _mean(d1_rep_y),
            "discovery_sample_sd_from_episode_table": DISCOVERY_SD,
            "m23_falsification_band_halfwidth": 1.96 * DISCOVERY_SD,
            "historical_critical_cases": 0,
        },
        "pre_intervention_spread": {
            "validation_pre_dot": _distribution([float(row["pre_dot"]) for row in d1_val]),
            "replication_pre_dot": _distribution([float(row["pre_dot"]) for row in d1_rep]),
            "validation_baseline_margin": _distribution([float(row["baseline_margin"]) for row in d1_val]),
            "correlation_pre_dot_delta_validation": _corr(
                [float(row["pre_dot"]) for row in d1_val], d1_val_y
            ),
            "correlation_baseline_delta_validation": _corr(
                [float(row["baseline_margin"]) for row in d1_val], d1_val_y
            ),
            "correlation_pre_dot_delta_replication": _corr(
                [float(row["pre_dot"]) for row in d1_rep], d1_rep_y
            ),
            "correlation_baseline_delta_replication": _corr(
                [float(row["baseline_margin"]) for row in d1_rep], d1_rep_y
            ),
        },
        "heldout_versus_validation_mean": heldout,
        "heldout_magnitude_versus_scaled_validation_mean": heldout_magnitude,
        "heldout_versus_m23_discovery_belief": versus_m23,
        "predictor_mae": {name: _mae(d1_rep_y, values) for name, values in predictors.items()},
        "belief_updates": updates,
        "main_versus_control": {
            "d1_discovery_mean": DISCOVERY_MEAN,
            "d2_discovery_mean_from_verdict": json.loads(
                (ROOT / "reports" / "M23_SCIENTIFIC_VERDICT.json").read_text(encoding="utf-8")
            )["discovery_d2_mean"],
            "d1_validation_mean": train_mean,
            "d2_validation_mean": _mean(d2_val_y),
            "predict_reads_intervention_identity": False,
            "predict_reads_pre_dot": False,
            "distinction_requires_experimenter_chosen_intervention_id": True,
        },
        "synthetic": _synthetic(),
    }
    out = ROOT / "reports" / "M23_DIAGNOSTIC_RESULTS.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = ROOT / "reports" / "M23_DIAGNOSTIC_EPISODES.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    print(out)


if __name__ == "__main__":
    main()
