# ============================================================
# M21.2.4.3.3
# Calibration Stress, Support, and Transfer Audit
#
# Scientific status:
# - PAVA implementation audit
# - Episode-disjoint calibration/test audit
# - Support-aware isotonic calibration analysis
# - Exact-output / perturbation / boundary stress audits
# - Track-wise and cross-track descriptive evaluation
#
# Important:
# q_invalid is NOT automatically universal probability semantics.
# Exact q_invalid=0 or q_invalid=1 is NOT epistemic certainty.
# ============================================================

import os
import json
import math
import shutil
import random
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 1. Configuration
# ============================================================

SEED = 20260912

CONFIG = {
    "experiment_name": "M21.2.4.3.3",
    "seed": SEED,

    # Operational clipping is not a calibration method.
    "operational_clip_eps": 1e-4,

    # Numerical clipping solely for stable LogLoss.
    "metric_clip_eps": 1e-12,

    # Threshold for empirical support tagging of exact 0/1 outputs.
    # Does NOT authorize epistemic certainty.
    "min_certainty_block_support": 20.0,

    "high_confidence_threshold": 0.99,
    "low_confidence_threshold": 0.01,

    "fixed_reliability_bins": 10,
    "quantile_reliability_bins": 10,

    # For initial Colab smoke run, use 300.
    # For final benchmark, use 1000 or more.
    "n_episode_bootstrap": 1000,
    "bootstrap_ci_alpha": 0.05,

    "perturbation_deltas": [
        1e-12,
        1e-9,
        1e-6,
        1e-4,
    ],

    # None means evaluate every internal PAVA boundary.
    "max_boundary_stress_points": 500,

    # Descriptive support thresholds; not universal scientific gates.
    "support_coverage_thresholds": [1, 5, 20, 50, 100],

    # Warning heuristic only; not a scientific theorem.
    "singleton_mass_warning_threshold": 0.50,

    "artifact_dir": "artifacts/M21.2.4.3.3",

    # Tracks exposed during isotonic fit.
    "calibration_domain_tracks": [
        "KNOWN_VALID",
        "CAUSAL_BREAK",
        "FALSE_ALARM",
    ],

    "all_expected_tracks": [
        "KNOWN_VALID",
        "FALSE_ALARM",
        "CAUSAL_BREAK",
        "REGIME_CHANGE_IN_SCOPE",
        "REGIME_CHANGE_OUT_OF_SCOPE",
        "NOVEL_BUT_VALID",
        "NOVEL_AND_INVALID",
    ],
}

np.random.seed(SEED)
random.seed(SEED)

plt.style.use("seaborn-v0_8-whitegrid")

ARTIFACT_DIR = Path(CONFIG["artifact_dir"])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

with open(ARTIFACT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

# ============================================================
# 2. Reproducible stable seed
# ============================================================

def stable_seed(*parts, base_seed: int = 0, modulus: int = 2**31 - 1) -> int:
    """
    Deterministic process-independent seed.

    Python built-in hash() must NOT be used for benchmark seeds because
    hash randomization can change its result between processes.
    """
    key = "::".join(map(str, parts))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int((int(base_seed) + offset) % modulus)

# ============================================================
# 3. Data contract and validation
# ============================================================

REQUIRED_KEYS = [
    "raw_probability_calibration",
    "y_calibration",
    "track_calibration",
    "episode_id_calibration",

    "raw_probability_test",
    "y_test",
    "track_test",
    "episode_id_test",
]


def load_prediction_bundle_npz(path: str) -> Dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=True)

    missing = [key for key in REQUIRED_KEYS if key not in bundle.files]
    if missing:
        raise KeyError(
            f"NPZ file is missing required keys: {missing}\n"
            f"Available keys: {bundle.files}"
        )

    return {key: np.asarray(bundle[key]) for key in REQUIRED_KEYS}


def ensure_binary_labels(y: np.ndarray, name: str):
    y = np.asarray(y)

    if not np.all(np.isfinite(y.astype(float))):
        raise ValueError(f"{name} contains non-finite values.")

    unique = np.unique(y)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{name} must contain only 0/1. Found {unique}")


def ensure_probability_vector(p: np.ndarray, name: str):
    p = np.asarray(p, dtype=float)

    if p.ndim != 1:
        raise ValueError(f"{name} must be 1D. Shape={p.shape}")

    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} contains NaN or infinity.")

    invalid = (p < 0.0) | (p > 1.0)
    if np.any(invalid):
        example = np.where(invalid)[0][:10]
        raise ValueError(
            f"{name} must be in [0,1]. Bad indices={example}, "
            f"values={p[example]}"
        )


def make_episode_key(track: np.ndarray, episode_id: np.ndarray) -> np.ndarray:
    track = np.asarray(track).astype(str)
    episode_id = np.asarray(episode_id).astype(str)
    return np.char.add(np.char.add(track, "::"), episode_id)


def validate_split_arrays(
    raw_probability: np.ndarray,
    y: np.ndarray,
    track: np.ndarray,
    episode_id: np.ndarray,
    split_name: str,
) -> Dict[str, np.ndarray]:

    raw_probability = np.asarray(raw_probability, dtype=float)
    y = np.asarray(y).astype(int)
    track = np.asarray(track).astype(str)
    episode_id = np.asarray(episode_id).astype(str)

    n = len(y)

    for name, arr in {
        "raw_probability": raw_probability,
        "y": y,
        "track": track,
        "episode_id": episode_id,
    }.items():
        if len(arr) != n:
            raise ValueError(
                f"{split_name}: {name} length={len(arr)}, expected={n}"
            )

    ensure_binary_labels(y, f"{split_name}.y")
    ensure_probability_vector(raw_probability, f"{split_name}.raw_probability")

    return {
        "raw_probability": raw_probability,
        "y": y,
        "track": track,
        "episode_id": episode_id,
        "episode_key": make_episode_key(track, episode_id),
    }

# ============================================================
# 4. Hardened weighted PAVA isotonic calibrator
# ============================================================

@dataclass
class PAVABlock:
    block_id: int
    start_group: int
    end_group: int
    x_left: float
    x_right: float
    positive_weight: float
    total_weight: float
    probability: float
    n_unique_scores: int
    n_observations: int


class IsotonicCalibrator:
    """
    Weighted PAVA isotonic regression for binary calibration.

    Properties:
    - Tied scores grouped before monotonic pooling.
    - PAVA merges violating blocks using weighted means.
    - Predicts via interpolation across fitted score knots.
    - Provides support / block / extrapolation metadata.
    """

    def __init__(self):
        self.is_fitted_ = False

    @staticmethod
    def _validate_inputs(scores, y, sample_weight=None):
        scores = np.asarray(scores, dtype=float)
        y = np.asarray(y)

        if scores.ndim != 1 or y.ndim != 1:
            raise ValueError("scores and y must be 1D.")

        if len(scores) != len(y):
            raise ValueError("scores and y lengths must match.")

        if len(scores) == 0:
            raise ValueError("Cannot fit on an empty calibration split.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("scores contains NaN or infinity.")

        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("PAVA calibration requires binary labels 0/1.")

        y = y.astype(int)

        if sample_weight is None:
            sample_weight = np.ones(len(scores), dtype=float)
        else:
            sample_weight = np.asarray(sample_weight, dtype=float)

            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1D.")

            if len(sample_weight) != len(scores):
                raise ValueError("sample_weight length must match scores.")

            if not np.all(np.isfinite(sample_weight)):
                raise ValueError("sample_weight contains NaN or infinity.")

            if np.any(sample_weight <= 0):
                raise ValueError("sample_weight must be strictly positive.")

        return scores, y, sample_weight

    def fit(self, scores, y, sample_weight=None):
        scores, y, sample_weight = self._validate_inputs(
            scores, y, sample_weight
        )

        order = np.argsort(scores, kind="mergesort")

        x_sorted = scores[order]
        y_sorted = y[order]
        w_sorted = sample_weight[order]

        unique_x, starts, counts = np.unique(
            x_sorted,
            return_index=True,
            return_counts=True,
        )

        n_groups = len(unique_x)

        group_weight = np.zeros(n_groups, dtype=float)
        group_positive_weight = np.zeros(n_groups, dtype=float)
        group_observation_count = np.zeros(n_groups, dtype=int)

        # Group tied scores.
        for g, (start, count) in enumerate(zip(starts, counts)):
            end = start + count
            w = w_sorted[start:end]
            yy = y_sorted[start:end]

            group_weight[g] = np.sum(w)
            group_positive_weight[g] = np.sum(w * yy)
            group_observation_count[g] = count

        # Weighted PAVA stack.
        stack = []

        for g in range(n_groups):
            block = {
                "start_group": g,
                "end_group": g,
                "positive_weight": float(group_positive_weight[g]),
                "total_weight": float(group_weight[g]),
                "n_unique_scores": 1,
                "n_observations": int(group_observation_count[g]),
            }

            stack.append(block)

            while len(stack) >= 2:
                left = stack[-2]
                right = stack[-1]

                left_mean = left["positive_weight"] / left["total_weight"]
                right_mean = right["positive_weight"] / right["total_weight"]

                if left_mean <= right_mean:
                    break

                merged = {
                    "start_group": left["start_group"],
                    "end_group": right["end_group"],
                    "positive_weight": (
                        left["positive_weight"] + right["positive_weight"]
                    ),
                    "total_weight": (
                        left["total_weight"] + right["total_weight"]
                    ),
                    "n_unique_scores": (
                        left["n_unique_scores"] + right["n_unique_scores"]
                    ),
                    "n_observations": (
                        left["n_observations"] + right["n_observations"]
                    ),
                }

                stack = stack[:-2]
                stack.append(merged)

        fitted_by_group = np.zeros(n_groups, dtype=float)
        block_index_by_group = np.zeros(n_groups, dtype=int)
        blocks = []

        for block_id, block in enumerate(stack):
            p = block["positive_weight"] / block["total_weight"]

            fitted_by_group[
                block["start_group"]:block["end_group"] + 1
            ] = p

            block_index_by_group[
                block["start_group"]:block["end_group"] + 1
            ] = block_id

            blocks.append(
                PAVABlock(
                    block_id=block_id,
                    start_group=block["start_group"],
                    end_group=block["end_group"],
                    x_left=float(unique_x[block["start_group"]]),
                    x_right=float(unique_x[block["end_group"]]),
                    positive_weight=float(block["positive_weight"]),
                    total_weight=float(block["total_weight"]),
                    probability=float(p),
                    n_unique_scores=int(block["n_unique_scores"]),
                    n_observations=int(block["n_observations"]),
                )
            )

        self.unique_x_ = unique_x
        self.group_weight_ = group_weight
        self.group_positive_weight_ = group_positive_weight
        self.group_observation_count_ = group_observation_count

        self.fitted_by_group_ = fitted_by_group
        self.block_index_by_group_ = block_index_by_group
        self.blocks_ = blocks

        self.score_min_ = float(unique_x[0])
        self.score_max_ = float(unique_x[-1])

        self.is_fitted_ = True
        return self

    def _require_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("IsotonicCalibrator is not fitted.")

    def predict(self, scores):
        self._require_fitted()

        scores = np.asarray(scores, dtype=float)

        if not np.all(np.isfinite(scores)):
            raise ValueError("Prediction scores contain NaN or infinity.")

        return np.interp(
            scores,
            self.unique_x_,
            self.fitted_by_group_,
            left=self.fitted_by_group_[0],
            right=self.fitted_by_group_[-1],
        )

    def blocks_dataframe(self) -> pd.DataFrame:
        self._require_fitted()
        return pd.DataFrame([asdict(block) for block in self.blocks_])

    def predict_with_metadata(
        self,
        scores,
        min_certainty_block_support: float = 20.0,
        atol: float = 1e-15,
    ) -> pd.DataFrame:

        self._require_fitted()

        scores = np.asarray(scores, dtype=float)
        probabilities = self.predict(scores)

        rows = []

        for score, p in zip(scores, probabilities):
            below = score < self.score_min_
            above = score > self.score_max_

            if below:
                left_group = 0
                right_group = 0
                extrapolation_status = "BELOW_CALIBRATION_SCORE_RANGE"

            elif above:
                left_group = len(self.unique_x_) - 1
                right_group = len(self.unique_x_) - 1
                extrapolation_status = "ABOVE_CALIBRATION_SCORE_RANGE"

            else:
                position = np.searchsorted(
                    self.unique_x_,
                    score,
                    side="left",
                )

                is_exact_knot = (
                    position < len(self.unique_x_)
                    and np.isclose(
                        score,
                        self.unique_x_[position],
                        atol=atol,
                        rtol=0.0,
                    )
                )

                if is_exact_knot:
                    left_group = position
                    right_group = position
                else:
                    right_group = min(position, len(self.unique_x_) - 1)
                    left_group = max(right_group - 1, 0)

                extrapolation_status = "IN_SUPPORT"

            left_block_id = int(
                self.block_index_by_group_[left_group]
            )
            right_block_id = int(
                self.block_index_by_group_[right_group]
            )

            left_block = self.blocks_[left_block_id]
            right_block = self.blocks_[right_block_id]

            interpolated = (
                left_group != right_group
                and not np.isclose(score, self.unique_x_[left_group], atol=atol)
                and not np.isclose(score, self.unique_x_[right_group], atol=atol)
            )

            local_support = min(
                left_block.total_weight,
                right_block.total_weight,
            )

            local_n_observations = min(
                left_block.n_observations,
                right_block.n_observations,
            )

            # Important:
            # These labels denote empirical support only.
            # They do not grant epistemic certainty.
            if np.isclose(p, 0.0, atol=atol):
                exact_output_status = (
                    "EMPIRICALLY_SUPPORTED_EXACT_ZERO"
                    if local_support >= min_certainty_block_support
                    else "LOW_SUPPORT_EXACT_ZERO"
                )

            elif np.isclose(p, 1.0, atol=atol):
                exact_output_status = (
                    "EMPIRICALLY_SUPPORTED_EXACT_ONE"
                    if local_support >= min_certainty_block_support
                    else "LOW_SUPPORT_EXACT_ONE"
                )

            else:
                exact_output_status = "NON_EXTREME"

            rows.append(
                {
                    "raw_probability": float(score),
                    "q_invalid_isotonic": float(p),

                    "left_group_index": int(left_group),
                    "right_group_index": int(right_group),

                    "left_block_id": left_block_id,
                    "right_block_id": right_block_id,

                    "left_block_probability": float(left_block.probability),
                    "right_block_probability": float(right_block.probability),

                    "left_block_support": float(left_block.total_weight),
                    "right_block_support": float(right_block.total_weight),

                    "left_block_n_observations": int(
                        left_block.n_observations
                    ),
                    "right_block_n_observations": int(
                        right_block.n_observations
                    ),

                    "local_support": float(local_support),
                    "local_n_observations": int(local_n_observations),

                    "interpolated": bool(interpolated),
                    "extrapolation_status": extrapolation_status,
                    "exact_output_status": exact_output_status,
                }
            )

        return pd.DataFrame(rows)

    def support_summary(self, atol: float = 1e-15) -> Dict[str, Any]:
        blocks = self.blocks_dataframe()

        exact_zero = blocks[
            np.isclose(blocks["probability"], 0.0, atol=atol)
        ]

        exact_one = blocks[
            np.isclose(blocks["probability"], 1.0, atol=atol)
        ]

        singleton_blocks = blocks[blocks["n_observations"] == 1]

        total_weight = float(blocks["total_weight"].sum())
        singleton_weight = float(singleton_blocks["total_weight"].sum())

        return {
            "n_calibration_samples": int(
                self.group_observation_count_.sum()
            ),
            "n_unique_scores": int(len(self.unique_x_)),
            "n_blocks": int(len(blocks)),

            "n_singleton_blocks": int(len(singleton_blocks)),
            "singleton_block_fraction": float(
                len(singleton_blocks) / max(1, len(blocks))
            ),
            "singleton_mass_fraction": float(
                singleton_weight / max(total_weight, 1e-12)
            ),

            "min_block_observations": int(blocks["n_observations"].min()),
            "max_block_observations": int(blocks["n_observations"].max()),
            "median_block_observations": float(
                blocks["n_observations"].median()
            ),

            "n_exact_zero_blocks": int(len(exact_zero)),
            "n_exact_one_blocks": int(len(exact_one)),

            "exact_zero_mass_fraction": float(
                exact_zero["total_weight"].sum()
                / max(total_weight, 1e-12)
            ),
            "exact_one_mass_fraction": float(
                exact_one["total_weight"].sum()
                / max(total_weight, 1e-12)
            ),
        }

# ============================================================
# 5. PAVA unit tests
# ============================================================

def run_pava_unit_tests() -> Dict[str, str]:
    results = {}

    # Monotonicity.
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 1, 0, 1])
    cal = IsotonicCalibrator().fit(x, y)
    assert np.all(np.diff(cal.predict(x)) >= -1e-15)
    results["monotonicity"] = "PASS"

    # Tied score grouping.
    x = np.array([0.2, 0.2, 0.8, 0.8])
    y = np.array([0, 1, 0, 1])
    cal = IsotonicCalibrator().fit(x, y)
    assert np.allclose(cal.predict(np.array([0.2, 0.8])), [0.5, 0.5])
    results["tied_score_grouping"] = "PASS"

    # Weighted pooling.
    x = np.array([0.1, 0.2])
    y = np.array([1, 0])
    w = np.array([3.0, 1.0])
    cal = IsotonicCalibrator().fit(x, y, w)
    assert np.allclose(cal.predict(x), [0.75, 0.75])
    results["weighted_pooling"] = "PASS"

    # Non-finite rejection.
    try:
        IsotonicCalibrator().fit(
            np.array([0.1, np.nan]),
            np.array([0, 1]),
        )
        raise AssertionError("Expected non-finite rejection.")
    except ValueError:
        results["nonfinite_rejection"] = "PASS"

    # Non-binary label rejection.
    try:
        IsotonicCalibrator().fit(
            np.array([0.1, 0.2]),
            np.array([0, 2]),
        )
        raise AssertionError("Expected non-binary rejection.")
    except ValueError:
        results["nonbinary_label_rejection"] = "PASS"

    return results

# ============================================================
# 6. Prediction table
# ============================================================

def build_prediction_table(
    split: Dict[str, np.ndarray],
    split_name: str,
    calibrator: IsotonicCalibrator,
    config: Dict[str, Any],
) -> pd.DataFrame:

    metadata = calibrator.predict_with_metadata(
        split["raw_probability"],
        min_certainty_block_support=config[
            "min_certainty_block_support"
        ],
    )

    df = pd.DataFrame(
        {
            "split": split_name,
            "y_invalid": split["y"].astype(int),
            "track": split["track"].astype(str),
            "episode_id": split["episode_id"].astype(str),
            "episode_key": split["episode_key"].astype(str),
        }
    )

    df = pd.concat([df, metadata], axis=1)

    df["q_invalid_operational"] = np.clip(
        df["q_invalid_isotonic"].to_numpy(),
        config["operational_clip_eps"],
        1.0 - config["operational_clip_eps"],
    )

    calibration_tracks = set(config["calibration_domain_tracks"])

    df["calibration_domain"] = np.where(
        df["track"].isin(calibration_tracks),
        "IN_DOMAIN_TRACK_FAMILY",
        "UNSEEN_TRACK_FAMILY",
    )

    return df

# ============================================================
# 7. Metrics and reliability
# ============================================================

PREDICTION_METHODS = {
    "raw": "raw_probability",
    "isotonic": "q_invalid_isotonic",
    "operational_clipped": "q_invalid_operational",
}


def safe_binary_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
    metric_clip_eps: float,
) -> Dict[str, Optional[float]]:

    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)

    p_logloss = np.clip(
        p,
        metric_clip_eps,
        1.0 - metric_clip_eps,
    )

    out = {
        "n_samples": int(len(y_true)),
        "prevalence_invalid": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, p_logloss, labels=[0, 1])),
        "auroc": None,
        "auprc": None,
    }

    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, p))
        out["auprc"] = float(average_precision_score(y_true, p))

    return out


def make_fixed_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def make_quantile_edges(p: np.ndarray, n_bins: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))

    if len(edges) < 2:
        return np.array([0.0, 1.0])

    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)

    return edges


def ece_from_edges(
    y_true: np.ndarray,
    p: np.ndarray,
    edges: np.ndarray,
) -> float:

    bin_index = np.digitize(p, edges[1:-1], right=False)
    ece = 0.0

    for b in range(len(edges) - 1):
        mask = bin_index == b

        if not np.any(mask):
            continue

        ece += (
            np.mean(mask)
            * abs(np.mean(y_true[mask]) - np.mean(p[mask]))
        )

    return float(ece)


def reliability_table_from_edges(
    y_true: np.ndarray,
    p: np.ndarray,
    edges: np.ndarray,
    method: str,
    binning: str,
) -> pd.DataFrame:

    bin_index = np.digitize(p, edges[1:-1], right=False)
    rows = []

    for b in range(len(edges) - 1):
        mask = bin_index == b

        if not np.any(mask):
            rows.append(
                {
                    "method": method,
                    "binning": binning,
                    "bin_index": b,
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                    "n_samples": 0,
                    "mean_prediction": np.nan,
                    "empirical_invalidity": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_prediction = float(np.mean(p[mask]))
        empirical_invalidity = float(np.mean(y_true[mask]))

        rows.append(
            {
                "method": method,
                "binning": binning,
                "bin_index": b,
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "n_samples": int(np.sum(mask)),
                "mean_prediction": mean_prediction,
                "empirical_invalidity": empirical_invalidity,
                "absolute_gap": abs(
                    empirical_invalidity - mean_prediction
                ),
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# 8. Episode bootstrap
# ============================================================

def episode_bootstrap_indices(
    episode_keys: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:

    episode_keys = np.asarray(episode_keys).astype(str)
    unique_episodes = np.unique(episode_keys)

    sampled_episodes = rng.choice(
        unique_episodes,
        size=len(unique_episodes),
        replace=True,
    )

    index_map = {
        episode: np.flatnonzero(episode_keys == episode)
        for episode in unique_episodes
    }

    return np.concatenate([
        index_map[episode]
        for episode in sampled_episodes
    ])


def bootstrap_metric_ci_episode_level(
    y_true: np.ndarray,
    p: np.ndarray,
    episode_keys: np.ndarray,
    metric_name: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
    metric_clip_eps: float,
) -> Tuple[Optional[float], Optional[float]]:

    local_rng = np.random.default_rng(seed)
    values = []

    for _ in range(n_bootstrap):
        idx = episode_bootstrap_indices(episode_keys, local_rng)

        yb = y_true[idx]
        pb = p[idx]

        if metric_name == "brier":
            value = brier_score_loss(yb, pb)

        elif metric_name == "logloss":
            value = log_loss(
                yb,
                np.clip(pb, metric_clip_eps, 1.0 - metric_clip_eps),
                labels=[0, 1],
            )

        elif metric_name == "ece_fixed":
            value = ece_from_edges(
                yb,
                pb,
                make_fixed_edges(CONFIG["fixed_reliability_bins"]),
            )

        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        values.append(value)

    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1.0 - alpha / 2)),
    )

# ============================================================
# 9. Metric reports
# ============================================================

def evaluate_prediction_methods(
    df: pd.DataFrame,
    split_name: str,
    config: Dict[str, Any],
) -> pd.DataFrame:

    rows = []

    for method_name, column in PREDICTION_METHODS.items():
        y = df["y_invalid"].to_numpy()
        p = df[column].to_numpy()

        metrics = safe_binary_metrics(
            y,
            p,
            config["metric_clip_eps"],
        )

        metrics["ece_fixed"] = ece_from_edges(
            y,
            p,
            make_fixed_edges(config["fixed_reliability_bins"]),
        )

        metrics["ece_quantile"] = ece_from_edges(
            y,
            p,
            make_quantile_edges(
                p,
                config["quantile_reliability_bins"],
            ),
        )

        for metric_name in ["brier", "logloss", "ece_fixed"]:
            low, high = bootstrap_metric_ci_episode_level(
                y_true=y,
                p=p,
                episode_keys=df["episode_key"].to_numpy(),
                metric_name=metric_name,
                n_bootstrap=config["n_episode_bootstrap"],
                alpha=config["bootstrap_ci_alpha"],
                seed=stable_seed(
                    "overall_episode_bootstrap",
                    split_name,
                    method_name,
                    metric_name,
                    base_seed=config["seed"],
                ),
                metric_clip_eps=config["metric_clip_eps"],
            )

            metrics[f"{metric_name}_ci95_low"] = low
            metrics[f"{metric_name}_ci95_high"] = high

        rows.append(
            {
                "split": split_name,
                "method": method_name,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def trackwise_audit(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:

    rows = []

    for track, track_df in df.groupby("track", sort=True):
        y = track_df["y_invalid"].to_numpy()

        for method_name, column in PREDICTION_METHODS.items():
            p = track_df[column].to_numpy()

            metrics = safe_binary_metrics(
                y,
                p,
                config["metric_clip_eps"],
            )

            metrics["ece_fixed"] = ece_from_edges(
                y,
                p,
                make_fixed_edges(config["fixed_reliability_bins"]),
            )

            row = {
                "track": track,
                "calibration_domain": track_df[
                    "calibration_domain"
                ].iloc[0],
                "method": method_name,
                "n_samples": int(len(track_df)),
                "n_episodes": int(track_df["episode_key"].nunique()),
                **metrics,
            }

            for metric_name in ["brier", "logloss", "ece_fixed"]:
                low, high = bootstrap_metric_ci_episode_level(
                    y_true=y,
                    p=p,
                    episode_keys=track_df["episode_key"].to_numpy(),
                    metric_name=metric_name,
                    n_bootstrap=config["n_episode_bootstrap"],
                    alpha=config["bootstrap_ci_alpha"],
                    seed=stable_seed(
                        "trackwise_episode_bootstrap",
                        track,
                        method_name,
                        metric_name,
                        base_seed=config["seed"],
                    ),
                    metric_clip_eps=config["metric_clip_eps"],
                )

                row[f"{metric_name}_ci95_low"] = low
                row[f"{metric_name}_ci95_high"] = high

            row["mean_prediction"] = float(np.mean(p))
            row["p95_prediction"] = float(np.quantile(p, 0.95))
            row["p99_prediction"] = float(np.quantile(p, 0.99))

            rows.append(row)

    return pd.DataFrame(rows)


def domain_transfer_audit(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:

    rows = []

    for domain, domain_df in df.groupby("calibration_domain", sort=True):
        y = domain_df["y_invalid"].to_numpy()

        row = {
            "domain": domain,
            "tracks": ", ".join(sorted(domain_df["track"].unique())),
            "n_samples": int(len(domain_df)),
            "n_episodes": int(domain_df["episode_key"].nunique()),
            "prevalence_invalid": float(np.mean(y)),
        }

        for method_name, column in PREDICTION_METHODS.items():
            p = domain_df[column].to_numpy()

            metrics = safe_binary_metrics(
                y,
                p,
                config["metric_clip_eps"],
            )

            row[f"{method_name}_brier"] = metrics["brier"]
            row[f"{method_name}_logloss"] = metrics["logloss"]
            row[f"{method_name}_auroc"] = metrics["auroc"]
            row[f"{method_name}_auprc"] = metrics["auprc"]

            row[f"{method_name}_ece_fixed"] = ece_from_edges(
                y,
                p,
                make_fixed_edges(config["fixed_reliability_bins"]),
            )

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# 10. Reliability artifacts
# ============================================================

def create_reliability_artifacts(
    df: pd.DataFrame,
    split_name: str,
    config: Dict[str, Any],
    artifact_dir: Path,
) -> pd.DataFrame:

    tables = []

    for method_name, column in PREDICTION_METHODS.items():
        y = df["y_invalid"].to_numpy()
        p = df[column].to_numpy()

        schemes = {
            "fixed": make_fixed_edges(config["fixed_reliability_bins"]),
            "quantile": make_quantile_edges(
                p,
                config["quantile_reliability_bins"],
            ),
        }

        for binning, edges in schemes.items():
            table = reliability_table_from_edges(
                y,
                p,
                edges,
                method_name,
                binning,
            )
            tables.append(table)

    reliability_df = pd.concat(tables, ignore_index=True)

    reliability_df.to_csv(
        artifact_dir / f"{split_name}_reliability_tables.csv",
        index=False,
    )

    for binning in ["fixed", "quantile"]:
        fig, ax = plt.subplots(figsize=(8, 7))

        ax.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle="--",
            label="Perfect calibration",
        )

        for method_name in PREDICTION_METHODS:
            d = reliability_df[
                (reliability_df["method"] == method_name)
                & (reliability_df["binning"] == binning)
                & (reliability_df["n_samples"] > 0)
            ]

            ax.plot(
                d["mean_prediction"],
                d["empirical_invalidity"],
                marker="o",
                linewidth=1.8,
                label=method_name,
            )

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted invalidity")
        ax.set_ylabel("Empirical invalidity")
        ax.set_title(
            f"{split_name}: Reliability Diagram ({binning} bins)"
        )
        ax.legend()

        plt.tight_layout()
        plt.savefig(
            artifact_dir / f"{split_name}_reliability_{binning}.png",
            dpi=180,
        )
        plt.close()

    return reliability_df

# ============================================================
# 11. Exact-output audit
# ============================================================

def exact_output_audit(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:

    p = df["q_invalid_isotonic"].to_numpy()
    y = df["y_invalid"].to_numpy()

    exact_zero = np.isclose(p, 0.0)
    exact_one = np.isclose(p, 1.0)

    valid_as_exact_one = (y == 0) & exact_one
    invalid_as_exact_zero = (y == 1) & exact_zero

    low_support_extreme = df["exact_output_status"].isin(
        [
            "LOW_SUPPORT_EXACT_ZERO",
            "LOW_SUPPORT_EXACT_ONE",
        ]
    )

    forensic_mask = (
        valid_as_exact_one
        | invalid_as_exact_zero
        | low_support_extreme.to_numpy()
    )

    forensic_columns = [
        "split",
        "track",
        "episode_id",
        "episode_key",
        "y_invalid",
        "raw_probability",
        "q_invalid_isotonic",
        "q_invalid_operational",
        "local_support",
        "local_n_observations",
        "interpolated",
        "extrapolation_status",
        "exact_output_status",
        "calibration_domain",
    ]

    forensic_df = df.loc[forensic_mask, forensic_columns].copy()

    summary = {
        "n_samples": int(len(df)),
        "n_exact_zero_predictions": int(np.sum(exact_zero)),
        "n_exact_one_predictions": int(np.sum(exact_one)),
        "valid_samples_mapped_to_exact_one": int(
            np.sum(valid_as_exact_one)
        ),
        "invalid_samples_mapped_to_exact_zero": int(
            np.sum(invalid_as_exact_zero)
        ),
        "low_support_extreme_count": int(
            np.sum(low_support_extreme)
        ),
        "high_confidence_false_alarm_count": int(
            np.sum(
                (y == 0)
                & (p >= config["high_confidence_threshold"])
            )
        ),
        "low_confidence_miss_count": int(
            np.sum(
                (y == 1)
                & (p <= config["low_confidence_threshold"])
            )
        ),
    }

    return summary, forensic_df

# ============================================================
# 12. Score perturbation stability audit
# ============================================================

def score_perturbation_stability_audit(
    calibrator: IsotonicCalibrator,
    df: pd.DataFrame,
    deltas: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    scores = df["raw_probability"].to_numpy(dtype=float)
    p_base = calibrator.predict(scores)

    rows = []

    for delta in deltas:
        score_minus = np.clip(scores - delta, 0.0, 1.0)
        score_plus = np.clip(scores + delta, 0.0, 1.0)

        p_minus = calibrator.predict(score_minus)
        p_plus = calibrator.predict(score_plus)

        span = np.abs(p_plus - p_minus)

        for i in range(len(scores)):
            rows.append(
                {
                    "row_index": int(i),
                    "track": df["track"].iloc[i],
                    "episode_key": df["episode_key"].iloc[i],
                    "delta": float(delta),
                    "score": float(scores[i]),
                    "p_base": float(p_base[i]),
                    "p_minus": float(p_minus[i]),
                    "p_plus": float(p_plus[i]),
                    "two_sided_span_change": float(span[i]),
                    "exact_output_status": df[
                        "exact_output_status"
                    ].iloc[i],
                }
            )

    detailed = pd.DataFrame(rows)
    summary_rows = []

    for delta, d in detailed.groupby("delta", sort=True):
        v = d["two_sided_span_change"].to_numpy()

        summary_rows.append(
            {
                "delta": float(delta),
                "n_scores": int(len(d)),
                "mean_span_change": float(np.mean(v)),
                "median_span_change": float(np.median(v)),
                "p95_span_change": float(np.quantile(v, 0.95)),
                "p99_span_change": float(np.quantile(v, 0.99)),
                "max_span_change": float(np.max(v)),
                "fraction_span_gt_1e_4": float(np.mean(v > 1e-4)),
                "fraction_span_gt_1e_2": float(np.mean(v > 1e-2)),
            }
        )

    return detailed, pd.DataFrame(summary_rows)

# ============================================================
# 13. PAVA block-boundary sensitivity audit
# ============================================================

def get_internal_boundary_scores(
    calibrator: IsotonicCalibrator,
) -> np.ndarray:

    blocks = calibrator.blocks_dataframe().sort_values("block_id")

    if len(blocks) < 2:
        return np.array([], dtype=float)

    boundaries = []

    for i in range(len(blocks) - 1):
        left = blocks.iloc[i]
        right = blocks.iloc[i + 1]

        boundaries.append(
            0.5 * (left["x_right"] + right["x_left"])
        )

    return np.asarray(boundaries, dtype=float)


def boundary_sensitivity_audit(
    calibrator: IsotonicCalibrator,
    deltas: List[float],
    max_boundary_points: Optional[int],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    boundaries = get_internal_boundary_scores(calibrator)

    if (
        max_boundary_points is not None
        and len(boundaries) > max_boundary_points
    ):
        local_rng = np.random.default_rng(
            stable_seed(
                "boundary_subsample",
                len(boundaries),
                base_seed=config["seed"],
            )
        )

        boundaries = np.sort(
            local_rng.choice(
                boundaries,
                size=max_boundary_points,
                replace=False,
            )
        )

    rows = []

    for boundary_index, boundary in enumerate(boundaries):
        for delta in deltas:
            left_score = max(0.0, boundary - delta)
            right_score = min(1.0, boundary + delta)

            p_left = float(calibrator.predict([left_score])[0])
            p_right = float(calibrator.predict([right_score])[0])

            rows.append(
                {
                    "boundary_index": int(boundary_index),
                    "boundary_score": float(boundary),
                    "delta": float(delta),
                    "p_left": p_left,
                    "p_right": p_right,
                    "two_sided_boundary_span": abs(p_right - p_left),
                }
            )

    detailed = pd.DataFrame(rows)

    if len(detailed) == 0:
        return detailed, pd.DataFrame()

    summary_rows = []

    for delta, d in detailed.groupby("delta", sort=True):
        v = d["two_sided_boundary_span"].to_numpy()

        summary_rows.append(
            {
                "delta": float(delta),
                "n_boundaries": int(len(d)),
                "mean_boundary_span": float(np.mean(v)),
                "median_boundary_span": float(np.median(v)),
                "p95_boundary_span": float(np.quantile(v, 0.95)),
                "p99_boundary_span": float(np.quantile(v, 0.99)),
                "max_boundary_span": float(np.max(v)),
                "fraction_span_gt_1e_4": float(np.mean(v > 1e-4)),
                "fraction_span_gt_1e_2": float(np.mean(v > 1e-2)),
            }
        )

    return detailed, pd.DataFrame(summary_rows)

# ============================================================
# 14. Support coverage audit
# ============================================================

def calibration_support_coverage_audit(
    df: pd.DataFrame,
    thresholds: List[float],
) -> pd.DataFrame:

    rows = []
    total_episodes = max(1, df["episode_key"].nunique())

    for threshold in thresholds:
        mask = df["local_support"].to_numpy() >= threshold

        rows.append(
            {
                "support_threshold": str(threshold),
                "n_samples": int(np.sum(mask)),
                "fraction_test_mass": float(np.mean(mask)),
                "n_episodes": int(
                    df.loc[mask, "episode_key"].nunique()
                ),
                "fraction_episodes": float(
                    df.loc[mask, "episode_key"].nunique()
                    / total_episodes
                ),
            }
        )

    extrapolated = (
        df["extrapolation_status"] != "IN_SUPPORT"
    )

    interpolated = df["interpolated"]

    low_support_extreme = df["exact_output_status"].isin(
        [
            "LOW_SUPPORT_EXACT_ZERO",
            "LOW_SUPPORT_EXACT_ONE",
        ]
    )

    for label, mask in {
        "INTERPOLATED": interpolated,
        "EXTRAPOLATED": extrapolated,
        "LOW_SUPPORT_EXTREME": low_support_extreme,
    }.items():
        rows.append(
            {
                "support_threshold": label,
                "n_samples": int(np.sum(mask)),
                "fraction_test_mass": float(np.mean(mask)),
                "n_episodes": int(
                    df.loc[mask, "episode_key"].nunique()
                ),
                "fraction_episodes": float(
                    df.loc[mask, "episode_key"].nunique()
                    / total_episodes
                ),
            }
        )

    return pd.DataFrame(rows)

# ============================================================
# 15. Verdict
# ============================================================

def generate_verdict(
    support_summary: Dict[str, Any],
    exact_summary: Dict[str, Any],
    support_coverage_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Dict[str, Any]:

    singleton_mass = support_summary["singleton_mass_fraction"]

    warning_triggered = (
        singleton_mass >= config["singleton_mass_warning_threshold"]
    )

    false_extreme = (
        exact_summary["valid_samples_mapped_to_exact_one"] > 0
        or exact_summary["invalid_samples_mapped_to_exact_zero"] > 0
    )

    extrapolation_row = support_coverage_df[
        support_coverage_df["support_threshold"] == "EXTRAPOLATED"
    ]

    extrapolation_fraction = (
        float(extrapolation_row["fraction_test_mass"].iloc[0])
        if len(extrapolation_row) else None
    )

    return {
        "engineering_pipeline_status": "PASS",
        "pava_implementation_status": "PASS",
        "episode_split_status": "PASS",

        "singleton_support_warning_heuristic": {
            "singleton_mass_fraction": singleton_mass,
            "warning_threshold": config[
                "singleton_mass_warning_threshold"
            ],
            "triggered": warning_triggered,
            "interpretation": (
                "Warning heuristic only; not a universal "
                "scientific pass/fail criterion."
            ),
        },

        "false_exact_extreme_counterexample_observed": false_extreme,

        "exact_zero_one_semantics": (
            "NOT AUTHORIZED AS EPISTEMIC CERTAINTY"
        ),

        "test_score_extrapolation_fraction": extrapolation_fraction,

        "cross_track_transfer_status": (
            "DESCRIPTIVE OUT-OF-TRACK-FAMILY AUDIT ONLY; "
            "UNIVERSAL TRANSFER NOT ESTABLISHED"
        ),

        "recommended_q_invalid_contract": (
            "support-aware, benchmark-conditionally calibrated "
            "invalidity evidence score"
        ),

        "mandatory_downstream_policy": (
            "Downstream policy must consume q_invalid jointly with "
            "local_support, exact_output_status, calibration_domain, "
            "and extrapolation_status. Exact 0/1 is not certainty."
        ),
    }

# ============================================================
# 16. Plot helpers
# ============================================================

def create_diagnostic_plots(
    blocks_df: pd.DataFrame,
    test_df: pd.DataFrame,
    perturbation_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    config: Dict[str, Any],
    artifact_dir: Path,
):
    # PAVA block support.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        blocks_df["n_observations"],
        bins=min(30, max(5, blocks_df["n_observations"].nunique())),
        ax=ax,
    )
    ax.set_title("PAVA Block Observation Support")
    ax.set_xlabel("Observations per PAVA block")
    plt.tight_layout()
    plt.savefig(
        artifact_dir / "pava_block_support_distribution.png",
        dpi=180,
    )
    plt.close()

    # Local support.
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        test_df["local_support"],
        bins=min(30, max(5, test_df["local_support"].nunique())),
        ax=ax,
    )
    ax.axvline(
        config["min_certainty_block_support"],
        color="red",
        linestyle="--",
        label="empirical-support annotation threshold",
    )
    ax.set_title("Final-Test Local Calibration Support")
    ax.set_xlabel("Local calibration support")
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        artifact_dir / "final_test_local_support_distribution.png",
        dpi=180,
    )
    plt.close()

    # Exact output status.
    fig, ax = plt.subplots(figsize=(9, 5))
    status_counts = (
        test_df["exact_output_status"]
        .value_counts()
        .rename_axis("status")
        .reset_index(name="count")
    )
    sns.barplot(data=status_counts, x="status", y="count", ax=ax)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Final-Test Exact-Output Status")
    plt.tight_layout()
    plt.savefig(
        artifact_dir / "final_test_exact_output_status.png",
        dpi=180,
    )
    plt.close()

    # Perturbation stress.
    if len(perturbation_summary):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            perturbation_summary["delta"],
            perturbation_summary["p95_span_change"],
            marker="o",
            label="P95 span",
        )
        ax.plot(
            perturbation_summary["delta"],
            perturbation_summary["max_span_change"],
            marker="s",
            label="Maximum span",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Score Perturbation Stability")
        ax.set_xlabel("δ")
        ax.set_ylabel("|p(s+δ) - p(s-δ)|")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            artifact_dir / "score_perturbation_stability.png",
            dpi=180,
        )
        plt.close()

    # Boundary stress.
    if len(boundary_summary):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            boundary_summary["delta"],
            boundary_summary["p95_boundary_span"],
            marker="o",
            label="P95 boundary span",
        )
        ax.plot(
            boundary_summary["delta"],
            boundary_summary["max_boundary_span"],
            marker="s",
            label="Maximum boundary span",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("PAVA Block-Boundary Sensitivity")
        ax.set_xlabel("δ")
        ax.set_ylabel("|p(b+δ) - p(b-δ)|")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            artifact_dir / "block_boundary_sensitivity.png",
            dpi=180,
        )
        plt.close()

# ============================================================
# 17. Main
# ============================================================

def main(input_npz: str):
    print("=" * 72)
    print("M21.2.4.3.3 — Calibration Stress, Support, Transfer Audit")
    print("=" * 72)

    # --------------------------------------------------------
    # Load and validate
    # --------------------------------------------------------
    data = load_prediction_bundle_npz(input_npz)

    calibration = validate_split_arrays(
        data["raw_probability_calibration"],
        data["y_calibration"],
        data["track_calibration"],
        data["episode_id_calibration"],
        split_name="calibration",
    )

    test = validate_split_arrays(
        data["raw_probability_test"],
        data["y_test"],
        data["track_test"],
        data["episode_id_test"],
        split_name="final_test",
    )

    overlap = set(calibration["episode_key"]).intersection(
        set(test["episode_key"])
    )

    if overlap:
        raise AssertionError(
            "EPISODE LEAKAGE DETECTED between calibration and final test. "
            f"Examples: {sorted(overlap)[:10]}"
        )

    print("Episode-disjoint split audit: PASS")
    print("Calibration samples:", len(calibration["y"]))
    print("Final-test samples:", len(test["y"]))

    # --------------------------------------------------------
    # PAVA implementation tests
    # --------------------------------------------------------
    pava_tests = run_pava_unit_tests()

    with open(
        ARTIFACT_DIR / "pava_unit_tests.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(pava_tests, f, indent=2)

    # --------------------------------------------------------
    # Fit only on calibration split
    # --------------------------------------------------------
    calibrator = IsotonicCalibrator().fit(
        calibration["raw_probability"],
        calibration["y"],
    )

    blocks_df = calibrator.blocks_dataframe()
    blocks_df.to_csv(
        ARTIFACT_DIR / "pava_blocks.csv",
        index=False,
    )

    support_summary = calibrator.support_summary()

    with open(
        ARTIFACT_DIR / "calibration_support_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(support_summary, f, indent=2)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    calibration_df = build_prediction_table(
        calibration,
        "calibration",
        calibrator,
        CONFIG,
    )

    test_df = build_prediction_table(
        test,
        "final_test",
        calibrator,
        CONFIG,
    )

    calibration_df.to_csv(
        ARTIFACT_DIR / "calibration_predictions_with_metadata.csv",
        index=False,
    )

    test_df.to_csv(
        ARTIFACT_DIR / "final_test_predictions_with_metadata.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    overall_df = evaluate_prediction_methods(
        test_df,
        "final_test",
        CONFIG,
    )
    overall_df.to_csv(
        ARTIFACT_DIR / "overall_final_test_metrics.csv",
        index=False,
    )

    trackwise_df = trackwise_audit(test_df, CONFIG)
    trackwise_df.to_csv(
        ARTIFACT_DIR / "trackwise_episode_bootstrap_audit.csv",
        index=False,
    )

    transfer_df = domain_transfer_audit(test_df, CONFIG)
    transfer_df.to_csv(
        ARTIFACT_DIR / "cross_track_transfer_audit.csv",
        index=False,
    )

    reliability_df = create_reliability_artifacts(
        test_df,
        "final_test",
        CONFIG,
        ARTIFACT_DIR,
    )

    # --------------------------------------------------------
    # Exact-output audit
    # --------------------------------------------------------
    exact_summary, forensic_df = exact_output_audit(test_df, CONFIG)

    with open(
        ARTIFACT_DIR / "exact_output_audit.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(exact_summary, f, indent=2)

    forensic_df.to_csv(
        ARTIFACT_DIR / "exact_output_forensic_records.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Stress audits
    # --------------------------------------------------------
    perturbation_detailed, perturbation_summary = (
        score_perturbation_stability_audit(
            calibrator,
            test_df,
            CONFIG["perturbation_deltas"],
        )
    )

    perturbation_detailed.to_csv(
        ARTIFACT_DIR / "score_perturbation_stability_detailed.csv",
        index=False,
    )

    perturbation_summary.to_csv(
        ARTIFACT_DIR / "score_perturbation_stability_summary.csv",
        index=False,
    )

    boundary_detailed, boundary_summary = boundary_sensitivity_audit(
        calibrator,
        CONFIG["perturbation_deltas"],
        CONFIG["max_boundary_stress_points"],
        CONFIG,
    )

    boundary_detailed.to_csv(
        ARTIFACT_DIR / "block_boundary_sensitivity_detailed.csv",
        index=False,
    )

    boundary_summary.to_csv(
        ARTIFACT_DIR / "block_boundary_sensitivity_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Support coverage
    # --------------------------------------------------------
    support_coverage_df = calibration_support_coverage_audit(
        test_df,
        CONFIG["support_coverage_thresholds"],
    )

    support_coverage_df.to_csv(
        ARTIFACT_DIR / "calibration_support_coverage.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------
    create_diagnostic_plots(
        blocks_df,
        test_df,
        perturbation_summary,
        boundary_summary,
        CONFIG,
        ARTIFACT_DIR,
    )

    # --------------------------------------------------------
    # Scientific verdict
    # --------------------------------------------------------
    verdict = generate_verdict(
        support_summary,
        exact_summary,
        support_coverage_df,
        CONFIG,
    )

    with open(
        ARTIFACT_DIR / "scientific_verdict.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(verdict, f, indent=2)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------
    manifest = {
        "experiment_name": CONFIG["experiment_name"],
        "seed": CONFIG["seed"],
        "input_npz": str(input_npz),

        "calibration_samples": int(len(calibration_df)),
        "calibration_episodes": int(
            calibration_df["episode_key"].nunique()
        ),

        "final_test_samples": int(len(test_df)),
        "final_test_episodes": int(
            test_df["episode_key"].nunique()
        ),

        "episode_overlap": 0,
        "calibration_tracks": sorted(
            calibration_df["track"].unique().tolist()
        ),
        "test_tracks": sorted(
            test_df["track"].unique().tolist()
        ),

        "support_summary": support_summary,
        "exact_output_summary": exact_summary,
        "scientific_verdict": verdict,

        "artifact_files": sorted(
            str(p.name)
            for p in ARTIFACT_DIR.glob("*")
        ),
    }

    with open(
        ARTIFACT_DIR / "artifact_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    # ZIP only artifact directory.
    zip_path = shutil.make_archive(
        str(ARTIFACT_DIR),
        "zip",
        root_dir=str(ARTIFACT_DIR),
    )

    print("\n" + "=" * 72)
    print("RUN COMPLETE")
    print("=" * 72)

    print("\nOverall metrics:")
    print(overall_df.to_string(index=False))

    print("\nCalibration support summary:")
    print(json.dumps(support_summary, indent=2))

    print("\nExact-output audit:")
    print(json.dumps(exact_summary, indent=2))

    print("\nScientific verdict:")
    print(json.dumps(verdict, indent=2))

    print("\nArtifact directory:", ARTIFACT_DIR)
    print("Artifact zip:", zip_path)

    return {
        "overall_metrics": overall_df,
        "trackwise_metrics": trackwise_df,
        "transfer_metrics": transfer_df,
        "support_coverage": support_coverage_df,
        "support_summary": support_summary,
        "exact_output_summary": exact_summary,
        "verdict": verdict,
        "artifact_dir": str(ARTIFACT_DIR),
        "zip_path": zip_path,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_npz",
        required=True,
        help="NPZ prediction bundle satisfying the M21.2.4.3.3 data contract.",
    )

    args = parser.parse_args()
    main(args.input_npz)
