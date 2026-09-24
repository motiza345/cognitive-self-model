"""
M21.2.4.3.1 pipeline orchestrator.

End-to-end: build episode-disjoint train/calibration/test splits, fit the raw
invalidity estimator on the train split, emit raw ``q_invalid`` for the
calibration and test splits, and write the eight-key NPZ bundle consumed by
M21.2.4.3.3. A reproducibility manifest (seeds, versions, dataset SHA256, git
commit/branch) is written alongside the bundle.

This module intentionally writes generated artifacts to a configurable output
directory (default ``reports/M21_2_4_3_1``) and does not commit them.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ...common import reproducibility as _repro
from .bundle import (
    BUNDLE_KEYS,
    binary_logloss,
    brier_score,
    file_sha256,
    write_bundle,
)
from .calibration import IsotonicCalibrator
from .data import collect_episode_track_data, flatten_split
from .estimator import SelfModelInvalidityEstimator
from .evidence import FEATURE_DIM, FEATURE_NAMES
from .tracks import EnvironmentTrack

PIPELINE_VERSION = "M21.2.4.3.1-clean-port"
BENCHMARK_VERSION = "frozen-reference-0.6x+0.5u"
EVIDENCE_SCHEMA_VERSION = "phi16-RR6-PT2-DD2-CC2-HH2-XX2"

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "m21_2_4_3_1_config.json"


def load_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the pipeline config JSON (defaults to ``configs/m21_2_4_3_1_config.json``)."""
    path = Path(config_path) if config_path is not None else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_split(spec: Dict[str, Any], env_cfg: Dict[str, Any], split_name: str) -> Dict[str, np.ndarray]:
    tracks = [EnvironmentTrack(t) for t in spec["tracks"]]
    episodes = collect_episode_track_data(
        tracks=tracks,
        num_episodes=int(spec["num_episodes"]),
        seed_start=int(spec["seed_start"]),
        steps_per_episode=int(env_cfg["steps_per_episode"]),
        noise_std=float(env_cfg["noise_std"]),
        probe_measurement_std=float(env_cfg["probe_measurement_std"]),
        probe_offset=float(env_cfg["probe_offset"]),
        window_size=int(env_cfg["window_size"]),
        onset_step=int(env_cfg["onset_step"]),
    )
    return flatten_split(episodes, split_name=split_name)


def build_scored_dataset(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build episode-disjoint splits, train the estimator, and score every split.

    Returns the train/calibration/test splits (each augmented with a
    ``raw_probability`` array = raw ``q_invalid``), the fitted estimator, and the
    per-split episode-id sets. Shared by :func:`run_pipeline` and the
    M21.2.4.3.4 identifiability dataset generator so the raw scores and labels
    always come from the same real benchmark oracle.
    """
    cfg = config if config is not None else load_config()
    env_cfg = cfg["environment"]
    splits_cfg = cfg["splits"]

    train = _build_split(splits_cfg["train"], env_cfg, "train")
    calibration = _build_split(splits_cfg["calibration"], env_cfg, "calibration")
    test = _build_split(splits_cfg["final_test"], env_cfg, "final_test")

    train_ids = set(train["episode_id"].tolist())
    calib_ids = set(calibration["episode_id"].tolist())
    test_ids = set(test["episode_id"].tolist())
    if train_ids & calib_ids or train_ids & test_ids or calib_ids & test_ids:
        raise AssertionError("Episode-disjoint split contract violated.")

    est_cfg = cfg["estimator"]
    estimator = SelfModelInvalidityEstimator(
        feature_dim=FEATURE_DIM, l2_reg=float(est_cfg["l2_reg"])
    )
    estimator.fit(
        train["X"], train["y"], epochs=int(est_cfg["epochs"]), lr=float(est_cfg["lr"])
    )

    for split in (train, calibration, test):
        split["raw_probability"] = estimator.predict_raw_probability(split["X"])

    return {
        "config": cfg,
        "estimator": estimator,
        "train": train,
        "calibration": calibration,
        "test": test,
        "episode_ids": {
            "train": train_ids,
            "calibration": calib_ids,
            "test": test_ids,
        },
    }


def run_pipeline(
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run the full M21.2.4.3.1 pipeline and return a summary dict."""
    cfg = config if config is not None else load_config()
    splits_cfg = cfg["splits"]

    out_dir = Path(output_dir) if output_dir is not None else Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = build_scored_dataset(cfg)
    train = scored["train"]
    calibration = scored["calibration"]
    test = scored["test"]
    estimator = scored["estimator"]
    train_ids = scored["episode_ids"]["train"]
    calib_ids = scored["episode_ids"]["calibration"]
    test_ids = scored["episode_ids"]["test"]

    raw_probability_calibration = calibration["raw_probability"]
    raw_probability_test = test["raw_probability"]

    bundle_path = out_dir / "M21_2_4_3_3_input.npz"
    write_bundle(
        bundle_path,
        raw_probability_calibration=raw_probability_calibration,
        y_calibration=calibration["y"],
        track_calibration=calibration["track"],
        episode_id_calibration=calibration["episode_id"],
        raw_probability_test=raw_probability_test,
        y_test=test["y"],
        track_test=test["track"],
        episode_id_test=test["episode_id"],
    )

    # Downstream (informational) isotonic calibration artifacts.
    calibrator = IsotonicCalibrator().fit(raw_probability_calibration, calibration["y"])
    isotonic_probability_test = calibrator.calibrate(raw_probability_test)

    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "label_semantics": cfg.get("label_semantics"),
        "splits": {
            "train": {"tracks": splits_cfg["train"]["tracks"], "n_samples": int(len(train["y"])), "n_episodes": len(train_ids)},
            "calibration": {"tracks": splits_cfg["calibration"]["tracks"], "n_samples": int(len(calibration["y"])), "n_episodes": len(calib_ids)},
            "final_test": {"tracks": splits_cfg["final_test"]["tracks"], "n_samples": int(len(test["y"])), "n_episodes": len(test_ids)},
        },
        "episode_disjoint_splits": True,
        "npz_bundle": str(bundle_path),
        "npz_bundle_keys": list(BUNDLE_KEYS),
        "npz_bundle_key_count": len(BUNDLE_KEYS),
        "npz_bundle_sha256": file_sha256(bundle_path),
        "git_commit": _repro.get_git_commit_hash(),
        "git_branch": _repro.get_git_branch_name(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "seeds": {
            "train_seed_start": splits_cfg["train"]["seed_start"],
            "calibration_seed_start": splits_cfg["calibration"]["seed_start"],
            "final_test_seed_start": splits_cfg["final_test"]["seed_start"],
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    summary = {
        "npz_bundle": str(bundle_path),
        "npz_bundle_sha256": manifest["npz_bundle_sha256"],
        "raw_brier_final_test": brier_score(test["y"], raw_probability_test),
        "raw_logloss_final_test": binary_logloss(test["y"], raw_probability_test),
        "isotonic_brier_final_test": brier_score(test["y"], isotonic_probability_test),
        "n_isotonic_blocks": int(len(calibrator.block_values)),
        "final_test_invalidity_prevalence": float(np.mean(test["y"])),
    }
    with open(out_dir / "pipeline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":  # pragma: no cover
    result = run_pipeline()
    print(json.dumps(result, indent=2))
