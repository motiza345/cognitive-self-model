"""End-to-end pipeline determinism and bundle contract."""

import numpy as np

from src.cognitive_self_model.benchmark.bundle import BUNDLE_KEYS
from src.cognitive_self_model.benchmark.pipeline import load_config, run_pipeline


def test_pipeline_produces_eight_key_bundle_and_is_deterministic(tmp_path):
    config = load_config()

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    summary_a = run_pipeline(config=config, output_dir=out_a)
    summary_b = run_pipeline(config=config, output_dir=out_b)

    # Bundle exists with exactly the eight contract keys.
    with np.load(out_a / "M21_2_4_3_3_input.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(BUNDLE_KEYS)

    # Deterministic: identical dataset SHA256 across independent runs.
    assert summary_a["npz_bundle_sha256"] == summary_b["npz_bundle_sha256"]


def test_bundle_split_sizes_match_config(tmp_path):
    config = load_config()
    run_pipeline(config=config, output_dir=tmp_path)
    with np.load(tmp_path / "M21_2_4_3_3_input.npz", allow_pickle=False) as saved:
        # train not in bundle; calibration = 3 tracks x 8 ep x 40 steps = 960
        assert len(saved["y_calibration"]) == 3 * 8 * 40
        # final_test = 6 tracks x 10 ep x 40 steps = 2400
        assert len(saved["y_test"]) == 6 * 10 * 40
        assert len(saved["raw_probability_test"]) == 2400
