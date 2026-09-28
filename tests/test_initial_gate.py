"""Checks for the IG-0 gate. The full run is the experiment, not a mock."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_self_model.initial_gate import (
    load_config,
    matches_truth,
    rotate_operands,
    run_gate,
    run_seed,
    sample_observational,
    select_hypothesis,
    structural,
)


class WorldTests(unittest.TestCase):
    def test_rotation_turns_addition_into_subtraction(self) -> None:
        rng = np.random.default_rng(0)
        state = rng.uniform(-1.0, 1.0, size=(32, 4))
        rotated = rotate_operands(state)
        added = structural("add", state, 0, 1)
        subtracted = structural("sub", rotated, 0, 1)
        np.testing.assert_allclose(added, subtracted)

    def test_observational_add_and_copy_are_tied(self) -> None:
        config = load_config()
        rng = np.random.default_rng(1)
        state, outcome = sample_observational(rng, ("add", 0, 1), 50, config)
        choice, error, margin = select_hypothesis(state, outcome)
        self.assertLess(margin, float(config["identification_margin"]))
        self.assertLess(error, 0.05)
        self.assertTrue(choice[0] in ("add", "copy"))

    def test_config_requires_a_majority_of_seeds(self) -> None:
        config = load_config()
        self.assertGreaterEqual(config["minimum_seed_passes"], 18)
        self.assertGreaterEqual(config["seeds"], config["minimum_seed_passes"])


class GateTests(unittest.TestCase):
    def test_single_seed_report_has_required_gates(self) -> None:
        report = run_seed(0, load_config())
        for name in (
            "trap_exists",
            "identify_training_mechanism",
            "prediction_is_not_identity",
            "rebind_with_new_evidence",
            "rotation_needs_reidentification",
            "novelty_is_not_invalidity",
            "observation_cannot_identify",
            "shuffle_collapses",
            "nonlinear_mechanism",
        ):
            self.assertIn(name, report["gates"])
        self.assertTrue(matches_truth(tuple(report["selected_hypothesis"]), ("add", 0, 1)))

    def test_full_gate_passes_pre_registered_criteria(self) -> None:
        result = run_gate()
        failed = [
            (report["seed"], [name for name, ok in report["gates"].items() if not ok])
            for report in result["seeds"]
            if not report["passed"]
        ]
        self.assertGreaterEqual(
            result["seed_passes"],
            result["minimum_seed_passes"],
            failed,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["disallowed_claims"])


if __name__ == "__main__":
    unittest.main()
