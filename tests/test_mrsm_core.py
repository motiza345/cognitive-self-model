"""Unit checks for the frozen MRSM rules. No checkpoint and no holdout scoring."""

from __future__ import annotations

import numpy as np

from src.mrsm import HEADS
from src.mrsm.baselines import predictions
from src.mrsm.evaluators import score_h1, score_h2, score_h3, score_h4
from src.mrsm.leakage import audit_sources
from src.mrsm.self_model import LeakageError, SelfModelCore
from src.mrsm.transforms import apply_transform, frozen_parameters, invert_transform


def _effects(true_pair: tuple[str, str] = ("L0H0", "L1H1")) -> tuple[dict[str, float], dict[str, float]]:
    singles = {name: 0.05 for name in HEADS}
    singles[true_pair[0]] = 1.2
    singles[true_pair[1]] = 1.1
    joints = {}
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            key = f"{left}+{right}"
            interaction = 0.0
            if {left, right} == set(true_pair):
                interaction = 0.8
            joints[key] = singles[left] + singles[right] + interaction
    return singles, joints


def test_self_model_rejects_ground_truth_field():
    core = SelfModelCore("scope")
    try:
        core.update({"ground_truth": 1}, {}, [], min_abs_interaction=1e-4, min_relative_gap=0.05)
    except LeakageError:
        return
    raise AssertionError("ground truth field was accepted")


def test_discovery_recovers_pair_without_a_label_argument():
    core = SelfModelCore("P:test")
    singles, joints = _effects()
    core.update(singles, joints, ["e0"], min_abs_interaction=1e-4, min_relative_gap=0.05)
    assert core.mechanism["ordered"] == ["L0H0", "L1H1"]
    assert core.predict("pair:L0H0+L1H1") == joints["L0H0+L1H1"]


def test_not_identifiable_is_not_a_pass():
    core = SelfModelCore("P:test")
    singles = {name: 0.0 for name in HEADS}
    joints = {}
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            joints[f"{left}+{right}"] = 0.0
    core.update(singles, joints, [], min_abs_interaction=1e-4, min_relative_gap=0.05)
    assert core.mechanism["status"] == "NOT_IDENTIFIABLE"
    h1 = score_h1(core.mechanism, singles, ground_truth_ordered=("L0H0", "L1H1"), sign_floor=1e-8)
    assert h1["status"] == "FAIL"
    cases = [
        {
            "original_prediction": 0.0,
            "transformed_prediction": 0.0,
            "mechanism_abstraction": "NOT_IDENTIFIABLE",
            "original_abstraction": "NOT_IDENTIFIABLE",
            "scope": "P:test",
            "original_scope": "P:test",
        }
    ]
    assert score_h3(cases, sign_floor=1e-8)["status"] == "NOT_EVALUATED"


def test_h1_pass_and_fail_are_all_or_nothing():
    mechanism = {"status": "IDENTIFIED", "ordered": ["L0H0", "L1H1"], "abstraction": "L0H0->L1H1"}
    effects = {name: 0.1 for name in HEADS}
    effects["L0H0"] = 1.0
    effects["L1H1"] = 1.0
    assert score_h1(mechanism, effects, ground_truth_ordered=("L0H0", "L1H1"), sign_floor=1e-8)["status"] == "PASS"
    wrong = {"status": "IDENTIFIED", "ordered": ["L0H2", "L1H3"], "abstraction": "L0H2->L1H3"}
    assert score_h1(wrong, effects, ground_truth_ordered=("L0H0", "L1H1"), sign_floor=1e-8)["status"] == "FAIL"


def test_h3_inverse_preserves_predictions():
    vec = np.linspace(-0.4, 1.1, 8)
    for name in ("T1", "T2", "T3"):
        recovered = invert_transform(name, apply_transform(name, vec))
        assert np.max(np.abs(recovered - vec)) < 1e-8
    frozen = frozen_parameters()
    assert frozen["T2"]["perm"] == [2, 0, 3, 1, 6, 4, 7, 5]


def test_h2_requires_a_real_margin_over_the_additive_baseline():
    singles, joints = _effects()
    self_pred = {f"single:{name}": singles[name] for name in HEADS}
    self_pred.update({f"pair:{key}": value for key, value in joints.items()})
    base = predictions(singles)
    actual = {key: [value] * 32 for key, value in self_pred.items()}
    # Additive baseline is wrong on the true pair; self-model matches the actual joint.
    scored = score_h2(self_pred, base, actual, resamples=200, seed=1729, sign_floor=1e-8)
    assert scored["strongest_baseline"] == "B2"
    assert scored["mae_self"] == 0.0
    assert scored["status"] == "PASS"


def test_h4_tie_is_a_fail():
    scored = score_h4([1] * 20, [1] * 20, resamples=100, seed=1729)
    assert scored["status"] == "FAIL"


def test_runtime_sources_do_not_embed_the_ground_truth_file():
    assert audit_sources()["pass"] is True
