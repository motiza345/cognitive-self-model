"""
Formal leakage audit (L1-L9) and negative controls for the M21.2.4.3.1 benchmark.

These tests turn the roadmap's leakage contract into executable guarantees:
the estimator must depend only on the frozen 16-D evidence at time <= t, and
must be powerless when the real signal is destroyed (shuffled labels) yet
trivially powerful when ground truth is injected (proving the audit can detect
leakage).
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from src.cognitive_self_model.benchmark import (
    EnvironmentTrack,
    FEATURE_NAMES,
    ScientificFrozenEncoder,
    ScientificFrozenEnvironment,
    SelfModelInvalidityEstimator,
)
from src.cognitive_self_model.benchmark.evidence import FEATURE_DIM
from src.cognitive_self_model.benchmark.pipeline import build_scored_dataset


def _episode_observations(track, seed=7, steps=40):
    env = ScientificFrozenEnvironment(track, seed=seed).reset()
    obs_list = []
    for step in range(steps):
        x = np.sin(step * 0.1)
        u = np.cos(step * 0.1)
        obs, _ = env.step(x, u, [u + 0.2])
        obs_list.append(obs)
    return obs_list


# ---------------------------------------------------------------- L3: causality
def test_L3_encoder_uses_only_past_and_present_no_future_leakage():
    obs_list = _episode_observations(EnvironmentTrack.CAUSAL_BREAK)

    full = ScientificFrozenEncoder()
    phis_full = np.asarray([full.encode(o) for o in obs_list])

    # Re-encoding only the prefix E_<=k must reproduce phi at every step <= k,
    # i.e. phi(t) is a function of E_<=t and never of future observations.
    for k in (5, 15, 25, 39):
        prefix = ScientificFrozenEncoder()
        phis_prefix = np.asarray([prefix.encode(o) for o in obs_list[: k + 1]])
        np.testing.assert_array_equal(phis_prefix, phis_full[: k + 1])


# ---------------------------------------------- L4/L5/L6/L7: no forbidden inputs
def test_L4_L5_L6_L7_evidence_excludes_forbidden_concepts():
    forbidden_substrings = (
        "track",
        "mechanism",
        "label",
        "invalid",
        "governance",
        "decision",
        "episode",
        "future",
    )
    joined = " ".join(FEATURE_NAMES).lower()
    for token in forbidden_substrings:
        assert token not in joined, f"evidence schema leaks '{token}'"
    assert FEATURE_DIM == 16


# ------------------------------------------------------- L8: order/identity free
def test_L8_predictions_are_per_row_and_order_invariant():
    scored = build_scored_dataset()
    test = scored["test"]
    estimator = scored["estimator"]

    perm = np.random.RandomState(0).permutation(len(test["y"]))
    p_original = estimator.predict_raw_probability(test["X"])
    p_permuted = estimator.predict_raw_probability(test["X"][perm])
    np.testing.assert_allclose(p_permuted, p_original[perm], rtol=0, atol=1e-12)


# --------------------------------------------------- positive baseline (signal)
def test_real_evidence_separates_valid_from_invalid():
    scored = build_scored_dataset()
    test = scored["test"]
    auroc = roc_auc_score(test["y"], test["raw_probability"])
    assert auroc > 0.9, f"evidence should carry the invalidity signal (AUROC={auroc:.3f})"


# ------------------------------------------- negative control: shuffled labels
def test_negative_control_shuffled_labels_collapse_auroc():
    scored = build_scored_dataset()
    train, test = scored["train"], scored["test"]

    rng = np.random.RandomState(0)
    y_shuffled = train["y"].copy()
    rng.shuffle(y_shuffled)

    est = SelfModelInvalidityEstimator(feature_dim=FEATURE_DIM).fit(
        train["X"], y_shuffled, epochs=800, lr=0.1
    )
    auroc = roc_auc_score(test["y"], est.predict_raw_probability(test["X"]))
    # Destroying the label-evidence association must destroy discriminative power.
    assert 0.3 < auroc < 0.7, f"shuffled-label AUROC should be ~chance (got {auroc:.3f})"


# --------------------------------------- negative control: ground-truth injection
def test_negative_control_ground_truth_injection_is_detectable():
    scored = build_scored_dataset()
    train, test = scored["train"], scored["test"]

    # Inject the label as an extra feature: the audit harness must be able to
    # detect this catastrophic leakage as near-perfect performance.
    Xtr = np.column_stack([train["X"], train["y"].astype(float)])
    Xte = np.column_stack([test["X"], test["y"].astype(float)])

    est = SelfModelInvalidityEstimator(feature_dim=FEATURE_DIM + 1).fit(
        Xtr, train["y"], epochs=800, lr=0.1
    )
    auroc = roc_auc_score(test["y"], est.predict_raw_probability(Xte))
    assert auroc > 0.99, f"injected ground truth must be detectable (AUROC={auroc:.3f})"
