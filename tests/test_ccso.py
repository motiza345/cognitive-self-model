"""CCSO rules. No Qwen download and no Self-Model."""

import numpy as np

from src.mrsm.ccso import (
    CONDITIONS,
    DECISIONS,
    DIRECTION_SEEDS,
    INCONCLUSIVE,
    LAYERS,
    LOCAL,
    MULTI,
    NO_ADDITIONAL,
    NOVEL_DIRECTION_SEEDS,
    PRIMARY_ALPHAS,
    PRIMARY_SLICES,
    READOUTS,
    SNAPSHOT,
    TRAIN_DIRECTION_SEEDS,
    beats,
    decide,
    feature_dim,
    make_projections,
    preregistration_document,
    row_features,
    run_readouts,
)


def _block(mae=1.0, sign=0.5, n_sign=20):
    return {
        "mae": mae,
        "sign_agreement": sign,
        "n_sign": n_sign,
        "pearson": 0.0,
        "n": 20,
        "decision_uses_pearson": False,
    }


def _metrics():
    linear = {name: {sl: _block() for sl in PRIMARY_SLICES} for name in READOUTS}
    small = {name: {sl: _block() for sl in PRIMARY_SLICES} for name in CONDITIONS}
    return {
        "linear": linear,
        "small_mlp": small,
        "mlp_large": {"B2": {sl: _block() for sl in ("S_val", "S_rep")}},
        "nulls": {
            name: {null: {"S_val": _block(mae=2.0)} for null in ("N1", "N2", "N4")}
            for name in ("C1", "C2")
        },
        "direction_sensitivity": {
            name: {"mean_pearson": 0.4, "n_defined": 8, "n_groups": 8}
            for name in ("C1", "C2")
        },
    }


def _set(metrics, family, name, mae, sign):
    for sl in PRIMARY_SLICES:
        metrics[family][name][sl] = _block(mae, sign)


def test_preregistration_locks_the_gate_before_a_self_model():
    document = preregistration_document()
    assert document["self_model_built"] is False
    assert document["cross_model"] is False
    assert document["fine_tuning"] is False
    assert document["historical_alpha_pm_1_in_primary_endpoint"] is False
    assert document["target_excluded_from_features"] is True
    assert "0.25" not in document["primary_alphas"]
    assert "1.00" not in document["primary_alphas"]
    assert set(TRAIN_DIRECTION_SEEDS).isdisjoint(NOVEL_DIRECTION_SEEDS)
    assert len(DIRECTION_SEEDS) == 8
    assert len(NOVEL_DIRECTION_SEEDS) == 2
    assert LAYERS == (0, 8, 15, 23)
    assert set(DECISIONS) == {
        NO_ADDITIONAL,
        SNAPSHOT,
        LOCAL,
        MULTI,
        INCONCLUSIVE,
    }


def test_feature_width_is_the_same_for_every_observation():
    projections = make_projections(32, 4)
    h = np.random.default_rng(0).normal(size=(4, 32))
    g = np.random.default_rng(1).normal(size=(4, 32))
    direction = np.random.default_rng(2).normal(size=32)
    direction /= np.linalg.norm(direction)
    widths = {
        kind: row_features(kind, h, g, direction, 0.01, 1, projections).shape
        for kind in CONDITIONS
    }
    assert set(widths.values()) == {(feature_dim(4),)}
    gradient = row_features("B3", h, g, direction, 0.01, 1, projections)
    assert gradient[3] == np.float64(0.01 * (g[1] @ direction))


def test_beats_requires_both_endpoints():
    assert beats(_block(0.2, 0.9), _block(0.5, 0.4))
    assert not beats(_block(0.2, 0.4), _block(0.5, 0.4))
    assert not beats(_block(0.5, 0.9), _block(0.5, 0.4))
    assert not beats(_block(0.2, 0.9, n_sign=0), _block(0.5, 0.4))


def test_gradient_without_added_state_or_depth_is_no_additional_information():
    metrics = _metrics()
    _set(metrics, "linear", "B3", 0.2, 0.9)
    decision = decide(metrics)
    assert decision["decision"] == NO_ADDITIONAL
    assert decision["self_model_built"] is False
    assert "cannot exist" in decision["primary_interpretation"]
    assert "mechanism" in decision["primary_interpretation"]


def test_snapshot_wins_when_richer_observations_do_not():
    metrics = _metrics()
    _set(metrics, "linear", "B2", 0.3, 0.8)
    assert decide(metrics)["decision"] == SNAPSHOT


def test_multi_depth_requires_holdouts_novel_directions_and_nulls():
    metrics = _metrics()
    _set(metrics, "linear", "C2", 0.1, 0.95)
    _set(metrics, "small_mlp", "C2", 0.1, 0.95)
    assert decide(metrics)["decision"] == MULTI
    metrics["nulls"]["C2"]["N1"]["S_val"] = _block(mae=0.05)
    assert decide(metrics)["decision"] == INCONCLUSIVE


def test_local_state_is_not_awarded_when_depth_also_wins():
    metrics = _metrics()
    _set(metrics, "linear", "C1", 0.2, 0.9)
    _set(metrics, "small_mlp", "C1", 0.2, 0.9)
    assert decide(metrics)["decision"] == LOCAL
    _set(metrics, "linear", "C2", 0.05, 0.99)
    assert decide(metrics)["decision"] != LOCAL


def test_capacity_and_estimator_disagreement_stay_inconclusive():
    metrics = _metrics()
    _set(metrics, "linear", "C2", 0.1, 0.95)
    assert decide(metrics)["decision"] == INCONCLUSIVE
    metrics = _metrics()
    _set(metrics, "linear", "B2", 0.4, 0.7)
    metrics["mlp_large"]["B2"]["S_val"] = _block(0.05, 0.99)
    metrics["mlp_large"]["B2"]["S_rep"] = _block(0.05, 0.99)
    metrics["small_mlp"]["B2"]["S_val"] = _block(0.4, 0.7)
    metrics["small_mlp"]["B2"]["S_rep"] = _block(0.4, 0.7)
    metrics["small_mlp"]["C2"]["S_val"] = _block(0.2, 0.8)
    metrics["small_mlp"]["C2"]["S_rep"] = _block(0.2, 0.8)
    metrics["linear"]["C2"]["S_val"] = _block(0.2, 0.8)
    metrics["linear"]["C2"]["S_rep"] = _block(0.2, 0.8)
    assert decide(metrics)["decision"] == INCONCLUSIVE


def test_linear_gradient_readout_recovers_the_directional_derivative():
    rng = np.random.default_rng(7)
    prompts = 6
    layers = 2
    width = 32
    h = rng.normal(size=(prompts, layers, width))
    g = rng.normal(size=(prompts, layers, width))
    directions = rng.normal(size=(8, width))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    alphas = np.asarray(list(PRIMARY_ALPHAS) + [-0.25, 0.25], dtype=np.float64)
    delta = np.zeros((prompts, layers, 8, alphas.size))
    for layer in range(layers):
        for direction in range(8):
            dot = g[:, layer] @ directions[direction]
            delta[:, layer, direction, :] = -alphas[None, :] * dot[:, None]
    roles = ["discovery", "discovery", "validation", "validation", "replication", "replication"]
    regimes = ["completion", "instruction", "syntax", "completion", "instruction", "syntax"]
    metrics = run_readouts(
        h,
        g,
        directions,
        list(DIRECTION_SEEDS),
        delta,
        alphas,
        roles,
        regimes,
        run_mlp=False,
        run_large=False,
        run_nulls=False,
    )
    assert metrics["linear"]["B3"]["S_val"]["mae"] < 1e-6
    assert metrics["linear"]["B3"]["S_val"]["mae"] < metrics["linear"]["B2"]["S_val"]["mae"]
    assert metrics["linear"]["B3"]["S_novel"]["sign_agreement"] == 1.0
