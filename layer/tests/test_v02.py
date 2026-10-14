import random

import numpy as np
import pytest

from cognitive_layer import (Action, CalibrationRegistry, CalibrationRule, CognitiveLayer, ExpectedUtilityPolicy,
                             LeakageError, Ledger, LedgerError, Scope, Status, UtilityTable, auroc, fit_platt,
                             from_token_logprobs, from_verbal, replay)
from cognitive_layer.evaluation import oracle_instance, paired_bootstrap, run_arm
from cognitive_layer.sim import declare_all, stream


def make(mode="scoped", scoped=True, shuffle=False, seed=0, ledger=None):
    def f():
        reg = CalibrationRegistry()
        if scoped:
            declare_all(reg)
        return CognitiveLayer(reg, mode=mode, shuffle_evidence=shuffle, seed=seed, ledger=ledger)
    return f


# ---- statistics -----------------------------------------------------------
def test_platt_recovers_known_map():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.02, 0.98, 20000)
    true_p = 1 / (1 + np.exp(-(0.5 * np.log(c / (1 - c)) - 0.7)))
    y = rng.random(c.size) < true_p
    fit = fit_platt(c, y, prior_strength=1.0)
    assert abs(fit.a - 0.5) < 0.05 and abs(fit.b + 0.7) < 0.05


def test_platt_without_data_is_identity():
    fit = fit_platt([], [])
    assert fit.apply(0.8) == pytest.approx(0.8)


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(1)
    c = rng.random(60).round(1)
    y = rng.random(60) < c
    pos, neg = c[y], c[~y]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert auroc(c, y).auroc == pytest.approx(brute)


def test_status_gate_requires_both_outcomes():
    rule = CalibrationRule(min_each=15)
    assert rule.status(100, 14, 0.9, 0.99) == Status.UNCERTAIN
    assert rule.status(15, 15, 0.6, 0.9) == Status.SUPPORTED
    assert rule.status(15, 15, 0.3, 0.52) == Status.CONTRADICTED


def test_sensors():
    assert from_token_logprobs([-0.1, -0.2]) == pytest.approx(np.exp(-0.3))
    assert from_token_logprobs([-0.1, -0.3], mode="mean") == pytest.approx(np.exp(-0.2))
    with pytest.raises(ValueError):
        from_token_logprobs([0.5])
    assert from_verbal("I am 83% sure") == pytest.approx(0.83)
    assert from_verbal("confidence: 0.4") == pytest.approx(0.4)
    assert from_verbal("no number") is None


# ---- policy ---------------------------------------------------------------
def test_expected_utility_thresholds_match_m34a():
    pol = ExpectedUtilityPolicy(UtilityTable(), epsilon=0.0)
    assert pol.greedy(0.86, False)[0] == Action.ACT      # 2p-1 > 0.7 iff p > 0.85
    assert pol.greedy(0.84, False)[0] == Action.VERIFY
    no_verify = ExpectedUtilityPolicy(UtilityTable(verify=-1.0), epsilon=0.0)
    assert no_verify.greedy(0.4, False)[0] == Action.ABSTAIN
    assert no_verify.greedy(0.4, True)[0] in (Action.ACT, Action.VERIFY)   # learning: reveal outcome


def test_s1_consumption_counterfactual():
    pol = ExpectedUtilityPolicy(epsilon=0.0)
    assert pol.counterfactual_changes_decision(p_supported=0.95, p_fallback=0.6)


def test_exploration_is_logged_with_propensity():
    pol = ExpectedUtilityPolicy(UtilityTable(verify=-1.0), epsilon=0.5, seed=3)
    seen = {pol.choose(0.2, False, False)[3:] for _ in range(200)}
    assert (True, 0.5) in seen and (False, 0.5) in seen


# ---- layer ----------------------------------------------------------------
def test_scoped_calibration_finds_planted_properties():
    res = run_arm(make(), stream(1500, 0), stream(10, 99))
    claims = res["layer"].registry.claims
    wrapped, clean, trivia = claims["arith_wrapped"], claims["arith_clean"], claims["trivia"]
    assert clean.status == Status.SUPPORTED and wrapped.status == Status.SUPPORTED
    assert wrapped.calibrated(0.9) < clean.calibrated(0.9) - 0.15   # execution bias detected
    assert wrapped.ece_calibrated < wrapped.ece_raw / 2
    assert trivia.status != Status.SUPPORTED                         # uninformative confidence not trusted


def test_holdout_ordering_against_controls():
    """Pooled over three fixed seeds (no seed selection). Scoped must beat every control."""
    arms = {"scoped": make(), "global": make(scoped=False), "raw": make("raw"),
            "base_rate_only": make("base_rate_only"), "shuffled": make(shuffle=True)}
    u = {k: [] for k in arms}
    orc = []
    for seed in (0, 1, 2):
        learn, hold = stream(1500, seed), stream(1000, 1000 + seed)
        for k, f in arms.items():
            u[k].append(run_arm(f, learn, hold)["holdout_utility"])
        orc.append(oracle_instance(hold))
    u = {k: np.concatenate(v) for k, v in u.items()}
    for k in ("global", "raw", "base_rate_only", "shuffled"):
        assert paired_bootstrap(u["scoped"], u[k])["lower"] > 0, k
    assert u["scoped"].mean() < np.concatenate(orc).mean()


def test_non_observing_outcome_must_be_none():
    reg = CalibrationRegistry()
    layer = CognitiveLayer(reg, ExpectedUtilityPolicy(UtilityTable(verify=-1.0), epsilon=0.0))
    layer.registry.claims["GLOBAL"] = layer.registry.claims["GLOBAL"].__class__(
        "GLOBAL", "x", Scope(), status=Status.CONTRADICTED)
    pred = layer.before({"task": "a"}, confidence=0.1)
    assert pred.action == Action.ABSTAIN
    with pytest.raises(ValueError):
        layer.after(pred.prediction_id, True, "test")
    layer.after(pred.prediction_id, None, "test")


def test_leakage_sealing_and_double_apply():
    layer = make()()
    with pytest.raises(LeakageError):
        layer.before({"task": "arith", "is_correct": True}, 0.5)
    pred = layer.before({"task": "arith", "wrapper": "off"}, 0.7)
    layer.after(pred.prediction_id, True, "t")
    with pytest.raises(ValueError):
        layer.after(pred.prediction_id, True, "t")


def test_historical_evidence_recorded_not_counted():
    layer = make()()
    pred = layer.before({"task": "arith", "wrapper": "off"}, 0.7)
    layer.after(pred.prediction_id, True, "M34a-v2-cache", source_is_current=False)
    c = layer.registry.claims["arith_clean"]
    assert c.n_pos == 0 and len(c.recorded_only) == 1


def test_ledger_chain_persistence_and_replay(tmp_path):
    path = tmp_path / "ledger.jsonl"
    res = run_arm(make(ledger=Ledger(path)), stream(400, 5), stream(20, 6))
    live = res["layer"]
    loaded = Ledger(path)
    assert loaded.head == live.ledger.head
    rebuilt = replay(loaded)
    for cid, c in live.registry.claims.items():
        r = rebuilt.claims[cid]
        assert (r.status, r.n_pos, r.n_neg) == (c.status, c.n_pos, c.n_neg)
        assert r.platt.a == pytest.approx(c.platt.a) and r.platt.b == pytest.approx(c.platt.b)
    loaded.events[5]["payload"]["p_hat"] = 0.99
    with pytest.raises(LedgerError):
        loaded.verify()


def test_ambiguous_scope_refused():
    reg = CalibrationRegistry()
    reg.declare("a", "x", Scope.of(task="t"))
    reg.declare("b", "y", Scope.of(domain="d"))
    with pytest.raises(ValueError):
        reg.matching({"task": "t", "domain": "d"})
