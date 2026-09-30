"""Evidence-belief trace. Synthetic signs only. No Qwen forward."""

import numpy as np

from src.mrsm.ccso import LAYERS, PRIMARY_ALPHAS, REGIMES
from src.mrsm.evidence_belief import (
    FAILS_PREDICTION,
    INCONCLUSIVE,
    MIN_SCOREABLE,
    NEGATIVE,
    NONE,
    POSITIVE,
    PREDICTION_BAR,
    TRACE_HOLDS,
    decide,
    discovery_claim,
    evaluate_measurements,
    preregistration_document,
    run_belief,
)
from src.mrsm.self_model import LeakageError


def _summary(**overrides):
    base = {
        "precondition_ok": True,
        "scoreable": MIN_SCOREABLE,
        "prediction_n": 10,
        "prediction_accuracy": 0.8,
        "confirmation_n": 8,
        "falsification_n": 2,
        "scope_conflict_n": 3,
        "scope_violations": 0,
        "falsification_violations": 0,
        "update_violations": 0,
        "persistence_violations": 0,
    }
    base.update(overrides)
    return base


def test_preregistration_is_a_sign_trace_not_a_new_model():
    document = preregistration_document()
    assert document["new_self_model"] is False
    assert document["mechanism_identity"] == "NOT_EVALUATED"
    assert document["prediction_bar"] == PREDICTION_BAR == 0.5
    assert document["min_scoreable"] == MIN_SCOREABLE == 12
    assert "magnitude tolerance and the multiplier 2" in document["excluded"]
    assert "M18.7 as a gate" in document["excluded"]
    assert "notebook PASS/FAIL labels as decision targets" in document["excluded"]
    assert "load a new Qwen forward" in document["does_not"]
    assert "start M22" in document["does_not"]
    assert "build a new Self-Model" in document["does_not"]
    assert "modify belief_update.classify" in document["does_not"]
    assert "modify classify_qwen" in document["does_not"]


def test_claim_requires_two_agreeing_nonzero_signs():
    assert discovery_claim([POSITIVE, POSITIVE]) == POSITIVE
    assert discovery_claim([NEGATIVE, NEGATIVE]) == NEGATIVE
    assert discovery_claim([POSITIVE, NEGATIVE]) is None
    assert discovery_claim([POSITIVE, NONE]) is None
    assert discovery_claim([NONE, NONE]) is None


def test_out_of_scope_conflict_keeps_the_local_claim():
    trace = run_belief(
        "completion",
        [POSITIVE, POSITIVE],
        [
            {"regime": "instruction", "role": "validation", "sign": NEGATIVE},
            {"regime": "completion", "role": "validation", "sign": POSITIVE},
            {"regime": "completion", "role": "replication", "sign": POSITIVE},
        ],
    )
    assert trace["claim"] == POSITIVE
    assert trace["active"] is True
    assert trace["contests"] == 0
    assert trace["support"] == 4
    assert trace["evidence"][2]["action"] == "ABSTAIN"
    assert trace["evidence"][2]["falsified"] is False
    assert [row["id"] for row in trace["evidence"][:2]] == ["d0", "d1"]
    assert trace["counters"]["scope_conflict_n"] == 1
    assert trace["counters"]["scope_violations"] == 0
    assert trace["counters"]["prediction_correct"] == 1


def test_in_scope_conflict_contests_and_does_not_resurrect():
    trace = run_belief(
        "instruction",
        [POSITIVE, POSITIVE],
        [
            {"regime": "instruction", "role": "validation", "sign": NEGATIVE},
            {"regime": "instruction", "role": "validation", "sign": POSITIVE},
            {"regime": "syntax", "role": "validation", "sign": NEGATIVE},
            {"regime": "instruction", "role": "replication", "sign": POSITIVE},
        ],
    )
    assert trace["active"] is False
    assert trace["contests"] == 1
    assert trace["support"] == 2
    assert trace["claim"] == POSITIVE
    assert trace["evidence"][2]["action"] == "PREDICT"
    assert trace["evidence"][2]["falsified"] is True
    assert trace["evidence"][3]["action"] == "ABSTAIN"
    assert trace["evidence"][5]["action"] == "ABSTAIN"
    assert [row["id"] for row in trace["evidence"] if row["role"] == "discovery"] == ["d0", "d1"]
    assert trace["counters"]["persistence_violations"] == 0
    assert trace["counters"]["update_violations"] == 0
    assert trace["counters"]["falsification_violations"] == 0
    assert trace["counters"]["prediction_n"] == 1
    assert trace["counters"]["prediction_correct"] == 0


def test_zero_effect_does_not_falsify():
    trace = run_belief(
        "completion",
        [NEGATIVE, NEGATIVE],
        [{"regime": "completion", "role": "validation", "sign": NONE}],
    )
    assert trace["active"] is True
    assert trace["contests"] == 0
    assert trace["counters"]["prediction_n"] == 0
    assert trace["counters"]["falsification_n"] == 0


def test_forbidden_field_is_rejected():
    try:
        run_belief("completion", [POSITIVE, POSITIVE], [{"regime": "completion", "role": "validation", "sign": POSITIVE, "mechanism": "named"}])
    except LeakageError:
        rejected = True
    else:
        rejected = False
    assert rejected


def test_decision_order_is_fixed():
    assert decide(_summary(precondition_ok=False))["decision"] == INCONCLUSIVE
    assert decide(_summary(scoreable=MIN_SCOREABLE - 1))["reason_code"] == "TOO_FEW_CLAIMS"
    assert decide(_summary(prediction_accuracy=PREDICTION_BAR))["decision"] == FAILS_PREDICTION
    assert decide(_summary(prediction_accuracy=0.49, falsification_n=0))["decision"] == FAILS_PREDICTION
    assert decide(_summary(scope_violations=1))["decision"] != TRACE_HOLDS
    assert decide(_summary(falsification_n=0))["reason_code"] == "NOT_EXERCISED"
    assert decide(_summary(scope_conflict_n=0))["reason_code"] == "NOT_EXERCISED"
    assert decide(_summary(confirmation_n=0))["reason_code"] == "NOT_EXERCISED"
    assert decide(_summary())["decision"] == TRACE_HOLDS


def _roles():
    roles = []
    regimes = []
    for role in ("discovery", "validation", "replication"):
        for regime in REGIMES:
            roles.extend([role, role])
            regimes.extend([regime, regime])
    return roles, regimes


def test_synthetic_world_exercises_the_trace_without_reading_qwen():
    roles, regimes = _roles()
    alphas = np.asarray(list(PRIMARY_ALPHAS) + [-0.25, 0.25], dtype=np.float64)
    delta = np.zeros((len(roles), len(LAYERS), 8, alphas.shape[0]), dtype=np.float64)
    for prompt, regime in enumerate(regimes):
        role = roles[prompt]
        sign = 1.0
        if regime == "syntax":
            sign = -1.0
        if regime == "instruction" and role == "validation" and roles[: prompt + 1].count("validation") == 3:
            sign = -1.0
        delta[prompt] = sign * alphas.reshape(1, 1, -1)
    evaluated = evaluate_measurements(delta, alphas, roles, regimes, list(range(23101, 23109)))
    decision = decide(evaluated["summary"])
    assert evaluated["summary"]["scoreable"] == 4 * 6 * 3
    assert evaluated["summary"]["prediction_accuracy"] > PREDICTION_BAR
    assert evaluated["summary"]["falsification_n"] > 0
    assert evaluated["summary"]["scope_conflict_n"] > 0
    assert evaluated["summary"]["confirmation_n"] > 0
    assert evaluated["summary"]["scope_violations"] == 0
    assert evaluated["summary"]["persistence_violations"] == 0
    assert decision["decision"] == TRACE_HOLDS
