"""Qwen scope-update rules. No new Qwen forward."""

import numpy as np

from src.mrsm.ccso import LAYERS, PRIMARY_ALPHAS, REGIMES
from src.mrsm.qwen_belief_update import (
    ABSTAIN,
    DOES_NOT_SEPARATE,
    HOLD,
    INCONCLUSIVE,
    REVISE_SCOPE,
    SEPARATES,
    TOLERANCE_MULTIPLIER,
    classify_qwen,
    decide_qwen,
    evaluate_cases,
    preregistration_document,
)
from src.mrsm.self_model import LeakageError


def _world(effect_by_regime: dict[str, float]):
    roles: list[str] = []
    regimes: list[str] = []
    effects: list[float] = []
    for regime in REGIMES:
        for role in ("discovery", "validation", "replication"):
            roles.append(role)
            regimes.append(regime)
            effects.append(effect_by_regime[regime])
    alphas = np.asarray(PRIMARY_ALPHAS, dtype=np.float64)
    delta = np.zeros((len(roles), len(LAYERS), 8, alphas.shape[0]), dtype=np.float64)
    for prompt, effect in enumerate(effects):
        delta[prompt] = float(effect) * alphas.reshape(1, 1, -1)
    return delta, alphas, roles, regimes, list(range(23101, 23109))


def test_preregistration_rejects_the_p_scale():
    document = preregistration_document()
    assert document["hypothesis_output"] is False
    assert document["mechanism_identity"] == "NOT_EVALUATED"
    assert document["tolerance"]["p_absolute_floor_1"] is False
    assert document["tolerance"]["p_head_table"] is False
    assert document["tolerance"]["multiplier"] == TOLERANCE_MULTIPLIER == 2.0
    assert "load a new Qwen forward" in document["does_not"]
    assert "start M22" in document["does_not"]
    assert "claim mechanism identity" in document["does_not"]
    assert "modify belief_update.classify" in document["does_not"]
    assert "alpha ±0.25" in document["excluded"]


def test_rule_has_no_mechanism_name():
    belief = {"scope": "completion", "predictions": {"L0:d1": 1.0}, "uncertainties": {"L0:d1": 0.1}}
    assert classify_qwen(belief, [{"intervention_id": "L0:d1", "actual_outcome": 1.0, "scope": "completion"}]) == HOLD
    assert (
        classify_qwen(belief, [{"intervention_id": "L0:d1", "actual_outcome": 1.0, "scope": "completion::label-only"}])
        == HOLD
    )
    assert (
        classify_qwen(belief, [{"intervention_id": "L0:d1", "actual_outcome": 5.0, "scope": "instruction"}])
        == REVISE_SCOPE
    )
    assert classify_qwen(belief, [{"intervention_id": "L0:d1", "actual_outcome": 5.0, "scope": "completion"}]) == ABSTAIN
    mixed = [
        {"intervention_id": "L0:d1", "actual_outcome": 5.0, "scope": "completion"},
        {"intervention_id": "L0:d1", "actual_outcome": 5.0, "scope": "instruction"},
    ]
    assert classify_qwen(belief, mixed) == ABSTAIN
    assert classify_qwen(belief, [{"intervention_id": "L0:d999", "actual_outcome": 0.0, "scope": "instruction"}]) == ABSTAIN
    try:
        classify_qwen({**belief, "mechanism": "L0H0->L1H1"}, [])
    except LeakageError:
        leaked = True
    else:
        leaked = False
    assert leaked


def test_floor_of_one_is_not_applied():
    belief = {"scope": "completion", "predictions": {"L0:d1": 0.0}, "uncertainties": {"L0:d1": 0.01}}
    # 2 * 0.01 = 0.02. A residual of 0.05 is inside the P floor of 1 and outside this tolerance.
    label = classify_qwen(belief, [{"intervention_id": "L0:d1", "actual_outcome": 0.05, "scope": "instruction"}])
    assert label == REVISE_SCOPE


def test_separated_world_and_quiet_regime_change():
    separated = evaluate_cases(*_world({"completion": 1.0, "instruction": 5.0, "syntax": -3.0}))
    assert separated["primary"] == {
        "same_regime": HOLD,
        "regime_change": REVISE_SCOPE,
        "two_changes": ABSTAIN,
    }
    assert separated["controls"]["scope_string_alone"] == HOLD
    decision = decide_qwen(
        separated["primary"],
        separated["controls"],
        split_ok=True,
        finite_ok=separated["finite"],
        self_ok=separated["self_ok"],
    )
    assert decision["decision"] == SEPARATES
    quiet = evaluate_cases(*_world({"completion": 1.0, "instruction": 1.0, "syntax": 1.0}))
    assert quiet["primary"]["regime_change"] == HOLD
    assert (
        decide_qwen(quiet["primary"], quiet["controls"], split_ok=True, finite_ok=True, self_ok=True)["decision"]
        == DOES_NOT_SEPARATE
    )


def test_decision_order():
    primary = {"same_regime": HOLD, "regime_change": REVISE_SCOPE, "two_changes": ABSTAIN}
    controls = {"scope_string_alone": HOLD}
    assert decide_qwen(primary, controls, split_ok=False, finite_ok=True, self_ok=True)["decision"] == INCONCLUSIVE
    assert (
        decide_qwen(primary, {"scope_string_alone": REVISE_SCOPE}, split_ok=True, finite_ok=True, self_ok=True)[
            "decision"
        ]
        == INCONCLUSIVE
    )
