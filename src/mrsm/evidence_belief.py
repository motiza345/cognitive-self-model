"""Sign-level evidence, belief, scope, falsification, and persistence trace.

This does not repeat the Qwen magnitude tolerance. A belief is a sign claim
fit on the two discovery prompts of one regime. Later prompts are applied in
frozen order. The rule is fixed here, before the frozen measurements are scored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.mrsm.ccso import (
    LAYERS,
    PRIMARY_ALPHAS,
    REGIMES,
    TRAIN_DIRECTION_SEEDS,
)
from src.mrsm.self_model import LeakageError

POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NONE = "NONE"
SIGNS = (POSITIVE, NEGATIVE, NONE)

PREDICT = "PREDICT"
ABSTAIN = "ABSTAIN"
FIT = "FIT"

TRACE_HOLDS = "SELF_KNOWLEDGE_TRACE_HOLDS"
FAILS_PREDICTION = "SELF_KNOWLEDGE_FAILS_PREDICTION"
FAILS_SCOPE = "SELF_KNOWLEDGE_FAILS_SCOPE"
FAILS_FALSIFICATION = "SELF_KNOWLEDGE_FAILS_FALSIFICATION"
FAILS_UPDATE = "SELF_KNOWLEDGE_FAILS_UPDATE"
FAILS_PERSISTENCE = "SELF_KNOWLEDGE_FAILS_PERSISTENCE"
INCONCLUSIVE = "INCONCLUSIVE"
DECISIONS = (
    TRACE_HOLDS,
    FAILS_PREDICTION,
    FAILS_SCOPE,
    FAILS_FALSIFICATION,
    FAILS_UPDATE,
    FAILS_PERSISTENCE,
    INCONCLUSIVE,
)

PREDICTION_BAR = 0.5
MIN_SCOREABLE = 12
POSSIBLE_CELLS = len(LAYERS) * len(TRAIN_DIRECTION_SEEDS) * len(REGIMES)

FORBIDDEN_KEYS = {
    "failure_type",
    "expected",
    "expected_label",
    "ground_truth",
    "mechanism",
    "mechanism_label",
    "planted",
    "injection",
}


def preregistration_document() -> dict[str, Any]:
    return {
        "diagnostic": "evidence_belief_trace",
        "status": "diagnostic",
        "model": "frozen Qwen measurements from the CCSO run",
        "question": (
            "Can a sign claim fit on two discovery prompts of one regime predict "
            "later prompts of that regime, abstain outside that regime, contest "
            "itself when an in-scope sign disagrees, keep every evidence record, "
            "and still use that revised state on the next prompt?"
        ),
        "mechanism_identity": "NOT_EVALUATED",
        "new_self_model": False,
        "decisions": list(DECISIONS),
        "prediction_bar": PREDICTION_BAR,
        "prediction_rule": "pass only if validation sign accuracy is strictly above 0.5",
        "min_scoreable": MIN_SCOREABLE,
        "possible_cells": POSSIBLE_CELLS,
        "effect": "mean of Δ/α over the six primary alphas, one prompt, one layer, one train direction",
        "claim": "the shared nonzero sign of the two discovery prompts; otherwise no claim",
        "scope": "the single discovery regime; other regimes are out of scope",
        "confidence": "support count and contest count; no mapped probability",
        "expansion": "forbidden; one or more out-of-scope matches do not enlarge scope",
        "resurrection": "forbidden; a later match does not reactivate a contested claim",
        "falsification": "an in-scope opposite sign while the claim is active; a zero effect does not falsify",
        "out_of_scope": "recorded, and it does not change the claim, support, contests, or active flag",
        "prediction_pool": "validation prompts only, and only when the claim is still active and the effect has a sign",
        "decision_order": [
            "split or finiteness fails -> INCONCLUSIVE",
            "fewer than 12 scoreable claims -> INCONCLUSIVE",
            "no validation prediction -> INCONCLUSIVE",
            "validation sign accuracy <= 0.5 -> SELF_KNOWLEDGE_FAILS_PREDICTION",
            "an out-of-scope prompt changes the claim or receives a prediction -> SELF_KNOWLEDGE_FAILS_SCOPE",
            "a sign conflict is flagged in the wrong place -> SELF_KNOWLEDGE_FAILS_FALSIFICATION",
            "evidence is dropped, a contest is invented, or a contested claim reactivates -> SELF_KNOWLEDGE_FAILS_UPDATE",
            "a contested claim is predicted again, or the stored sign changes -> SELF_KNOWLEDGE_FAILS_PERSISTENCE",
            "confirmation, falsification, or an out-of-scope sign conflict never occurs -> INCONCLUSIVE",
            "otherwise -> SELF_KNOWLEDGE_TRACE_HOLDS",
        ],
        "excluded": [
            "magnitude tolerance and the multiplier 2",
            "P head table",
            "P floor 1",
            "novel directions 23107 and 23108",
            "alpha ±0.25",
            "M18.7 as a gate",
            "notebook PASS/FAIL labels as decision targets",
            "a new Qwen forward",
            "M22",
        ],
        "does_not": [
            "modify belief_update.classify",
            "modify classify_qwen",
            "modify SelfModelCore",
            "modify the CCSO measurements",
            "load a new Qwen forward",
            "start M22",
            "claim mechanism identity",
            "build a new Self-Model",
            "change a bar after scoring",
            "treat a historical notebook status as a validated label",
        ],
    }


def _reject(obj: Any) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS or lowered.startswith("planted"):
                raise LeakageError(f"forbidden input field: {key}")
            _reject(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _reject(item)


def discovery_claim(signs: list[str]) -> str | None:
    if len(signs) != 2:
        raise ValueError("a claim uses exactly two discovery signs")
    if signs[0] in (POSITIVE, NEGATIVE) and signs[0] == signs[1]:
        return signs[0]
    return None


def _sign_name(mean: float) -> str:
    if not np.isfinite(mean) or mean == 0.0:
        return NONE
    if mean > 0.0:
        return POSITIVE
    return NEGATIVE


def _primary_alpha_indices(alphas: np.ndarray) -> list[int]:
    wanted = {format(value, ".2f") for value in PRIMARY_ALPHAS}
    found = [index for index, value in enumerate(alphas) if format(float(value), ".2f") in wanted]
    if len(found) != len(PRIMARY_ALPHAS):
        raise ValueError("primary alphas are not all present")
    return found


def prompt_sign(
    delta: np.ndarray,
    alphas: np.ndarray,
    prompt: int,
    layer_index: int,
    direction_index: int,
    alpha_indices: list[int],
) -> str:
    samples = []
    for alpha_index in alpha_indices:
        alpha = float(alphas[alpha_index])
        if alpha == 0.0:
            raise ValueError("primary alpha is zero")
        samples.append(float(delta[prompt, layer_index, direction_index, alpha_index]) / alpha)
    return _sign_name(float(np.mean(np.asarray(samples, dtype=np.float64))))


def run_belief(scope: str, discovery_signs: list[str], later: list[dict[str, str]]) -> dict[str, Any]:
    """Apply later events in the given order. `later` must already be time-ordered."""
    _reject({"scope": scope, "discovery_signs": discovery_signs, "later": later})
    claim = discovery_claim(discovery_signs)
    evidence: list[dict[str, Any]] = []
    for index, sign in enumerate(discovery_signs):
        evidence.append(
            {
                "id": f"d{index}",
                "regime": scope,
                "role": "discovery",
                "sign": sign,
                "action": FIT,
                "falsified": False,
            }
        )
    active = claim is not None
    support = 2 if active else 0
    contests = 0
    counters = {
        "prediction_n": 0,
        "prediction_correct": 0,
        "confirmation_n": 0,
        "falsification_n": 0,
        "scope_n": 0,
        "scope_conflict_n": 0,
        "scope_violations": 0,
        "falsification_violations": 0,
        "update_violations": 0,
        "persistence_violations": 0,
    }
    discovery_ids = [row["id"] for row in evidence]
    for event in later:
        regime = str(event["regime"])
        role = str(event["role"])
        sign = str(event["sign"])
        if role not in ("validation", "replication") or sign not in SIGNS:
            raise ValueError("later event is outside the preregistered roles and signs")
        before = (active, claim, support, contests, len(evidence))
        record: dict[str, Any] = {
            "id": f"e{len(evidence)}",
            "regime": regime,
            "role": role,
            "sign": sign,
            "falsified": False,
            "action": ABSTAIN,
            "predicted": None,
        }
        if regime != scope:
            counters["scope_n"] += 1
            if claim is not None and sign in (POSITIVE, NEGATIVE) and sign != claim:
                counters["scope_conflict_n"] += 1
            evidence.append(record)
        elif not active or claim is None:
            evidence.append(record)
            if role == "replication" and contests > 0:
                pass
        elif sign == NONE:
            record["predicted"] = claim
            evidence.append(record)
        else:
            record["action"] = PREDICT
            record["predicted"] = claim
            match = sign == claim
            record["falsified"] = not match
            if role == "validation":
                counters["prediction_n"] += 1
                counters["prediction_correct"] += int(match)
            if match:
                support += 1
                counters["confirmation_n"] += 1
            else:
                contests += 1
                active = False
                counters["falsification_n"] += 1
            evidence.append(record)
        if regime != scope and (active, claim, support, contests) != before[:4]:
            counters["scope_violations"] += 1
        if record["falsified"] and (regime != scope or sign == claim or sign == NONE or not before[0]):
            counters["falsification_violations"] += 1
        if (not record["falsified"]) and regime == scope and before[0] and sign in (POSITIVE, NEGATIVE) and sign != claim:
            counters["falsification_violations"] += 1
        if regime != scope and record["falsified"]:
            counters["falsification_violations"] += 1
        if active and not before[0]:
            counters["update_violations"] += 1
        if claim != before[1]:
            counters["update_violations"] += 1
        if before[3] > 0 and record["action"] == PREDICT:
            counters["persistence_violations"] += 1
        if len(evidence) != before[4] + 1:
            counters["update_violations"] += 1
    if any(item not in {row["id"] for row in evidence} for item in discovery_ids):
        counters["update_violations"] += 1
        counters["persistence_violations"] += 1
    if contests > 0 and active:
        counters["persistence_violations"] += 1
    return {
        "scope": scope,
        "claim": claim,
        "scoreable": claim is not None,
        "active": active,
        "support": support,
        "contests": contests,
        "evidence": evidence,
        "counters": counters,
    }


def _prompts(roles: list[str], regimes: list[str], role: str, regime: str) -> list[int]:
    return [
        index
        for index, (row_role, row_regime) in enumerate(zip(roles, regimes))
        if row_role == role and row_regime == regime
    ]


def _empty_summary(
    *,
    precondition_ok: bool,
    finite_ok: bool,
    split_ok: bool,
    by_regime: dict[str, dict[str, int]],
) -> dict[str, Any]:
    return {
        "precondition_ok": precondition_ok,
        "finite_ok": finite_ok,
        "split_ok": split_ok,
        "scoreable": 0,
        "possible_cells": POSSIBLE_CELLS,
        "prediction_n": 0,
        "prediction_correct": 0,
        "prediction_accuracy": None,
        "confirmation_n": 0,
        "falsification_n": 0,
        "scope_n": 0,
        "scope_conflict_n": 0,
        "scope_violations": 0,
        "falsification_violations": 0,
        "update_violations": 0,
        "persistence_violations": 0,
        "by_regime": by_regime,
    }


def _compact(trace: dict[str, Any]) -> dict[str, Any]:
    counters = trace["counters"]
    return {
        "layer": trace["layer"],
        "direction_seed": trace["direction_seed"],
        "scope": trace["scope"],
        "claim": trace["claim"],
        "active": trace["active"],
        "support": trace["support"],
        "contests": trace["contests"],
        "evidence_n": len(trace["evidence"]),
        "prediction_n": counters["prediction_n"],
        "prediction_correct": counters["prediction_correct"],
        "confirmation_n": counters["confirmation_n"],
        "falsification_n": counters["falsification_n"],
        "scope_conflict_n": counters["scope_conflict_n"],
        "scope_violations": counters["scope_violations"],
        "falsification_violations": counters["falsification_violations"],
        "update_violations": counters["update_violations"],
        "persistence_violations": counters["persistence_violations"],
    }


def evaluate_measurements(
    delta: np.ndarray,
    alphas: np.ndarray,
    roles: list[str],
    regimes: list[str],
    direction_seeds: list[int],
) -> dict[str, Any]:
    _reject({"roles": roles, "regimes": regimes})
    finite = bool(np.isfinite(delta).all() and np.isfinite(alphas).all())
    split_ok = True
    for regime in REGIMES:
        for role, expected in (("discovery", 2), ("validation", 2), ("replication", 2)):
            if len(_prompts(roles, regimes, role, regime)) != expected:
                split_ok = False
    train = [index for index, seed in enumerate(direction_seeds) if int(seed) in TRAIN_DIRECTION_SEEDS]
    if len(train) != len(TRAIN_DIRECTION_SEEDS):
        split_ok = False
    if not split_ok or not finite:
        empty = {regime: {"prediction_n": 0, "prediction_correct": 0, "falsification_n": 0} for regime in REGIMES}
        return {
            "summary": _empty_summary(precondition_ok=False, finite_ok=finite, split_ok=split_ok, by_regime=empty),
            "traces": [],
        }
    alpha_indices = _primary_alpha_indices(np.asarray(alphas))
    later_index = [
        index
        for index, role in enumerate(roles)
        if role in ("validation", "replication")
    ]
    traces = []
    compact: list[dict[str, Any]] = []
    totals = {
        "prediction_n": 0,
        "prediction_correct": 0,
        "confirmation_n": 0,
        "falsification_n": 0,
        "scope_n": 0,
        "scope_conflict_n": 0,
        "scope_violations": 0,
        "falsification_violations": 0,
        "update_violations": 0,
        "persistence_violations": 0,
    }
    by_regime = {regime: {"prediction_n": 0, "prediction_correct": 0, "falsification_n": 0} for regime in REGIMES}
    scoreable = 0
    for layer_index, layer in enumerate(LAYERS):
        for direction_index in train:
            seed = int(direction_seeds[direction_index])
            for regime in REGIMES:
                discovery = [
                    prompt_sign(delta, alphas, prompt, layer_index, direction_index, alpha_indices)
                    for prompt in _prompts(roles, regimes, "discovery", regime)
                ]
                later = [
                    {
                        "regime": regimes[prompt],
                        "role": roles[prompt],
                        "sign": prompt_sign(delta, alphas, prompt, layer_index, direction_index, alpha_indices),
                    }
                    for prompt in later_index
                ]
                trace = run_belief(regime, discovery, later)
                trace["layer"] = int(layer)
                trace["direction_seed"] = seed
                traces.append(trace)
                if not trace["scoreable"]:
                    continue
                compact.append(_compact(trace))
                scoreable += 1
                for key in totals:
                    totals[key] += int(trace["counters"][key])
                by_regime[regime]["prediction_n"] += int(trace["counters"]["prediction_n"])
                by_regime[regime]["prediction_correct"] += int(trace["counters"]["prediction_correct"])
                by_regime[regime]["falsification_n"] += int(trace["counters"]["falsification_n"])
    accuracy = None
    if totals["prediction_n"]:
        accuracy = totals["prediction_correct"] / totals["prediction_n"]
    summary = {
        "precondition_ok": bool(split_ok and finite),
        "finite_ok": finite,
        "split_ok": split_ok,
        "scoreable": scoreable,
        "possible_cells": POSSIBLE_CELLS,
        "prediction_n": totals["prediction_n"],
        "prediction_correct": totals["prediction_correct"],
        "prediction_accuracy": accuracy,
        "confirmation_n": totals["confirmation_n"],
        "falsification_n": totals["falsification_n"],
        "scope_n": totals["scope_n"],
        "scope_conflict_n": totals["scope_conflict_n"],
        "scope_violations": totals["scope_violations"],
        "falsification_violations": totals["falsification_violations"],
        "update_violations": totals["update_violations"],
        "persistence_violations": totals["persistence_violations"],
        "by_regime": by_regime,
    }
    return {"summary": summary, "traces": compact}


def decide(summary: dict[str, Any]) -> dict[str, str]:
    if not summary.get("precondition_ok"):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "PRECONDITION",
            "reason": "The split or a finite effect failed.",
        }
    if int(summary["scoreable"]) < MIN_SCOREABLE:
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "TOO_FEW_CLAIMS",
            "reason": "Discovery did not form enough shared-sign claims.",
        }
    if int(summary["prediction_n"]) == 0:
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "NO_PREDICTIONS",
            "reason": "No active claim was asked to predict a validation sign.",
        }
    accuracy = float(summary["prediction_accuracy"])
    if accuracy <= PREDICTION_BAR:
        return {
            "decision": FAILS_PREDICTION,
            "reason_code": "PREDICTION_BAR",
            "reason": "Validation sign accuracy was not strictly above one half.",
        }
    if int(summary["scope_violations"]) > 0:
        return {
            "decision": FAILS_SCOPE,
            "reason_code": "SCOPE_MUTATION",
            "reason": "An out-of-scope prompt changed the claim or its counts.",
        }
    if int(summary["falsification_violations"]) > 0:
        return {
            "decision": FAILS_FALSIFICATION,
            "reason_code": "FALSIFICATION_FLAG",
            "reason": "A sign conflict was flagged outside the preregistered case, or missed inside it.",
        }
    if int(summary["update_violations"]) > 0:
        return {
            "decision": FAILS_UPDATE,
            "reason_code": "UPDATE_RULE",
            "reason": "Evidence was dropped, a contest was invented, or a contested claim reactivated.",
        }
    if int(summary["persistence_violations"]) > 0:
        return {
            "decision": FAILS_PERSISTENCE,
            "reason_code": "PERSISTENCE",
            "reason": "A contested claim was predicted again, or the stored sign changed.",
        }
    if (
        int(summary["confirmation_n"]) == 0
        or int(summary["falsification_n"]) == 0
        or int(summary["scope_conflict_n"]) == 0
    ):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "NOT_EXERCISED",
            "reason": "Confirmation, in-scope falsification, or an out-of-scope sign conflict did not occur.",
        }
    return {
        "decision": TRACE_HOLDS,
        "reason_code": "TRACE_HOLDS",
        "reason": "The five checks held on the frozen sign trace, and each required case occurred.",
    }
