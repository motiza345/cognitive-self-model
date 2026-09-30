"""Causal-belief update diagnostic on frozen arm P.

The classifier sees a belief and evidence records. It does not see which
contradiction was injected. It does not modify SelfModelCore, the planted
holdout, or the frozen P result. It does not load Qwen and it does not start M22.
"""

from __future__ import annotations

from typing import Any

from src.mrsm import HEADS
from src.mrsm.self_model import LeakageError, SelfModelCore, pair_name

REVISE_HYPOTHESIS = "REVISE_HYPOTHESIS"
REVISE_SCOPE = "REVISE_SCOPE"
ADD_INTERACTION = "ADD_INTERACTION"
ABSTAIN = "ABSTAIN"
HOLD = "HOLD"

RULE_LABELS = (
    REVISE_HYPOTHESIS,
    REVISE_SCOPE,
    ADD_INTERACTION,
    ABSTAIN,
    HOLD,
)

SEPARATES = "UPDATE_RULE_SEPARATES_FAILURES"
COLLAPSES = "UPDATE_RULE_COLLAPSES_FAILURES"
INCONCLUSIVE = "INCONCLUSIVE"
DECISIONS = (SEPARATES, COLLAPSES, INCONCLUSIVE)

# Copied from src/mrsm/run_p.py. Not a new falsification bar.
FALSIFY_ABS_FLOOR = 1.0
FALSIFY_UNCERTAINTY_MULTIPLIER = 2.0

# Used only to build a residual that is strictly past the existing tolerance.
# The classifier does not read this number.
CONSTRUCTION_CLEARANCE = 1.0
OUTSIDE_SCOPE_SUFFIX = "::outside"

PRIMARY_EXPECTATION = {
    "hypothesis": REVISE_HYPOTHESIS,
    "scope": REVISE_SCOPE,
    "interaction": ADD_INTERACTION,
    "none": ABSTAIN,
}
CONTROL_EXPECTATION = {
    "holdout": HOLD,
    "double": ABSTAIN,
    "scope_string_alone": HOLD,
}

BELIEF_FIELDS = (
    "scope",
    "effects",
    "joints",
    "mechanism_ordered",
    "uncertainty",
    "min_abs_interaction",
    "min_relative_gap",
)
RECORD_FIELDS = ("intervention_id", "actual_outcome", "scope")
FORBIDDEN_KEYS = {
    "failure_type",
    "injection",
    "expected_label",
    "expected",
    "ground_truth",
    "planted",
    "mechanism_label",
    "holdout_outcome",
    "holdout_accuracy",
    "holdout_margin",
}


def tolerance_of(uncertainty: float) -> float:
    """Same expression as the frozen P falsification record."""
    return max(FALSIFY_ABS_FLOOR, FALSIFY_UNCERTAINTY_MULTIPLIER * float(uncertainty))


def preregistration_document() -> dict[str, Any]:
    return {
        "diagnostic": "causal_belief_update",
        "status": "diagnostic",
        "arm": "P",
        "question": (
            "Given the frozen P belief and a new set of evidence records, "
            "does one preregistered rule map four single contradictions onto "
            "four different updates?"
        ),
        "rule_labels": list(RULE_LABELS),
        "decisions": list(DECISIONS),
        "primary_expectation": dict(PRIMARY_EXPECTATION),
        "control_expectation": dict(CONTROL_EXPECTATION),
        "belief_fields_read_by_the_rule": list(BELIEF_FIELDS),
        "record_fields_read_by_the_rule": list(RECORD_FIELDS),
        "forbidden_keys": sorted(FORBIDDEN_KEYS),
        "tolerance": {
            "expression": "max(1.0, 2.0 * uncertainty)",
            "source": "src/mrsm/run_p.py falsification tolerance",
            "new_threshold": False,
        },
        "identifiability_source": "configs/mrsm_prereg.yaml identifiability",
        "construction_clearance": {
            "value": CONSTRUCTION_CLEARANCE,
            "decision_input": False,
            "role": "places an injected residual strictly beyond the existing tolerance",
        },
        "predicates": {
            "scope": (
                "At least one record has a scope string different from the belief "
                "and a residual beyond tolerance."
            ),
            "hypothesis": (
                "The in-scope records contain every single and every pair, and the "
                "existing selection rule identifies a different ordered pair."
            ),
            "interaction": (
                "An in-scope pair has singles inside tolerance and a joint beyond "
                "tolerance, and a complete in-scope table does not identify a "
                "different pair."
            ),
            "exactly_one": "One supported predicate selects its update.",
            "more_than_one": ABSTAIN,
            "none_but_a_residual_beyond_tolerance": ABSTAIN,
            "none_and_no_residual_beyond_tolerance": HOLD,
            "undefined_prediction": ABSTAIN,
        },
        "injections": {
            "hypothesis": "Move the winning interaction onto the lexicographic first other pair and zero the old pair.",
            "scope": "Shift every actual by tolerance plus clearance and set scope to the belief scope plus '::outside'.",
            "interaction": "Add a signed clearance step to the lexicographic first non-winning pair whose new interaction keeps the same identified pair.",
            "none": "Shift one in-scope single by tolerance plus clearance and supply no pair.",
            "undefined_construction": "If the interaction move has no legal pair, that run is INCONCLUSIVE. The clearance is not searched.",
        },
        "controls": {
            "holdout": "Frozen holdout_scored records. The stored residual is not an input.",
            "double": "Hypothesis table plus one out-of-scope record beyond tolerance.",
            "scope_string_alone": "Matching actuals with a different scope string.",
        },
        "decision_order": [
            "reconstruction or recorded-tolerance mismatch -> INCONCLUSIVE",
            "undefined construction -> INCONCLUSIVE",
            "primary expectation fails -> UPDATE_RULE_COLLAPSES_FAILURES",
            "a control expectation fails -> INCONCLUSIVE",
            "otherwise -> UPDATE_RULE_SEPARATES_FAILURES",
        ],
        "does_not": [
            "modify SelfModelCore",
            "modify artifacts/mrsm",
            "modify the holdout",
            "modify thresholds",
            "load Qwen",
            "start M22",
            "use M18.7 as evidence",
            "claim mechanism identity",
        ],
        "negative_result_scope": (
            "A collapse is local to this frozen P belief and these injections. "
            "It does not say a belief object cannot exist."
        ),
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


def _check_belief(belief: dict[str, Any]) -> None:
    _reject(belief)
    missing = [name for name in BELIEF_FIELDS if name not in belief]
    if missing:
        raise ValueError(f"belief missing fields: {missing}")
    extra = [name for name in belief if name not in BELIEF_FIELDS]
    if extra:
        raise LeakageError(f"belief has fields the rule does not read: {extra}")


def _check_record(record: dict[str, Any]) -> None:
    _reject(record)
    missing = [name for name in RECORD_FIELDS if name not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    extra = [name for name in record if name not in RECORD_FIELDS]
    if extra:
        raise LeakageError(f"record has fields the rule does not read: {extra}")


def _predict(belief: dict[str, Any], intervention_id: str) -> float:
    kind, name = intervention_id.split(":", 1)
    if kind == "single":
        return float(belief["effects"][name])
    if kind == "pair":
        return float(belief["joints"][name])
    raise KeyError(intervention_id)


def select_mechanism(
    effects: dict[str, float],
    joints: dict[str, float],
    min_abs: float,
    min_gap: float,
) -> dict[str, Any]:
    """Run the existing selection rule on a throwaway core."""
    core = SelfModelCore("belief-update-copy")
    core.update(
        {name: float(effects[name]) for name in HEADS},
        {name: float(joints[name]) for name in joints},
        [],
        min_abs_interaction=float(min_abs),
        min_relative_gap=float(min_gap),
    )
    mechanism = core.mechanism
    if mechanism is None:
        raise RuntimeError("selection did not return a mechanism")
    return mechanism


def _complete(records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]] | None:
    singles = {}
    joints = {}
    for record in records:
        kind, name = str(record["intervention_id"]).split(":", 1)
        value = float(record["actual_outcome"])
        if kind == "single":
            singles[name] = value
        elif kind == "pair":
            joints[name] = value
    if any(name not in singles for name in HEADS):
        return None
    expected = {
        pair_name(left, right)
        for i, left in enumerate(HEADS)
        for right in HEADS[i + 1 :]
    }
    if set(joints) != expected:
        return None
    return singles, joints


def classify(belief: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Return exactly one rule label. Injection names are not an input."""
    _check_belief(belief)
    for record in records:
        _check_record(record)
    tol = tolerance_of(float(belief["uncertainty"]))
    scope = str(belief["scope"])
    ordered = tuple(belief["mechanism_ordered"])
    min_abs = float(belief["min_abs_interaction"])
    min_gap = float(belief["min_relative_gap"])

    residuals: list[tuple[dict[str, Any], float | None]] = []
    for record in records:
        try:
            residual = float(record["actual_outcome"]) - _predict(belief, str(record["intervention_id"]))
        except KeyError:
            residuals.append((record, None))
            continue
        residuals.append((record, residual))
    if any(residual is None for _, residual in residuals):
        return ABSTAIN

    def _beyond(residual: float | None) -> bool:
        return residual is not None and abs(residual) > tol

    in_scope = [record for record, _ in residuals if str(record["scope"]) == scope]
    scope_supported = any(
        str(record["scope"]) != scope and _beyond(residual) for record, residual in residuals
    )
    table = _complete(in_scope)
    hypothesis_supported = False
    same_identity = False
    if table is not None:
        selected = select_mechanism(table[0], table[1], min_abs, min_gap)
        selected_ordered = tuple(selected["ordered"] or ())
        hypothesis_supported = (
            selected["status"] == "IDENTIFIED" and selected_ordered != ordered
        )
        same_identity = selected["status"] == "IDENTIFIED" and selected_ordered == ordered

    singles_in = {
        str(record["intervention_id"]).split(":", 1)[1]: float(record["actual_outcome"])
        for record in in_scope
        if str(record["intervention_id"]).startswith("single:")
    }
    joints_in = {
        str(record["intervention_id"]).split(":", 1)[1]: float(record["actual_outcome"])
        for record in in_scope
        if str(record["intervention_id"]).startswith("pair:")
    }
    local_interaction = False
    for name, actual in joints_in.items():
        left, right = name.split("+")
        if left not in singles_in or right not in singles_in:
            continue
        if abs(singles_in[left] - float(belief["effects"][left])) > tol:
            continue
        if abs(singles_in[right] - float(belief["effects"][right])) > tol:
            continue
        if abs(actual - float(belief["joints"][name])) > tol:
            local_interaction = True
            break
    if table is None:
        interaction_supported = local_interaction
    elif hypothesis_supported or not same_identity:
        interaction_supported = False
    else:
        interaction_supported = local_interaction

    supported = []
    if scope_supported:
        supported.append(REVISE_SCOPE)
    if hypothesis_supported:
        supported.append(REVISE_HYPOTHESIS)
    if interaction_supported:
        supported.append(ADD_INTERACTION)
    if len(supported) == 1:
        return supported[0]
    if len(supported) > 1:
        return ABSTAIN
    if any(_beyond(residual) for _, residual in residuals):
        return ABSTAIN
    return HOLD


def _record(intervention_id: str, actual: float, scope: str) -> dict[str, Any]:
    return {
        "intervention_id": intervention_id,
        "actual_outcome": float(actual),
        "scope": scope,
    }


def table_records(
    effects: dict[str, float],
    joints: dict[str, float],
    scope: str,
) -> list[dict[str, Any]]:
    records = [_record(f"single:{name}", effects[name], scope) for name in HEADS]
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            name = pair_name(left, right)
            records.append(_record(f"pair:{name}", joints[name], scope))
    return records


def _copy_joints(belief: dict[str, Any]) -> dict[str, float]:
    return {name: float(value) for name, value in belief["joints"].items()}


def _winner(belief: dict[str, Any]) -> str:
    ordered = belief["mechanism_ordered"]
    return pair_name(str(ordered[0]), str(ordered[1]))


def build_primary_cases(
    belief: dict[str, Any],
) -> dict[str, list[dict[str, Any]] | None]:
    """Build the four injections. None means the interaction case is undefined."""
    _check_belief(belief)
    scope = str(belief["scope"])
    tol = tolerance_of(float(belief["uncertainty"]))
    step = tol + CONSTRUCTION_CLEARANCE
    effects = {name: float(belief["effects"][name]) for name in HEADS}
    joints = _copy_joints(belief)
    winner = _winner(belief)
    winner_interaction = float(joints[winner]) - float(effects[winner.split("+")[0]]) - float(
        effects[winner.split("+")[1]]
    )

    hypothesis_joints = _copy_joints(belief)
    others = sorted(name for name in hypothesis_joints if name != winner)
    moved = others[0]
    left, right = winner.split("+")
    hypothesis_joints[winner] = float(effects[left]) + float(effects[right])
    moved_left, moved_right = moved.split("+")
    hypothesis_joints[moved] = (
        float(effects[moved_left]) + float(effects[moved_right]) + winner_interaction
    )
    hypothesis = table_records(effects, hypothesis_joints, scope)

    outside = scope + OUTSIDE_SCOPE_SUFFIX
    scope_case = [
        _record(row["intervention_id"], float(row["actual_outcome"]) + step, outside)
        for row in table_records(effects, joints, scope)
    ]

    ceiling = abs(winner_interaction) * (1.0 - float(belief["min_relative_gap"])) - 1e-6
    interaction_case = None
    for name in sorted(joints):
        if name == winner:
            continue
        old = float(joints[name])
        pair_left, pair_right = name.split("+")
        base = float(effects[pair_left]) + float(effects[pair_right])
        old_interaction = old - base
        for sign in (-1.0, 1.0):
            delta = sign * step
            new_interaction = old_interaction + delta
            if abs(new_interaction) > ceiling or abs(delta) <= tol:
                continue
            trial = _copy_joints(belief)
            trial[name] = old + delta
            selected = select_mechanism(
                effects,
                trial,
                float(belief["min_abs_interaction"]),
                float(belief["min_relative_gap"]),
            )
            if selected["status"] != "IDENTIFIED":
                continue
            if tuple(selected["ordered"]) != tuple(belief["mechanism_ordered"]):
                continue
            interaction_case = table_records(effects, trial, scope)
            break
        if interaction_case is not None:
            break

    none_case = [_record(f"single:{HEADS[0]}", float(effects[HEADS[0]]) + step, scope)]
    return {
        "hypothesis": hypothesis,
        "scope": scope_case,
        "interaction": interaction_case,
        "none": none_case,
    }


def build_controls(
    belief: dict[str, Any],
    holdout_records: list[dict[str, Any]],
    hypothesis_records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    scope = str(belief["scope"])
    tol = tolerance_of(float(belief["uncertainty"]))
    step = tol + CONSTRUCTION_CLEARANCE
    effects = {name: float(belief["effects"][name]) for name in HEADS}
    matched = table_records(effects, _copy_joints(belief), scope + OUTSIDE_SCOPE_SUFFIX)
    foreign = _record(f"single:{HEADS[0]}", float(effects[HEADS[0]]) + step, scope + OUTSIDE_SCOPE_SUFFIX)
    return {
        "holdout": [
            _record(str(row["intervention_id"]), float(row["actual_outcome"]), str(row["scope"]))
            for row in holdout_records
        ],
        "double": list(hypothesis_records) + [foreign],
        "scope_string_alone": matched,
    }


def decide(
    primary: dict[str, str | None],
    controls: dict[str, str],
    *,
    reconstruction_ok: bool,
    tolerance_ok: bool,
    construction_defined: bool,
) -> dict[str, Any]:
    """Apply the preregistered decision order. Do not retune it after scoring."""
    if not reconstruction_ok or not tolerance_ok or not construction_defined:
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "PRECONDITION",
            "reason": "A reconstruction check, the recorded tolerance, or case construction failed.",
        }
    if any(primary.get(name) is None for name in PRIMARY_EXPECTATION):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "MISSING_PRIMARY",
            "reason": "A primary case has no rule output.",
        }
    if any(primary[name] != expected for name, expected in PRIMARY_EXPECTATION.items()):
        return {
            "decision": COLLAPSES,
            "reason_code": "PRIMARY_MISMATCH",
            "reason": "The four injected contradictions did not receive four different updates.",
        }
    if any(controls.get(name) != expected for name, expected in CONTROL_EXPECTATION.items()):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "CONTROL_MISMATCH",
            "reason": "The primary cases separated, and a preregistered control did not match.",
        }
    return {
        "decision": SEPARATES,
        "reason_code": "SEPARATED",
        "reason": "The four injected contradictions received four different updates, and the controls matched.",
    }
