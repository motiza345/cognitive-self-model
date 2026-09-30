"""World-change check for the frozen belief-update rule.

The worlds are copies or evaluation contexts of arm P. The records they emit
are scored by belief_update.classify. This module does not change that rule,
the frozen checkpoint, the holdout, or the P result.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from src.mrsm import HEADS
from src.mrsm.belief_update import (
    ABSTAIN,
    ADD_INTERACTION,
    HOLD,
    INCONCLUSIVE,
    REVISE_HYPOTHESIS,
    REVISE_SCOPE,
    classify,
)
from src.mrsm.self_model import pair_name

WORLD_SEPARATES = "WORLD_CHANGE_SEPARATES"
WORLD_DOES_NOT_SEPARATE = "WORLD_CHANGE_DOES_NOT_SEPARATE"
DECISIONS = (WORLD_SEPARATES, WORLD_DOES_NOT_SEPARATE, INCONCLUSIVE)

HYPOTHESIS_HEADS = ("L0H2", "L1H2")
NONE_HEAD = "L0H3"
SCOPE_NAME = "P:shuffled-target:seed=21011"
SCOPE_PERMUTATION_SEED = 21011
DISCOVERY_SEED = 1929
DISCOVERY_N = 512
TRAIN_SEED = 1829
TRAIN_N = 8192
MODEL_SEED = 1729

PRIMARY_EXPECTATION = {
    "hypothesis": REVISE_HYPOTHESIS,
    "scope": REVISE_SCOPE,
    "interaction": ADD_INTERACTION,
    "none": ABSTAIN,
}
CONTROL_EXPECTATION = {"holdout": HOLD}

INSTRUMENT_ATOL = 1.0e-4


def preregistration_document(declaration: dict[str, Any] | None = None) -> dict[str, Any]:
    document = {
        "diagnostic": "belief_world_change",
        "status": "diagnostic",
        "arm": "P",
        "question": (
            "When evidence is measured from a changed planted world, does the "
            "unchanged belief-update rule map each change to its own update?"
        ),
        "rule": "src.mrsm.belief_update.classify",
        "rule_modified": False,
        "decisions": list(DECISIONS),
        "primary_expectation": dict(PRIMARY_EXPECTATION),
        "control_expectation": dict(CONTROL_EXPECTATION),
        "worlds": {
            "hypothesis": (
                "Train a new copy from the same model seed with the construction "
                "mask on L0H2 and L1H2. Measure discovery ablations. Do not finetune "
                "the frozen checkpoint."
            ),
            "scope": (
                "Keep the frozen weights. Permute discovery targets with seed 21011 "
                "and label the records with a scope string other than the belief scope."
            ),
            "interaction": (
                "Keep the frozen weights. On the lexicographic first non-winning pair, "
                "subtract the frozen winner's absolute interaction from the target logit "
                "only when both heads of that pair are ablated."
            ),
            "none": (
                "Keep the frozen weights. Subtract that same magnitude from the target "
                "logit whenever L0H3 is ablated, and publish the full measured table."
            ),
        },
        "magnitude_source": "absolute interaction of the frozen belief's identified pair",
        "magnitude_is_searched": False,
        "ceiling_search": False,
        "holdout_remeasured": False,
        "ground_truth_file_loaded": False,
        "decision_order": [
            "reconstruction, tolerance, instrument, or non-finite measurement -> INCONCLUSIVE",
            "primary expectation fails -> WORLD_CHANGE_DOES_NOT_SEPARATE",
            "holdout is not HOLD -> INCONCLUSIVE",
            "otherwise -> WORLD_CHANGE_SEPARATES",
        ],
        "does_not": [
            "modify belief_update.classify",
            "modify SelfModelCore",
            "modify artifacts/mrsm",
            "modify the holdout",
            "modify thresholds",
            "load Qwen",
            "start M22",
            "finetune the frozen checkpoint",
            "choose a new magnitude after scoring",
        ],
    }
    if declaration is not None:
        document["declaration"] = declaration
    return document


def declare_from_belief(belief: dict[str, Any]) -> dict[str, Any]:
    """Name the interaction pair and magnitude from the frozen belief only."""
    ordered = [str(name) for name in belief["mechanism_ordered"]]
    winner = pair_name(ordered[0], ordered[1])
    left, right = winner.split("+")
    interaction = (
        float(belief["joints"][winner]) - float(belief["effects"][left]) - float(belief["effects"][right])
    )
    others = sorted(name for name in belief["joints"] if name != winner)
    return {
        "hypothesis_heads": list(HYPOTHESIS_HEADS),
        "interaction_pair": others[0],
        "magnitude": abs(interaction),
        "none_head": NONE_HEAD,
        "scope_name": SCOPE_NAME,
        "scope_permutation_seed": SCOPE_PERMUTATION_SEED,
        "winner_pair": winner,
    }


def target_penalty(logits: torch.Tensor, targets: torch.Tensor, amount: float) -> torch.Tensor:
    if amount == 0.0:
        return logits
    shifted = logits.clone()
    shifted[torch.arange(shifted.shape[0]), targets] = shifted[torch.arange(shifted.shape[0]), targets] - float(amount)
    return shifted


def penalty_amount(kind: str, ablated: frozenset[str], declaration: dict[str, Any]) -> float:
    magnitude = float(declaration["magnitude"])
    if kind == "interaction":
        left, right = str(declaration["interaction_pair"]).split("+")
        if left in ablated and right in ablated:
            return magnitude
        return 0.0
    if kind == "none":
        if str(declaration["none_head"]) in ablated:
            return magnitude
        return 0.0
    return 0.0


def _margins(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    gathered = logits.gather(1, targets[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, targets[:, None], -torch.inf)
    return gathered - other.max(dim=-1).values


def measure_effects(model, ds, penalty: Callable[[frozenset[str]], float]) -> tuple[dict[str, float], dict[str, float]]:
    """Ablation drops. penalty is part of the world, applied inside the forward result."""
    import mrsm_p_v1
    from src.mrsm.interventions import _hooks, parse_head

    def drop(heads: list[str]) -> float:
        amount_intact = float(penalty(frozenset()))
        amount_treated = float(penalty(frozenset(heads)))
        intact_logits = mrsm_p_v1.logits_at(model, ds, None)
        treated_hooks = _hooks([parse_head(name) for name in heads]) if heads else None
        treated_logits = mrsm_p_v1.logits_at(model, ds, treated_hooks)
        intact = _margins(target_penalty(intact_logits, ds.targets, amount_intact), ds.targets)
        treated = _margins(target_penalty(treated_logits, ds.targets, amount_treated), ds.targets)
        return float((intact - treated).mean())

    singles = {name: drop([name]) for name in HEADS}
    joints = {}
    for index, left in enumerate(HEADS):
        for right in HEADS[index + 1 :]:
            joints[pair_name(left, right)] = drop([left, right])
    return singles, joints


def records_from_effects(singles: dict[str, float], joints: dict[str, float], scope: str) -> list[dict[str, Any]]:
    rows = [
        {"intervention_id": f"single:{name}", "actual_outcome": float(singles[name]), "scope": scope}
        for name in HEADS
    ]
    for index, left in enumerate(HEADS):
        for right in HEADS[index + 1 :]:
            name = pair_name(left, right)
            rows.append(
                {"intervention_id": f"pair:{name}", "actual_outcome": float(joints[name]), "scope": scope}
            )
    return rows


def finite_effects(singles: dict[str, float], joints: dict[str, float]) -> bool:
    values = list(singles.values()) + list(joints.values())
    return all(value == value and abs(value) != float("inf") for value in values)


def decide_world(
    primary: dict[str, str | None],
    controls: dict[str, str],
    *,
    reconstruction_ok: bool,
    tolerance_ok: bool,
    instrument_ok: bool,
    finite_ok: bool,
) -> dict[str, Any]:
    if not reconstruction_ok or not tolerance_ok or not instrument_ok or not finite_ok:
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "PRECONDITION",
            "reason": "A reconstruction, tolerance, instrument, or finiteness check failed.",
        }
    if any(primary.get(name) != expected for name, expected in PRIMARY_EXPECTATION.items()):
        return {
            "decision": WORLD_DOES_NOT_SEPARATE,
            "reason_code": "PRIMARY_MISMATCH",
            "reason": "The measured worlds did not receive the four different updates.",
        }
    if any(controls.get(name) != expected for name, expected in CONTROL_EXPECTATION.items()):
        return {
            "decision": INCONCLUSIVE,
            "reason_code": "CONTROL_MISMATCH",
            "reason": "The worlds separated, and the frozen holdout did not stay HOLD.",
        }
    return {
        "decision": WORLD_SEPARATES,
        "reason_code": "SEPARATED",
        "reason": "Each measured world received its own update, and the frozen holdout stayed HOLD.",
    }


def score_records(belief: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Call the unchanged rule. The world name is not an argument."""
    return classify(belief, records)
