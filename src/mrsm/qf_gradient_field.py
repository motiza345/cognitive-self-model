"""Preregistered gradient-field diagnostic. Not M3 and not an MRSM gate.

Δ is the margin drop used by the regime measurements:

    Δ(α) = m(h) - m(h + α v)

For any α, the expansion is

    Δ(α) ≈ -α vᵀ g(x) - (α² / 2) vᵀ H_m(x) v

with g(x) = ∇_h m(x) at the residual site. For α > 0 the first-order term
has the opposite sign of vᵀ g(x). This module fixes that sign. It does not
treat a regime-effect sign matrix as a gradient, and it does not treat an
α = 1 measurement as the linear term.

The rules below are the preregistration. They are not adjusted after the
model measurements. A negative result means the declared representation was
not predictable for these models, prompts, and this map. It does not say the
structure cannot exist.
"""

from __future__ import annotations

import itertools
import math
import statistics
from typing import Any, Callable

import numpy as np
import torch

from src.mrsm import HEADS
from src.mrsm.qf_m2 import cosine
from src.mrsm.qwen_bridge import SIGN_FLOOR

SMALL_ALPHAS = (-0.25, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.25)
REFERENCE_ALPHAS = (-1.0, 1.0)
SMALLEST_ABS_ALPHA = 0.01
# 1e-4 is below float32 resolution of the logit-margin difference.
# 1e-2 is the instrument step. The scientific alpha grid is unchanged.
GRADIENT_CHECK_EPSILON = 1e-2
GRADIENT_CHECK_RELATIVE = 1e-2
GRADIENT_CHECK_ABSOLUTE = 1e-4
PROMPTS_PER_SPLIT = 6
PERMUTATION_COUNT = 720
SPLITS = ("discovery", "validation", "replication")
HELDOUT_SPLITS = ("validation", "replication")
PRIMARY_MAP = "functional_slot_alignment"
FIELD_REPRESENTATION = "phi_s = v_s^T g_layer(s)"
QWEN_FUNCTIONAL_LAYERS = (0, 8, 15, 23)
GPT2_FUNCTIONAL_LAYERS = (0, 4, 7, 11)
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
QWEN_POSITIVE_ID = 9834
QWEN_NEGATIVE_ID = 902
GPT2_POSITIVE_ID = 3763
GPT2_NEGATIVE_ID = 645

LINEAR_MATCH = "LINEAR_SIGN_MATCHES_GRADIENT"
LINEAR_SHRINKS = "SIGN_AGREEMENT_SHRINKS_AS_ALPHA_GROWS"
LINEAR_MISS = "LINEAR_SIGN_DOES_NOT_MATCH_AT_SMALLEST_ALPHA"
LINEAR_UNDEFINED = "LINEAR_SIGN_UNDEFINED"
LINEAR_DIFFERS = "HELD_OUT_DIFFERS_FROM_DISCOVERY"
INCONCLUSIVE = "INCONCLUSIVE"
FIELD_SUPPORTED = "SUPPORTED"
FIELD_PARTIAL = "PARTIAL"
FIELD_TIED = "TIED"
FIELD_NOT = "NOT_SUPPORTED"
FIELD_NO_POWER = "INCONCLUSIVE_NULL_HAS_NO_POWER"
FIELD_PREDICTABLE = "FIELD_LOW_DIMENSIONAL_PREDICTABLE_UNDER_DECLARED_MAP"
FIELD_NOT_PREDICTABLE = "FIELD_LOW_DIMENSIONAL_NOT_PREDICTABLE_UNDER_DECLARED_MAP"
FIELD_PARTIAL_OVERALL = "PARTIAL_FIELD_PREDICTABILITY"

RELATED_NOT_THIS_TEST = (
    {
        "name": "Curse of Multiple Mediators",
        "arxiv": "2606.27510",
        "relation": (
            "Discusses state-dependent interactions in activation-patching estimates. "
            "Related context, not this additive residual protocol."
        ),
    },
    {
        "name": "Distributed Alignment Search",
        "arxiv": "2303.02536",
        "relation": (
            "Learns an alignment to a prespecified high-level variable. "
            "It does not by itself remove context dependence, and it is not the map used here."
        ),
    },
    {
        "name": "Boundless DAS",
        "arxiv": "2305.08809",
        "relation": (
            "Extends DAS on Alpaca. Included as related work only. "
            "This diagnostic does not learn a rotation between models."
        ),
    },
)


def alpha_key(alpha: float) -> str:
    return format(float(alpha), ".2f")


def small_alpha_keys() -> tuple[str, ...]:
    keys = tuple(alpha_key(alpha) for alpha in SMALL_ALPHAS)
    if len(keys) != len(set(keys)):
        raise RuntimeError("Small-alpha keys are not unique.")
    return keys


def reference_alpha_keys() -> tuple[str, ...]:
    return tuple(alpha_key(alpha) for alpha in REFERENCE_ALPHAS)


def smallest_alpha_keys() -> tuple[str, ...]:
    keys = tuple(key for key in small_alpha_keys() if abs(float(key)) == SMALLEST_ABS_ALPHA)
    if keys != ("-0.01", "0.01"):
        raise RuntimeError("Smallest alpha keys drifted.")
    return keys


def last_position_margin_gradient(
    run_with_hooks: Callable[..., torch.Tensor],
    tokens: torch.Tensor | None,
    layer: int,
    positive_id: int,
    negative_id: int,
) -> np.ndarray:
    """Direct ∇_h m at blocks.{layer}.hook_resid_post, last position.

    A zero perturbation with its own graph is added at the hook. The returned
    vector is ∂m/∂h at the unperturbed residual. This is not a finite difference.
    """
    saved: dict[str, torch.Tensor] = {}
    hook_name = f"blocks.{int(layer)}.hook_resid_post"

    def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
        if residual.ndim != 3:
            raise ValueError("Residual must have shape [batch, position, d_model].")
        delta = torch.zeros_like(residual)
        delta.requires_grad_(True)
        saved["delta"] = delta
        return residual + delta

    with torch.enable_grad():
        logits = run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        if logits.ndim != 3 or int(logits.shape[0]) != 1:
            raise ValueError("Gradient measurement scores one prompt.")
        margin = logits[0, -1, int(positive_id)] - logits[0, -1, int(negative_id)]
        if "delta" not in saved:
            raise RuntimeError(f"{hook_name} did not run.")
        margin.backward()
        grad = saved["delta"].grad
        if grad is None:
            raise RuntimeError("Direct residual gradient is missing.")
        vector = grad[0, -1].detach().to(dtype=torch.float64).cpu().numpy().copy()
    saved.clear()
    return vector


def linear_prediction(alpha: float, projection: float) -> float:
    """First-order term of Δ = m(h) - m(h + α v). Opposite the projection when α > 0."""
    return -float(alpha) * float(projection)


def pole(value: float) -> str:
    if not math.isfinite(float(value)) or abs(float(value)) <= SIGN_FLOOR:
        return "NEAR_ZERO"
    return "POSITIVE" if float(value) > 0.0 else "NEGATIVE"


def compare_prediction(alpha: float, projection: float, delta: float) -> dict[str, Any]:
    """Compare the preregistered linear term with a measured margin drop."""
    linear = linear_prediction(alpha, projection)
    residual = float(delta) - linear
    ratio = None if abs(linear) <= SIGN_FLOOR else float(delta) / linear
    linear_sign = pole(linear)
    delta_sign = pole(delta)
    if linear_sign == "NEAR_ZERO" or delta_sign == "NEAR_ZERO":
        relation = "NEAR_ZERO"
    elif linear_sign == delta_sign:
        relation = "AGREE"
    else:
        relation = "DISAGREE"
    return {
        "alpha": float(alpha),
        "alpha_key": alpha_key(alpha),
        "projection": float(projection),
        "linear": linear,
        "delta": float(delta),
        "residual": residual,
        "ratio": ratio,
        "projection_sign": pole(projection),
        "linear_sign": linear_sign,
        "delta_sign": delta_sign,
        "relation": relation,
        "in_small_grid": alpha_key(alpha) in small_alpha_keys(),
        "historical_scale": alpha_key(alpha) in reference_alpha_keys(),
    }


def gradient_check_passes(projection: float, central_difference: float) -> bool:
    """Instrument check. Passes when the absolute or the relative discrepancy is small.

    The absolute floor is the existing 1e-4 effect floor. This is not a
    scientific threshold on mechanisms.
    """
    if not math.isfinite(float(projection)) or not math.isfinite(float(central_difference)):
        return False
    error = abs(float(projection) - float(central_difference))
    scale = max(abs(float(projection)), abs(float(central_difference)), SIGN_FLOOR)
    return error <= GRADIENT_CHECK_ABSOLUTE or (error / scale) <= GRADIENT_CHECK_RELATIVE


def alpha0_passes(delta: float) -> bool:
    """α = 0 must leave the margin drop at numerical zero. Not a scientific alpha."""
    return math.isfinite(float(delta)) and abs(float(delta)) <= GRADIENT_CHECK_ABSOLUTE


def _rows_for(cells: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [cell for cell in cells if cell.get("alpha_key") == key]


def _agreement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    compared = [row for row in rows if row.get("relation") != "NEAR_ZERO"]
    return {
        "n": len(rows),
        "compared": len(compared),
        "agree": sum(1 for row in compared if row.get("relation") == "AGREE"),
        "disagree": sum(1 for row in compared if row.get("relation") == "DISAGREE"),
        "near_zero": sum(1 for row in rows if row.get("relation") == "NEAR_ZERO"),
    }


def classify_linear(cells: list[dict[str, Any]], *, gradient_ok: bool, alpha0_ok: bool) -> str:
    """Sign agreement of Δ and -α vᵀg. Magnitudes are not an input."""
    if not gradient_ok or not alpha0_ok:
        return INCONCLUSIVE
    required = small_alpha_keys()
    present = {cell.get("alpha_key") for cell in cells}
    if not set(required).issubset(present):
        return INCONCLUSIVE
    counts = {key: _agreement_counts(_rows_for(cells, key)) for key in required}
    if any(item["n"] == 0 for item in counts.values()):
        return INCONCLUSIVE
    smallest = smallest_alpha_keys()
    larger = tuple(key for key in required if key not in smallest)

    def full_match(key: str) -> bool:
        item = counts[key]
        return item["compared"] > 0 and item["disagree"] == 0

    def has_disagreement(key: str) -> bool:
        return counts[key]["disagree"] > 0

    if any(has_disagreement(key) for key in smallest):
        return LINEAR_MISS
    if any(counts[key]["compared"] == 0 for key in smallest):
        return LINEAR_UNDEFINED
    if any(has_disagreement(key) for key in larger):
        return LINEAR_SHRINKS
    if any(counts[key]["compared"] == 0 for key in larger):
        return LINEAR_UNDEFINED
    if all(full_match(key) for key in required):
        return LINEAR_MATCH
    return INCONCLUSIVE


def classify_model_linear(split_labels: dict[str, str]) -> str:
    if set(split_labels) != set(SPLITS):
        return INCONCLUSIVE
    labels = [split_labels[split] for split in SPLITS]
    if any(label == INCONCLUSIVE for label in labels):
        return INCONCLUSIVE
    if len(set(labels)) == 1:
        return labels[0]
    return LINEAR_DIFFERS


def magnitude_summary(cells: list[dict[str, Any]], alpha: str) -> dict[str, Any]:
    """Descriptive only. Not an input to classify_linear."""
    ratios = []
    residuals = []
    for cell in _rows_for(cells, alpha):
        linear = cell.get("linear")
        delta = cell.get("delta")
        if linear is None or delta is None or abs(float(linear)) <= SIGN_FLOOR:
            continue
        ratios.append(float(delta) / float(linear))
        residuals.append(abs(float(delta) - float(linear)))
    return {
        "alpha_key": alpha,
        "n_ratios": len(ratios),
        "median_delta_over_linear": statistics.median(ratios) if ratios else None,
        "median_abs_residual": statistics.median(residuals) if residuals else None,
        "decision_input": False,
    }


def field_null(qwen: list[list[float]], gpt2: list[list[float]]) -> dict[str, Any]:
    """Exhaustive prompt permutation within one split. 6! = 720, not a tuned draw count."""
    if len(qwen) != PROMPTS_PER_SPLIT or len(gpt2) != PROMPTS_PER_SPLIT:
        return {
            "status": INCONCLUSIVE,
            "reason": "a split does not contain the preregistered six prompts",
            "n_permutations": 0,
        }
    permutations = list(itertools.permutations(range(PROMPTS_PER_SPLIT)))
    if len(permutations) != PERMUTATION_COUNT:
        raise RuntimeError("Prompt permutation count drifted.")
    identity_perm = tuple(range(PROMPTS_PER_SPLIT))
    scores: list[float] = []
    identity_cosines: list[float] | None = None
    identity_mean: float | None = None
    for perm in permutations:
        cosines = [cosine(qwen[index], gpt2[perm[index]]) for index in range(PROMPTS_PER_SPLIT)]
        if any(value is None or not math.isfinite(float(value)) for value in cosines):
            return {
                "status": INCONCLUSIVE,
                "reason": "a field cosine is undefined",
                "n_permutations": PERMUTATION_COUNT,
            }
        typed = [float(value) for value in cosines if value is not None]
        score = sum(typed) / PROMPTS_PER_SPLIT
        scores.append(score)
        if perm == identity_perm:
            identity_cosines = typed
            identity_mean = score
    if identity_mean is None or identity_cosines is None:
        return {"status": INCONCLUSIVE, "reason": "identity permutation missing", "n_permutations": PERMUTATION_COUNT}
    n_strictly_better = sum(1 for score in scores if score > identity_mean)
    n_equal = sum(1 for score in scores if score == identity_mean)
    others = [score for score, perm in zip(scores, permutations) if perm != identity_perm]
    best_other = max(others) if others else None
    all_positive = all(value > 0.0 for value in identity_cosines)
    if all(score == identity_mean for score in scores):
        status = FIELD_NO_POWER
    elif n_strictly_better == 0 and all(identity_mean > score for score in others) and all_positive:
        status = FIELD_SUPPORTED
    elif n_strictly_better == 0 and all(identity_mean > score for score in others):
        status = FIELD_PARTIAL
    elif n_strictly_better == 0 and n_equal > 1 and all_positive:
        status = FIELD_TIED
    else:
        status = FIELD_NOT
    return {
        "status": status,
        "n_permutations": PERMUTATION_COUNT,
        "identity_mean_cosine": identity_mean,
        "identity_cosines": identity_cosines,
        "n_strictly_better": n_strictly_better,
        "n_equal_to_identity": n_equal,
        "best_other_mean_cosine": best_other,
        "null": "exhaustive permutation of the six GPT-2 prompt rows within the split",
        "within_regime_null": "not used; two prompts per regime cannot support this claim",
    }


def classify_field_splits(split_results: dict[str, dict[str, Any]]) -> str:
    if set(split_results) != set(SPLITS):
        return INCONCLUSIVE
    labels = [split_results[split]["status"] for split in SPLITS]
    if any(str(label).startswith("INCONCLUSIVE") for label in labels):
        return INCONCLUSIVE
    if all(label == FIELD_SUPPORTED for label in labels):
        return FIELD_PREDICTABLE
    if all(label == FIELD_NOT for label in labels):
        return FIELD_NOT_PREDICTABLE
    return FIELD_PARTIAL_OVERALL


def preregistration_document() -> dict[str, Any]:
    return {
        "experiment": "QF-GRADIENT-FIELD",
        "m3": False,
        "mrsm_rerun": False,
        "self_model_modified": False,
        "delta_definition": "m(h) - m(h + alpha * v)",
        "expansion": "delta ≈ -alpha * v^T g(x) - (alpha^2 / 2) * v^T H_m(x) v",
        "linear_term": "-alpha * v^T g(x)",
        "sign_correction": (
            "For alpha > 0 the first-order term in delta has the opposite sign of v^T g(x)."
        ),
        "small_alphas": [alpha_key(alpha) for alpha in SMALL_ALPHAS],
        "reference_alphas_not_in_linear_test": [alpha_key(alpha) for alpha in REFERENCE_ALPHAS],
        "smallest_abs_alpha": SMALLEST_ABS_ALPHA,
        "gradient_check_epsilon": GRADIENT_CHECK_EPSILON,
        "gradient_check_epsilon_reason": (
            "1e-4 is below float32 resolution of the logit-margin difference. "
            "1e-2 is the instrument step. The scientific alpha grid is unchanged."
        ),
        "gradient_check_absolute": GRADIENT_CHECK_ABSOLUTE,
        "gradient_check_relative": GRADIENT_CHECK_RELATIVE,
        "sign_floor": SIGN_FLOOR,
        "representation": FIELD_REPRESENTATION,
        "map": {
            "name": PRIMARY_MAP,
            "qwen_layers": list(QWEN_FUNCTIONAL_LAYERS),
            "gpt2_layers": list(GPT2_FUNCTIONAL_LAYERS),
            "slot_order": list(HEADS),
            "direction_seeds": {"primary": 22101, "orthogonal": 22103},
            "directions_resampled_at_each_d_model": True,
            "learned_rotation": False,
            "layers_15_and_23_on_gpt2": "NOT_COMPARABLE",
            "imputation": False,
        },
        "splits": {
            "discovery": "same frozen discovery prompts as the regime matrix; primary linear reading",
            "validation": "held-out repeat",
            "replication": "held-out repeat",
            "pooling": False,
        },
        "field_null": "all 720 permutations of GPT-2 prompts inside one split",
        "assumes_context_independent_mechanism": False,
        "assumes_model_independent_mechanism": False,
        "negative_result_scope": (
            "A negative or partial result says this structure was not found for these models, "
            "these prompts, and this declared map. It does not say the structure cannot exist."
        ),
        "related_not_this_test": list(RELATED_NOT_THIS_TEST),
    }


def _linear_block(cells: list[dict[str, Any]], *, gradient_ok: bool, alpha0_ok: bool) -> dict[str, Any]:
    label = classify_linear(cells, gradient_ok=gradient_ok, alpha0_ok=alpha0_ok)
    by_alpha = {}
    for key in small_alpha_keys() + reference_alpha_keys():
        counts = _agreement_counts(_rows_for(cells, key))
        counts.update(magnitude_summary(cells, key))
        counts["in_linear_test"] = key in small_alpha_keys()
        by_alpha[key] = counts
    return {"label": label, "by_alpha": by_alpha, "gradient_ok": gradient_ok, "alpha0_ok": alpha0_ok}


def assemble_decision(measurements: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Classify recorded cells. Does not measure a model and does not rank models."""
    if set(measurements) != {"qwen", "gpt2"}:
        raise ValueError("The diagnostic compares the preregistered Qwen and GPT-2 measurements.")
    linear: dict[str, Any] = {}
    field_splits: dict[str, Any] = {}
    for model in ("qwen", "gpt2"):
        split_labels = {}
        blocks = {}
        for split in SPLITS:
            payload = measurements[model][split]
            block = _linear_block(
                list(payload["cells"]),
                gradient_ok=bool(payload["gradient_ok"]),
                alpha0_ok=bool(payload["alpha0_ok"]),
            )
            blocks[split] = block
            split_labels[split] = block["label"]
        linear[model] = {
            "splits": blocks,
            "label": classify_model_linear(split_labels),
        }
    for split in SPLITS:
        qwen_split = measurements["qwen"][split]
        gpt_split = measurements["gpt2"][split]
        if not qwen_split["gradient_ok"] or not gpt_split["gradient_ok"]:
            field_splits[split] = {
                "status": INCONCLUSIVE,
                "reason": "a gradient check failed, so the field was not compared",
            }
            continue
        qwen_rows = [list(row["vector"]) for row in qwen_split["field"]]
        gpt_rows = [list(row["vector"]) for row in gpt_split["field"]]
        field_splits[split] = field_null(qwen_rows, gpt_rows)
    field_label = classify_field_splits(field_splits)
    interpretation = (
        f"Qwen linear classification is {linear['qwen']['label']}. "
        f"GPT-2 linear classification is {linear['gpt2']['label']}. "
        "The models are reported separately and are not ranked. "
        f"The eight-dimensional field of v^T g under {PRIMARY_MAP} is {field_label}. "
        "This experiment does not assume a context-independent or model-independent mechanism. "
        "A negative or partial result says this structure was not found for these models, "
        "these prompts, and this declared map. It does not say the structure cannot exist. "
        "M3 was not run. No mechanism was identified. "
        "The regime-effect sign matrix was not used as a gradient, and alpha = 1 was not part of the linear test."
    )
    return {
        "experiment": "QF-GRADIENT-FIELD",
        "m3_executed": False,
        "mechanism_identified": False,
        "models_ranked": False,
        "assumes_context_independent_mechanism": False,
        "assumes_model_independent_mechanism": False,
        "linear": linear,
        "field_splits": field_splits,
        "field_decision": field_label,
        "primary_interpretation": interpretation,
        "map": PRIMARY_MAP,
        "representation": FIELD_REPRESENTATION,
    }
