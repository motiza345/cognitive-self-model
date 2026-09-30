"""Cross-model diagnostic rules. Not an MRSM gate and not a mechanism claim."""

from __future__ import annotations

from typing import Any

from cognitive_self_model.m22_1.direction import (
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)
from cognitive_self_model.m22_1.protocol import candidate_layers

from src.mrsm import HEADS
from src.mrsm.qf_regime import FROZEN_REGIMES

QWEN_N_LAYERS = 24
N_CANDIDATE_LAYERS = 4
PRIMARY_SEED = 22101
ORTHOGONAL_SEED = 22103
ALPHA = 1.0
QWEN_COORDINATE_LAYERS = (0, 8, 15, 23)

FROZEN_QWEN_MATRIX = {
    "L0H0": {"completion": "+", "instruction": "-", "syntax": "-"},
    "L0H1": {"completion": "-", "instruction": "-", "syntax": "-"},
    "L0H2": {"completion": "+", "instruction": "-", "syntax": "+"},
    "L0H3": {"completion": "+", "instruction": "+", "syntax": "-"},
    "L1H0": {"completion": "+", "instruction": "-", "syntax": "-"},
    "L1H1": {"completion": "-", "instruction": "-", "syntax": "-"},
    "L1H2": {"completion": "-", "instruction": "-", "syntax": "-"},
    "L1H3": {"completion": "-", "instruction": "-", "syntax": "-"},
}


def qwen_coordinate_layers() -> list[int]:
    layers = candidate_layers(QWEN_N_LAYERS, N_CANDIDATE_LAYERS)
    if layers != list(QWEN_COORDINATE_LAYERS):
        raise RuntimeError("Qwen candidate-layer grid does not match the frozen catalog.")
    return layers


def functional_layers(n_layers: int) -> list[int] | None:
    """Same candidate-layer rule used to build the Qwen slots. Not fit to effects."""
    layers = candidate_layers(int(n_layers), N_CANDIDATE_LAYERS)
    if len(layers) != N_CANDIDATE_LAYERS:
        return None
    return layers


def _directions(d_model: int):
    primary = primary_direction(int(d_model), PRIMARY_SEED)
    orthogonal = orthogonal_direction(int(d_model), ORTHOGONAL_SEED, primary)
    return {"primary": primary, "orthogonal": orthogonal}


def slot_catalog(layers: list[int], d_model: int, *, n_layers: int) -> list[dict[str, Any]]:
    """Eight residual slots. Names are labels, not attention-head indices."""
    if len(layers) != len(HEADS) // 2:
        raise ValueError("Slot catalog expects four layers.")
    directions = _directions(d_model)
    rows = []
    index = 0
    for layer in layers:
        for direction_name in ("primary", "orthogonal"):
            vector = directions[direction_name]
            comparable = 0 <= int(layer) < int(n_layers)
            rows.append({
                "slot": HEADS[index],
                "layer": int(layer),
                "direction_name": direction_name,
                "direction_seed": PRIMARY_SEED if direction_name == "primary" else ORTHOGONAL_SEED,
                "alpha": ALPHA,
                "vector": vector,
                "vector_sha256": vector_sha256(vector),
                "comparable": comparable,
                "status": "SCHEDULED" if comparable else "NOT_COMPARABLE",
                "reason": None if comparable else "layer index is outside this model",
            })
            index += 1
    return rows


def public_slots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hidden = {"vector"}
    return [{key: value for key, value in row.items() if key not in hidden} for row in rows]


def decide_m1(
    *,
    comparable: bool,
    cells: dict[str, dict[str, dict[str, Any]]] | None,
    qwen_matrix: dict[str, dict[str, str]] | None = None,
    incomparability_reason: str | None = None,
) -> dict[str, Any]:
    """Compare the predeclared functional counterpart to the frozen Qwen sign matrix."""
    reference = qwen_matrix or FROZEN_QWEN_MATRIX
    if not comparable:
        return _pack(
            "MODEL_B_NOT_COMPARABLE",
            key_difference=incomparability_reason or "Model B could not be compared under the frozen protocol.",
            sign_differences=[],
        )
    if cells is None or set(cells) != set(HEADS):
        return _pack(
            "M1_INCONCLUSIVE",
            key_difference="A functional-counterpart slot is missing, so the matrices were not compared.",
            sign_differences=[],
        )
    separable = 0
    total = 0
    signs_match = True
    differences = []
    for slot in HEADS:
        regimes = cells.get(slot, {})
        if set(regimes) != set(FROZEN_REGIMES):
            return _pack(
                "M1_INCONCLUSIVE",
                key_difference=f"{slot} is missing a frozen regime.",
                sign_differences=[],
            )
        for regime in FROZEN_REGIMES:
            cell = regimes[regime]
            if cell.get("status") == "INSUFFICIENT_DATA" or "sign_symbol" not in cell:
                return _pack(
                    "M1_INCONCLUSIVE",
                    key_difference="A regime cell has no measured sign.",
                    sign_differences=[],
                )
            total += 1
            if cell.get("separable_from_regime_null"):
                separable += 1
            observed = cell["sign_symbol"]
            expected = reference[slot][regime]
            if observed != expected:
                signs_match = False
                differences.append(f"{slot} {regime}: Qwen {expected}, Model B {observed}")
    if separable == 0:
        label = "MODEL_B_SIGNAL_NOT_SEPARATED_FROM_NULL"
        difference = "No functional-counterpart regime cell on Model B separated from that regime's own null."
    elif separable == total and signs_match:
        label = "MODEL_B_SIGNAL_REPRODUCED"
        difference = (
            "The functional-counterpart sign matrix matches the frozen Qwen matrix, "
            "and every Model B cell separated from its own regime null."
        )
    elif separable > 0:
        label = "MODEL_B_SIGNAL_PRESENT_BUT_DIFFERENT"
        shown = "; ".join(differences) if differences else "separability differs while signs match"
        difference = (
            "Model B has regime cells separated from their own nulls, "
            f"but the pattern is not the frozen Qwen pattern. Differences: {shown}."
        )
    else:
        label = "M1_INCONCLUSIVE"
        difference = "The measured cells did not determine a comparison."
    return _pack(label, key_difference=difference, sign_differences=differences)


def _pack(label: str, *, key_difference: str, sign_differences: list[str]) -> dict[str, Any]:
    inferred = {
        "MODEL_B_SIGNAL_REPRODUCED": (
            "The same conceptual intervention protocol produced the same regime sign pattern "
            "on Model B. This does not say that MRSM is wrong, and it does not identify a mechanism."
        ),
        "MODEL_B_SIGNAL_PRESENT_BUT_DIFFERENT": (
            "A regime-level signal separates from Model B's own nulls, and the sign pattern "
            "differs from Qwen. The Qwen profile may be model-dependent. "
            "This is not a claim that Model B is better."
        ),
        "MODEL_B_SIGNAL_NOT_SEPARATED_FROM_NULL": (
            "The cross-model causal signal was not reproduced. "
            "That does not prove the intervention method is invalid."
        ),
        "MODEL_B_NOT_COMPARABLE": (
            "No cross-model signal comparison was made."
        ),
        "M1_INCONCLUSIVE": (
            "The measurements do not decide whether Model B reproduces the Qwen regime pattern."
        ),
    }
    return {
        "m1_decision": label,
        "key_difference": key_difference,
        "sign_differences": sign_differences,
        "mechanism_identity": "NOT_EVALUATED",
        "m2_executed": False,
        "m3_executed": False,
        "inferred": inferred[label],
        "not_established": [
            "mechanism identity",
            "that one model is better",
            "that MRSM is wrong",
            "a cross-model regime pattern from M2",
            "a model-independent causal signature from M3",
        ],
    }
