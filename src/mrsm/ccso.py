"""Contextual Causal State Observation. Diagnostic, before any Self-Model.

The question is whether a richer observation predicts the intervention drop
Δ = m(h) - m(h + α v) on held-out prompts and unseen directions, beyond a
snapshot or the local gradient. This is not a Self-Model, not an MRSM gate,
and not a mechanism-identity claim.

O1 is operationalized as the local pair (h, g), with g = ∇_h m. The drop
being predicted is never copied into the features. O2 is a depth profile of
those pairs at the frozen Qwen layers, not a temporal trajectory.

Rules below are fixed before the Qwen measurement for this diagnostic.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from src.mrsm.qf_gradient_field import (
    QWEN_FUNCTIONAL_LAYERS,
    QWEN_NEGATIVE_ID,
    QWEN_POSITIVE_ID,
    QWEN_REVISION,
)
from src.mrsm.qf_m2 import pearson
from src.mrsm.qwen_bridge import SIGN_FLOOR

LAYERS = QWEN_FUNCTIONAL_LAYERS
PRIMARY_ALPHAS = (-0.10, -0.05, -0.01, 0.01, 0.05, 0.10)
NONLINEAR_ALPHAS = (-0.25, 0.25)
DIRECTION_SEEDS = tuple(range(23101, 23109))
TRAIN_DIRECTION_SEEDS = DIRECTION_SEEDS[:6]
NOVEL_DIRECTION_SEEDS = DIRECTION_SEEDS[6:]
REGIMES = ("completion", "instruction", "syntax")
CONDITIONS = ("B2", "B3", "C1", "C2")
READOUTS = ("B0", "B1", "B2", "B3", "C1", "C2")
PRIMARY_SLICES = ("S_val", "S_rep", "S_novel", "S_regime")
PROJ_DIM = 16
PROTECTED_DIM = 8
LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
MLP_EPOCHS = 400
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-2
MLP_HIDDEN = 16
MLP_LARGE_HIDDEN = 128
MLP_SEED = 24101
MLP_LARGE_SEED = 24102
NULL_SEEDS = {"N1": 25101, "N2": 25102, "N4": 25104}
PROJ_SEEDS = {"B2": 24100, "B3": 24110, "C1": 24120, "C2": 24130}

NO_ADDITIONAL = "NO_ADDITIONAL_INFORMATION"
SNAPSHOT = "SNAPSHOT_SUFFICIENT"
LOCAL = "LOCAL_CAUSAL_STATE_ADDS_INFORMATION"
MULTI = "MULTI_DEPTH_PROFILE_ADDS_INFORMATION"
INCONCLUSIVE = "INCONCLUSIVE"
DECISIONS = (NO_ADDITIONAL, SNAPSHOT, LOCAL, MULTI, INCONCLUSIVE)


def feature_dim(n_layers: int = len(LAYERS)) -> int:
    return PROTECTED_DIM + PROJ_DIM + int(n_layers)


def fixed_projection(in_dim: int, out_dim: int, seed: int) -> np.ndarray:
    if int(in_dim) < int(out_dim):
        raise ValueError("Projection input must be at least the output width.")
    raw = np.random.default_rng(int(seed)).normal(size=(int(in_dim), int(out_dim)))
    q, _ = np.linalg.qr(raw)
    return np.ascontiguousarray(q[:, : int(out_dim)].T)


def make_projections(d_model: int, n_layers: int = len(LAYERS)) -> dict[str, np.ndarray]:
    width = int(d_model)
    depth = int(n_layers)
    return {
        "B2": fixed_projection(width, PROJ_DIM, PROJ_SEEDS["B2"]),
        "B3": fixed_projection(width, PROJ_DIM, PROJ_SEEDS["B3"]),
        "C1": fixed_projection(2 * width, PROJ_DIM, PROJ_SEEDS["C1"]),
        "C2": fixed_projection(depth * 2 * width, PROJ_DIM, PROJ_SEEDS["C2"]),
    }


def _protected(
    kind: str,
    alpha: float,
    dot_h: float,
    dot_g: float,
    norm_h: float,
    norm_g: float,
    depth_dot_h: float,
    depth_dot_g: float,
) -> np.ndarray:
    if kind == "B2":
        values = [1.0, alpha, dot_h, alpha * dot_h, norm_h, 0.0, 0.0, 0.0]
    elif kind == "B3":
        values = [1.0, alpha, dot_g, alpha * dot_g, norm_g, 0.0, 0.0, 0.0]
    elif kind == "C1":
        values = [1.0, alpha, dot_h, alpha * dot_h, dot_g, alpha * dot_g, norm_h, norm_g]
    elif kind == "C2":
        values = [1.0, alpha, dot_h, alpha * dot_h, dot_g, alpha * dot_g, depth_dot_g, depth_dot_h]
    else:
        raise ValueError(f"Unknown observation {kind}")
    return np.asarray(values, dtype=np.float64)


def row_features(
    kind: str,
    h_all: np.ndarray,
    g_all: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    layer_index: int,
    projections: dict[str, np.ndarray],
) -> np.ndarray:
    """One readout row. The predicted drop is not an input."""
    h_layer = np.asarray(h_all[layer_index], dtype=np.float64)
    g_layer = np.asarray(g_all[layer_index], dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    dot_h = float(h_layer @ direction)
    dot_g = float(g_layer @ direction)
    others = [index for index in range(h_all.shape[0]) if index != int(layer_index)]
    if others:
        depth_dot_h = float(np.mean([h_all[index] @ direction for index in others]))
        depth_dot_g = float(np.mean([g_all[index] @ direction for index in others]))
    else:
        depth_dot_h = 0.0
        depth_dot_g = 0.0
    protected = _protected(
        kind,
        float(alpha),
        dot_h,
        dot_g,
        float(np.linalg.norm(h_layer)),
        float(np.linalg.norm(g_layer)),
        depth_dot_h,
        depth_dot_g,
    )
    if kind == "B2":
        projected = projections["B2"] @ h_layer
    elif kind == "B3":
        projected = projections["B3"] @ g_layer
    elif kind == "C1":
        projected = projections["C1"] @ np.concatenate([h_layer, g_layer])
    else:
        packed = np.concatenate([np.concatenate([h_all[index], g_all[index]]) for index in range(h_all.shape[0])])
        projected = projections["C2"] @ packed
    one_hot = np.zeros(h_all.shape[0], dtype=np.float64)
    one_hot[int(layer_index)] = 1.0
    features = np.concatenate([protected, np.asarray(projected, dtype=np.float64), one_hot])
    if features.shape != (feature_dim(h_all.shape[0]),):
        raise RuntimeError("Observation feature width drifted.")
    return features


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape or actual.size == 0:
        return {"mae": None, "sign_agreement": None, "n_sign": 0, "pearson": None, "n": int(actual.size)}
    mae = float(np.mean(np.abs(actual - predicted)))
    comparable = np.abs(actual) > SIGN_FLOOR
    n_sign = int(comparable.sum())
    if n_sign == 0:
        agreement = None
    else:
        hits = 0
        for left, right in zip(actual[comparable], predicted[comparable]):
            if abs(float(right)) <= SIGN_FLOOR:
                continue
            if np.sign(left) == np.sign(right):
                hits += 1
        agreement = hits / n_sign
    return {
        "mae": mae,
        "sign_agreement": agreement,
        "n_sign": n_sign,
        "pearson": pearson(actual.tolist(), predicted.tolist()),
        "n": int(actual.size),
        "decision_uses_pearson": False,
    }


def beats(left: dict[str, Any] | None, right: dict[str, Any] | None) -> bool:
    """Strict MAE drop and strict sign-agreement gain. Pearson is not an input."""
    if not left or not right:
        return False
    if int(left.get("n_sign", 0)) <= 0 or int(right.get("n_sign", 0)) <= 0:
        return False
    if left.get("mae") is None or right.get("mae") is None:
        return False
    if left.get("sign_agreement") is None or right.get("sign_agreement") is None:
        return False
    return float(left["mae"]) < float(right["mae"]) and float(left["sign_agreement"]) > float(right["sign_agreement"])


def _fit_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body = features[:, 1:]
    center = body.mean(axis=0)
    scale = body.std(axis=0)
    constant = scale <= 1e-8
    safe = np.where(constant, 1.0, scale)
    return center, safe, constant


def _apply_scaler(
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    constant: np.ndarray,
) -> np.ndarray:
    scaled = np.array(features, dtype=np.float64, copy=True)
    scaled[:, 1:] = (features[:, 1:] - center) / scale
    scaled[:, 1:][:, constant] = 0.0
    return scaled


def ridge_fit(features: np.ndarray, target: np.ndarray, lam: float) -> np.ndarray:
    gram = features.T @ features
    penalty = float(lam) * np.eye(features.shape[1])
    penalty[0, 0] = 0.0
    solved = np.linalg.solve(gram + penalty, features.T @ target)
    return np.asarray(solved, dtype=np.float64)


def choose_lambda(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    prompt_index: np.ndarray,
) -> float:
    prompts = sorted({int(value) for value in prompt_index[train_mask]})
    best_lam = float(LAMBDA_GRID[0])
    best_score = None
    for lam in LAMBDA_GRID:
        losses = []
        for held in prompts:
            train = train_mask & (prompt_index != held)
            held_rows = train_mask & (prompt_index == held)
            if int(train.sum()) == 0 or int(held_rows.sum()) == 0:
                continue
            center, scale, constant = _fit_scaler(features[train])
            beta = ridge_fit(_apply_scaler(features[train], center, scale, constant), target[train], lam)
            pred = _apply_scaler(features[held_rows], center, scale, constant) @ beta
            losses.append(float(np.mean(np.abs(target[held_rows] - pred))))
        if not losses:
            continue
        score = float(np.mean(losses))
        if best_score is None or score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam


def linear_predict(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    prompt_index: np.ndarray,
) -> tuple[np.ndarray, float]:
    lam = choose_lambda(features, target, train_mask, prompt_index)
    center, scale, constant = _fit_scaler(features[train_mask])
    beta = ridge_fit(_apply_scaler(features[train_mask], center, scale, constant), target[train_mask], lam)
    pred = _apply_scaler(features, center, scale, constant) @ beta
    return pred, lam


def mlp_predict(
    features: np.ndarray,
    target: np.ndarray,
    train_mask: np.ndarray,
    *,
    seed: int,
    hidden: int,
    epochs: int = MLP_EPOCHS,
) -> np.ndarray:
    center, scale, constant = _fit_scaler(features[train_mask])
    train_x = _apply_scaler(features[train_mask], center, scale, constant)
    all_x = _apply_scaler(features, center, scale, constant)
    torch.manual_seed(int(seed))
    if int(hidden) == MLP_LARGE_HIDDEN:
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], int(hidden)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden), int(hidden)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden), 1),
        )
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], int(hidden)),
            torch.nn.ReLU(),
            torch.nn.Linear(int(hidden), 1),
        )
    x_train = torch.tensor(train_x, dtype=torch.float32)
    y_train = torch.tensor(target[train_mask], dtype=torch.float32).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(all_x, dtype=torch.float32)).squeeze(1).cpu().numpy()
    return np.asarray(pred, dtype=np.float64)


def preregistration_document() -> dict[str, Any]:
    return {
        "experiment": "CCSO",
        "status": "diagnostic_pre_self_model",
        "self_model_built": False,
        "mrsm_modified": False,
        "qwen_modified": False,
        "fine_tuning": False,
        "cross_model": False,
        "model": "Qwen/Qwen2.5-0.5B",
        "revision": QWEN_REVISION,
        "layers": list(LAYERS),
        "primary_alphas": [format(alpha, ".2f") for alpha in PRIMARY_ALPHAS],
        "nonlinear_probe_alphas": [format(alpha, ".2f") for alpha in NONLINEAR_ALPHAS],
        "historical_alpha_pm_1_in_primary_endpoint": False,
        "direction_seeds": list(DIRECTION_SEEDS),
        "train_direction_seeds": list(TRAIN_DIRECTION_SEEDS),
        "novel_direction_seeds": list(NOVEL_DIRECTION_SEEDS),
        "observation": {
            "O0": "h at the intervened layer",
            "O1": "(h, g) at the intervened layer; g is the direct residual gradient, not the target drop",
            "O2": "h and g at every frozen layer; a depth profile, not a temporal trajectory",
        },
        "target": "delta = m(h) - m(h + alpha v)",
        "target_excluded_from_features": True,
        "held_out_prompts_excluded_from_fitting": True,
        "feature_dim": feature_dim(),
        "same_readout_width_for_B2_B3_C1_C2": True,
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_selection": "leave-one-discovery-prompt-out MAE; smallest lambda wins ties",
        "mlp_epochs": MLP_EPOCHS,
        "mlp_hidden": MLP_HIDDEN,
        "mlp_large_hidden": MLP_LARGE_HIDDEN,
        "decision_uses_pearson": False,
        "decision_uses_nonlinear_probe": False,
        "decision_uses_regime_permutation": False,
        "nulls_that_can_block_a_positive_claim": ["N1", "N2", "N4"],
        "question": (
            "Does a richer observation supply counterfactual information about Qwen's "
            "causal response that is not available from the snapshot or the local gradient?"
        ),
        "negative_result_scope": (
            "A negative result says that additional information was not shown on this frozen "
            "Qwen revision and these frozen prompts. It does not say a richer observation cannot exist."
        ),
    }


def _slice_ok(block: dict[str, Any] | None) -> bool:
    return bool(block) and int(block.get("n_sign", 0)) > 0 and block.get("mae") is not None and block.get("sign_agreement") is not None


def _family_ready(metrics: dict[str, Any], family: str, names: tuple[str, ...], slices: tuple[str, ...]) -> bool:
    family_metrics = metrics.get(family, {})
    return all(_slice_ok(family_metrics.get(name, {}).get(sl)) for name in names for sl in slices)


def _capacity_confound(metrics: dict[str, Any]) -> bool:
    large = metrics.get("mlp_large", {}).get("B2", {})
    small = metrics.get("small_mlp", {}).get("B2", {})
    if not all(_slice_ok(large.get(sl)) and _slice_ok(small.get(sl)) for sl in ("S_val", "S_rep")):
        return False
    changes = all(float(large[sl]["mae"]) < float(small[sl]["mae"]) for sl in ("S_val", "S_rep"))
    if not changes:
        return False
    for sl in ("S_val", "S_rep"):
        if float(large[sl]["mae"]) > float(metrics["small_mlp"]["C2"][sl]["mae"]):
            return False
        if float(large[sl]["mae"]) > float(metrics["linear"]["C2"][sl]["mae"]):
            return False
    return True


def _null_explained(metrics: dict[str, Any], name: str) -> bool:
    real = metrics["linear"][name]["S_val"]["mae"]
    for null in ("N1", "N2", "N4"):
        block = metrics.get("nulls", {}).get(name, {}).get(null, {}).get("S_val")
        if not block or block.get("mae") is None:
            return True
        if float(block["mae"]) <= float(real):
            return True
    return False


def _sensitivity_ok(metrics: dict[str, Any], name: str) -> bool:
    block = metrics.get("direction_sensitivity", {}).get(name)
    if not block:
        return False
    if int(block["n_defined"]) * 2 < int(block["n_groups"]):
        return False
    pearson_value = block.get("mean_pearson")
    return pearson_value is not None and float(pearson_value) > 0.0


def _linear_multi(metrics: dict[str, Any]) -> bool:
    lin = metrics["linear"]
    if not all(beats(lin["C2"][sl], lin["B2"][sl]) for sl in PRIMARY_SLICES):
        return False
    if not all(beats(lin["C2"][sl], lin["B3"][sl]) for sl in ("S_val", "S_rep")):
        return False
    if not all(beats(lin["C2"][sl], lin["C1"][sl]) for sl in ("S_val", "S_rep")):
        return False
    return all(beats(lin["C2"][sl], lin["B1"][sl]) for sl in ("S_val", "S_rep"))


def _mlp_multi(metrics: dict[str, Any]) -> bool:
    mlp = metrics["small_mlp"]
    if not all(beats(mlp["C2"][sl], mlp["B2"][sl]) for sl in ("S_val", "S_rep", "S_novel", "S_regime")):
        return False
    if not all(beats(mlp["C2"][sl], mlp["B3"][sl]) for sl in ("S_val", "S_rep")):
        return False
    return all(beats(mlp["C2"][sl], mlp["C1"][sl]) for sl in ("S_val", "S_rep"))


def _linear_local(metrics: dict[str, Any]) -> bool:
    lin = metrics["linear"]
    if not all(beats(lin["C1"][sl], lin["B2"][sl]) for sl in PRIMARY_SLICES):
        return False
    if not all(beats(lin["C1"][sl], lin["B3"][sl]) for sl in ("S_val", "S_rep")):
        return False
    if not all(beats(lin["C1"][sl], lin["B1"][sl]) for sl in ("S_val", "S_rep")):
        return False
    depth_also = beats(lin["C2"]["S_val"], lin["C1"]["S_val"]) and beats(lin["C2"]["S_rep"], lin["C1"]["S_rep"])
    return not depth_also


def _mlp_local(metrics: dict[str, Any]) -> bool:
    mlp = metrics["small_mlp"]
    if not all(beats(mlp["C1"][sl], mlp["B2"][sl]) for sl in ("S_val", "S_rep", "S_novel", "S_regime")):
        return False
    return all(beats(mlp["C1"][sl], mlp["B3"][sl]) for sl in ("S_val", "S_rep"))


def _snapshot(metrics: dict[str, Any]) -> bool:
    lin = metrics["linear"]
    if not all(beats(lin["B2"][sl], lin["B0"][sl]) for sl in ("S_val", "S_rep")):
        return False
    if not all(beats(lin["B2"][sl], lin["B1"][sl]) for sl in ("S_val", "S_rep")):
        return False
    for richer in ("B3", "C1", "C2"):
        if any(beats(lin[richer][sl], lin["B2"][sl]) for sl in ("S_val", "S_rep")):
            return False
    return True


def _no_additional(metrics: dict[str, Any]) -> bool:
    lin = metrics["linear"]
    if not all(beats(lin["B3"][sl], lin["B2"][sl]) for sl in ("S_val", "S_rep")):
        return False
    return not any(beats(lin[richer][sl], lin["B3"][sl]) for richer in ("C1", "C2") for sl in ("S_val", "S_rep"))


def decide(metrics: dict[str, Any]) -> dict[str, Any]:
    ready = _family_ready(metrics, "linear", READOUTS, PRIMARY_SLICES) and _family_ready(
        metrics, "small_mlp", CONDITIONS, PRIMARY_SLICES
    )
    if not ready or "mlp_large" not in metrics:
        label = INCONCLUSIVE
        reason = "A required slice or readout is missing."
    elif _capacity_confound(metrics):
        label = INCONCLUSIVE
        reason = "A larger snapshot readout matches the richer observations, so representation is not separated from capacity."
    elif _linear_multi(metrics):
        if _mlp_multi(metrics) and not _null_explained(metrics, "C2") and _sensitivity_ok(metrics, "C2"):
            label = MULTI
            reason = (
                "The multi-depth profile improves on the snapshot, the local gradient, and the "
                "single-layer state on the preregistered holdouts, unseen directions, and held-out regimes."
            )
        else:
            label = INCONCLUSIVE
            reason = "The linear multi-depth comparison passed and a confirmation check did not."
    elif _linear_local(metrics):
        if _mlp_local(metrics) and not _null_explained(metrics, "C1") and _sensitivity_ok(metrics, "C1"):
            label = LOCAL
            reason = (
                "State plus the local gradient improves on the snapshot and on the gradient alone. "
                "The depth profile does not add a further consistent gain."
            )
        else:
            label = INCONCLUSIVE
            reason = "The linear local-state comparison passed and a confirmation check did not."
    elif _snapshot(metrics):
        label = SNAPSHOT
        reason = "The snapshot readout predicts the held-out drop, and the richer observations do not strictly improve both endpoints."
    elif _no_additional(metrics):
        label = NO_ADDITIONAL
        reason = (
            "The local gradient improves on the snapshot. Adding state or depth does not strictly "
            "improve both endpoints. Observation enrichment beyond the local gradient was not shown."
        )
    else:
        label = INCONCLUSIVE
        reason = "The preregistered comparisons do not meet one decision rule."
    return {
        "decision": label,
        "reason": reason,
        "mechanism_identity": "NOT_EVALUATED",
        "self_model_built": False,
        "mrsm_modified": False,
        "qwen_modified": False,
        "trajectory_encoder_built": False,
        "primary_interpretation": _interpretation(label, reason),
    }


def _interpretation(label: str, reason: str) -> str:
    scope = (
        " This result is local to the frozen Qwen revision and the frozen prompts. "
        "It does not say a richer observation cannot exist. "
        "No mechanism was identified. A Self-Model was not built. "
        "Pearson correlation, the alpha = ±0.25 probe, and the regime-label permutation are not decision inputs."
    )
    if label == MULTI:
        follow = (
            " The next experiment, which was not run, would ask whether this information can support "
            "a falsifiable causal model rather than only a predictor."
        )
    elif label == NO_ADDITIONAL:
        follow = " This result does not support building a trajectory encoder from these observations."
    else:
        follow = ""
    return reason + follow + scope


def residual_state_and_gradient(
    run_with_hooks: Callable[..., torch.Tensor],
    tokens: torch.Tensor,
    layer: int,
    positive_id: int,
    negative_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return last-token h and ∇_h m at one residual site. The drop is not computed here."""
    saved: dict[str, torch.Tensor] = {}
    hook_name = f"blocks.{int(layer)}.hook_resid_post"

    def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
        if residual.ndim != 3:
            raise ValueError("Residual must have shape [batch, position, d_model].")
        saved["h"] = residual.detach()[0, -1].to(dtype=torch.float64).cpu().numpy().copy()
        delta = torch.zeros_like(residual)
        delta.requires_grad_(True)
        saved["delta"] = delta
        return residual + delta

    with torch.enable_grad():
        logits = run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
        margin = logits[0, -1, int(positive_id)] - logits[0, -1, int(negative_id)]
        if "delta" not in saved or "h" not in saved:
            raise RuntimeError(f"{hook_name} did not run.")
        margin.backward()
        grad = saved["delta"].grad
        if grad is None:
            raise RuntimeError("Direct residual gradient is missing.")
        gradient = grad[0, -1].detach().to(dtype=torch.float64).cpu().numpy().copy()
    state = np.asarray(saved["h"], dtype=np.float64).copy()
    saved.clear()
    return state, gradient


def _roles_masks(roles: list[str], regimes: list[str], meta: np.ndarray, alphas: np.ndarray, seeds: list[int]):
    prompt = meta[:, 0]
    direction = meta[:, 2]
    alpha_index = meta[:, 3]
    role = np.asarray([roles[int(index)] for index in prompt])
    regime = np.asarray([regimes[int(index)] for index in prompt])
    primary = np.asarray([format(float(alphas[int(index)]), ".2f") in {format(value, ".2f") for value in PRIMARY_ALPHAS} for index in alpha_index])
    nonlinear = np.asarray([format(float(alphas[int(index)]), ".2f") in {format(value, ".2f") for value in NONLINEAR_ALPHAS} for index in alpha_index])
    train_dir = np.isin(direction, [index for index, seed in enumerate(seeds) if int(seed) in TRAIN_DIRECTION_SEEDS])
    novel_dir = np.isin(direction, [index for index, seed in enumerate(seeds) if int(seed) in NOVEL_DIRECTION_SEEDS])
    discovery = role == "discovery"
    validation = role == "validation"
    replication = role == "replication"
    return {
        "prompt": prompt,
        "regime": regime,
        "train": discovery & train_dir & primary,
        "S_val": validation & train_dir & primary,
        "S_rep": replication & train_dir & primary,
        "S_novel": discovery & novel_dir & primary,
        "nonlinear_val": validation & train_dir & nonlinear,
        "train_dir": train_dir,
        "primary": primary,
        "discovery": discovery,
        "validation": validation,
        "replication": replication,
    }


def _build_design(
    h: np.ndarray,
    g: np.ndarray,
    directions: np.ndarray,
    alphas: np.ndarray,
    projections: dict[str, np.ndarray],
    donor: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    n_prompts, n_layers, _width = h.shape
    if donor is None:
        donor = np.arange(n_prompts)
    blocks = {kind: [] for kind in CONDITIONS}
    meta = []
    for prompt_index in range(n_prompts):
        state_index = int(donor[prompt_index])
        for layer_index in range(n_layers):
            for direction_index in range(directions.shape[0]):
                for alpha_index, alpha in enumerate(alphas):
                    meta.append((prompt_index, layer_index, direction_index, alpha_index))
                    for kind in CONDITIONS:
                        blocks[kind].append(
                            row_features(
                                kind,
                                h[state_index],
                                g[state_index],
                                directions[direction_index],
                                float(alpha),
                                layer_index,
                                projections,
                            )
                        )
    return {kind: np.vstack(rows) for kind, rows in blocks.items()}, np.asarray(meta, dtype=np.int64)


def _flat_target(delta: np.ndarray, meta: np.ndarray) -> np.ndarray:
    return np.asarray(
        [delta[int(row[0]), int(row[1]), int(row[2]), int(row[3])] for row in meta],
        dtype=np.float64,
    )


def _mean_baseline(target: np.ndarray, train_mask: np.ndarray, regime: np.ndarray, by_regime: bool) -> np.ndarray:
    pred = np.empty(target.shape[0], dtype=np.float64)
    global_mean = float(target[train_mask].mean()) if int(train_mask.sum()) else 0.0
    if not by_regime:
        pred.fill(global_mean)
        return pred
    means = {}
    for name in REGIMES:
        rows = train_mask & (regime == name)
        means[name] = float(target[rows].mean()) if int(rows.sum()) else global_mean
    for index, name in enumerate(regime):
        pred[index] = means.get(str(name), global_mean)
    return pred


def _macro(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks or any(block.get("mae") is None or block.get("sign_agreement") is None or int(block["n_sign"]) == 0 for block in blocks):
        return {"mae": None, "sign_agreement": None, "n_sign": 0, "pearson": None, "n": int(sum(int(block["n"]) for block in blocks)), "decision_uses_pearson": False}
    return {
        "mae": float(np.mean([block["mae"] for block in blocks])),
        "sign_agreement": float(np.mean([block["sign_agreement"] for block in blocks])),
        "n_sign": int(sum(int(block["n_sign"]) for block in blocks)),
        "pearson": None,
        "n": int(sum(int(block["n"]) for block in blocks)),
        "decision_uses_pearson": False,
    }


def _pack_slices(actual: np.ndarray, predicted: np.ndarray, masks: dict[str, np.ndarray], regime: np.ndarray, discovery: np.ndarray) -> dict[str, Any]:
    packed = {}
    for name in ("S_val", "S_rep", "S_novel"):
        packed[name] = prediction_metrics(actual[masks[name]], predicted[masks[name]])
    regime_blocks = []
    for name in REGIMES:
        test = (~discovery) & (regime == name) & masks["train_dir"] & masks["primary"]
        # Caller passes the fit-specific test via masks only for the main split.
        # Regime-holdout metrics are supplied by _macro from separate fits.
        regime_blocks.append(prediction_metrics(actual[test], predicted[test]))
    packed["S_regime_pooled_not_used"] = _macro(regime_blocks)
    return packed


def _shuffle_targets(target: np.ndarray, meta: np.ndarray, train_mask: np.ndarray, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    shuffled_rows = np.array(target, copy=True)
    train_rows = np.flatnonzero(train_mask)
    permutation = np.random.default_rng(NULL_SEEDS["N1"]).permutation(train_rows.size)
    shuffled_rows[train_rows] = target[train_rows][permutation]
    direction_shuffled = np.array(target, copy=True)
    train_directions = [index for index, seed in enumerate(seeds) if int(seed) in TRAIN_DIRECTION_SEEDS]
    direction_perm = np.random.default_rng(NULL_SEEDS["N2"]).permutation(len(train_directions))
    lookup = {(int(row[0]), int(row[1]), int(row[2]), int(row[3])): index for index, row in enumerate(meta)}
    prompts = sorted({int(row[0]) for row in meta[train_mask]})
    layers = sorted({int(row[1]) for row in meta})
    alpha_ids = sorted({int(row[3]) for row in meta[train_mask]})
    for prompt_index in prompts:
        for layer_index in layers:
            for alpha_index in alpha_ids:
                rows = [lookup[(prompt_index, layer_index, direction, alpha_index)] for direction in train_directions]
                for position, row in enumerate(rows):
                    direction_shuffled[row] = target[rows[int(direction_perm[position])]]
    return shuffled_rows, direction_shuffled


def _direction_sensitivity(
    actual: np.ndarray,
    predicted: np.ndarray,
    meta: np.ndarray,
    masks: dict[str, np.ndarray],
    alphas: np.ndarray,
    seeds: list[int],
) -> dict[str, Any]:
    alpha_index = next(index for index, value in enumerate(alphas) if format(float(value), ".2f") == "0.01")
    train_directions = [index for index, seed in enumerate(seeds) if int(seed) in TRAIN_DIRECTION_SEEDS]
    lookup = {(int(row[0]), int(row[1]), int(row[2]), int(row[3])): index for index, row in enumerate(meta)}
    prompts = sorted({int(row[0]) for row in meta[masks["S_val"]]})
    layers = sorted({int(row[1]) for row in meta})
    values = []
    groups = 0
    for prompt_index in prompts:
        for layer_index in layers:
            groups += 1
            rows = [lookup[(prompt_index, layer_index, direction, alpha_index)] for direction in train_directions]
            score = pearson(actual[rows].tolist(), predicted[rows].tolist())
            if score is not None:
                values.append(float(score))
    return {
        "mean_pearson": float(np.mean(values)) if values else None,
        "n_defined": len(values),
        "n_groups": groups,
        "decision_input": True,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_readouts(
    h: np.ndarray,
    g: np.ndarray,
    directions: np.ndarray,
    direction_seeds: list[int],
    delta: np.ndarray,
    alphas: np.ndarray,
    roles: list[str],
    regimes: list[str],
    *,
    mlp_epochs: int = MLP_EPOCHS,
    run_mlp: bool = True,
    run_large: bool = True,
    run_nulls: bool = True,
) -> dict[str, Any]:
    """Fit readouts on discovery rows only. Held-out prompts never enter the fit."""
    if list(direction_seeds) != list(DIRECTION_SEEDS):
        raise RuntimeError("Direction seeds do not match the preregistration.")
    if set(TRAIN_DIRECTION_SEEDS) & set(NOVEL_DIRECTION_SEEDS):
        raise RuntimeError("Novel directions overlap the training directions.")
    projections = make_projections(h.shape[-1], h.shape[1])
    features, meta = _build_design(h, g, directions, alphas, projections)
    actual = _flat_target(delta, meta)
    masks = _roles_masks(roles, regimes, meta, alphas, direction_seeds)
    if np.any(masks["train"] & masks["S_val"]) or np.any(masks["train"] & masks["S_rep"]) or np.any(masks["train"] & masks["S_novel"]):
        raise RuntimeError("A training row leaked into a primary evaluation slice.")
    prompt_index = masks["prompt"]
    regime = masks["regime"]

    def fit_family(train_mask: np.ndarray, feature_bank: dict[str, np.ndarray], target: np.ndarray, mlp: bool, large: bool = False) -> dict[str, Any]:
        out = {}
        out["B0"] = _slice_predictions(actual, _mean_baseline(target, train_mask, regime, False), masks, regime)
        out["B1"] = _slice_predictions(actual, _mean_baseline(target, train_mask, regime, True), masks, regime)
        for kind in CONDITIONS:
            pred, lam = linear_predict(feature_bank[kind], target, train_mask, prompt_index)
            block = _slice_predictions(actual, pred, masks, regime)
            block["lambda"] = lam
            block["predictor"] = "mlp_large" if large else ("small_mlp" if mlp else "linear")
            out[kind] = block
            out[kind]["_pred"] = pred
        if mlp:
            hidden = MLP_LARGE_HIDDEN if large else MLP_HIDDEN
            seed = MLP_LARGE_SEED if large else MLP_SEED
            names = ("B2",) if large else CONDITIONS
            for kind in names:
                pred = mlp_predict(feature_bank[kind], target, train_mask, seed=seed, hidden=hidden, epochs=mlp_epochs)
                block = _slice_predictions(actual, pred, masks, regime)
                block["_pred"] = pred
                out[kind] = block
        return out

    def _slice_predictions(actual_values, predicted, mask_book, regime_values):
        packed = {name: prediction_metrics(actual_values[mask_book[name]], predicted[mask_book[name]]) for name in ("S_val", "S_rep", "S_novel")}
        packed["S_regime"] = {"mae": None, "sign_agreement": None, "n_sign": 0, "pearson": None, "n": 0, "decision_uses_pearson": False}
        packed["_regime_values"] = regime_values
        return packed

    linear = fit_family(masks["train"], features, actual, mlp=False)
    # Regime holdout refits. Macro-average replaces the placeholder S_regime.
    for family_name, use_mlp, large in (("linear", False, False), ("small_mlp", True, False)):
        if family_name == "small_mlp" and not run_mlp:
            continue
        bank = {}
        for kind in READOUTS:
            per_regime = []
            for held_regime in REGIMES:
                train = masks["discovery"] & masks["train_dir"] & masks["primary"] & (regime != held_regime)
                if kind in ("B0", "B1"):
                    pred = _mean_baseline(actual, train, regime, kind == "B1")
                elif family_name == "linear":
                    pred, _lam = linear_predict(features[kind], actual, train, prompt_index)
                else:
                    pred = mlp_predict(features[kind], actual, train, seed=MLP_SEED, hidden=MLP_HIDDEN, epochs=mlp_epochs)
                test = (~masks["discovery"]) & (regime == held_regime) & masks["train_dir"] & masks["primary"]
                per_regime.append(prediction_metrics(actual[test], pred[test]))
            bank[kind] = _macro(per_regime)
        if family_name == "linear":
            for kind in READOUTS:
                linear[kind]["S_regime"] = bank[kind]
        else:
            pass
        if family_name == "linear":
            linear_regime = bank
        else:
            mlp_regime = bank if run_mlp else None

    small = {}
    if run_mlp:
        small = fit_family(masks["train"], features, actual, mlp=True)
        for kind in READOUTS:
            small[kind]["S_regime"] = mlp_regime[kind]
    large_metrics = {}
    if run_large and run_mlp:
        large_fit = fit_family(masks["train"], features, actual, mlp=True, large=True)
        # fit_family with large=True still fills linear predictions first, then overwrites B2 with the large MLP.
        large_metrics = {"B2": {key: value for key, value in large_fit["B2"].items() if key != "_pred"}}

    nulls = {}
    if run_nulls:
        shuffled, direction_shuffled = _shuffle_targets(actual, meta, masks["train"], direction_seeds)
        donors = np.arange(h.shape[0])
        discovery_prompts = [index for index, role in enumerate(roles) if role == "discovery"]
        for position, donor_prompt in enumerate(discovery_prompts):
            donors[donor_prompt] = discovery_prompts[(position + 1) % len(discovery_prompts)]
        shifted, _meta = _build_design(h, g, directions, alphas, projections, donors)
        for kind in CONDITIONS:
            nulls[kind] = {}
            for null_name, target, feature_bank in (
                ("N1", shuffled, features),
                ("N2", direction_shuffled, features),
                ("N4", actual, shifted),
            ):
                pred, _lam = linear_predict(feature_bank[kind], target, masks["train"], prompt_index)
                nulls[kind][null_name] = {"S_val": prediction_metrics(actual[masks["S_val"]], pred[masks["S_val"]])}

    sensitivity = {kind: _direction_sensitivity(actual, linear[kind]["_pred"], meta, masks, alphas, direction_seeds) for kind in ("C1", "C2")}
    nonlinear = {
        kind: prediction_metrics(actual[masks["nonlinear_val"]], linear[kind]["_pred"][masks["nonlinear_val"]])
        for kind in CONDITIONS
    }
    cycled = {"completion": "instruction", "instruction": "syntax", "syntax": "completion"}
    global_mean = float(actual[masks["train"]].mean())
    permuted_means = {}
    for name in REGIMES:
        rows = masks["train"] & np.asarray([cycled[str(item)] == name for item in regime])
        permuted_means[name] = float(actual[rows].mean()) if int(rows.sum()) else global_mean
    regime_perm_pred = np.asarray([permuted_means[str(name)] for name in regime], dtype=np.float64)

    def _public(block: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in block.items() if key not in {"_pred", "_regime_values"}}

    metrics = {
        "linear": {kind: _public(linear[kind]) for kind in READOUTS},
        "small_mlp": {kind: _public(small[kind]) for kind in READOUTS} if small else {},
        "mlp_large": large_metrics,
        "nulls": nulls,
        "direction_sensitivity": sensitivity,
        "nonlinear_probe": {"S_val": nonlinear, "decision_input": False},
        "regime_permutation_b1": {"S_val": prediction_metrics(actual[masks["S_val"]], regime_perm_pred[masks["S_val"]]), "decision_input": False},
        "linear_regime_holdout": linear_regime,
    }
    return _jsonable(metrics)
