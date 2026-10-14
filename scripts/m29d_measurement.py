"""M29-D execution-freeze measurement implementation.

This module computes the frozen pre-outcome quantities g, F''(0), and
kappa for the inherited D1 mechanism family. It never runs a scored
intervention and never accepts an observed outcome.

Frozen implementation:
    F(t) = logit_margin(model with t*d added at the last token of
                         the frozen residual hook)
    g = F'(0)
    kappa = alpha**2 / 2 * F''(0)
    r_hat = kappa
    y_hat = g + kappa

The second derivative is obtained by automatic differentiation:
    grad_F = dF/dx
    g = <grad_F, d>
    hv = d/dx <grad_F, d> = H*d
    F''(0) = <hv, d>

No polynomial fitting or finite differences are used.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
HOOKS = {
    "M22.1-D1-L0": "blocks.0.hook_resid_post",
    "M22.1-D1-L8": "blocks.8.hook_resid_post",
    "M22.1-D1-L15": "blocks.15.hook_resid_post",
}
LAYERS = (0, 8, 15)
DIRECTION_SEED = 22101
DIRECTION_DIMENSION = 896
DIRECTION_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
ALPHA = 1.0
POSITIVE_ID = 9834
NEGATIVE_ID = 902
NUMERICAL_TOLERANCE = 1e-6
DTYPE_NAME = "float32"
MEASUREMENT_DEVICE_POLICY = "cuda_if_available_else_cpu"


def _direction() -> np.ndarray:
    from src.cognitive_self_model.m23.m22_reuse import primary_direction, vector_sha256
    d = primary_direction(DIRECTION_DIMENSION, DIRECTION_SEED)
    if vector_sha256(d) != DIRECTION_SHA256:
        raise RuntimeError("D1 direction hash mismatch")
    return d


def _as_logits(output: Any):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("model forward did not return logits")


def _margin(logits, positive_id: int = POSITIVE_ID, negative_id: int = NEGATIVE_ID):
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("expected logits [1, position, vocab]")
    value = logits[0, -1, positive_id] - logits[0, -1, negative_id]
    if not torch.isfinite(value):
        raise ValueError("non-finite margin")
    return value


def measure_kappa(model, tokens, layer: int, direction: np.ndarray) -> dict[str, float]:
    import torch

    hook_name = HOOKS[f"M22.1-D1-L{layer}"]
    captured: dict[str, torch.Tensor] = {}

    def capture(resid, hook):
        if not resid.requires_grad:
            raise RuntimeError("hook residual does not require grad")
        captured["resid"] = resid
        return resid

    device = next(model.parameters()).device
    d = torch.tensor(direction, dtype=torch.float32, device=device)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        logits = _as_logits(model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, capture)],
        ))
        margin = logits[0, -1, POSITIVE_ID] - logits[0, -1, NEGATIVE_ID]
        resid = captured.get("resid")
        if resid is None:
            raise RuntimeError("measurement hook did not fire")
        if resid.shape[-1] != DIRECTION_DIMENSION:
            raise RuntimeError("unexpected residual dimension")

        grad = torch.autograd.grad(
            margin,
            resid,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]

        local_grad = grad[0, -1]
        g_tensor = torch.sum(local_grad * d)

        hv = torch.autograd.grad(
            g_tensor,
            resid,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        h_d_d = torch.sum(hv[0, -1] * d)
        kappa = (ALPHA * ALPHA / 2.0) * h_d_d

    values = {
        "g": float(g_tensor.detach().cpu().item()),
        "f_second": float(h_d_d.detach().cpu().item()),
        "kappa": float(kappa.detach().cpu().item()),
        "r_hat_m29": float(kappa.detach().cpu().item()),
        "y_hat_m29": float((g_tensor + kappa).detach().cpu().item()),
    }
    if not all(np.isfinite(v) for v in values.values()):
        raise RuntimeError("non-finite M29 measurement")
    model.zero_grad(set_to_none=True)
    return values


def measurement_manifest() -> dict[str, Any]:
    payload = {
        "model": MODEL_ID,
        "model_revision": REVISION,
        "hooks": HOOKS,
        "layers": list(LAYERS),
        "direction_seed": DIRECTION_SEED,
        "direction_dimension": DIRECTION_DIMENSION,
        "direction_sha256": DIRECTION_SHA256,
        "alpha": ALPHA,
        "positive_token_id": POSITIVE_ID,
        "negative_token_id": NEGATIVE_ID,
        "dtype": DTYPE_NAME,
        "device_policy": MEASUREMENT_DEVICE_POLICY,
        "numerical_tolerance": NUMERICAL_TOLERANCE,
        "autodiff": "torch.autograd.grad_create_graph_hessian_vector_product",
        "finite_difference": False,
        "outcome_inputs": False,
        "residual_inputs": False,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
