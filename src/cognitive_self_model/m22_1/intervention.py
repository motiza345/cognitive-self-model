"""Additive last-token residual intervention.

The intervention is h <- h + alpha * direction at the final position of one
named residual hook. Alpha 0 returns the input tensor unchanged.
"""

from __future__ import annotations

from typing import Any

import torch


def apply_last_token_additive(
    residual: torch.Tensor,
    alpha: float,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Return a residual tensor with an additive update at the last position."""
    if residual.ndim != 3:
        raise ValueError("Residual must have shape [batch, position, d_model].")
    if direction.ndim != 1 or direction.shape[0] != residual.shape[-1]:
        raise ValueError("Direction must be a vector matching d_model.")
    if not torch.isfinite(residual).all():
        raise ValueError("Residual contains non-finite values.")
    if float(alpha) == 0.0:
        return residual
    updated = residual.clone()
    step = float(alpha) * direction.to(device=updated.device, dtype=updated.dtype)
    updated[:, -1, :] = updated[:, -1, :] + step
    return updated


def make_resid_hook(
    alpha: float,
    direction: torch.Tensor,
    recorder: dict[str, Any] | None = None,
):
    """Build a TransformerLens hook that records whether it actually wrote."""

    def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
        if recorder is not None:
            recorder["fired"] = int(recorder.get("fired", 0)) + 1
            recorder["hook_name"] = str(getattr(hook, "name", ""))
            recorder["shape"] = tuple(int(size) for size in residual.shape)
        updated = apply_last_token_additive(residual, alpha, direction)
        if recorder is not None:
            last_delta = (updated[:, -1, :] - residual[:, -1, :]).abs().max().item()
            if residual.shape[1] > 1:
                other_delta = (updated[:, :-1, :] - residual[:, :-1, :]).abs().max().item()
            else:
                other_delta = 0.0
            recorder["max_abs_delta_last"] = float(last_delta)
            recorder["max_abs_delta_other"] = float(other_delta)
            recorder["last_modified"] = bool(last_delta > 0.0)
            recorder["other_unchanged"] = bool(other_delta == 0.0)
        return updated

    return hook_fn
