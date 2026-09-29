"""P intervention interface. Executes the existing MRSM-P v1 head ablation hooks."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mrsm_p_v1  # noqa: E402
import torch  # noqa: E402

from src.mrsm import HEADS


def parse_head(name: str) -> tuple[int, int]:
    if name not in HEADS:
        raise KeyError(name)
    return int(name[1]), int(name[3])


def _hooks(heads: list[tuple[int, int]]):
    if not heads:
        return None
    if len(heads) == 1:
        return [mrsm_p_v1.zero_head_hook(heads[0])]
    return mrsm_p_v1.multi_zero_hooks(heads)


def representation_summary(model, ds, heads: list[tuple[int, int]]) -> dict[str, float]:
    """Mean absolute head output at the query position. Not a mechanism label."""
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(z: torch.Tensor, hook):
            captured[layer] = z.detach()
            return z
        return hook

    hooks = [(f"blocks.{layer}.attn.hook_z", make_hook(layer)) for layer in range(2)]
    with torch.no_grad():
        model.run_with_hooks(ds.tokens, fwd_hooks=hooks)
    qpos = ds.query_pos
    summary = {}
    for name in HEADS:
        layer, head = parse_head(name)
        block = captured[layer][:, qpos, head, :]
        summary[name] = float(block.abs().mean())
    summary["ablated"] = [f"L{layer}H{head}" for layer, head in heads]
    return summary


def _example_margins(model, ds, hooks) -> torch.Tensor:
    logits = mrsm_p_v1.logits_at(model, ds, hooks)
    target = logits.gather(1, ds.targets[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, ds.targets[:, None], -torch.inf)
    return target - other.max(dim=-1).values


def execute(model, ds, mode: str, heads: list[tuple[int, int]], *, with_representation: bool = True) -> dict[str, Any]:
    if mode not in {"baseline", "intervention", "counterfactual"}:
        raise KeyError(mode)
    active = [] if mode == "baseline" else list(heads)
    hooks = _hooks(active)
    intact = _example_margins(model, ds, None)
    treated = _example_margins(model, ds, hooks)
    drop = intact - treated
    logits = mrsm_p_v1.logits_at(model, ds, hooks)
    correct = (logits.argmax(-1) == ds.targets).to(torch.int64)
    return {
        "mode": mode,
        "heads": [f"L{layer}H{head}" for layer, head in active],
        "observable_outcome": {
            "accuracy": float(correct.float().mean()),
            "margin": float(treated.mean()),
            "correct": [int(v) for v in correct.tolist()],
        },
        "internal_representation": representation_summary(model, ds, active) if with_representation else {},
        "behavioral_delta": {
            "margin_drop": float(drop.mean()),
            "per_example_margin_drop": [float(v) for v in drop.tolist()],
        },
    }


def discovery_effects(model, ds) -> tuple[dict[str, float], dict[str, float]]:
    singles: dict[str, float] = {}
    for name in HEADS:
        result = execute(model, ds, "intervention", [parse_head(name)], with_representation=False)
        singles[name] = float(result["behavioral_delta"]["margin_drop"])
    joints: dict[str, float] = {}
    for i, left in enumerate(HEADS):
        for right in HEADS[i + 1 :]:
            result = execute(model, ds, "counterfactual", [parse_head(left), parse_head(right)], with_representation=False)
            joints[f"{left}+{right}"] = float(result["behavioral_delta"]["margin_drop"])
    return singles, joints
