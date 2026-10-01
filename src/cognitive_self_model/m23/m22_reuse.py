"""M22.1 functions copied for M23.

Source: origin/cursor/mrsm-first-scientific-run-4e19
commit aba115119f81c1737e5a964458b701ddd82be5eb
files: m22_1/prompts.py, direction.py, intervention.py, outcome.py

Behavior of those functions is unchanged. The residual hook records one extra
field, pre_dot, which M23 logs and does not pass to predict.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

ROLES = ("discovery", "validation", "replication")

_TEXTS: dict[str, tuple[str, ...]] = {
    "completion": (
        "The capital of France is",
        "Water freezes at a temperature of",
        "The opposite of hot is",
        "Two plus two equals",
        "The sun rises in the",
        "A cat is a type of",
    ),
    "syntax": (
        "If it rains, then the ground is",
        "She opened the door and",
        "Because the road was closed,",
        "After the meeting ended,",
        "Although the box was empty,",
        "Before sunrise, the sky was",
    ),
    "instruction": (
        "Reply with one word. Color of the sky:",
        "Reply with one word. A common pet:",
        "Reply with one word. Result of 1+1:",
        "Name a day of the week:",
        "Name a primary color:",
        "Name a season of the year:",
    ),
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    regime_id: str
    role: str
    text: str
    prompt_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_role(index_within_regime: int) -> str:
    if index_within_regime < 0:
        raise ValueError("Prompt index must be non-negative.")
    return ROLES[index_within_regime % 3]


def frozen_prompts() -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for regime_id in sorted(_TEXTS):
        texts = _TEXTS[regime_id]
        for index, text in enumerate(texts):
            prompt_id = f"{regime_id}-{index + 1:02d}"
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    regime_id=regime_id,
                    role=assign_role(index),
                    text=text,
                    prompt_sha256=prompt_sha256(text),
                )
            )
    records.sort(key=lambda record: record.prompt_id)
    return records


def prompt_manifest_sha256(prompts: list[PromptRecord] | None = None) -> str:
    catalog = prompts if prompts is not None else frozen_prompts()
    payload = [record.to_dict() for record in catalog]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unit_gaussian(dimension: int, seed: int) -> np.ndarray:
    if dimension < 2:
        raise ValueError("Direction dimension must be at least 2.")
    vector = np.random.default_rng(int(seed)).normal(size=int(dimension)).astype(np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Failed to sample a finite direction.")
    return vector / norm


def vector_sha256(vector: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(vector, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def primary_direction(dimension: int, seed: int) -> np.ndarray:
    return _unit_gaussian(dimension, seed)


def orthogonal_direction(dimension: int, seed: int, primary: np.ndarray) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    if primary.shape != (dimension,):
        raise ValueError("Primary direction shape does not match dimension.")
    primary_norm = float(np.linalg.norm(primary))
    if abs(primary_norm - 1.0) > 1e-6:
        raise ValueError("Primary direction must be unit length.")
    raw = _unit_gaussian(dimension, seed)
    projected = raw - float(np.dot(raw, primary)) * primary
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError("Orthogonal direction collapsed.")
    orthogonal = projected / norm
    if abs(float(np.dot(orthogonal, primary))) > 1e-6:
        raise RuntimeError("Orthogonalization failed.")
    return orthogonal


def resolve_outcome_tokens(tokenizer: Any, positive_text: str, negative_text: str) -> dict[str, Any]:
    positive_ids = [int(token_id) for token_id in tokenizer.encode(positive_text, add_special_tokens=False)]
    negative_ids = [int(token_id) for token_id in tokenizer.encode(negative_text, add_special_tokens=False)]
    if len(positive_ids) == 0 or len(negative_ids) == 0:
        raise RuntimeError("Outcome text produced an empty tokenization.")
    positive_id = positive_ids[0]
    negative_id = negative_ids[0]
    if positive_id == negative_id:
        raise RuntimeError("Outcome poles resolved to the same token id.")
    return {
        "type": "logit_margin",
        "positive_text": positive_text,
        "negative_text": negative_text,
        "positive_token_ids": positive_ids,
        "negative_token_ids": negative_ids,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "token_ids": [positive_id, negative_id],
        "single_piece": len(positive_ids) == 1 and len(negative_ids) == 1,
        "token_position": "last",
    }


def logit_margin(logits: torch.Tensor, positive_id: int, negative_id: int) -> float:
    if logits.ndim != 3:
        raise ValueError("Logits must have shape [batch, position, vocab].")
    if logits.shape[0] != 1:
        raise ValueError("M22.1 scores one prompt at a time.")
    last = logits[0, -1]
    if not torch.isfinite(last).all():
        raise ValueError("Logits contain non-finite values.")
    value = last[int(positive_id)] - last[int(negative_id)]
    if not torch.isfinite(value):
        raise ValueError("Outcome margin is non-finite.")
    return float(value.item())


def apply_last_token_additive(
    residual: torch.Tensor,
    alpha: float,
    direction: torch.Tensor,
) -> torch.Tensor:
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
    def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
        if recorder is not None:
            recorder["fired"] = int(recorder.get("fired", 0)) + 1
            recorder["hook_name"] = str(getattr(hook, "name", ""))
            recorder["shape"] = tuple(int(size) for size in residual.shape)
            pre = residual[:, -1, :].detach()
            direction_on_pre = direction.to(device=pre.device, dtype=pre.dtype)
            recorder["pre_dot"] = float(torch.dot(pre[0].float(), direction_on_pre.float()).item())
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
