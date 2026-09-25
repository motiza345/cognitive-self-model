"""Frozen logit-margin outcome.

s = last-position logit of the first id of the positive text
    minus the last-position logit of the first id of the negative text.
"""

from __future__ import annotations

from typing import Any

import torch


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
