"""Outcome-independent masks for the frozen prompt-by-direction grid."""

from __future__ import annotations

import numpy as np


def held_out_mask(
    n_prompts: int,
    n_directions: int,
    seed: int,
    base_pairs: list[list[int]],
) -> np.ndarray:
    """Return a boolean mask that is True on held-out cells.

    The base pairs are a fixed design. The seed only permutes direction
    labels and prompt assignment. Observed deltas are not an argument.
    """
    if len(base_pairs) != n_prompts:
        raise ValueError("The frozen mask design must have one pair per prompt.")
    rng = np.random.default_rng(int(seed))
    direction_labels = rng.permutation(n_directions)
    prompt_order = rng.permutation(n_prompts)
    mask = np.zeros((n_prompts, n_directions), dtype=bool)
    for template_row, pair in enumerate(base_pairs):
        if len(pair) != 2 or len(set(pair)) != 2:
            raise ValueError("Each frozen mask pair must contain two distinct directions.")
        prompt_index = int(prompt_order[template_row])
        for slot in pair:
            direction_index = int(direction_labels[int(slot)])
            if direction_index < 0 or direction_index >= n_directions:
                raise ValueError("Mask pair points outside the direction panel.")
            mask[prompt_index, direction_index] = True
    _assert_coverage(mask)
    return mask


def _assert_coverage(mask: np.ndarray) -> None:
    held_per_prompt = mask.sum(axis=1)
    held_per_direction = mask.sum(axis=0)
    train_per_direction = (~mask).sum(axis=0)
    if not np.all(held_per_prompt == 2):
        raise RuntimeError("Mask did not hold out exactly two cells per prompt.")
    if not np.all(held_per_direction >= 1):
        raise RuntimeError("Mask left a direction without a held-out cell.")
    if not np.all(train_per_direction >= 1):
        raise RuntimeError("Mask left a direction without a training cell.")
