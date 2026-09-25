"""Pre-registered response predictors.

Every fit reads only cells where ``train`` is true. Values outside that mask
are ignored, including when they are present in the array.
"""

from __future__ import annotations

import numpy as np


def _observed(matrix: np.ndarray, train: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)[train]
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Training cells must be finite and non-empty.")
    return values


def predict_global_mean(matrix: np.ndarray, train: np.ndarray, **_ignored) -> np.ndarray:
    return np.full(matrix.shape, float(np.mean(_observed(matrix, train))), dtype=np.float64)


def predict_prompt_mean(matrix: np.ndarray, train: np.ndarray, **_ignored) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64)
    prediction = np.empty(source.shape, dtype=np.float64)
    for prompt in range(source.shape[0]):
        if not np.any(train[prompt]):
            raise ValueError("Prompt mean requires a training cell in every prompt.")
        prediction[prompt, :] = float(np.mean(source[prompt, train[prompt]]))
    return prediction


def predict_direction_mean(matrix: np.ndarray, train: np.ndarray, **_ignored) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64)
    prediction = np.empty(source.shape, dtype=np.float64)
    for direction in range(source.shape[1]):
        if not np.any(train[:, direction]):
            raise ValueError("Direction mean requires a training cell in every direction.")
        prediction[:, direction] = float(np.mean(source[train[:, direction], direction]))
    return prediction


def predict_additive(matrix: np.ndarray, train: np.ndarray, *, ridge_lambda: float, **_ignored) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float64)
    n_prompts, n_directions = source.shape
    rows = []
    targets = []
    for prompt, direction in zip(*np.where(train)):
        features = np.zeros(1 + n_prompts + n_directions, dtype=np.float64)
        features[0] = 1.0
        features[1 + prompt] = 1.0
        features[1 + n_prompts + direction] = 1.0
        rows.append(features)
        targets.append(source[prompt, direction])
    design = np.vstack(rows)
    response = np.asarray(targets, dtype=np.float64)
    penalty = np.ones(design.shape[1], dtype=np.float64)
    penalty[0] = 0.0
    gram = design.T @ design + float(ridge_lambda) * np.diag(penalty)
    coefficient = np.linalg.solve(gram, design.T @ response)
    offset = coefficient[0]
    prompt_effect = coefficient[1 : 1 + n_prompts]
    direction_effect = coefficient[1 + n_prompts :]
    return offset + prompt_effect[:, None] + direction_effect[None, :]


def predict_imputed_svd(
    matrix: np.ndarray,
    train: np.ndarray,
    *,
    rank: int,
    **_ignored,
) -> np.ndarray:
    """Closed form on the training-mean-imputed matrix. Not used for the label."""
    source = np.asarray(matrix, dtype=np.float64)
    center = float(np.mean(_observed(source, train)))
    filled = np.where(train, source, center)
    left, singular, right = np.linalg.svd(filled - center, full_matrices=False)
    kept = min(int(rank), singular.size)
    reconstruction = (left[:, :kept] * singular[:kept]) @ right[:kept, :]
    return center + reconstruction


def predict_ridge_als(
    matrix: np.ndarray,
    train: np.ndarray,
    *,
    rank: int,
    ridge_lambda: float,
    als_iterations: int,
    als_tolerance: float,
    unseen_prompt_factor: str,
    **_ignored,
) -> np.ndarray:
    """Ridge alternating least squares for mu + U V^T.

    Missing entries are initialized at the training mean and then ignored.
    A prompt with no training cells receives the mean training prompt factor
    after fitting, which is the frozen transfer rule.
    """
    source = np.asarray(matrix, dtype=np.float64)
    n_prompts, n_directions = source.shape
    center = float(np.mean(_observed(source, train)))
    filled = np.where(train, source, center)
    left, singular, right = np.linalg.svd(filled - center, full_matrices=False)
    kept = min(int(rank), int(singular.size))
    if kept < 1:
        return np.full(source.shape, center, dtype=np.float64)
    scales = np.sqrt(np.maximum(singular[:kept], 0.0))
    prompt_factors = left[:, :kept] * scales
    direction_factors = right[:kept, :].T * scales
    identity = np.eye(kept, dtype=np.float64)
    previous = None
    for _ in range(int(als_iterations)):
        updated_prompts = np.zeros_like(prompt_factors)
        for prompt in range(n_prompts):
            observed = np.where(train[prompt])[0]
            if observed.size == 0:
                continue
            design = direction_factors[observed]
            target = source[prompt, observed] - center
            updated_prompts[prompt] = np.linalg.solve(
                design.T @ design + float(ridge_lambda) * identity,
                design.T @ target,
            )
        prompt_factors = updated_prompts
        updated_directions = np.zeros_like(direction_factors)
        for direction in range(n_directions):
            observed = np.where(train[:, direction])[0]
            if observed.size == 0:
                continue
            design = prompt_factors[observed]
            target = source[observed, direction] - center
            updated_directions[direction] = np.linalg.solve(
                design.T @ design + float(ridge_lambda) * identity,
                design.T @ target,
            )
        direction_factors = updated_directions
        reconstruction = prompt_factors @ direction_factors.T
        center = float(np.mean(source[train] - reconstruction[train]))
        loss = float(np.sum((source[train] - center - reconstruction[train]) ** 2))
        if previous is not None and abs(previous - loss) <= float(als_tolerance):
            break
        previous = loss
    if unseen_prompt_factor != "mean_of_training_prompt_factors":
        raise ValueError("Unknown unseen-prompt rule.")
    missing_prompts = np.where(~np.any(train, axis=1))[0]
    seen_prompts = np.where(np.any(train, axis=1))[0]
    if missing_prompts.size and seen_prompts.size:
        prompt_factors[missing_prompts] = np.mean(prompt_factors[seen_prompts], axis=0)
    return center + prompt_factors @ direction_factors.T


PREDICTORS = {
    "B0_global_mean": predict_global_mean,
    "B1_prompt_mean": predict_prompt_mean,
    "B2_direction_mean": predict_direction_mean,
    "B3_additive": predict_additive,
    "B4_rank1": predict_ridge_als,
    "B5_rank2": predict_ridge_als,
    "B6_rank3": predict_ridge_als,
}

RANKS = {"B4_rank1": 1, "B5_rank2": 2, "B6_rank3": 3}
