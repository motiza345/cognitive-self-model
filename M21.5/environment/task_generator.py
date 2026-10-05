"""Observable task generation. Task difficulty does not depend on hidden state."""

from __future__ import annotations

import numpy as np

from environment.state import ACTIONS, TASK_REGIMES


def sample_regimes(seed: int, n_episodes: int, probabilities: dict[str, float]) -> list[str]:
    order = list(TASK_REGIMES)
    probs = np.array([probabilities[name] for name in order], dtype=float)
    # Equal thirds are not exact in binary. Renormalizing preserves the
    # specified distribution up to that float error.
    probs = probs / probs.sum()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 17]))
    drawn = rng.choice(order, size=int(n_episodes), p=probs)
    return [str(item) for item in drawn]


def make_task(episode_id: int, regime: str) -> dict:
    if regime not in TASK_REGIMES:
        raise ValueError(f"unknown task regime {regime}")
    return {
        "task_id": f"m21.5-{episode_id:03d}",
        "task_text": (
            f"Complete benchmark item {episode_id:03d} in the {regime} regime."
        ),
        "task_regime": regime,
        "difficulty": 1,
    }


def available_actions() -> list[str]:
    return list(ACTIONS)


def uniform_draw(seed: int, episode_id: int, action_index: int) -> float:
    """Common random number for one potential outcome.

    Indexed by seed, episode, and action, so the draw does not depend on
    which action an agent selects.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(episode_id), int(action_index), 29])
    )
    return float(rng.random())
