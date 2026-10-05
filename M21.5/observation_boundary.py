"""Agent-visible payload checks. This module is not an agent."""

from __future__ import annotations

BANNED_KEYS = frozenset(
    {
        "hidden_self_state",
        "true_probabilities",
        "optimal_action",
        "optimal_by_regime",
        "future_state",
        "future_outcome",
        "transition_schedule",
        "environment_seed",
        "ground_truth",
        "probability_matrix",
        "causal_graph",
        "schedule",
        "self_state",
    }
)

# Hidden-state names. They are evaluator vocabulary, not agent observations.
HIDDEN_STATE_NAMES = frozenset({"HIGH", "MEDIUM", "LOW"})


def assert_agent_visible(payload) -> None:
    """Raise if a payload carries evaluator-only information."""
    _walk(payload)


def _walk(payload) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in BANNED_KEYS:
                raise ValueError(f"agent payload contains banned key {key}")
            _walk(value)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            _walk(value)
        return
    if isinstance(payload, str) and payload in HIDDEN_STATE_NAMES:
        raise ValueError("agent payload contains a hidden-state name")
