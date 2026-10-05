"""Frozen state, regime, and action labels for M21.5-env-v1.0."""

from __future__ import annotations

BENCHMARK_VERSION = "M21.5-env-v1.0"

STATES = ("HIGH", "MEDIUM", "LOW")
TASK_REGIMES = ("REASONING", "TOOL", "BALANCED")
ACTIONS = ("DIRECT", "EXTENDED_REASONING", "TOOL_ASSISTED")

PRIMARY_SEEDS = (11, 23, 47, 71, 89, 101, 137, 163, 191, 223)
N_EPISODES = 100
