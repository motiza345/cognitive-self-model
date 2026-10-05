"""B0 raw baseline: one regime-action table, greedy mean, no latent state."""

from __future__ import annotations

from agents.base import Agent


class RawAgent(Agent):
    name = "B0"

    def __init__(self, hparams: dict):
        super().__init__(hparams)
        self.ucb_c = 0.0
        self.decay = 1.0
        self.reset(0)
