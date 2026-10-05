"""B1 confidence baseline: the same table as B0, with UCB selection.

There is no latent self-state and no self-transition model.
"""

from __future__ import annotations

from agents.base import Agent


class ConfidenceAgent(Agent):
    name = "B1"

    def __init__(self, hparams: dict):
        super().__init__(hparams)
        self.ucb_c = float(hparams["b1_ucb_c"])
        self.decay = 1.0
        self.reset(0)
