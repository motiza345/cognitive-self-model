"""B2 memory baseline: recency-weighted table plus a capped reflection log.

The reflection log is not a latent causal state. Selection uses only the
decayed regime-action means.
"""

from __future__ import annotations

from agents.base import Agent


class MemoryAgent(Agent):
    name = "B2"

    def __init__(self, hparams: dict):
        super().__init__(hparams)
        self.ucb_c = 0.0
        self.decay = float(hparams["b2_decay"])
        self.reflection_cap = int(hparams["b2_reflection_cap"])
        self.reflections: list[str] = []
        self.reset(0)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.table.decay = self.decay
        self.reflections = []

    def _after_update(self, outcome: int) -> None:
        regime = self._last["regime"]
        action = self._last["action"]
        note = f"regime {regime} action {action} outcome {int(outcome)}"
        self.reflections.append(note)
        if len(self.reflections) > self.reflection_cap:
            self.reflections = self.reflections[-self.reflection_cap :]

    def get_state(self) -> dict:
        state = super().get_state()
        state["reflections"] = list(self.reflections)
        return state
