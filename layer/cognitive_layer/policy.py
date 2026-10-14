"""Expected-utility decision policy with logged exploration.

Origins: real_claim_self_model.decide (decision reads only the self-claim)
and counterfactual_changes_decision (S1 check). v0.2 replaces the fixed
status->action table with expected utility over a calibrated probability,
because in M34a v2 the instance-level own-answer probability carried most of
the decision value (+0.089) and the per-level track record little (+0.025).

Where the probability comes from (the only epistemic input):
  SUPPORTED             -> calibrated(confidence) of the claim used
  CONTRADICTED          -> base rate of the claim used
  UNCERTAIN / no claim  -> base rate, and only outcome-revealing actions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .core import OBSERVING_ACTIONS, Action, Status


@dataclass(frozen=True)
class UtilityTable:
    """Default values are the M34a utilities: answer +1 / -1, abstain 0, verify 0.7."""

    act_correct: float = 1.0
    act_wrong: float = -1.0
    verify: float = 0.7
    abstain: float = 0.0
    delegate: float | None = None      # None: action disabled

    def expected(self, p: float) -> dict[Action, float]:
        eu = {
            Action.ACT: p * self.act_correct + (1.0 - p) * self.act_wrong,
            Action.VERIFY: self.verify,
            Action.ABSTAIN: self.abstain,
        }
        if self.delegate is not None:
            eu[Action.DELEGATE] = self.delegate
        return eu

    def realized(self, action: Action, correct: bool) -> float:
        if action == Action.ACT:
            return self.act_correct if correct else self.act_wrong
        if action == Action.VERIFY:
            return self.verify
        if action == Action.ABSTAIN:
            return self.abstain
        if action == Action.DELEGATE and self.delegate is not None:
            return self.delegate
        raise ValueError("action not in utility table")


_ORDER = [Action.VERIFY, Action.ABSTAIN, Action.DELEGATE, Action.ACT]  # conservative tie-break: ACT only if strictly better


@dataclass
class ExpectedUtilityPolicy:
    utility: UtilityTable = field(default_factory=UtilityTable)
    epsilon: float = 0.05              # exploration rate for non-observing greedy actions
    explore_action: Action = Action.VERIFY
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if self.explore_action not in OBSERVING_ACTIONS:
            raise ValueError("exploration must reveal the outcome")
        self._rng = random.Random(self.seed)

    def greedy(self, p: float, learning: bool) -> tuple[Action, dict[Action, float]]:
        eu = self.utility.expected(p)
        allowed = [a for a in _ORDER if a in eu and (not learning or a in OBSERVING_ACTIONS)]
        best = max(allowed, key=lambda a: (eu[a], -_ORDER.index(a)))
        return best, eu

    def choose(self, p: float, learning: bool, frozen: bool) -> tuple[Action, Action, dict[Action, float], bool, float]:
        """Returns (action, greedy_action, eu, explored, propensity)."""
        greedy, eu = self.greedy(p, learning)
        if frozen or greedy in OBSERVING_ACTIONS or self.epsilon == 0.0:
            return greedy, greedy, eu, False, 1.0
        if self._rng.random() < self.epsilon:
            return self.explore_action, greedy, eu, True, self.epsilon
        return greedy, greedy, eu, False, 1.0 - self.epsilon

    def counterfactual_changes_decision(self, p_supported: float, p_fallback: float) -> bool:
        """S1 check: does the self-claim's status change behaviour for this instance?"""
        return self.greedy(p_supported, False)[0] != self.greedy(p_fallback, False)[0]

    @staticmethod
    def source_for(status: Status) -> str:
        return {Status.SUPPORTED: "calibrated", Status.CONTRADICTED: "base_rate"}.get(status, "base_rate")
