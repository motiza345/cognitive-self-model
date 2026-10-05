"""Shared agent contract. No evaluator-only information is accepted."""

from __future__ import annotations

import math

from observation_boundary import assert_agent_visible

REGIMES = ("REASONING", "TOOL", "BALANCED")
DEFAULT_ACTIONS = ("DIRECT", "EXTENDED_REASONING", "TOOL_ASSISTED")


class BetaTable:
    def __init__(self, actions, regimes, prior: float, decay: float = 1.0):
        self.actions = list(actions)
        self.regimes = list(regimes)
        self.prior = float(prior)
        self.decay = float(decay)
        self.counts = {
            (regime, action): [self.prior, self.prior]
            for regime in self.regimes
            for action in self.actions
        }

    def decay_all(self) -> None:
        if self.decay == 1.0:
            return
        prior = self.prior
        gamma = self.decay
        for key, (successes, failures) in self.counts.items():
            self.counts[key] = [
                prior + (successes - prior) * gamma,
                prior + (failures - prior) * gamma,
            ]

    def update(self, regime: str, action: str, outcome: int) -> None:
        successes, failures = self.counts[(regime, action)]
        if int(outcome) == 1:
            successes += 1.0
        else:
            failures += 1.0
        self.counts[(regime, action)] = [successes, failures]

    def remove(self, regime: str, action: str, outcome: int) -> None:
        successes, failures = self.counts[(regime, action)]
        if int(outcome) == 1:
            successes -= 1.0
        else:
            failures -= 1.0
        self.counts[(regime, action)] = [
            max(self.prior, successes),
            max(self.prior, failures),
        ]

    def mean_std(self, regime: str, action: str) -> tuple[float, float]:
        successes, failures = self.counts[(regime, action)]
        total = successes + failures
        mean = successes / total
        variance = (successes * failures) / (total * total * (total + 1.0))
        return mean, math.sqrt(variance)


class Agent:
    name = "agent"

    def __init__(self, hparams: dict):
        self.hparams = dict(hparams)
        self.tie_break = list(hparams["tie_break"])
        self.prior = float(hparams["beta_prior"])
        self.ucb_c = 0.0
        self.decay = 1.0
        self.reset(0)

    def reset(self, seed: int) -> None:
        # The environment seed is intentionally not stored.
        del seed
        self.episodes_seen = 0
        self._current = None
        self._last = None
        self.table = BetaTable(self.tie_break, REGIMES, self.prior, self.decay)

    def observe_episode(self, episode: dict) -> None:
        assert_agent_visible(episode)
        if "outcome" in episode:
            raise ValueError("observation includes an outcome before the decision")
        self._current = episode

    def predict(self, task, available_actions, history) -> dict:
        assert_agent_visible(task)
        assert_agent_visible(history)
        self._check_history(history)
        if list(available_actions) != self.tie_break:
            raise ValueError("available actions differ from the frozen action list")
        regime = task["task_regime"]
        predictions = {}
        uncertainty = {}
        for action in self.tie_break:
            mean, std = self._score_parts(regime, action)
            predictions[action] = mean
            uncertainty[action] = std
        self._last = {"regime": regime, "predictions": predictions, "uncertainty": uncertainty}
        return {"predictions": predictions, "uncertainty": uncertainty}

    def select_action(self, predictions: dict) -> str:
        assert_agent_visible(predictions)
        best_action = None
        best_score = None
        for action in self.tie_break:
            score = float(predictions["predictions"][action]) + self.ucb_c * float(
                predictions["uncertainty"][action]
            )
            if best_score is None or score > best_score:
                best_action = action
                best_score = score
        if self._last is None:
            raise RuntimeError("select_action called before predict")
        self._last["action"] = best_action
        self._last["decision_mean"] = float(predictions["predictions"][best_action])
        return best_action

    def update(self, outcome: int) -> None:
        if self._last is None or "action" not in self._last:
            raise RuntimeError("update called before an action was selected")
        outcome = 1 if int(outcome) == 1 else 0
        self.table.decay_all()
        self.table.update(self._last["regime"], self._last["action"], outcome)
        self.episodes_seen += 1
        self._after_update(outcome)

    def get_state(self) -> dict:
        return {
            "agent": self.name,
            "episodes_seen": self.episodes_seen,
            "mismatch_detected": False,
        }

    def _score_parts(self, regime: str, action: str) -> tuple[float, float]:
        return self.table.mean_std(regime, action)

    def _after_update(self, outcome: int) -> None:
        del outcome

    def _check_history(self, history) -> None:
        if self._current is None:
            raise RuntimeError("predict called before observe_episode")
        current = int(self._current["episode_id"])
        for event in history:
            if int(event["episode_id"]) >= current:
                raise ValueError("history contains the current or a future episode")
            if "outcome" not in event:
                raise ValueError("history episode has no outcome")
        if len(history) != self.episodes_seen:
            raise ValueError("history is not the complete permitted past")
