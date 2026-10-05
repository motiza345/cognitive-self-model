"""B3 causal self-model.

Internal labels S0, S1, and S2 are hypotheses. They are not aligned to the
evaluator's hidden-state names, and the outcome tables start from a shared
prior rather than the evaluator matrix.

Belief is one-hot on the active hypothesis. A switch keeps the old table and
moves recent observations onto a dormant hypothesis. Selection uses the same
UCB coefficient as B1 unless an ablation removes uncertainty.
"""

from __future__ import annotations

from agents.base import REGIMES, Agent, BetaTable


def confident_mismatch(abs_errors, confidences, error_threshold: float, min_confidence: float) -> bool:
    if not abs_errors:
        return False
    mean_error = sum(abs_errors) / len(abs_errors)
    mean_confidence = sum(confidences) / len(confidences)
    return mean_error > error_threshold and mean_confidence > min_confidence


class SelfModelAgent(Agent):
    name = "B3"

    def __init__(self, hparams: dict, ablation: str | None = None):
        self.ablation = ablation
        self.latent_states = list(hparams["latent_states"])
        if ablation == "no_causal_structure":
            self.latent_states = [self.latent_states[0]]
        super().__init__(hparams)
        self.window = int(hparams["b3_window"])
        self.error_threshold = float(hparams["b3_abs_error_threshold"])
        self.min_confidence = float(hparams["b3_min_confidence"])
        self.min_episodes = int(hparams["b3_min_episodes"])
        self.min_gap = int(hparams["b3_min_gap"])
        self.ucb_c = 0.0 if ablation == "no_uncertainty" else float(hparams["b3_ucb_c"])
        self.pool_actions = ablation == "no_intervention_prediction"
        self.freeze_updates = ablation == "no_update"
        self.allow_switch = ablation in (None, "no_uncertainty", "no_intervention_prediction")
        self.reset(0)

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self.tables = {
            label: BetaTable(self.tie_break, REGIMES, self.prior, 1.0) for label in self.latent_states
        }
        self.pooled = {
            label: {regime: [self.prior, self.prior] for regime in REGIMES} for label in self.latent_states
        }
        self.belief = {label: 0.0 for label in self.latent_states}
        self.map_state = self.latent_states[0]
        self.belief[self.map_state] = 1.0
        self.assigned = {label: 0 for label in self.latent_states}
        self.recent: list[dict] = []
        self.switch_count = 0
        self.mismatch_detected = False
        self.last_switch_episode = 0

    def _score_parts(self, regime: str, action: str) -> tuple[float, float]:
        if self.pool_actions:
            mean, std = self._pooled_mean_std(self.map_state, regime)
        else:
            mean, std = self.tables[self.map_state].mean_std(regime, action)
        if self.ablation == "no_uncertainty":
            return mean, 0.0
        return mean, std

    def _pooled_mean_std(self, label: str, regime: str) -> tuple[float, float]:
        successes, failures = self.pooled[label][regime]
        total = successes + failures
        mean = successes / total
        variance = (successes * failures) / (total * total * (total + 1.0))
        return mean, variance ** 0.5

    def update(self, outcome: int) -> None:
        if self._last is None or "action" not in self._last:
            raise RuntimeError("update called before an action was selected")
        outcome = 1 if int(outcome) == 1 else 0
        regime = self._last["regime"]
        action = self._last["action"]
        decision_mean = float(self._last["predictions"][action])
        self.mismatch_detected = False
        if not self.freeze_updates:
            self._apply(self.map_state, regime, action, outcome)
            self.recent.append(
                {
                    "regime": regime,
                    "action": action,
                    "outcome": outcome,
                    "abs_error": abs(outcome - decision_mean),
                    "confidence": max(decision_mean, 1.0 - decision_mean),
                }
            )
            if len(self.recent) > self.window:
                self.recent = self.recent[-self.window :]
            self._maybe_switch()
        self.episodes_seen += 1

    def _apply(self, label: str, regime: str, action: str, outcome: int) -> None:
        self.tables[label].update(regime, action, outcome)
        successes, failures = self.pooled[label][regime]
        if outcome == 1:
            successes += 1.0
        else:
            failures += 1.0
        self.pooled[label][regime] = [successes, failures]
        self.assigned[label] += 1

    def _revert(self, label: str, regime: str, action: str, outcome: int) -> None:
        self.tables[label].remove(regime, action, outcome)
        successes, failures = self.pooled[label][regime]
        if outcome == 1:
            successes -= 1.0
        else:
            failures -= 1.0
        self.pooled[label][regime] = [max(self.prior, successes), max(self.prior, failures)]
        self.assigned[label] = max(0, self.assigned[label] - 1)

    def _maybe_switch(self) -> None:
        if not self.allow_switch or len(self.latent_states) < 2:
            return
        if self.episodes_seen + 1 < self.min_episodes:
            return
        if (self.episodes_seen + 1) - self.last_switch_episode < self.min_gap:
            return
        if len(self.recent) < self.window:
            return
        if not confident_mismatch(
            [row["abs_error"] for row in self.recent],
            [row["confidence"] for row in self.recent],
            self.error_threshold,
            self.min_confidence,
        ):
            return
        target = self._choose_target()
        moved = list(self.recent)
        for row in moved:
            self._revert(self.map_state, row["regime"], row["action"], row["outcome"])
        self.map_state = target
        self.belief = {label: 0.0 for label in self.latent_states}
        self.belief[target] = 1.0
        for row in moved:
            self._apply(target, row["regime"], row["action"], row["outcome"])
        self.recent = []
        self.switch_count += 1
        self.mismatch_detected = True
        self.last_switch_episode = self.episodes_seen + 1

    def _choose_target(self) -> str:
        dormant = [label for label in self.latent_states if label != self.map_state]
        unused = [label for label in dormant if self.assigned[label] == 0]
        if unused:
            return unused[0]
        best_label = dormant[0]
        best_score = None
        for label in dormant:
            score = 0.0
            for row in self.recent:
                mean, _std = self.tables[label].mean_std(row["regime"], row["action"])
                score += mean if row["outcome"] == 1 else 1.0 - mean
            if best_score is None or score > best_score:
                best_label = label
                best_score = score
        return best_label

    def get_state(self) -> dict:
        return {
            "agent": self.name,
            "ablation": self.ablation,
            "episodes_seen": self.episodes_seen,
            "map_state": self.map_state,
            "belief": dict(self.belief),
            "mismatch_detected": self.mismatch_detected,
            "switch_count": self.switch_count,
        }
