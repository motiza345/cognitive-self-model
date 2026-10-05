"""Frozen benchmark environment. The object is evaluator-only."""

from __future__ import annotations

import json
from pathlib import Path

from environment.outcome_model import as_float_map, optimal_action, success_probability
from environment.state import (
    ACTIONS,
    BENCHMARK_VERSION,
    N_EPISODES,
    PRIMARY_SEEDS,
    STATES,
    TASK_REGIMES,
)
from environment.task_generator import available_actions, make_task, sample_regimes, uniform_draw

GROUND_TRUTH_PATH = Path(__file__).with_name("ground_truth.json")


def load_ground_truth(path: Path | None = None) -> dict:
    source = GROUND_TRUTH_PATH if path is None else Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    _validate_ground_truth(data)
    return data


def _validate_ground_truth(data: dict) -> None:
    if data.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("ground truth benchmark version does not match M21.5-env-v1.0")
    if data.get("episodes") != N_EPISODES:
        raise ValueError("episode count is not 100")
    if tuple(data.get("states", [])) != STATES:
        raise ValueError("state space does not match the frozen list")
    if tuple(data.get("task_regimes", [])) != TASK_REGIMES:
        raise ValueError("task regimes do not match the frozen list")
    if tuple(data.get("actions", [])) != ACTIONS:
        raise ValueError("action space does not match the frozen list")
    if tuple(data.get("seeds", [])) != PRIMARY_SEEDS:
        raise ValueError("seed list does not match the frozen list")
    if data.get("reward") != {"SUCCESS": 1, "FAILURE": 0}:
        raise ValueError("reward definition does not match the frozen specification")
    if data.get("intervention_cost") != 0:
        raise ValueError("v1 has no intervention cost")
    _validate_schedule(data["schedule"], N_EPISODES)
    _validate_schedule(data["negative_control_schedule"], N_EPISODES)
    matrix = data["probability_matrix"]
    optimal = data["optimal_action"]
    seen_optimal = set()
    for state in STATES:
        for regime in TASK_REGIMES:
            row = matrix[state][regime]
            if tuple(row) != ACTIONS:
                raise ValueError(f"probability row keys differ for {state} {regime}")
            for action, value in row.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("probability entries must be numbers")
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError("probability out of range")
            chosen = optimal_action({action: float(value) for action, value in row.items()})
            if optimal[state][regime] != chosen:
                raise ValueError(f"declared optimal action disagrees with the row for {state} {regime}")
            seen_optimal.add(chosen)
    if seen_optimal != set(ACTIONS):
        raise ValueError("not every action is optimal somewhere")
    as_float_map(data["regime_probabilities"])
    as_float_map(data["ood_regime_probabilities"])


def _validate_schedule(schedule: list[dict], n_episodes: int) -> None:
    covered = []
    for block in schedule:
        if block["state"] not in STATES:
            raise ValueError("schedule contains an unknown state")
        covered.extend(range(int(block["start"]), int(block["end"]) + 1))
    if covered != list(range(1, n_episodes + 1)):
        raise ValueError("schedule does not cover episodes 1..100 exactly once")


class BenchmarkEnvironment:
    """One reproducible episode sequence. Do not pass this object to agents."""

    def __init__(
        self,
        ground_truth: dict,
        seed: int,
        schedule_name: str = "primary",
        regime_distribution: str = "primary",
    ):
        if ground_truth.get("benchmark_version") != BENCHMARK_VERSION:
            raise ValueError("refusing to build an environment for a different version")
        self.ground_truth = ground_truth
        self.seed = int(seed)
        self.schedule_name = schedule_name
        self.regime_distribution = regime_distribution
        self.n_episodes = int(ground_truth["episodes"])
        if schedule_name == "primary":
            self.schedule = ground_truth["schedule"]
        elif schedule_name == "static":
            self.schedule = ground_truth["negative_control_schedule"]
        else:
            raise ValueError(f"unknown schedule {schedule_name}")
        if regime_distribution == "primary":
            raw_probs = ground_truth["regime_probabilities"]
        elif regime_distribution == "ood":
            raw_probs = ground_truth["ood_regime_probabilities"]
        else:
            raise ValueError(f"unknown regime distribution {regime_distribution}")
        self.regime_probabilities = as_float_map(raw_probs)
        self.regimes = sample_regimes(self.seed, self.n_episodes, self.regime_probabilities)
        self._draws = {
            (episode_id, action): uniform_draw(self.seed, episode_id, index)
            for episode_id in range(1, self.n_episodes + 1)
            for index, action in enumerate(ACTIONS)
        }

    def hidden_state(self, episode_id: int) -> str:
        for block in self.schedule:
            if int(block["start"]) <= int(episode_id) <= int(block["end"]):
                return str(block["state"])
        raise ValueError(f"episode {episode_id} is outside the schedule")

    def public_observation(self, episode_id: int) -> dict:
        regime = self.regimes[episode_id - 1]
        task = make_task(episode_id, regime)
        return {
            "episode_id": int(episode_id),
            "task": task,
            "task_regime": regime,
            "available_actions": available_actions(),
        }

    def true_probabilities(self, episode_id: int) -> dict[str, float]:
        state = self.hidden_state(episode_id)
        regime = self.regimes[episode_id - 1]
        row = self.ground_truth["probability_matrix"][state][regime]
        return {action: float(row[action]) for action in ACTIONS}

    def optimal_by_regime(self, episode_id: int) -> dict[str, str]:
        state = self.hidden_state(episode_id)
        return dict(self.ground_truth["optimal_action"][state])

    def optimal_action(self, episode_id: int) -> str:
        regime = self.regimes[episode_id - 1]
        return self.optimal_by_regime(episode_id)[regime]

    def outcome(self, episode_id: int, action: str) -> int:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action}")
        probability = success_probability(
            self.ground_truth["probability_matrix"],
            self.hidden_state(episode_id),
            self.regimes[episode_id - 1],
            action,
        )
        success = self._draws[(episode_id, action)] < probability
        return 1 if success else 0

    def evaluator_event(self, episode_id: int, action: str, outcome: int) -> dict:
        return {
            "episode_id": int(episode_id),
            "hidden_self_state": self.hidden_state(episode_id),
            "true_probabilities": self.true_probabilities(episode_id),
            "optimal_action": self.optimal_action(episode_id),
            "optimal_by_regime": self.optimal_by_regime(episode_id),
            "selected_action": action,
            "outcome": int(outcome),
        }
