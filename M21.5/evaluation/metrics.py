"""Scoring functions. Hidden-state fields are read only from evaluator events."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

PHASE_REGIMES = ("REASONING", "TOOL", "BALANCED")


def mean_utility(agent_events: list[dict]) -> float:
    if not agent_events:
        raise ValueError("utility requires episodes")
    return sum(int(event["outcome"]) for event in agent_events) / len(agent_events)


def slice_utility(agent_events: list[dict], start: int, end: int) -> float:
    chosen = [event for event in agent_events if start <= int(event["episode_id"]) <= end]
    if not chosen:
        raise ValueError("utility slice is empty")
    return mean_utility(chosen)


def mean_regret(agent_events: list[dict], eval_events: list[dict]) -> float:
    by_id = {int(event["episode_id"]): event for event in eval_events}
    total = 0.0
    for event in agent_events:
        truth = by_id[int(event["episode_id"])]
        probabilities = truth["true_probabilities"]
        optimal = float(probabilities[truth["optimal_action"]])
        selected = float(probabilities[event["selected_action"]])
        total += optimal - selected
    return total / len(agent_events)


def prediction_pairs(agent_events: list[dict]) -> tuple[list[float], list[int]]:
    probabilities = []
    outcomes = []
    for event in agent_events:
        probabilities.append(float(event["predictions"][event["selected_action"]]))
        outcomes.append(int(event["outcome"]))
    return probabilities, outcomes


def mean_absolute_error(probabilities, outcomes) -> float:
    return sum(abs(p - y) for p, y in zip(probabilities, outcomes)) / len(probabilities)


def brier_score(probabilities, outcomes) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def log_loss(probabilities, outcomes, clip: float) -> float:
    total = 0.0
    for probability, outcome in zip(probabilities, outcomes):
        clipped = min(1.0 - clip, max(clip, float(probability)))
        if int(outcome) == 1:
            total += -math.log(clipped)
        else:
            total += -math.log(1.0 - clipped)
    return total / len(probabilities)


def expected_calibration_error(probabilities, outcomes, n_bins: int) -> float:
    probs = np.asarray(probabilities, dtype=float)
    ys = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = 0.0
    for index in range(int(n_bins)):
        lower = edges[index]
        upper = edges[index + 1]
        if index == int(n_bins) - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / len(probs)) * abs(float(ys[mask].mean()) - float(probs[mask].mean()))
    return float(total)


def prediction_metrics(agent_events: list[dict], clip: float, n_bins: int) -> dict:
    probabilities, outcomes = prediction_pairs(agent_events)
    return {
        "mae": mean_absolute_error(probabilities, outcomes),
        "brier": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes, clip),
        "calibration_error": expected_calibration_error(probabilities, outcomes, n_bins),
    }


def hidden_phases(eval_events: list[dict]) -> list[list[dict]]:
    phases: list[list[dict]] = []
    current: list[dict] = []
    current_state = None
    for event in sorted(eval_events, key=lambda item: int(item["episode_id"])):
        state = event["hidden_self_state"]
        if current_state is None or state == current_state:
            current.append(event)
            current_state = state
        else:
            phases.append(current)
            current = [event]
            current_state = state
    if current:
        phases.append(current)
    return phases


def transition_points(eval_events: list[dict]) -> list[dict]:
    points = []
    previous = None
    for event in sorted(eval_events, key=lambda item: int(item["episode_id"])):
        state = event["hidden_self_state"]
        if previous is not None and state != previous:
            points.append(
                {
                    "episode_id": int(event["episode_id"]),
                    "from_state": previous,
                    "to_state": state,
                    "label": f"{previous}->{state}",
                }
            )
        previous = state
    return points


def _unique_mode(actions: list[str]):
    if not actions:
        return None
    counts = Counter(actions)
    best = max(counts.values())
    winners = [action for action, count in counts.items() if count == best]
    if len(winners) != 1:
        return None
    return winners[0]


def decision_separation(agent_events: list[dict], eval_events: list[dict]) -> dict:
    by_agent = {int(event["episode_id"]): event for event in agent_events}
    phases = hidden_phases(eval_events)
    opportunities = 0
    correct = 0
    rows = []
    for left, right in zip(phases, phases[1:]):
        for regime in PHASE_REGIMES:
            before = left[0]["optimal_by_regime"][regime]
            after = right[0]["optimal_by_regime"][regime]
            if before == after:
                continue
            opportunities += 1
            left_ids = [int(event["episode_id"]) for event in left]
            right_ids = [int(event["episode_id"]) for event in right]
            left_actions = [
                by_agent[episode_id]["selected_action"]
                for episode_id in left_ids
                if by_agent[episode_id]["task_regime"] == regime
            ]
            right_actions = [
                by_agent[episode_id]["selected_action"]
                for episode_id in right_ids
                if by_agent[episode_id]["task_regime"] == regime
            ]
            changed = _unique_mode(left_actions) == before and _unique_mode(right_actions) == after
            if changed:
                correct += 1
            rows.append(
                {
                    "regime": regime,
                    "from_action": before,
                    "to_action": after,
                    "correct": changed,
                }
            )
    return {
        "opportunities": opportunities,
        "correct_changes": correct,
        "dsr": (correct / opportunities) if opportunities else None,
        "rows": rows,
    }


def window_means(agent_events: list[dict], center: int, window: int, last_episode: int) -> tuple[float, float] | None:
    by_id = {int(event["episode_id"]): int(event["outcome"]) for event in agent_events}
    before_ids = [episode for episode in range(center - window, center) if 1 <= episode <= last_episode]
    after_ids = [episode for episode in range(center, center + window) if 1 <= episode <= last_episode]
    if len(before_ids) != window or len(after_ids) != window:
        return None
    before = sum(by_id[episode] for episode in before_ids) / window
    after = sum(by_id[episode] for episode in after_ids) / window
    return before, after


def transition_update_gains(agent_events: list[dict], eval_events: list[dict], window: int) -> dict:
    last_episode = max(int(event["episode_id"]) for event in agent_events)
    gains = {}
    for point in transition_points(eval_events):
        means = window_means(agent_events, int(point["episode_id"]), window, last_episode)
        gains[point["label"]] = None if means is None else means[1] - means[0]
    return gains


def detection_update_gains(agent_events, eval_events, states, window: int) -> dict:
    scored = mismatch_detection(states, eval_events)
    last_episode = max(int(event["episode_id"]) for event in agent_events)
    gains = {}
    for item in scored["transitions"]:
        detected = item["detection_episode"]
        if detected is None:
            gains[item["label"]] = None
            continue
        means = window_means(agent_events, int(detected), window, last_episode)
        gains[item["label"]] = None if means is None else means[1] - means[0]
    return gains


def mismatch_detection(states: list[dict], eval_events: list[dict]) -> dict:
    points = transition_points(eval_events)
    detected = [
        index + 1
        for index, state in enumerate(states)
        if bool(state.get("mismatch_detected"))
    ]
    boundaries = [int(point["episode_id"]) for point in points]
    boundaries.append(max(int(event["episode_id"]) for event in eval_events) + 1)
    true_positive = 0
    false_positive = 0
    false_negative = 0
    delays = []
    details = []
    consumed = set()
    for point, end in zip(points, boundaries[1:]):
        start = int(point["episode_id"])
        hits = [episode for episode in detected if start <= episode < end]
        if hits:
            true_positive += 1
            false_positive += len(hits) - 1
            consumed.update(hits)
            delay = hits[0] - start
        else:
            false_negative += 1
            delay = None
        delays.append(delay)
        details.append(
            {
                "label": point["label"],
                "transition_episode": start,
                "detection_episode": None if delay is None else start + delay,
                "delay": delay,
            }
        )
    for episode in detected:
        if episode not in consumed:
            false_positive += 1
    precision = None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    recall = None if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    finite_delays = [delay for delay in delays if delay is not None]
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "delays": delays,
        "mean_delay": None if not finite_delays else sum(finite_delays) / len(finite_delays),
        "transitions": details,
    }
