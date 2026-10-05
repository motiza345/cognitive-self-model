"""Aggregate sealed logs. This module does not call the agents or the model."""

from __future__ import annotations

from evaluation.metrics import (
    decision_separation,
    detection_update_gains,
    mean_regret,
    mean_utility,
    mismatch_detection,
    prediction_metrics,
    slice_utility,
    transition_points,
    transition_update_gains,
)
from evaluation.statistics import bootstrap_indices, summarize_paired

PRIMARY_AGENTS = ("B0", "B1", "B2", "B3")
EXPECTED_DYNAMIC_TRANSITIONS = ("HIGH->LOW", "LOW->MEDIUM")


def score_bundle(bundle: dict, choices: dict) -> dict:
    events = bundle["agent_events"]
    eval_events = bundle["eval_events"]
    states = bundle["states"]
    _reject_disagreement(events, eval_events)
    window = int(choices["update_gain_window"])
    post_start, post_end = choices["post_transition_episodes"]
    separation = decision_separation(events, eval_events)
    return {
        "utility": mean_utility(events),
        "post_utility": slice_utility(events, int(post_start), int(post_end)),
        "regret": mean_regret(events, eval_events),
        "prediction": prediction_metrics(
            events, float(choices["log_loss_clip"]), int(choices["ece_bins"])
        ),
        "decision_separation": {
            "opportunities": separation["opportunities"],
            "correct_changes": separation["correct_changes"],
            "dsr": separation["dsr"],
        },
        "transition_update_gain": transition_update_gains(events, eval_events, window),
        "detection_update_gain": detection_update_gains(events, eval_events, states, window),
        "mismatch": _compact_mismatch(mismatch_detection(states, eval_events)),
        "transitions": [point["label"] for point in transition_points(eval_events)],
    }


def score_condition(seed_rows: list[dict], choices: dict, expect_dynamic_transitions: bool) -> dict:
    if not seed_rows:
        raise RuntimeError("STOP: condition has no seeds")
    seeds = [int(row["seed"]) for row in seed_rows]
    if seeds != sorted(seeds):
        raise RuntimeError("STOP: seeds are not in increasing order")
    agent_names = tuple(seed_rows[0]["agents"])
    for row in seed_rows:
        if tuple(row["agents"]) != agent_names:
            raise RuntimeError("STOP: agents differ across seeds")
    per_seed = []
    for row in seed_rows:
        scored = {
            name: score_bundle(bundle, choices) for name, bundle in row["agents"].items()
        }
        if expect_dynamic_transitions:
            for name, record in scored.items():
                labels = tuple(record["transitions"])
                if labels != EXPECTED_DYNAMIC_TRANSITIONS:
                    raise RuntimeError(
                        "STOP: hidden-state transitions differ from the frozen schedule "
                        f"{EXPECTED_DYNAMIC_TRANSITIONS}; found {labels} for {name}"
                    )
        per_seed.append({"seed": row["seed"], "agents": scored})
    per_agent = {
        name: _aggregate_agent([row["agents"][name] for row in per_seed]) for name in agent_names
    }
    summary = {"seeds": seeds, "agent_names": list(agent_names), "per_agent": per_agent, "per_seed": per_seed}
    if set(PRIMARY_AGENTS).issubset(agent_names):
        summary.update(_baseline_contrasts(per_seed, choices))
    return summary


def score_ablations(ablation_rows: list[dict], primary: dict, choices: dict) -> dict:
    if [row["seed"] for row in ablation_rows] != primary["seeds"]:
        raise RuntimeError("STOP: ablation seeds differ from the primary seeds")
    indices = _indices(len(primary["seeds"]), choices)
    best = primary["best_simple_utilities"]
    full = primary["per_agent"]["B3"]["utilities"]
    scored = {}
    for name in ablation_rows[0]["agents"]:
        utilities = []
        for row, full_row in zip(ablation_rows, primary["per_seed"]):
            bundle_score = score_bundle(row["agents"][name], choices)
            if tuple(bundle_score["transitions"]) != tuple(full_row["agents"]["B3"]["transitions"]):
                raise RuntimeError("STOP: ablation transitions differ from the primary run")
            utilities.append(bundle_score["utility"])
        scored[name] = {
            "mean_utility": _mean(utilities),
            "utilities": utilities,
            "minus_best_simple": summarize_paired(
                [utility - base for utility, base in zip(utilities, best)], indices
            ),
            "full_minus_ablation": summarize_paired(
                [reference - utility for reference, utility in zip(full, utilities)], indices
            ),
        }
    return scored


def _baseline_contrasts(per_seed: list[dict], choices: dict) -> dict:
    indices = _indices(len(per_seed), choices)
    series = {
        name: [row["agents"][name]["utility"] for row in per_seed] for name in PRIMARY_AGENTS
    }
    best_utilities = [
        max(series["B0"][index], series["B1"][index], series["B2"][index])
        for index in range(len(per_seed))
    ]
    deltas = [series["B3"][index] - best_utilities[index] for index in range(len(per_seed))]
    post_deltas = []
    for row in per_seed:
        best_post = max(row["agents"][name]["post_utility"] for name in ("B0", "B1", "B2"))
        post_deltas.append(row["agents"]["B3"]["post_utility"] - best_post)
    comparisons = {
        "B3-B0": summarize_paired(
            [series["B3"][i] - series["B0"][i] for i in range(len(per_seed))], indices
        ),
        "B3-B1": summarize_paired(
            [series["B3"][i] - series["B1"][i] for i in range(len(per_seed))], indices
        ),
        "B3-B2": summarize_paired(
            [series["B3"][i] - series["B2"][i] for i in range(len(per_seed))], indices
        ),
        "B3-BestSimple": summarize_paired(deltas, indices),
    }
    return {
        "best_simple_utilities": best_utilities,
        "best_simple_mean_utility": _mean(best_utilities),
        "post_transition_delta": _mean(post_deltas),
        "comparisons": comparisons,
    }


def _aggregate_agent(records: list[dict]) -> dict:
    return {
        "utilities": [record["utility"] for record in records],
        "mean_utility": _mean([record["utility"] for record in records]),
        "mean_regret": _mean([record["regret"] for record in records]),
        "prediction": {
            key: _mean([record["prediction"][key] for record in records])
            for key in ("mae", "brier", "log_loss", "calibration_error")
        },
        "dsr": _pool_dsr(records),
        "mismatch": _pool_mismatch(records),
        "mean_transition_update_gain": _mean_labeled(records, "transition_update_gain"),
        "mean_detection_update_gain": _mean_labeled(records, "detection_update_gain"),
    }


def _pool_dsr(records: list[dict]) -> dict:
    opportunities = sum(record["decision_separation"]["opportunities"] for record in records)
    correct = sum(record["decision_separation"]["correct_changes"] for record in records)
    return {
        "opportunities": opportunities,
        "correct_changes": correct,
        "dsr": None if opportunities == 0 else correct / opportunities,
    }


def _pool_mismatch(records: list[dict]) -> dict:
    true_positive = sum(record["mismatch"]["true_positive"] for record in records)
    false_positive = sum(record["mismatch"]["false_positive"] for record in records)
    false_negative = sum(record["mismatch"]["false_negative"] for record in records)
    delays = [
        delay
        for record in records
        for delay in record["mismatch"]["delays"]
        if delay is not None
    ]
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": None
        if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive),
        "recall": None
        if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative),
        "mean_delay": None if not delays else sum(delays) / len(delays),
    }


def _mean_labeled(records: list[dict], key: str) -> dict:
    labels = list(records[0][key])
    output = {}
    for label in labels:
        values = [record[key][label] for record in records]
        output[label] = None if any(value is None for value in values) else _mean(values)
    return output


def _compact_mismatch(result: dict) -> dict:
    return {
        "true_positive": result["true_positive"],
        "false_positive": result["false_positive"],
        "false_negative": result["false_negative"],
        "precision": result["precision"],
        "recall": result["recall"],
        "delays": result["delays"],
        "mean_delay": result["mean_delay"],
    }


def _indices(n_seeds: int, choices: dict):
    return bootstrap_indices(
        n_seeds, int(choices["bootstrap_replicates"]), int(choices["bootstrap_seed"])
    )


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _reject_disagreement(agent_events: list[dict], eval_events: list[dict]) -> None:
    if len(agent_events) != len(eval_events):
        raise RuntimeError("STOP: agent and evaluator logs differ in length")
    for agent_event, eval_event in zip(agent_events, eval_events):
        if int(agent_event["episode_id"]) != int(eval_event["episode_id"]):
            raise RuntimeError("STOP: agent and evaluator episode ids differ")
        if agent_event["selected_action"] != eval_event["selected_action"]:
            raise RuntimeError("STOP: agent and evaluator actions differ")
        if int(agent_event["outcome"]) != int(eval_event["outcome"]):
            raise RuntimeError("STOP: agent and evaluator outcomes differ")
        if float(eval_event["true_probabilities"][eval_event["optimal_action"]]) + 1e-12 < max(
            float(value) for value in eval_event["true_probabilities"].values()
        ):
            raise RuntimeError("STOP: logged optimal action is not maximal")
