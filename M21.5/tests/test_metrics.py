from evaluation.metrics import (
    brier_score,
    decision_separation,
    expected_calibration_error,
    mean_absolute_error,
    mean_regret,
    mean_utility,
    mismatch_detection,
)
from evaluation.report import decide_claim
from evaluation.statistics import bootstrap_indices, strictly_positive, summarize_paired


def _agent_event(episode, regime, action, outcome, probability=0.5):
    return {
        "episode_id": episode,
        "task_regime": regime,
        "selected_action": action,
        "outcome": outcome,
        "predictions": {
            "DIRECT": probability,
            "EXTENDED_REASONING": probability,
            "TOOL_ASSISTED": probability,
        },
    }


def _eval_event(episode, state, regime, optimal, probabilities, optimal_by_regime):
    return {
        "episode_id": episode,
        "hidden_self_state": state,
        "task_regime": regime,
        "optimal_action": optimal,
        "optimal_by_regime": optimal_by_regime,
        "true_probabilities": probabilities,
        "selected_action": optimal,
        "outcome": 1,
    }


def test_known_utility_regret_and_prediction_scores():
    events = [
        _agent_event(1, "REASONING", "DIRECT", 1, 1.0),
        _agent_event(2, "REASONING", "DIRECT", 0, 0.0),
        _agent_event(3, "REASONING", "DIRECT", 1, 1.0),
        _agent_event(4, "REASONING", "DIRECT", 0, 0.0),
    ]
    assert mean_utility(events) == 0.5
    probabilities = [1.0, 0.0, 1.0, 0.0]
    outcomes = [1, 0, 1, 0]
    assert mean_absolute_error(probabilities, outcomes) == 0.0
    assert brier_score(probabilities, outcomes) == 0.0
    assert expected_calibration_error([1, 1, 1, 1], [1, 1, 1, 1], 10) == 0.0
    assert expected_calibration_error([0, 0, 0, 0], [1, 1, 1, 1], 10) == 1.0
    eval_events = [
        _eval_event(
            index,
            "HIGH",
            "REASONING",
            "EXTENDED_REASONING",
            {"DIRECT": 0.5, "EXTENDED_REASONING": 0.9, "TOOL_ASSISTED": 0.2},
            {"REASONING": "EXTENDED_REASONING", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"},
        )
        for index in range(1, 5)
    ]
    for event in eval_events:
        event["selected_action"] = "DIRECT"
    regret = mean_regret(events, eval_events)
    assert abs(regret - 0.4) < 1e-12


def test_decision_separation_counts_a_correct_change():
    optimal_high = {"REASONING": "EXTENDED_REASONING", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"}
    optimal_low = {"REASONING": "TOOL_ASSISTED", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"}
    eval_events = []
    agent_events = []
    for episode, action in ((1, "EXTENDED_REASONING"), (2, "EXTENDED_REASONING")):
        eval_events.append(
            _eval_event(episode, "HIGH", "REASONING", "EXTENDED_REASONING", {"DIRECT": 0.1, "EXTENDED_REASONING": 0.9, "TOOL_ASSISTED": 0.2}, optimal_high)
        )
        agent_events.append(_agent_event(episode, "REASONING", action, 1))
    for episode, action in ((3, "TOOL_ASSISTED"), (4, "TOOL_ASSISTED")):
        eval_events.append(
            _eval_event(episode, "LOW", "REASONING", "TOOL_ASSISTED", {"DIRECT": 0.1, "EXTENDED_REASONING": 0.2, "TOOL_ASSISTED": 0.8}, optimal_low)
        )
        agent_events.append(_agent_event(episode, "REASONING", action, 1))
    scored = decision_separation(agent_events, eval_events)
    assert scored["opportunities"] == 1
    assert scored["correct_changes"] == 1
    assert scored["dsr"] == 1.0


def test_mismatch_delay_and_bootstrap_sign():
    states = [{"mismatch_detected": False}, {"mismatch_detected": False}, {"mismatch_detected": True}]
    eval_events = [
        {"episode_id": 1, "hidden_self_state": "HIGH"},
        {"episode_id": 2, "hidden_self_state": "LOW"},
        {"episode_id": 3, "hidden_self_state": "LOW"},
    ]
    scored = mismatch_detection(states, eval_events)
    assert scored["true_positive"] == 1
    assert scored["delays"] == [1]
    indices = bootstrap_indices(4, 200, 2150)
    positive = summarize_paired([1, 1, 1, 1], indices)
    mixed = summarize_paired([1, -1, 1, -1], indices)
    assert strictly_positive(positive)
    assert not strictly_positive(mixed)


def _summary(low, mean, n=10, dsr=0.5, gain=0.1, post=0.1, brier=0.2):
    per_agent = {}
    for name in ("B0", "B1", "B2", "B3"):
        per_agent[name] = {
            "dsr": {"dsr": 0.1 if name != "B3" else dsr},
            "prediction": {"brier": 0.4 if name != "B3" else brier},
            "mean_detection_update_gain": {"HIGH->LOW": gain, "LOW->MEDIUM": gain},
        }
    return {
        "per_agent": per_agent,
        "post_transition_delta": post,
        "comparisons": {
            "B3-B0": {"mean": mean, "low": low, "high": mean + 0.1, "n": n},
            "B3-B1": {"mean": mean, "low": low, "high": mean + 0.1, "n": n},
            "B3-B2": {"mean": mean, "low": low, "high": mean + 0.1, "n": n},
            "B3-BestSimple": {"mean": mean, "low": low, "high": mean + 0.1, "n": n},
        },
    }


def test_claim_accepts_only_the_full_predeclared_list():
    integrity = {
        "leakage_passed": True,
        "major_compute_confound": False,
        "raw_hashes_match": True,
        "source_hashes_match": True,
        "fingerprints_match": True,
    }
    primary = _summary(0.02, 0.08)
    static_summary = _summary(-0.05, 0.0)
    ood = _summary(0.01, 0.04)
    ablations = {
        "B3-no-causal-structure": {"full_minus_ablation": {"mean": 0.05, "low": 0.01, "high": 0.1, "n": 10}}
    }
    accepted = decide_claim(primary, static_summary, ood, ablations, integrity)
    assert accepted["verdict"] == "M21.5_V1_CLAIM_ACCEPTED"
    missed = decide_claim(_summary(-0.01, 0.0), static_summary, ood, ablations, integrity)
    assert missed["verdict"] == "M21.5_V1_CLAIM_NOT_ACCEPTED"
    leaked = dict(integrity)
    leaked["leakage_passed"] = False
    invalid = decide_claim(primary, static_summary, ood, ablations, leaked)
    assert invalid["verdict"] == "BENCHMARK_INVALID"
    assert invalid["interpretation"] == ["Leakage audit failed. No scientific interpretation is offered."]
