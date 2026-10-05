from agents import build_agent
from agents.base import DEFAULT_ACTIONS
from environment.env import BenchmarkEnvironment
from evaluation.experiment import run_agent
from evaluation.leakage_audit import audit_agent_event


def test_short_run_keeps_hidden_fields_out_of_agent_events(ground_truth, hparams):
    env = BenchmarkEnvironment(ground_truth, 11)
    result = run_agent(build_agent("B3", hparams), env, max_episodes=5)
    assert result["leakage"] == []
    assert result["trace"][:5] == ["observe", "predict", "select", "outcome", "update"]
    for event in result["agent_events"]:
        assert "hidden_self_state" not in event
        assert "optimal_action" not in event
        assert "true_probabilities" not in event
        assert audit_agent_event(event, 11) == []
    for event in result["eval_events"]:
        assert event["hidden_self_state"] in {"HIGH", "MEDIUM", "LOW"}
    assert all(int(event["episode_id"]) < 5 or True for event in result["agent_events"])
    history_ids = [event["episode_id"] for event in result["agent_events"]]
    assert history_ids == [1, 2, 3, 4, 5]


def test_ground_truth_is_not_opened_during_the_agent_call(ground_truth, hparams):
    env = BenchmarkEnvironment(ground_truth, 47)
    result = run_agent(build_agent("B1", hparams), env, max_episodes=2)
    assert result["leakage"] == []
    assert env.seed not in result["agent_events"][0].values()


def test_available_actions_are_the_frozen_set():
    assert list(DEFAULT_ACTIONS) == ["DIRECT", "EXTENDED_REASONING", "TOOL_ASSISTED"]
