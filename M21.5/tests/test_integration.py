from agents import ABLATION_NAMES, PRIMARY_AGENTS, build_agent
from environment.env import BenchmarkEnvironment
from evaluation.experiment import run_agent
from evaluation.scoring import score_bundle


def test_pipeline_separates_prediction_from_outcome(ground_truth, hparams):
    env = BenchmarkEnvironment(ground_truth, 11)
    seen = {}
    for name in PRIMARY_AGENTS:
        result = run_agent(build_agent(name, hparams), env, max_episodes=5)
        assert result["leakage"] == []
        assert [event["episode_id"] for event in result["agent_events"]] == [1, 2, 3, 4, 5]
        assert len(result["eval_events"]) == 5
        seen[name] = [event["outcome"] for event in result["agent_events"]]
        for agent_event, eval_event in zip(result["agent_events"], result["eval_events"]):
            assert agent_event["outcome"] == eval_event["outcome"]
            assert "hidden_self_state" not in agent_event
            assert "hidden_self_state" in eval_event
    again = run_agent(build_agent("B0", hparams), BenchmarkEnvironment(ground_truth, 11), max_episodes=5)
    assert [event["outcome"] for event in again["agent_events"]] == seen["B0"]
    assert [event["selected_action"] for event in again["agent_events"]] == [
        event["selected_action"]
        for event in run_agent(build_agent("B0", hparams), BenchmarkEnvironment(ground_truth, 11), max_episodes=5)[
            "agent_events"
        ]
    ]


def test_full_seed_completes_without_mixing_logs(ground_truth, hparams, package_dir):
    del package_dir
    env = BenchmarkEnvironment(ground_truth, 11)
    choices = hparams
    for name in PRIMARY_AGENTS + ABLATION_NAMES:
        fresh = BenchmarkEnvironment(ground_truth, 11)
        result = run_agent(build_agent(name, choices), fresh)
        assert result["leakage"] == []
        assert len(result["agent_events"]) == 100
        scored = score_bundle(
            {
                "agent_events": result["agent_events"],
                "eval_events": result["eval_events"],
                "states": result["states"],
            },
            choices,
        )
        assert scored["transitions"] == ["HIGH->LOW", "LOW->MEDIUM"]
        assert 0.0 <= scored["utility"] <= 1.0
    assert env.n_episodes == 100
