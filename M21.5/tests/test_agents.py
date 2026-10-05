import numpy as np

from agents import PRIMARY_AGENTS, build_agent
from agents.b3_self_model import SelfModelAgent, confident_mismatch
from agents.base import DEFAULT_ACTIONS
from evaluation.leakage_audit import scan_agent_sources


def _drive(agent, outcome_fn, n=20, regime="REASONING"):
    agent.reset(0)
    history = []
    actions = []
    flags = []
    for episode in range(1, n + 1):
        task = {
            "task_id": f"t-{episode}",
            "task_text": f"Complete benchmark item {episode:03d} in the {regime} regime.",
            "task_regime": regime,
            "difficulty": 1,
        }
        public = {
            "episode_id": episode,
            "task": task,
            "task_regime": regime,
            "available_actions": list(DEFAULT_ACTIONS),
        }
        agent.observe_episode(public)
        prediction = agent.predict(task, public["available_actions"], history)
        action = agent.select_action(prediction)
        outcome = int(outcome_fn(action, episode))
        agent.update(outcome)
        history.append(
            {
                "episode_id": episode,
                "task_id": task["task_id"],
                "task_text": task["task_text"],
                "task_regime": regime,
                "available_actions": list(DEFAULT_ACTIONS),
                "predictions": prediction["predictions"],
                "uncertainties": prediction["uncertainty"],
                "selected_action": action,
                "outcome": outcome,
            }
        )
        actions.append(action)
        flags.append(bool(agent.get_state().get("mismatch_detected")))
    return actions, flags


def test_four_agents_share_the_contract(hparams):
    for name in PRIMARY_AGENTS:
        agent = build_agent(name, hparams)
        actions, _flags = _drive(agent, lambda _action, _episode: 1, n=3)
        assert actions[0] in DEFAULT_ACTIONS
        state = agent.get_state()
        assert state["agent"] == name
        assert state["episodes_seen"] == 3
        if name != "B3":
            assert "belief" not in state


def test_agent_sources_do_not_contain_the_evaluator_table(package_dir):
    assert scan_agent_sources(package_dir / "agents") == []


def test_b1_matches_the_no_causal_ablation_on_a_stationary_oracle(hparams):
    rng = np.random.default_rng(0)
    probabilities = {"DIRECT": 0.15, "EXTENDED_REASONING": 0.85, "TOOL_ASSISTED": 0.20}

    def outcome(action, _episode):
        return 1 if rng.random() < probabilities[action] else 0

    b1 = build_agent("B1", hparams)
    ablated = build_agent("B3-no-causal-structure", hparams)
    full = build_agent("B3", hparams)
    # Separate generators with the same seed keep the oracle identical per agent.
    actions = {}
    for name, agent, seed in (("B1", b1, 0), ("ablated", ablated, 0), ("B3", full, 0)):
        local = np.random.default_rng(seed)

        def outcome_fn(action, _episode, generator=local):
            return 1 if generator.random() < probabilities[action] else 0

        actions[name], _flags = _drive(agent, outcome_fn, n=40)
    assert actions["B1"] == actions["ablated"]
    assert full.get_state()["switch_count"] == 0
    assert actions["B3"] == actions["B1"]


def test_confident_error_rule_and_a_forced_shift(hparams):
    assert confident_mismatch([0.9, 0.9, 0.9, 0.9, 0.9], [0.9, 0.9, 0.9, 0.9, 0.9], 0.71, 0.75)
    assert not confident_mismatch([0.2, 0.2, 0.2, 0.2, 0.2], [0.8, 0.8, 0.8, 0.8, 0.8], 0.71, 0.75)
    shifted = dict(hparams)
    shifted["b3_ucb_c"] = 0.0
    agent = SelfModelAgent(shifted)

    def outcome(_action, episode):
        return 1 if episode <= 15 else 0

    _actions, flags = _drive(agent, outcome, n=25)
    assert any(flags)
    assert agent.get_state()["switch_count"] >= 1


def test_history_cannot_contain_the_future(hparams):
    agent = build_agent("B0", hparams)
    task = {
        "task_id": "t-1",
        "task_text": "Complete benchmark item 001 in the REASONING regime.",
        "task_regime": "REASONING",
        "difficulty": 1,
    }
    agent.observe_episode(
        {
            "episode_id": 1,
            "task": task,
            "task_regime": "REASONING",
            "available_actions": list(DEFAULT_ACTIONS),
        }
    )
    try:
        agent.predict(task, list(DEFAULT_ACTIONS), [{"episode_id": 1, "outcome": 1}])
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("future history was accepted")


def test_hidden_key_is_rejected(hparams):
    agent = build_agent("B2", hparams)
    try:
        agent.observe_episode({"episode_id": 1, "hidden_self_state": "HIGH"})
    except ValueError:
        return
    raise AssertionError("hidden state was accepted")
