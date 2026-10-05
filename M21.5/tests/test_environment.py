from environment.env import BenchmarkEnvironment, load_ground_truth
from environment.outcome_model import optimal_action
from environment.state import ACTIONS, STATES, TASK_REGIMES
from environment.task_generator import uniform_draw
from observation_boundary import assert_agent_visible

EXPECTED = {
    "HIGH": {
        "REASONING": {"DIRECT": 0.78, "EXTENDED_REASONING": 0.90, "TOOL_ASSISTED": 0.45},
        "TOOL": {"DIRECT": 0.70, "EXTENDED_REASONING": 0.62, "TOOL_ASSISTED": 0.88},
        "BALANCED": {"DIRECT": 0.82, "EXTENDED_REASONING": 0.76, "TOOL_ASSISTED": 0.68},
    },
    "MEDIUM": {
        "REASONING": {"DIRECT": 0.55, "EXTENDED_REASONING": 0.72, "TOOL_ASSISTED": 0.50},
        "TOOL": {"DIRECT": 0.58, "EXTENDED_REASONING": 0.60, "TOOL_ASSISTED": 0.78},
        "BALANCED": {"DIRECT": 0.68, "EXTENDED_REASONING": 0.64, "TOOL_ASSISTED": 0.60},
    },
    "LOW": {
        "REASONING": {"DIRECT": 0.45, "EXTENDED_REASONING": 0.40, "TOOL_ASSISTED": 0.68},
        "TOOL": {"DIRECT": 0.38, "EXTENDED_REASONING": 0.35, "TOOL_ASSISTED": 0.86},
        "BALANCED": {"DIRECT": 0.62, "EXTENDED_REASONING": 0.48, "TOOL_ASSISTED": 0.44},
    },
}
OPTIMAL = {
    "HIGH": {"REASONING": "EXTENDED_REASONING", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"},
    "MEDIUM": {"REASONING": "EXTENDED_REASONING", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"},
    "LOW": {"REASONING": "TOOL_ASSISTED", "TOOL": "TOOL_ASSISTED", "BALANCED": "DIRECT"},
}


def test_probability_table_matches_the_frozen_specification(ground_truth):
    assert ground_truth["probability_matrix"] == EXPECTED
    assert ground_truth["optimal_action"] == OPTIMAL
    seen = set()
    for state in STATES:
        for regime in TASK_REGIMES:
            row = {action: float(value) for action, value in EXPECTED[state][regime].items()}
            chosen = optimal_action(row)
            assert chosen == OPTIMAL[state][regime]
            seen.add(chosen)
    assert seen == set(ACTIONS)


def test_schedule_and_reward(ground_truth):
    env = BenchmarkEnvironment(ground_truth, 11)
    for episode in range(1, 41):
        assert env.hidden_state(episode) == "HIGH"
    for episode in range(41, 61):
        assert env.hidden_state(episode) == "LOW"
    for episode in range(61, 101):
        assert env.hidden_state(episode) == "MEDIUM"
    assert load_ground_truth()["reward"] == {"SUCCESS": 1, "FAILURE": 0}
    for action in ACTIONS:
        outcome = env.outcome(1, action)
        assert outcome in (0, 1)
        probability = env.true_probabilities(1)[action]
        draw = uniform_draw(11, 1, ACTIONS.index(action))
        assert outcome == (1 if draw < probability else 0)


def test_same_seed_is_identical_and_another_seed_can_differ(ground_truth):
    first = BenchmarkEnvironment(ground_truth, 11)
    second = BenchmarkEnvironment(ground_truth, 11)
    other = BenchmarkEnvironment(ground_truth, 23)
    assert first.regimes == second.regimes
    assert first._draws == second._draws
    assert first.regimes != other.regimes


def test_static_and_ood_controls_follow_the_specification(ground_truth):
    static = BenchmarkEnvironment(ground_truth, 11, schedule_name="static")
    assert [static.hidden_state(episode) for episode in range(1, 101)] == ["HIGH"] * 100
    primary = BenchmarkEnvironment(ground_truth, 11)
    ood = BenchmarkEnvironment(ground_truth, 11, regime_distribution="ood")
    assert static.regimes == primary.regimes
    assert ood.regimes != primary.regimes
    assert ood._draws == primary._draws


def test_public_observation_has_no_hidden_fields(ground_truth):
    env = BenchmarkEnvironment(ground_truth, 11)
    observation = env.public_observation(7)
    assert_agent_visible(observation)
    assert observation["task"]["difficulty"] == 1
    assert "HIGH" not in observation["task"]["task_text"]
    assert "LOW" not in observation["task"]["task_text"]
    assert "MEDIUM" not in observation["task"]["task_text"]
