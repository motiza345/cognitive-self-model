"""Run agents on evaluator-owned environments and write separated logs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents import PRIMARY_AGENTS, build_agent
from environment.env import BenchmarkEnvironment
from environment.state import ACTIONS
from evaluation.leakage_audit import OpenGuard, audit_agent_event
from observation_boundary import assert_agent_visible


def _guard_hits(guard: OpenGuard, episode_id: int) -> list[str]:
    return [f"episode {episode_id} opened {hit}" for hit in guard.hits]


def run_agent(agent, env: BenchmarkEnvironment, max_episodes: int | None = None) -> dict:
    """Execute one agent. The environment object is not passed into the agent."""
    limit = env.n_episodes if max_episodes is None else int(max_episodes)
    if limit < 1 or limit > env.n_episodes:
        raise ValueError("episode limit is outside the frozen environment")
    history: list[dict] = []
    agent_events: list[dict] = []
    eval_events: list[dict] = []
    states: list[dict] = []
    trace: list[str] = []
    leakage: list[str] = []
    agent.reset(0)
    for episode_id in range(1, limit + 1):
        public = json.loads(json.dumps(env.public_observation(episode_id)))
        history_view = json.loads(json.dumps(history))
        with OpenGuard() as guard:
            trace.append("observe")
            agent.observe_episode(public)
            trace.append("predict")
            prediction = agent.predict(public["task"], list(public["available_actions"]), history_view)
            trace.append("select")
            action = agent.select_action(prediction)
            leakage.extend(_guard_hits(guard, episode_id))
        if action not in ACTIONS:
            raise ValueError(f"agent selected an unknown action {action}")
        trace.append("outcome")
        outcome = int(env.outcome(episode_id, action))
        with OpenGuard() as guard:
            trace.append("update")
            agent.update(outcome)
            state = json.loads(json.dumps(agent.get_state()))
            leakage.extend(_guard_hits(guard, episode_id))
        event = {
            "episode_id": episode_id,
            "task_id": public["task"]["task_id"],
            "task_text": public["task"]["task_text"],
            "task_regime": public["task_regime"],
            "available_actions": list(public["available_actions"]),
            "predictions": {key: float(value) for key, value in prediction["predictions"].items()},
            "uncertainties": {key: float(value) for key, value in prediction["uncertainty"].items()},
            "selected_action": action,
            "outcome": outcome,
        }
        leakage.extend(audit_agent_event(event, env.seed))
        try:
            assert_agent_visible(state)
        except ValueError as exc:
            leakage.append(f"episode {episode_id} state: {exc}")
        expected = ["observe", "predict", "select", "outcome", "update"]
        if trace[-5:] != expected:
            raise RuntimeError("prediction was not sealed before the outcome")
        history.append(event)
        agent_events.append(event)
        eval_events.append(env.evaluator_event(episode_id, action, outcome))
        states.append(state)
    return {
        "agent_events": agent_events,
        "eval_events": eval_events,
        "states": states,
        "trace": trace,
        "leakage": leakage,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run(destination: Path, result: dict) -> None:
    write_jsonl(destination / "agent_events.jsonl", result["agent_events"])
    write_jsonl(destination / "evaluator_events.jsonl", result["eval_events"])
    write_jsonl(destination / "agent_state.jsonl", result["states"])


def environment_fingerprint(env: BenchmarkEnvironment) -> dict:
    regime_payload = json.dumps(env.regimes, separators=(",", ":"))
    draw_payload = json.dumps(
        [
            [episode_id, action, env._draws[(episode_id, action)]]
            for episode_id in range(1, env.n_episodes + 1)
            for action in ACTIONS
        ],
        separators=(",", ":"),
    )
    return {
        "seed": env.seed,
        "schedule_name": env.schedule_name,
        "regime_distribution": env.regime_distribution,
        "regimes_sha256": hashlib.sha256(regime_payload.encode("utf-8")).hexdigest(),
        "draws_sha256": hashlib.sha256(draw_payload.encode("utf-8")).hexdigest(),
    }


def replay_fingerprint(ground_truth: dict, seed: int, schedule_name: str, regime_distribution: str) -> dict:
    env = BenchmarkEnvironment(
        ground_truth,
        seed,
        schedule_name=schedule_name,
        regime_distribution=regime_distribution,
    )
    return environment_fingerprint(env)


def run_condition(
    ground_truth: dict,
    hparams: dict,
    seeds: list[int],
    agents: tuple[str, ...],
    schedule_name: str,
    regime_distribution: str,
    destination: Path,
) -> list[str]:
    """Write a complete condition. Caller supplies an empty destination."""
    leakage: list[str] = []
    for seed in seeds:
        env = BenchmarkEnvironment(
            ground_truth,
            int(seed),
            schedule_name=schedule_name,
            regime_distribution=regime_distribution,
        )
        seed_dir = destination / f"seed_{seed}"
        fingerprint = environment_fingerprint(env)
        (seed_dir / "environment_fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        repeated = replay_fingerprint(ground_truth, int(seed), schedule_name, regime_distribution)
        if repeated != fingerprint:
            raise RuntimeError(f"environment fingerprint was not reproducible for seed {seed}")
        for agent_name in agents:
            agent = build_agent(agent_name, hparams)
            result = run_agent(agent, env)
            if result["leakage"]:
                leakage.extend(f"seed {seed} {agent_name}: {item}" for item in result["leakage"])
            if len(result["agent_events"]) != env.n_episodes:
                raise RuntimeError("episode count drifted")
            write_run(seed_dir / agent_name, result)
    return leakage


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    return rows


def iter_seed_runs(condition_dir: Path):
    for seed_dir in sorted(condition_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_", 1)[1])
        fingerprint = json.loads((seed_dir / "environment_fingerprint.json").read_text(encoding="utf-8"))
        agents = {}
        for agent_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            agents[agent_dir.name] = {
                "agent_events": load_jsonl(agent_dir / "agent_events.jsonl"),
                "eval_events": load_jsonl(agent_dir / "evaluator_events.jsonl"),
                "states": load_jsonl(agent_dir / "agent_state.jsonl"),
            }
        yield seed, fingerprint, agents


def primary_agent_names() -> tuple[str, ...]:
    return PRIMARY_AGENTS
