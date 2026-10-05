"""Leakage checks for saved runs and agent source."""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

from observation_boundary import BANNED_KEYS, HIDDEN_STATE_NAMES, assert_agent_visible

AGENT_EVENT_KEYS = {
    "episode_id",
    "task_id",
    "task_text",
    "task_regime",
    "available_actions",
    "predictions",
    "uncertainties",
    "selected_action",
    "outcome",
}

FORBIDDEN_AGENT_TOKENS = (
    "HIGH",
    "MEDIUM",
    "LOW",
    "ground_truth",
    "optimal_action",
    "hidden_self_state",
    "probability_matrix",
    "0.78",
    "0.90",
    "0.88",
    "0.82",
    "0.76",
    "0.72",
    "0.68",
    "0.64",
    "0.62",
    "0.60",
    "0.58",
    "0.55",
    "0.50",
    "0.48",
    "0.45",
    "0.44",
    "0.40",
    "0.38",
    "0.35",
    "0.86",
)


class OpenGuard:
    """Record attempts to open the evaluator matrix during an agent call."""

    def __init__(self):
        self.hits: list[str] = []
        self._open = None
        self._read_text = None

    def __enter__(self):
        self._open = builtins.open
        path_type = Path

        def guarded(file, *args, **kwargs):
            self._note(file)
            return self._open(file, *args, **kwargs)

        builtins.open = guarded
        self._read_text = path_type.read_text

        def read_text(path_obj, *args, **kwargs):
            self._note(path_obj)
            return self._read_text(path_obj, *args, **kwargs)

        path_type.read_text = read_text
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.open = self._open
        Path.read_text = self._read_text
        return False

    def _note(self, file) -> None:
        name = str(file)
        if name.endswith("ground_truth.json") or name.endswith("m21_5_v1.yaml"):
            self.hits.append(name)


def scan_agent_sources(agents_dir: Path) -> list[str]:
    problems = []
    for path in sorted(agents_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_AGENT_TOKENS:
            if token in text:
                problems.append(f"{path.name} contains {token}")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            problems.append(f"{path.name} did not parse: {exc}")
    return problems


def audit_agent_event(event: dict, environment_seed: int) -> list[str]:
    problems = []
    try:
        assert_agent_visible(event)
    except ValueError as exc:
        problems.append(str(exc))
    extra = set(event) - AGENT_EVENT_KEYS
    missing = AGENT_EVENT_KEYS - set(event)
    if extra:
        problems.append(f"unexpected agent event keys {sorted(extra)}")
    if missing:
        problems.append(f"missing agent event keys {sorted(missing)}")
    if event.get("environment_seed") == environment_seed:
        problems.append("environment seed stored on the agent event")
    text = json.dumps(event, sort_keys=True)
    for name in HIDDEN_STATE_NAMES:
        if f'"{name}"' in text:
            problems.append(f"hidden state name {name} in agent event")
    return problems


def audit_history_order(events: list[dict]) -> list[str]:
    problems = []
    for index, event in enumerate(events, start=1):
        if int(event["episode_id"]) != index:
            problems.append(f"episode order broken at {event['episode_id']}")
    return problems


def audit_saved_run(raw_dir: Path, environment_seed: int) -> list[str]:
    problems = []
    for path in sorted(raw_dir.rglob("agent_events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            problems.extend(audit_agent_event(json.loads(line), environment_seed))
        events = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        problems.extend(audit_history_order(events))
    return problems
