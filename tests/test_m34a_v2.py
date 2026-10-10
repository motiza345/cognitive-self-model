"""Mock tests for M34a-v2 (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.m34a_v2_common import (
    SEED_V2_POOLS,
    build_v2_pool_plan,
    forbidden_operand_keys,
    load_pilot2_mul_n1,
    operand_keys_from_problems,
    select_v2_window,
    tercile_bin,
    tercile_edges,
)
from scripts.m34a_v2_generate import MockQwenV2Client, run_problem
from scripts import m34a_v2_analyze
from scripts import m34a_v2_collect


def test_select_v2_window_picks_2_to_7() -> None:
    acc = load_pilot2_mul_n1()
    choice = select_v2_window(acc)
    assert choice["stop"] is False
    assert choice["chosen_levels"] == [2, 3, 4, 5, 6, 7]
    assert choice["best"]["gap_hat"] >= 0.05


def test_select_v2_stop_and_tie_lower() -> None:
    flat = {n: 0.6 for n in range(2, 11)}
    choice = select_v2_window(flat)
    assert choice["stop"] is True
    # equal gap_hat (0): first/lowest window is the recorded best
    assert choice["best"]["levels"] == [2, 3, 4, 5, 6, 7]


def test_pool_plan_disjoint_from_pilots_and_v1() -> None:
    plan = build_v2_pool_plan(SEED_V2_POOLS, [2, 3, 4, 5, 6, 7])
    assert plan["n_problems"] == 1200
    used = forbidden_operand_keys()
    keys = operand_keys_from_problems(plan["problems"])
    assert keys.isdisjoint(used)
    assert len(keys) == 1200
    plan2 = build_v2_pool_plan(SEED_V2_POOLS, [2, 3, 4, 5, 6, 7])
    assert plan["problems"] == plan2["problems"]


def test_tercile_ties_go_lower() -> None:
    edges = tercile_edges([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert tercile_bin(edges[0], edges) == 0
    assert tercile_bin(edges[1], edges) == 1
    assert tercile_bin(edges[1] + 1e-9, edges) == 2 or tercile_bin(edges[1] + 1.0, edges) == 2


def test_mock_collect_and_hash_gate(tmp_path: Path) -> None:
    plan = build_v2_pool_plan(SEED_V2_POOLS, [2, 3, 4, 5, 6, 7])
    slim = []
    counts: dict[tuple[str, int], int] = {}
    for p in plan["problems"]:
        key = (p["pool"], p["level"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= 2:
            slim.append(p)
    plan["problems"] = slim
    plan["n_problems"] = len(slim)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    out = tmp_path / "raw"
    assert m34a_v2_collect.main(["--pool-plan", str(plan_path), "--out-dir", str(out), "--mock"]) == 0
    cache = out / "responses.jsonl"
    n1 = len(cache.read_text(encoding="utf-8").strip().splitlines())
    assert n1 == len(slim)
    row = json.loads(cache.read_text(encoding="utf-8").splitlines()[0])
    assert "answer_logprob" in row and "token_logprobs" in row
    assert "parsed_confidence" not in row
    assert m34a_v2_collect.main(["--pool-plan", str(plan_path), "--out-dir", str(out), "--mock"]) == 0
    n2 = len(cache.read_text(encoding="utf-8").strip().splitlines())
    assert n2 == n1
    cache.write_text(cache.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    try:
        m34a_v2_analyze.main(
            ["--raw-dir", str(out), "--report", str(tmp_path / "r.md"), "--allow-dirty"]
        )
        assert False, "expected hash mismatch"
    except SystemExit as exc:
        assert "hash mismatch" in str(exc)


def test_dirty_tree_refuses_without_flag(monkeypatch) -> None:
    def fake_check_output(cmd, cwd=None, text=None):
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return " M scripts/m34a_v2_analyze.py\n"
        return "deadbeef\n"

    monkeypatch.setattr(m34a_v2_analyze.subprocess, "check_output", fake_check_output)
    try:
        m34a_v2_analyze.require_clean_git()
        assert False, "expected dirty refusal"
    except SystemExit as exc:
        assert "dirty" in str(exc)


def test_mock_generate_logprobs() -> None:
    client = MockQwenV2Client(fail_from_digits=99)
    problem = {
        "problem_id": "t",
        "pool": "test",
        "family": "mul_n1",
        "level": 2,
        "a": 12,
        "b": 3,
        "op": "x",
        "target": 36,
        "product": 36,
        "prompt": "Compute 12 x 3. Reply with only the integer.",
    }
    row = run_problem(client, problem)
    assert row["correct"] is True
    assert row["token_logprobs"]
    assert abs(row["answer_logprob"] - sum(row["token_logprobs"])) < 1e-12
