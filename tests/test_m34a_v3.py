"""Mock tests for M34a-v3 (no model download)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.m34a_v2_common import operand_keys_from_problems
from scripts.m34a_v3_carries import n_carries_add_nn, n_carries_mul_n1
from scripts.m34a_v3_common import (
    SEED_S1_POOLS,
    SEED_S2_PILOT,
    SEED_S2_POOLS,
    aggregate_c,
    build_s1_pool_plan,
    build_s2_pilot_problems,
    build_s2_pool_plan,
    forbidden_operand_keys_s1,
    forbidden_operand_keys_s2_pilot,
    load_pilot2_family,
    select_gap_window,
)
from scripts.m34a_v3_generate import MockV3Client, run_problem
from scripts import m34a_v3_analyze
from scripts import m34a_v3_collect


def test_s1_selects_3_to_8() -> None:
    acc = load_pilot2_family("add_nn")
    choice = select_gap_window(acc, "add_nn", "test")
    assert choice["stop"] is False
    assert choice["chosen_levels"] == [3, 4, 5, 6, 7, 8]
    assert choice["best"]["gap_hat"] >= 0.05


def test_s1_pool_disjoint() -> None:
    plan = build_s1_pool_plan(SEED_S1_POOLS)
    assert plan["n_problems"] == 1200
    assert plan["family"] == "add_nn"
    keys = operand_keys_from_problems(plan["problems"])
    assert keys.isdisjoint(forbidden_operand_keys_s1())
    assert plan["problems"][0]["prompt"].startswith("Compute ")
    assert " + " in plan["problems"][0]["prompt"]


def test_s2_pilot_and_pools_disjoint() -> None:
    pilot = build_s2_pilot_problems(SEED_S2_PILOT)
    assert len(pilot) == 20 * 9
    pkeys = operand_keys_from_problems(pilot)
    assert pkeys.isdisjoint(forbidden_operand_keys_s2_pilot())
    # pools also exclude pilot
    plan = build_s2_pool_plan([2, 3, 4, 5, 6, 7], SEED_S2_POOLS)
    assert plan["n_problems"] == 1200
    assert operand_keys_from_problems(plan["problems"]).isdisjoint(
        pkeys | forbidden_operand_keys_s2_pilot()
    )


def test_aggregate_c_table_exhaustive() -> None:
    expected = {
        ("INTERNAL_VALUE", "INTERNAL_VALUE"): "C_GENERAL",
        ("INTERNAL_VALUE", "INTERNAL_NONE"): "C_MIXED",
        ("INTERNAL_NONE", "INTERNAL_VALUE"): "C_MIXED",
        ("INTERNAL_VALUE", "NOT_INFORMATIVE"): "C_PARTIAL",
        ("NOT_INFORMATIVE", "INTERNAL_VALUE"): "C_PARTIAL",
        ("INTERNAL_NONE", "INTERNAL_NONE"): "C_NOT_REPLICATED",
        ("INTERNAL_NONE", "NOT_INFORMATIVE"): "C_NOT_REPLICATED",
        ("NOT_INFORMATIVE", "INTERNAL_NONE"): "C_NOT_REPLICATED",
        ("NOT_INFORMATIVE", "NOT_INFORMATIVE"): "C_NOT_INFORMATIVE",
    }
    for (a, b), label in expected.items():
        assert aggregate_c(a, b) == label
    assert aggregate_c("INTERNAL_VALUE", "NOT_RUN") == "C_PARTIAL"


def test_carries() -> None:
    assert n_carries_add_nn(999, 1) >= 1
    assert n_carries_mul_n1(25, 4) >= 1


def test_mock_s1_collect_and_analyze(tmp_path: Path) -> None:
    plan = build_s1_pool_plan(SEED_S1_POOLS)
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
    assert (
        m34a_v3_collect.main(
            ["--setting", "s1", "--pool-plan", str(plan_path), "--out-dir", str(out), "--mock"]
        )
        == 0
    )
    row = json.loads((out / "responses.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "answer_logprob" in row
    assert "+" in row["prompt"] or row["op"] == "+"
    # hash gate
    (out / "responses.jsonl").write_text(
        (out / "responses.jsonl").read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
    )
    try:
        m34a_v3_analyze.analyze_setting(out)
        assert False, "expected hash mismatch"
    except SystemExit as exc:
        assert "hash mismatch" in str(exc)


def test_dirty_tree_refuses(monkeypatch) -> None:
    def fake_check_output(cmd, cwd=None, text=None):
        if cmd[:3] == ["git", "status", "--porcelain"]:
            return " M scripts/m34a_v3_analyze.py\n"
        return "deadbeef\n"

    monkeypatch.setattr(m34a_v3_analyze.subprocess, "check_output", fake_check_output)
    try:
        m34a_v3_analyze.require_clean_git()
        assert False
    except SystemExit as exc:
        assert "dirty" in str(exc)


def test_mock_generate_add() -> None:
    client = MockV3Client(fail_from_digits=99)
    problem = {
        "problem_id": "t",
        "pool": "test",
        "family": "add_nn",
        "level": 3,
        "a": 123,
        "b": 456,
        "op": "+",
        "target": 579,
        "product": 579,
        "prompt": "Compute 123 + 456. Reply with only the integer.",
    }
    row = run_problem(client, problem)
    assert row["correct"] is True
    assert abs(row["answer_logprob"] - sum(row["token_logprobs"])) < 1e-12
