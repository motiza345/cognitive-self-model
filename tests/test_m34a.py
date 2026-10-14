"""Mock-model tests for M34a (no real model download)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.m34a_common import (
    FAMILY_SPECS,
    build_pilot2_problems,
    build_pilot_problems,
    build_pool_plan,
    choose_action,
    choose_levels,
    parse_response,
    random_derangement,
    realized_utility,
    select_pilot2_window,
    sha256_file,
)
from scripts.m34a_generate import MockQwenClient, run_problem
from scripts import m34a_analyze
from scripts import m34a_collect
from scripts import m34a_make_pools
from scripts import m34a_pilot


def test_parse_response_rules() -> None:
    ans, conf = parse_response("Answer: 1,234; Confidence: 80")
    assert ans == 1234 and conf == 80.0
    ans, conf = parse_response("nope")
    assert ans is None and conf == 50.0
    ans, conf = parse_response("42")
    assert ans == 42 and conf == 50.0


def test_choose_levels_prefers_lower_on_tie() -> None:
    acc = {2: 1.0, 3: 0.5, 4: 0.0, 5: 0.5, 6: 1.0, 7: 0.5, 8: 0.0}
    choice = choose_levels(acc)
    assert choice["chosen_levels"] == [2, 3, 4, 5, 6, 7]
    assert choice["stop"] is False


def test_select_pilot2_stop_when_score_lt_4() -> None:
    # All accuracies outside [0.15,0.85] → score 0 → STOP
    fam = {
        "add_nn": {n: 0.0 if n < 10 else 1.0 for n in range(3, 16)},
        "mul_n2": {n: 0.0 for n in range(2, 9)},
        "mul_n1": {n: 1.0 for n in range(2, 11)},
    }
    choice = select_pilot2_window(fam)
    assert choice["stop"] is True
    assert choice["best_score"] < 4
    assert choice["chosen_levels"] is None


def test_select_pilot2_ties_prefer_span_then_family_then_lower() -> None:
    # add_nn and mul_n2 both can get score 6; add_nn wins family order when span equal.
    mid = {n: 0.5 for n in range(3, 9)}  # 3..8 six levels all in band
    # extend add_nn levels
    add = {n: 0.5 for n in range(3, 16)}
    mul2 = {n: 0.5 for n in range(2, 9)}
    mul1 = {n: 0.0 for n in range(2, 11)}
    choice = select_pilot2_window({"add_nn": add, "mul_n2": mul2, "mul_n1": mul1})
    assert choice["stop"] is False
    assert choice["best_score"] == 6
    assert choice["chosen_family"] == "add_nn"
    assert choice["chosen_levels"] == [3, 4, 5, 6, 7, 8]

    # Same score; larger span wins over smaller span even if later family
    add2 = {n: 0.5 for n in range(3, 16)}  # span 0
    mul2b = {2: 0.15, 3: 0.85, 4: 0.5, 5: 0.5, 6: 0.5, 7: 0.5, 8: 0.5}  # score 6, span 0.7
    choice2 = select_pilot2_window({"add_nn": add2, "mul_n2": mul2b, "mul_n1": mul1})
    assert choice2["chosen_family"] == "mul_n2"
    assert choice2["chosen_levels"] == [2, 3, 4, 5, 6, 7]
    assert choice2["accuracy_range"] == 0.7


def test_select_pilot2_lower_levels_on_remaining_tie() -> None:
    # add_nn: two windows score 4 with same span; prefer lower start
    acc = {}
    for n in range(3, 16):
        acc[n] = 0.5 if n <= 8 else 0.0
    # window 3-8: all 0.5 → score 6
    # Force only score-4 ties with equal span via mul only one family
    # Levels 3..14 flat 0.5 except make two windows... simpler: all 0.5 → lower window
    flat = {n: 0.5 for n in range(3, 16)}
    choice = select_pilot2_window(
        {"add_nn": flat, "mul_n2": {n: 0.0 for n in range(2, 9)}, "mul_n1": {n: 0.0 for n in range(2, 11)}}
    )
    assert choice["chosen_family"] == "add_nn"
    assert choice["chosen_levels"] == [3, 4, 5, 6, 7, 8]


def test_pool_plan_amendment1_disjoint_and_sized() -> None:
    p1 = build_pilot_problems(34001)
    p2 = build_pilot2_problems(34002)
    plan = build_pool_plan(34001, [3, 4, 5, 6, 7, 8], "add_nn")
    assert plan["n_problems"] == 6 * (100 + 60 + 40)
    assert plan["family"] == "add_nn"
    used = {(p["op"], p["a"], p["b"]) for p in p1 + p2}
    pool_keys = {(p["op"], p["a"], p["b"]) for p in plan["problems"]}
    assert used.isdisjoint(pool_keys)
    plan2 = build_pool_plan(34001, [3, 4, 5, 6, 7, 8], "add_nn")
    assert plan["problems"] == plan2["problems"]


def test_action_ties_and_utilities() -> None:
    assert choose_action(0.5, "V3") == "verify"
    assert choose_action(0.5, "V2") == "answer"
    assert realized_utility("answer", True) == 1.0
    assert realized_utility("verify", False) == 0.7


def test_derangement_has_no_fixed_points() -> None:
    import random

    rng = random.Random(0)
    items = [2, 3, 4, 5, 6, 7]
    for _ in range(20):
        d = random_derangement(rng, items)
        assert all(d[i] != i for i in items)


def test_mock_collect_resumable_and_analyze_hash_gate(tmp_path: Path) -> None:
    levels = {
        "status": "LEVELS_CHOSEN",
        "chosen_levels": [3, 4, 5, 6, 7, 8],
        "chosen_family": "add_nn",
        "choice": {
            "chosen_levels": [3, 4, 5, 6, 7, 8],
            "chosen_family": "add_nn",
            "best_score": 6,
            "stop": False,
        },
    }
    levels_path = tmp_path / "levels.json"
    levels_path.write_text(json.dumps(levels), encoding="utf-8")
    plan_path = tmp_path / "pool_plan.json"
    assert m34a_make_pools.main(["--levels", str(levels_path), "--out", str(plan_path)]) == 0

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    slim = []
    counts: dict[tuple[str, int], int] = {}
    for p in plan["problems"]:
        key = (p["pool"], p["level"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= 2:
            slim.append(p)
    plan["problems"] = slim
    plan["n_problems"] = len(slim)
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    out_dir = tmp_path / "raw"
    assert (
        m34a_collect.main(["--pool-plan", str(plan_path), "--out-dir", str(out_dir), "--mock"])
        == 0
    )
    cache = out_dir / "responses.jsonl"
    lines1 = cache.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines1) == len(slim)
    assert (
        m34a_collect.main(["--pool-plan", str(plan_path), "--out-dir", str(out_dir), "--mock"])
        == 0
    )
    assert cache.read_text(encoding="utf-8").strip().splitlines() == lines1

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_sha256"] == sha256_file(cache)
    cache.write_text(cache.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    try:
        m34a_analyze.main(["--raw-dir", str(out_dir), "--report", str(tmp_path / "r.md")])
        assert False, "expected hash mismatch abort"
    except SystemExit as exc:
        assert "hash mismatch" in str(exc)


def test_mock_pilot2_smoke() -> None:
    client = MockQwenClient(fail_from_digits=99)
    problem = {
        "problem_id": "t",
        "pool": "pilot2",
        "family": "add_nn",
        "level": 3,
        "a": 100,
        "b": 200,
        "op": "+",
        "target": 300,
        "product": 300,
        "prompt": "Compute 100 + 200. Reply with only the integer.",
    }
    row = run_problem(client, problem)
    assert row["correct"] is True
    assert FAMILY_SPECS["add_nn"]["levels"][0] == 3
    assert len(build_pilot2_problems(34002)) == 13 * 20 + 7 * 20 + 9 * 20
