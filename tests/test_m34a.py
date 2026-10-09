"""Mock-model tests for M34a (no real model download)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.m34a_common import (
    build_pilot_problems,
    build_pool_plan,
    choose_action,
    choose_levels,
    parse_response,
    random_derangement,
    realized_utility,
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
    ans, conf = parse_response("Answer: abc; Confidence: xx")
    assert ans is None and conf == 50.0


def test_choose_levels_prefers_lower_on_tie() -> None:
    acc = {2: 1.0, 3: 0.5, 4: 0.0, 5: 0.5, 6: 1.0, 7: 0.5, 8: 0.0}
    choice = choose_levels(acc)
    assert choice["chosen_levels"] == [2, 3, 4, 5, 6, 7]
    assert choice["accuracy_range"] == 1.0
    assert choice["stop"] is False


def test_choose_levels_stop_small_range() -> None:
    acc = {n: 0.5 + 0.01 * (n % 3) for n in range(2, 9)}
    choice = choose_levels(acc)
    assert choice["stop"] is True


def test_pool_plan_disjoint_from_pilot_and_sizes() -> None:
    pilot = build_pilot_problems(34001)
    plan = build_pool_plan(34001, [2, 3, 4, 5, 6, 7])
    assert plan["n_problems"] == 6 * (100 + 60 + 40)
    pilot_pairs = {(p["a"], p["b"]) for p in pilot}
    pool_pairs = {(p["a"], p["b"]) for p in plan["problems"]}
    assert pilot_pairs.isdisjoint(pool_pairs)
    plan2 = build_pool_plan(34001, [2, 3, 4, 5, 6, 7])
    assert plan["problems"] == plan2["problems"]


def test_action_ties_and_utilities() -> None:
    assert choose_action(0.5, "V3") == "verify"
    assert choose_action(0.5, "V2") == "answer"
    assert realized_utility("answer", True) == 1.0
    assert realized_utility("answer", False) == -1.0
    assert realized_utility("verify", False) == 0.7


def test_derangement_has_no_fixed_points() -> None:
    import random

    rng = random.Random(0)
    items = [2, 3, 4, 5, 6, 7]
    for _ in range(20):
        d = random_derangement(rng, items)
        assert set(d.keys()) == set(items)
        assert set(d.values()) == set(items)
        assert all(d[i] != i for i in items)


def test_mock_collect_resumable_and_analyze_hash_gate(tmp_path: Path) -> None:
    levels = {
        "status": "LEVELS_CHOSEN",
        "chosen_levels": [2, 3, 4, 5, 6, 7],
        "choice": {
            "chosen_levels": [2, 3, 4, 5, 6, 7],
            "accuracy_range": 1.0,
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
    lines2 = cache.read_text(encoding="utf-8").strip().splitlines()
    assert lines2 == lines1

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_sha256"] == sha256_file(cache)
    cache.write_text(cache.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    try:
        m34a_analyze.main(["--raw-dir", str(out_dir), "--report", str(tmp_path / "r.md")])
        assert False, "expected hash mismatch abort"
    except SystemExit as exc:
        assert "hash mismatch" in str(exc)


def test_mock_pilot_and_generate_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(m34a_pilot, "OUT_DIR", tmp_path / "pilot")
    rc = m34a_pilot.main(["--mock"])
    assert rc in (0, 2)
    assert (tmp_path / "pilot" / "levels.json").exists()
    client = MockQwenClient()
    problem = {
        "problem_id": "t",
        "pool": "pilot",
        "level": 2,
        "a": 12,
        "b": 34,
        "product": 408,
        "prompt": (
            "Compute 12 x 34. Reply exactly in this format: "
            "Answer: <integer>; Confidence: <0-100 probability your answer is exactly correct>"
        ),
    }
    row = run_problem(client, problem)
    assert row["correct"] is True
    assert 0 <= row["parsed_confidence"] <= 100
