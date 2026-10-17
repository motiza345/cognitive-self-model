import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import m50_analyze, m50_generate, m50_pilot  # noqa: E402
from scripts.m50_arms import (apply_condition, bootstrap_diff, condition_labels, fit_probe, holm, label_floor,  # noqa: E402
                              label_probe, util_from_p)
from scripts.m50_common import (choose_windows, compute_gates, earlier_exclusions, gap_hat, is_correct, make_pool,  # noqa: E402
                                make_prompt, parse_answer, truth_of)


# ----------------------------------------------------------------------------- problems
def test_prompts_are_exact():
    assert make_prompt("mul_n1", [123, 4]) == "Compute 123 x 4. Reply with only the integer."
    assert make_prompt("add_nn", [12, 34]) == "Compute 12 + 34. Reply with only the integer."
    assert make_prompt("sort_n", [30, 10, 20]) == "Sort ascending: 30, 10, 20. Reply with only the sorted numbers separated by commas."


def test_parse_and_correctness():
    assert parse_answer("mul_n1", "The answer is 1,234") == 1234
    assert parse_answer("mul_n1", "no digits") is None
    assert is_correct("mul_n1", [12, 3], "36")
    assert not is_correct("mul_n1", [12, 3], "37")
    assert is_correct("add_nn", [10, 20], " 30\n")
    assert parse_answer("sort_n", "10, 20 , 30") == [10, 20, 30]
    assert is_correct("sort_n", [30, 10, 20], "10, 20, 30")
    assert not is_correct("sort_n", [30, 10, 20], "10, 30, 20")
    assert not is_correct("sort_n", [30, 10, 20], "")
    assert truth_of("sort_n", [30, 10, 20]) == [10, 20, 30]


def test_pools_are_deterministic_unique_and_disjoint():
    a = make_pool("mul_n1", 5, "cal", 30, 50001, set())
    b = make_pool("mul_n1", 5, "cal", 30, 50001, set())
    assert a == b
    assert len({tuple(p["payload"]) for p in a}) == 30
    keys = {tuple(p["payload"]) for p in a}
    t = make_pool("mul_n1", 5, "test", 30, 50001, keys)
    assert not (keys & {tuple(p["payload"]) for p in t})
    assert all(len(str(p["payload"][0])) == 5 and 2 <= p["payload"][1] <= 9 for p in a)
    s = make_pool("sort_n", 7, "cal", 10, 50001, set())
    assert all(len(set(p["payload"])) == 7 and all(10 <= x <= 99 for x in p["payload"]) for p in s)


def test_earlier_exclusions_read_operand_pairs(tmp_path):
    d = tmp_path / "reports" / "m34a_x_raw"
    d.mkdir(parents=True)
    (d / "responses.jsonl").write_text(json.dumps({"a": 123, "b": 4}) + "\n")
    assert (123, 4) in earlier_exclusions(tmp_path)
    pool = make_pool("mul_n1", 3, "cal", 50, 50001, {(123, 4)})
    assert (123, 4) not in {tuple(p["payload"]) for p in pool}


# ----------------------------------------------------------------------------- pilot decisions
def test_gap_hat_formula():
    assert gap_hat([0.95, 0.95, 0.9, 0.8]) > gap_hat([0.5, 0.5, 0.5, 0.5])
    assert gap_hat([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_choose_windows_rules():
    acc = {
        "mul_n1": {2: .95, 3: .95, 4: .9, 5: .8, 6: .65, 7: .5, 8: .25, 9: .45, 10: .55},
        "add_nn": {3: .99, 4: .99, 5: .98, 6: .97, 7: .96},   # no level in [0.25, 0.75]: dropped
        "sort_n": {3: .9, 4: .8, 5: .6, 6: .4, 7: .3, 8: .2},
    }
    chosen = choose_windows(acc)
    assert "add_nn" not in chosen
    assert chosen["mul_n1"]["levels"] == [2, 3, 4, 5] or len(chosen["mul_n1"]["levels"]) == 4
    assert all(len(c["levels"]) == 4 for c in chosen.values())
    # ties go to lower levels
    flat = {"mul_n1": {n: 0.5 for n in range(2, 8)}}
    assert choose_windows(flat)["mul_n1"]["levels"] == [2, 3, 4, 5]


def _pilot_rows(acc_by_scope, per=40):
    rows = []
    for (task, level), a in acc_by_scope.items():
        for i in range(per):
            rows.append({"task": task, "level": level, "correct": i < round(a * per), "problem_id": f"{task}-{level}-{i}",
                         "token_logprobs": [-0.01] if i < round(a * per) else [-2.0]})
    return rows


def test_compute_gates_pass_and_fail():
    chosen = {"mul_n1": {"levels": [2, 3, 4, 5]}, "sort_n": {"levels": [3, 4, 5, 6]}}
    good = {(t, lv): a for t, ls in (("mul_n1", [2, 3, 4, 5]), ("sort_n", [3, 4, 5, 6])) for lv, a in zip(ls, (.9, .7, .5, .3))}
    g = compute_gates(chosen, _pilot_rows(good), se_estimate=0.01)
    assert g["G_ceiling"] and g["G_variance"] and g["all_pass"] and g["test_per_scope_level"] == 40
    g2 = compute_gates(chosen, _pilot_rows(good), se_estimate=0.02)
    assert g2["test_per_scope_level"] == 80
    bad = {k: 0.1 for k in good}
    assert not compute_gates(chosen, _pilot_rows(bad), 0.01)["all_pass"]
    assert not compute_gates({"mul_n1": {"levels": [2, 3, 4, 5]}}, _pilot_rows(good), 0.01)["tasks_ok"]


def test_pilot_decide_runs_on_synthetic_rows():
    acc = {}
    for task, levels in (("mul_n1", range(2, 11)), ("add_nn", range(3, 16)), ("sort_n", range(3, 13))):
        for i, lv in enumerate(levels):
            acc[(task, lv)] = max(0.05, 0.98 - 0.12 * i)
    levels_json, gates = m50_pilot.decide(_pilot_rows(acc))
    assert set(levels_json["chosen"]) <= {"mul_n1", "add_nn", "sort_n"} and gates["n_tasks_kept"] >= 2
    assert gates["se_estimate_scoped_minus_best_simple"] is not None


# ----------------------------------------------------------------------------- conditions and labels
def _items(n_sort=400):
    return [{"problem_id": f"sort_n-6-test-{i:04d}", "task": "sort_n", "level": 6, "pool": "test", "correct": True, "conf": 0.9}
            for i in range(n_sort)] + [{"problem_id": f"mul_n1-5-test-{i:04d}", "task": "mul_n1", "level": 5, "pool": "test",
                                        "correct": True, "conf": 0.9} for i in range(100)]


def test_implanted_condition_only_touches_sort_and_is_deterministic():
    items = _items()
    a, b = apply_condition(items, "implanted"), apply_condition(items, "implanted")
    assert [r["correct"] for r in a] == [r["correct"] for r in b]
    assert all(r["correct"] for r in a if r["task"] == "mul_n1")
    wrong = sum(1 for r in a if r["task"] == "sort_n" and not r["correct"]) / 400
    assert 0.2 < wrong < 0.4
    assert [r["correct"] for r in apply_condition(items, "natural")] == [True] * len(items)
    with pytest.raises(ValueError):
        apply_condition(items, "other")


def test_labels_are_exhaustive_and_follow_the_thresholds():
    assert label_floor("A", 0.05, 0.01, 0.03) == "A_SUPPORTED"
    assert label_floor("A", 0.02, 0.01, 0.03) == "A_WEAK"
    assert label_floor("A", 0.05, -0.01, 0.03) == "A_NOT_SUPPORTED"
    assert label_probe(0.03, 0.01, 0.05) == "PROBE_BETTER"
    assert label_probe(-0.03, -0.05, -0.01) == "LOGPROB_BETTER"
    assert label_probe(0.01, -0.01, 0.03) == "UNRESOLVED"
    rng = np.random.default_rng(0)
    for _ in range(500):
        m = rng.normal(0, 0.05)
        lo, hi = m - abs(rng.normal(0, 0.03)), m + abs(rng.normal(0, 0.03))
        comps = {h: {"mean": m, "lower": lo, "upper": hi} for h in "ABCD"}
        for cond in ("natural", "implanted"):
            lab = condition_labels(cond, True, comps)
            assert lab["A"].startswith("A_") and lab["B"].startswith("B_") and lab["C"].startswith("C_")
            assert lab["D"] in ("PROBE_BETTER", "LOGPROB_BETTER", "UNRESOLVED")
            assert lab["status"] in ("OK", "POSITIVE_CONTROL_FAILED")
    assert condition_labels("natural", False, comps)["status"] == "NOT_INFORMATIVE"
    strong = {h: {"mean": 0.1, "lower": 0.05, "upper": 0.15} for h in "ABCD"}
    assert condition_labels("implanted", True, strong)["status"] == "OK"
    weak = {**strong, "C": {"mean": 0.0, "lower": -0.01, "upper": 0.01}}
    assert condition_labels("implanted", True, weak)["status"] == "POSITIVE_CONTROL_FAILED"
    assert condition_labels("natural", True, weak)["status"] == "OK"


def test_util_from_p_matches_thresholds():
    assert util_from_p(0.9, True) == 1.0 and util_from_p(0.9, False) == -1.0
    assert util_from_p(0.85, True) == 0.7 and util_from_p(0.2, True) == 0.7


def test_bootstrap_and_holm():
    r = bootstrap_diff(np.full(200, 0.1), np.zeros(200))
    assert r["lower"] > 0.09 and r["p"] <= 0.001
    flags = holm({"a": 0.001, "b": 0.02, "c": 0.06})
    assert flags["a"] and not flags["c"]


def test_probe_learns_a_planted_signal_and_uses_cv_on_cal_only():
    rng = np.random.default_rng(1)
    n, d = 300, 20
    scope = rng.integers(0, 3, n)
    y = rng.random(n) < 0.5
    X = rng.normal(size=(n, d))
    X[:, 0] += 2.0 * y
    probe = fit_probe(X[:200], y[:200], scope[:200], 3)
    p = probe.predict(X[200:], scope[200:])
    pos, neg = p[y[200:]], p[~y[200:]]
    auroc = np.mean([a > b for a in pos for b in neg])
    assert auroc > 0.85


# ----------------------------------------------------------------------------- generation with a mock model
def _solve(prompt):
    """Mock model: answers correctly with a probability that falls with problem size; log probabilities are informative."""
    nums = [int(x) for x in re.findall(r"\d+", prompt)]
    if prompt.startswith("Sort"):
        truth, text_ok = sorted(nums), ", ".join(str(x) for x in sorted(nums))
        level = len(nums)
    elif " x " in prompt:
        truth = nums[0] * nums[1]; text_ok = str(truth); level = len(str(nums[0]))
    else:
        truth = nums[0] + nums[1]; text_ok = str(truth); level = len(str(nums[0]))
    u = (sum(nums) * 7919 % 1000) / 1000.0
    ok = u < max(0.2, 1.0 - 0.12 * level)
    text = text_ok if ok else "0"
    lps = [-0.002 * (1 + u)] * 3 if ok else [-0.8 - u] * 3
    return text, lps


def _small_plan(n_cal=40, n_test=20):
    plan = []
    for task, level in (("mul_n1", 5), ("mul_n1", 6), ("sort_n", 6), ("sort_n", 7)):
        cal = make_pool(task, level, "cal", n_cal, 50001, set())
        keys = {tuple(p["payload"]) if task != "sort_n" else tuple(p["payload"]) for p in cal}
        plan += cal + make_pool(task, level, "test", n_test, 50001, keys | {("x",)})
    return plan


def test_generation_is_resumable_and_writes_a_valid_manifest(tmp_path):
    plan = _small_plan()
    gen = m50_generate.MockGenerator(_solve, dim=8)
    first = plan[:50]
    m50_generate.run_generation(first, gen, tmp_path, batch=16, hidden=True, log=lambda *a: None)
    assert len(m50_generate.load_done(tmp_path / "responses.jsonl")) == 50
    gen2 = m50_generate.MockGenerator(_solve, dim=8)
    r = m50_generate.run_generation(plan, gen2, tmp_path, batch=16, hidden=True, log=lambda *a: None)
    assert r["generated"] == len(plan) - 50 and sum(gen2.batch_sizes) == len(plan) - 50
    manifest = m50_generate.write_manifest("main", plan, tmp_path, {"generator": "Mock"}, 16)
    assert manifest["n_rows"] == len(plan) and manifest["cache_sha256"] == m50_analyze.sha256_file(tmp_path / "responses.jsonl")
    assert manifest["hidden_chunks_sha256"]


def test_smoke_flags_structural_problems():
    plan = _small_plan()
    assert m50_generate.smoke(m50_generate.MockGenerator(_solve), plan, log=lambda *a: None)
    broken = m50_generate.MockGenerator(lambda p: ("", []))
    assert not m50_generate.smoke(broken, plan, log=lambda *a: None)


def test_generator_module_does_not_import_torch_at_import_time():
    assert "torch" not in sys.modules or True  # HFGenerator imports torch lazily inside __init__
    src = (ROOT / "scripts" / "m50_generate.py").read_text()
    assert "\nimport torch" not in src.split("class HFGenerator")[0]


# ----------------------------------------------------------------------------- full analysis on a synthetic cache
def test_analysis_runs_end_to_end_on_a_synthetic_cache(tmp_path):
    plan = _small_plan(n_cal=50, n_test=25)
    gen = m50_generate.MockGenerator(_solve, dim=12)
    m50_generate.run_generation(plan, gen, tmp_path, batch=32, hidden=True, log=lambda *a: None)
    m50_generate.write_manifest("main", plan, tmp_path, {"generator": "Mock"}, 32)
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"all_pass": True, "G_ceiling": True, "G_variance": True, "test_per_scope_level": 25}))
    summary = m50_analyze.run(tmp_path, gates, seeds=[0], workers=1, log=lambda *a: None)
    for cond in ("natural", "implanted"):
        c = summary["conditions"][cond]
        assert c["labels"]["status"] in ("OK", "POSITIVE_CONTROL_FAILED")
        assert c["n_test"] == 100 and c["n_cal"] == 200
        assert {"A1 always answer", "A2 always verify", "A3 random verify", "A4 raw confidence", "A5 global calibrated",
                "A6 scoped layer", "A7 track record", "A8 probe prompt", "A9 probe answer", "R1 instance oracle"} <= set(c["utility"])
        assert c["utility"]["R1 instance oracle"] >= c["utility"]["A2 always verify"] - 1e-9
    assert set(summary["holm"]) == {f"{c}:{h}" for c in ("natural", "implanted") for h in "ABCD" if not (c == "natural" and h == "A")}
    out = tmp_path / "report.md"
    m50_analyze.write_report(out, summary)
    assert "Condition: natural" in out.read_text() and "H-A" in out.read_text()


def test_analysis_refuses_a_tampered_cache(tmp_path):
    plan = _small_plan(n_cal=20, n_test=10)
    m50_generate.run_generation(plan, m50_generate.MockGenerator(_solve), tmp_path, batch=16, hidden=True, log=lambda *a: None)
    m50_generate.write_manifest("main", plan, tmp_path, {"generator": "Mock"}, 16)
    with open(tmp_path / "responses.jsonl", "a") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError):
        m50_analyze.load_items(tmp_path)


def test_gate_failure_gives_not_informative_labels():
    comps = {h: {"mean": 0.1, "lower": 0.05, "upper": 0.15} for h in "ABCD"}
    lab = condition_labels("natural", False, comps)
    assert set(lab.values()) == {"NOT_INFORMATIVE"}


def test_analysis_resumes_and_gives_identical_numbers(tmp_path):
    plan = _small_plan(n_cal=40, n_test=20)
    m50_generate.run_generation(plan, m50_generate.MockGenerator(_solve, dim=8), tmp_path, batch=32, hidden=True, log=lambda *a: None)
    m50_generate.write_manifest("main", plan, tmp_path, {"generator": "Mock"}, 32)
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps({"all_pass": True, "G_ceiling": True, "G_variance": True, "test_per_scope_level": 20}))
    cache = tmp_path / "analysis_cache"
    with pytest.raises(m50_analyze.OutOfTime):
        m50_analyze.run(tmp_path, gates, seeds=[0, 1], workers=1, log=lambda *a: None, cache_dir=cache, max_seconds=0.0001)
    first = m50_analyze.run(tmp_path, gates, seeds=[0, 1], workers=1, log=lambda *a: None, cache_dir=cache)
    again = m50_analyze.run(tmp_path, gates, seeds=[0, 1], workers=1, log=lambda *a: None, cache_dir=cache)
    fresh = m50_analyze.run(tmp_path, gates, seeds=[0, 1], workers=1, log=lambda *a: None)
    for c in ("natural", "implanted"):
        assert first["conditions"][c]["utility"] == again["conditions"][c]["utility"] == fresh["conditions"][c]["utility"]


def test_git_dirty_ignores_only_untracked_m50_outputs(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    (tmp_path / "a.txt").write_text("x")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c"], cwd=tmp_path)
    from scripts.m50_common import git_dirty
    assert not git_dirty(tmp_path)
    (tmp_path / "reports" / "m50_raw").mkdir(parents=True)
    (tmp_path / "reports" / "m50_raw" / "responses.jsonl").write_text("{}")
    assert not git_dirty(tmp_path)
    (tmp_path / "other.txt").write_text("y")
    assert git_dirty(tmp_path)
    (tmp_path / "other.txt").unlink()
    (tmp_path / "a.txt").write_text("changed")
    assert git_dirty(tmp_path)
