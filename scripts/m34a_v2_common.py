"""M34a-v2 helpers: level selection, pool plan, utilities. No model calls."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from scripts.m34a_common import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    FAMILY_SPECS,
    N_H_VALUES,
    N_HISTORY_DRAWS,
    PILOT_PER_LEVEL,
    POOL_SIZES,
    PRIMARY_N_H,
    PROMPT_TEMPLATE_A1,
    SEED_PILOT,
    SEED_PILOT2,
    build_pilot2_problems,
    build_pilot_problems,
    choose_action,
    load_json,
    make_family_problem,
    n_digit_int,
    paired_mean_ci,
    parse_response,
    random_derangement,
    realized_utility,
    sha256_file,
    write_json,
)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
FAMILY = "mul_n1"
SEED_V2_POOLS = 34003
VERIFY_U = 0.7
WINDOW = 6
GAP_FLOOR = 0.05

V1_POOL_PLAN = Path("reports/m34a_pilot2/pool_plan.json")
PILOT2_LEVELS = Path("reports/m34a_pilot2/levels.json")


def eu_star(p: float) -> float:
    """Best of answer vs verify (abstain 0 is never better than verify)."""
    return max(2.0 * p - 1.0, VERIFY_U)


def select_v2_window(mul_n1_acc: dict[int, float]) -> dict[str, Any]:
    """Choose 6 contiguous mul_n1 levels maximizing gap_hat; ties -> lower levels."""
    levels = sorted(mul_n1_acc)
    if len(levels) < WINDOW:
        raise ValueError("need at least 6 mul_n1 levels")
    windows: list[dict[str, Any]] = []
    for i in range(len(levels) - WINDOW + 1):
        w = levels[i : i + WINDOW]
        ps = [float(mul_n1_acc[L]) for L in w]
        pbar = sum(ps) / len(ps)
        per = [eu_star(p) for p in ps]
        u_level = sum(per) / len(per)
        u_global = eu_star(pbar)
        gap = u_level - u_global
        windows.append(
            {
                "levels": list(w),
                "p": {str(L): p for L, p in zip(w, ps)},
                "pbar": pbar,
                "u_level_mean": u_level,
                "u_global": u_global,
                "gap_hat": gap,
            }
        )
    # max gap_hat, then lower start level (windows already low-to-high; only update on >)
    best = windows[0]
    for row in windows[1:]:
        if row["gap_hat"] > best["gap_hat"]:
            best = row
    stop = best["gap_hat"] < GAP_FLOOR
    return {
        "stop": stop,
        "chosen_levels": None if stop else best["levels"],
        "best": best,
        "all_windows": windows,
        "rule": (
            "gap_hat = mean_L max(2p_L-1, 0.7) - max(2*pbar-1, 0.7); "
            "max gap_hat; ties -> lower levels; STOP if gap_hat < 0.05"
        ),
        "source": "reports/m34a_pilot2/levels.json choice.family_accuracies.mul_n1 only",
        "family": FAMILY,
    }


def load_pilot2_mul_n1(path: Path = PILOT2_LEVELS) -> dict[int, float]:
    doc = load_json(path)
    raw = doc["choice"]["family_accuracies"]["mul_n1"]
    return {int(k): float(v) for k, v in raw.items()}


def operand_keys_from_problems(problems: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    return {(str(p["op"]), int(p["a"]), int(p["b"])) for p in problems}


def forbidden_operand_keys(v1_plan_path: Path = V1_POOL_PLAN) -> set[tuple[str, int, int]]:
    used = operand_keys_from_problems(build_pilot_problems(SEED_PILOT))
    used |= operand_keys_from_problems(build_pilot2_problems(SEED_PILOT2))
    v1 = load_json(v1_plan_path)
    used |= operand_keys_from_problems(v1["problems"])
    return used


def build_v2_pool_plan(
    seed: int,
    chosen_levels: list[int],
    used: set[tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    if used is None:
        used = forbidden_operand_keys()
    used = set(used)
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in chosen_levels:
        for pool, n in POOL_SIZES.items():
            for i in range(n):
                while True:
                    cand = make_family_problem(
                        rng, FAMILY, int(level), f"v2-{pool}-L{level}-{i:03d}", pool
                    )
                    key = (str(cand["op"]), int(cand["a"]), int(cand["b"]))
                    if key not in used:
                        used.add(key)
                        problems.append(cand)
                        break
    return {
        "seed": seed,
        "family": FAMILY,
        "chosen_levels": list(chosen_levels),
        "n_problems": len(problems),
        "pool_sizes": dict(POOL_SIZES),
        "version": "m34a-v2",
        "prompt_template": PROMPT_TEMPLATE_A1,
        "problems": problems,
    }


def tercile_edges(values: list[float]) -> tuple[float, float]:
    """Inclusive-lower terciles: edges at 1/3 and 2/3 sample quantiles (type-7-ish via rank)."""
    if not values:
        raise ValueError("empty values for terciles")
    xs = sorted(float(v) for v in values)
    n = len(xs)

    def q(frac: float) -> float:
        if n == 1:
            return xs[0]
        idx = (n - 1) * frac
        lo = int(idx)
        hi = min(n - 1, lo + 1)
        w = idx - lo
        return xs[lo] * (1.0 - w) + xs[hi] * w

    return q(1.0 / 3.0), q(2.0 / 3.0)


def tercile_bin(lp: float, edges: tuple[float, float]) -> int:
    """0/1/2. Ties at an edge go to the lower bin (inclusive upper bound of the lower bin)."""
    e1, e2 = edges
    if lp <= e1:
        return 0
    if lp <= e2:
        return 1
    return 2


def mann_whitney_auroc(scores: list[float], labels: list[bool]) -> float | None:
    """AUROC of score for positive label; None if a class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    # rank-sum: count (pos>neg) + 0.5 (pos==neg)
    better = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                better += 1.0
            elif p == n:
                better += 0.5
    return better / (len(pos) * len(neg))
