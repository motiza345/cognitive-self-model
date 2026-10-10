"""M34a-v3 helpers: parameterized level selection and pool plans. Reuses v2 gap_hat rule."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from scripts.m34a_common import (
    PILOT_PER_LEVEL,
    POOL_SIZES,
    PROMPT_TEMPLATE_A1,
    SEED_PILOT,
    SEED_PILOT2,
    build_pilot2_problems,
    build_pilot_problems,
    load_json,
    make_family_problem,
    write_json,
)
from scripts.m34a_v2_common import (
    GAP_FLOOR,
    WINDOW,
    eu_star,
    operand_keys_from_problems,
    select_v2_window,
)

QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
QWEN_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PHI_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

SEED_S1_POOLS = 34004
SEED_S2_PILOT = 34005
SEED_S2_POOLS = 34006

PILOT2_LEVELS = Path("reports/m34a_pilot2/levels.json")
V1_POOL_PLAN = Path("reports/m34a_pilot2/pool_plan.json")
V2_POOL_PLAN = Path("reports/m34a_v2/pool_plan.json")
S1_POOL_PLAN = Path("reports/m34a_v3_s1/pool_plan.json")
S2_PILOT_LEVELS = Path("reports/m34a_v3_s2_pilot/levels.json")
S2_POOL_PLAN = Path("reports/m34a_v3_s2/pool_plan.json")

C_LABELS = ("INTERNAL_VALUE", "INTERNAL_NONE", "NOT_INFORMATIVE")


def select_gap_window(acc: dict[int, float], family: str, source: str) -> dict[str, Any]:
    """Same rule as v2 select_v2_window; annotate family/source."""
    choice = select_v2_window(acc)
    choice["family"] = family
    choice["source"] = source
    return choice


def load_pilot2_family(family: str, path: Path = PILOT2_LEVELS) -> dict[int, float]:
    doc = load_json(path)
    raw = doc["choice"]["family_accuracies"][family]
    return {int(k): float(v) for k, v in raw.items()}


def forbidden_operand_keys_s1() -> set[tuple[str, int, int]]:
    used = operand_keys_from_problems(build_pilot_problems(SEED_PILOT))
    used |= operand_keys_from_problems(build_pilot2_problems(SEED_PILOT2))
    used |= operand_keys_from_problems(load_json(V1_POOL_PLAN)["problems"])
    used |= operand_keys_from_problems(load_json(V2_POOL_PLAN)["problems"])
    return used


def forbidden_operand_keys_s2_pilot() -> set[tuple[str, int, int]]:
    used = forbidden_operand_keys_s1()
    if S1_POOL_PLAN.exists():
        used |= operand_keys_from_problems(load_json(S1_POOL_PLAN)["problems"])
    return used


def forbidden_operand_keys_s2_pools() -> set[tuple[str, int, int]]:
    used = forbidden_operand_keys_s2_pilot()
    used |= operand_keys_from_problems(build_s2_pilot_problems(SEED_S2_PILOT))
    return used


def build_pool_plan(
    seed: int,
    family: str,
    chosen_levels: list[int],
    used: set[tuple[str, int, int]],
    *,
    version: str,
    id_prefix: str,
) -> dict[str, Any]:
    used = set(used)
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in chosen_levels:
        for pool, n in POOL_SIZES.items():
            for i in range(n):
                while True:
                    cand = make_family_problem(
                        rng, family, int(level), f"{id_prefix}-{pool}-L{level}-{i:03d}", pool
                    )
                    key = (str(cand["op"]), int(cand["a"]), int(cand["b"]))
                    if key not in used:
                        used.add(key)
                        problems.append(cand)
                        break
    return {
        "seed": seed,
        "family": family,
        "chosen_levels": list(chosen_levels),
        "n_problems": len(problems),
        "pool_sizes": dict(POOL_SIZES),
        "version": version,
        "prompt_template": PROMPT_TEMPLATE_A1,
        "problems": problems,
    }


def build_s1_pool_plan(seed: int = SEED_S1_POOLS) -> dict[str, Any]:
    acc = load_pilot2_family("add_nn")
    choice = select_gap_window(
        acc,
        "add_nn",
        "reports/m34a_pilot2/levels.json choice.family_accuracies.add_nn only",
    )
    if choice["stop"] or not choice["chosen_levels"]:
        raise RuntimeError("S1 STOP: gap_hat < 0.05")
    return build_pool_plan(
        seed,
        "add_nn",
        [int(x) for x in choice["chosen_levels"]],
        forbidden_operand_keys_s1(),
        version="m34a-v3-s1",
        id_prefix="v3s1",
    )


def build_s2_pilot_problems(seed: int = SEED_S2_PILOT) -> list[dict[str, Any]]:
    used = forbidden_operand_keys_s2_pilot()
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in range(2, 11):
        for i in range(PILOT_PER_LEVEL):
            while True:
                cand = make_family_problem(
                    rng, "mul_n1", level, f"v3s2-pilot-L{level}-{i:02d}", "pilot"
                )
                key = (str(cand["op"]), int(cand["a"]), int(cand["b"]))
                if key not in used:
                    used.add(key)
                    problems.append(cand)
                    break
    return problems


def build_s2_pool_plan(
    chosen_levels: list[int], seed: int = SEED_S2_POOLS
) -> dict[str, Any]:
    return build_pool_plan(
        seed,
        "mul_n1",
        list(chosen_levels),
        forbidden_operand_keys_s2_pools(),
        version="m34a-v3-s2",
        id_prefix="v3s2",
    )


def aggregate_c(c_s1: str, c_s2: str) -> str:
    """Exhaustive aggregate over S1×S2 C labels. NOT_RUN counts as NOT_INFORMATIVE."""
    if c_s1 == "NOT_RUN":
        c_s1 = "NOT_INFORMATIVE"
    if c_s2 == "NOT_RUN":
        c_s2 = "NOT_INFORMATIVE"
    if c_s1 not in C_LABELS or c_s2 not in C_LABELS:
        raise ValueError(f"bad C labels: {c_s1}, {c_s2}")
    vals = {c_s1, c_s2}
    if vals == {"INTERNAL_VALUE"}:
        return "C_GENERAL"
    if vals == {"INTERNAL_VALUE", "INTERNAL_NONE"}:
        return "C_MIXED"
    if vals == {"INTERNAL_VALUE", "NOT_INFORMATIVE"}:
        return "C_PARTIAL"
    if "INTERNAL_VALUE" not in vals and "INTERNAL_NONE" in vals:
        return "C_NOT_REPLICATED"
    if vals == {"NOT_INFORMATIVE"}:
        return "C_NOT_INFORMATIVE"
    raise RuntimeError(f"unmapped aggregate: {c_s1}, {c_s2}")


def write_s1_selection_and_plan(out_dir: Path) -> dict[str, Any]:
    acc = load_pilot2_family("add_nn")
    choice = select_gap_window(
        acc,
        "add_nn",
        "reports/m34a_pilot2/levels.json choice.family_accuracies.add_nn only",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    levels_doc = {
        "version": "m34a-v3-s1",
        "family": "add_nn",
        "pilot2_add_nn": {str(k): v for k, v in sorted(acc.items())},
        "choice": choice,
        "chosen_levels": choice["chosen_levels"],
        "status": "STOP_GAP_LT_0.05" if choice["stop"] else "LEVELS_CHOSEN",
        "model_id": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
    }
    write_json(out_dir / "levels.json", levels_doc)
    if choice["stop"]:
        return levels_doc
    plan = build_s1_pool_plan(SEED_S1_POOLS)
    write_json(out_dir / "pool_plan.json", plan)
    return {"levels": levels_doc, "plan_n": plan["n_problems"], "chosen": plan["chosen_levels"]}
