"""Shared helpers for M34a (Qwen local generation + offline analysis contracts)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

PRIMARY_MODEL = "Qwen/Qwen2.5-3B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Historical pilot-1 prompt (mul n x n with confidence). Kept for regenerating pilot-1 exclusions.
PROMPT_TEMPLATE_P1 = (
    "Compute {a} x {b}. Reply exactly in this format: "
    "Answer: <integer>; Confidence: <0-100 probability your answer is exactly correct>"
)
# Amendment 1 prompt
PROMPT_TEMPLATE_A1 = "Compute {a} {op} {b}. Reply with only the integer."

MAX_NEW_TOKENS = 60
SEED_PILOT = 34001  # pilot 1 + pool plan seed
SEED_PILOT2 = 34002
BOOTSTRAP_SEED = 23001
BOOTSTRAP_DRAWS = 5000
CANDIDATE_LEVELS = list(range(2, 9))  # pilot 1
PILOT_PER_LEVEL = 20
POOL_SIZES = {"cal": 100, "hist": 60, "test": 40}
N_H_VALUES = (2, 5, 10, 25)
PRIMARY_N_H = 10
N_HISTORY_DRAWS = 30
CONF_BINS = ((0, 59), (60, 79), (80, 89), (90, 94), (95, 100))

# Amendment 1 families: id -> (op_symbol, levels, b_digits or "same")
FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "add_nn": {"op": "+", "levels": list(range(3, 16)), "b_digits": "same"},
    "mul_n2": {"op": "x", "levels": list(range(2, 9)), "b_digits": 2},
    "mul_n1": {"op": "x", "levels": list(range(2, 11)), "b_digits": 1},
}
FAMILY_ORDER = ("add_nn", "mul_n2", "mul_n1")

ANSWER_RE = re.compile(r"Answer:\s*([^\n;]+)", re.IGNORECASE)
CONF_RE = re.compile(r"Confidence:\s*([^\n;]+)", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def n_digit_int(rng: random.Random, n: int) -> int:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    return rng.randint(lo, hi)


def make_problem(rng: random.Random, level: int, problem_id: str, pool: str) -> dict[str, Any]:
    """Pilot-1 style: n-digit x n-digit with confidence prompt."""
    a = n_digit_int(rng, level)
    b = n_digit_int(rng, level)
    target = a * b
    return {
        "problem_id": problem_id,
        "pool": pool,
        "family": "mul_nn_p1",
        "level": level,
        "a": a,
        "b": b,
        "op": "x",
        "target": target,
        "product": target,
        "prompt": PROMPT_TEMPLATE_P1.format(a=a, b=b),
    }


def make_family_problem(
    rng: random.Random,
    family: str,
    level: int,
    problem_id: str,
    pool: str,
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    op = spec["op"]
    a = n_digit_int(rng, level)
    if spec["b_digits"] == "same":
        b = n_digit_int(rng, level)
    else:
        b = n_digit_int(rng, int(spec["b_digits"]))
    target = a + b if op == "+" else a * b
    return {
        "problem_id": problem_id,
        "pool": pool,
        "family": family,
        "level": level,
        "a": a,
        "b": b,
        "op": op,
        "target": target,
        "product": target,
        "prompt": PROMPT_TEMPLATE_A1.format(a=a, op=op, b=b),
    }


def parse_response(text: str) -> tuple[int | None, float]:
    """Answer from Answer: field if present, else first integer; confidence default 50."""
    ans: int | None = None
    am = ANSWER_RE.search(text or "")
    if am:
        raw = am.group(1).replace(",", "").strip()
        im = INT_RE.search(raw)
        if im:
            try:
                ans = int(im.group(0))
            except ValueError:
                ans = None
    if ans is None:
        im2 = INT_RE.search((text or "").replace(",", ""))
        if im2:
            try:
                ans = int(im2.group(0))
            except ValueError:
                ans = None
    conf = 50.0
    cm = CONF_RE.search(text or "")
    if cm:
        raw_c = cm.group(1).replace("%", "").strip()
        try:
            conf_v = float(raw_c)
            if conf_v == conf_v and 0.0 <= conf_v <= 100.0:
                conf = conf_v
        except ValueError:
            conf = 50.0
    return ans, conf


def choose_levels(accuracies: dict[int, float]) -> dict[str, Any]:
    """Pilot-1 rule (historical)."""
    levels = sorted(accuracies)
    if len(levels) < 6:
        raise ValueError("need at least 6 candidate levels")
    best: tuple[float, list[int]] | None = None
    for start_idx in range(len(levels) - 5):
        window = levels[start_idx : start_idx + 6]
        vals = [accuracies[L] for L in window]
        span = max(vals) - min(vals)
        if best is None or span > best[0]:
            best = (span, list(window))
    assert best is not None
    return {
        "chosen_levels": best[1],
        "accuracy_range": best[0],
        "accuracies": {str(k): float(accuracies[k]) for k in levels},
        "stop": best[0] < 0.5,
        "rule": "6 contiguous maximizing (max-min); ties -> lower levels; STOP if range < 0.5",
    }


def select_pilot2_window(
    family_accuracies: dict[str, dict[int, float]],
) -> dict[str, Any]:
    """Amendment 1 selection: pure function.

    score = #levels in a 6-contiguous window with acc in [0.15, 0.85].
    Max score; ties -> larger (max-min); then family order add_nn, mul_n2, mul_n1;
    then lower levels. STOP if best score < 4.
    """
    # Each row: (score, span, fam_rank, start_level, family, window, vals)
    candidates: list[tuple[int, float, int, int, str, list[int], list[float]]] = []
    for fam in FAMILY_ORDER:
        if fam not in family_accuracies:
            continue
        acc = family_accuracies[fam]
        levels = sorted(acc)
        if len(levels) < 6:
            continue
        fam_rank = FAMILY_ORDER.index(fam)
        for start_idx in range(len(levels) - 5):
            window = levels[start_idx : start_idx + 6]
            vals = [float(acc[L]) for L in window]
            score = sum(1 for v in vals if 0.15 <= v <= 0.85)
            span = max(vals) - min(vals)
            candidates.append((score, span, fam_rank, window[0], fam, list(window), vals))

    if not candidates:
        return {
            "stop": True,
            "best_score": 0,
            "chosen_family": None,
            "chosen_levels": None,
            "rule": "Amendment 1 select_pilot2_window",
            "reason": "no_windows",
        }

    # Maximize score, then span, then earlier family, then lower start level
    candidates.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
    score, span, _fr, _start, fam, window, vals = candidates[0]
    stop = score < 4
    return {
        "stop": stop,
        "best_score": score,
        "accuracy_range": span,
        "chosen_family": fam,
        "chosen_levels": None if stop else window,
        "window_accuracies": {str(L): float(a) for L, a in zip(window, vals)},
        "family_accuracies": {
            f: {str(k): float(v) for k, v in sorted(family_accuracies[f].items())}
            for f in FAMILY_ORDER
            if f in family_accuracies
        },
        "rule": (
            "score=#acc in [0.15,0.85] over 6 contiguous; max score; "
            "ties->larger (max-min), then add_nn/mul_n2/mul_n1, then lower levels; STOP if score<4"
        ),
    }


def build_pilot_problems(seed: int = SEED_PILOT) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in CANDIDATE_LEVELS:
        for i in range(PILOT_PER_LEVEL):
            problems.append(make_problem(rng, level, f"pilot-L{level}-{i:02d}", "pilot"))
    return problems


def build_pilot2_problems(seed: int = SEED_PILOT2) -> list[dict[str, Any]]:
    """All Amendment 1 family/level cells; disjointness from pilot 1 enforced via used set."""
    used = {(p["op"], p["a"], p["b"]) for p in build_pilot_problems(SEED_PILOT)}
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        for level in FAMILY_SPECS[family]["levels"]:
            for i in range(PILOT_PER_LEVEL):
                while True:
                    cand = make_family_problem(
                        rng, family, level, f"pilot2-{family}-L{level}-{i:02d}", "pilot2"
                    )
                    key = (cand["op"], cand["a"], cand["b"])
                    if key not in used:
                        used.add(key)
                        problems.append(cand)
                        break
    return problems


def build_pool_plan(
    seed: int,
    chosen_levels: list[int],
    family: str,
) -> dict[str, Any]:
    """Pools for Amendment 1 family; disjoint from pilots 1 and 2."""
    if family not in FAMILY_SPECS:
        raise ValueError(f"unknown family {family}")
    used = {(p["op"], p["a"], p["b"]) for p in build_pilot_problems(SEED_PILOT)}
    used |= {(p["op"], p["a"], p["b"]) for p in build_pilot2_problems(SEED_PILOT2)}
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in chosen_levels:
        for pool, n in POOL_SIZES.items():
            for i in range(n):
                while True:
                    cand = make_family_problem(
                        rng, family, int(level), f"{pool}-L{level}-{i:03d}", pool
                    )
                    key = (cand["op"], cand["a"], cand["b"])
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
        "amendment": 1,
        "problems": problems,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def conf_bin(conf: float) -> tuple[int, int]:
    c = float(conf)
    for lo, hi in CONF_BINS:
        if lo <= c <= hi:
            return (lo, hi)
    return CONF_BINS[0]


def random_derangement(rng: random.Random, items: list[Any]) -> dict[Any, Any]:
    n = len(items)
    if n < 2:
        raise ValueError("derangement needs >= 2 items")
    while True:
        perm = list(items)
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return {a: b for a, b in zip(items, perm)}


def eu_actions(p: float) -> dict[str, float]:
    return {"answer": 2.0 * p - 1.0, "abstain": 0.0, "verify": 0.7}


def choose_action(p: float, variant: str) -> str:
    eus = eu_actions(p)
    if variant == "V2":
        candidates = ["answer", "abstain"]
    elif variant == "V3":
        candidates = ["answer", "abstain", "verify"]
    else:
        raise ValueError(variant)
    best = max(eus[c] for c in candidates)
    for name in ("answer", "verify", "abstain"):
        if name in candidates and eus[name] == best:
            return name
    raise RuntimeError("unreachable")


def realized_utility(action: str, correct: bool) -> float:
    if action == "abstain":
        return 0.0
    if action == "verify":
        return 0.7
    if action == "answer":
        return 1.0 if correct else -1.0
    raise ValueError(action)


def paired_mean_ci(
    deltas: list[float], draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED
) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(deltas, dtype=float)
    n = len(arr)
    if n == 0:
        raise ValueError("empty deltas")
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(draws, dtype=float)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(arr[idx].mean())
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"mean": mean, "low": float(low), "high": float(high), "n": n, "draws": draws, "seed": seed}
