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
PROMPT_TEMPLATE = (
    "Compute {a} x {b}. Reply exactly in this format: "
    "Answer: <integer>; Confidence: <0-100 probability your answer is exactly correct>"
)
MAX_NEW_TOKENS = 60
SEED_PILOT = 34001
BOOTSTRAP_SEED = 23001
BOOTSTRAP_DRAWS = 5000
CANDIDATE_LEVELS = list(range(2, 9))
PILOT_PER_LEVEL = 20
POOL_SIZES = {"cal": 100, "hist": 60, "test": 40}
N_H_VALUES = (2, 5, 10, 25)
PRIMARY_N_H = 10
N_HISTORY_DRAWS = 30
CONF_BINS = ((0, 59), (60, 79), (80, 89), (90, 94), (95, 100))

ANSWER_RE = re.compile(r"Answer:\s*([^\n;]+)", re.IGNORECASE)
CONF_RE = re.compile(r"Confidence:\s*([^\n;]+)", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def n_digit_int(rng: random.Random, n: int) -> int:
    lo = 10 ** (n - 1)
    hi = 10**n - 1
    return rng.randint(lo, hi)


def make_problem(rng: random.Random, level: int, problem_id: str, pool: str) -> dict[str, Any]:
    a = n_digit_int(rng, level)
    b = n_digit_int(rng, level)
    return {
        "problem_id": problem_id,
        "pool": pool,
        "level": level,
        "a": a,
        "b": b,
        "product": a * b,
        "prompt": PROMPT_TEMPLATE.format(a=a, b=b),
    }


def parse_response(text: str) -> tuple[int | None, float]:
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


def build_pilot_problems(seed: int = SEED_PILOT) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    problems: list[dict[str, Any]] = []
    for level in CANDIDATE_LEVELS:
        for i in range(PILOT_PER_LEVEL):
            problems.append(make_problem(rng, level, f"pilot-L{level}-{i:02d}", "pilot"))
    return problems


def build_pool_plan(seed: int, chosen_levels: list[int]) -> dict[str, Any]:
    """Deterministic pools disjoint from pilot operands; continues the pilot RNG stream."""
    rng = random.Random(seed)
    pilot = build_pilot_problems(seed)
    # Rebuild with same seed so the stream matches pilot generation, then continue.
    rng = random.Random(seed)
    for _ in pilot:
        n_digit_int(rng, _["level"])
        n_digit_int(rng, _["level"])
    used = {(p["a"], p["b"]) for p in pilot}
    problems: list[dict[str, Any]] = []
    for level in chosen_levels:
        for pool, n in POOL_SIZES.items():
            for i in range(n):
                while True:
                    a = n_digit_int(rng, level)
                    b = n_digit_int(rng, level)
                    if (a, b) not in used:
                        used.add((a, b))
                        break
                pid = f"{pool}-L{level}-{i:03d}"
                problems.append(
                    {
                        "problem_id": pid,
                        "pool": pool,
                        "level": level,
                        "a": a,
                        "b": b,
                        "product": a * b,
                        "prompt": PROMPT_TEMPLATE.format(a=a, b=b),
                    }
                )
    return {
        "seed": seed,
        "chosen_levels": list(chosen_levels),
        "n_problems": len(problems),
        "pool_sizes": dict(POOL_SIZES),
        "pilot_ids": [p["problem_id"] for p in pilot],
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
    """Return a derangement mapping item -> image; retries until none fixed."""
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
    """variant V3: answer/abstain/verify; V2: answer/abstain. Ties: answer then verify."""
    eus = eu_actions(p)
    if variant == "V2":
        candidates = ["answer", "abstain"]
    elif variant == "V3":
        candidates = ["answer", "abstain", "verify"]
    else:
        raise ValueError(variant)
    best = max(eus[c] for c in candidates)
    # tie order: answer, then verify, then abstain
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
