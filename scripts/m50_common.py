"""M50 shared definitions: tasks, problem generation, parsing, pool plans, level choice and gates.

Pure Python (numpy only). No model code here, so everything is testable offline.
Protocol: docs/M50_PREREG.md. Constants below are the fixed parameters of that file.
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
HIDDEN_LAYER = 18          # HF hidden_states index (output of the 18th block)
MAX_NEW_TOKENS = 64

TASK_LEVELS = {"mul_n1": list(range(2, 11)), "add_nn": list(range(3, 16)), "sort_n": list(range(3, 13))}
SEED_POOLS, SEED_PILOT, SEED_CORRUPT, SEED_RANDOM_VERIFY, SEED_BOOT = 50001, 50002, 50003, 50004, 23001
WINDOW = 4
PILOT_PER_LEVEL = 40
CAL_PER_SCOPE = 100
TEST_PER_SCOPE_DEFAULT = 40
TEST_PER_SCOPE_LARGE = 80
VERIFY_VALUE = 0.7
CORRUPTION_PROBABILITY = 0.3
CORRUPTED_TASK = "sort_n"


# ----------------------------------------------------------------------------- problems
def scope_id(task: str, level: int) -> str:
    return f"{task}:{level}"


def make_prompt(task: str, payload) -> str:
    if task == "mul_n1":
        return f"Compute {payload[0]} x {payload[1]}. Reply with only the integer."
    if task == "add_nn":
        return f"Compute {payload[0]} + {payload[1]}. Reply with only the integer."
    if task == "sort_n":
        return "Sort ascending: " + ", ".join(str(x) for x in payload) + ". Reply with only the sorted numbers separated by commas."
    raise ValueError(task)


def truth_of(task: str, payload):
    if task == "mul_n1":
        return int(payload[0]) * int(payload[1])
    if task == "add_nn":
        return int(payload[0]) + int(payload[1])
    if task == "sort_n":
        return sorted(int(x) for x in payload)
    raise ValueError(task)


def parse_answer(task: str, text: str):
    """Integer tasks: first integer after stripping commas and spaces. Sort: all integers in order."""
    if task in ("mul_n1", "add_nn"):
        match = re.search(r"-?\d+", text.replace(",", "").replace(" ", ""))
        return int(match.group()) if match else None
    if task == "sort_n":
        numbers = re.findall(r"\d+", text)
        return [int(x) for x in numbers] if numbers else None
    raise ValueError(task)


def is_correct(task: str, payload, text: str) -> bool:
    parsed = parse_answer(task, text)
    return parsed is not None and parsed == truth_of(task, payload)


def payload_key(task: str, payload) -> tuple:
    return (task, tuple(int(x) for x in payload))


def _draw(task: str, level: int, rng: random.Random):
    if task == "mul_n1":
        return [rng.randint(10 ** (level - 1), 10 ** level - 1), rng.randint(2, 9)]
    if task == "add_nn":
        return [rng.randint(10 ** (level - 1), 10 ** level - 1), rng.randint(10 ** (level - 1), 10 ** level - 1)]
    if task == "sort_n":
        return rng.sample(range(10, 100), level)
    raise ValueError(task)


def make_pool(task: str, level: int, pool: str, count: int, seed: int, exclude: set) -> list[dict]:
    """Deterministic, unique problems for (task, level, pool), disjoint from `exclude` (a set of payload keys
    and of operand pairs)."""
    rng = random.Random(f"{seed}:{task}:{level}:{pool}")
    out, seen, guard = [], set(), 0
    while len(out) < count:
        guard += 1
        if guard > 100000:
            raise RuntimeError("could not draw enough distinct problems")
        payload = _draw(task, level, rng)
        key = payload_key(task, payload)
        pair = tuple(int(x) for x in payload) if task != "sort_n" else None
        if key in seen or key in exclude or (pair is not None and pair in exclude):
            continue
        seen.add(key)
        out.append({
            "problem_id": f"{task}-{level}-{pool}-{len(out):04d}",
            "task": task, "level": level, "pool": pool, "scope": scope_id(task, level),
            "payload": payload, "prompt": make_prompt(task, payload),
        })
    return out


def earlier_exclusions(root: Path = ROOT) -> set:
    """Operand pairs and payload keys of all earlier pools in the repository (M34a pilots, pools)."""
    exclude: set = set()
    patterns = ["reports/m34a_*/responses.jsonl", "reports/m34a_*raw/responses.jsonl"]
    for pattern in patterns:
        for path in glob.glob(str(root / pattern)):
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "a" in row and "b" in row:
                    exclude.add((int(row["a"]), int(row["b"])))
    return exclude


# ----------------------------------------------------------------------------- pilot decisions
def gap_hat(accuracies: list[float]) -> float:
    pbar = sum(accuracies) / len(accuracies)
    mean_levels = sum(max(2 * p - 1, VERIFY_VALUE) for p in accuracies) / len(accuracies)
    return mean_levels - max(2 * pbar - 1, VERIFY_VALUE)


def choose_windows(pilot_accuracy: dict[str, dict[int, float]], window: int = WINDOW, min_mid: int = 2) -> dict[str, dict]:
    """Per task, the window of `window` contiguous levels with maximal gap_hat subject to at least `min_mid`
    levels with accuracy in [0.25, 0.75]; ties go to lower levels. Tasks without a qualifying window are dropped."""
    chosen: dict[str, dict] = {}
    for task, by_level in pilot_accuracy.items():
        levels = sorted(by_level)
        best = None
        for start in levels:
            win = list(range(start, start + window))
            if any(level not in by_level for level in win):
                continue
            accs = [by_level[level] for level in win]
            mid = sum(1 for a in accs if 0.25 <= a <= 0.75)
            if mid < min_mid:
                continue
            score = gap_hat(accs)
            if best is None or score > best[0] + 1e-12:
                best = (score, win, accs)
        if best is not None:
            chosen[task] = {"levels": best[1], "gap_hat": best[0], "pilot_accuracy": best[2]}
    return chosen


def compute_gates(chosen: dict[str, dict], pilot_items: list[dict], se_estimate: float | None) -> dict:
    """G-ceiling, G-variance and the sample size rule, on the pilot items of the chosen windows."""
    scopes = {scope_id(t, lv) for t, c in chosen.items() for lv in c["levels"]}
    items = [r for r in pilot_items if scope_id(r["task"], r["level"]) in scopes]
    accuracy = sum(1 for r in items if r["correct"]) / len(items) if items else 0.0
    oracle = sum(1.0 if r["correct"] else VERIFY_VALUE for r in items) / len(items) if items else 0.0
    ceiling = oracle - VERIFY_VALUE
    per_scope = {}
    for s in sorted(scopes):
        sub = [r for r in items if scope_id(r["task"], r["level"]) == s]
        per_scope[s] = sum(1 for r in sub if r["correct"]) / len(sub) if sub else None
    n_mid = sum(1 for a in per_scope.values() if a is not None and 0.25 <= a <= 0.75)
    n_tasks = len(chosen)
    test_per_scope = TEST_PER_SCOPE_LARGE if (se_estimate is not None and se_estimate > 0.015) else TEST_PER_SCOPE_DEFAULT
    gates = {
        "n_tasks_kept": n_tasks,
        "tasks_ok": n_tasks >= 2,
        "pilot_accuracy_overall": accuracy,
        "oracle_minus_always_verify": ceiling,
        "G_ceiling": bool(ceiling >= 0.15),
        "n_scope_levels_mid_accuracy": n_mid,
        "G_variance": bool(n_mid >= 5),
        "se_estimate_scoped_minus_best_simple": se_estimate,
        "test_per_scope_level": test_per_scope,
        "per_scope_pilot_accuracy": per_scope,
    }
    gates["all_pass"] = bool(gates["tasks_ok"] and gates["G_ceiling"] and gates["G_variance"])
    return gates


# ----------------------------------------------------------------------------- misc
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path = ROOT) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()


IGNORED_UNTRACKED_PREFIXES = ("reports/m50_pilot/", "reports/m50_raw/")


def git_dirty(root: Path = ROOT, ignore_untracked: tuple = IGNORED_UNTRACKED_PREFIXES) -> bool:
    """True when tracked files are modified, or untracked files exist outside the M50 output folders."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True,
                         text=True).stdout.splitlines()
    for line in out:
        status, path = line[:2], line[3:].strip().strip('"')
        if status == "??" and any(path.startswith(prefix) for prefix in ignore_untracked):
            continue
        return True
    return False


def add_layer_to_path() -> None:
    layer = str(ROOT / "layer")
    if layer not in sys.path:
        sys.path.insert(0, layer)
