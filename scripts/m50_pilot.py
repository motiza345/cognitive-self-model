"""M50 pilot decisions (offline): level choice per task, gates, sample size rule.

Reads reports/m50_pilot/responses.jsonl, writes reports/m50_pilot/levels.json and gates.json.
Protocol: docs/M50_PREREG.md, section 4. Commit both files before collecting the main data.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.m50_common import ROOT, choose_windows, compute_gates, scope_id
from scripts.m50_arms import arm_always_answer, arm_always_verify, run_layer_avg

from cognitive_layer import from_token_logprobs  # noqa: E402  (layer/ is added to sys.path by m50_arms)

PILOT_DIR = ROOT / "reports" / "m50_pilot"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pilot_accuracy(rows: list[dict]) -> dict[str, dict[int, float]]:
    acc: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["task"]][r["level"]].append(bool(r["correct"]))
    return {t: {lv: sum(v) / len(v) for lv, v in by.items()} for t, by in acc.items()}


def estimate_se(rows: list[dict], chosen: dict[str, dict]) -> float | None:
    """Standard error of (scoped layer minus the better of always-answer and always-verify), from running the
    layer on the pilot items of the chosen windows split 50/50 (first half learns, second half is scored)."""
    if len(chosen) < 2:
        return None
    scopes = [(t, lv) for t, c in chosen.items() for lv in c["levels"]]
    cal, test = [], []
    for task, level in scopes:
        sub = sorted((r for r in rows if r["task"] == task and r["level"] == level), key=lambda r: r["problem_id"])
        half = len(sub) // 2
        for part, target in ((sub[:half], cal), (sub[half:], test)):
            for r in part:
                target.append({"task": task, "level": level, "correct": bool(r["correct"]),
                               "conf": from_token_logprobs(r["token_logprobs"])})
    u_layer, _, _ = run_layer_avg(cal, test, scopes, [0])
    simple = {"answer": arm_always_answer(test), "verify": arm_always_verify(test)}
    best = max(simple.values(), key=lambda a: a.mean())
    diff = u_layer - best
    return float(np.std(diff, ddof=1) / np.sqrt(len(diff)))


def decide(rows: list[dict]) -> tuple[dict, dict]:
    acc = pilot_accuracy(rows)
    chosen = choose_windows(acc)
    se = estimate_se(rows, chosen)
    gates = compute_gates(chosen, rows, se)
    levels = {"chosen": chosen, "pilot_accuracy": {t: {str(k): v for k, v in by.items()} for t, by in acc.items()}}
    return levels, gates


def main() -> int:
    rows = load_rows(PILOT_DIR / "responses.jsonl")
    levels, gates = decide(rows)
    (PILOT_DIR / "levels.json").write_text(json.dumps(levels, indent=2), encoding="utf-8")
    (PILOT_DIR / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print("chosen windows:", {t: c["levels"] for t, c in levels["chosen"].items()})
    print("gates:", {k: v for k, v in gates.items() if k != "per_scope_pilot_accuracy"})
    print("ALL GATES PASS: collection allowed." if gates["all_pass"] else "A GATE FAILED: collection is forbidden (label NOT_INFORMATIVE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
