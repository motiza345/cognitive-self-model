"""S2 pilot for Phi mul_n1: 20/level, seed 34005, gap_hat selection. No pool analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import write_json  # noqa: E402
from scripts.m34a_v3_common import (  # noqa: E402
    PHI_MODEL_ID,
    SEED_S2_PILOT,
    build_s2_pilot_problems,
    select_gap_window,
)
from scripts.m34a_v3_generate import make_client, run_problem  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "m34a_v3_s2_pilot")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=PHI_MODEL_ID)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument(
        "--revision-pin",
        type=Path,
        default=ROOT / "reports" / "m34a_v3_s2" / "revision.json",
    )
    args = parser.parse_args(argv)

    revision = args.revision
    if revision is None and not args.mock:
        if not args.revision_pin.exists():
            raise SystemExit("pin S2 revision first: python -m scripts.m34a_v3_pin_s2_revision")
        revision = json.loads(args.revision_pin.read_text(encoding="utf-8"))["revision"]

    problems = build_s2_pilot_problems(SEED_S2_PILOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "responses.jsonl"

    try:
        client: Any = make_client(mock=args.mock, model_id=args.model, revision=revision)
    except Exception as exc:  # noqa: BLE001
        write_json(
            args.out_dir / "NOT_RUN.json",
            {"reason": "model_load_failed", "error": repr(exc), "model_id": args.model},
        )
        print(f"NOT_RUN: {exc}")
        return 3

    done = set()
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["problem_id"])

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    try:
        with cache.open("a", encoding="utf-8") as f:
            for problem in problems:
                if problem["problem_id"] in done:
                    continue
                row = run_problem(client, problem)
                row["model_id"] = getattr(client, "model_id", args.model)
                row["revision"] = getattr(client, "revision", revision)
                f.write(json.dumps(row, sort_keys=True) + "\n")
                rows.append(row)
    except Exception as exc:  # noqa: BLE001
        write_json(
            args.out_dir / "NOT_RUN.json",
            {"reason": "logprob_or_runtime_failure", "error": repr(exc)},
        )
        print(f"NOT_RUN: {exc}")
        return 3

    all_rows = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            all_rows.append(json.loads(line))

    by: dict[int, list[bool]] = defaultdict(list)
    for r in all_rows:
        by[int(r["level"])].append(bool(r["correct"]))
    acc = {L: sum(v) / len(v) for L, v in sorted(by.items())}
    choice = select_gap_window(
        acc,
        "mul_n1",
        "S2 Phi pilot seed 34005 accuracies",
    )
    doc = {
        "version": "m34a-v3-s2-pilot",
        "family": "mul_n1",
        "seed": SEED_S2_PILOT,
        "model_id": getattr(client, "model_id", args.model),
        "revision": getattr(client, "revision", revision),
        "device_name": getattr(client, "device_name", "cpu"),
        "dtype": getattr(client, "dtype_name", "float32"),
        "n_calls": len(all_rows),
        "runtime_seconds": time.perf_counter() - t0,
        "git_commit": git_commit(),
        "family_accuracies": {str(k): v for k, v in acc.items()},
        "choice": choice,
        "chosen_levels": choice["chosen_levels"],
        "status": "STOP_GAP_LT_0.05" if choice["stop"] else "LEVELS_CHOSEN",
    }
    write_json(args.out_dir / "levels.json", doc)
    stale = args.out_dir / "NOT_RUN.json"
    if stale.exists():
        stale.unlink()
    print(json.dumps({"status": doc["status"], "chosen_levels": doc["chosen_levels"], "best_gap": choice["best"]["gap_hat"]}, indent=2))
    return 2 if choice["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
