"""M34a STEP 0 pilot (Colab): 20 problems per level 2..8; write levels.json."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import (  # noqa: E402
    CANDIDATE_LEVELS,
    SEED_PILOT,
    build_pilot_problems,
    choose_levels,
    write_json,
)
from scripts.m34a_generate import MockQwenClient, TransformersQwenClient, run_problem  # noqa: E402

OUT_DIR = ROOT / "reports" / "m34a_pilot"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M34a pilot")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED_PILOT)
    parser.add_argument("--model", type=str, default=None, help="Override model repo id")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    client: Any
    if args.mock:
        client = MockQwenClient()
    else:
        client = TransformersQwenClient(model_id=args.model)

    problems = build_pilot_problems(args.seed)
    records = []
    for i, problem in enumerate(problems):
        rec = run_problem(client, problem)
        records.append(rec)
        if (i + 1) % 20 == 0:
            print(f"pilot {i + 1}/{len(problems)}", flush=True)

    accuracies = {
        level: sum(1 for r in records if r["level"] == level and r["correct"])
        / sum(1 for r in records if r["level"] == level)
        for level in CANDIDATE_LEVELS
    }
    choice = choose_levels(accuracies)
    runtime = time.perf_counter() - t0
    payload = {
        "seed": args.seed,
        "model_id": getattr(client, "model_id", "mock"),
        "revision": getattr(client, "revision", "mock"),
        "device_name": getattr(client, "device_name", "cpu"),
        "dtype": getattr(client, "dtype_name", "float32"),
        "n_calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "runtime_seconds": runtime,
        "per_level_accuracy": {str(k): accuracies[k] for k in CANDIDATE_LEVELS},
        "choice": choice,
        "chosen_levels": None if choice["stop"] else choice["chosen_levels"],
        "status": "STOP_RANGE_TOO_SMALL" if choice["stop"] else "LEVELS_CHOSEN",
        "float32_rerun_ids": [r["problem_id"] for r in records if r.get("float32_rerun")],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "levels.json", payload)
    write_json(OUT_DIR / "pilot_raw.json", records)
    print(json.dumps({"status": payload["status"], "choice": choice}, indent=2))
    return 2 if choice["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
