"""M34a pilots: pilot1 (historical) or pilot2 (Amendment 1 families)."""

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
    FAMILY_ORDER,
    FAMILY_SPECS,
    SEED_PILOT,
    SEED_PILOT2,
    build_pilot2_problems,
    build_pilot_problems,
    choose_levels,
    select_pilot2_window,
    write_json,
)
from scripts.m34a_generate import MockQwenClient, TransformersQwenClient, run_problem  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M34a pilot")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--pilot",
        choices=("1", "2"),
        default="2",
        help="1=historical mul n x n; 2=Amendment 1 families (default)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Optional: run only one Amendment 1 family during pilot 2",
    )
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    client: Any
    if args.mock:
        client = MockQwenClient()
    else:
        client = TransformersQwenClient(model_id=args.model)

    if args.pilot == "1":
        seed = SEED_PILOT if args.seed is None else args.seed
        problems = build_pilot_problems(seed)
        out_dir = ROOT / "reports" / "m34a_pilot"
        records = []
        for i, problem in enumerate(problems):
            records.append(run_problem(client, problem))
            if (i + 1) % 20 == 0:
                print(f"pilot1 {i + 1}/{len(problems)}", flush=True)
        accuracies = {
            level: sum(1 for r in records if r["level"] == level and r["correct"])
            / sum(1 for r in records if r["level"] == level)
            for level in CANDIDATE_LEVELS
        }
        choice = choose_levels(accuracies)
        payload = {
            "pilot": 1,
            "seed": seed,
            "model_id": getattr(client, "model_id", "mock"),
            "revision": getattr(client, "revision", "mock"),
            "device_name": getattr(client, "device_name", "cpu"),
            "dtype": getattr(client, "dtype_name", "float32"),
            "n_calls": len(records),
            "input_tokens": sum(r["input_tokens"] for r in records),
            "output_tokens": sum(r["output_tokens"] for r in records),
            "runtime_seconds": time.perf_counter() - t0,
            "per_level_accuracy": {str(k): accuracies[k] for k in CANDIDATE_LEVELS},
            "choice": choice,
            "chosen_levels": None if choice["stop"] else choice["chosen_levels"],
            "status": "STOP_RANGE_TOO_SMALL" if choice["stop"] else "LEVELS_CHOSEN",
            "float32_rerun_ids": [r["problem_id"] for r in records if r.get("float32_rerun")],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "levels.json", payload)
        write_json(out_dir / "pilot_raw.json", records)
        print(json.dumps({"status": payload["status"], "choice": choice}, indent=2))
        return 2 if choice["stop"] else 0

    # Pilot 2
    seed = SEED_PILOT2 if args.seed is None else args.seed
    problems = build_pilot2_problems(seed)
    if args.family:
        if args.family not in FAMILY_SPECS:
            raise SystemExit(f"unknown family {args.family}")
        problems = [p for p in problems if p["family"] == args.family]
    out_dir = ROOT / "reports" / "m34a_pilot2"
    records = []
    for i, problem in enumerate(problems):
        records.append(run_problem(client, problem))
        if (i + 1) % 20 == 0:
            print(f"pilot2 {i + 1}/{len(problems)}", flush=True)

    family_accuracies: dict[str, dict[int, float]] = {}
    for fam in FAMILY_ORDER:
        fam_rows = [r for r in records if r.get("family") == fam]
        if not fam_rows:
            continue
        levels = sorted({int(r["level"]) for r in fam_rows})
        family_accuracies[fam] = {
            level: sum(1 for r in fam_rows if r["level"] == level and r["correct"])
            / sum(1 for r in fam_rows if r["level"] == level)
            for level in levels
        }
    choice = select_pilot2_window(family_accuracies)
    payload = {
        "pilot": 2,
        "amendment": 1,
        "seed": seed,
        "model_id": getattr(client, "model_id", "mock"),
        "revision": getattr(client, "revision", "mock"),
        "device_name": getattr(client, "device_name", "cpu"),
        "dtype": getattr(client, "dtype_name", "float32"),
        "n_calls": len(records),
        "input_tokens": sum(r["input_tokens"] for r in records),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "runtime_seconds": time.perf_counter() - t0,
        "choice": choice,
        "chosen_family": choice.get("chosen_family"),
        "chosen_levels": choice.get("chosen_levels"),
        "status": "STOP_SCORE_LT_4" if choice["stop"] else "LEVELS_CHOSEN",
        "float32_rerun_ids": [r["problem_id"] for r in records if r.get("float32_rerun")],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "levels.json", payload)
    write_json(out_dir / "pilot_raw.json", records)
    print(json.dumps({"status": payload["status"], "choice": choice}, indent=2))
    return 2 if choice["stop"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
