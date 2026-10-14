"""Write S2 pool plan from pilot levels.json. No model calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import load_json, write_json  # noqa: E402
from scripts.m34a_v3_common import SEED_S2_POOLS, build_s2_pool_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        type=Path,
        default=ROOT / "reports" / "m34a_v3_s2_pilot" / "levels.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "m34a_v3_s2")
    parser.add_argument("--seed", type=int, default=SEED_S2_POOLS)
    args = parser.parse_args(argv)
    doc = load_json(args.levels)
    if doc.get("status") == "STOP_GAP_LT_0.05" or not doc.get("chosen_levels"):
        print("STOP: no S2 pool plan")
        return 2
    levels = [int(x) for x in doc["chosen_levels"]]
    plan = build_s2_pool_plan(levels, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "pool_plan.json", plan)
    write_json(
        args.out_dir / "levels.json",
        {
            "version": "m34a-v3-s2",
            "family": "mul_n1",
            "chosen_levels": levels,
            "from_pilot": str(args.levels.as_posix()),
            "gap_hat": doc.get("choice", {}).get("best", {}).get("gap_hat"),
        },
    )
    print(f"levels={levels} n={plan['n_problems']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
