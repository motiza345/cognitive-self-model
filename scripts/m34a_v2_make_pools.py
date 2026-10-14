"""Write v2 level selection and pool plan (no model calls)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_v2_common import (  # noqa: E402
    SEED_V2_POOLS,
    build_v2_pool_plan,
    load_pilot2_mul_n1,
    select_v2_window,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "m34a_v2")
    parser.add_argument("--seed", type=int, default=SEED_V2_POOLS)
    args = parser.parse_args(argv)
    acc = load_pilot2_mul_n1()
    choice = select_v2_window(acc)
    write_json(
        args.out_dir / "levels.json",
        {
            "version": "m34a-v2",
            "family": "mul_n1",
            "pilot2_mul_n1": {str(k): v for k, v in sorted(acc.items())},
            "choice": choice,
            "chosen_levels": choice["chosen_levels"],
            "status": "STOP_GAP_LT_0.05" if choice["stop"] else "LEVELS_CHOSEN",
        },
    )
    if choice["stop"]:
        print("STOP: gap_hat < 0.05")
        return 2
    plan = build_v2_pool_plan(args.seed, [int(x) for x in choice["chosen_levels"]])
    write_json(args.out_dir / "pool_plan.json", plan)
    print(
        f"levels={plan['chosen_levels']} n={plan['n_problems']} "
        f"gap_hat={choice['best']['gap_hat']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
