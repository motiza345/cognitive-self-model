"""Build committed pool plan from pilot2 levels.json (no model calls)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import (  # noqa: E402
    SEED_PILOT,
    build_pool_plan,
    load_json,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M34a pool plan from chosen levels")
    parser.add_argument(
        "--levels",
        type=Path,
        default=ROOT / "reports" / "m34a_pilot2" / "levels.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "m34a_pilot2" / "pool_plan.json",
    )
    parser.add_argument("--seed", type=int, default=SEED_PILOT)
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Override family (default: from levels.json)",
    )
    args = parser.parse_args(argv)

    levels_doc = load_json(args.levels)
    if levels_doc.get("status", "").startswith("STOP") or levels_doc.get("choice", {}).get(
        "stop"
    ):
        raise SystemExit("pilot STOP: refusing to build pools")
    chosen = levels_doc.get("chosen_levels") or levels_doc.get("choice", {}).get(
        "chosen_levels"
    )
    family = args.family or levels_doc.get("chosen_family") or levels_doc.get("choice", {}).get(
        "chosen_family"
    )
    if not family:
        raise SystemExit("family missing in levels.json (Amendment 1 requires chosen_family)")
    if not chosen or len(chosen) != 6:
        raise SystemExit(f"need exactly 6 chosen levels, got {chosen!r}")
    plan = build_pool_plan(args.seed, [int(x) for x in chosen], str(family))
    plan["levels_status"] = levels_doc.get("status")
    plan["levels_best_score"] = levels_doc.get("choice", {}).get("best_score")
    write_json(args.out, plan)
    print(f"wrote {args.out} family={family} n={plan['n_problems']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
