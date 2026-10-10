"""Write S1 (add_nn) level selection and pool plan. No model calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_v3_common import write_s1_selection_and_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "m34a_v3_s1")
    args = parser.parse_args(argv)
    out = write_s1_selection_and_plan(args.out_dir)
    if isinstance(out, dict) and out.get("status") == "STOP_GAP_LT_0.05":
        print("STOP: gap_hat < 0.05")
        return 2
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
