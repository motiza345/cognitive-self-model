"""Resolve Phi-3.5-mini-instruct revision and write the pin BEFORE S2 collection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import write_json  # noqa: E402
from scripts.m34a_v3_common import PHI_MODEL_ID  # noqa: E402
from scripts.m34a_v3_generate import resolve_revision  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "m34a_v3_s2" / "revision.json",
    )
    parser.add_argument("--model", type=str, default=PHI_MODEL_ID)
    args = parser.parse_args(argv)
    rev = resolve_revision(args.model)
    doc = {
        "model_id": args.model,
        "revision": rev,
        "pinned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Pinned before any S2 collection. Do not change after outcomes exist.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, doc)
    print(doc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
