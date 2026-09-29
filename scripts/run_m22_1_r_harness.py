"""Run the M22.1-R harness.

The default command does not load Qwen and does not execute the preflight.
It writes an execution record. ``--execute-forward`` is required before any
pinned load, and a missing offline snapshot still stops with REPLAY_BLOCKED.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cognitive_self_model.m22_1_r.harness import run_harness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M22.1-R replay harness")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--execute-forward",
        action="store_true",
        help="Attempt a fail-closed pinned load. Does not fall back to an unpinned revision.",
    )
    parser.add_argument(
        "--update-recovery",
        action="store_true",
        help="Splice the execution record into control_plane/RECOVERY.yaml.",
    )
    args = parser.parse_args(argv)
    record = run_harness(
        Path(args.repo_root),
        execute_forward=args.execute_forward,
        update_recovery=args.update_recovery,
    )
    print(record["replay_tier"])
    print(record["block_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
