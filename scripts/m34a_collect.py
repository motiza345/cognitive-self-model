"""M34a collection: resumable JSONL cache from committed pool_plan.json (no analysis)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import load_json, sha256_file, write_json  # noqa: E402
from scripts.m34a_generate import MockQwenClient, TransformersQwenClient, run_problem  # noqa: E402

RAW_DIR = ROOT / "reports" / "m34a_raw"
CACHE_NAME = "responses.jsonl"
MANIFEST_NAME = "manifest.json"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        )
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def load_done_ids(cache_path: Path) -> set[str]:
    done: set[str] = set()
    if not cache_path.exists():
        return done
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(row["problem_id"])
    return done


def append_row(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M34a resumable collection")
    parser.add_argument(
        "--pool-plan",
        type=Path,
        default=ROOT / "reports" / "m34a_pilot2" / "pool_plan.json",
    )
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(argv)

    plan = load_json(args.pool_plan)
    problems = plan["problems"]
    cache_path = args.out_dir / CACHE_NAME
    done = load_done_ids(cache_path)
    print(f"pool plan {len(problems)} problems; already cached {len(done)}", flush=True)

    t0 = time.perf_counter()
    if args.mock:
        client: Any = MockQwenClient()
    else:
        client = TransformersQwenClient(model_id=args.model)

    versions = {"torch": None, "transformers": None}
    try:
        import torch
        import transformers

        versions["torch"] = torch.__version__
        versions["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass

    new_count = 0
    for problem in problems:
        pid = problem["problem_id"]
        if pid in done:
            continue
        row = run_problem(client, problem)
        row["model_id"] = getattr(client, "model_id", row.get("request_params", {}).get("model"))
        row["revision"] = getattr(client, "revision", row.get("request_params", {}).get("revision"))
        append_row(cache_path, row)
        done.add(pid)
        new_count += 1
        if new_count % 25 == 0:
            print(f"collected +{new_count} (total cached {len(done)})", flush=True)

    runtime = time.perf_counter() - t0
    # Reload cache for totals / float32 flags
    rows = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    cache_sha = sha256_file(cache_path)
    float32_ids = [r["problem_id"] for r in rows if r.get("float32_rerun")]
    manifest = {
        "model_id": getattr(client, "model_id", "mock"),
        "revision": getattr(client, "revision", "mock"),
        "device_name": getattr(client, "device_name", "cpu"),
        "dtype_default": getattr(client, "dtype_name", "float32"),
        "torch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "git_commit": git_commit(),
        "pool_plan_path": str(args.pool_plan.as_posix()),
        "pool_plan_n": len(problems),
        "n_cached": len(rows),
        "n_new_this_run": new_count,
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in rows),
        "runtime_seconds_this_run": runtime,
        "cache_file": CACHE_NAME,
        "cache_sha256": cache_sha,
        "float32_rerun_ids": float32_ids,
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost_usd_estimate": None,
    }
    write_json(args.out_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"n_cached": len(rows), "cache_sha256": cache_sha}, indent=2))
    if len(rows) != len(problems):
        print(
            f"WARNING: cache has {len(rows)} rows but plan has {len(problems)}",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
