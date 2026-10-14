"""M34a-v3 resumable collection (parameterized setting). No analysis."""

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
from scripts.m34a_v3_common import PHI_MODEL_ID, QWEN_MODEL_ID, QWEN_REVISION  # noqa: E402
from scripts.m34a_v3_generate import make_client, run_problem  # noqa: E402

CACHE_NAME = "responses.jsonl"
MANIFEST_NAME = "manifest.json"

SETTINGS = {
    "s1": {
        "pool_plan": ROOT / "reports" / "m34a_v3_s1" / "pool_plan.json",
        "out_dir": ROOT / "reports" / "m34a_v3_s1_raw",
        "model_id": QWEN_MODEL_ID,
        "revision": QWEN_REVISION,
        "version": "m34a-v3-s1",
    },
    "s2": {
        "pool_plan": ROOT / "reports" / "m34a_v3_s2" / "pool_plan.json",
        "out_dir": ROOT / "reports" / "m34a_v3_s2_raw",
        "model_id": PHI_MODEL_ID,
        "revision": None,  # from pin file
        "version": "m34a-v3-s2",
        "revision_pin": ROOT / "reports" / "m34a_v3_s2" / "revision.json",
    },
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def load_done_ids(cache_path: Path) -> set[str]:
    done: set[str] = set()
    if not cache_path.exists():
        return done
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["problem_id"])
    return done


def append_row(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_s2_revision(pin_path: Path) -> str:
    if not pin_path.exists():
        raise SystemExit(
            f"missing S2 revision pin {pin_path}; run scripts.m34a_v3_pin_s2_revision first"
        )
    doc = load_json(pin_path)
    rev = doc.get("revision")
    if not rev:
        raise SystemExit(f"empty revision in {pin_path}")
    return str(rev)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=sorted(SETTINGS), required=True)
    parser.add_argument("--pool-plan", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = SETTINGS[args.setting]
    pool_plan = args.pool_plan or cfg["pool_plan"]
    out_dir = args.out_dir or cfg["out_dir"]
    model_id = args.model or cfg["model_id"]
    revision = args.revision
    if revision is None:
        revision = cfg.get("revision")
    if args.setting == "s2" and revision is None and not args.mock:
        revision = resolve_s2_revision(cfg["revision_pin"])

    plan = load_json(pool_plan)
    problems = plan["problems"]
    cache_path = out_dir / CACHE_NAME
    done = load_done_ids(cache_path)
    print(f"setting={args.setting} plan {len(problems)}; cached {len(done)}", flush=True)

    t0 = time.perf_counter()
    try:
        client: Any = make_client(mock=args.mock, model_id=model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001
        write_json(
            out_dir / "NOT_RUN.json",
            {
                "setting": args.setting,
                "reason": "model_load_failed",
                "error": repr(exc),
                "model_id": model_id,
                "revision": revision,
                "git_commit": git_commit(),
            },
        )
        print(f"NOT_RUN: model load failed: {exc}", flush=True)
        return 3

    versions = {"torch": None, "transformers": None}
    if not args.mock:
        try:
            import torch
            import transformers

            versions["torch"] = torch.__version__
            versions["transformers"] = transformers.__version__
        except Exception:  # noqa: BLE001
            pass

    new_count = 0
    try:
        for problem in problems:
            if problem["problem_id"] in done:
                continue
            row = run_problem(client, problem)
            # logprob rule sanity: empty non-EOS token logprobs on a non-empty generation
            if (
                not args.mock
                and row["generated_token_ids"]
                and not row["token_logprobs"]
                and not row["nonfinite_detected"]
            ):
                raise RuntimeError("answer-token logprob rule broken (no token logprobs)")
            row["model_id"] = getattr(client, "model_id", model_id)
            row["revision"] = getattr(client, "revision", revision)
            append_row(cache_path, row)
            done.add(problem["problem_id"])
            new_count += 1
            if new_count % 25 == 0:
                print(f"collected +{new_count} (cached {len(done)})", flush=True)
    except Exception as exc:  # noqa: BLE001
        write_json(
            out_dir / "NOT_RUN.json",
            {
                "setting": args.setting,
                "reason": "logprob_or_runtime_failure",
                "error": repr(exc),
                "model_id": model_id,
                "revision": getattr(client, "revision", revision),
                "git_commit": git_commit(),
                "n_cached_before_fail": len(done),
            },
        )
        print(f"NOT_RUN: {exc}", flush=True)
        return 3

    rows = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    cache_sha = sha256_file(cache_path)
    manifest = {
        "setting": args.setting,
        "model_id": getattr(client, "model_id", model_id),
        "revision": getattr(client, "revision", revision),
        "device_name": getattr(client, "device_name", "cpu"),
        "dtype_default": getattr(client, "dtype_name", "float32"),
        "torch_version": versions["torch"],
        "transformers_version": versions["transformers"],
        "git_commit": git_commit(),
        "pool_plan_path": str(Path(pool_plan).as_posix()),
        "pool_plan_n": len(problems),
        "n_cached": len(rows),
        "n_new_this_run": new_count,
        "input_tokens": sum(int(r.get("input_tokens", 0)) for r in rows),
        "output_tokens": sum(int(r.get("output_tokens", 0)) for r in rows),
        "runtime_seconds_this_run": time.perf_counter() - t0,
        "cache_file": CACHE_NAME,
        "cache_sha256": cache_sha,
        "float32_rerun_ids": [r["problem_id"] for r in rows if r.get("float32_rerun")],
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": cfg["version"],
    }
    write_json(out_dir / MANIFEST_NAME, manifest)
    print(json.dumps({"n_cached": len(rows), "cache_sha256": cache_sha}, indent=2))
    return 0 if len(rows) == len(problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
