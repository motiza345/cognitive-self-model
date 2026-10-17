"""M50 analysis: offline, once, on a clean tree. Protocol: docs/M50_PREREG.md.

  python -m scripts.m50_analyze                 # official run (8 shuffle seeds, both conditions)
  python -m scripts.m50_analyze --seeds 1 --allow-dirty --out /tmp/m50_check.md      # quick check only

Reads reports/m50_raw/{responses.jsonl, hidden/chunk_*.npz, manifest.json} and reports/m50_pilot/gates.json,
refuses to run on a dirty tree or on a hash mismatch, writes reports/m50_raw/analysis_summary.json and the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.m50_arms import (U, apply_condition, arm_always_answer, arm_always_verify, arm_from_probabilities,
                              arm_oracle_instance, arm_random_verify, bootstrap_diff, condition_labels, fit_probe, holm,
                              run_layer)
from scripts.m50_common import ROOT, git_commit, git_dirty, sha256_file
from cognitive_layer import from_token_logprobs  # noqa: E402

RAW_DIR = ROOT / "reports" / "m50_raw"
PILOT_DIR = ROOT / "reports" / "m50_pilot"
REPORT = ROOT / "reports" / "M50_REPORT.md"
CONDITIONS = ("natural", "implanted")


# ----------------------------------------------------------------------------- loading
def load_items(raw_dir: Path, verify_hashes: bool = True) -> tuple[list[dict], dict]:
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    cache = raw_dir / "responses.jsonl"
    if verify_hashes:
        if sha256_file(cache) != manifest["cache_sha256"]:
            raise RuntimeError("responses.jsonl does not match the manifest hash")
        for name, digest in manifest.get("hidden_chunks_sha256", {}).items():
            if sha256_file(raw_dir / "hidden" / name) != digest:
                raise RuntimeError(f"{name} does not match the manifest hash")
    hidden: dict[str, tuple] = {}
    for chunk in sorted((raw_dir / "hidden").glob("chunk_*.npz")) if (raw_dir / "hidden").exists() else []:
        z = np.load(chunk, allow_pickle=False)
        for i, pid in enumerate(z["ids"]):
            hidden[str(pid)] = (z["h_prompt"][i].astype(np.float64), z["h_answer"][i].astype(np.float64))
    items = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        h = hidden.get(r["problem_id"])
        items.append({"problem_id": r["problem_id"], "task": r["task"], "level": r["level"], "pool": r["pool"],
                      "correct": bool(r["correct"]), "conf": from_token_logprobs(r["token_logprobs"]),
                      "h_prompt": h[0] if h else None, "h_answer": h[1] if h else None})
    return items, manifest


# ----------------------------------------------------------------------------- one condition
class OutOfTime(Exception):
    pass


def _layer_seeds(cal, test, scopes, seeds, workers, key="", cache_dir=None, deadline=None, **kw):
    """Mean over shuffle seeds. Each (key, seed) result is stored in cache_dir, so a stopped analysis resumes and
    gives exactly the same numbers; `deadline` (epoch seconds) lets a time-limited machine stop and resume later."""
    done, todo = {}, []
    for s in seeds:
        path = cache_dir / f"{key}_seed{s}.npz" if cache_dir else None
        if path is not None and path.exists():
            z = np.load(path)
            done[s] = (z["util"], z["verified"], float(z["learn"]))
        else:
            todo.append(s)
    if todo and workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {s: pool.submit(run_layer, cal, test, scopes, s, **kw) for s in todo}
            computed = {s: f.result() for s, f in futures.items()}
        todo_results = [(s, computed[s]) for s in todo]
    else:
        todo_results = []
        for s in todo:
            if deadline is not None and time.time() > deadline:
                raise OutOfTime()
            todo_results.append((s, run_layer(cal, test, scopes, s, **kw)))
    for s, res in todo_results:
        done[s] = res
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_dir / f"{key}_seed{s}.npz", util=res[0], verified=res[1], learn=res[2])
    res = [done[s] for s in seeds]
    return (np.mean([r[0] for r in res], axis=0), np.mean([r[1] for r in res], axis=0), float(np.mean([r[2] for r in res])))


def _decision_verify_rate(probs) -> float:
    return float(np.mean([0.0 if 2.0 * float(p) - 1.0 > U.verify else 1.0 for p in probs]))


def _answered_accuracy(test, verify_rate_per_item) -> float | None:
    w = 1.0 - np.asarray(verify_rate_per_item)
    if w.sum() <= 0:
        return None
    return float(np.sum(w * np.array([1.0 if r["correct"] else 0.0 for r in test])) / w.sum())


def analyse_condition(items: list[dict], condition: str, seeds: list[int], workers: int, log=print,
                      cache_dir: Path | None = None, deadline: float | None = None) -> dict:
    items = apply_condition(items, condition)
    cal = [r for r in items if r["pool"] == "cal"]
    test = [r for r in items if r["pool"] == "test"]
    scopes = sorted({(r["task"], r["level"]) for r in items})
    scope_idx = {s: i for i, s in enumerate(scopes)}
    arms, meta = {}, {}

    arms["A1 always answer"], meta["A1 always answer"] = arm_always_answer(test), {"verify_rate": 0.0}
    arms["A2 always verify"], meta["A2 always verify"] = arm_always_verify(test), {"verify_rate": 1.0}
    log(f"  [{condition}] layer arms ({len(seeds)} seeds, {len(cal)} learning items, {len(test)} test items)")
    kw = {"cache_dir": cache_dir, "deadline": deadline}
    u6, v6, l6 = _layer_seeds(cal, test, scopes, seeds, workers, key=f"{condition}_scoped", **kw)
    u5, v5, l5 = _layer_seeds(cal, test, scopes, seeds, workers, key=f"{condition}_global", scoped=False, **kw)
    u7, v7, l7 = _layer_seeds(cal, test, scopes, seeds, workers, key=f"{condition}_baserate", mode="base_rate_only", **kw)
    arms["A6 scoped layer"], meta["A6 scoped layer"] = u6, {"verify_rate": float(v6.mean()), "learning_utility": l6,
                                                           "answered_accuracy": _answered_accuracy(test, v6)}
    arms["A5 global calibrated"], meta["A5 global calibrated"] = u5, {"verify_rate": float(v5.mean()), "learning_utility": l5,
                                                                    "answered_accuracy": _answered_accuracy(test, v5)}
    arms["A7 track record"], meta["A7 track record"] = u7, {"verify_rate": float(v7.mean()), "learning_utility": l7,
                                                          "answered_accuracy": _answered_accuracy(test, v7)}
    q = float(v6.mean())
    arms["A3 random verify"], meta["A3 random verify"] = arm_random_verify(test, q), {"verify_rate": q}
    raw_p = [r["conf"] for r in test]
    arms["A4 raw confidence"] = arm_from_probabilities(test, raw_p)
    meta["A4 raw confidence"] = {"verify_rate": _decision_verify_rate(raw_p)}
    probe_arms = []
    if all(r["h_prompt"] is not None for r in items):
        y = np.array([1.0 if r["correct"] else 0.0 for r in cal])
        for name, key in (("A8 probe prompt", "h_prompt"), ("A9 probe answer", "h_answer")):
            probe = fit_probe(np.stack([r[key] for r in cal]), y, [scope_idx[(r["task"], r["level"])] for r in cal], len(scopes))
            p = probe.predict(np.stack([r[key] for r in test]), [scope_idx[(r["task"], r["level"])] for r in test])
            arms[name] = arm_from_probabilities(test, p)
            meta[name] = {"verify_rate": _decision_verify_rate(p)}
            probe_arms.append(name)
    arms["R1 instance oracle"], meta["R1 instance oracle"] = arm_oracle_instance(test), {"verify_rate": None}

    simple = max(("A1 always answer", "A2 always verify", "A3 random verify"), key=lambda k: arms[k].mean())
    comps = {
        "A": {"desc": f"A6 minus best simple policy ({simple})", **bootstrap_diff(arms["A6 scoped layer"], arms[simple])},
        "B": {"desc": "A6 minus A4 raw confidence", **bootstrap_diff(arms["A6 scoped layer"], arms["A4 raw confidence"])},
        "C": {"desc": "A6 minus A5 global calibrated", **bootstrap_diff(arms["A6 scoped layer"], arms["A5 global calibrated"])},
    }
    if probe_arms:
        best_probe = max(probe_arms, key=lambda k: arms[k].mean())
        comps["D"] = {"desc": f"{best_probe} minus A6", **bootstrap_diff(arms[best_probe], arms["A6 scoped layer"])}
    else:
        comps["D"] = {"desc": "no hidden states available", "mean": 0.0, "lower": -1.0, "upper": 1.0, "p": 1.0}
    extra = {
        "A6 minus A7 track record": bootstrap_diff(arms["A6 scoped layer"], arms["A7 track record"]),
        "A6 minus A1 always answer": bootstrap_diff(arms["A6 scoped layer"], arms["A1 always answer"]),
        "A6 minus A2 always verify": bootstrap_diff(arms["A6 scoped layer"], arms["A2 always verify"]),
        "R1 minus A6": bootstrap_diff(arms["R1 instance oracle"], arms["A6 scoped layer"]),
    }
    return {"condition": condition, "n_cal": len(cal), "n_test": len(test), "n_scope_levels": len(scopes),
            "utility": {k: float(v.mean()) for k, v in arms.items()}, "meta": meta, "comparisons": comps, "extra": extra}


# ----------------------------------------------------------------------------- report
def write_report(path: Path, summary: dict) -> None:
    g, m = summary["gates"], summary["manifest"]
    lines = [
        "# M50 report (generated by scripts/m50_analyze.py)", "",
        "Protocol: `docs/M50_PREREG.md`. This file contains the numbers and labels produced by the committed analysis. "
        "The Claimed / NOT claimed / Scope section is added after review.", "",
        f"- analysis commit: `{summary['analysis_commit']}`; cache sha256 `{m['cache_sha256']}`; rows {m['n_rows']}; "
        f"model `{m['model_id']}` revision `{m['model_revision']}`", f"- gates: all_pass = {g['all_pass']}, "
        f"G_ceiling = {g['G_ceiling']}, G_variance = {g['G_variance']}, test per scope-level = {g['test_per_scope_level']}",
        f"- shuffle seeds: {summary['seeds']}", ""]
    for cond in CONDITIONS:
        c = summary["conditions"][cond]
        lab = c["labels"]
        lines += [f"## Condition: {cond}", "", f"Scope-levels {c['n_scope_levels']}, learning items {c['n_cal']}, test items {c['n_test']}. "
                  f"Status: **{lab['status']}**.", "", "| arm | mean utility | verify rate | answered accuracy | learning-phase utility |", "|---|---:|---:|---:|---:|"]
        for arm, u in c["utility"].items():
            mt = c["meta"].get(arm, {})
            fmt = lambda x: "" if x is None else f"{x:.4f}"
            lines.append(f"| {arm} | {u:.4f} | {fmt(mt.get('verify_rate'))} | {fmt(mt.get('answered_accuracy'))} | {fmt(mt.get('learning_utility'))} |")
        lines += ["", "| hypothesis | comparison | mean difference | 95% interval | label | Holm significant |", "|---|---|---:|---|---|---|"]
        for h in "ABCD":
            comp = c["comparisons"][h]
            hs = summary["holm"].get(f"{cond}:{h}")
            lines.append(f"| H-{h} | {comp['desc']} | {comp['mean']:+.4f} | [{comp['lower']:+.4f}, {comp['upper']:+.4f}] | {lab[h]} | "
                         f"{'primary' if (cond == 'natural' and h == 'A') else hs} |")
        lines += ["", "Descriptive comparisons:", ""]
        for name, comp in c["extra"].items():
            lines.append(f"- {name}: {comp['mean']:+.4f} [{comp['lower']:+.4f}, {comp['upper']:+.4f}]")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def run(raw_dir: Path, gates_path: Path, seeds: list[int], workers: int, log=print, cache_dir: Path | None = None,
        max_seconds: float | None = None) -> dict:
    items, manifest = load_items(raw_dir)
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    summary = {"analysis_commit": git_commit(), "seeds": seeds, "manifest": manifest, "gates": gates, "conditions": {},
               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pvals = {}
    deadline = time.time() + max_seconds if max_seconds else None
    for cond in CONDITIONS:
        res = analyse_condition(items, cond, seeds, workers, log=log, cache_dir=cache_dir, deadline=deadline)
        res["labels"] = condition_labels(cond, bool(gates["all_pass"]), res["comparisons"])
        summary["conditions"][cond] = res
        for h in "ABCD":
            if not (cond == "natural" and h == "A"):
                pvals[f"{cond}:{h}"] = res["comparisons"][h]["p"]
    summary["holm"] = holm(pvals)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--gates", type=Path, default=PILOT_DIR / "gates.json")
    ap.add_argument("--out", type=Path, default=REPORT)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--allow-dirty", action="store_true", help="quick checks only")
    ap.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after this long; rerun to resume (cached results are reused)")
    args = ap.parse_args(argv)
    if git_dirty() and not args.allow_dirty:
        print("The working tree is not clean; the official analysis must run on a committed tree.")
        return 2
    try:
        summary = run(args.raw_dir, args.gates, list(range(args.seeds)), args.workers, cache_dir=args.raw_dir / "analysis_cache",
                      max_seconds=args.max_seconds)
    except OutOfTime:
        print("Stopped on the time limit; progress is saved. Run the same command again to resume.")
        return 4
    (args.raw_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.out, summary)
    for cond in CONDITIONS:
        print(cond, summary["conditions"][cond]["labels"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
