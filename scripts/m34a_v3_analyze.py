"""M34a-v3 offline analysis. Reuses v2 arms/gates/labels. Refuses dirty tree."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import (  # noqa: E402
    N_H_VALUES,
    PRIMARY_N_H,
    load_json,
    sha256_file,
    write_json,
)
from scripts.m34a_v2_analyze import (  # noqa: E402
    cal_level_maps,
    cal_var_ok,
    index_by_pool,
    labels_from,
    level_accuracy,
    load_cache,
    mean_utilities,
    risk_coverage,
)
from scripts.m34a_v2_common import mann_whitney_auroc  # noqa: E402
from scripts.m34a_v3_carries import n_carries_for_row  # noqa: E402
from scripts.m34a_v3_common import aggregate_c  # noqa: E402

SETTINGS = {
    "s1": {
        "raw_dir": ROOT / "reports" / "m34a_v3_s1_raw",
        "summary": ROOT / "reports" / "m34a_v3_s1_raw" / "analysis_summary.json",
        "name": "S1",
        "desc": "Qwen2.5-3B-Instruct add_nn levels 3–8",
    },
    "s2": {
        "raw_dir": ROOT / "reports" / "m34a_v3_s2_raw",
        "summary": ROOT / "reports" / "m34a_v3_s2_raw" / "analysis_summary.json",
        "name": "S2",
        "desc": "Phi-3.5-mini-instruct mul_n1",
    },
}
REPORT_PATH = ROOT / "reports" / "M34A_V3_REPORT.md"


def require_clean_git() -> str:
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read git status: {exc}") from exc
    if dirty.strip():
        raise SystemExit("refusing analysis: dirty working tree (tracked files)\n" + dirty)
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"cannot read HEAD: {exc}") from exc


def mean_level_auroc(rows: list[dict[str, Any]], score_fn) -> dict[str, Any]:
    per: dict[str, float | None] = {}
    finite: list[float] = []
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[int(r["level"])].append(r)
    for L in sorted(by):
        subset = by[L]
        val = mann_whitney_auroc(
            [float(score_fn(r)) for r in subset],
            [bool(r["correct"]) for r in subset],
        )
        per[str(L)] = val
        if val is not None:
            finite.append(val)
    return {"per_level": per, "mean": (sum(finite) / len(finite)) if finite else None}


def analyze_setting(raw_dir: Path) -> dict[str, Any]:
    not_run = raw_dir / "NOT_RUN.json"
    if not_run.exists() and not (raw_dir / "responses.jsonl").exists():
        doc = load_json(not_run)
        reason = str(doc.get("reason", ""))
        if reason == "STOP_GAP_LT_0.05":
            return {
                "status": "STOP",
                "A": "NOT_INFORMATIVE",
                "C": "NOT_INFORMATIVE",
                "not_run": doc,
            }
        return {
            "status": "NOT_RUN",
            "A": "NOT_RUN",
            "C": "NOT_INFORMATIVE",
            "not_run": doc,
        }
    manifest = load_json(raw_dir / "manifest.json")
    cache_path = raw_dir / "responses.jsonl"
    cache_sha = sha256_file(cache_path)
    if manifest.get("cache_sha256") != cache_sha:
        raise SystemExit(
            f"cache hash mismatch: manifest={manifest.get('cache_sha256')} file={cache_sha}"
        )
    rows = load_cache(cache_path)
    by = index_by_pool(rows)
    test_rows = sorted(by["test"], key=lambda r: r["problem_id"])
    if len(test_rows) != 240:
        raise SystemExit(f"expected 240 TEST rows, got {len(test_rows)}")
    hist_by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in by["hist"]:
        hist_by[int(r["level"])].append(r)
    oracle, level_cal, edges, lp_cal = cal_level_maps(by["cal"])
    gate_c = cal_var_ok(by["cal"])
    utils = mean_utilities(
        test_rows, hist_by, PRIMARY_N_H, "V3", oracle, level_cal, edges, lp_cal, 34011
    )
    primary = labels_from(utils, gate_c)
    arm_means = {arm: sum(v) / len(v) for arm, v in utils.items()}
    nh_table = {}
    for nh in N_H_VALUES:
        u = mean_utilities(
            test_rows, hist_by, nh, "V3", oracle, level_cal, edges, lp_cal, 34011 + nh
        )
        nh_table[nh] = {
            "self": sum(u["self"]) / len(u["self"]),
            "global": sum(u["global"]) / len(u["global"]),
            "shuffled": sum(u["shuffled"]) / len(u["shuffled"]),
        }
    auroc_lp = mean_level_auroc(rows, lambda r: float(r["answer_logprob"]))
    auroc_carry = mean_level_auroc(rows, lambda r: -float(n_carries_for_row(r)))
    rc = risk_coverage(test_rows)
    return {
        "status": "OK",
        "manifest": manifest,
        "cache_sha256": cache_sha,
        "primary": primary,
        "A": primary["A"],
        "C": primary["C"],
        "arm_means_v3_nh10": arm_means,
        "nh_table": {str(k): v for k, v in nh_table.items()},
        "per_level_accuracy": {str(k): v for k, v in level_accuracy(rows).items()},
        "auroc_logprob": auroc_lp,
        "auroc_neg_carries": auroc_carry,
        "risk_coverage": rc,
    }


def render_report(
    results: dict[str, dict[str, Any]], git_commit: str
) -> str:
    s1, s2 = results["s1"], results["s2"]
    agg = aggregate_c(s1["C"], s2["C"])
    lines = [
        "# M34A-v3 report",
        "",
        f"**Aggregate C:** `{agg}` · S1 A=`{s1['A']}` C=`{s1['C']}` · S2 A=`{s2['A']}` C=`{s2['C']}`",
        "",
        f"- analysis git commit: `{git_commit}` (clean tree required)",
        "",
    ]
    for key in ("s1", "s2"):
        cfg = SETTINGS[key]
        r = results[key]
        lines += [f"## {cfg['name']}: {cfg['desc']}", ""]
        if r["status"] == "NOT_RUN":
            lines += [
                f"- status: `NOT_RUN`",
                f"- detail: `{json.dumps(r.get('not_run', {}), sort_keys=True)}`",
                "",
            ]
            continue
        p = r["primary"]
        m = r["manifest"]
        lines += [
            f"**Verdict:** A=`{r['A']}` · C=`{r['C']}` · GATE_A=`{p['gate_a']}` · GATE_C=`{p['gate_c']}`",
            "",
            f"- cache sha256: `{r['cache_sha256']}`",
            f"- model: `{m.get('model_id')}` revision `{m.get('revision')}`",
            f"- device/dtype: `{m.get('device_name')}` / `{m.get('dtype_default')}`",
            f"- collection git commit: `{m.get('git_commit')}`",
            f"- GATE_A gap = {p['gate_gap']:.6f}",
            f"- self-global: mean={p['self_minus_global']['mean']:.6f} CI=[{p['self_minus_global']['low']:.6f}, {p['self_minus_global']['high']:.6f}]",
            f"- self-shuffled: mean={p['self_minus_shuffled']['mean']:.6f} CI=[{p['self_minus_shuffled']['low']:.6f}, {p['self_minus_shuffled']['high']:.6f}]",
            f"- level_lp_cal-level_cal: mean={p['level_lp_cal_minus_level_cal']['mean']:.6f} CI=[{p['level_lp_cal_minus_level_cal']['low']:.6f}, {p['level_lp_cal_minus_level_cal']['high']:.6f}]",
            f"- mean within-level AUROC answer_logprob: {r['auroc_logprob']['mean']}",
            f"- mean within-level AUROC (−n_carries): {r['auroc_neg_carries']['mean']}",
            "",
            "### V3 mean utilities (n_h=10)",
            "",
            "| arm | mean U |",
            "| --- | ---: |",
        ]
        for arm, val in r["arm_means_v3_nh10"].items():
            lines.append(f"| `{arm}` | {val:.6f} |")
        lines += [
            "",
            "### Descriptive: n_h curve (V3)",
            "",
            "| n_h | self | global | shuffled |",
            "| ---: | ---: | ---: | ---: |",
        ]
        for nh, row in sorted(r["nh_table"].items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {nh} | {row['self']:.6f} | {row['global']:.6f} | {row['shuffled']:.6f} |"
            )
        lines.append("")
    lines += [
        "## Claimed",
        "",
        f"- Per-setting A/C under `docs/M34A_V3_PREREG.md` (v2 gates/labels). Aggregate C=`{agg}`.",
        "",
        "## NOT claimed",
        "",
        "- Privileged access (logprobs are API-visible); mechanistic κ; planning/games; other models/operations beyond S1/S2.",
        "",
        "## Scope",
        "",
        "- S1: one Qwen2.5-3B-Instruct, addition (`add_nn`) levels 3–8 only.",
        "- S2: one Phi-3.5-mini-instruct, multiplication by one digit (`mul_n1`) only (or NOT_RUN / STOP).",
        "- Static warm-start history, n_h=10 primary, greedy float16. M34a closed after v3.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["s1", "s2", "both"], default="both")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--allow-dirty", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    git_commit = "TEST" if args.allow_dirty else require_clean_git()
    keys = ["s1", "s2"] if args.setting == "both" else [args.setting]
    results: dict[str, dict[str, Any]] = {}
    for key in keys:
        cfg = SETTINGS[key]
        raw = cfg["raw_dir"]
        if not raw.exists() and args.setting == "both":
            # allow writing report when one setting missing only if NOT_RUN marker planned
            results[key] = {
                "status": "NOT_RUN",
                "A": "NOT_RUN",
                "C": "NOT_INFORMATIVE",
                "not_run": {"reason": "missing_raw_dir"},
            }
            continue
        out = analyze_setting(raw)
        results[key] = out
        write_json(cfg["summary"], {**out, "analysis_git_commit": git_commit})

    if args.setting == "both" or set(results) == {"s1", "s2"}:
        # fill missing side for single-setting runs when writing partial
        for key in ("s1", "s2"):
            if key not in results:
                results[key] = {
                    "status": "NOT_RUN",
                    "A": "NOT_RUN",
                    "C": "NOT_INFORMATIVE",
                    "not_run": {"reason": "not_analyzed_this_invocation"},
                }
        report = render_report(results, git_commit)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")

    summary = {
        k: {"A": results[k]["A"], "C": results[k]["C"], "status": results[k]["status"]}
        for k in results
    }
    if "s1" in results and "s2" in results:
        summary["aggregate_C"] = aggregate_c(results["s1"]["C"], results["s2"]["C"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
