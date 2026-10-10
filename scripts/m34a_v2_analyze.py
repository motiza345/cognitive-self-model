"""M34a-v2 offline analysis. Refuses dirty tree and cache-hash mismatch."""

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
    N_HISTORY_DRAWS,
    PRIMARY_N_H,
    choose_action,
    load_json,
    paired_mean_ci,
    random_derangement,
    realized_utility,
    sha256_file,
    write_json,
)
from scripts.m34a_v2_common import mann_whitney_auroc, tercile_bin, tercile_edges  # noqa: E402

RAW_DIR = ROOT / "reports" / "m34a_v2_raw"
REPORT_PATH = ROOT / "reports" / "M34A_V2_REPORT.md"

HIST_ARMS = ("naive", "global", "self", "shuffled")
CAL_ARMS = ("oracle_level", "oracle_instance", "level_cal", "level_lp_cal")
ALL_ARMS = HIST_ARMS + CAL_ARMS


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


def load_cache(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_by_pool(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[r["pool"]].append(r)
    return out


def level_accuracy(rows: list[dict[str, Any]]) -> dict[int, float]:
    by: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        by[int(r["level"])].append(bool(r["correct"]))
    return {L: sum(v) / len(v) for L, v in by.items()}


def sample_history(hist_by_level: dict[int, list[dict[str, Any]]], n_h: int, rng):
    sample = {}
    for L, rows in hist_by_level.items():
        if len(rows) < n_h:
            raise ValueError(f"HIST level {L} has {len(rows)} < n_h={n_h}")
        idxs = rng.sample(range(len(rows)), n_h)
        sample[L] = [rows[i] for i in idxs]
    return sample


def self_p(sample: dict[int, list[dict[str, Any]]], n_h: int) -> dict[int, float]:
    return {
        L: (sum(1 for r in rows if r["correct"]) + 1) / (n_h + 2) for L, rows in sample.items()
    }


def global_p(sample: dict[int, list[dict[str, Any]]]) -> float:
    rows = [r for rs in sample.values() for r in rs]
    return sum(1 for r in rows if r["correct"]) / len(rows)


def cal_level_maps(cal_rows: list[dict[str, Any]]):
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in cal_rows:
        by[int(r["level"])].append(r)
    oracle = {L: sum(1 for r in rows if r["correct"]) / len(rows) for L, rows in by.items()}
    level_cal = {
        L: (sum(1 for r in rows if r["correct"]) + 1) / (len(rows) + 2) for L, rows in by.items()
    }
    edges = {L: tercile_edges([float(r["answer_logprob"]) for r in rows]) for L, rows in by.items()}
    cells: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for L, rows in by.items():
        for r in rows:
            b = tercile_bin(float(r["answer_logprob"]), edges[L])
            cells[(L, b)].append(bool(r["correct"]))
    lp_cal = {
        key: (sum(vals) + 1) / (len(vals) + 2) for key, vals in cells.items() if vals
    }
    return oracle, level_cal, edges, lp_cal


def risk_coverage(rows: list[dict[str, Any]], fracs: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> dict[str, Any]:
    """Selective classification: keep top fraction by answer_logprob; risk = 1 - accuracy."""
    ordered = sorted(rows, key=lambda r: float(r["answer_logprob"]), reverse=True)
    n = len(ordered)
    out: dict[str, Any] = {"overall": {}, "per_level": {}}
    for frac in fracs:
        k = max(1, int(round(frac * n)))
        kept = ordered[:k]
        acc = sum(1 for r in kept if r["correct"]) / len(kept)
        out["overall"][str(frac)] = {
            "coverage": len(kept) / n,
            "n_kept": len(kept),
            "accuracy": acc,
            "risk": 1.0 - acc,
        }
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[int(r["level"])].append(r)
    for L, subset in sorted(by.items()):
        so = sorted(subset, key=lambda r: float(r["answer_logprob"]), reverse=True)
        half = max(1, int(round(0.5 * len(so))))
        kept = so[:half]
        acc = sum(1 for r in kept if r["correct"]) / len(kept)
        out["per_level"][str(L)] = {
            "n": len(so),
            "coverage_0.5_n": half,
            "accuracy": acc,
            "risk": 1.0 - acc,
        }
    return out


def cal_var_ok(cal_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[int, list[bool]] = defaultdict(list)
    for r in cal_rows:
        by[int(r["level"])].append(bool(r["correct"]))
    n_var = 0
    detail = {}
    for L, vals in sorted(by.items()):
        ok = sum(vals)
        bad = len(vals) - ok
        detail[str(L)] = {"correct": ok, "wrong": bad, "n": len(vals), "accuracy": ok / len(vals)}
        if ok >= 15 and bad >= 15:
            n_var += 1
    return {"var_ok": n_var >= 3, "n_levels_var": n_var, "per_level": detail}


def p_for_arm(arm: str, row: dict[str, Any], **kw: Any) -> float | None:
    L = int(row["level"])
    if arm == "naive":
        return None
    if arm == "global":
        return kw["g_p"]
    if arm == "self":
        return kw["self_ps"][L]
    if arm == "shuffled":
        return kw["self_ps"][kw["shuffled_map"][L]]
    if arm == "oracle_level":
        return kw["oracle"][L]
    if arm == "oracle_instance":
        return 1.0 if row["correct"] else 0.0
    if arm == "level_cal":
        return kw["level_cal"][L]
    if arm == "level_lp_cal":
        b = tercile_bin(float(row["answer_logprob"]), kw["edges"][L])
        return kw["lp_cal"].get((L, b), kw["level_cal"][L])
    raise ValueError(arm)


def utility_for_arm(arm: str, row: dict[str, Any], variant: str, **kw: Any) -> float:
    if arm == "naive":
        return realized_utility("answer", bool(row["correct"]))
    p = p_for_arm(arm, row, **kw)
    assert p is not None
    return realized_utility(choose_action(float(p), variant), bool(row["correct"]))


def mean_utilities(
    test_rows: list[dict[str, Any]],
    hist_by_level: dict[int, list[dict[str, Any]]],
    n_h: int,
    variant: str,
    oracle: dict[int, float],
    level_cal: dict[int, float],
    edges: dict[int, tuple[float, float]],
    lp_cal: dict[tuple[int, int], float],
    seed: int,
) -> dict[str, list[float]]:
    import random

    rng = random.Random(seed)
    levels = sorted(hist_by_level)
    sums = {arm: [0.0] * len(test_rows) for arm in ALL_ARMS}
    for _ in range(N_HISTORY_DRAWS):
        sample = sample_history(hist_by_level, n_h, rng)
        kw = {
            "g_p": global_p(sample),
            "self_ps": self_p(sample, n_h),
            "shuffled_map": random_derangement(rng, levels),
            "oracle": oracle,
            "level_cal": level_cal,
            "edges": edges,
            "lp_cal": lp_cal,
        }
        for i, row in enumerate(test_rows):
            for arm in ALL_ARMS:
                sums[arm][i] += utility_for_arm(arm, row, variant, **kw)
    return {arm: [s / N_HISTORY_DRAWS for s in vals] for arm, vals in sums.items()}


def labels_from(utils: dict[str, list[float]], gate_c: dict[str, Any]) -> dict[str, Any]:
    def delta(a: str, b: str) -> list[float]:
        return [x - y for x, y in zip(utils[a], utils[b])]

    u_oracle = sum(utils["oracle_level"]) / len(utils["oracle_level"])
    u_global = sum(utils["global"]) / len(utils["global"])
    gate_a = (u_oracle - u_global) >= 0.05
    self_global = paired_mean_ci(delta("self", "global"))
    self_shuf = paired_mean_ci(delta("self", "shuffled"))
    if not gate_a:
        a_label = "NOT_INFORMATIVE"
    elif self_global["low"] > 0 and self_global["mean"] >= 0.05 and self_shuf["low"] > 0:
        a_label = "SUPPORTED"
    elif self_global["low"] > 0:
        a_label = "SUPPORTED_WEAK"
    else:
        a_label = "NOT_SUPPORTED"

    lp_vs_cal = paired_mean_ci(delta("level_lp_cal", "level_cal"))
    if not gate_c["var_ok"]:
        c_label = "NOT_INFORMATIVE"
    elif lp_vs_cal["low"] > 0 and lp_vs_cal["mean"] >= 0.03:
        c_label = "INTERNAL_VALUE"
    else:
        c_label = "INTERNAL_NONE"
    return {
        "gate_a": gate_a,
        "gate_c": gate_c["var_ok"],
        "U_oracle_level": u_oracle,
        "U_global": u_global,
        "gate_gap": u_oracle - u_global,
        "A": a_label,
        "C": c_label,
        "self_minus_global": self_global,
        "self_minus_shuffled": self_shuf,
        "level_lp_cal_minus_level_cal": lp_vs_cal,
        "cal_var": gate_c,
    }


def render_report(
    manifest: dict[str, Any],
    primary: dict[str, Any],
    arm_means: dict[str, float],
    v2_means: dict[str, float],
    nh_table: dict[int, dict[str, float]],
    per_level: dict[int, float],
    auroc: dict[str, Any],
    rc: dict[str, Any],
    git_commit: str,
    cache_sha: str,
) -> str:
    lines = [
        "# M34A-v2 report",
        "",
        f"**Primary verdict (V3, n_h={PRIMARY_N_H}):** A=`{primary['A']}` · C=`{primary['C']}` · GATE_A=`{primary['gate_a']}` · GATE_C=`{primary['gate_c']}`",
        "",
        f"- cache sha256: `{cache_sha}`",
        f"- model: `{manifest.get('model_id')}` revision `{manifest.get('revision')}`",
        f"- device/dtype: `{manifest.get('device_name')}` / `{manifest.get('dtype_default')}`",
        f"- torch/transformers: `{manifest.get('torch_version')}` / `{manifest.get('transformers_version')}`",
        f"- collection git commit: `{manifest.get('git_commit')}`",
        f"- analysis git commit: `{git_commit}` (clean tree required)",
        f"- GATE_A gap U(oracle_level)-U(global) = {primary['gate_gap']:.6f} (need >= 0.05)",
        f"- self-global: mean={primary['self_minus_global']['mean']:.6f} CI=[{primary['self_minus_global']['low']:.6f}, {primary['self_minus_global']['high']:.6f}]",
        f"- self-shuffled: mean={primary['self_minus_shuffled']['mean']:.6f} CI=[{primary['self_minus_shuffled']['low']:.6f}, {primary['self_minus_shuffled']['high']:.6f}]",
        f"- level_lp_cal-level_cal: mean={primary['level_lp_cal_minus_level_cal']['mean']:.6f} CI=[{primary['level_lp_cal_minus_level_cal']['low']:.6f}, {primary['level_lp_cal_minus_level_cal']['high']:.6f}]",
        "",
        "## V3 mean utilities (n_h=10)",
        "",
        "| arm | mean U |",
        "| --- | ---: |",
    ]
    for arm, val in arm_means.items():
        lines.append(f"| `{arm}` | {val:.6f} |")
    lines += ["", "## Descriptive: V2 mean utilities (n_h=10)", "", "| arm | mean U |", "| --- | ---: |"]
    for arm, val in v2_means.items():
        lines.append(f"| `{arm}` | {val:.6f} |")
    lines += [
        "",
        "## Descriptive: n_h curve (V3)",
        "",
        "| n_h | self | global | shuffled |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for nh, row in sorted(nh_table.items()):
        lines.append(f"| {nh} | {row['self']:.6f} | {row['global']:.6f} | {row['shuffled']:.6f} |")
    lines += ["", "## Per-level accuracy (all pools, descriptive)", "", "| level | accuracy |", "| ---: | ---: |"]
    for L in sorted(per_level):
        lines.append(f"| {L} | {per_level[L]:.4f} |")
    lines += [
        "",
        "## Descriptive: AUROC of answer_logprob vs correctness",
        "",
        f"- mean of finite per-level AUROCs: {auroc.get('mean')}",
        "",
        "| level | AUROC |",
        "| ---: | ---: |",
    ]
    for L, v in auroc.get("per_level", {}).items():
        lines.append(f"| {L} | {v} |")
    lines += [
        "",
        "## Descriptive: risk-coverage (TEST, ranked by answer_logprob)",
        "",
        "| coverage | n_kept | accuracy | risk |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for frac, row in rc.get("overall", {}).items():
        lines.append(
            f"| {frac} | {row['n_kept']} | {row['accuracy']:.4f} | {row['risk']:.4f} |"
        )
    lines += [
        "",
        "Per-level at 50% coverage:",
        "",
        "| level | n | n_kept | accuracy | risk |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for L, row in rc.get("per_level", {}).items():
        lines.append(
            f"| {L} | {row['n']} | {row['coverage_0.5_n']} | {row['accuracy']:.4f} | {row['risk']:.4f} |"
        )
    lines += [
        "",
        "## Claimed",
        "",
        f"- Labels A=`{primary['A']}` and C=`{primary['C']}` under `docs/M34A_V2_PREREG.md` on this cache.",
        "",
        "## NOT claimed",
        "",
        "- Privileged internal access in the mechanistic sense; Qwen residual κ; planning/games; other models or operations.",
        "",
        "## Scope",
        "",
        "- One small local model (`Qwen/Qwen2.5-3B-Instruct`), multiplication by one digit (`mul_n1`) only, "
        "levels 2–7, greedy float16, V3 primary, static warm-start history, n_h=10.",
        "",
    ]
    if primary["A"] == "SUPPORTED" and primary["C"] == "INTERNAL_VALUE":
        lines.append(
            "**Interpretation:** track-record value plus instance-level logprob value on this task."
        )
    elif primary["A"] == "SUPPORTED" and primary["C"] == "INTERNAL_NONE":
        lines.append(
            "**Interpretation:** track-record calibration only; no instance-level claim."
        )
    else:
        lines.append("**Interpretation:** no support at this scale.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--allow-dirty", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    git_commit = "TEST" if args.allow_dirty else require_clean_git()
    manifest = load_json(args.raw_dir / "manifest.json")
    cache_path = args.raw_dir / "responses.jsonl"
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
    utils_v2 = mean_utilities(
        test_rows, hist_by, PRIMARY_N_H, "V2", oracle, level_cal, edges, lp_cal, 34011
    )
    v2_means = {arm: sum(v) / len(v) for arm, v in utils_v2.items()}
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
    per_level = level_accuracy(rows)
    auroc_pl = {}
    finite = []
    for L in sorted({int(r["level"]) for r in rows}):
        subset = [r for r in rows if int(r["level"]) == L]
        val = mann_whitney_auroc(
            [float(r["answer_logprob"]) for r in subset],
            [bool(r["correct"]) for r in subset],
        )
        auroc_pl[str(L)] = val
        if val is not None:
            finite.append(val)
    auroc = {"per_level": auroc_pl, "mean": (sum(finite) / len(finite)) if finite else None}
    rc = risk_coverage(test_rows)

    report = render_report(
        manifest, primary, arm_means, v2_means, nh_table, per_level, auroc, rc, git_commit, cache_sha
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    write_json(
        args.raw_dir / "analysis_summary.json",
        {
            "primary": primary,
            "arm_means_v3_nh10": arm_means,
            "arm_means_v2_nh10": v2_means,
            "nh_table": {str(k): v for k, v in nh_table.items()},
            "per_level_accuracy": {str(k): v for k, v in per_level.items()},
            "auroc": auroc,
            "risk_coverage": rc,
            "cache_sha256": cache_sha,
            "analysis_git_commit": git_commit,
        },
    )
    print(json.dumps({"A": primary["A"], "C": primary["C"], "gate_a": primary["gate_a"], "gate_c": primary["gate_c"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
