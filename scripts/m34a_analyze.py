"""M34a offline analysis from cached responses only (refuses on hash mismatch)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    N_H_VALUES,
    N_HISTORY_DRAWS,
    PRIMARY_N_H,
    choose_action,
    conf_bin,
    load_json,
    paired_mean_ci,
    random_derangement,
    realized_utility,
    sha256_file,
    write_json,
)

RAW_DIR = ROOT / "reports" / "m34a_raw"
REPORT_PATH = ROOT / "reports" / "M34A_REPORT.md"


def load_cache(cache_path: Path) -> list[dict[str, Any]]:
    rows = []
    with cache_path.open("r", encoding="utf-8") as f:
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


def sample_history(
    hist_by_level: dict[int, list[dict[str, Any]]],
    n_h: int,
    rng,
) -> dict[int, list[dict[str, Any]]]:
    sample: dict[int, list[dict[str, Any]]] = {}
    for L, rows in hist_by_level.items():
        if len(rows) < n_h:
            raise ValueError(f"HIST level {L} has {len(rows)} < n_h={n_h}")
        idxs = rng.sample(range(len(rows)), n_h)
        sample[L] = [rows[i] for i in idxs]
    return sample


def self_p(sample: dict[int, list[dict[str, Any]]], n_h: int) -> dict[int, float]:
    return {
        L: (sum(1 for r in rows if r["correct"]) + 1) / (n_h + 2)
        for L, rows in sample.items()
    }


def global_p(sample: dict[int, list[dict[str, Any]]]) -> float:
    rows = [r for rs in sample.values() for r in rs]
    return sum(1 for r in rows if r["correct"]) / len(rows)


def verbal_cal_map(sample: dict[int, list[dict[str, Any]]]) -> dict[tuple[int, int], float]:
    buckets: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for rows in sample.values():
        for r in rows:
            buckets[conf_bin(float(r["parsed_confidence"]))].append(bool(r["correct"]))
    out: dict[tuple[int, int], float] = {}
    for key, vals in buckets.items():
        # Beta(1,1) posterior mean
        out[key] = (sum(vals) + 1) / (len(vals) + 2)
    return out


def p_for_arm(
    arm: str,
    row: dict[str, Any],
    *,
    g_p: float,
    self_ps: dict[int, float],
    shuffled_map: dict[int, int] | None,
    vcal: dict[tuple[int, int], float],
    oracle_level: dict[int, float],
) -> float | None:
    L = int(row["level"])
    if arm == "naive":
        return None
    if arm == "none":
        return 0.5
    if arm == "global":
        return g_p
    if arm == "self":
        return self_ps[L]
    if arm == "shuffled":
        assert shuffled_map is not None
        return self_ps[shuffled_map[L]]
    if arm == "verbal_raw":
        return float(row["parsed_confidence"]) / 100.0
    if arm == "verbal_cal":
        key = conf_bin(float(row["parsed_confidence"]))
        return vcal.get(key, g_p)
    if arm == "combined":
        key = conf_bin(float(row["parsed_confidence"]))
        return 0.5 * (self_ps[L] + vcal.get(key, g_p))
    if arm == "oracle_level":
        return oracle_level[L]
    if arm == "oracle_instance":
        return 1.0 if row["correct"] else 0.0
    raise ValueError(arm)


def utility_for_arm(
    arm: str,
    row: dict[str, Any],
    variant: str,
    **kwargs: Any,
) -> float:
    if arm == "naive":
        return realized_utility("answer", bool(row["correct"]))
    p = p_for_arm(arm, row, **kwargs)
    assert p is not None
    action = choose_action(float(p), variant)
    return realized_utility(action, bool(row["correct"]))


ARMS = (
    "naive",
    "none",
    "global",
    "self",
    "shuffled",
    "verbal_raw",
    "verbal_cal",
    "combined",
    "oracle_level",
    "oracle_instance",
)


def mean_utilities_over_draws(
    test_rows: list[dict[str, Any]],
    hist_by_level: dict[int, list[dict[str, Any]]],
    oracle_level: dict[int, float],
    n_h: int,
    variant: str,
    seed: int = 34011,
) -> dict[str, list[float]]:
    """Per TEST problem, mean utility over N_HISTORY_DRAWS; returns arm -> list[len(test)]."""
    import random

    rng = random.Random(seed)
    levels = sorted(hist_by_level)
    # accumulate sum utilities per problem index
    sums = {arm: [0.0] * len(test_rows) for arm in ARMS}
    for _draw in range(N_HISTORY_DRAWS):
        sample = sample_history(hist_by_level, n_h, rng)
        g_p = global_p(sample)
        s_ps = self_p(sample, n_h)
        der = random_derangement(rng, levels)
        vcal = verbal_cal_map(sample)
        kwargs = {
            "g_p": g_p,
            "self_ps": s_ps,
            "shuffled_map": der,
            "vcal": vcal,
            "oracle_level": oracle_level,
        }
        for i, row in enumerate(test_rows):
            for arm in ARMS:
                sums[arm][i] += utility_for_arm(arm, row, variant, **kwargs)
    return {arm: [s / N_HISTORY_DRAWS for s in vals] for arm, vals in sums.items()}


def ece_stated(rows: list[dict[str, Any]], n_bins: int = 10) -> float:
    # simple equal-width ECE on [0,1]
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for r in rows:
        p = float(r["parsed_confidence"]) / 100.0
        p = min(1.0, max(0.0, p))
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append((p, bool(r["correct"])))
    ece = 0.0
    n = len(rows)
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(1 for _, c in bucket if c) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece


def labels_from_utils(utils: dict[str, list[float]]) -> dict[str, Any]:
    def delta(a: str, b: str) -> list[float]:
        return [x - y for x, y in zip(utils[a], utils[b])]

    u_oracle = sum(utils["oracle_level"]) / len(utils["oracle_level"])
    u_global = sum(utils["global"]) / len(utils["global"])
    gate_ok = (u_oracle - u_global) >= 0.05

    if not gate_ok:
        a_label = "NOT_INFORMATIVE"
        self_global = paired_mean_ci(delta("self", "global"))
        self_shuf = paired_mean_ci(delta("self", "shuffled"))
    else:
        self_global = paired_mean_ci(delta("self", "global"))
        self_shuf = paired_mean_ci(delta("self", "shuffled"))
        if self_global["low"] > 0 and self_global["mean"] >= 0.05 and self_shuf["low"] > 0:
            a_label = "SUPPORTED"
        elif self_global["low"] > 0:
            a_label = "SUPPORTED_WEAK"
        else:
            a_label = "NOT_SUPPORTED"

    verbal_self = paired_mean_ci(delta("verbal_cal", "self"))
    if verbal_self["low"] > 0 and verbal_self["mean"] >= 0.03:
        b_label = "INTROSPECTION_VALUE"
    else:
        b_label = "INTROSPECTION_NONE"

    return {
        "gate_ok": gate_ok,
        "U_oracle_level": u_oracle,
        "U_global": u_global,
        "gate_gap": u_oracle - u_global,
        "A": a_label,
        "B": b_label,
        "self_minus_global": self_global,
        "self_minus_shuffled": self_shuf,
        "verbal_cal_minus_self": verbal_self,
    }


def render_report(
    manifest: dict[str, Any],
    primary: dict[str, Any],
    arm_means: dict[str, float],
    per_level_acc: dict[int, float],
    ece: float,
    git_commit: str,
    cache_sha: str,
    nh_table: dict[int, dict[str, float]],
    v2_means: dict[str, float],
) -> str:
    lines = [
        "# M34A report",
        "",
        f"**Primary verdict (V3, n_h={PRIMARY_N_H}):** A=`{primary['A']}` · B=`{primary['B']}` · GATE_OK=`{primary['gate_ok']}`",
        "",
        f"- cache sha256: `{cache_sha}`",
        f"- manifest model: `{manifest.get('model_id')}` revision `{manifest.get('revision')}`",
        f"- device/dtype: `{manifest.get('device_name')}` / `{manifest.get('dtype_default')}`",
        f"- torch/transformers: `{manifest.get('torch_version')}` / `{manifest.get('transformers_version')}`",
        f"- collection git commit: `{manifest.get('git_commit')}`",
        f"- analysis git commit: `{git_commit}`",
        f"- GATE gap U(oracle_level)-U(global) = {primary['gate_gap']:.6f} (need >= 0.05)",
        f"- self-global: mean={primary['self_minus_global']['mean']:.6f} CI=[{primary['self_minus_global']['low']:.6f}, {primary['self_minus_global']['high']:.6f}]",
        f"- self-shuffled: mean={primary['self_minus_shuffled']['mean']:.6f} CI=[{primary['self_minus_shuffled']['low']:.6f}, {primary['self_minus_shuffled']['high']:.6f}]",
        f"- verbal_cal-self: mean={primary['verbal_cal_minus_self']['mean']:.6f} CI=[{primary['verbal_cal_minus_self']['low']:.6f}, {primary['verbal_cal_minus_self']['high']:.6f}]",
        "",
        "## V3 mean utilities (n_h=10)",
        "",
        "| arm | mean U |",
        "| --- | ---: |",
    ]
    for arm, val in arm_means.items():
        lines.append(f"| `{arm}` | {val:.6f} |")
    lines += [
        "",
        "## Descriptive: V2 mean utilities (n_h=10)",
        "",
        "| arm | mean U |",
        "| --- | ---: |",
    ]
    for arm, val in v2_means.items():
        lines.append(f"| `{arm}` | {val:.6f} |")
    lines += [
        "",
        "## Descriptive: self mean U by n_h (V3)",
        "",
        "| n_h | self | global | shuffled |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for nh, row in sorted(nh_table.items()):
        lines.append(
            f"| {nh} | {row['self']:.6f} | {row['global']:.6f} | {row['shuffled']:.6f} |"
        )
    lines += [
        "",
        "## Per-level accuracy (all pools pooled descriptive)",
        "",
        "| level | accuracy |",
        "| ---: | ---: |",
    ]
    for L in sorted(per_level_acc):
        lines.append(f"| {L} | {per_level_acc[L]:.4f} |")
    lines += [
        "",
        f"ECE of stated confidence (TEST+HIST+CAL pooled): {ece:.6f}",
        "",
        "## Claimed",
        "",
        f"- Labels A=`{primary['A']}` and B=`{primary['B']}` under the frozen M34a prereg on this cache.",
        "",
        "## NOT claimed",
        "",
        "- Privileged internal access; Qwen mechanistic κ; agent planning/games (M34b/c); other models.",
        "",
        "## Scope",
        "",
        "- One Instruct Qwen (3B primary / 1.5B fallback), multiplication digit levels from the pilot, "
        "greedy decode, V3 primary, static warm-start history, n_h=10 primary.",
        "",
    ]
    if primary["A"] == "SUPPORTED" and primary["B"] == "INTROSPECTION_VALUE":
        lines.append(
            "**Interpretation:** real LLM shows state-dependent self-knowledge value plus instance-level calibrated verbal signal."
        )
    elif primary["A"] == "SUPPORTED" and primary["B"] == "INTROSPECTION_NONE":
        lines.append(
            "**Interpretation:** the value is plain calibration from a track record; no project-specific introspection claim."
        )
    else:
        lines.append("**Interpretation:** no support at this scale.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="M34a offline analysis")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    manifest_path = args.raw_dir / "manifest.json"
    cache_path = args.raw_dir / "responses.jsonl"
    manifest = load_json(manifest_path)
    cache_sha = sha256_file(cache_path)
    expected = manifest.get("cache_sha256")
    if expected != cache_sha:
        raise SystemExit(
            f"cache hash mismatch: manifest={expected} file={cache_sha}. Refusing analysis."
        )

    rows = load_cache(cache_path)
    by_pool = index_by_pool(rows)
    test_rows = sorted(by_pool["test"], key=lambda r: r["problem_id"])
    hist_rows = by_pool["hist"]
    cal_rows = by_pool["cal"]
    if len(test_rows) != 240:
        raise SystemExit(f"expected 240 TEST rows, got {len(test_rows)}")

    hist_by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in hist_rows:
        hist_by_level[int(r["level"])].append(r)
    oracle_level = level_accuracy(cal_rows)

    utils = mean_utilities_over_draws(
        test_rows, hist_by_level, oracle_level, PRIMARY_N_H, "V3", seed=34011
    )
    primary = labels_from_utils(utils)
    arm_means = {arm: sum(vals) / len(vals) for arm, vals in utils.items()}

    utils_v2 = mean_utilities_over_draws(
        test_rows, hist_by_level, oracle_level, PRIMARY_N_H, "V2", seed=34011
    )
    v2_means = {arm: sum(vals) / len(vals) for arm, vals in utils_v2.items()}

    nh_table: dict[int, dict[str, float]] = {}
    for nh in N_H_VALUES:
        u = mean_utilities_over_draws(
            test_rows, hist_by_level, oracle_level, nh, "V3", seed=34011 + nh
        )
        nh_table[nh] = {
            "self": sum(u["self"]) / len(u["self"]),
            "global": sum(u["global"]) / len(u["global"]),
            "shuffled": sum(u["shuffled"]) / len(u["shuffled"]),
        }

    per_level = level_accuracy(rows)
    ece = ece_stated(rows)

    try:
        import subprocess

        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        git_commit = "UNKNOWN"

    report = render_report(
        manifest,
        primary,
        arm_means,
        per_level,
        ece,
        git_commit,
        cache_sha,
        nh_table,
        v2_means,
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
            "ece": ece,
            "cache_sha256": cache_sha,
        },
    )
    print(json.dumps({"A": primary["A"], "B": primary["B"], "gate_ok": primary["gate_ok"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
