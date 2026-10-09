"""S2 finite-difference quadratic control.

The rule is docs/S2_PREREG.md. M1, M2, and BB_3 misses are read from the
S1 rows. This script measures only FDQ.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m30_catalog import catalog
from scripts.run_m29d_outcome import _load_model
from scripts.run_m30_pre_outcome import (
    DEVICE_NAME,
    DTYPE_NAME,
    MODEL_ID,
    MODEL_REVISION,
    assert_catalog_lock,
    assert_clean_git,
    load_directions,
)
from scripts.run_s1_steering import (
    CELL_HOOK,
    CELL_LAYER,
    EXPECTED_PREDICTION_SHA256,
    PREDICTIONS_PATH,
    SCALES,
    _resolve,
    _sha256,
    clip_alpha,
    margin_at,
    pick_root,
    solve_m2,
)

S1_RAW_PATH = ROOT / "reports" / "s1_raw" / "ROWS.json"
REPORT_PATH = ROOT / "reports" / "S2_REPORT.md"
RAW_PATH = ROOT / "reports" / "s2_raw" / "ROWS.json"
ARMS = ("new_identity", "anchor")
DECISION_SCALES = (4, 8)
DRAWS = 5000
SEED = 23001
METHODS = ("M1", "M2", "FDQ", "BB_3")


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _num(value: float | None) -> str:
    if value is None:
        return "undefined"
    return json.dumps(value)


def fdq_coefficients(f0: float, f_plus: float, f_minus: float) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in (f0, f_plus, f_minus)):
        raise RuntimeError("FDQ margins are non-finite")
    linear = (f_plus - f_minus) / 2.0
    quad = (f_plus + f_minus) / 2.0 - f0
    return linear, quad


def solve_fdq(linear: float, quad: float, target: float, s: float) -> float:
    """M2 root rule with (b, c) in place of (g, kappa)."""
    if not all(math.isfinite(value) for value in (linear, quad, target, s)):
        raise RuntimeError("FDQ solve inputs are non-finite")
    if quad == 0.0:
        if linear == 0.0:
            raise RuntimeError("FDQ linear and quadratic coefficients are both zero")
        alpha = target / linear
    else:
        disc = linear * linear + 4.0 * quad * target
        if disc < 0.0:
            alpha = -linear / (2.0 * quad)
        else:
            sqrt_disc = math.sqrt(disc)
            denom = 2.0 * quad
            alpha = pick_root((-linear + sqrt_disc) / denom, (-linear - sqrt_disc) / denom, float(s))
    if not math.isfinite(alpha):
        raise RuntimeError("FDQ alpha is non-finite")
    return clip_alpha(alpha, s)


def trimmed_mean(values: list[float], proportion: float = 0.1) -> float:
    ordered = sorted(values)
    drop = int(proportion * len(ordered))
    kept = ordered[drop : len(ordered) - drop]
    if not kept:
        raise RuntimeError("trimmed mean has no remaining values")
    return sum(kept) / len(kept)


def paired_median_ci(differences: list[float]) -> dict[str, float]:
    """Bootstrap the median. Same generator and percentile indexes as paired_mean_ci."""
    if len(differences) != 24:
        raise RuntimeError("median bootstrap requires 24 prompt differences")
    point = float(statistics.median(differences))
    rng = random.Random(SEED)
    medians: list[float] = []
    for _ in range(DRAWS):
        sample = [differences[rng.randrange(24)] for _ in range(24)]
        medians.append(float(statistics.median(sample)))
    medians.sort()
    low_index = int(math.floor(0.025 * (DRAWS - 1)))
    high_index = int(math.ceil(0.975 * (DRAWS - 1)))
    return {"high": medians[high_index], "low": medians[low_index], "median": point}


def prompt_level(rows: list[dict[str, Any]], arm: str, scale: int, field: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm or int(row["s"]) != int(scale):
            continue
        grouped[str(row["prompt_id"])].append(float(row[field]))
    if len(grouped) != 24:
        raise RuntimeError(f"{arm} s={scale} does not have 24 prompts")
    scores = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{prompt_id} does not have 3 finite cell misses")
        scores.append(sum(values) / len(values))
    return scores


def predicates(median_m2: float, median_fdq: float, ci_low: float) -> tuple[bool, bool]:
    white = median_m2 <= 0.75 * median_fdq and ci_low > 0.0
    no_advantage = median_fdq <= median_m2
    if white and no_advantage:
        raise RuntimeError("a condition meets both decision predicates")
    return white, no_advantage


def verdict_from_counts(n_white: int, n_no_advantage: int) -> str:
    if n_white >= 3 and n_no_advantage >= 3:
        raise RuntimeError("WHITE_BOX_ADVANTAGE and NO_ADVANTAGE both reached 3")
    if n_white >= 3:
        return "WHITE_BOX_ADVANTAGE"
    if n_no_advantage >= 3:
        return "NO_ADVANTAGE"
    return "MIXED"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_scale = []
    n_white = 0
    n_no_advantage = 0
    for arm in ARMS:
        for scale in SCALES:
            scores = {method: prompt_level(rows, arm, scale, f"miss_{method.lower()}") for method in METHODS}
            differences = [fdq - m2 for fdq, m2 in zip(scores["FDQ"], scores["M2"])]
            interval = paired_median_ci(differences)
            white, no_advantage = predicates(float(statistics.median(scores["M2"])), float(statistics.median(scores["FDQ"])), interval["low"])
            if scale in DECISION_SCALES:
                n_white += int(white)
                n_no_advantage += int(no_advantage)
            per_scale.append(
                {
                    "arm": arm,
                    "ci_high": interval["high"],
                    "ci_low": interval["low"],
                    "decision_condition": scale in DECISION_SCALES,
                    "mean": {method: sum(scores[method]) / len(scores[method]) for method in METHODS},
                    "median": {method: float(statistics.median(scores[method])) for method in METHODS},
                    "median_difference_fdq_minus_m2": interval["median"],
                    "no_advantage": no_advantage,
                    "s": scale,
                    "trimmed_mean": {method: trimmed_mean(scores[method]) for method in METHODS},
                    "white": white,
                }
            )
    n_decision = len(ARMS) * len(DECISION_SCALES)
    n_neither = n_decision - n_white - n_no_advantage
    return {
        "n_decision": n_decision,
        "n_neither": n_neither,
        "n_no_advantage": n_no_advantage,
        "n_white": n_white,
        "per_scale": per_scale,
        "verdict": verdict_from_counts(n_white, n_no_advantage),
    }


def _physical_rows(s1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in s1_rows if int(row["s"]) == 1]
    if len(rows) != 144:
        raise RuntimeError(f"expected 144 S1 physical rows, got {len(rows)}")
    return rows


def _s1_by_key(s1_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    indexed = {}
    for row in s1_rows:
        key = (str(row["arm"]), str(row["prompt_id"]), str(row["intervention_id"]), int(row["s"]))
        if key in indexed:
            raise RuntimeError(f"duplicate S1 row {key}")
        indexed[key] = row
    return indexed


def _timed_margins(model: Any, tokens: Any, hook: str, direction: Any, positive_id: int, negative_id: int) -> tuple[float, float, float, float]:
    started = time.perf_counter()
    f0 = margin_at(model, tokens, hook, direction, 0.0, positive_id, negative_id)
    after_f0 = time.perf_counter()
    f_plus = margin_at(model, tokens, hook, direction, 1.0, positive_id, negative_id)
    after_plus = time.perf_counter()
    f_minus = margin_at(model, tokens, hook, direction, -1.0, positive_id, negative_id)
    after_minus = time.perf_counter()
    elapsed = (after_f0 - started) + (after_plus - after_f0) + (after_minus - after_plus)
    return f0, f_plus, f_minus, elapsed


def measure_fdq(model: Any, physical: list[dict[str, Any]], s1_index: dict[tuple[str, str, str, int], dict[str, Any]], tokens_of: dict[str, Any], directions: dict[str, Any], resolved: dict[str, tuple[int, int]]) -> tuple[list[dict[str, Any]], list[float]]:
    first = physical[0]
    positive_id, negative_id = resolved[str(first["arm"])]
    _timed_margins(model, tokens_of[first["prompt_id"]], first["hook"], directions[first["arm"]], positive_id, negative_id)
    rows: list[dict[str, Any]] = []
    costs: list[float] = []
    baselines: dict[tuple[str, str], float] = {}
    for index, physical_row in enumerate(physical, start=1):
        arm = str(physical_row["arm"])
        prompt_id = str(physical_row["prompt_id"])
        cell = str(physical_row["intervention_id"])
        positive_id, negative_id = resolved[arm]
        f0, f_plus, f_minus, elapsed = _timed_margins(
            model,
            tokens_of[prompt_id],
            physical_row["hook"],
            directions[arm],
            positive_id,
            negative_id,
        )
        costs.append(elapsed)
        cache_key = (arm, prompt_id)
        if cache_key not in baselines:
            baselines[cache_key] = f0
        elif baselines[cache_key] != f0:
            raise RuntimeError("unperturbed margin changed across cells of one prompt")
        linear, quad = fdq_coefficients(f0, f_plus, f_minus)
        for scale in SCALES:
            source = s1_index[(arm, prompt_id, cell, int(scale))]
            target = float(source["T"])
            alpha = solve_fdq(linear, quad, target, float(scale))
            effect = margin_at(model, tokens_of[prompt_id], physical_row["hook"], directions[arm], alpha, positive_id, negative_id) - f0
            if not math.isfinite(effect) or target == 0.0:
                raise RuntimeError("FDQ verification is undefined")
            rows.append(
                {
                    "alpha_fdq": alpha,
                    "arm": arm,
                    "b": linear,
                    "c": quad,
                    "f0": f0,
                    "g": float(source["g"]),
                    "hook": physical_row["hook"],
                    "intervention_id": cell,
                    "miss_bb_3": float(source["methods"]["BB_3"]["miss"]),
                    "miss_fdq": abs(effect - target) / abs(target),
                    "miss_m1": float(source["methods"]["M1"]["miss"]),
                    "miss_m2": float(source["methods"]["M2"]["miss"]),
                    "prompt_id": prompt_id,
                    "s": int(scale),
                    "T": target,
                }
            )
        if index % 6 == 0:
            print(f"fdq {index}/{len(physical)} {arm} {prompt_id}", flush=True)
    return rows, costs


def render_report(payload: dict[str, Any]) -> str:
    scored = payload["scored"]
    lines = [
        "# S2 finite-difference control",
        "",
        f"Verdict: `{scored['verdict']}`.",
        "",
        f"Conditions in class: `WHITE` {scored['n_white']}, `NO_ADVANTAGE` {scored['n_no_advantage']}, `NEITHER` {scored['n_neither']}.",
        "",
        f"Pre-registration `docs/S2_PREREG.md`. Runner commit `{payload['git_commit']}`.",
        f"FDQ seconds `{_num(payload['cost_fdq'])}`. M2 seconds `{_num(payload['cost_m2'])}`.",
        "FDQ forward-equivalents: 3. The M2 time is the S1 median and was not measured again.",
        "",
        "## Median miss",
        "",
        "| arm | s | M1 | M2 | FDQ | BB_3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scored["per_scale"]:
        lines.append(
            "| {arm} | {s} | {m1} | {m2} | {fdq} | {bb3} |".format(
                arm=row["arm"],
                s=row["s"],
                m1=_num(row["median"]["M1"]),
                m2=_num(row["median"]["M2"]),
                fdq=_num(row["median"]["FDQ"]),
                bb3=_num(row["median"]["BB_3"]),
            )
        )
    lines.extend(
        [
            "",
            "## Median of miss_FDQ - miss_M2",
            "",
            "| arm | s | median difference | CI low | CI high | W | N | in decision |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for row in scored["per_scale"]:
        lines.append(
            "| {arm} | {s} | {median} | {low} | {high} | {white} | {none} | {decision} |".format(
                arm=row["arm"],
                s=row["s"],
                median=_num(row["median_difference_fdq_minus_m2"]),
                low=_num(row["ci_low"]),
                high=_num(row["ci_high"]),
                white=json.dumps(bool(row["white"])),
                none=json.dumps(bool(row["no_advantage"])),
                decision=json.dumps(bool(row["decision_condition"])),
            )
        )
    lines.extend(
        [
            "",
            "Prompt-level means of three cells. Bootstrap of the median: 24 prompts, 5000 draws, seed 23001. Scales 1 and 2 are descriptive.",
            "",
            "## Mean and 10% trimmed mean",
            "",
            "| arm | s | method | mean | trimmed mean |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in scored["per_scale"]:
        for method in METHODS:
            lines.append(
                "| {arm} | {s} | {method} | {mean} | {trimmed} |".format(
                    arm=row["arm"],
                    s=row["s"],
                    method=method,
                    mean=_num(row["mean"][method]),
                    trimmed=_num(row["trimmed_mean"][method]),
                )
            )
    lines.extend(["", f"Runtime seconds: {_num(payload['runtime_seconds'])}.", ""])
    return "\n".join(lines)


def _load_s1() -> dict[str, Any]:
    if not S1_RAW_PATH.is_file():
        raise SystemExit("S1 raw rows are missing")
    payload = json.loads(S1_RAW_PATH.read_text(encoding="utf-8"))
    if payload.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256:
        raise SystemExit("S1 prediction hash does not match the frozen file")
    if _sha256(PREDICTIONS_PATH) != EXPECTED_PREDICTION_SHA256:
        raise SystemExit("prediction sha256 mismatch")
    if int(payload.get("k_eq")) != 3:
        raise SystemExit("S1 equal-cost black box is not BB_3")
    return payload


def _refuse_overwrite() -> None:
    if REPORT_PATH.exists() or RAW_PATH.exists():
        raise SystemExit("S2 artifacts already exist; refusing to overwrite")


def main() -> None:
    _refuse_overwrite()
    catalog_digest = assert_catalog_lock()
    s1 = _load_s1()
    physical = _physical_rows(s1["rows"])
    s1_index = _s1_by_key(s1["rows"])
    for row in physical:
        cell = str(row["intervention_id"])
        if cell not in CELL_HOOK or row["hook"] != CELL_HOOK[cell]:
            raise RuntimeError(f"S1 row is not a frozen cell: {cell}")
    commit = assert_clean_git()
    direction_d1, direction_d2 = load_directions()
    directions = {"anchor": direction_d1, "new_identity": direction_d2}
    started = time.perf_counter()
    model, device = _load_model()
    if device != DEVICE_NAME:
        raise SystemExit("S2 must stay on CPU")
    resolved = _resolve(model.tokenizer)
    by_prompt = {row["prompt_id"]: row for row in catalog()}
    tokens_of = {}
    for row in physical:
        prompt_id = row["prompt_id"]
        if prompt_id not in tokens_of:
            source = by_prompt[prompt_id]
            if source["partition"] != "holdout":
                raise RuntimeError(f"catalog partition differs for {prompt_id}")
            tokens_of[prompt_id] = model.to_tokens(source["text"], prepend_bos=True)
    rows, costs = measure_fdq(model, physical, s1_index, tokens_of, directions, resolved)
    if len(costs) != 144:
        raise RuntimeError("FDQ cost sample is not 144 rows")
    cost_fdq = float(statistics.median(costs))
    scored = summarize(rows)
    runtime_seconds = time.perf_counter() - started
    payload = {
        "catalog_sha256": catalog_digest,
        "cost_fdq": cost_fdq,
        "cost_fdq_forward_equivalents": 3,
        "cost_m2": float(s1["cost"]["cost_m2"]),
        "device": DEVICE_NAME,
        "dtype": DTYPE_NAME,
        "git_commit": commit,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prediction_sha256": EXPECTED_PREDICTION_SHA256,
        "rows": rows,
        "runtime_seconds": runtime_seconds,
        "scored": scored,
        "timing_seconds": costs,
    }
    report = render_report(payload)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_PATH.with_suffix(".json.tmp")
    temporary.write_text(_dump(payload), encoding="utf-8")
    os.replace(temporary, RAW_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"status": "S2_COMPLETE", "verdict": scored["verdict"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
