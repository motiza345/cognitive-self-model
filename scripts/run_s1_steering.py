"""S1 steering run. Formulas and the decision rule are docs/S1_PREREG.md.

g and kappa are read from the frozen M30 predictions. Timed gradient and
Hessian-vector products are discarded and do not choose alpha.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.m29d_measurement as measurement
from scripts.m30_catalog import catalog
from scripts.run_m29d_outcome import _as_logits, _load_model
from scripts.run_m30_pre_outcome import (
    ANCHOR_CELLS,
    DEVICE_NAME,
    DTYPE_NAME,
    LOCKED_CATALOG_SHA256,
    MANIFEST_PATH,
    MODEL_ID,
    MODEL_REVISION,
    NEW_CELLS,
    PREDICTIONS_PATH,
    assert_catalog_lock,
    assert_clean_git,
    load_directions,
    resolve_anchor_tokens,
    resolve_new_identity_tokens,
)
from src.cognitive_self_model.m23.m22_reuse import logit_margin, make_resid_hook
from src.cognitive_self_model.m23.stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_mean_ci

if BOOTSTRAP_DRAWS != 5000 or BOOTSTRAP_SEED != 23001:
    raise RuntimeError("bootstrap constants are not the pre-registered values")

EXPECTED_PREDICTION_SHA256 = "f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5"
OUTCOMES_PATH = ROOT / "reports" / "m30_raw" / "OUTCOMES.json"
REPORT_PATH = ROOT / "reports" / "S1_REPORT.md"
RAW_PATH = ROOT / "reports" / "s1_raw" / "ROWS.json"
SCALES = (1, 2, 4, 8)
DECISION_SCALES = (4, 8)
BB_KS = (1, 2, 3, 4, 6, 8)
REDUCTION_MIN = 0.25
ARMS = ("new_identity", "anchor")
CELL_LAYER = {cell[0]: cell[1] for cell in (*NEW_CELLS, *ANCHOR_CELLS)}
CELL_HOOK = {cell[0]: cell[2] for cell in (*NEW_CELLS, *ANCHOR_CELLS)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _num(value: float | None) -> str:
    if value is None:
        return "undefined"
    return json.dumps(value)


def assert_prediction_lock(path: Path = PREDICTIONS_PATH) -> tuple[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise SystemExit("prediction file is missing")
    digest = _sha256(path)
    if digest != EXPECTED_PREDICTION_SHA256:
        raise SystemExit(f"prediction sha256 mismatch: {digest}")
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise SystemExit("prediction artifact is not a list")
    return digest, predictions


def clip_alpha(alpha: float, s: float) -> float:
    limit = 4.0 * float(s)
    if alpha > limit:
        return limit
    if alpha < -limit:
        return -limit
    return float(alpha)


def pick_root(root_a: float, root_b: float, s: float) -> float:
    distance_a = abs(root_a - s)
    distance_b = abs(root_b - s)
    if distance_a < distance_b:
        return root_a
    if distance_b < distance_a:
        return root_b
    return min(root_a, root_b)


def solve_m2(g: float, kappa: float, s: float) -> float:
    """Real root of kappa*a^2 + g*a = s*g nearest to s, then clip."""
    if not math.isfinite(g) or not math.isfinite(kappa) or g == 0.0:
        raise RuntimeError("M2 inputs are undefined")
    if kappa == 0.0:
        alpha = float(s)
    else:
        target = float(s) * float(g)
        disc = g * g + 4.0 * kappa * target
        if disc < 0.0:
            alpha = -g / (2.0 * kappa)
        else:
            sqrt_disc = math.sqrt(disc)
            denom = 2.0 * kappa
            alpha = pick_root((-g + sqrt_disc) / denom, (-g - sqrt_disc) / denom, float(s))
    if not math.isfinite(alpha):
        raise RuntimeError("M2 alpha is non-finite")
    return clip_alpha(alpha, s)


def probe_sequence(evaluate: Callable[[float], float], target: float, gbar: float, n_probes: int) -> tuple[list[dict[str, float]], list[float]]:
    """k probes, then the secant estimate from the latest probe and (0, 0)."""
    if n_probes < 1 or not math.isfinite(gbar) or gbar == 0.0 or not math.isfinite(target):
        raise RuntimeError("black-box probe setup is undefined")
    alpha = target / gbar
    probes: list[dict[str, float]] = []
    estimates: list[float] = []
    for _ in range(n_probes):
        if not math.isfinite(alpha):
            raise RuntimeError("black-box alpha is non-finite")
        effect = float(evaluate(alpha))
        if not math.isfinite(effect):
            raise RuntimeError("black-box probe margin is non-finite")
        estimate = alpha if effect == 0.0 else target * alpha / effect
        if not math.isfinite(estimate):
            raise RuntimeError("secant estimate is non-finite")
        probes.append({"alpha": alpha, "effect": effect})
        estimates.append(estimate)
        alpha = estimate
    return probes, estimates


def gbar_from_update(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    """Mean observed effect at alpha +1. Holdout rows are skipped."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("partition") != "update":
            continue
        grouped[(str(row["arm"]), str(row["intervention_id"]))].append(float(row["observed_effect"]))
    means: dict[tuple[str, str], float] = {}
    expected_keys = {(arm, cell[0]) for arm, cells in (("new_identity", NEW_CELLS), ("anchor", ANCHOR_CELLS)) for cell in cells}
    if set(grouped) != expected_keys:
        raise RuntimeError("UPDATE cells for gbar are not the frozen six")
    for key, values in grouped.items():
        if len(values) != 12 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"gbar inputs are undefined for {key}")
        mean = sum(values) / len(values)
        if mean == 0.0 or not math.isfinite(mean):
            raise RuntimeError(f"gbar is undefined for {key}")
        means[key] = mean
    return means


def executed_ks(k_eq: int) -> tuple[int, ...]:
    ordered = list(BB_KS)
    if k_eq not in ordered:
        ordered.append(k_eq)
    return tuple(ordered)


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("mean requires values")
    return sum(values) / len(values)


def prompt_misses(rows: list[dict[str, Any]], arm: str, scale: int, method: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm or int(row["s"]) != int(scale):
            continue
        grouped[str(row["prompt_id"])].append(float(row["methods"][method]["miss"]))
    if len(grouped) != 24:
        raise RuntimeError(f"{arm} s={scale} {method} does not have 24 prompts")
    scores = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{prompt_id} does not have 3 finite cell misses")
        scores.append(_mean(values))
    return scores


def contrast(baseline: list[float], challenger: list[float]) -> dict[str, Any]:
    """Paired difference baseline minus challenger. A new bootstrap at seed 23001."""
    if len(baseline) != 24 or len(challenger) != 24:
        raise RuntimeError("contrast requires 24 prompt scores")
    diffs = [base - other for base, other in zip(baseline, challenger)]
    interval = paired_mean_ci(diffs)
    mean_baseline = _mean(baseline)
    mean_challenger = _mean(challenger)
    if abs(float(interval["mean"]) - _mean(diffs)) > 1e-9:
        raise RuntimeError("bootstrap mean does not equal the paired difference")
    reduction = None if not mean_baseline > 0.0 else 1.0 - (mean_challenger / mean_baseline)
    return {
        "ci_high": float(interval["high"]),
        "ci_low": float(interval["low"]),
        "delta": _mean(diffs),
        "mean_baseline": mean_baseline,
        "mean_challenger": mean_challenger,
        "median_baseline": float(statistics.median(baseline)),
        "median_challenger": float(statistics.median(challenger)),
        "n_prompts": 24,
        "reduction": reduction,
    }


def beats_m1(summary: dict[str, Any]) -> bool:
    reduction = summary["reduction"]
    return reduction is not None and reduction >= REDUCTION_MIN and float(summary["ci_low"]) > 0.0


def decide(pairs: list[dict[str, Any]]) -> str:
    """First matching clause of the pre-registered rule. `pairs` is the set S."""
    if len(pairs) != 4:
        raise RuntimeError("the decision set S has four arm-scale pairs")
    all_beat = all(bool(pair["beat"]) for pair in pairs)
    all_m2_not_worse = all(float(pair["median_m2"]) <= float(pair["median_bb"]) for pair in pairs)
    all_bb_lower = all(float(pair["median_bb"]) < float(pair["median_m2"]) for pair in pairs)
    if all_beat and all_m2_not_worse:
        return "GO"
    if all_beat and all_bb_lower:
        return "PIVOT"
    if not all_beat:
        return "STOP"
    return "MIXED"


def summarize(rows: list[dict[str, Any]], k_eq: int) -> dict[str, Any]:
    ks = executed_ks(k_eq)
    bb_eq = f"BB_{k_eq}"
    per_scale = []
    decision_pairs = []
    for arm in ARMS:
        for scale in SCALES:
            m1 = prompt_misses(rows, arm, scale, "M1")
            m2 = prompt_misses(rows, arm, scale, "M2")
            versus_m1 = contrast(m1, m2)
            versus_eq = contrast(prompt_misses(rows, arm, scale, bb_eq), m2)
            versus_bb2 = contrast(prompt_misses(rows, arm, scale, "BB_2"), m2)
            medians = {"M1": float(statistics.median(m1)), "M2": float(statistics.median(m2))}
            for k in ks:
                medians[f"BB_{k}"] = float(statistics.median(prompt_misses(rows, arm, scale, f"BB_{k}")))
            beat = beats_m1(versus_m1)
            record = {
                "arm": arm,
                "beat_m1": beat,
                "median_miss": medians,
                "s": scale,
                "versus_bb2": versus_bb2,
                "versus_bb_eq": versus_eq,
                "versus_m1": versus_m1,
            }
            per_scale.append(record)
            if scale in DECISION_SCALES:
                decision_pairs.append(
                    {
                        "arm": arm,
                        "beat": beat,
                        "median_bb": medians[bb_eq],
                        "median_m2": medians["M2"],
                        "s": scale,
                    }
                )
    return {"decision": decide(decision_pairs), "k_eq": k_eq, "pairs": decision_pairs, "per_scale": per_scale}


def margin_at(model: Any, tokens: Any, hook: str, direction: Any, alpha: float, positive_id: int, negative_id: int) -> float:
    import torch

    if not math.isfinite(alpha):
        raise RuntimeError("alpha is non-finite")
    with torch.no_grad():
        if float(alpha) == 0.0:
            value = logit_margin(_as_logits(model(tokens)), positive_id, negative_id)
        else:
            recorder: dict[str, Any] = {}
            vector = torch.tensor(direction, dtype=torch.float32)
            hook_fn = make_resid_hook(float(alpha), vector, recorder)
            value = logit_margin(
                _as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook, hook_fn)])),
                positive_id,
                negative_id,
            )
            if int(recorder.get("fired", 0)) != 1:
                raise RuntimeError("intervention hook did not fire once")
            if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
                raise RuntimeError("intervention did not stay on the last token")
    if not math.isfinite(value):
        raise RuntimeError("measured margin is non-finite")
    return float(value)


def _restore_measurement(saved_hooks: dict[str, str], saved_positive: int, saved_negative: int) -> None:
    measurement.HOOKS.clear()
    measurement.HOOKS.update(saved_hooks)
    measurement.POSITIVE_ID = saved_positive
    measurement.NEGATIVE_ID = saved_negative


def time_forward(model: Any, tokens: Any, hook: str, direction: Any, positive_id: int, negative_id: int) -> float:
    started = time.perf_counter()
    margin_at(model, tokens, hook, direction, 1.0, positive_id, negative_id)
    return time.perf_counter() - started


def time_gradient(model: Any, tokens: Any, hook: str, direction: Any, positive_id: int, negative_id: int) -> float:
    """First-order margin gradient. The value is discarded by the caller."""
    import torch

    started = time.perf_counter()
    captured: dict[str, Any] = {}

    def capture(resid: Any, hook_obj: Any) -> Any:
        del hook_obj
        if not resid.requires_grad:
            raise RuntimeError("hook residual does not require grad")
        captured["resid"] = resid
        return resid

    direction_t = torch.tensor(direction, dtype=torch.float32, device=next(model.parameters()).device)
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = _as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook, capture)]))
        margin = logits[0, -1, int(positive_id)] - logits[0, -1, int(negative_id)]
        resid = captured.get("resid")
        if resid is None:
            raise RuntimeError("gradient hook did not fire")
        grad = torch.autograd.grad(margin, resid, create_graph=False, retain_graph=False)[0]
        dotted = torch.sum(grad[0, -1] * direction_t)
        if not torch.isfinite(dotted):
            raise RuntimeError("timed gradient is non-finite")
    model.zero_grad(set_to_none=True)
    return time.perf_counter() - started


def time_hvp(model: Any, tokens: Any, layer: int, hook: str, direction: Any, positive_id: int, negative_id: int) -> float:
    """Time measure_kappa only. Restore hooks outside the clock. Discard the values."""
    key = f"M22.1-D1-L{layer}"
    saved_hooks = dict(measurement.HOOKS)
    saved_positive = measurement.POSITIVE_ID
    saved_negative = measurement.NEGATIVE_ID
    measurement.HOOKS[key] = hook
    measurement.POSITIVE_ID = int(positive_id)
    measurement.NEGATIVE_ID = int(negative_id)
    try:
        started = time.perf_counter()
        measurement.measure_kappa(model, tokens, layer, direction)
        return time.perf_counter() - started
    finally:
        _restore_measurement(saved_hooks, saved_positive, saved_negative)


def _arm_tokens(arm: str, resolved: dict[str, tuple[int, int]]) -> tuple[int, int]:
    return resolved[arm]


def _direction_for(arm: str, direction_d1: Any, direction_d2: Any) -> Any:
    if arm == "new_identity":
        return direction_d2
    if arm == "anchor":
        return direction_d1
    raise RuntimeError(f"unknown arm {arm}")


def cost_summary(samples: list[dict[str, float]], k_eq: int) -> dict[str, Any]:
    if len(samples) != 144:
        raise RuntimeError(f"expected 144 timing rows, got {len(samples)}")
    cost_forward = float(statistics.median([row["forward_seconds"] for row in samples]))
    cost_m1 = float(statistics.median([row["m1_seconds"] for row in samples]))
    cost_m2 = float(statistics.median([row["m2_seconds"] for row in samples]))
    if not cost_forward > 0.0:
        raise RuntimeError("forward cost is not positive")
    methods = []
    for name, seconds, forwards in (
        ("M1", cost_m1, cost_m1 / cost_forward),
        ("M2", cost_m2, cost_m2 / cost_forward),
    ):
        methods.append({"forward_equivalents": forwards, "method": name, "seconds": seconds})
    for k in executed_ks(k_eq):
        methods.append({"forward_equivalents": float(k), "method": f"BB_{k}", "seconds": float(k) * cost_forward})
    return {
        "cost_forward": cost_forward,
        "cost_m1": cost_m1,
        "cost_m2": cost_m2,
        "k_eq": k_eq,
        "methods": methods,
        "n_rows": 144,
    }


def render_report(payload: dict[str, Any]) -> str:
    scored = payload["scored"]
    cost = payload["cost"]
    lines = [
        "# S1 steering",
        "",
        f"Decision: `{scored['decision']}`.",
        "",
        f"Pre-registration `docs/S1_PREREG.md`. Runner commit `{payload['git_commit']}`.",
        f"Predictions `{payload['prediction_sha256']}`. Catalog `{payload['catalog_sha256']}`.",
        f"Equal-cost black box is `BB_{scored['k_eq']}` "
        f"(`ceil({_num(cost['cost_m2'])} / {_num(cost['cost_forward'])})`).",
        "",
        "## Decision pairs",
        "",
        "| arm | s | beats M1 | median M2 | median BB_eq | reduction | CI low | CI high |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key = {(row["arm"], row["s"]): row for row in scored["per_scale"]}
    for pair in scored["pairs"]:
        row = by_key[(pair["arm"], pair["s"])]
        versus = row["versus_m1"]
        lines.append(
            "| {arm} | {s} | {beat} | {m2} | {bb} | {reduction} | {low} | {high} |".format(
                arm=pair["arm"],
                s=pair["s"],
                beat=json.dumps(bool(pair["beat"])),
                m2=_num(pair["median_m2"]),
                bb=_num(pair["median_bb"]),
                reduction=_num(versus["reduction"]),
                low=_num(versus["ci_low"]),
                high=_num(versus["ci_high"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-scale misses",
            "",
            "| arm | s | Delta M1-M2 | CI low | CI high | reduction | median M1 | median M2 | median BB_eq | median BB_2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scored["per_scale"]:
        versus = row["versus_m1"]
        lines.append(
            "| {arm} | {s} | {delta} | {low} | {high} | {reduction} | {m1} | {m2} | {eq} | {bb2} |".format(
                arm=row["arm"],
                s=row["s"],
                delta=_num(versus["delta"]),
                low=_num(versus["ci_low"]),
                high=_num(versus["ci_high"]),
                reduction=_num(versus["reduction"]),
                m1=_num(row["median_miss"]["M1"]),
                m2=_num(row["median_miss"]["M2"]),
                eq=_num(row["median_miss"][f"BB_{scored['k_eq']}"]),
                bb2=_num(row["median_miss"]["BB_2"]),
            )
        )
    lines.extend(
        [
            "",
            "Prompt-level means of three cells, 24 prompts, 5000 draws, seed 23001. "
            "Delta is `miss_M1 - miss_M2`.",
            "",
            "## Accuracy versus cost",
            "",
            "| method | seconds | forward-equivalents | new s=1 | new s=2 | new s=4 | new s=8 | anchor s=1 | anchor s=2 | anchor s=4 | anchor s=8 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in cost["methods"]:
        name = method["method"]
        cells = []
        for arm in ARMS:
            for scale in SCALES:
                cells.append(_num(by_key[(arm, scale)]["median_miss"][name]))
        lines.append(
            "| {name} | {seconds} | {fe} | {cells} |".format(
                name=name,
                seconds=_num(method["seconds"]),
                fe=_num(method["forward_equivalents"]),
                cells=" | ".join(cells),
            )
        )
    lines.extend(
        [
            "",
            "Seconds for M1 and M2 are median wall-clock per holdout row. "
            "Black-box seconds are `k` times the median forward. "
            "Verification forwards and `f0` are not charged.",
            "",
            f"Timed rows: {cost['n_rows']}. Runtime seconds: {_num(payload['runtime_seconds'])}.",
            "",
        ]
    )
    return "\n".join(lines)


def _holdout_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in predictions if row.get("partition") == "holdout"]
    if len(rows) != 144:
        raise RuntimeError(f"expected 144 holdout predictions, got {len(rows)}")
    for row in rows:
        if float(row["alpha"]) != 1.0 or not math.isfinite(float(row["g"])) or float(row["g"]) == 0.0:
            raise RuntimeError(f"holdout g is undefined for {row['prompt_id']}")
        if not math.isfinite(float(row["kappa"])):
            raise RuntimeError(f"holdout kappa is undefined for {row['prompt_id']}")
        cell = str(row["intervention_id"])
        if cell not in CELL_HOOK or row["hook"] != CELL_HOOK[cell] or str(row["arm"]) not in ARMS:
            raise RuntimeError(f"holdout row is not a frozen cell: {cell}")
    return rows


def _time_rows(model: Any, rows: list[dict[str, Any]], tokens_of: dict[str, Any], directions: dict[str, Any], resolved: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    samples = []
    for row in rows:
        arm = str(row["arm"])
        positive_id, negative_id = _arm_tokens(arm, resolved)
        samples.append(
            {
                "arm": arm,
                "forward_seconds": time_forward(model, tokens_of[row["prompt_id"]], row["hook"], directions[arm], positive_id, negative_id),
                "intervention_id": row["intervention_id"],
                "m1_seconds": time_gradient(model, tokens_of[row["prompt_id"]], row["hook"], directions[arm], positive_id, negative_id),
                "m2_seconds": time_hvp(
                    model,
                    tokens_of[row["prompt_id"]],
                    CELL_LAYER[row["intervention_id"]],
                    row["hook"],
                    directions[arm],
                    positive_id,
                    negative_id,
                ),
                "prompt_id": row["prompt_id"],
            }
        )
    return samples


def _evaluate_rows(
    model: Any,
    rows: list[dict[str, Any]],
    tokens_of: dict[str, Any],
    directions: dict[str, Any],
    resolved: dict[str, tuple[int, int]],
    gbar: dict[tuple[str, str], float],
    n_probes: int,
    ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], float] = {}
    scored_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        arm = str(row["arm"])
        prompt_id = str(row["prompt_id"])
        positive_id, negative_id = _arm_tokens(arm, resolved)
        tokens = tokens_of[prompt_id]
        cache_key = (arm, prompt_id)
        if cache_key not in baselines:
            baselines[cache_key] = margin_at(model, tokens, row["hook"], directions[arm], 0.0, positive_id, negative_id)
        f0 = baselines[cache_key]
        g = float(row["g"])
        kappa = float(row["kappa"])
        cell_gbar = gbar[(arm, str(row["intervention_id"]))]

        def evaluate(alpha: float, hook: str = str(row["hook"])) -> float:
            return margin_at(model, tokens, hook, directions[arm], alpha, positive_id, negative_id) - f0

        for scale in SCALES:
            target = float(scale) * g
            alpha_m1 = float(scale)
            alpha_m2 = solve_m2(g, kappa, float(scale))
            probes, estimates = probe_sequence(evaluate, target, cell_gbar, n_probes)
            methods: dict[str, dict[str, float]] = {}
            chosen = {"M1": alpha_m1, "M2": alpha_m2}
            for k in ks:
                chosen[f"BB_{k}"] = estimates[k - 1]
            for name, alpha in chosen.items():
                effect = evaluate(alpha)
                methods[name] = {"alpha": alpha, "effect": effect, "miss": abs(effect - target) / abs(target)}
            scored_rows.append(
                {
                    "arm": arm,
                    "f0": f0,
                    "family": row["family"],
                    "g": g,
                    "gbar": cell_gbar,
                    "hook": row["hook"],
                    "intervention_id": row["intervention_id"],
                    "kappa": kappa,
                    "methods": methods,
                    "probes": probes,
                    "prompt_id": prompt_id,
                    "s": scale,
                    "T": target,
                }
            )
        if index % 6 == 0:
            print(f"steered {index}/{len(rows)} {arm} {prompt_id}", flush=True)
    return scored_rows


def _refuse_overwrite() -> None:
    existing = [path for path in (REPORT_PATH, RAW_PATH) if path.exists()]
    if existing:
        raise SystemExit("S1 artifacts already exist; refusing to overwrite")


def _resolve(tokenizer: Any) -> dict[str, tuple[int, int]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    recorded = manifest["token_ids"]
    new_ids = resolve_new_identity_tokens(tokenizer)
    anchor_ids = resolve_anchor_tokens(tokenizer)
    expected_new = (recorded["new_identity"]["positive_id"], recorded["new_identity"]["negative_id"])
    if new_ids != expected_new:
        raise SystemExit(f"new-identity token ids are {new_ids[0]}, {new_ids[1]}")
    if anchor_ids != (9834, 902):
        raise SystemExit(f"anchor token ids are {anchor_ids[0]}, {anchor_ids[1]}")
    return {"anchor": anchor_ids, "new_identity": new_ids}


def main() -> None:
    _refuse_overwrite()
    prediction_digest, predictions = assert_prediction_lock()
    catalog_digest = assert_catalog_lock()
    holdout = _holdout_rows(predictions)
    outcomes = json.loads(OUTCOMES_PATH.read_text(encoding="utf-8"))
    gbar = gbar_from_update(outcomes)
    commit = assert_clean_git()
    direction_d1, direction_d2 = load_directions()
    directions = {"anchor": direction_d1, "new_identity": direction_d2}

    started = time.perf_counter()
    model, device = _load_model()
    if device != DEVICE_NAME:
        raise SystemExit("S1 must stay on CPU")
    resolved = _resolve(model.tokenizer)
    by_prompt = {row["prompt_id"]: row for row in catalog()}
    tokens_of = {}
    for row in holdout:
        prompt_id = row["prompt_id"]
        if prompt_id not in tokens_of:
            source = by_prompt[prompt_id]
            if source["partition"] != "holdout" or source["family"] != row["family"]:
                raise RuntimeError(f"catalog fields differ for {prompt_id}")
            tokens_of[prompt_id] = model.to_tokens(source["text"], prepend_bos=True)
    first = holdout[0]
    positive_id, negative_id = _arm_tokens(str(first["arm"]), resolved)
    time_forward(model, tokens_of[first["prompt_id"]], first["hook"], directions[first["arm"]], positive_id, negative_id)
    time_gradient(model, tokens_of[first["prompt_id"]], first["hook"], directions[first["arm"]], positive_id, negative_id)
    time_hvp(
        model,
        tokens_of[first["prompt_id"]],
        CELL_LAYER[first["intervention_id"]],
        first["hook"],
        directions[first["arm"]],
        positive_id,
        negative_id,
    )
    samples = _time_rows(model, holdout, tokens_of, directions, resolved)
    cost_forward = float(statistics.median([row["forward_seconds"] for row in samples]))
    cost_m2 = float(statistics.median([row["m2_seconds"] for row in samples]))
    if not cost_forward > 0.0:
        raise RuntimeError("forward cost is not positive")
    k_eq = int(math.ceil(cost_m2 / cost_forward))
    ks = executed_ks(k_eq)
    n_probes = max(8, k_eq)
    print(f"k_eq {k_eq} probes {n_probes} forward_s {cost_forward} m2_s {cost_m2}", flush=True)
    scored_rows = _evaluate_rows(model, holdout, tokens_of, directions, resolved, gbar, n_probes, ks)
    scored = summarize(scored_rows, k_eq)
    cost = cost_summary(samples, k_eq)
    runtime_seconds = time.perf_counter() - started
    payload = {
        "catalog_sha256": catalog_digest,
        "cost": cost,
        "device": DEVICE_NAME,
        "dtype": DTYPE_NAME,
        "gbar": {f"{arm}|{cell}": value for (arm, cell), value in sorted(gbar.items())},
        "git_commit": commit,
        "k_eq": k_eq,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_probes": n_probes,
        "prediction_sha256": prediction_digest,
        "rows": scored_rows,
        "runtime_seconds": runtime_seconds,
        "scored": scored,
        "timing": samples,
        "token_ids": {
            "anchor": {"negative_id": resolved["anchor"][1], "positive_id": resolved["anchor"][0]},
            "new_identity": {"negative_id": resolved["new_identity"][1], "positive_id": resolved["new_identity"][0]},
        },
    }
    report = render_report(payload)
    if _sha256(PREDICTIONS_PATH) != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError("prediction artifact changed before the S1 write")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RAW_PATH.with_suffix(".json.tmp")
    temporary.write_text(_dump(payload), encoding="utf-8")
    os.replace(temporary, RAW_PATH)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"decision": scored["decision"], "k_eq": k_eq, "status": "S1_COMPLETE"}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
