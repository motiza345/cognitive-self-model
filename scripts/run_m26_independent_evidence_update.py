"""Execute the frozen M26 independent-evidence update once.

The update equation, seeds, sigma values, catalog, and verdict map are
imported from the frozen modules. This file only runs that sequence:
update predictions, update outcomes, corrections, holdout predictions,
holdout outcomes, then scoring.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feasibility_gate_catalog import catalog as f_catalog
from scripts.m24_consumption_catalog import catalog as c_catalog
from scripts.m26_update_catalog import CATALOG_SHA256, catalog, catalog_sha256
from scripts.validate_m23_g_protocol import catalog as g_catalog
from scripts.validate_post_m23_single_measurement import catalog as s_catalog
from src.cognitive_self_model.m23.belief import CRITICAL_Z
from src.cognitive_self_model.m23.m22_reuse import (
    frozen_prompts,
    logit_margin,
    make_resid_hook,
    primary_direction,
    vector_sha256,
)
from src.cognitive_self_model.m23.score import _agreement
from src.cognitive_self_model.m23.stats import clopper_pearson, paired_mean_ci
from src.cognitive_self_model.mechanism_response import (
    DIRECTION_SHA256,
    PreInterventionState,
    frozen_mechanism_response_models,
)
from src.cognitive_self_model.residual_update import (
    FAMILY,
    FROZEN_SIGMA,
    PRIOR_COUNT,
    PRIOR_MEAN,
    PROTOCOL_VERSION,
    SHUFFLE_SEED,
    cell_deltas,
    global_delta,
    posterior_sd,
    prediction_after,
    prediction_before,
    predictive_sd,
    shuffled_residuals,
    update_residuals,
    update_verdict,
)

FREEZE_COMMIT = "a81388248d13b5cd602ee6bd786fe871d7909fbc"
RAW = ROOT / "reports" / "m26_raw"
ORDER_LOG = RAW / "order_log.jsonl"
RESULTS_PATH = ROOT / "reports" / "M26_INDEPENDENT_EVIDENCE_UPDATE_RESULTS.json"
EPISODES_PATH = ROOT / "reports" / "M26_INDEPENDENT_EVIDENCE_UPDATE_EPISODES.csv"
EXECUTION_PATH = ROOT / "reports" / "M26_INDEPENDENT_EVIDENCE_UPDATE_EXECUTION.md"
PROTOCOL_PATH = ROOT / "reports" / "M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md"
M25_RESULTS = ROOT / "reports" / "M25_SCOPED_MECHANISM_BELIEF_RESULTS.json"
BANNED_PREDICTION_KEYS = {"observed_effect", "baseline_output", "intervened_output"}
CLOSURE_CHAIN = (
    "reports/m26_raw/update_predictions.json",
    "reports/m26_raw/update_outcomes.json",
    "reports/m26_raw/corrections.json",
    "reports/m26_raw/holdout_predictions.json",
    "reports/m26_raw/holdout_outcomes.json",
)


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, payload: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "path": str(path.relative_to(ROOT)),
        "utc": _stamp(),
        "sha256": digest,
        "mtime_ns": str(path.stat().st_mtime_ns),
    }
    with ORDER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def _contains_banned(payload: Any) -> set[str]:
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in BANNED_PREDICTION_KEYS or key == "pre_dot":
                    found.add(str(key))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    return found


def _assert_prediction_file(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    found = _contains_banned(payload)
    if found:
        raise RuntimeError(f"{path.name} contains outcome fields: {sorted(found)}")


def _prompts(partition: str) -> list[dict[str, str]]:
    rows = [row for row in catalog() if row["partition"] == partition]
    if len(rows) != 12:
        raise RuntimeError(f"{partition} partition is not 12 prompts")
    return sorted(rows, key=lambda row: row["prompt_id"])


def _assert_catalog_frozen() -> None:
    if catalog_sha256() != CATALOG_SHA256:
        raise RuntimeError("catalog hash does not match the frozen constant")
    if CATALOG_SHA256 != "015c3d41855da5fa630c382fe0e4a9ea63c868e12e4a1243e104e1c4e371ea6f":
        raise RuntimeError("catalog hash constant is not the frozen digest")
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    if CATALOG_SHA256 not in text:
        raise RuntimeError("protocol is missing the catalog hash")
    rows = catalog()
    if len(rows) != 24:
        raise RuntimeError("catalog is not 24 prompts")
    old_texts = {record.text for record in frozen_prompts()}
    old_ids = {record.prompt_id for record in frozen_prompts()}
    for previous in (g_catalog(), s_catalog(), f_catalog(), c_catalog()):
        old_texts.update(row["text"] for row in previous)
        old_ids.update(row["prompt_id"] for row in previous)
    for row in rows:
        if row["text"] not in text or row["prompt_id"] not in text:
            raise RuntimeError(f"protocol is missing {row['prompt_id']}")
        if row["text"] in old_texts or row["prompt_id"] in old_ids:
            raise RuntimeError(f"catalog reuses an old prompt: {row['prompt_id']}")
        if not row["prompt_id"].startswith("u-"):
            raise RuntimeError("catalog id is outside the u-* namespace")
    for cell, sigma in FROZEN_SIGMA.items():
        if cell not in text or str(sigma) not in text:
            raise RuntimeError(f"protocol is missing frozen sigma for {cell}")


def _assert_sigma_matches_m25() -> None:
    payload = json.loads(M25_RESULTS.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 3:
        raise RuntimeError("M25 results do not contain the three frozen cells")
    seen = set()
    for cell in cells:
        mechanism_id = str(cell["mechanism_id"])
        seen.add(mechanism_id)
        if mechanism_id not in FROZEN_SIGMA:
            raise RuntimeError(f"M25 cell is outside the frozen family: {mechanism_id}")
        for condition in ("consistent", "contradictory", "unsupported_catalog"):
            snapshot = cell[condition]
            if float(snapshot["residual_sd"]) != FROZEN_SIGMA[mechanism_id]:
                raise RuntimeError(f"M25 residual_sd drifted for {mechanism_id} {condition}")
            if snapshot.get("response_rule") != "prediction = g":
                raise RuntimeError("M25 response rule is not prediction = g")
    if seen != set(FAMILY):
        raise RuntimeError("M25 cells do not match the frozen family")


def _assert_git_before_execution() -> dict[str, str]:
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    if status:
        raise SystemExit("working tree is not clean; refusing to run")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise SystemExit("HEAD is not the frozen commit or a descendant")
    changed = {line for line in _git("diff", "--name-only", FREEZE_COMMIT, "HEAD").splitlines() if line}
    allowed = {"scripts/run_m26_independent_evidence_update.py"}
    if head != FREEZE_COMMIT and not changed <= allowed:
        raise SystemExit(f"descendant changes are outside the execution runner: {sorted(changed)}")
    if head == FREEZE_COMMIT and changed:
        raise SystemExit("freeze commit reports a diff against itself")
    return {"head": head, "status": status, "changed_since_freeze": " ".join(sorted(changed))}


def _gradients(model, runner, text: str, positive_id: int, negative_id: int) -> dict[int, tuple[float, ...]]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    saved: dict[int, torch.Tensor] = {}

    def _hook(layer: int):
        def hook_fn(residual: torch.Tensor, hook: Any) -> torch.Tensor:
            if residual.requires_grad:
                residual.retain_grad()
            saved[layer] = residual
            return residual

        return hook_fn

    layers = (0, 8, 15)
    hooks = [(f"blocks.{layer}.hook_resid_post", _hook(layer)) for layer in layers]
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        logits = runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=hooks))
        margin = logits[0, -1, positive_id] - logits[0, -1, negative_id]
        if not margin.requires_grad:
            raise RuntimeError("margin is not connected to the residual graph")
        margin.backward()
    found: dict[int, tuple[float, ...]] = {}
    for layer in layers:
        residual = saved.get(layer)
        if residual is None or residual.grad is None:
            raise RuntimeError(f"no gradient at layer {layer}")
        gradient = residual.grad[0, -1].detach().float().cpu()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"non-finite gradient at layer {layer}")
        found[layer] = tuple(float(value) for value in gradient.tolist())
    model.zero_grad(set_to_none=True)
    return found


def _g_rows(model, runner, tokens, mechanisms, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        gradients = _gradients(
            model,
            runner,
            prompt["text"],
            tokens["positive_token_id"],
            tokens["negative_token_id"],
        )
        for mechanism in mechanisms:
            state = PreInterventionState(margin_gradient=gradients[mechanism.layer])
            predicted = float(mechanism.predict(state))
            if mechanism.response_rule != "prediction = g" or float(mechanism.alpha) != 1.0:
                raise RuntimeError("response model is not the frozen g rule")
            rows.append(
                {
                    "intervention_id": mechanism.intervention_id,
                    "layer": mechanism.layer,
                    "hook": mechanism.hook,
                    "direction_id": mechanism.direction_id,
                    "alpha": mechanism.alpha,
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt["family"],
                    "partition": prompt["partition"],
                    "g": predicted,
                    "prediction_before": prediction_before(predicted),
                    "baseline_prediction": float(mechanism.training_baseline_mean),
                    "response_rule": mechanism.response_rule,
                    "outcome_present": False,
                }
            )
        print(f"predict {prompt['partition']} {prompt['prompt_id']}", flush=True)
    if _contains_banned(rows):
        raise RuntimeError("prediction rows contain an outcome")
    return rows


def _effect(model, runner, text: str, positive_id: int, negative_id: int, mechanism) -> dict[str, float]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        baseline = logit_margin(runner._as_logits(model(tokens)), positive_id, negative_id)
        recorder: dict[str, Any] = {}
        vector = torch.tensor(mechanism.direction, dtype=torch.float32)
        hook_fn = make_resid_hook(1.0, vector, recorder)
        intervened = logit_margin(
            runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(mechanism.hook, hook_fn)])),
            positive_id,
            negative_id,
        )
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("intervention hook did not fire once")
    if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not stay on the last token")
    return {
        "baseline_output": float(baseline),
        "intervened_output": float(intervened),
        "observed_effect": float(intervened - baseline),
    }


def _outcome_rows(model, runner, tokens, mechanisms, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        for mechanism in mechanisms:
            measured = _effect(
                model,
                runner,
                prompt["text"],
                tokens["positive_token_id"],
                tokens["negative_token_id"],
                mechanism,
            )
            rows.append(
                {
                    "intervention_id": mechanism.intervention_id,
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt["family"],
                    "partition": prompt["partition"],
                    "alpha": 1.0,
                    "baseline_output": measured["baseline_output"],
                    "intervened_output": measured["intervened_output"],
                    "observed_effect": measured["observed_effect"],
                }
            )
        print(f"outcome {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _join(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcome_by = {(str(row["intervention_id"]), str(row["prompt_id"])): row for row in outcomes}
    if len(outcome_by) != len(outcomes):
        raise RuntimeError("outcome keys are not unique")
    joined = []
    for prediction in predictions:
        key = (str(prediction["intervention_id"]), str(prediction["prompt_id"]))
        outcome = outcome_by.get(key)
        if outcome is None:
            raise RuntimeError(f"missing outcome for {key}")
        if prediction["partition"] != outcome["partition"]:
            raise RuntimeError("prediction and outcome partitions differ")
        merged = dict(prediction)
        merged["baseline_output"] = float(outcome["baseline_output"])
        merged["intervened_output"] = float(outcome["intervened_output"])
        merged["observed_effect"] = float(outcome["observed_effect"])
        joined.append(merged)
    if len(joined) != len(predictions):
        raise RuntimeError("join dropped a prediction")
    return joined


def _prompt_means(rows: list[dict[str, Any]], field: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(float(row[field]))
    scores = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3:
            raise RuntimeError(f"{prompt_id} does not have three cells")
        scores.append(sum(values) / 3.0)
    if len(scores) != 12:
        raise RuntimeError("prompt aggregation does not match the 12 holdout prompts")
    return scores


def _interval(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return dict(paired_mean_ci(_prompt_means(rows, field)))


def _cell_interval(rows: list[dict[str, Any]], intervention_id: str, field: str) -> dict[str, Any]:
    cell_rows = sorted(
        (row for row in rows if row["intervention_id"] == intervention_id),
        key=lambda row: str(row["prompt_id"]),
    )
    if len(cell_rows) != 12:
        raise RuntimeError(f"{intervention_id} does not have 12 holdout rows")
    return dict(paired_mean_ci([float(row[field]) for row in cell_rows]))


def _mae(values: list[float]) -> float:
    if not values:
        raise RuntimeError("MAE requires rows")
    return float(sum(values) / len(values))


def _sign_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))
    flags = [_agreement(float(row[field]), float(row["observed_effect"])) for row in ordered]
    decided = [flag for flag in flags if flag is not None]
    correct = sum(1 for flag in decided if flag)
    n = len(decided)
    if n == 0:
        low, high = None, None
    else:
        low, high = clopper_pearson(correct, n)
    return {
        "correct": correct,
        "n_nonzero": n,
        "n_excluded_zero": len(flags) - n,
        "accuracy": (correct / n) if n else None,
        "low": low,
        "high": high,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (str(row["intervention_id"]), str(row["prompt_id"])))
    flags = [bool(row["covered"]) for row in ordered]
    covered = sum(1 for flag in flags if flag)
    n = len(flags)
    low, high = clopper_pearson(covered, n)
    return {"covered": covered, "n": n, "rate": covered / n, "low": low, "high": high}


def _attach_holdout_scores(rows: list[dict[str, Any]], n_by_cell: dict[str, int]) -> None:
    for row in rows:
        effect = float(row["observed_effect"])
        g_value = float(row["g"])
        cell_pred = float(row["prediction_after"])
        global_pred = float(row["prediction_global"])
        shuffled_pred = float(row["prediction_shuffled"])
        baseline = float(row["baseline_prediction"])
        cell = str(row["intervention_id"])
        n_update = int(n_by_cell[cell])
        scale = predictive_sd(FROZEN_SIGMA[cell], n_update)
        row["residual"] = effect - g_value
        row["abs_error_g"] = abs(effect - g_value)
        row["abs_error_cell"] = abs(effect - cell_pred)
        row["abs_error_global"] = abs(effect - global_pred)
        row["abs_error_shuffled"] = abs(effect - shuffled_pred)
        row["abs_error_baseline"] = abs(effect - baseline)
        row["paired_vs_g"] = row["abs_error_g"] - row["abs_error_cell"]
        row["paired_vs_global"] = row["abs_error_global"] - row["abs_error_cell"]
        row["paired_shuffled"] = row["abs_error_g"] - row["abs_error_shuffled"]
        row["predictive_scale"] = scale
        row["covered"] = abs(effect - cell_pred) <= CRITICAL_Z * scale
        row["n_update"] = n_update


def _attach_update_record(rows: list[dict[str, Any]], corrections: dict[str, Any]) -> None:
    delta_cell = corrections["delta_cell"]
    delta_shuffled = corrections["delta_shuffled"]
    delta_global = float(corrections["delta_global"])
    tau = corrections["tau"]
    n_by_cell = corrections["n"]
    for row in rows:
        cell = str(row["intervention_id"])
        g_value = float(row["g"])
        effect = float(row["observed_effect"])
        cell_delta = float(delta_cell[cell])
        row["delta_cell"] = cell_delta
        row["delta_global"] = delta_global
        row["delta_shuffled"] = float(delta_shuffled[cell])
        row["tau"] = float(tau[cell])
        row["sigma"] = float(FROZEN_SIGMA[cell])
        row["prediction_before"] = prediction_before(g_value)
        row["prediction_after"] = prediction_after(g_value, cell_delta)
        row["prediction_global"] = prediction_after(g_value, delta_global)
        row["prediction_shuffled"] = prediction_after(g_value, float(delta_shuffled[cell]))
        row["n_update"] = int(n_by_cell[cell])
        row["residual"] = effect - g_value


def _leakage_ok(
    *,
    closure: list[dict[str, str]],
    update_predictions_unchanged: bool,
    holdout_predictions_unchanged: bool,
    prediction_files_clean: bool,
    model_unchanged: bool,
    sigma_unchanged: bool,
    holdout_excluded: bool,
) -> bool:
    by_path = {record["path"]: record for record in closure}
    if any(path not in by_path for path in CLOSURE_CHAIN):
        return False
    times = [int(by_path[path]["mtime_ns"]) for path in CLOSURE_CHAIN]
    if times != sorted(times) or len(set(times)) != len(times):
        return False
    return bool(
        update_predictions_unchanged
        and holdout_predictions_unchanged
        and prediction_files_clean
        and model_unchanged
        and sigma_unchanged
        and holdout_excluded
    )


def _interpretation(verdict: str) -> str:
    if verdict == "UPDATE_SUPPORTED":
        return (
            "On this holdout, the per-cell residual correction reduced absolute error relative to raw g, "
            "and that reduction was not accounted for by one pooled correction or by the frozen shuffle of the same update residuals. "
            "g was not refit. This is not a self-model, and it does not clear the M25 contradiction record."
        )
    if verdict == "UPDATE_NOT_SUPPORTED":
        return (
            "On this holdout, the per-cell residual correction increased absolute error relative to raw g. "
            "g was not refit. This catalog does not support the correction. It does not show that a residual update is impossible under every other catalog."
        )
    return (
        "The recorded holdout classes do not meet the supported rule or the not-supported rule. "
        "The correction is unestablished. No second prior, second shrinkage count, or new observable is authorized."
    )


def _markdown(payload: dict[str, Any]) -> str:
    primary = payload["primary"]
    versus = payload["versus_global"]
    shuffled = payload["shuffled"]
    lines = [
        "# M26 independent evidence update execution",
        "",
        "The protocol in `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md` is unchanged. The update equation, seeds, sigma values, and verdict map are unchanged.",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        payload["interpretation"],
        "",
        "## Primary intervals",
        "",
        "Prompt-level means of the three cells, then `paired_mean_ci` with 5000 draws and seed 23001.",
        "",
        f"- A, MAE(g) - MAE(g + δ_cell): mean `{json.dumps(primary['mean'])}`, interval `[{json.dumps(primary['low'])}, {json.dumps(primary['high'])}]`, class `{primary['class']}`",
        f"- B, MAE(g + δ_global) - MAE(g + δ_cell): mean `{json.dumps(versus['mean'])}`, interval `[{json.dumps(versus['low'])}, {json.dumps(versus['high'])}]`, class `{versus['class']}`",
        f"- shuffled, MAE(g) - MAE(g + δ_shuffled): mean `{json.dumps(shuffled['mean'])}`, interval `[{json.dumps(shuffled['low'])}, {json.dumps(shuffled['high'])}]`, class `{shuffled['class']}`",
        f"- leakage_ok: `{json.dumps(payload['leakage_ok'])}`",
        "",
        "## Corrections",
        "",
        "| quantity | M22.1-D1-L0 | M22.1-D1-L8 | M22.1-D1-L15 |",
        "| --- | ---: | ---: | ---: |",
        "| δ_cell | "
        + " | ".join(json.dumps(payload["delta_cell"][cell]) for cell in FAMILY)
        + " |",
        "| δ_shuffled | "
        + " | ".join(json.dumps(payload["delta_shuffled"][cell]) for cell in FAMILY)
        + " |",
        "| τ | " + " | ".join(json.dumps(payload["tau"][cell]) for cell in FAMILY) + " |",
        "| σ | " + " | ".join(json.dumps(payload["sigma"][cell]) for cell in FAMILY) + " |",
        "| n | " + " | ".join(str(payload["n"][cell]) for cell in FAMILY) + " |",
        "",
        f"- δ_global: `{json.dumps(payload['delta_global'])}`",
        "",
        "## Holdout descriptive scores",
        "",
        "Sign agreement and coverage are not verdict inputs.",
        "",
        f"- MAE g: `{json.dumps(payload['holdout_mae']['g'])}`",
        f"- MAE g + δ_cell: `{json.dumps(payload['holdout_mae']['cell'])}`",
        f"- MAE g + δ_global: `{json.dumps(payload['holdout_mae']['global'])}`",
        f"- MAE g + δ_shuffled: `{json.dumps(payload['holdout_mae']['shuffled'])}`",
        f"- MAE stored scalar: `{json.dumps(payload['holdout_mae']['baseline'])}`",
        f"- sign g: `{json.dumps(payload['sign']['g'])}`",
        f"- sign g + δ_cell: `{json.dumps(payload['sign']['cell'])}`",
        f"- coverage of g + δ_cell: `{json.dumps(payload['coverage'])}`",
        "",
        "## Per-cell holdout intervals",
        "",
        "| cell | A mean | A class | B mean | B class | shuffled mean | shuffled class |",
        "| --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for cell in payload["per_cell"]:
        lines.append(
            "| `{intervention_id}` | {a_mean} | `{a_class}` | {b_mean} | `{b_class}` | {s_mean} | `{s_class}` |".format(
                intervention_id=cell["intervention_id"],
                a_mean=json.dumps(cell["primary"]["mean"]),
                a_class=cell["primary"]["class"],
                b_mean=json.dumps(cell["versus_global"]["mean"]),
                b_class=cell["versus_global"]["class"],
                s_mean=json.dumps(cell["shuffled"]["mean"]),
                s_class=cell["shuffled"]["class"],
            )
        )
    lines.extend(
        [
            "",
            "## Closure",
            "",
            "| artifact | utc | sha256 |",
            "| --- | --- | --- |",
        ]
    )
    for record in payload["closure"]:
        lines.append(f"| `{record['path']}` | `{record['utc']}` | `{record['sha256']}` |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- model: `{payload['model_id']}`",
            f"- revision: `{payload['model_revision']}`",
            f"- direction sha256: `{payload['direction_sha256']}`",
            f"- catalog sha256: `{payload['catalog_sha256']}`",
            f"- protocol version: `{payload['protocol_version']}`",
            f"- git commit at execution: `{payload['git_commit']}`",
            f"- freeze commit: `{payload['freeze_commit']}`",
            f"- git status at execution: `{payload['git_status'] or 'clean'}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if RAW.exists() and any(RAW.iterdir()):
        raise SystemExit("raw directory already has files; refusing to rerun")
    if RESULTS_PATH.exists() or EPISODES_PATH.exists() or EXECUTION_PATH.exists():
        raise SystemExit("results already exist; refusing to rerun")
    git_info = _assert_git_before_execution()
    _assert_catalog_frozen()
    _assert_sigma_matches_m25()
    RAW.mkdir(parents=True, exist_ok=True)
    mechanisms = frozen_mechanism_response_models()
    if [model.intervention_id for model in mechanisms] != list(FAMILY):
        raise RuntimeError("mechanism objects are not the frozen family")
    before = [asdict(model) for model in mechanisms]
    sigma_before = dict(FROZEN_SIGMA)
    means = {model.intervention_id: float(model.training_baseline_mean) for model in mechanisms}
    _write(
        RAW / "catalog_freeze.json",
        {
            "catalog_sha256": CATALOG_SHA256,
            "prompts": catalog(),
            "family": list(FAMILY),
            "sigma": sigma_before,
            "prior_mean": PRIOR_MEAN,
            "prior_count": PRIOR_COUNT,
            "shuffle_seed": SHUFFLE_SEED,
            "bootstrap_draws": 5000,
            "bootstrap_seed": 23001,
            "source": "reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md",
        },
    )
    runner = _runner()
    model, device = runner.load_model()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("architecture check failed")
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"token ids changed: {tokens['token_ids']}")
    d1 = primary_direction(896, 22101)
    if vector_sha256(d1) != DIRECTION_SHA256:
        raise RuntimeError("direction hash does not match the frozen vector")
    update_prompts = _prompts("update")
    holdout_prompts = _prompts("holdout")
    if {row["prompt_id"] for row in update_prompts} & {row["prompt_id"] for row in holdout_prompts}:
        raise RuntimeError("update and holdout prompts overlap")

    update_prediction_rows = _g_rows(model, runner, tokens, mechanisms, update_prompts)
    update_prediction_record = _write(RAW / "update_predictions.json", update_prediction_rows)
    _assert_prediction_file(RAW / "update_predictions.json")
    stored_update_predictions = json.loads((RAW / "update_predictions.json").read_text(encoding="utf-8"))
    if hashlib.sha256((RAW / "update_predictions.json").read_bytes()).hexdigest() != update_prediction_record["sha256"]:
        raise RuntimeError("update prediction file changed before outcomes")

    update_outcome_rows = _outcome_rows(model, runner, tokens, mechanisms, update_prompts)
    update_outcome_record = _write(RAW / "update_outcomes.json", update_outcome_rows)
    _assert_prediction_file(RAW / "update_predictions.json")

    update_joined = _join(stored_update_predictions, update_outcome_rows)
    if any(row["partition"] != "update" for row in update_joined):
        raise RuntimeError("update join contains a non-update row")
    residual_rows = [
        {
            "intervention_id": row["intervention_id"],
            "prompt_id": row["prompt_id"],
            "partition": row["partition"],
            "g": row["g"],
            "observed_effect": row["observed_effect"],
        }
        for row in update_joined
    ]
    residuals = update_residuals(residual_rows)
    deltas = cell_deltas(residuals)
    shared = global_delta(residuals)
    shuffled = cell_deltas(shuffled_residuals(residual_rows))
    counts = {cell: len(residuals[cell]) for cell in FAMILY}
    if any(count != 12 for count in counts.values()):
        raise RuntimeError("each cell does not have 12 update residuals")
    tau = {cell: posterior_sd(FROZEN_SIGMA[cell], counts[cell]) for cell in FAMILY}
    corrections = {
        "delta_cell": deltas,
        "delta_global": shared,
        "delta_shuffled": shuffled,
        "tau": tau,
        "sigma": dict(FROZEN_SIGMA),
        "n": counts,
        "prior_mean": PRIOR_MEAN,
        "prior_count": PRIOR_COUNT,
        "shuffle_seed": SHUFFLE_SEED,
        "residuals": residuals,
        "shuffled_residuals": shuffled_residuals(residual_rows),
    }
    corrections_record = _write(RAW / "corrections.json", corrections)
    stored_corrections = json.loads((RAW / "corrections.json").read_text(encoding="utf-8"))
    if cell_deltas(residuals) != {cell: float(stored_corrections["delta_cell"][cell]) for cell in FAMILY}:
        reloaded_deltas = {cell: float(stored_corrections["delta_cell"][cell]) for cell in FAMILY}
        if reloaded_deltas != {cell: float(deltas[cell]) for cell in FAMILY}:
            raise RuntimeError("persisted cell deltas do not match the frozen equation")

    holdout_rows = _g_rows(model, runner, tokens, mechanisms, holdout_prompts)
    for row in holdout_rows:
        cell = str(row["intervention_id"])
        g_value = float(row["g"])
        cell_delta = float(stored_corrections["delta_cell"][cell])
        row["delta_cell"] = cell_delta
        row["delta_global"] = float(stored_corrections["delta_global"])
        row["delta_shuffled"] = float(stored_corrections["delta_shuffled"][cell])
        row["tau"] = float(stored_corrections["tau"][cell])
        row["sigma"] = float(FROZEN_SIGMA[cell])
        row["n_update"] = int(stored_corrections["n"][cell])
        row["prediction_before"] = prediction_before(g_value)
        row["prediction_after"] = prediction_after(g_value, cell_delta)
        row["prediction_global"] = prediction_after(g_value, float(stored_corrections["delta_global"]))
        row["prediction_shuffled"] = prediction_after(g_value, float(stored_corrections["delta_shuffled"][cell]))
    holdout_prediction_record = _write(RAW / "holdout_predictions.json", holdout_rows)
    _assert_prediction_file(RAW / "holdout_predictions.json")
    stored_holdout_predictions = json.loads((RAW / "holdout_predictions.json").read_text(encoding="utf-8"))
    if hashlib.sha256((RAW / "holdout_predictions.json").read_bytes()).hexdigest() != holdout_prediction_record["sha256"]:
        raise RuntimeError("holdout prediction file changed before outcomes")

    holdout_outcome_rows = _outcome_rows(model, runner, tokens, mechanisms, holdout_prompts)
    holdout_outcome_record = _write(RAW / "holdout_outcomes.json", holdout_outcome_rows)
    _assert_prediction_file(RAW / "update_predictions.json")
    _assert_prediction_file(RAW / "holdout_predictions.json")

    holdout_joined = _join(stored_holdout_predictions, holdout_outcome_rows)
    if any(row["partition"] != "holdout" for row in holdout_joined):
        raise RuntimeError("holdout join contains a non-holdout row")
    try:
        update_residuals(holdout_joined)
    except ValueError:
        holdout_excluded = True
    else:
        holdout_excluded = False
    _attach_holdout_scores(holdout_joined, {cell: int(stored_corrections["n"][cell]) for cell in FAMILY})
    _attach_update_record(update_joined, stored_corrections)

    primary = _interval(holdout_joined, "paired_vs_g")
    versus_global = _interval(holdout_joined, "paired_vs_global")
    shuffled_interval = _interval(holdout_joined, "paired_shuffled")
    after = [asdict(model) for model in mechanisms]
    sigma_unchanged = dict(FROZEN_SIGMA) == sigma_before
    update_unchanged = (
        hashlib.sha256((RAW / "update_predictions.json").read_bytes()).hexdigest() == update_prediction_record["sha256"]
    )
    holdout_unchanged = (
        hashlib.sha256((RAW / "holdout_predictions.json").read_bytes()).hexdigest()
        == holdout_prediction_record["sha256"]
    )
    closure = [
        update_prediction_record,
        update_outcome_record,
        corrections_record,
        holdout_prediction_record,
        holdout_outcome_record,
    ]
    clean = _leakage_ok(
        closure=closure,
        update_predictions_unchanged=update_unchanged,
        holdout_predictions_unchanged=holdout_unchanged,
        prediction_files_clean=True,
        model_unchanged=before == after,
        sigma_unchanged=sigma_unchanged,
        holdout_excluded=holdout_excluded,
    )
    verdict = update_verdict(str(primary["class"]), str(versus_global["class"]), str(shuffled_interval["class"]), clean)
    per_cell = []
    for cell in FAMILY:
        per_cell.append(
            {
                "intervention_id": cell,
                "primary": _cell_interval(holdout_joined, cell, "paired_vs_g"),
                "versus_global": _cell_interval(holdout_joined, cell, "paired_vs_global"),
                "shuffled": _cell_interval(holdout_joined, cell, "paired_shuffled"),
                "mae_g": _mae([float(row["abs_error_g"]) for row in holdout_joined if row["intervention_id"] == cell]),
                "mae_cell": _mae(
                    [float(row["abs_error_cell"]) for row in holdout_joined if row["intervention_id"] == cell]
                ),
                "sign_cell": _sign_summary(
                    [row for row in holdout_joined if row["intervention_id"] == cell],
                    "prediction_after",
                ),
                "coverage": _coverage([row for row in holdout_joined if row["intervention_id"] == cell]),
            }
        )
    payload = {
        "verdict": verdict,
        "leakage_ok": clean,
        "device": device,
        "alpha": 1.0,
        "family": list(FAMILY),
        "protocol_version": PROTOCOL_VERSION,
        "catalog_sha256": CATALOG_SHA256,
        "response_rule": "prediction = g",
        "model_id": runner.MODEL_ID,
        "model_revision": runner.REVISION,
        "direction_sha256": DIRECTION_SHA256,
        "freeze_commit": FREEZE_COMMIT,
        "git_commit": git_info["head"],
        "git_status": git_info["status"],
        "prior_mean": PRIOR_MEAN,
        "prior_count": PRIOR_COUNT,
        "shuffle_seed": SHUFFLE_SEED,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 23001,
        "critical_z": CRITICAL_Z,
        "sigma": dict(FROZEN_SIGMA),
        "n": {cell: int(stored_corrections["n"][cell]) for cell in FAMILY},
        "delta_cell": {cell: float(stored_corrections["delta_cell"][cell]) for cell in FAMILY},
        "delta_global": float(stored_corrections["delta_global"]),
        "delta_shuffled": {cell: float(stored_corrections["delta_shuffled"][cell]) for cell in FAMILY},
        "tau": {cell: float(stored_corrections["tau"][cell]) for cell in FAMILY},
        "update_residuals": stored_corrections["residuals"],
        "shuffled_residuals": stored_corrections["shuffled_residuals"],
        "primary": primary,
        "versus_global": versus_global,
        "shuffled": shuffled_interval,
        "holdout_mae": {
            "g": _mae([float(row["abs_error_g"]) for row in holdout_joined]),
            "cell": _mae([float(row["abs_error_cell"]) for row in holdout_joined]),
            "global": _mae([float(row["abs_error_global"]) for row in holdout_joined]),
            "shuffled": _mae([float(row["abs_error_shuffled"]) for row in holdout_joined]),
            "baseline": _mae([float(row["abs_error_baseline"]) for row in holdout_joined]),
        },
        "baseline_means": means,
        "sign": {
            "g": _sign_summary(holdout_joined, "g"),
            "cell": _sign_summary(holdout_joined, "prediction_after"),
            "global": _sign_summary(holdout_joined, "prediction_global"),
            "shuffled": _sign_summary(holdout_joined, "prediction_shuffled"),
        },
        "coverage": _coverage(holdout_joined),
        "per_cell": per_cell,
        "closure": closure,
        "interpretation": _interpretation(verdict),
    }
    _write(RESULTS_PATH, payload)
    fields = [
        "intervention_id",
        "layer",
        "hook",
        "prompt_id",
        "family",
        "partition",
        "g",
        "prediction_before",
        "prediction_after",
        "prediction_global",
        "prediction_shuffled",
        "baseline_prediction",
        "delta_cell",
        "delta_global",
        "delta_shuffled",
        "tau",
        "sigma",
        "n_update",
        "baseline_output",
        "intervened_output",
        "observed_effect",
        "residual",
        "abs_error_g",
        "abs_error_cell",
        "abs_error_global",
        "abs_error_shuffled",
        "abs_error_baseline",
        "paired_vs_g",
        "paired_vs_global",
        "paired_shuffled",
        "predictive_scale",
        "covered",
        "response_rule",
    ]
    update_for_csv = []
    for row in update_joined:
        copied = dict(row)
        effect = float(copied["observed_effect"])
        copied["abs_error_g"] = abs(effect - float(copied["g"]))
        copied["abs_error_cell"] = abs(effect - float(copied["prediction_after"]))
        copied["abs_error_global"] = abs(effect - float(copied["prediction_global"]))
        copied["abs_error_shuffled"] = abs(effect - float(copied["prediction_shuffled"]))
        copied["abs_error_baseline"] = abs(effect - float(copied["baseline_prediction"]))
        copied["paired_vs_g"] = copied["abs_error_g"] - copied["abs_error_cell"]
        copied["paired_vs_global"] = copied["abs_error_global"] - copied["abs_error_cell"]
        copied["paired_shuffled"] = copied["abs_error_g"] - copied["abs_error_shuffled"]
        copied["predictive_scale"] = predictive_sd(float(copied["sigma"]), int(copied["n_update"]))
        copied["covered"] = abs(effect - float(copied["prediction_after"])) <= CRITICAL_Z * float(
            copied["predictive_scale"]
        )
        update_for_csv.append(copied)
    with EPISODES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in update_for_csv + holdout_joined:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    EXECUTION_PATH.write_text(_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "leakage_ok": clean,
                "primary": primary,
                "versus_global": versus_global,
                "shuffled": shuffled_interval,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
