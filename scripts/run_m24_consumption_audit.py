"""Execute the frozen M24 consumption audit.

Predictions are written for validation and evaluation before either
partition's outcomes. The train partition is not forwarded. The candidate
is MechanismResponseModel.predict. No slope is fit.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m24_consumption_catalog import CATALOG_SHA256, catalog, catalog_sha256
from scripts.m24_consumption_protocol import (
    EXECUTABLE_PARTITIONS,
    FAMILY,
    baseline_prediction,
    consumption_verdict,
    join_rows,
    leakage_ok,
    prediction_file_clean,
    rows_lack_outcomes,
    run_prediction_before_outcome,
    score_split,
    attach_shuffled_predictions,
)
from src.cognitive_self_model.m23.m22_reuse import (
    frozen_prompts,
    logit_margin,
    make_resid_hook,
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)
from src.cognitive_self_model.mechanism_response import (
    DIRECTION_SHA256,
    PreInterventionState,
    directional_derivative,
    frozen_mechanism_response_models,
)
from scripts.feasibility_gate_catalog import catalog as f_catalog
from scripts.validate_m23_g_protocol import catalog as g_catalog
from scripts.validate_post_m23_single_measurement import catalog as s_catalog

D2_SHA = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
RAW = ROOT / "reports" / "m24_raw"
ORDER_LOG = RAW / "order_log.jsonl"
RESULTS_PATH = ROOT / "reports" / "M24_CONSUMPTION_RESULTS.json"
EPISODES_PATH = ROOT / "reports" / "M24_CONSUMPTION_EPISODES.csv"
EXECUTION_PATH = ROOT / "reports" / "M24_CONSUMPTION_EXECUTION.md"
PROTOCOL_PATH = ROOT / "reports" / "M24_CONSUMPTION_PROTOCOL.md"


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with ORDER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"path": str(path.relative_to(ROOT)), "utc": _stamp(), "sha256": digest}) + "\n")
    return digest


def _prompts(partition: str) -> list[dict[str, str]]:
    if partition not in EXECUTABLE_PARTITIONS:
        raise RuntimeError(f"{partition} is not forwarded")
    return sorted((row for row in catalog() if row["partition"] == partition), key=lambda row: row["prompt_id"])


def _assert_catalog_frozen() -> None:
    if catalog_sha256() != CATALOG_SHA256:
        raise RuntimeError("catalog hash does not match the frozen constant")
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    if CATALOG_SHA256 not in text:
        raise RuntimeError("protocol is missing the catalog hash")
    rows = catalog()
    if len(rows) != 36:
        raise RuntimeError("catalog is not 36 prompts")
    old_texts = {record.text for record in frozen_prompts()}
    old_ids = {record.prompt_id for record in frozen_prompts()}
    for previous in (g_catalog(), s_catalog(), f_catalog()):
        old_texts.update(row["text"] for row in previous)
        old_ids.update(row["prompt_id"] for row in previous)
    for row in rows:
        if row["text"] not in text or row["prompt_id"] not in text:
            raise RuntimeError(f"protocol is missing {row['prompt_id']}")
        if row["text"] in old_texts or row["prompt_id"] in old_ids:
            raise RuntimeError(f"catalog reuses an old prompt: {row['prompt_id']}")
        if not row["prompt_id"].startswith("c-"):
            raise RuntimeError("catalog id is outside the c-* namespace")


def _assert_no_effect(path: Path) -> None:
    if not prediction_file_clean(path.read_text(encoding="utf-8")):
        raise RuntimeError(f"{path.name} contains an outcome field")


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


def _prediction_rows(model, runner, tokens, mechanisms, d2, prompts: list[dict[str, str]]) -> list[dict[str, Any]]:
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
            gradient = gradients[mechanism.layer]
            state = PreInterventionState(margin_gradient=gradient)
            predicted = float(mechanism.predict(state))
            orth = float(directional_derivative(np.asarray(gradient, dtype=np.float64), d2))
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
                    "g_orth": orth,
                    "baseline_prediction": baseline_prediction(mechanism),
                    "candidate_prediction": predicted,
                    "response_rule": mechanism.response_rule,
                    "outcome_present": False,
                }
            )
        print(f"predict {prompt['partition']} {prompt['prompt_id']}", flush=True)
    if not rows_lack_outcomes(rows):
        raise RuntimeError("prediction rows contain an outcome")
    return attach_shuffled_predictions(rows)


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
                    **measured,
                }
            )
        print(f"outcome {prompt['partition']} {prompt['prompt_id']}", flush=True)
    return rows


def _markdown(payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation"]
    validation = payload["validation"]
    lines = [
        "# M24 consumption execution",
        "",
        "The protocol in `reports/M24_CONSUMPTION_PROTOCOL.md` is unchanged.",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        payload["interpretation"],
        "",
        "## Evaluation",
        "",
        f"- baseline MAE: `{evaluation['baseline_mae']}`",
        f"- mechanism MAE: `{evaluation['mechanism_mae']}`",
        f"- paired mean: `{evaluation['paired_mean']}`",
        f"- interval: `[{evaluation['low']}, {evaluation['high']}]`",
        f"- class: `{evaluation['class']}`",
        f"- shuffled class: `{evaluation['shuffled_class']}`",
        f"- shuffled paired mean: `{evaluation['shuffled_paired_mean']}`",
        f"- sign decision does not lose: `{evaluation['sign_does_not_lose']}`",
        f"- leakage_ok: `{payload['leakage_ok']}`",
        "",
        "## Validation",
        "",
        "Validation is not an input to the verdict.",
        "",
        f"- class: `{validation['class']}`",
        f"- paired mean: `{validation['paired_mean']}`",
        f"- interval: `[{validation['low']}, {validation['high']}]`",
        "",
        "## Per-cell evaluation",
        "",
        "| cell | paired mean | class | baseline MAE | mechanism MAE |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for cell in evaluation["cells"]:
        lines.append(
            f"| `{cell['intervention_id']}` | {cell['paired_mean']} | `{cell['class']}` | {cell['baseline_mae']} | {cell['mechanism_mae']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _interpretation(verdict: str) -> str:
    if verdict == "CONSUMPTION_SUPPORTED":
        return (
            "On the frozen evaluation prompts, the consumed derivative reduced absolute error relative to the stored cell mean, "
            "the sign decision did not lose to that mean, and the shuffled predictions did not clear the same interval. "
            "This is a consumption result for the frozen object on this catalog. It is not a self-model."
        )
    if verdict == "CONSUMPTION_NOT_SUPPORTED":
        return (
            "On the frozen evaluation prompts, the consumed derivative increased absolute error relative to the stored cell mean. "
            "The object was consumed as specified. This catalog does not support that consumption. It is not a claim about self-models in general."
        )
    return (
        "The recorded evaluation classes do not meet the supported rule or the not-supported rule. "
        "The derivative is not declared useless. No new observable is authorized."
    )


def main() -> None:
    if RAW.exists() and any(RAW.iterdir()):
        raise SystemExit("raw directory already has files; refusing to rerun")
    if RESULTS_PATH.exists():
        raise SystemExit("results already exist; refusing to rerun")
    _assert_catalog_frozen()
    RAW.mkdir(parents=True, exist_ok=True)
    mechanisms = frozen_mechanism_response_models()
    if [model.intervention_id for model in mechanisms] != list(FAMILY):
        raise RuntimeError("mechanism objects are not the frozen family")
    before = [asdict(model) for model in mechanisms]
    means = {model.intervention_id: baseline_prediction(model) for model in mechanisms}
    _write(
        RAW / "catalog_freeze.json",
        {
            "catalog_sha256": CATALOG_SHA256,
            "prompts": catalog(),
            "executed_partitions": list(EXECUTABLE_PARTITIONS),
            "baseline_means": means,
            "source": "reports/M24_CONSUMPTION_PROTOCOL.md",
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
    d2 = orthogonal_direction(896, 22103, d1)
    if vector_sha256(d1) != DIRECTION_SHA256 or vector_sha256(d2) != D2_SHA:
        raise RuntimeError("direction hash does not match the frozen vectors")
    heldout_prompts = {name: _prompts(name) for name in EXECUTABLE_PARTITIONS}
    prediction_paths = {}
    prediction_digests = {}

    def prepare(name: str) -> list[dict[str, Any]]:
        rows = _prediction_rows(model, runner, tokens, mechanisms, d2, heldout_prompts[name])
        for row in rows:
            if float(row["baseline_prediction"]) != means[row["intervention_id"]]:
                raise RuntimeError("baseline drifted from the stored cell mean")
            if float(row["candidate_prediction"]) != float(row["g"]):
                raise RuntimeError("candidate prediction is not g")
        path = RAW / f"{name}_predictions.json"
        digest = _write(path, rows)
        _assert_no_effect(path)
        prediction_paths[name] = path
        prediction_digests[name] = digest
        return rows

    def finish(name: str) -> list[dict[str, Any]]:
        path = prediction_paths[name]
        _assert_no_effect(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != prediction_digests[name]:
            raise RuntimeError("prediction file changed before outcomes")
        return _outcome_rows(model, runner, tokens, mechanisms, heldout_prompts[name])

    prepared, measured = run_prediction_before_outcome(EXECUTABLE_PARTITIONS, prepare, finish)
    outcome_paths = {}
    for name in EXECUTABLE_PARTITIONS:
        outcome_paths[name] = RAW / f"{name}_outcomes.json"
        _write(outcome_paths[name], measured[name])
        _assert_no_effect(prediction_paths[name])
    after = [asdict(model) for model in mechanisms]
    model_unchanged = before == after
    files_ordered = True
    rows_clean = True
    baseline_ok = True
    for name in EXECUTABLE_PARTITIONS:
        if prediction_paths[name].stat().st_mtime > outcome_paths[name].stat().st_mtime:
            files_ordered = False
        if hashlib.sha256(prediction_paths[name].read_bytes()).hexdigest() != prediction_digests[name]:
            rows_clean = False
        if not prediction_file_clean(prediction_paths[name].read_text(encoding="utf-8")):
            rows_clean = False
        if means != {model.intervention_id: baseline_prediction(model) for model in mechanisms}:
            baseline_ok = False
    clean = leakage_ok(
        prediction_before_outcome=files_ordered,
        rows_clean=rows_clean,
        model_unchanged=model_unchanged,
        baseline_from_model=baseline_ok,
    )
    joined = {name: join_rows(prepared[name], measured[name]) for name in EXECUTABLE_PARTITIONS}
    scored = {name: score_split(joined[name], expected_prompts=12) for name in EXECUTABLE_PARTITIONS}
    verdict = consumption_verdict(
        scored["evaluation"]["class"],
        bool(scored["evaluation"]["sign_does_not_lose"]),
        scored["evaluation"]["shuffled_class"],
        clean,
    )
    payload = {
        "verdict": verdict,
        "leakage_ok": clean,
        "device": device,
        "alpha": 1.0,
        "family": list(FAMILY),
        "catalog_sha256": CATALOG_SHA256,
        "response_rule": "prediction = g",
        "baseline": "training_baseline_mean",
        "baseline_means": means,
        "shuffle_seed": 24001,
        "bootstrap_draws": 5000,
        "bootstrap_seed": 23001,
        "decision_threshold": 0.0,
        "direction_sha256": DIRECTION_SHA256,
        "control_direction_sha256": D2_SHA,
        "evaluation": scored["evaluation"],
        "validation": scored["validation"],
        "interpretation": _interpretation(verdict),
    }
    _write(RESULTS_PATH, payload)
    fields = [
        "intervention_id",
        "layer",
        "hook",
        "direction_id",
        "alpha",
        "prompt_id",
        "family",
        "partition",
        "g",
        "g_orth",
        "baseline_prediction",
        "candidate_prediction",
        "shuffled_prediction",
        "response_rule",
        "baseline_output",
        "intervened_output",
        "observed_effect",
        "abs_error_baseline",
        "abs_error_mechanism",
        "paired_difference",
        "shuffled_paired_difference",
        "orth_paired_difference",
    ]
    with EPISODES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for name in EXECUTABLE_PARTITIONS:
            for row in joined[name]:
                writer.writerow(row)
    EXECUTION_PATH.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "leakage_ok": clean, "evaluation_class": scored["evaluation"]["class"]}, indent=2))


if __name__ == "__main__":
    main()
