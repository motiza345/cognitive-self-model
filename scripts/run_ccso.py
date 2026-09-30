"""CCSO diagnostic on the frozen Qwen revision. Not a Self-Model and not an MRSM rerun.

The readout rules live in src/mrsm/ccso.py. This script writes that
preregistration before any Qwen forward.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cognitive_self_model.m22_1.direction import primary_direction, vector_sha256  # noqa: E402
from cognitive_self_model.m22_1.outcome import resolve_outcome_tokens  # noqa: E402
from cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role  # noqa: E402
from src.mrsm.ccso import (  # noqa: E402
    CONDITIONS,
    DIRECTION_SEEDS,
    LAYERS,
    NONLINEAR_ALPHAS,
    PRIMARY_ALPHAS,
    PRIMARY_SLICES,
    QWEN_NEGATIVE_ID,
    QWEN_POSITIVE_ID,
    QWEN_REVISION,
    READOUTS,
    beats,
    decide,
    preregistration_document,
    residual_state_and_gradient,
    run_readouts,
)
from src.mrsm.run_q import _load_model, _margins  # noqa: E402

OUT = ROOT / "artifacts" / "ccso"
REPORT = ROOT / "reports" / "CCSO_OBSERVATION.md"
ALPHAS = np.asarray(list(PRIMARY_ALPHAS) + list(NONLINEAR_ALPHAS), dtype=np.float64)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stop(message: str) -> None:
    raise SystemExit(message)


def _prompts() -> list:
    grouped = prompts_by_role(frozen_prompts())
    ordered = []
    for split in ("discovery", "validation", "replication"):
        ordered.extend(sorted(grouped[split], key=lambda record: record.prompt_id))
    roles = {record.role for record in ordered}
    if roles != {"discovery", "validation", "replication"} or len(ordered) != 18:
        _stop("STOP: frozen prompt split drifted")
    discovery = {record.prompt_id for record in ordered if record.role == "discovery"}
    held = {record.prompt_id for record in ordered if record.role != "discovery"}
    if discovery & held:
        _stop("STOP: held-out prompts overlap discovery")
    return ordered


def _freeze(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _measure(model, records, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    texts = [record.text for record in records]
    tokens_outcome = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    if not tokens_outcome["single_piece"]:
        _stop("STOP: outcome is not a single piece")
    if int(tokens_outcome["positive_token_id"]) != QWEN_POSITIVE_ID or int(tokens_outcome["negative_token_id"]) != QWEN_NEGATIVE_ID:
        _stop("STOP: outcome token ids do not match the frozen pair")
    h = np.zeros((len(records), len(LAYERS), int(model.cfg.d_model)), dtype=np.float64)
    g = np.zeros_like(h)
    total = len(records) * len(LAYERS)
    done = 0
    for prompt_index, record in enumerate(records):
        tokens = model.to_tokens([record.text], prepend_bos=True)
        for layer_index, layer in enumerate(LAYERS):
            hook_name = f"blocks.{layer}.hook_resid_post"
            if hook_name not in model.hook_dict:
                _stop(f"STOP: {hook_name} is not a hook")
            started = time.perf_counter()
            state, gradient = residual_state_and_gradient(
                model.run_with_hooks, tokens, layer, QWEN_POSITIVE_ID, QWEN_NEGATIVE_ID
            )
            h[prompt_index, layer_index] = state
            g[prompt_index, layer_index] = gradient
            done += 1
            print(f"state {done}/{total} {record.prompt_id} L{layer} {time.perf_counter() - started:.1f}s", flush=True)
        del tokens
        gc.collect()
    base = _margins(model, texts, [], QWEN_POSITIVE_ID, QWEN_NEGATIVE_ID)
    delta = np.zeros((len(records), len(LAYERS), directions.shape[0], ALPHAS.size), dtype=np.float64)
    step = 0
    planned = len(LAYERS) * directions.shape[0] * ALPHAS.size
    for layer_index, layer in enumerate(LAYERS):
        for direction_index in range(directions.shape[0]):
            for alpha_index, alpha in enumerate(ALPHAS):
                item = {
                    "name": f"L{layer}-D{direction_index}",
                    "layer": int(layer),
                    "direction_name": "ccso",
                    "alpha": float(alpha),
                    "vector": directions[direction_index],
                }
                treated = _margins(model, texts, [item], QWEN_POSITIVE_ID, QWEN_NEGATIVE_ID)
                delta[:, layer_index, direction_index, alpha_index] = np.asarray(base) - np.asarray(treated)
                step += 1
                print(f"delta {step}/{planned} layer {layer} dir {direction_index} alpha {alpha:.2f}", flush=True)
    if not np.isfinite(delta).all() or not np.isfinite(h).all() or not np.isfinite(g).all():
        _stop("STOP: non-finite state or drop")
    return h, g, delta


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _report(decision: dict, metrics: dict) -> str:
    lines = [
        "# CCSO — Contextual Causal State Observation",
        "",
        "Diagnostic / Pre-Self-Model. Not an MRSM gate and not a mechanism claim.",
        "",
        "## 1. Question",
        "",
        "Does a richer observation supply counterfactual information about this frozen Qwen's",
        "intervention drop that is not available from the snapshot or the local gradient?",
        "",
        f"- Decision: `{decision['decision']}`.",
        f"- {decision['primary_interpretation']}",
        "",
        "## 2. What was fixed before measurement",
        "",
        f"- Model `Qwen/Qwen2.5-0.5B` revision `{QWEN_REVISION}`. No GPT-2 and no fine-tuning.",
        "- Prompts are the frozen discovery, validation, and replication split. Held-out prompts are not fit.",
        "- Layers `0, 8, 15, 23`. Directions seeds `23101`–`23108`; `23107` and `23108` are unseen.",
        "- Primary alphas are ±0.01, ±0.05, and ±0.10. ±0.25 is a nonlinear probe and is not a decision input.",
        "- ±1 is outside the primary endpoint.",
        "- O1 is `(h, g)`. The predicted drop is not a feature. O2 is a depth profile, not a trajectory.",
        "- B2, B3, C1, and C2 share one readout width. B0 is the global mean and B1 is the regime mean.",
        "",
        "## 3. Linear readouts",
        "",
        "A win requires a strict MAE decrease and a strict sign-agreement increase. Pearson is reported only.",
        "",
        "| Readout | Slice | MAE | Sign agreement | Pearson |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in READOUTS:
        for sl in PRIMARY_SLICES:
            block = metrics["linear"][name][sl]
            lines.append(
                "| " + " | ".join([name, sl, _fmt(block["mae"]), _fmt(block["sign_agreement"]), _fmt(block["pearson"])]) + " |"
            )
    lines.extend(["", "## 4. Small MLP", "", "| Readout | Slice | MAE | Sign agreement |", "| --- | --- | ---: | ---: |"])
    for name in CONDITIONS:
        for sl in PRIMARY_SLICES:
            block = metrics["small_mlp"][name][sl]
            lines.append("| " + " | ".join([name, sl, _fmt(block["mae"]), _fmt(block["sign_agreement"])]) + " |")
    large = metrics["mlp_large"]["B2"]
    lines.extend(
        [
            "",
            "## 5. Capacity, nulls, and the nonlinear probe",
            "",
            f"- Large MLP snapshot validation MAE `{_fmt(large['S_val']['mae'])}`, replication MAE `{_fmt(large['S_rep']['mae'])}`.",
            "- Null MAE on validation, linear family. A positive claim is blocked when a null MAE is at most the real MAE.",
            "",
            "| Observation | N1 shuffled target | N2 shuffled direction | N4 shifted state | Real S_val MAE |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in CONDITIONS:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(metrics["nulls"][name]["N1"]["S_val"]["mae"]),
                    _fmt(metrics["nulls"][name]["N2"]["S_val"]["mae"]),
                    _fmt(metrics["nulls"][name]["N4"]["S_val"]["mae"]),
                    _fmt(metrics["linear"][name]["S_val"]["mae"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"- Regime-label permutation B1 validation MAE `{_fmt(metrics['regime_permutation_b1']['S_val']['mae'])}`. Not a decision input.",
            "",
            "| Observation | Nonlinear-probe validation MAE | Sign agreement |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in CONDITIONS:
        block = metrics["nonlinear_probe"]["S_val"][name]
        lines.append("| " + " | ".join([name, _fmt(block["mae"]), _fmt(block["sign_agreement"])]) + " |")
    lines.extend(
        [
            "",
            "## 6. Same state, different direction",
            "",
            "Mean within-prompt Pearson at alpha = +0.01 on validation, across the six training directions.",
            "",
            "| Observation | Mean Pearson | Defined groups | Groups |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name in ("C1", "C2"):
        block = metrics["direction_sensitivity"][name]
        lines.append(
            "| "
            + " | ".join([name, _fmt(block["mean_pearson"]), str(block["n_defined"]), str(block["n_groups"])])
            + " |"
        )
    lines.extend(
        [
            "",
            "## 7. What this does not say",
            "",
            "No Self-Model was built. Qwen was not modified. MRSM was not rerun.",
            "No mechanism was identified. A trajectory encoder was not built.",
            "The result is local to this frozen revision and these frozen prompts.",
            "It does not say a richer observation cannot exist.",
            "",
            "## 8. Which strict comparisons fired",
            "",
            "A cell is true only when MAE falls and sign agreement rises. This trace does not change the decision.",
            "",
            "| Comparison | Validation | Replication | Unseen direction | Held-out regime |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for left, right in (
        ("B3", "B2"),
        ("C1", "B2"),
        ("C1", "B3"),
        ("C2", "B2"),
        ("C2", "B3"),
        ("C2", "C1"),
        ("B2", "B0"),
        ("B2", "B1"),
    ):
        flags = [str(beats(metrics["linear"][left][sl], metrics["linear"][right][sl])) for sl in PRIMARY_SLICES]
        lines.append("| " + " | ".join([f"{left} over {right}", *flags]) + " |")
    lines.extend(["", "The table uses the same `beats` rule as the decision. It does not reopen the label.", ""])
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "preregistration.json", preregistration_document())
    records = _prompts()
    measurement = OUT / "measurements.npz"
    index_path = OUT / "measurement_index.json"
    if os.environ.get("CCSO_FIT_ONLY") == "1" and measurement.exists():
        stored = np.load(measurement)
        h = stored["h"]
        g = stored["g"]
        directions = stored["directions"]
        delta = stored["delta"]
        print("loaded stored measurements", flush=True)
    else:
        print("preregistration written; loading Qwen", flush=True)
        model = _load_model()
        _freeze(model)
        if int(model.cfg.d_model) != 896 or int(model.cfg.n_layers) != 24:
            _stop("STOP: loaded Qwen shape drifted")
        directions = np.vstack([primary_direction(896, seed) for seed in DIRECTION_SEEDS])
        h, g, delta = _measure(model, records, directions)
        del model
        gc.collect()
        np.savez(
            measurement,
            h=h,
            g=g,
            directions=directions,
            delta=delta,
            alphas=ALPHAS,
        )
    _write(
        index_path,
        {
            "prompt_ids": [record.prompt_id for record in records],
            "roles": [record.role for record in records],
            "regimes": [record.regime_id for record in records],
            "layers": list(LAYERS),
            "direction_seeds": list(DIRECTION_SEEDS),
            "direction_sha256": [vector_sha256(directions[index]) for index in range(directions.shape[0])],
            "alphas": [float(value) for value in ALPHAS],
            "revision": QWEN_REVISION,
        },
    )
    print("fitting readouts", flush=True)
    metrics = run_readouts(
        h,
        g,
        directions,
        list(DIRECTION_SEEDS),
        delta,
        ALPHAS,
        [record.role for record in records],
        [record.regime_id for record in records],
    )
    decision = decide(metrics)
    _write(OUT / "metrics.json", metrics)
    _write(OUT / "decision.json", decision)
    manifest = [f"{_sha(path)}  {path.name}" for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "manifest.sha256"]
    (OUT / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    REPORT.write_text(_report(decision, metrics), encoding="utf-8")
    print(decision["decision"], flush=True)


if __name__ == "__main__":
    main()
