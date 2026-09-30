"""Preregistered gradient-field diagnostic. Not an MRSM rerun and not M3.

The rules live in src/mrsm/qf_gradient_field.py. This script writes that
preregistration, then measures. It does not choose alphas, the map, or the
null after seeing agreement.
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

from cognitive_self_model.m22_1.direction import vector_sha256  # noqa: E402
from cognitive_self_model.m22_1.outcome import resolve_outcome_tokens  # noqa: E402
from cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role  # noqa: E402
from src.mrsm import HEADS  # noqa: E402
from src.mrsm.qf_cross_model import functional_layers, public_slots, slot_catalog  # noqa: E402
from src.mrsm.qf_gradient_field import (  # noqa: E402
    GPT2_FUNCTIONAL_LAYERS,
    GPT2_NEGATIVE_ID,
    GPT2_POSITIVE_ID,
    GPT2_REVISION,
    QWEN_FUNCTIONAL_LAYERS,
    QWEN_NEGATIVE_ID,
    QWEN_POSITIVE_ID,
    QWEN_REVISION,
    REFERENCE_ALPHAS,
    SMALL_ALPHAS,
    SPLITS,
    alpha0_passes,
    assemble_decision,
    compare_prediction,
    gradient_check_passes,
    last_position_margin_gradient,
    preregistration_document,
)
from src.mrsm.qf_gradient_field import GRADIENT_CHECK_EPSILON as EPSILON  # noqa: E402
from src.mrsm.run_q import _load_model, _margins  # noqa: E402

OUT = ROOT / "artifacts" / "qf_gradient_field"
REPORT = ROOT / "reports" / "QF_GRADIENT_FIELD.md"
M1_TASK = ROOT / "artifacts" / "qf_cross_model" / "m1" / "task_mapping.json"


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


def _freeze_parameters(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _grouped_prompts() -> dict:
    grouped = {split: sorted(rows, key=lambda record: record.prompt_id) for split, rows in prompts_by_role(frozen_prompts()).items()}
    if set(grouped) != set(SPLITS):
        _stop("STOP: frozen prompt splits drifted")
    for split, rows in grouped.items():
        if len(rows) != 6:
            _stop(f"STOP: {split} does not have six prompts")
    discovery = {record.prompt_id for record in grouped["discovery"]}
    stored = set(json.loads(M1_TASK.read_text(encoding="utf-8"))["prompt_ids"])
    if discovery != stored:
        _stop("STOP: discovery prompts are not the regime-matrix prompts")
    return grouped


def _slots(layers: tuple[int, ...] | list[int], d_model: int, n_layers: int) -> list[dict]:
    expected = functional_layers(n_layers)
    if expected != list(layers):
        _stop("STOP: functional layer map does not match the preregistered layers")
    rows = slot_catalog(list(layers), d_model, n_layers=n_layers)
    if [row["slot"] for row in rows] != list(HEADS):
        _stop("STOP: slot order drifted")
    if any(not row["comparable"] for row in rows):
        _stop("STOP: a functional slot is outside the model")
    for row in rows:
        name = f"blocks.{row['layer']}.hook_resid_post"
        row["hook_name"] = name
    return rows


def _load_gpt2():
    from huggingface_hub import snapshot_download
    from transformer_lens import HookedTransformer

    snapshot_download("gpt2", revision=GPT2_REVISION)
    model = HookedTransformer.from_pretrained(
        "gpt2",
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        device="cpu",
        dtype=torch.float32,
        revision=GPT2_REVISION,
        local_files_only=True,
    )
    _freeze_parameters(model)
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if int(model.cfg.n_layers) != 12 or int(model.cfg.d_model) != 768:
        _stop("STOP: GPT-2 shape does not match the M1 manifest")
    return model


def _outcome(model, positive_id: int, negative_id: int) -> tuple[int, int]:
    tokens = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    if not tokens["single_piece"]:
        _stop("STOP: outcome is not a single piece")
    if int(tokens["positive_token_id"]) != positive_id or int(tokens["negative_token_id"]) != negative_id:
        _stop("STOP: outcome token ids do not match the frozen pair")
    return positive_id, negative_id


def _item(row: dict, alpha: float) -> dict:
    return {
        "name": row["slot"],
        "layer": row["layer"],
        "direction_name": row["direction_name"],
        "alpha": float(alpha),
        "vector": row["vector"],
    }


def _project(gradient: np.ndarray, vector: np.ndarray) -> float:
    return float(np.dot(np.asarray(gradient, dtype=np.float64), np.asarray(vector, dtype=np.float64)))


def _gradients(model, records, layers: list[int], positive_id: int, negative_id: int) -> dict[tuple[str, int], np.ndarray]:
    found = {}
    total = len(records) * len(layers)
    done = 0
    for record in records:
        tokens = model.to_tokens([record.text], prepend_bos=True)
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            if hook_name not in model.hook_dict:
                _stop(f"STOP: {hook_name} is not a hook")
            started = time.perf_counter()
            gradient = last_position_margin_gradient(
                model.run_with_hooks, tokens, layer, positive_id, negative_id
            )
            found[(record.prompt_id, layer)] = gradient
            done += 1
            print(
                f"gradient {done}/{total} {record.prompt_id} layer {layer} {time.perf_counter() - started:.1f}s",
                flush=True,
            )
        del tokens
        gc.collect()
    return found


def _margins_for(model, texts, row: dict, alpha: float, positive_id: int, negative_id: int) -> list[float]:
    return _margins(model, texts, [_item(row, alpha)], positive_id, negative_id)


def measure(model, records_by_split, slots, positive_id: int, negative_id: int) -> dict:
    layers = sorted({int(row["layer"]) for row in slots})
    flat = [record for split in SPLITS for record in records_by_split[split]]
    gradients = _gradients(model, flat, layers, positive_id, negative_id)
    texts = [record.text for record in flat]
    base = _margins(model, texts, [], positive_id, negative_id)
    projections = {record.prompt_id: {} for record in flat}
    cells_by_split = {split: [] for split in SPLITS}
    failures = {split: [] for split in SPLITS}
    alpha0_failures = {split: [] for split in SPLITS}
    audits = []
    for slot_index, row in enumerate(slots, start=1):
        print(f"intervention slot {slot_index}/{len(slots)} {row['slot']}", flush=True)
        plus = _margins_for(model, texts, row, EPSILON, positive_id, negative_id)
        minus = _margins_for(model, texts, row, -EPSILON, positive_id, negative_id)
        zero = _margins_for(model, texts, row, 0.0, positive_id, negative_id)
        treated = {
            alpha: _margins_for(model, texts, row, alpha, positive_id, negative_id)
            for alpha in SMALL_ALPHAS + REFERENCE_ALPHAS
        }
        for index, record in enumerate(flat):
            projection = _project(gradients[(record.prompt_id, row["layer"])], row["vector"])
            projections[record.prompt_id][row["slot"]] = projection
            central = (plus[index] - minus[index]) / (2.0 * EPSILON)
            check = gradient_check_passes(projection, central)
            if not check:
                failures[record.role].append(record.prompt_id + ":" + row["slot"])
            zero_delta = float(base[index] - zero[index])
            if not alpha0_passes(zero_delta):
                alpha0_failures[record.role].append(record.prompt_id + ":" + row["slot"])
            audits.append({
                "split": record.role,
                "prompt_id": record.prompt_id,
                "slot": row["slot"],
                "layer": row["layer"],
                "projection": projection,
                "central_difference": float(central),
                "gradient_check_passed": check,
                "alpha0_delta": zero_delta,
                "alpha0_passed": alpha0_passes(zero_delta),
                "gradient_sha256": vector_sha256(gradients[(record.prompt_id, row["layer"])]),
            })
            for alpha, margins in treated.items():
                delta = float(base[index] - margins[index])
                cell = compare_prediction(alpha, projection, delta)
                cell.update({
                    "split": record.role,
                    "prompt_id": record.prompt_id,
                    "regime_id": record.regime_id,
                    "slot": row["slot"],
                    "layer": row["layer"],
                    "direction_name": row["direction_name"],
                })
                cells_by_split[record.role].append(cell)
    measured = {}
    for split in SPLITS:
        field = [
            {
                "prompt_id": record.prompt_id,
                "regime_id": record.regime_id,
                "vector": [projections[record.prompt_id][slot] for slot in HEADS],
            }
            for record in records_by_split[split]
        ]
        measured[split] = {
            "cells": cells_by_split[split],
            "field": field,
            "gradient_ok": len(failures[split]) == 0,
            "alpha0_ok": len(alpha0_failures[split]) == 0,
            "gradient_check_failures": failures[split],
            "alpha0_failures": alpha0_failures[split],
        }
        print(
            f"{split}: gradient_ok={measured[split]['gradient_ok']} alpha0_ok={measured[split]['alpha0_ok']} "
            f"failures={len(failures[split])} alpha0_failures={len(alpha0_failures[split])}",
            flush=True,
        )
    return {"splits": measured, "audits": audits, "layers": layers}


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _report(decision: dict, qwen_audit: dict, gpt_audit: dict, descriptive: dict | None = None) -> str:
    lines = [
        "# QF Gradient Field",
        "",
        "QF-GRADIENT-FIELD. Not an MRSM gate and not M3.",
        "",
        "## 1. Sign correction",
        "",
        "Δ is the margin drop `m(h) - m(h + α v)`, the same drop as the regime measurements.",
        "The expansion fixed before this run is",
        "",
        "`Δ ≈ -α vᵀ g(x) - (α² / 2) vᵀ H_m(x) v`.",
        "",
        "For α > 0 the first-order term has the opposite sign of `vᵀ g(x)`.",
        "The regime-effect sign matrix is not a gradient.",
        "α = ±1 is reported as the historical scale and is not an input to the linear classification.",
        "",
        "## 2. Preregistered question",
        "",
        "The same frozen prompts, functional layers, and residual site are used.",
        "`g(x)` is the direct derivative of the yes/no margin at `blocks.{layer}.hook_resid_post` on the last token.",
        "The linear prediction `-α vᵀ g(x)` is compared with measured Δ on the small grid "
        + ", ".join(f"{alpha:.2f}" for alpha in SMALL_ALPHAS)
        + ".",
        "Validation and replication are held-out repeats. They are not pooled with discovery.",
        "The low-dimensional field is `φ_s(x) = v_sᵀ g(x)` on the eight functional slots.",
        "The map is the M1 functional slot alignment: Qwen layers "
        + ", ".join(str(layer) for layer in QWEN_FUNCTIONAL_LAYERS)
        + " and GPT-2 layers "
        + ", ".join(str(layer) for layer in GPT2_FUNCTIONAL_LAYERS)
        + ".",
        "Directions are the frozen seeds resampled in each model width. No rotation is learned.",
        "GPT-2 has no layers 15 or 23; those coordinate cells stay `NOT_COMPARABLE` and are not imputed.",
        "Predictability is the mean cosine of prompt-aligned fields against all 720 GPT-2 prompt permutations inside one split.",
        "",
        "## 3. Decisions",
        "",
        f"- Qwen linear: `{decision['linear']['qwen']['label']}`.",
        f"- GPT-2 linear: `{decision['linear']['gpt2']['label']}`.",
        f"- Field: `{decision['field_decision']}`.",
        f"- {decision['primary_interpretation']}",
        "",
        "These labels are the preregistered instrument result. "
        "A failed gradient check blocks the linear and field classifications. "
        "The check is not loosened after the run, and the blocked readings in section 8 do not replace the labels.",
        "",
        "## 4. Instrument audit",
        "",
        "A directional derivative must match the central difference at ε = "
        + f"{EPSILON:g}, "
        "and an α = 0 hook must leave Δ at most 1e-4.",
        "A failed check makes that comparison inconclusive. It is not reclassified.",
        "",
        "| Model | Split | Gradient check | Alpha 0 | Gradient failures | Alpha-0 failures |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for name, audit in (("Qwen", qwen_audit), ("GPT-2", gpt_audit)):
        for split in SPLITS:
            payload = audit["splits"][split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        split,
                        str(payload["gradient_ok"]),
                        str(payload["alpha0_ok"]),
                        str(len(payload["gradient_check_failures"])),
                        str(len(payload["alpha0_failures"])),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## 5. Linear sign comparison", ""])
    lines.append("| Model | Split | Alpha | In linear test | Agree | Disagree | Near zero | Median Δ/linear |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
    for name, key in (("Qwen", "qwen"), ("GPT-2", "gpt2")):
        for split in SPLITS:
            block = decision["linear"][key]["splits"][split]
            for alpha, counts in block["by_alpha"].items():
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            name,
                            split,
                            alpha,
                            str(counts["in_linear_test"]),
                            str(counts["agree"]),
                            str(counts["disagree"]),
                            str(counts["near_zero"]),
                            _fmt(counts["median_delta_over_linear"]),
                        ]
                    )
                    + " |"
                )
            lines.append(f"| {name} | {split} | label |  | `{block['label']}` |  |  |  |")
    lines.extend(
        [
            "",
            "## 6. Cross-model field",
            "",
            "Each row is one split of six prompts. The null is exhaustive inside that split.",
            "A within-regime permutation is not used, because a regime has two discovery prompts.",
            "",
            "| Split | Status | Identity mean cosine | Permutations strictly better | Permutations tied with identity |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        result = decision["field_splits"][split]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    f"`{result['status']}`",
                    _fmt(result.get("identity_mean_cosine")),
                    _fmt(result.get("n_strictly_better")),
                    _fmt(result.get("n_equal_to_identity")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 7. What this does not say",
            "",
            "This experiment does not assume a context-independent or model-independent mechanism.",
            "A negative or partial result says the declared field was not predictable for these models, these prompts, and this map.",
            "It does not say that no such structure exists.",
            "DAS and the Curse of Multiple Mediators paper are related citations, not this protocol.",
            "M3 was not run. MRSM was not rerun. The Self-Model was not modified.",
            "No model is ranked.",
            "",
            "## 8. Descriptive readings blocked by the instrument check",
            "",
            "The gradient check compares `vᵀ g` with a central difference at ε = "
            + f"{EPSILON:g}. "
            "It fails when the absolute discrepancy exceeds 1e-4 and the relative discrepancy exceeds 1e-2. "
            "Alpha 0 is a separate audit. "
            "Section 5 is the sign and magnitude comparison. "
            "Neither table reopens the classification.",
            "",
        ]
    )
    if descriptive:
        lines.extend(
            [
                "| Model | Median relative discrepancy | Median absolute discrepancy | Maximum relative discrepancy | Failed cells |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ("qwen", "gpt2"):
            item = descriptive["gradient_check"][name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(item["median_relative"]),
                        _fmt(item["median_absolute"]),
                        _fmt(item["max_relative"]),
                        str(item["failed"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "The field permutation below uses the recorded projections. `decision_input` is false.",
                "",
                "| Split | Blocked status | Identity mean cosine | Permutations strictly better |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for split in SPLITS:
            item = descriptive["field"][split]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        f"`{item['status']}`",
                        _fmt(item.get("identity_mean_cosine")),
                        _fmt(item.get("n_strictly_better")),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "Where a small-grid sign disagrees, the linear term is small. "
                "Those cells are listed in `descriptive_not_decision.json`. "
                "They are not a new threshold and not a repaired label.",
                "",
                descriptive["scope"],
                "",
            ]
        )
    return "\n".join(lines)


def build_descriptive(qwen_doc: dict, gpt_doc: dict) -> dict:
    """Readings that stay outside the decision when the instrument check fails."""
    from statistics import median

    from src.mrsm.qf_gradient_field import field_null

    def check_summary(doc: dict) -> dict:
        relative = []
        absolute = []
        failed = 0
        for row in doc["audits"]:
            error = abs(float(row["projection"]) - float(row["central_difference"]))
            scale = max(abs(float(row["projection"])), abs(float(row["central_difference"])), 1e-12)
            relative.append(error / scale)
            absolute.append(error)
            if not row["gradient_check_passed"]:
                failed += 1
        return {
            "n": len(relative),
            "failed": failed,
            "median_relative": median(relative),
            "median_absolute": median(absolute),
            "max_relative": max(relative),
        }

    disagreements = []
    for model, doc in (("qwen", qwen_doc), ("gpt2", gpt_doc)):
        for split in SPLITS:
            for cell in doc["splits"][split]["cells"]:
                if cell["in_small_grid"] and cell["relation"] == "DISAGREE":
                    disagreements.append({
                        "model": model,
                        "split": split,
                        "prompt_id": cell["prompt_id"],
                        "slot": cell["slot"],
                        "alpha_key": cell["alpha_key"],
                        "projection": cell["projection"],
                        "linear": cell["linear"],
                        "delta": cell["delta"],
                        "ratio": cell["ratio"],
                    })
    field = {}
    for split in SPLITS:
        result = field_null(
            [row["vector"] for row in qwen_doc["splits"][split]["field"]],
            [row["vector"] for row in gpt_doc["splits"][split]["field"]],
        )
        result["decision_input"] = False
        field[split] = result
    return {
        "decision_input": False,
        "reason": "The preregistered gradient check failed, so these readings do not replace the inconclusive labels.",
        "gradient_check": {"qwen": check_summary(qwen_doc), "gpt2": check_summary(gpt_doc)},
        "small_grid_disagreements": disagreements,
        "field": field,
        "scope": (
            "The blocked field reading is about these models, these prompts, and the functional slot map. "
            "It does not say the structure cannot exist."
        ),
    }


def _public_measurement(model_name: str, revision: str, payload: dict, slots: list[dict]) -> dict:
    return {
        "model_name": model_name,
        "revision": revision,
        "slots": public_slots(slots),
        "splits": payload["splits"],
        "audits": payload["audits"],
        "layers": payload["layers"],
        "ambient_gradient_retained": False,
        "retained_representation": "phi_s = v_s^T g_layer(s)",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "preregistration.json", preregistration_document())
    grouped = _grouped_prompts()
    print("preregistration written; loading Qwen", flush=True)
    qwen = _load_model()
    _freeze_parameters(qwen)
    _outcome(qwen, QWEN_POSITIVE_ID, QWEN_NEGATIVE_ID)
    qwen_slots = _slots(QWEN_FUNCTIONAL_LAYERS, int(qwen.cfg.d_model), int(qwen.cfg.n_layers))
    qwen_payload = measure(qwen, grouped, qwen_slots, QWEN_POSITIVE_ID, QWEN_NEGATIVE_ID)
    _write(OUT / "qwen_measurements.json", _public_measurement("Qwen/Qwen2.5-0.5B", QWEN_REVISION, qwen_payload, qwen_slots))
    del qwen
    gc.collect()
    print("loading GPT-2", flush=True)
    gpt2 = _load_gpt2()
    _outcome(gpt2, GPT2_POSITIVE_ID, GPT2_NEGATIVE_ID)
    gpt_slots = _slots(GPT2_FUNCTIONAL_LAYERS, int(gpt2.cfg.d_model), int(gpt2.cfg.n_layers))
    gpt_payload = measure(gpt2, grouped, gpt_slots, GPT2_POSITIVE_ID, GPT2_NEGATIVE_ID)
    _write(OUT / "gpt2_measurements.json", _public_measurement("gpt2", GPT2_REVISION, gpt_payload, gpt_slots))
    del gpt2
    gc.collect()
    for split in SPLITS:
        qwen_ids = [row["prompt_id"] for row in qwen_payload["splits"][split]["field"]]
        gpt_ids = [row["prompt_id"] for row in gpt_payload["splits"][split]["field"]]
        if qwen_ids != gpt_ids:
            _stop("STOP: cross-model prompt alignment drifted")
    decision = assemble_decision(
        {
            "qwen": qwen_payload["splits"],
            "gpt2": gpt_payload["splits"],
        }
    )
    _write(OUT / "decision.json", decision)
    descriptive = build_descriptive(
        _public_measurement("Qwen/Qwen2.5-0.5B", QWEN_REVISION, qwen_payload, qwen_slots),
        _public_measurement("gpt2", GPT2_REVISION, gpt_payload, gpt_slots),
    )
    _write(OUT / "descriptive_not_decision.json", descriptive)
    manifest_lines = [f"{_sha(path)}  {path.name}" for path in sorted(OUT.glob("*.json"))]
    (OUT / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    REPORT.write_text(_report(decision, qwen_payload, gpt_payload, descriptive), encoding="utf-8")
    print(decision["linear"]["qwen"]["label"])
    print(decision["linear"]["gpt2"]["label"])
    print(decision["field_decision"])


if __name__ == "__main__":
    main()
