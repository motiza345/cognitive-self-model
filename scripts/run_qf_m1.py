"""QF-M1 cross-model signal diagnostic. Not an MRSM rerun. Stops before M2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

# Read before huggingface_hub import. The Xet route 404s for this revision.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cognitive_self_model.m22_1.outcome import resolve_outcome_tokens  # noqa: E402
from cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role  # noqa: E402
from src.mrsm import HEADS  # noqa: E402
from src.mrsm.qf_cross_model import (  # noqa: E402
    ALPHA,
    FROZEN_QWEN_MATRIX,
    N_CANDIDATE_LAYERS,
    ORTHOGONAL_SEED,
    PRIMARY_SEED,
    decide_m1,
    functional_layers,
    public_slots,
    qwen_coordinate_layers,
    slot_catalog,
)
from src.mrsm.qf_regime import FROZEN_REGIMES, heterogeneity, regime_cell  # noqa: E402
from src.mrsm.run_q import _margins  # noqa: E402

OUT = ROOT / "artifacts" / "qf_cross_model"
M1 = OUT / "m1"
REPORT = ROOT / "reports" / "QF_M1_CROSS_MODEL.md"
MODEL_NAME = "gpt2"
P_FREEZE = "2cccafd047044332828eaf602b69f4267852cba2"
DECISION_SPLIT = "discovery"


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


def _resolve_revision() -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_NAME)
    revision = str(info.sha or "")
    if len(revision) != 40:
        _stop(f"STOP: no exact revision for {MODEL_NAME}")
    return revision


def _load(revision: str):
    import os

    from huggingface_hub import snapshot_download
    from transformer_lens import HookedTransformer

    # The Xet token route 404s for this revision. The bytes are still that revision.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    snapshot_download(MODEL_NAME, revision=revision)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
        fold_value_biases=True,
        device="cpu",
        dtype=torch.float32,
        revision=revision,
        local_files_only=True,
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if str(getattr(model.cfg, "model_name", MODEL_NAME)) not in {MODEL_NAME, "gpt2"}:
        _stop("STOP: loaded model name does not match the frozen gpt2 selection")
    return model


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _measure(model, records, slots, positive_id: int, negative_id: int) -> dict:
    texts = [record.text for record in records]
    base = _margins(model, texts, [], positive_id, negative_id)
    cache = {}
    for row in slots:
        if not row["comparable"]:
            continue
        key = (row["layer"], row["direction_name"])
        if key in cache:
            continue
        item = {
            "name": row["slot"],
            "layer": row["layer"],
            "direction_name": row["direction_name"],
            "alpha": ALPHA,
            "vector": row["vector"],
            "vector_sha256": row["vector_sha256"],
        }
        treated = _margins(model, texts, [item], positive_id, negative_id)
        null_item = dict(item)
        null_item["alpha"] = 0.0
        null_treated = _margins(model, texts, [null_item], positive_id, negative_id)
        cache[key] = (treated, null_treated)
    measured = {}
    for row in slots:
        if not row["comparable"]:
            measured[row["slot"]] = {
                "comparable": False,
                "status": "NOT_COMPARABLE",
                "layer": row["layer"],
                "regimes": {
                    regime: {"status": "NOT_COMPARABLE", "sign_symbol": "NOT_COMPARABLE"}
                    for regime in FROZEN_REGIMES
                },
            }
            continue
        treated, null_treated = cache[(row["layer"], row["direction_name"])]
        by_regime = {name: {"effects": [], "nulls": [], "prompts": []} for name in FROZEN_REGIMES}
        for record, base_margin, treated_margin, null_margin in zip(records, base, treated, null_treated):
            bucket = by_regime[record.regime_id]
            bucket["effects"].append(float(base_margin - treated_margin))
            bucket["nulls"].append(float(base_margin - null_margin))
            bucket["prompts"].append(record.prompt_id)
        cells = {}
        for regime in FROZEN_REGIMES:
            cell = regime_cell(by_regime[regime]["effects"], by_regime[regime]["nulls"])
            cell["prompt_ids"] = by_regime[regime]["prompts"]
            cell["effects"] = by_regime[regime]["effects"]
            cell["nulls"] = by_regime[regime]["nulls"]
            cells[regime] = cell
        pooled = mean(value for regime in FROZEN_REGIMES for value in by_regime[regime]["effects"])
        measured[row["slot"]] = {
            "comparable": True,
            "layer": row["layer"],
            "direction_name": row["direction_name"],
            "vector_sha256": row["vector_sha256"],
            "regimes": cells,
            "pooled_mean_descriptive_only": pooled,
            "heterogeneity": heterogeneity({regime: by_regime[regime]["effects"] for regime in FROZEN_REGIMES}),
        }
    return measured


def _symbols(measured: dict) -> dict:
    return {
        slot: {regime: measured[slot]["regimes"][regime]["sign_symbol"] for regime in FROZEN_REGIMES}
        for slot in HEADS
    }


def _report(selection, task, mapping, effects, matrix, decision) -> str:
    functional = effects["functional_counterpart"]
    lines = [
        "# QF-M1 Cross-Model Identity / Signal Diagnostic",
        "",
        "QF-CROSS-MODEL-DIAGNOSTIC. Not an MRSM gate and not a mechanism discovery.",
        "M2 and M3 were not executed.",
        "",
        "## 1. Frozen State",
        "",
        "| Item | State |",
        "| --- | --- |",
        "| P | H1 PASS, H2 PASS, H3 PASS, H4 PASS, Leakage PASS |",
        "| Q | H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL, Leakage PASS |",
        "| DEC-010 | `REDEFINE_SCALE` |",
        f"| p_freeze | `{P_FREEZE}` |",
        "| Qwen revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |",
        "| Qwen regime decision | `MIXED_SIGNAL` |",
        "| MRSM rerun | not performed |",
        "",
        "## 2. Model B Selection",
        "",
        "One model was selected. Candidates were not ranked.",
        "",
        f"- Model: `{selection['model_name']}`",
        f"- Revision: `{selection['revision']}`",
        f"- Architecture: `{selection['architecture']}`",
        f"- Parameter count: `{selection['parameter_count']}`",
        f"- Parameter count definition: {selection.get('parameter_count_definition', '')}",
        f"- dtype: `{selection['dtype']}`",
        f"- device: `{selection['device']}`",
        f"- Tokenizer: `{selection['tokenizer']}`",
        f"- Compatibility: `{selection['compatibility_status']}`",
        f"- Reason: {selection['reason_for_selection']}",
        "",
        "## 3. Identity Freeze",
        "",
        "The loader requested this revision and `local_files_only`. No other revision was substituted.",
        "",
        "## 4. Task Mapping",
        "",
        task["minimum_transformation"],
        "Regimes kept: completion, instruction, syntax. None were dropped.",
        f"Primary split: `{DECISION_SPLIT}`, the same split as the frozen Qwen matrix.",
        "",
        "## 5. Intervention Mapping",
        "",
        "Declared before any Model B effect was measured.",
        "Slot names are residual-stream labels from the frozen catalog, not attention-head indices.",
        f"Alpha conversion: `{mapping['alpha_conversion']}`. Alpha stays `{ALPHA}`.",
        f"Direction seeds: primary `{PRIMARY_SEED}`, orthogonal `{ORTHOGONAL_SEED}`, width `{mapping['d_model']}`.",
        "",
        "| Slot | Qwen layer | Functional layer | Coordinate status |",
        "| --- | ---: | --- | --- |",
    ]
    functional_rows = {row["slot"]: row for row in mapping["functional_counterpart"]}
    coordinate_rows = {row["slot"]: row for row in mapping["coordinate"]}
    for slot in HEADS:
        functional_layer = functional_rows[slot]["layer"] if slot in functional_rows else "NOT_COMPARABLE"
        coordinate = coordinate_rows.get(slot)
        status = "comparable" if coordinate and coordinate["comparable"] else "NOT_COMPARABLE"
        coordinate_layer = coordinate["layer"] if coordinate else "NOT_COMPARABLE"
        lines.append(f"| {slot} | {coordinate_layer} | {functional_layer} | {status} |")
    lines.extend([
        "",
        "## 6. Functional-Counterpart Results",
        "",
        "Primary comparison. Statistics use the QF-Regime definitions.",
        "",
        "| Slot | Regime | n | Mean | Median | Std | SE | Effect minus null | Sign | Consistency | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ])
    for slot in HEADS:
        if slot not in functional:
            continue
        for regime in FROZEN_REGIMES:
            cell = functional[slot]["regimes"][regime]
            if cell.get("status") == "NOT_COMPARABLE":
                lines.append(f"| {slot} | {regime} |  |  |  |  |  |  |  |  | NOT_COMPARABLE |")
                continue
            lines.append(
                f"| {slot} | {regime} | {cell['n']} | {_fmt(cell['mean_effect'])} | {_fmt(cell['median_effect'])} | "
                f"{_fmt(cell['std_effect'])} | {_fmt(cell['standard_error'])} | {_fmt(cell['effect_minus_null'])} | "
                f"{cell['sign_class']} | {_fmt(cell['sign_consistency'])} | {cell['status']} |"
            )
    lines.extend([
        "",
        "## 7. Regime-Specific Nulls",
        "",
        "Alpha 0 inside the same slot and the same regime. Not one global null.",
        "",
        "| Slot | Regime | Null mean | Null std |",
        "| --- | --- | ---: | ---: |",
    ])
    for slot in HEADS:
        if slot not in functional:
            continue
        for regime in FROZEN_REGIMES:
            cell = functional[slot]["regimes"][regime]
            if "null_mean" not in cell:
                lines.append(f"| {slot} | {regime} |  |  |")
                continue
            lines.append(f"| {slot} | {regime} | {_fmt(cell['null_mean'])} | {_fmt(cell['null_std'])} |")
    lines.extend([
        "",
        "## 8. Effect Matrices",
        "",
        "Qwen matrix is the frozen discovery matrix. It was not recomputed.",
        "A symbol is the sign of the regime mean. UNSTABLE in the results table means the two discovery prompts disagree; the matrix still shows the mean sign.",
        "",
        "| Slot | Qwen completion | Qwen instruction | Qwen syntax | B completion | B instruction | B syntax |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    model_b_matrix = matrix.get("model_b_functional_discovery") or {}
    for slot in HEADS:
        qwen = matrix["qwen_frozen_discovery"][slot]
        other = model_b_matrix.get(slot) or {regime: "NOT_COMPARABLE" for regime in FROZEN_REGIMES}
        lines.append(
            f"| {slot} | {qwen['completion']} | {qwen['instruction']} | {qwen['syntax']} | "
            f"{other['completion']} | {other['instruction']} | {other['syntax']} |"
        )
    lines.extend([
        "",
        "Coordinate comparison, same Qwen layer index where the index exists:",
        "",
        "| Slot | completion | instruction | syntax |",
        "| --- | --- | --- | --- |",
    ])
    coordinate_matrix = matrix.get("model_b_coordinate_discovery") or {}
    for slot in HEADS:
        row = coordinate_matrix.get(slot) or {regime: "NOT_COMPARABLE" for regime in FROZEN_REGIMES}
        lines.append(f"| {slot} | {row['completion']} | {row['instruction']} | {row['syntax']} |")
    lines.extend([
        "",
        "## 9. Decision",
        "",
        f"`{decision['m1_decision']}`",
        "",
        decision["key_difference"],
        "",
        "## 10. Observed, Inferred, Not Established",
        "",
        "### Observed",
        "",
        "- Model B identity, the unchanged prompts, the regime-specific effects, the regime-specific nulls, and the two matrices above.",
        "- The Qwen matrix remains the frozen discovery matrix.",
        "",
        "### Inferred",
        "",
        decision["inferred"],
        "",
        "Recorded associations, not an isolated cause:",
        "",
        "- Architecture identity differs: GPT-2 versus Qwen2.",
        "- Task strings do not differ.",
        "- Intervention type and alpha do not differ. Alpha was not tuned.",
        "- Direction width differs because the residual widths differ. The numeric Qwen vector was not reused.",
        "- No experiment here changes only one of those factors, so a cell difference is not attributed to one of them.",
        "",
        "### Not Established",
        "",
    ])
    for item in decision["not_established"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 11. Stop",
        "",
        "M1 is complete. M2 was not started. M3 was not started.",
        "The Self-Model was not modified. MRSM was not rerun.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    frozen_matrix_path = ROOT / "artifacts" / "qf_regime" / "regime_decision.json"
    recorded = json.loads(frozen_matrix_path.read_text(encoding="utf-8"))["effect_matrix"]
    if recorded != FROZEN_QWEN_MATRIX:
        _stop("STOP: frozen Qwen effect matrix does not match the recorded regime decision")
    revision = _resolve_revision()
    print(f"pinned {MODEL_NAME} {revision}", flush=True)
    model = _load(revision)
    n_layers = int(model.cfg.n_layers)
    d_model = int(model.cfg.d_model)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    outcome = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    layers = functional_layers(n_layers)
    hook_ok = "blocks.0.hook_resid_post" in model.hook_dict
    comparable = bool(outcome["single_piece"]) and layers is not None and hook_ok
    reason = None
    if not outcome["single_piece"]:
        reason = "The strings ' yes' and ' no' are not single Model B tokens. No prompt was rewritten."
    elif layers is None:
        reason = "The frozen four-layer candidate grid is not defined for this depth."
    elif not hook_ok:
        reason = "Model B has no residual hook at blocks.0.hook_resid_post."
    architecture = str(getattr(model.cfg, "original_architecture", None) or getattr(model.cfg, "model_name", MODEL_NAME))
    tokenizer_name = f"{type(model.tokenizer).__name__}:{getattr(model.tokenizer, 'name_or_path', MODEL_NAME)}"
    selection = {
        "model_name": MODEL_NAME,
        "revision": revision,
        "architecture": architecture,
        "parameter_count": parameter_count,
        "parameter_count_definition": (
            "Sum of parameters in the loaded HookedTransformer. "
            "The source checkpoint has 12 layers, d_model 768, and 12 heads. "
            "Loading unties the token embedding, so this count is larger than the tied checkpoint."
        ),
        "dtype": "float32",
        "device": "cpu",
        "tokenizer": tokenizer_name,
        "selection_criteria": {
            "open_weight": True,
            "reproducible_checkpoint": True,
            "activation_access": True,
            "hook_access": "blocks.0.hook_resid_post" in model.hook_dict,
            "hook_name": "blocks.{layer}.hook_resid_post",
            "practical_size": True,
            "tokenizer_available": True,
            "different_family_from_qwen": True,
            "not_another_qwen_checkpoint": True,
        },
        "reason_for_selection": (
            "gpt2 is an open-weight GPT-2 checkpoint with a pinned Hugging Face revision, "
            "a tokenizer, and TransformerLens residual hooks. Its family is GPT-2, not Qwen2. "
            "The checkpoint is small enough for repeated residual interventions. "
            "It is not another Qwen checkpoint. Candidates were not ranked."
        ),
        "compatibility_status": "IDENTITY_FROZEN" if comparable else "NOT_COMPARABLE",
        "ranked_candidates": False,
        "declared_before_effects": True,
    }
    grouped = prompts_by_role(frozen_prompts())
    discovery = grouped[DECISION_SPLIT]
    task = {
        "regimes": list(FROZEN_REGIMES),
        "dropped_regimes": [],
        "primary_split": DECISION_SPLIT,
        "prompt_ids": [record.prompt_id for record in discovery],
        "prompt_sha256": {record.prompt_id: record.prompt_sha256 for record in discovery},
        "texts_modified": False,
        "minimum_transformation": "No prompt text was changed. Batching uses left padding, the same call setting as the Qwen measurement.",
        "outcome": outcome,
        "prepend_bos": True,
    }
    functional = slot_catalog(layers, d_model, n_layers=n_layers) if layers is not None else []
    coordinate = slot_catalog(qwen_coordinate_layers(), d_model, n_layers=n_layers)
    mapping = {
        "declared_before_effects": True,
        "slot_meaning": "residual-stream label, not an attention head",
        "qwen_coordinate_layers": qwen_coordinate_layers(),
        "functional_layers": layers,
        "functional_rule": f"candidate_layers(n_layers, {N_CANDIDATE_LAYERS}), declared before effects",
        "coordinate_rule": "Qwen layer index if it exists in Model B; otherwise NOT_COMPARABLE",
        "alpha": ALPHA,
        "alpha_conversion": "NONE",
        "alpha_rule": "Alpha remains 1.0 times a unit direction in this model's residual stream. It was not rescaled.",
        "direction_seeds": {"primary": PRIMARY_SEED, "orthogonal": ORTHOGONAL_SEED},
        "d_model": d_model,
        "n_layers": n_layers,
        "hook": "blocks.{layer}.hook_resid_post",
        "metric": "margin_drop_unhooked_minus_intervened",
        "null": "alpha_0_inside_the_same_slot_and_regime",
        "functional_counterpart": public_slots(functional),
        "coordinate": public_slots(coordinate),
    }
    manifest_identity = {
        "model_name": MODEL_NAME,
        "revision": revision,
        "tokenizer": tokenizer_name,
        "dtype": "float32",
        "device": "cpu",
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": int(model.cfg.n_heads),
        "parameter_count": parameter_count,
        "loading_config": {
            "fold_ln": True,
            "center_writing_weights": True,
            "center_unembed": True,
            "fold_value_biases": True,
            "revision": revision,
            "local_files_only": True,
            "fallback_revision": None,
            "hf_hub_disable_xet": True,
        },
        "original_architecture": architecture,
    }
    _write(OUT / "m1_model_selection.json", selection)
    _write(M1 / "model_manifest.json", manifest_identity)
    _write(M1 / "task_mapping.json", task)
    _write(M1 / "intervention_mapping.json", mapping)
    if not comparable:
        decision = decide_m1(comparable=False, cells=None, incomparability_reason=reason)
        matrix = {
            "qwen_frozen_discovery": FROZEN_QWEN_MATRIX,
            "model_b_functional_discovery": None,
            "model_b_coordinate_discovery": None,
            "decision_uses": "model_b_functional_discovery",
        }
        effects = {"functional_counterpart": {}, "measured": False}
        _write(M1 / "regime_effects.json", effects)
        _write(M1 / "regime_nulls.json", {"measured": False, "global_null": False})
        _write(M1 / "effect_matrix.json", matrix)
        _finish(selection, task, mapping, effects, matrix, decision)
        print(decision["m1_decision"])
        return
    positive_id = int(outcome["positive_token_id"])
    negative_id = int(outcome["negative_token_id"])
    print("measuring functional counterpart and valid coordinates", flush=True)
    functional_measured = _measure(model, discovery, functional, positive_id, negative_id)
    coordinate_measured = _measure(model, discovery, coordinate, positive_id, negative_id)
    functional_cells = {slot: functional_measured[slot]["regimes"] for slot in HEADS}
    decision = decide_m1(comparable=True, cells=functional_cells)
    effects = {
        "metric": "margin_drop_unhooked_minus_intervened",
        "primary_split": DECISION_SPLIT,
        "statistics": "QF-Regime regime_cell",
        "functional_counterpart": functional_measured,
        "coordinate": coordinate_measured,
    }
    nulls = {
        "control": "alpha_0_inside_the_same_slot_and_regime",
        "global_null": False,
        "functional_counterpart": {
            slot: {
                regime: {
                    "null_mean": functional_measured[slot]["regimes"][regime].get("null_mean"),
                    "null_std": functional_measured[slot]["regimes"][regime].get("null_std"),
                    "nulls": functional_measured[slot]["regimes"][regime].get("nulls"),
                }
                for regime in FROZEN_REGIMES
            }
            for slot in HEADS
        },
    }
    matrix = {
        "qwen_frozen_discovery": FROZEN_QWEN_MATRIX,
        "model_b_functional_discovery": _symbols(functional_measured),
        "model_b_coordinate_discovery": _symbols(coordinate_measured),
        "decision_uses": "model_b_functional_discovery",
    }
    _write(M1 / "regime_effects.json", effects)
    _write(M1 / "regime_nulls.json", nulls)
    _write(M1 / "effect_matrix.json", matrix)
    _finish(selection, task, mapping, effects, matrix, decision)
    print(decision["m1_decision"])
    print(decision["key_difference"])


def _finish(selection, task, mapping, effects, matrix, decision) -> None:
    frozen_paths = {
        "reports/MRSM_FINAL_RESULT.md": ROOT / "reports/MRSM_FINAL_RESULT.md",
        "reports/Q_FAILURE_ANALYSIS.md": ROOT / "reports/Q_FAILURE_ANALYSIS.md",
        "reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md": ROOT / "reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md",
        "reports/QF_REGIME_SEPARATION.md": ROOT / "reports/QF_REGIME_SEPARATION.md",
        "configs/mrsm_prereg.yaml": ROOT / "configs/mrsm_prereg.yaml",
    }
    payload = {
        "m1_decision": decision["m1_decision"],
        "model_b": selection["model_name"],
        "revision": selection["revision"],
        "key_difference": decision["key_difference"],
        "sign_differences": decision["sign_differences"],
        "mechanism_identity": "NOT_EVALUATED",
        "m2_executed": False,
        "m3_executed": False,
        "inferred": decision["inferred"],
        "not_established": decision["not_established"],
        "qwen_effect_matrix": FROZEN_QWEN_MATRIX,
        "model_b_effect_matrix": matrix.get("model_b_functional_discovery"),
    }
    _write(M1 / "decision.json", payload)
    digest_lines = []
    for path in sorted(M1.glob("*.json")):
        if path.name == "reproducibility_manifest.json":
            continue
        digest_lines.append(f"{_sha(path)}  {path.name}")
    reproducibility = {
        "artifact_sha256": digest_lines,
        "frozen_file_sha256": {name: _sha(path) for name, path in frozen_paths.items()},
        "p_freeze": P_FREEZE,
        "qwen_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "model_b_revision": selection["revision"],
        "m2_executed": False,
        "m3_executed": False,
    }
    _write(M1 / "reproducibility_manifest.json", reproducibility)
    REPORT.write_text(_report(selection, task, mapping, effects, matrix, decision), encoding="utf-8")


if __name__ == "__main__":
    main()
