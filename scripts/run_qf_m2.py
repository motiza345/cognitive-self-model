"""QF-M2 regime comparison from stored M1 and QF-Regime effects. No new forwards."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.mrsm import HEADS  # noqa: E402
from src.mrsm.qf_cross_model import FROZEN_QWEN_MATRIX  # noqa: E402
from src.mrsm.qf_m2 import (  # noqa: E402
    PRIMARY_METRIC,
    cosine,
    cross_model_null,
    decide,
    level_a,
    level_b,
    level_c,
    matched_cosines,
    pearson,
    r2_standardized,
    sign_cell,
    slot_heterogeneity,
    _pairwise,
)
from src.mrsm.qf_regime import FROZEN_REGIMES  # noqa: E402

M1 = ROOT / "artifacts" / "qf_cross_model" / "m1"
M2 = ROOT / "artifacts" / "qf_cross_model" / "m2"
QWEN = ROOT / "artifacts" / "qf_regime" / "regime_effects.json"
REPORT = ROOT / "reports" / "QF_M2_CROSS_MODEL_REGIMES.md"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _load_cells(measured: dict) -> dict:
    cells = {}
    for slot in HEADS:
        cells[slot] = {}
        for regime in FROZEN_REGIMES:
            cell = measured[slot]["regimes"][regime]
            cells[slot][regime] = {
                "raw_effect": float(cell["mean_effect"]),
                "null_mean": float(cell["null_mean"]),
                "null_std": float(cell["null_std"]),
                "null_corrected_effect": float(cell["mean_effect"]) - float(cell["null_mean"]),
                "qf_regime_standardized_effect": cell.get("standardized_effect"),
                "r2": r2_standardized(cell["mean_effect"], cell["null_mean"], cell["null_std"]),
                "sign": cell["sign_symbol"],
                "effects": [float(value) for value in cell["effects"]],
            }
    return cells


def _vectors(cells: dict, field: str) -> dict[str, list[float]]:
    return {regime: [float(cells[slot][regime][field]) for slot in HEADS] for regime in FROZEN_REGIMES}


def _vector_table(title: str, qwen: dict, gpt2: dict) -> list[str]:
    lines = [f"### {title}", "", "| Slot | Qwen completion | Qwen instruction | Qwen syntax | GPT-2 completion | GPT-2 instruction | GPT-2 syntax |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for index, slot in enumerate(HEADS):
        lines.append(
            f"| {slot} | {_fmt(qwen['completion'][index])} | {_fmt(qwen['instruction'][index])} | {_fmt(qwen['syntax'][index])} | "
            f"{_fmt(gpt2['completion'][index])} | {_fmt(gpt2['instruction'][index])} | {_fmt(gpt2['syntax'][index])} |"
        )
    lines.append("")
    return lines


def _report(bundle: dict) -> str:
    decision = bundle["decision"]
    lines = [
        "# QF-M2 Cross-Model Regime Comparison",
        "",
        "QF-CROSS-MODEL-DIAGNOSTIC. Not an MRSM gate and not a mechanism claim.",
        "M3 was not executed.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Decision: `{decision['m2_decision']}`.",
        f"- Level A literal slots: `{bundle['levels']['A']}`.",
        f"- Level B regime profiles: `{bundle['levels']['B']}`.",
        f"- Level C regime relationships: `{bundle['levels']['C']}`.",
        f"- Interpretation: {decision['primary_interpretation']}",
        f"- M3: {decision['recommendation_for_m3']}",
        "",
        "## 2. Frozen Inputs",
        "",
        "- Qwen revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.",
        "- GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.",
        "- Functional counterpart layers `0, 4, 7, 11`, declared in M1 before measurement.",
        "- Qwen coordinate layers 15 and 23 remain `NOT_COMPARABLE`.",
        "- No new forward pass was run. Effects come from the stored QF-Regime and M1 artifacts.",
        "- MRSM was not rerun. The Self-Model was not modified.",
        "",
        "## 3. Qwen vs GPT-2 Comparability",
        "",
        "The comparison uses the eight functional-counterpart slots. Those slots are comparable.",
        "The coordinate reading of layers 15 and 23 is preserved as `NOT_COMPARABLE` and is not imputed.",
        "Slot names are residual-stream labels, not attention heads.",
        "",
        "## 4. Raw Effect Profiles",
        "",
        "R0 is the regime mean margin drop. The profile for a regime is the eight slot values in catalog order.",
        "",
    ]
    lines.extend(_vector_table("R0", bundle["r0"]["qwen"], bundle["r0"]["gpt2"]))
    lines.extend(["## 5. Null-Corrected Profiles", "", "R1 is `mean_effect - regime_null_mean`. This is the primary profile.", ""])
    lines.extend(_vector_table("R1", bundle["r1"]["qwen"], bundle["r1"]["gpt2"]))
    lines.extend([
        "## 6. Standardized Profiles",
        "",
        "The M2 formula is `(effect - null_mean) / null_std`.",
        f"Status: `{bundle['r2_status']}`.",
        "No substitute denominator was inserted. The existing QF-Regime standardized effect, which divides by the effect standard deviation, is stored in the artifact and is not a decision input.",
        "",
        "## 7. Regime-to-Regime Relationships",
        "",
        f"Primary metric, declared before the comparison: `{PRIMARY_METRIC}`.",
        "Pearson correlation is exploratory and was not used to choose the decision.",
        "",
        "| Pair | Qwen cosine | GPT-2 cosine | Qwen Pearson (exploratory) | GPT-2 Pearson (exploratory) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for pair, qwen_value in bundle["within_model"]["qwen_cosine"].items():
        lines.append(
            f"| {pair} | {_fmt(qwen_value)} | {_fmt(bundle['within_model']['gpt2_cosine'][pair])} | "
            f"{_fmt(bundle['within_model']['qwen_pearson'][pair])} | {_fmt(bundle['within_model']['gpt2_pearson'][pair])} |"
        )
    lines.extend([
        "",
        "## 8. Cross-Model Regime Comparison",
        "",
        "| Regime | R1 cosine | R1 Pearson (exploratory) |",
        "| --- | ---: | ---: |",
    ])
    for regime, value in bundle["cross_model"]["cosine"].items():
        lines.append(f"| {regime} | {_fmt(value)} | {_fmt(bundle['cross_model']['pearson'][regime])} |")
    lines.extend([
        "",
        "## 9. Sign Agreement",
        "",
        "Sign agreement describes intervention behavior. It is not mechanism identity.",
        "",
        f"- Agreement: `{bundle['signs']['sign_agreement']}`.",
        f"- Disagreement: `{bundle['signs']['sign_disagreement']}`.",
        f"- Near zero: `{bundle['signs']['near_zero']}`.",
        f"- Not comparable: `{bundle['signs']['not_comparable']}`.",
        "",
        "| Slot | completion | instruction | syntax |",
        "| --- | --- | --- | --- |",
    ])
    for slot, row in bundle["signs"]["functional_cells"].items():
        lines.append(f"| {slot} | {row['completion']} | {row['instruction']} | {row['syntax']} |")
    lines.extend([
        "",
        "## 10. Heterogeneity",
        "",
        "Same exploratory QF-Regime variance split. Not a gate and not a ranking.",
        "",
        "| Slot | Qwen between | Qwen within | Qwen ratio | GPT-2 between | GPT-2 within | GPT-2 ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for slot in HEADS:
        qwen = bundle["heterogeneity"]["qwen"][slot]
        gpt2 = bundle["heterogeneity"]["gpt2"][slot]
        lines.append(
            f"| {slot} | {_fmt(qwen['between_regime_variance'])} | {_fmt(qwen['within_regime_variance'])} | {_fmt(qwen['heterogeneity_ratio'])} | "
            f"{_fmt(gpt2['between_regime_variance'])} | {_fmt(gpt2['within_regime_variance'])} | {_fmt(gpt2['heterogeneity_ratio'])} |"
        )
    lines.extend([
        "",
        "## 11. Context Cancellation",
        "",
        "The two results stay separate. Neither model is ranked.",
        "",
        f"- `QWEN_CONTEXT_CANCELLATION`: slots {bundle['cancellation']['qwen']['slots']}.",
        f"- `GPT2_CONTEXT_CANCELLATION`: slots {bundle['cancellation']['gpt2']['slots']}.",
        "",
        "| Slot | Qwen | GPT-2 |",
        "| --- | --- | --- |",
    ])
    for slot in HEADS:
        lines.append(f"| {slot} | {bundle['cancellation']['qwen']['per_slot'][slot]} | {bundle['cancellation']['gpt2']['per_slot'][slot]} |")
    lines.extend([
        "",
        "## 12. Cross-Model Null",
        "",
        "GPT-2 regime labels are permuted exhaustively. There are six permutations because there are three regimes.",
        "The count was not tuned. No numeric significance cutoff was added.",
        "",
        f"- Identity mean cosine: `{_fmt(bundle['null']['identity_mean_cosine'])}`.",
        f"- Identity strictly best: `{bundle['null']['identity_strictly_best']}`.",
        "",
        "| GPT-2 order | Identity | Mean cosine |",
        "| --- | --- | ---: |",
    ])
    for row in bundle["null"]["rows"]:
        lines.append(f"| {', '.join(row['gpt2_regime_order'])} | {row['identity']} | {_fmt(row['mean_cosine'])} |")
    lines.extend([
        "",
        "## 13. Three-Level Consistency Analysis",
        "",
        f"- Level A, literal slot consistency: `{bundle['levels']['A']}`.",
        f"- Level B, regime-profile consistency: `{bundle['levels']['B']}`.",
        f"- Level C, higher-order context structure: `{bundle['levels']['C']}`.",
        "",
        "A asks whether the same slot has the same sign. B asks whether matched regime profiles align above the regime-label permutation null. C asks whether the three within-model regime relationships have the same shape.",
        "",
        f"- Cosine of the two within-model relationship triples: `{_fmt(bundle['relationship_cosine'])}`.",
        f"- Strongest Qwen pair: `{bundle['peaks']['qwen']}`.",
        f"- Strongest GPT-2 pair: `{bundle['peaks']['gpt2']}`.",
        "",
        "## 14. Decision",
        "",
        f"`{decision['m2_decision']}`",
        "",
        decision["primary_interpretation"],
        "",
        "## 15. What This Does Not Prove",
        "",
        "- It does not prove mechanism identity.",
        "- It does not prove that causal mechanisms are model-specific in general.",
        "- It does not rank Qwen and GPT-2.",
        "- It does not repair Q or change `REDEFINE_SCALE`.",
        "- It does not make layers 15 and 23 comparable.",
        "",
        "## 16. Implication for Q Failure Diagnosis",
        "",
        "`INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED` remains the Qwen diagnosis.",
        bundle["q_implication"],
        "",
        "## 17. Recommendation for M3",
        "",
        decision["recommendation_for_m3"],
        "",
        "## 18. Reproducibility Manifest",
        "",
        "See `artifacts/qf_cross_model/m2/manifest.sha256`.",
        "M1 artifacts were read and not overwritten.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    qwen_raw = json.loads(QWEN.read_text(encoding="utf-8"))["splits"]["discovery"]
    gpt2_raw = json.loads((M1 / "regime_effects.json").read_text(encoding="utf-8"))
    matrix = json.loads((M1 / "effect_matrix.json").read_text(encoding="utf-8"))
    if matrix["qwen_frozen_discovery"] != FROZEN_QWEN_MATRIX:
        raise SystemExit("STOP: stored Qwen matrix does not match the frozen matrix")
    qwen_cells = _load_cells(qwen_raw)
    gpt2_cells = _load_cells(gpt2_raw["functional_counterpart"])
    for slot in HEADS:
        for regime in FROZEN_REGIMES:
            if qwen_cells[slot][regime]["sign"] != FROZEN_QWEN_MATRIX[slot][regime]:
                raise SystemExit(f"STOP: Qwen sign drifted at {slot} {regime}")
            if gpt2_cells[slot][regime]["sign"] != matrix["model_b_functional_discovery"][slot][regime]:
                raise SystemExit(f"STOP: GPT-2 sign drifted at {slot} {regime}")
    r0 = {"qwen": _vectors(qwen_cells, "raw_effect"), "gpt2": _vectors(gpt2_cells, "raw_effect")}
    r1 = {"qwen": _vectors(qwen_cells, "null_corrected_effect"), "gpt2": _vectors(gpt2_cells, "null_corrected_effect")}
    r2_values = [
        qwen_cells[slot][regime]["r2"]["status"] == "DEFINED" or gpt2_cells[slot][regime]["r2"]["status"] == "DEFINED"
        for slot in HEADS for regime in FROZEN_REGIMES
    ]
    r2_status = "DEFINED" if any(r2_values) else "UNDEFINED_BECAUSE_NULL_STD_IS_ZERO"
    within = {
        "qwen_cosine": _pairwise(r1["qwen"]),
        "gpt2_cosine": _pairwise(r1["gpt2"]),
        "qwen_pearson": {f"{left}__{right}": pearson(r1["qwen"][left], r1["qwen"][right]) for left, right in (
            ("completion", "instruction"), ("completion", "syntax"), ("instruction", "syntax")
        )},
        "gpt2_pearson": {f"{left}__{right}": pearson(r1["gpt2"][left], r1["gpt2"][right]) for left, right in (
            ("completion", "instruction"), ("completion", "syntax"), ("instruction", "syntax")
        )},
        "pearson_role": "EXPLORATORY",
        "decision_input": "cosine",
    }
    cross = {
        "cosine": {regime: cosine(r1["qwen"][regime], r1["gpt2"][regime]) for regime in FROZEN_REGIMES},
        "pearson": {regime: pearson(r1["qwen"][regime], r1["gpt2"][regime]) for regime in FROZEN_REGIMES},
        "pearson_role": "EXPLORATORY",
    }
    null = cross_model_null(r1["qwen"], r1["gpt2"])
    functional_cells = {}
    sign_counts = {"sign_agreement": 0, "sign_disagreement": 0, "near_zero": 0, "not_comparable": 0}
    level_a_cells = []
    for slot in HEADS:
        functional_cells[slot] = {}
        for regime in FROZEN_REGIMES:
            label = sign_cell(qwen_cells[slot][regime]["sign"], gpt2_cells[slot][regime]["sign"], True)
            functional_cells[slot][regime] = label
            sign_counts[label] += 1
            level_a_cells.append(label)
    coordinate = matrix["model_b_coordinate_discovery"]
    for slot in HEADS:
        for regime in FROZEN_REGIMES:
            if coordinate[slot][regime] == "NOT_COMPARABLE":
                sign_counts["not_comparable"] += 1
    signs = {**sign_counts, "functional_cells": functional_cells, "coordinate_not_comparable_excluded_from_profiles": True}
    heterogeneity = {"qwen": {}, "gpt2": {}}
    cancellation = {"qwen": {"per_slot": {}, "slots": []}, "gpt2": {"per_slot": {}, "slots": []}}
    for slot in HEADS:
        for name, cells in (("qwen", qwen_cells), ("gpt2", gpt2_cells)):
            stats = slot_heterogeneity({regime: cells[slot][regime]["effects"] for regime in FROZEN_REGIMES})
            heterogeneity[name][slot] = stats
            cancellation[name]["per_slot"][slot] = bool(stats["context_cancellation"])
            if stats["context_cancellation"]:
                cancellation[name]["slots"].append(slot)
    relationship_cosine = cosine(
        [within["qwen_cosine"][key] for key in within["qwen_cosine"]],
        [within["gpt2_cosine"][key] for key in within["gpt2_cosine"]],
    )
    peaks = {
        "qwen": max(within["qwen_cosine"], key=within["qwen_cosine"].get),
        "gpt2": max(within["gpt2_cosine"], key=within["gpt2_cosine"].get),
    }
    levels = {
        "A": level_a(level_a_cells),
        "B": level_b(null, matched_cosines(r1["qwen"], r1["gpt2"], tuple(FROZEN_REGIMES))),
        "C": level_c(within["qwen_cosine"], within["gpt2_cosine"]),
    }
    decision = decide(levels["B"], levels["C"])
    implication = {
        "CROSS_MODEL_REGIME_PATTERN_SUPPORTED": (
            "The Qwen regime response has a counterpart on GPT-2 under this protocol. "
            "That does not explain the frozen Q failure and does not identify its mechanism."
        ),
        "PARTIAL_CROSS_MODEL_CONSISTENCY": (
            "Qwen context dependence remains a Qwen diagnosis. GPT-2 shows only a partial regime correspondence, "
            "so the Q failure is not given a shared regime representation."
        ),
        "CROSS_MODEL_REGIME_PATTERN_NOT_SUPPORTED": (
            "The Qwen regime pattern was not recovered on GPT-2. This does not say that causal mechanisms "
            "are model-specific in general, and it does not change the frozen Q result."
        ),
        "INCONCLUSIVE": (
            "M2 does not add a regime explanation to the frozen Q failure diagnosis."
        ),
    }
    bundle = {
        "r0": r0,
        "r1": r1,
        "r2_status": r2_status,
        "within_model": within,
        "cross_model": cross,
        "null": null,
        "signs": signs,
        "heterogeneity": heterogeneity,
        "cancellation": cancellation,
        "levels": levels,
        "relationship_cosine": relationship_cosine,
        "peaks": peaks,
        "decision": decision,
        "q_implication": implication[decision["m2_decision"]],
    }
    normalized = {
        "primary_representation": "R1",
        "r2_formula": "(effect - null_mean) / null_std",
        "r2_status": r2_status,
        "models": {
            "qwen": qwen_cells,
            "gpt2": gpt2_cells,
        },
        "coordinate_layers_15_and_23": "NOT_COMPARABLE",
    }
    profiles = {"slot_order": list(HEADS), "r0": r0, "r1": r1, "r2_status": r2_status}
    relationships = {"primary_metric": PRIMARY_METRIC, "within_model": within, "cross_model": cross}
    M2.mkdir(parents=True, exist_ok=True)
    _write(M2 / "normalized_effect_matrix.json", normalized)
    _write(M2 / "regime_profiles.json", profiles)
    _write(M2 / "regime_relationships.json", relationships)
    _write(M2 / "sign_agreement.json", signs)
    _write(M2 / "heterogeneity_comparison.json", {"label": "EXPLORATORY", "used_as_gate": False, "models": heterogeneity})
    _write(M2 / "context_cancellation.json", {
        "QWEN_CONTEXT_CANCELLATION": cancellation["qwen"],
        "GPT2_CONTEXT_CANCELLATION": cancellation["gpt2"],
        "collapsed_into_one_result": False,
    })
    _write(M2 / "cross_model_null.json", null)
    _write(M2 / "decision.json", {
        **decision,
        "levels": levels,
        "regime_profile_result": {
            "matched_r1_cosine": cross["cosine"],
            "identity_mean_cosine": null["identity_mean_cosine"],
            "identity_strictly_best": null["identity_strictly_best"],
        },
    })
    manifest_lines = [f"{_sha(path)}  {path.name}" for path in sorted(M2.glob("*.json"))]
    (M2 / "manifest.sha256").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    REPORT.write_text(_report(bundle), encoding="utf-8")
    print(decision["m2_decision"])
    print(json.dumps({"levels": levels, "cosine": cross["cosine"], "null_best": null["identity_strictly_best"], "identity_mean": null["identity_mean_cosine"]}, indent=2))


if __name__ == "__main__":
    main()
