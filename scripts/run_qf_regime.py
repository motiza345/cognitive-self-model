"""Characterize frozen regimes separately. This is not an MRSM rerun."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role  # noqa: E402
from src.mrsm.qf_regime import (  # noqa: E402
    FROZEN_REGIMES,
    decide_regimes,
    heterogeneity,
    regime_cell,
    slot_context_dependent,
)
from src.mrsm.qwen_bridge import is_context_cancellation  # noqa: E402
from src.cognitive_self_model.m22_1.outcome import resolve_outcome_tokens  # noqa: E402
from src.mrsm.run_q import _catalog, _load_model, _margins  # noqa: E402

OUT = ROOT / "artifacts" / "qf_regime"
REPORT = ROOT / "reports" / "QF_REGIME_SEPARATION.md"
P_FREEZE = "2cccafd047044332828eaf602b69f4267852cba2"
DECISION_SPLIT = "discovery"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _measure_role(model, records, catalog, positive_id: int, negative_id: int) -> dict:
    texts = [record.text for record in records]
    base = _margins(model, texts, [], positive_id, negative_id)
    measured = {}
    for item in catalog:
        treated = _margins(model, texts, [item], positive_id, negative_id)
        null_item = dict(item)
        null_item["alpha"] = 0.0
        null_treated = _margins(model, texts, [null_item], positive_id, negative_id)
        by_regime: dict[str, dict[str, list]] = {name: {"effects": [], "nulls": [], "prompts": []} for name in FROZEN_REGIMES}
        for record, base_margin, treated_margin, null_margin in zip(records, base, treated, null_treated):
            bucket = by_regime[record.regime_id]
            bucket["effects"].append(float(base_margin - treated_margin))
            bucket["nulls"].append(float(base_margin - null_margin))
            bucket["prompts"].append(record.prompt_id)
        cells = {}
        for name in FROZEN_REGIMES:
            cell = regime_cell(by_regime[name]["effects"], by_regime[name]["nulls"])
            cell["prompt_ids"] = by_regime[name]["prompts"]
            cell["effects"] = by_regime[name]["effects"]
            cell["nulls"] = by_regime[name]["nulls"]
            cells[name] = cell
        pooled_effects = [value for name in FROZEN_REGIMES for value in by_regime[name]["effects"]]
        pooled = mean(pooled_effects)
        means = {name: cells[name]["mean_effect"] for name in FROZEN_REGIMES}
        measured[item["name"]] = {
            "regimes": cells,
            "pooled_mean_descriptive_only": pooled,
            "context_cancellation": is_context_cancellation(means, pooled),
            "context_dependent": slot_context_dependent(cells, pooled),
            "heterogeneity": heterogeneity({name: by_regime[name]["effects"] for name in FROZEN_REGIMES}),
        }
    return measured


def _matrix(measured: dict) -> dict:
    rows = {}
    for name, slot in measured.items():
        rows[name] = {regime: slot["regimes"][regime]["sign_symbol"] for regime in FROZEN_REGIMES}
    return rows


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _report(frozen_def, effects, nulls, hetero, matrix, decision) -> str:
    primary = effects["splits"][DECISION_SPLIT]
    lines = [
        "# QF Regime Separation",
        "",
        "Diagnostic characterization. Not an MRSM result and not a mechanism identification.",
        "",
        "## 1. Executive Summary",
        "",
        f"- Regime decision: `{decision['regime_decision']}`.",
        f"- Previous diagnosis retained: `{decision['previous_diagnosis_retained']}`.",
        f"- Mechanism identity: `{decision['mechanism_identity']}`.",
        f"- Primary split: `{DECISION_SPLIT}`. Validation and replication are conditioning views only.",
        f"- Interpretation: {decision['primary_interpretation']}",
        "",
        "## 2. Frozen MRSM Status",
        "",
        "| Item | State |",
        "| --- | --- |",
        "| P | H1 PASS, H2 PASS, H3 PASS, H4 PASS, Leakage PASS |",
        "| Q | H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL, Leakage PASS |",
        "| DEC-010 | `REDEFINE_SCALE` |",
        f"| p_freeze | `{P_FREEZE}` |",
        "| Qwen revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |",
        "| MRSM rerun | not performed |",
        "",
        "## 3. Frozen Regime Definitions",
        "",
        "Regimes are the three ids in `src/cognitive_self_model/m22_1/prompts.py`.",
        "None were dropped, merged, renamed, relabeled, reweighted, or downsampled.",
        "Discovery, validation, and replication are splits, not regimes.",
        "",
        "| Regime | Discovery prompts | Validation prompts | Replication prompts |",
        "| --- | --- | --- | --- |",
    ]
    for name in FROZEN_REGIMES:
        row = frozen_def["prompts"][name]
        lines.append(
            f"| {name} | {', '.join(row['discovery'])} | {', '.join(row['validation'])} | {', '.join(row['replication'])} |"
        )
    lines.extend([
        "",
        "## 4. Experimental Contract",
        "",
        "- Metric: MRSM margin drop, unhooked minus alpha-1 intervention.",
        "- Candidates are the eight existing MRSM slots. A slot is not a mechanism.",
        "- Each regime has its own alpha-0 null. There is no shared null.",
        "- Separability reuses `effect_floor` on that regime's null sample.",
        "- Sign zero floor is the existing `1e-8`. Unstable reuses the existing `0.75` consistency bar.",
        "- The confidence interval reuses the frozen percentile bootstrap, 2000 draws, seed 22102.",
        "- Pooled means are descriptive. They are not the primary result.",
        "- Heterogeneity is labeled `EXPLORATORY` and is not a gate.",
        "- The decision uses the discovery split only.",
        "",
        "## 5. Regime-Specific Intervention Results",
        "",
        "Discovery is primary.",
        "",
        "| Slot | Regime | n | Mean | Median | Std | SE | Effect minus null | Sign | Consistency | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ])
    for slot, payload in primary.items():
        for regime in FROZEN_REGIMES:
            cell = payload["regimes"][regime]
            lines.append(
                f"| {slot} | {regime} | {cell['n']} | {_fmt(cell['mean_effect'])} | "
                f"{_fmt(cell['median_effect'])} | {_fmt(cell['std_effect'])} | {_fmt(cell['standard_error'])} | "
                f"{_fmt(cell['effect_minus_null'])} | {cell['sign_class']} | "
                f"{_fmt(cell['sign_consistency'])} | {cell['status']} |"
            )
    lines.extend([
        "",
        "## 6. Regime-Specific Null Results",
        "",
        "Alpha is 0 inside the same slot and the same regime. The null is not global.",
        "",
        "| Slot | Regime | Null mean | Null std | Effect floor |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for slot, payload in primary.items():
        for regime in FROZEN_REGIMES:
            cell = payload["regimes"][regime]
            lines.append(
                f"| {slot} | {regime} | {_fmt(cell['null_mean'])} | {_fmt(cell['null_std'])} | {_fmt(cell['effect_floor'])} |"
            )
    lines.extend([
        "",
        "## 7. Effect Consistency",
        "",
        "Symbols are the sign of the measured regime mean. `+` positive, `-` negative, `~` inside the existing zero floor.",
        "",
        "| Slot | completion | instruction | syntax |",
        "| --- | --- | --- | --- |",
    ])
    for slot, row in matrix["discovery_primary"].items():
        lines.append(f"| {slot} | {row['completion']} | {row['instruction']} | {row['syntax']} |")
    lines.extend([
        "",
        "## 8. Heterogeneity",
        "",
        "Exploratory only. These ratios did not choose the decision.",
        "",
        "| Slot | Between | Within | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ])
    for slot, payload in hetero["splits"]["discovery"].items():
        lines.append(
            f"| {slot} | {_fmt(payload['between_regime_variance'])} | "
            f"{_fmt(payload['within_regime_variance'])} | {_fmt(payload['heterogeneity_ratio'])} |"
        )
    lines.extend([
        "",
        "## 9. Context Cancellation",
        "",
        "Cancellation uses the existing QF-Bridge rule on discovery regime means.",
        "A true flag means the effect depends on regime. It does not mean multiple mechanisms.",
        "",
        "| Slot | Context cancellation |",
        "| --- | --- |",
    ])
    for slot, payload in primary.items():
        lines.append(f"| {slot} | {payload['context_cancellation']} |")
    lines.extend([
        "",
        "## 10. Pooled vs Regime-Specific Comparison",
        "",
        "| Representation | Role in this diagnostic |",
        "| --- | --- |",
        "| R0 pooled | Secondary descriptive mean within discovery. Not used to drop a regime. |",
        "| R1 regime-specific | Primary. One estimate per slot and regime on discovery. |",
        "| R2 regime-conditioned | Validation and replication are reported separately in the artifacts. They are not a Self-Model. |",
        "",
        "| Slot | R0 pooled discovery mean |",
        "| --- | ---: |",
    ])
    for slot, payload in primary.items():
        lines.append(f"| {slot} | {_fmt(payload['pooled_mean_descriptive_only'])} |")
    lines.extend([
        "",
        "## 11. Decision",
        "",
        f"`{decision['regime_decision']}`",
        "",
        decision["primary_interpretation"],
        "",
        "Signal existence and sign consistency were examined. Mechanism identity was not.",
        "",
        "## 12. What This Does Not Prove",
        "",
        "- It does not prove one mechanism.",
        "- It does not prove several independent mechanisms.",
        "- It does not repair Q.",
        "- It does not change `REDEFINE_SCALE`.",
        "- It does not select a best, worst, or representative regime.",
        "",
        "## 13. Implication for Intervention Bottleneck",
        "",
        "`INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED` stays in force.",
        f"This diagnostic refines it to `{decision['regime_decision']}`.",
        "The refinement is about how the effect varies by regime, not about mechanism identity.",
        "",
        "## 14. Single Next Intervention",
        "",
        decision["next_single_intervention"],
        "",
        "## 15. Reproducibility Manifest",
        "",
        "See `artifacts/qf_regime/manifest.sha256`.",
        "",
        "| Frozen file | sha256 |",
        "| --- | --- |",
    ])
    for name, digest in frozen_def["frozen_file_sha256"].items():
        lines.append(f"| {name} | `{digest}` |")
    lines.extend(["", f"Null artifact slots: {len(nulls['splits'][DECISION_SPLIT])}.", ""])
    return "\n".join(lines)


def main() -> None:
    frozen_paths = {
        "reports/MRSM_FINAL_RESULT.md": ROOT / "reports/MRSM_FINAL_RESULT.md",
        "reports/Q_FAILURE_ANALYSIS.md": ROOT / "reports/Q_FAILURE_ANALYSIS.md",
        "reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md": ROOT / "reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md",
        "configs/mrsm_prereg.yaml": ROOT / "configs/mrsm_prereg.yaml",
        "configs/m22_1_preflight.json": ROOT / "configs/m22_1_preflight.json",
    }
    prompts = {name: {} for name in FROZEN_REGIMES}
    for role, records in prompts_by_role(frozen_prompts()).items():
        for record in records:
            prompts.setdefault(record.regime_id, {}).setdefault(role, []).append(record.prompt_id)
    if set(prompts) != set(FROZEN_REGIMES):
        raise SystemExit("frozen regime set changed")
    frozen_def = {
        "source": "src/cognitive_self_model/m22_1/prompts.py",
        "regimes": list(FROZEN_REGIMES),
        "dropped": [],
        "merged": [],
        "renamed": [],
        "prompts": prompts,
        "frozen_file_sha256": {name: _sha(path) for name, path in frozen_paths.items()},
        "p_freeze": P_FREEZE,
        "qwen_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
    }
    model = _load_model()
    tokens = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    positive_id = int(tokens["positive_token_id"])
    negative_id = int(tokens["negative_token_id"])
    catalog = _catalog()
    grouped = prompts_by_role(frozen_prompts())
    splits = {}
    for role in ("discovery", "validation", "replication"):
        print(f"measuring {role}", flush=True)
        splits[role] = _measure_role(model, grouped[role], catalog, positive_id, negative_id)
    effects = {"metric": "margin_drop_unhooked_minus_intervened", "splits": splits, "primary_split": DECISION_SPLIT}
    nulls = {
        "control": "alpha_0_inside_the_same_slot_and_regime",
        "global_null": False,
        "splits": {
            role: {
                slot: {regime: {"null_mean": cell["null_mean"], "null_std": cell["null_std"], "nulls": cell["nulls"]}
                       for regime, cell in payload["regimes"].items()}
                for slot, payload in measured.items()
            }
            for role, measured in splits.items()
        },
    }
    hetero = {
        "label": "EXPLORATORY",
        "used_as_gate": False,
        "splits": {role: {slot: payload["heterogeneity"] for slot, payload in measured.items()} for role, measured in splits.items()},
    }
    matrix = {"note": "symbols are measured mean signs; not a mechanism matrix", "splits": {role: _matrix(measured) for role, measured in splits.items()}}
    matrix["discovery_primary"] = matrix["splits"]["discovery"]
    decision_slots = []
    for slot, payload in splits[DECISION_SPLIT].items():
        decision_slots.append({
            "slot": slot,
            "regimes": payload["regimes"],
            "context_dependent": payload["context_dependent"],
        })
    decision = decide_regimes(decision_slots)
    decision["primary_split"] = DECISION_SPLIT
    decision["effect_matrix"] = matrix["discovery_primary"]
    OUT.mkdir(parents=True, exist_ok=True)
    _write("frozen_regimes.json", frozen_def)
    _write("regime_effects.json", effects)
    _write("regime_nulls.json", nulls)
    _write("regime_heterogeneity.json", hetero)
    _write("effect_consistency_matrix.json", matrix)
    _write("regime_decision.json", decision)
    digest_lines = []
    for path in sorted(OUT.glob("*.json")):
        digest_lines.append(f"{_sha(path)}  {path.name}")
    (OUT / "manifest.sha256").write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
    REPORT.write_text(_report(frozen_def, effects, nulls, hetero, matrix, decision), encoding="utf-8")
    print(decision["regime_decision"])
    print(json.dumps(matrix["discovery_primary"], sort_keys=True))


if __name__ == "__main__":
    main()
