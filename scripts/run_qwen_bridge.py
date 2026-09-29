"""Reproduce the selected historical Qwen intervention on the pinned revision.

Writes only artifacts/qwen_bridge and reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md.
Does not call load_qwen and does not write reports/M22_1.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cognitive_self_model.m22_1.config import hook_name, load_config
from src.cognitive_self_model.m22_1.direction import orthogonal_direction, primary_direction, vector_sha256
from src.cognitive_self_model.m22_1.leakage import audit_package_imports
from src.cognitive_self_model.m22_1.loader import LoadedModel, direction_tensor, forward_record
from src.cognitive_self_model.m22_1.outcome import resolve_outcome_tokens
from src.cognitive_self_model.m22_1.prompts import assert_split_integrity, frozen_prompts, prompt_manifest_sha256
from src.cognitive_self_model.m22_1.protocol import execute_protocol
from src.cognitive_self_model.m22_1_r.strict_loader import load_pinned_revision
from src.mrsm.p_data import sha256_file
from src.mrsm.qwen_bridge import (
    FLOOR,
    MATCH_TOLERANCE,
    PINNED_REVISION,
    RECORDED_DIRECTION_SHA,
    RECORDED_LAYER,
    SIGN_FLOOR,
    classify_identity,
    decide,
    file_sha256,
    is_context_cancellation,
    select_bridge,
)

FROZEN_FILES = (
    "configs/mrsm_prereg.yaml",
    "reports/MRSM_FINAL_RESULT.md",
    "reports/Q_FAILURE_ANALYSIS.md",
    "control_plane/DEC-010_decision_table.md",
    "artifacts/mrsm/holdout_manifest.json",
    "artifacts/mrsm/q_run_001/result.json",
    "artifacts/q_failure/diagnosis.json",
    "src/mrsm/run_q.py",
    "src/mrsm/self_model.py",
    "src/mrsm/qf_localize.py",
)

RECORDED_MEANS = {
    "discovery_mean": 0.029387950897216797,
    "validation_mean": 0.028637091318766277,
    "replication_mean": 0.029693762461344402,
    "validation_control_mean": 0.10541534423828125,
    "replication_control_mean": 0.10765441258748372,
}


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _load_json(rel: str):
    return json.loads((_ROOT / rel).read_text(encoding="utf-8"))


def _inventory() -> list[dict]:
    certificate = _load_json("reports/M22_1/certificate.json")
    manifest = _load_json("reports/M22_1/manifest.json")
    panel_env = _load_json("reports/M22_1_1/direction_panel_results.json")["environment"]
    panel_label = _load_json("reports/M22_1_1/intervention_space_report.json")["classification"]
    response = _load_json("reports/m22_1_2_response_space_results.json")
    replay = _load_json("artifacts/m22_1_r/execution_record.json")
    evidence = certificate["evidence"]
    outcome = manifest["outcome"]
    common_unknown_tokenizer = "UNKNOWN"
    records = [
        {
            "experiment_id": "M22.1",
            "commit": certificate.get("git_commit", "UNKNOWN"),
            "commit_source": "reports/M22_1/certificate.json git_commit",
            "model_name": manifest.get("model_id", "UNKNOWN"),
            "model_revision": manifest.get("model_revision", "UNKNOWN"),
            "dtype": manifest.get("dtype", "UNKNOWN"),
            "device": manifest.get("device", "UNKNOWN"),
            "tokenizer": common_unknown_tokenizer,
            "outcome_token_ids": outcome.get("token_ids", "UNKNOWN"),
            "prompt_task": "frozen M22.1 prompts; logit margin of ' yes' minus ' no'",
            "input_distribution": "18 prompts, roles discovery/validation/replication, regimes completion/instruction/syntax",
            "layer_head": "candidate layers 0, 8, 15, 23; selected layer 23; not an attention head",
            "intervention_type": "additive last-token residual, blocks.{layer}.hook_resid_post",
            "intervention_strength": manifest.get("primary_alpha", "UNKNOWN"),
            "magnitude_grid": manifest.get("magnitude_grid", "UNKNOWN"),
            "target_metric": "actual_delta = intervened logit margin minus unhooked margin",
            "baseline": "unhooked forward",
            "effect_size": evidence.get("discovery_mean", "UNKNOWN"),
            "validation_effect": evidence.get("validation_mean", "UNKNOWN"),
            "replication_effect": evidence.get("replication_mean", "UNKNOWN"),
            "control": "alpha 0 null and orthogonal direction at the selected layer",
            "reported_result": certificate.get("status", "UNKNOWN"),
            "artifact_path": "reports/M22_1/certificate.json",
            "direct_causal_intervention": True,
            "negative_control_present": True,
            "reproducible_code": True,
            "code_path": "src/cognitive_self_model/m22_1/",
            "external_dependency_count": 0,
        },
        {
            "experiment_id": "M22.1.1",
            "commit": "UNKNOWN",
            "commit_source": "no git_commit field in reports/M22_1_1/intervention_space_report.json",
            "path_last_commit": _git(["log", "-1", "--format=%H", "--", "reports/M22_1_1/README.md"]),
            "model_name": panel_env.get("model_id", "UNKNOWN"),
            "model_revision": panel_env.get("model_revision", "UNKNOWN"),
            "dtype": panel_env.get("dtype", "UNKNOWN"),
            "device": panel_env.get("device", "UNKNOWN"),
            "tokenizer": common_unknown_tokenizer,
            "outcome_token_ids": "UNKNOWN",
            "prompt_task": "same frozen M22.1 prompts; README states logit(9834)-logit(902)",
            "input_distribution": "same 18-prompt split",
            "layer_head": "blocks.23.hook_resid_post only; directions D1-D8",
            "intervention_type": "additive last-token residual",
            "intervention_strength": 1.0,
            "magnitude_grid": [-2, -1, 0, 1, 2],
            "target_metric": "paired delta",
            "baseline": "unhooked margin",
            "effect_size": "README discovery D1 mean at +1 is +0.029388",
            "control": "D2 is the M22.1 orthogonal control; alpha 0 is on the grid",
            "reported_result": panel_label,
            "artifact_path": "reports/M22_1_1/intervention_space_report.json",
            "direct_causal_intervention": True,
            "negative_control_present": True,
            "reproducible_code": True,
            "code_path": "scripts/run_m22_1_1_audit.py",
            "external_dependency_count": 1,
        },
        {
            "experiment_id": "M22.1.2",
            "commit": "UNKNOWN",
            "commit_source": "no git_commit field in reports/m22_1_2_response_space_results.json",
            "model_name": "UNKNOWN",
            "model_revision": "UNKNOWN",
            "dtype": "UNKNOWN",
            "device": "UNKNOWN",
            "tokenizer": common_unknown_tokenizer,
            "prompt_task": "reuses M22.1.1 measurements; no new model forward",
            "input_distribution": "validation and replication matrices from M22.1.1",
            "layer_head": "UNKNOWN",
            "intervention_type": "none; response-space reanalysis",
            "intervention_strength": "UNKNOWN",
            "target_metric": "blocked-cell MAE",
            "baseline": "B0 through B7 in reports/m22_1_2_response_space_report.md",
            "effect_size": "UNKNOWN",
            "control": "B7 cell permutation is reported",
            "reported_result": response["label"]["label"],
            "artifact_path": "reports/m22_1_2_response_space_results.json",
            "direct_causal_intervention": False,
            "negative_control_present": True,
            "reproducible_code": True,
            "code_path": "scripts/run_m22_1_2_audit.py",
            "external_dependency_count": 2,
        },
        {
            "experiment_id": "M22.1.3",
            "commit": "UNKNOWN",
            "commit_source": "report does not contain a run commit field",
            "model_name": "Qwen/Qwen2.5-0.5B",
            "model_revision": PINNED_REVISION,
            "dtype": "UNKNOWN",
            "device": "UNKNOWN",
            "tokenizer": common_unknown_tokenizer,
            "prompt_task": "readout reconstruction of recorded residual effects",
            "input_distribution": "discovery, validation, replication",
            "layer_head": "final residual through RMSNorm and unembedding; site inherited from M22.1",
            "intervention_type": "reconstruction, not a new intervention",
            "intervention_strength": 1.0,
            "target_metric": "reconstructed delta versus recorded delta",
            "baseline": "recorded M22.1.1 deltas",
            "effect_size": "report states validation D1 reconstructed delta +0.028637",
            "control": "alpha grid inherited; not a new null experiment",
            "reported_result": "READOUT_RECONSTRUCTION_EXACT",
            "artifact_path": "reports/m22_1_3_readout_null_report.md",
            "direct_causal_intervention": False,
            "negative_control_present": False,
            "reproducible_code": True,
            "code_path": "scripts/run_m22_1_3_audit.py",
            "external_dependency_count": 2,
        },
        {
            "experiment_id": "M22.1.3b",
            "commit": "UNKNOWN",
            "model_name": "UNKNOWN",
            "model_revision": "UNKNOWN",
            "dtype": "UNKNOWN",
            "device": "UNKNOWN",
            "tokenizer": common_unknown_tokenizer,
            "prompt_task": "reinterpretation of stored measurements",
            "input_distribution": "UNKNOWN",
            "layer_head": "UNKNOWN",
            "intervention_type": "none",
            "intervention_strength": "UNKNOWN",
            "target_metric": "energy and cosine summaries",
            "baseline": "UNKNOWN",
            "effect_size": "UNKNOWN",
            "control": "UNKNOWN",
            "reported_result": "REINTERPRETATION_DESCRIPTIVELY_SUPPORTED",
            "artifact_path": "reports/M22_1_3b/README.md",
            "direct_causal_intervention": False,
            "negative_control_present": False,
            "reproducible_code": True,
            "code_path": "scripts/run_m22_1_3b_audit.py",
            "external_dependency_count": 3,
        },
        {
            "experiment_id": "M22.1-R",
            "commit": replay.get("git_commit", "UNKNOWN"),
            "model_name": "Qwen/Qwen2.5-0.5B",
            "model_revision": replay.get("model_revision", "UNKNOWN"),
            "dtype": "UNKNOWN",
            "device": "UNKNOWN",
            "tokenizer": common_unknown_tokenizer,
            "prompt_task": "replay harness; no forward in the recorded attempt",
            "input_distribution": "UNKNOWN",
            "layer_head": "UNKNOWN",
            "intervention_type": "not executed",
            "intervention_strength": "UNKNOWN",
            "target_metric": "UNKNOWN",
            "baseline": "UNKNOWN",
            "effect_size": "UNKNOWN",
            "control": "UNKNOWN",
            "reported_result": replay.get("replay_tier", "UNKNOWN"),
            "artifact_path": "artifacts/m22_1_r/execution_record.json",
            "direct_causal_intervention": False,
            "negative_control_present": False,
            "reproducible_code": True,
            "code_path": "scripts/run_m22_1_r_harness.py",
            "external_dependency_count": 1,
        },
    ]
    return records


def _identity(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        same_revision = record.get("model_revision") == PINNED_REVISION
        if record["experiment_id"] == "M22.1":
            checks = {
                "model_family": "yes",
                "model_revision": "yes" if same_revision else "no",
                "tokenizer": "unknown",
                "dtype": "yes" if record.get("dtype") == "float32" else "no",
                "activation_extraction": "yes",
                "indexing": "no",
                "prompt_distribution": "yes",
                "intervention_location": "no",
                "intervention_type": "yes",
                "intervention_magnitude": "yes",
            }
        elif record["experiment_id"] == "M22.1.1":
            checks = {
                "model_family": "yes",
                "model_revision": "yes" if same_revision else "no",
                "tokenizer": "unknown",
                "dtype": "yes" if record.get("dtype") == "float32" else "no",
                "activation_extraction": "yes",
                "indexing": "no",
                "prompt_distribution": "yes",
                "intervention_location": "no",
                "intervention_type": "yes",
                "intervention_magnitude": "yes",
            }
        elif record["experiment_id"] in {"M22.1.2", "M22.1.3", "M22.1.3b", "M22.1-R"}:
            checks = {
                "model_family": "yes" if record["experiment_id"] != "M22.1.2" else "unknown",
                "model_revision": "yes" if same_revision else ("unknown" if record.get("model_revision") in (None, "UNKNOWN") else "no"),
                "tokenizer": "unknown",
                "dtype": "unknown" if record.get("dtype") in (None, "UNKNOWN") else ("yes" if record.get("dtype") == "float32" else "no"),
                "activation_extraction": "unknown",
                "indexing": "unknown",
                "prompt_distribution": "unknown",
                "intervention_location": "unknown",
                "intervention_type": "no" if record["experiment_id"] != "M22.1-R" else "unknown",
                "intervention_magnitude": "unknown",
            }
        else:
            checks = {"model_revision": "unknown", "model_family": "unknown"}
        out.append({
            "experiment_id": record["experiment_id"],
            "checks": checks,
            "category": classify_identity(checks),
            "notes": _identity_note(record["experiment_id"]),
        })
    return out


def _identity_note(experiment_id: str) -> str:
    notes = {
        "M22.1": (
            "Same revision, dtype, hook, prompts, alpha 1.0, and outcome texts. "
            "Tokenizer name is not stored. Indexing differs because M22.1 selects one layer "
            "and MRSM names eight residual slots. The historical metric is intervened-minus-baseline; "
            "MRSM stores baseline-minus-intervened."
        ),
        "M22.1.1": "Same revision and layer-23 hook, but eight directions at one layer rather than the MRSM eight-slot map.",
        "M22.1.2": "No new forward. Model revision is not restated in the results JSON.",
        "M22.1.3": "Reconstruction of stored effects, not the MRSM eight-slot protocol.",
        "M22.1.3b": "Stored-number reinterpretation. Revision is not restated in the README front matter.",
        "M22.1-R": "The recorded attempt did not run a forward. The pin matches, and the snapshot was absent at that time.",
    }
    return notes.get(experiment_id, "")


def _sign(value: float) -> int:
    if abs(value) <= SIGN_FLOOR:
        return 0
    return 1 if value > 0 else -1


def _summary(rows: list[dict], value_key: str = "actual_delta") -> dict:
    values = [float(row[value_key]) for row in rows]
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array))
    sign = _sign(mean)
    active = [value for value in values if abs(value) > SIGN_FLOOR]
    consistency = float(np.mean([_sign(value) == sign for value in active])) if active and sign else 0.0
    regimes: dict[str, list[float]] = {}
    for row, value in zip(rows, values):
        regimes.setdefault(str(row["regime_id"]), []).append(value)
    regime_stats = {}
    for regime, items in sorted(regimes.items()):
        arr = np.asarray(items, dtype=np.float64)
        regime_stats[regime] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "sign": _sign(float(np.mean(arr))),
            "n": int(arr.size),
        }
    regime_signs = {item["sign"] for item in regime_stats.values() if item["sign"] != 0 and abs(item["mean"]) > FLOOR}
    return {
        "n": int(array.size),
        "raw_effect": mean,
        "std": std,
        "variance": float(np.var(array)),
        "median": float(np.median(array)),
        "sign": sign,
        "sign_consistency": consistency,
        "snr": None if std <= SIGN_FLOOR else abs(mean) / std,
        "standardized_effect": None if std <= SIGN_FLOOR else mean / std,
        "regime": regime_stats,
        "regime_sign_agreement": len(regime_signs) <= 1,
    }


def _mrsm_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        copied = dict(row)
        copied["actual_delta"] = -float(row["actual_delta"])
        converted.append(copied)
    return converted


def _match(evidence: dict, selected_layer: int, direction_sha: str, prompt_sha: str, status: str) -> tuple[bool, str | None]:
    reasons = []
    if direction_sha != RECORDED_DIRECTION_SHA:
        reasons.append("INTERVENTION")
    if prompt_sha != "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db":
        reasons.append("DATA")
    if int(selected_layer) != RECORDED_LAYER:
        reasons.append("INTERVENTION")
    for key, recorded in RECORDED_MEANS.items():
        if abs(float(evidence[key]) - recorded) > MATCH_TOLERANCE:
            reasons.append("IMPLEMENTATION")
            break
    if status != "CANDIDATE":
        reasons.append("METRIC")
    unique = list(dict.fromkeys(reasons))
    reproduced = not unique
    if reproduced:
        return True, None
    if len(unique) == 1:
        return False, unique[0]
    return False, "UNKNOWN"


def _strength(block: dict | None) -> list[dict]:
    if block is None:
        return []
    grouped: dict[float, list[dict]] = {}
    for row in block["rows"]:
        grouped.setdefault(float(row["alpha"]), []).append(row)
    null_rows = grouped.get(0.0, [])
    null_summary = _summary(null_rows) if null_rows else {"raw_effect": None, "std": None}
    out = []
    for alpha in sorted(grouped):
        summary = _summary(grouped[alpha])
        out.append({
            "alpha": alpha,
            "effect": summary["raw_effect"],
            "null": null_summary.get("raw_effect"),
            "effect_minus_null": None if null_summary.get("raw_effect") is None else summary["raw_effect"] - null_summary["raw_effect"],
            "snr": summary["snr"],
            "sign": summary["sign"],
            "sign_consistency": summary["sign_consistency"],
            "context_consistency": summary["regime_sign_agreement"],
            "variance": summary["variance"],
            "n": summary["n"],
        })
    return out


def _qf2_stratification() -> dict:
    payload = _load_json("artifacts/q_failure/qf2_intervention_signal.json")
    rows = []
    cancelling = []
    for row in payload["rows"]:
        regimes = {}
        for regime, mean in row["regime_means"].items():
            regimes[regime] = {
                "mean": mean,
                "std": "UNKNOWN",
                "median": "UNKNOWN",
                "sign": _sign(float(mean)),
                "n": 2,
                "n_source": "frozen discovery split has two prompts in each regime",
                "null_mean": row["null_distribution"]["mean"],
                "null_std": row["null_distribution"]["std"],
            }
        cancellation = is_context_cancellation(row["regime_means"], row["raw_effect"])
        if cancellation:
            cancelling.append(row["intervention"])
        pooled_sign = _sign(float(row["raw_effect"]))
        signed = [item["sign"] for item in regimes.values() if item["sign"] != 0]
        consistency = float(np.mean([sign == pooled_sign for sign in signed])) if signed else None
        rows.append({
            "slot": row["intervention"],
            "not_a_discovered_mechanism": True,
            "intervention_strength": 1.0,
            "pooled_mean": row["raw_effect"],
            "pooled_std": row["std"],
            "regimes": regimes,
            "sign_consistency_across_regimes": consistency,
            "context_cancellation": cancellation,
        })
    return {
        "source": "artifacts/q_failure/qf2_intervention_signal.json",
        "within_regime_std": "UNKNOWN",
        "within_regime_median": "UNKNOWN",
        "rows": rows,
        "cancelling_slots": cancelling,
        "pooled_versus_stratified": (
            "Stratification reveals opposing regime signs whose absolute values exceed the pooled mean."
            if cancelling else
            "No slot met the precommitted cancellation rule."
        ),
    }


def _load_bundle(config: dict) -> LoadedModel:
    from transformer_lens import HookedTransformer

    model = load_pinned_revision(
        "Qwen/Qwen2.5-0.5B",
        PINNED_REVISION,
        from_pretrained=HookedTransformer.from_pretrained,
    )
    model.eval()
    if model.tokenizer is not None:
        model.tokenizer.padding_side = "left"
        if model.tokenizer.pad_token_id is None and model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("loaded shape does not match the pinned Qwen config")
    from importlib.metadata import version

    return LoadedModel(
        model=model,
        model_id="Qwen/Qwen2.5-0.5B",
        model_revision=PINNED_REVISION,
        revision_pinned=True,
        revision_source="local snapshot pin",
        revision_error=None,
        load_used_revision=True,
        device="cpu",
        dtype="float32",
        versions={
            "torch": version("torch"),
            "transformer_lens": version("transformer-lens"),
            "transformers": version("transformers"),
            "numpy": version("numpy"),
        },
        n_layers=24,
        d_model=896,
        config_sha256="not-used",
    )


def _dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "null"
    return str(value)


def _report(payload: dict) -> str:
    decision = payload["decision"]
    lines = [
        "# Qwen bridge intervention analysis",
        "",
        "Label: `QF-BRIDGE-DIAGNOSTIC`",
        "",
        "This file does not replace `reports/MRSM_FINAL_RESULT.md` or `reports/Q_FAILURE_ANALYSIS.md`.",
        "",
        "## 1. Executive Summary",
        "",
        f"Selected bridge experiment: `{payload['selection']['experiment_id']}`.",
        f"Bridge result: `{decision['bridge_result']}`.",
        f"Primary diagnosis: `{decision['primary_diagnosis']}`.",
        "",
        decision["next_intervention"],
        "",
        "## 2. Frozen MRSM Context",
        "",
        "P H1–H4 remain PASS. Q remains H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL.",
        "Leakage remains PASS on both arms. DEC-010 remains `REDEFINE_SCALE`.",
        "`p_freeze` remains `2cccafd047044332828eaf602b69f4267852cba2`.",
        f"Qwen revision remains `{PINNED_REVISION}`.",
        "No MRSM gate was rescored.",
        "",
        "## 3. Historical Qwen Experiment Inventory",
        "",
        "M21 configs in this repository do not name Qwen. They are not in this inventory.",
        "",
    ]
    for record in payload["inventory"]:
        lines.append(f"### {record['experiment_id']}")
        lines.append("")
        for key in (
            "commit", "model_name", "model_revision", "dtype", "device", "tokenizer",
            "prompt_task", "input_distribution", "layer_head", "intervention_type",
            "intervention_strength", "target_metric", "baseline", "effect_size",
            "control", "reported_result", "artifact_path",
        ):
            lines.append(f"- {key}: `{record.get(key, 'UNKNOWN')}`")
        lines.append("")
    lines.extend(["## 4. Model / Revision Identity Audit", ""])
    for row in payload["identity"]:
        lines.append(f"- `{row['experiment_id']}`: `{row['category']}`. {row['notes']}")
    lines.extend([
        "",
        "No ranking is assigned.",
        "",
        "## 5. Selected Bridge Experiment",
        "",
        f"`{payload['selection']['experiment_id']}`",
        "",
        "Rule, in order: " + "; ".join(payload["selection"]["rule"]) + ".",
        f"Eligible ids: `{payload['selection']['eligible_ids']}`.",
        "M22.1 is the direct causal intervention with a null, an orthogonal control, a reported effect, and no dependence on a later audit.",
        "Its recorded status is `CANDIDATE`, not `VALIDATED_FOR_M22`. This bridge does not promote that status.",
        "",
        "## 6. Historical Setup Reproduction",
        "",
        "The historical protocol ran through `execute_protocol` on the local pinned snapshot.",
        "`load_qwen` was not used, so there was no unpinned retry.",
        f"Reproduced status: `{payload['historical']['status']}`.",
        f"Selected layer: `{payload['historical']['selected_layer']}`.",
        f"Match to the recorded certificate: `{payload['historical']['reproduced']}`.",
    ])
    for key, recorded in RECORDED_MEANS.items():
        lines.append(f"- {key}: reproduced `{_fmt(payload['historical']['evidence'][key])}`, recorded `{_fmt(recorded)}`")
    lines.extend([
        "",
        "## 7. MRSM-Compatible Reproduction",
        "",
        "The same forwards are reported as MRSM margin drop, which is the negation of `actual_delta`.",
        "No self-model was fit and no H gate was scored.",
        "",
    ])
    for split, block in payload["mrsm"].items():
        lines.append(f"- {split} drop mean `{_fmt(block['raw_effect'])}`, sign consistency `{_fmt(block['sign_consistency'])}`, SNR `{_fmt(block['snr'])}`")
    lines.extend([
        "",
        "## 8. Intervention Signal Comparison",
        "",
        "Historical metric is preserved. The MRSM metric is additional and does not replace it.",
        "",
        "| split | historical delta | MRSM drop | null delta | control delta |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in payload["comparison_rows"]:
        lines.append(
            f"| {row['split']} | {_fmt(row['historical_effect'])} | {_fmt(row['mrsm_effect'])} | {_fmt(row['null_effect'])} | {_fmt(row['control_effect'])} |"
        )
    lines.extend(["", "## 9. Context Stratification", ""])
    lines.append("QF-2 regime means are reused. Within-regime standard deviation and median were not stored, so they are `UNKNOWN`.")
    lines.append("")
    lines.append("| slot | completion | instruction | syntax | pooled | cancellation |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in payload["stratification"]["rows"]:
        regimes = row["regimes"]
        lines.append(
            "| {slot} | {c} | {i} | {s} | {p} | {flag} |".format(
                slot=row["slot"],
                c=_fmt(regimes["completion"]["mean"]),
                i=_fmt(regimes["instruction"]["mean"]),
                s=_fmt(regimes["syntax"]["mean"]),
                p=_fmt(row["pooled_mean"]),
                flag=row["context_cancellation"],
            )
        )
    lines.extend([
        "",
        "These slots are intervention names. They are not discovered mechanisms.",
        "",
        "## 10. Pooled vs Stratified Analysis",
        "",
        payload["stratification"]["pooled_versus_stratified"],
        f"Cancelling slots: `{payload['stratification']['cancelling_slots']}`.",
        "",
        "## 11. Simpson/Cancellation Audit",
        "",
        "A slot is `CONTEXT_CANCELLATION` when two regimes have opposite signs, each absolute regime mean clears the existing 1e-4 floor, and the absolute pooled mean is smaller than both.",
        "No regime was removed.",
        "",
        "## 12. Intervention Strength Analysis",
        "",
        "Strength values are the recorded M22.1 magnitude grid. The config does not define weak, medium, or strong, so those names are not assigned.",
        "",
    ])
    for split, rows in payload["strength"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| alpha | effect | null | SNR | sign | context consistency |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {_fmt(row['alpha'])} | {_fmt(row['effect'])} | {_fmt(row['null'])} | {_fmt(row['snr'])} | {_fmt(row['sign'])} | {row['context_consistency']} |"
            )
        lines.append("")
    lines.extend([
        "## 13. Negative Controls",
        "",
        "- Intact: unhooked logit margin.",
        "- Null: alpha 0 on the historical grid. Recorded max absolute null delta is 0.",
        f"- Orthogonal control at alpha 1, validation `{_fmt(payload['historical']['evidence']['validation_control_mean'])}`, replication `{_fmt(payload['historical']['evidence']['replication_control_mean'])}`.",
        "- Shuffled or permuted intervention: `NOT_AVAILABLE`. The M22.1 protocol does not define one, and none was added.",
        "",
        "## 14. Interpretation",
        "",
        payload["interpretation"],
        "",
        "## 15. Primary Diagnosis",
        "",
        decision["primary_diagnosis"],
        "",
        f"Bridge result: `{decision['bridge_result']}`.",
        "",
        "The previous QF label `INTERVENTION_BOTTLENECK` is refined by this result. It is not deleted from `reports/Q_FAILURE_ANALYSIS.md`.",
        "",
        "## 16. What This Does NOT Prove",
        "",
        "This diagnostic does not discover a mechanism and does not name a discovered mechanism.",
        "It does not change MRSM H1–H4.",
        "Q.H3 is not used as causal evidence.",
        "`CANDIDATE` is not promoted to `VALIDATED_FOR_M22`.",
        "A context cancellation is not a failed mechanism.",
        "",
        "## 17. Recommended Next Single Intervention",
        "",
        decision["next_intervention"],
        "",
        "No architectural change was made.",
        "",
        "## 18. Reproducibility Manifest",
        "",
    ])
    for key, value in payload["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "Scientific status: `QF-BRIDGE-DIAGNOSTIC`.", ""])
    return "\n".join(lines)


def main() -> int:
    before = {rel: sha256_file(_ROOT / rel) for rel in FROZEN_FILES}
    records = _inventory()
    identity = _identity(records)
    selection = select_bridge(records)
    config = load_config()
    prompts = frozen_prompts()
    assert_split_integrity(prompts)
    prompt_sha = prompt_manifest_sha256(prompts)
    leakage = audit_package_imports()
    bundle = _load_bundle(config)
    if bundle.model_revision != PINNED_REVISION:
        raise RuntimeError("revision mismatch")
    primary = primary_direction(bundle.d_model, int(config["direction_seed"]))
    control = orthogonal_direction(bundle.d_model, int(config["control_direction_seed"]), primary)
    direction_sha = vector_sha256(primary)
    outcome = resolve_outcome_tokens(bundle.model.tokenizer, " yes", " no")
    if int(outcome["positive_token_id"]) != 9834 or int(outcome["negative_token_id"]) != 902:
        tokenizer_match = "no"
    else:
        tokenizer_match = "yes_token_ids_only"
    tensors = {
        "primary": direction_tensor(primary, bundle.device),
        "control": direction_tensor(control, bundle.device),
    }
    progress = {"n": 0}

    def measure(prompt, *, hook, layer, alpha, direction_id):
        progress["n"] += 1
        if progress["n"] == 1 or progress["n"] % 25 == 0:
            print(f"forward {progress['n']}", flush=True)
        vector = None if direction_id in {None, "none"} else tensors[direction_id]
        name = None if layer is None else hook_name(int(layer), config)
        return forward_record(
            bundle,
            text=prompt.text,
            prepend_bos=bool(config["prepend_bos"]),
            hook=hook,
            layer=layer,
            alpha=float(alpha),
            direction=vector,
            positive_id=int(outcome["positive_token_id"]),
            negative_id=int(outcome["negative_token_id"]),
            expected_hook_name=name,
        )

    protocol = execute_protocol(
        prompts,
        n_layers=bundle.n_layers,
        measure=measure,
        config=config,
        revision_pinned=True,
        definition_reproducible=vector_sha256(primary_direction(bundle.d_model, int(config["direction_seed"]))) == direction_sha,
        leakage_pass=bool(leakage["pass"]),
        on_freeze=None,
    )
    evidence = protocol["evidence"]
    reproduced, locus = _match(evidence, int(protocol["selected_layer"]), direction_sha, prompt_sha, protocol["status"])
    selected = int(protocol["selected_layer"])
    historical_splits = {}
    mrsm_splits = {}
    comparison = []
    discovery_null = [row for row in protocol["null"]["rows"] if int(row["layer"]) == selected]
    for split, rows, control_rows, null_rows in (
        ("discovery", protocol["detectability"]["discovery"]["per_prompt"], [], discovery_null),
        (
            "validation",
            [row for row in protocol["validation"]["rows"] if float(row["alpha"]) == 1.0],
            protocol["validation"]["control_rows"],
            [row for row in protocol["validation"]["rows"] if float(row["alpha"]) == 0.0],
        ),
        (
            "replication",
            [row for row in protocol["replication"]["rows"] if float(row["alpha"]) == 1.0],
            protocol["replication"]["control_rows"],
            [row for row in protocol["replication"]["rows"] if float(row["alpha"]) == 0.0],
        ),
    ):
        historical_splits[split] = _summary(rows)
        mrsm_splits[split] = _summary(_mrsm_rows(rows))
        null_effect = None if not null_rows else _summary(null_rows)["raw_effect"]
        control_effect = None if not control_rows else _summary(control_rows)["raw_effect"]
        if split == "discovery":
            control_effect = "UNKNOWN"
        comparison.append({
            "split": split,
            "historical_effect": historical_splits[split]["raw_effect"],
            "mrsm_effect": mrsm_splits[split]["raw_effect"],
            "null_effect": null_effect,
            "control_effect": control_effect,
            "historical_snr": historical_splits[split]["snr"],
            "mrsm_snr": mrsm_splits[split]["snr"],
            "historical_sign_consistency": historical_splits[split]["sign_consistency"],
            "mrsm_sign_consistency": mrsm_splits[split]["sign_consistency"],
            "standardized_effect": historical_splits[split]["standardized_effect"],
            "variance": historical_splits[split]["variance"],
            "regime_consistency": historical_splits[split]["regime_sign_agreement"],
        })
    mrsm_signal = all(
        abs(block["raw_effect"]) > FLOOR and block["sign_consistency"] >= 0.75
        and abs(abs(block["raw_effect"]) - abs(RECORDED_MEANS[f"{split}_mean"])) <= MATCH_TOLERANCE
        for split, block in mrsm_splits.items()
    )
    stratification = _qf2_stratification()
    decision = decide(
        historical_reproduced=reproduced,
        failure_locus=locus,
        mrsm_signal_present=mrsm_signal,
        context_cancellation=bool(stratification["cancelling_slots"]),
    )
    environment = {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "python": platform.python_version(),
        "torch": bundle.versions["torch"],
        "transformer_lens": bundle.versions["transformer_lens"],
        "numpy": bundle.versions["numpy"],
        "model_revision": bundle.model_revision,
        "device": bundle.device,
        "dtype": bundle.dtype,
        "weight_sha256": sha256_file(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots" / PINNED_REVISION / "model.safetensors"),
        "direction_sha256": direction_sha,
        "prompt_manifest_sha256": prompt_sha,
        "tokenizer_token_ids": tokenizer_match,
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forwards": progress["n"],
    }
    interpretation = (
        f"Historical reproduction is {reproduced}. MRSM-metric signal present is {mrsm_signal}. "
        f"QF-2 cancellation slots are {stratification['cancelling_slots']}. "
        "The bridge target is the M22.1 primary direction at layer 23, which MRSM names as slot L1H2. "
        "That name is a slot, not a discovered mechanism. "
        "The orthogonal control remains larger than the primary contrast, which is why the historical status stays CANDIDATE."
    )
    payload = {
        "label": "QF-BRIDGE-DIAGNOSTIC",
        "inventory": records,
        "identity": identity,
        "selection": selection,
        "historical": {
            "status": protocol["status"],
            "selected_layer": protocol["selected_layer"],
            "evidence": evidence,
            "reproduced": reproduced,
            "failure_locus": locus,
            "splits": historical_splits,
        },
        "mrsm": mrsm_splits,
        "comparison_rows": comparison,
        "stratification": stratification,
        "strength": {
            "validation": _strength(protocol["validation"]),
            "replication": _strength(protocol["replication"]),
        },
        "decision": decision,
        "interpretation": interpretation,
        "environment": environment,
        "shuffled_control": "NOT_AVAILABLE",
    }
    out = _ROOT / "artifacts" / "qwen_bridge"
    out.mkdir(parents=True, exist_ok=True)
    _dump(out / "experiment_inventory.json", records)
    _dump(out / "identity_audit.json", identity)
    _dump(out / "bridge_reproduction.json", {
        "historical": payload["historical"],
        "mrsm": mrsm_splits,
        "comparison_rows": comparison,
        "shuffled_control": "NOT_AVAILABLE",
    })
    _dump(out / "context_stratification.json", stratification)
    _dump(out / "intervention_strength.json", payload["strength"])
    _dump(out / "decision.json", decision)
    lines = []
    for path in sorted(out.glob("*.json")):
        lines.append(f"{file_sha256(path)}  {path.name}")
    (out / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (_ROOT / "reports" / "QWEN_BRIDGE_INTERVENTION_ANALYSIS.md").write_text(_report(payload), encoding="utf-8")
    after = {rel: sha256_file(_ROOT / rel) for rel in FROZEN_FILES}
    if after != before:
        changed = [rel for rel in FROZEN_FILES if after[rel] != before[rel]]
        raise RuntimeError(f"frozen files changed: {changed}")
    print(json.dumps({
        "selected": selection["experiment_id"],
        "bridge_result": decision["bridge_result"],
        "primary_diagnosis": decision["primary_diagnosis"],
        "reproduced": reproduced,
        "mrsm_signal": mrsm_signal,
        "cancelling": stratification["cancelling_slots"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
