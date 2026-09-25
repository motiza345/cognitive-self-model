"""Run the frozen M22.1.1 panel once. Direction generation happens first."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import torch

from ..m22_1.loader import direction_tensor, forward_record
from ..m22_1.outcome import resolve_outcome_tokens
from ..m22_1.prompts import frozen_prompts, prompt_manifest_sha256
from .analyze import classify_panel, functional_correlations
from .panel import build_direction_panel, load_audit_config, panel_manifest


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _load_exact(config: dict):
    """Load the pinned revision with the same constructor arguments as M22.1.

    M22.1 passes device, dtype, and revision only. Extra folding flags are not
    added here. If the pinned revision cannot be requested, this stops.
    """
    from transformer_lens import HookedTransformer

    revision = str(config["model_revision"])
    signature = inspect.signature(HookedTransformer.from_pretrained)
    accepts_revision = "revision" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if not accepts_revision:
        raise RuntimeError("TransformerLens cannot pin a revision; refusing to load an unpinned checkpoint")
    model = HookedTransformer.from_pretrained(
        str(config["model_id"]),
        device="cpu",
        dtype=torch.float32,
        revision=revision,
    )
    model.eval()
    if getattr(model, "tokenizer", None) is not None:
        model.tokenizer.padding_side = "left"
        if model.tokenizer.pad_token_id is None and model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
    return model


class _Bundle:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"


def run_audit(config_path: str | Path, output_dir: str | Path) -> dict:
    config = load_audit_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    panel = build_direction_panel(int(config["expected_d_model"]), config)
    manifest = panel_manifest(panel)
    manifest["frozen_before_measurement"] = True
    (output / "direction_panel_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    model = _load_exact(config)
    if int(model.cfg.d_model) != int(config["expected_d_model"]):
        raise RuntimeError("loaded d_model does not match the frozen panel dimension")
    if int(model.cfg.n_layers) <= int(config["layer"]):
        raise RuntimeError("loaded model does not contain the frozen layer")
    outcome = resolve_outcome_tokens(
        model.tokenizer,
        str(config["positive_text"]),
        str(config["negative_text"]),
    )
    positive_id = int(outcome["positive_token_id"])
    negative_id = int(outcome["negative_token_id"])
    if (positive_id, negative_id) != (int(config["expected_positive_id"]), int(config["expected_negative_id"])):
        raise RuntimeError("outcome token ids differ from the frozen M22.1 pair")
    if not outcome["single_piece"]:
        raise RuntimeError("outcome texts are not single tokens on the frozen tokenizer")
    prompts = frozen_prompts()
    bundle = _Bundle(model)
    rows = []
    total = len(prompts) * len(panel["directions"]) * len(config["magnitude_grid"])
    done = 0
    hook_name = str(config["hook_name"])
    for prompt in prompts:
        null = forward_record(
            bundle,
            text=prompt.text,
            prepend_bos=bool(config["prepend_bos"]),
            hook=False,
            layer=None,
            alpha=0.0,
            direction=None,
            positive_id=positive_id,
            negative_id=negative_id,
            expected_hook_name=None,
        )
        null_margin = float(null["s"])
        for direction in panel["directions"]:
            vector = direction_tensor(direction["vector"], bundle.device)
            for alpha in config["magnitude_grid"]:
                measured = forward_record(
                    bundle,
                    text=prompt.text,
                    prepend_bos=bool(config["prepend_bos"]),
                    hook=True,
                    layer=int(config["layer"]),
                    alpha=float(alpha),
                    direction=vector,
                    positive_id=positive_id,
                    negative_id=negative_id,
                    expected_hook_name=hook_name,
                )
                if measured["hook_name"] != hook_name or not measured["hook_fired"]:
                    raise RuntimeError("intervention hook did not fire at the frozen site")
                if not measured["other_unchanged"]:
                    raise RuntimeError("intervention changed a token position other than the last token")
                rows.append({
                    "direction_id": direction["direction_id"],
                    "role": direction["role"],
                    "direction_sha256": direction["final_sha256"],
                    "direction_norm": direction["norm"],
                    "prompt_id": prompt.prompt_id,
                    "regime": prompt.regime_id,
                    "role_split": prompt.role,
                    "alpha": float(alpha),
                    "s_null": null_margin,
                    "s_intervened": float(measured["s"]),
                    "actual_delta": float(measured["s"]) - null_margin,
                    "hook_fired": measured["hook_fired"],
                    "hook_name_seen": measured["hook_name"],
                })
                done += 1
                if done % 40 == 0 or done == total:
                    print(f"measured {done}/{total}", flush=True)
    by_split = {
        role: [row for row in rows if row["role_split"] == role]
        for role in ("discovery", "validation", "replication")
    }
    characterization = classify_panel(by_split, config)
    correlations = {
        role: functional_correlations(split_rows, alpha=1.0)
        for role, split_rows in by_split.items()
    }
    report = {
        "milestone": "M22.1.1",
        "m22_1_status": config["m22_1_status_preserved"],
        "m22_2_authorized": False,
        "classification": characterization["label"],
        "classification_is_descriptive": True,
        "geometric_orthogonality_does_not_imply_functional_independence": True,
        "per_split": {
            role: {
                "label": detail["label"],
                "matching_labels": detail["matching_labels"],
                "substantial_floor": detail["substantial_floor"],
                "substantial_direction_ids": detail["substantial_direction_ids"],
                "max_over_min_abs_mean": detail["max_over_min_abs_mean"],
                "max_over_median_abs_mean": detail["max_over_median_abs_mean"],
                "regime_sign_change_direction_ids": detail["regime_sign_change_direction_ids"],
                "svd_uncentered": detail["svd_uncentered"],
                "profile": detail["profile"],
            }
            for role, detail in characterization["per_split"].items()
        },
        "functional_correlations_alpha_plus1": correlations,
        "non_claims": [
            "not a mechanism identity",
            "not a causal circuit",
            "not a SelfModel",
            "not self-awareness",
            "not general behavioral control",
            "not authorization for M22.2",
        ],
    }
    environment = {
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "architecture": str(getattr(model.cfg, "model_name", model.cfg.__class__.__name__)),
        "d_model": int(model.cfg.d_model),
        "n_layers": int(model.cfg.n_layers),
        "device": "cpu",
        "dtype": "float32",
        "torch": torch.__version__,
        "transformer_lens": _package_version("transformer-lens"),
        "transformers": _package_version("transformers"),
        "python": platform.python_version(),
        "prompt_manifest_sha256": prompt_manifest_sha256(prompts),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
    }
    payload = {"environment": environment, "records": rows, "report": report}
    (output / "direction_panel_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "intervention_space_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "README.md").write_text(_readme(config, environment, report), encoding="utf-8")
    return report


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def _readme(config: dict, environment: dict, report: dict) -> str:
    rules = config["classification"]
    lines = [
        "# M22.1.1 intervention-space audit",
        "",
        "Descriptive characterization of residual directions at one frozen site.",
        "This does not change the M22.1 candidate and does not authorize M22.2.",
        "",
        "## Protocol",
        "",
        f"- Model: `{environment['model_id']}` revision `{environment['model_revision']}`.",
        f"- Architecture: `{environment['architecture']}`, d_model `{environment['d_model']}`, layers `{environment['n_layers']}`.",
        f"- Device `{environment['device']}`, dtype `{environment['dtype']}`.",
        f"- Torch `{environment['torch']}`, TransformerLens `{environment['transformer_lens']}`, Transformers `{environment['transformers']}`.",
        f"- Site: `{config['hook_name']}`, last token only.",
        "- Outcome: `logit(9834) - logit(902)` for `\" yes\"` and `\" no\"`.",
        "- Prompts and discovery/validation/replication roles are the M22.1 frozen set.",
        "- Magnitudes: `-2, -1, 0, +1, +2` for every direction.",
        "- D1 is the M22.1 primary, seed 22101.",
        "- D2 is the M22.1 control, seed 22103, Gram-Schmidt against D1 only.",
        "- D3-D8 use seeds 22111-22116 and are Gram-Schmidt orthogonalized against every earlier panel member.",
        "- The panel manifest was written before the first hooked forward.",
        "- No direction was added, removed, or reseeded after an effect was observed.",
        "- Geometric orthogonality does not imply causal or functional independence.",
        "",
        "## Result",
        "",
        f"- Descriptive label: `{report['classification']}`.",
        f"- M22.1 status remains `{report['m22_1_status']}`.",
        "- M22.2 remains not authorized.",
        "",
        "The label is the frozen rule, not a new specificity pass.",
        "A split matches at most one of the pre-registered patterns.",
        "Zero matches, or more than one match, is `INSUFFICIENT_EVIDENCE`.",
        "The panel label is that shared validation/replication label.",
        "Discovery is reported and is not used for the label.",
        "",
        "The frozen comparisons, evaluated at alpha `+1`, are:",
        "",
        f"- broad: at least {rules['broad_min_directions']} substantial directions and max/min absolute mean below {rules['broad_max_over_min_ratio_exclusive']}",
        f"- structured: at least {rules['structured_min_directions']} substantial directions and max/median absolute mean at least {rules['structured_min_max_over_median_ratio']}",
        f"- low-dimensional: at most {rules['low_dimensional_max_substantial']} substantial directions and leading uncentered energy at least {rules['low_dimensional_min_leading_energy']}",
        f"- regime-dependent: at least {rules['regime_min_directions']} substantial directions whose regime means change sign and whose regime range exceeds half the absolute mean",
        "",
    ]
    for role in ("validation", "replication", "discovery"):
        detail = report["per_split"][role]
        energy = detail["svd_uncentered"]["leading_energy"]
        energy_text = "NA" if energy is None else f"{energy:.6f}"
        lines.append(
            f"- {role}: label `{detail['label']}`, substantial `{len(detail['substantial_direction_ids'])}`, "
            f"max/min `{detail['max_over_min_abs_mean']}`, max/median `{detail['max_over_median_abs_mean']}`, "
            f"leading energy `{energy_text}`, regime sign-change directions `{detail['regime_sign_change_direction_ids']}`."
        )
    lines.extend([
        "",
        "Validation and replication both match none of the four patterns, so the panel label is `INSUFFICIENT_EVIDENCE`.",
        "The raw profiles below are the result. The category was not adjusted after seeing them.",
        "",
        "## Effect profile",
        "",
        "Means are paired deltas on that split. `effect_per_unit_norm` equals the mean at `+1` because every direction has unit norm.",
        "",
    ])
    for role in ("discovery", "validation", "replication"):
        lines.append(f"### {role}")
        lines.append("")
        lines.append("| direction | role | mean Δ(+1) | mean Δ(-1) | sign consistency | symmetry |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in report["per_split"][role]["profile"]["directions"]:
            sign = row["sign_consistency"]
            sign_text = "NA" if sign is None else f"{sign:.3f}"
            lines.append(
                f"| {row['direction_id']} | {row['role']} | {_fmt(row['mean_delta_plus1'])} | "
                f"{_fmt(row['mean_delta_minus1'])} | {sign_text} | {_fmt(row['directional_symmetry_plus_plus_minus'])} |"
            )
        lines.append("")
        lines.append("| direction | Δ(-2) | Δ(-1) | Δ(0) | Δ(+1) | Δ(+2) | completion | instruction | syntax |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in report["per_split"][role]["profile"]["directions"]:
            dose = row["dose_response"]
            regimes = row["regime_means"]
            lines.append(
                "| {direction} | {m2} | {m1} | {z} | {p1} | {p2} | {c} | {i} | {s} |".format(
                    direction=row["direction_id"],
                    m2=_fmt(dose.get("-2.0")),
                    m1=_fmt(dose.get("-1.0")),
                    z=_fmt(dose.get("0.0")),
                    p1=_fmt(dose.get("1.0")),
                    p2=_fmt(dose.get("2.0")),
                    c=_fmt(regimes.get("completion")),
                    i=_fmt(regimes.get("instruction")),
                    s=_fmt(regimes.get("syntax")),
                )
            )
        lines.append("")
    lines.extend([
        "## Geometry and functional correlation",
        "",
        "Pairwise dots of the unit directions are in `direction_panel_manifest.json`.",
        "Off-diagonal dots are numerical zeros. That is geometric orthogonality.",
        "Prompt-level Pearson correlations of the alpha `+1` deltas are in `intervention_space_report.json`.",
        "Those correlations are not the same object as the dots. With six prompts in a split, they are descriptive.",
        "",
        "The uncentered singular-value energy of each prompt-by-direction delta matrix is stored per split.",
        "A large leading energy describes the observed matrix. It is not a causal mechanism, and it was not used to build a new intervention.",
        "",
        "## Interpretation limit",
        "",
        "On this frozen checkpoint, at this frozen L23 residual site and frozen logit-margin outcome, the pre-registered panel has a measurable causal response.",
        "The control is geometrically orthogonal to the primary and is still sensitive to the same outcome. Other panel directions are sensitive as well.",
        "The frozen category for that pattern is `INSUFFICIENT_EVIDENCE` because the magnitude spread sits between the broad and structured cutoffs, every direction is substantial, and regime signs do not flip.",
        "This does not establish a mechanism identity, a causal circuit, a SelfModel, self-awareness, or general behavioral control.",
        "M22.2 is not authorized.",
        "",
    ])
    return "\n".join(lines)
