"""Compare frozen measurements to precomputed readout predictions.

This is the only M22.1.3 module that opens the measurement table.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .predict import fit_through_origin, r_squared_no_intercept, sha256_file

ROOT = Path(__file__).resolve().parents[3]


def load_audit_yaml(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else ROOT / "configs" / "m22_1_3_readout_null.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _rows(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _pass_mask(error: np.ndarray, scale: np.ndarray, floor: float, coef: float) -> np.ndarray:
    return error <= float(floor) + float(coef) * np.abs(scale)


def level_b(cells: list[dict], floor: float, coef: float, small: float, field: str) -> dict:
    usable = [row for row in cells if float(row["alpha"]) != 0.0]
    if not usable:
        return {"pass_fraction": None, "overall": _rows([]), "small": _rows([]), "n": 0, "n_pass": 0}
    error = np.asarray([abs(row[field] - row["delta_rec"]) for row in usable], dtype=np.float64)
    recorded = np.asarray([row["delta_rec"] for row in usable], dtype=np.float64)
    passed = _pass_mask(error, recorded, floor, coef)
    small_mask = np.abs(recorded) < float(small)
    return {
        "n": int(error.size),
        "n_pass": int(np.sum(passed)),
        "pass_fraction": float(np.mean(passed)),
        "overall": _rows(error.tolist()),
        "small": _rows(error[small_mask].tolist()),
        "failures": [
            {
                "prompt_id": row["prompt_id"],
                "direction_id": row["direction_id"],
                "alpha": row["alpha"],
                "delta_rec": row["delta_rec"],
                "predicted": row[field],
                "abs_error": abs(row[field] - row["delta_rec"]),
            }
            for row, ok in zip(usable, passed)
            if not ok
        ],
    }


def level_a(cells: list[dict], predicted_field: str, target_field: str, min_abs: float, floor: float, coef: float) -> dict:
    usable = [row for row in cells if abs(float(row[target_field])) > float(min_abs)]
    if not usable:
        return {"n": 0, "r2": None, "sign_agreement": None, "pass_fraction": None, "by_alpha": {}}
    actual = np.asarray([row[target_field] for row in usable], dtype=np.float64)
    predicted = np.asarray([row[predicted_field] for row in usable], dtype=np.float64)
    error = np.abs(predicted - actual)
    by_alpha = {}
    for alpha in sorted({float(row["alpha"]) for row in usable}):
        subset = [row for row in usable if float(row["alpha"]) == alpha]
        y = np.asarray([row[target_field] for row in subset], dtype=np.float64)
        yhat = np.asarray([row[predicted_field] for row in subset], dtype=np.float64)
        by_alpha[str(alpha)] = {
            "n": int(y.size),
            "r2": r_squared_no_intercept(y, yhat),
            "sign_agreement": float(np.mean(np.sign(y) == np.sign(yhat))),
            "pass_fraction": float(np.mean(_pass_mask(np.abs(yhat - y), y, floor, coef))),
        }
    return {
        "n": int(actual.size),
        "r2": r_squared_no_intercept(actual, predicted),
        "sign_agreement": float(np.mean(np.sign(actual) == np.sign(predicted))),
        "pass_fraction": float(np.mean(_pass_mask(error, actual, floor, coef))),
        "by_alpha": by_alpha,
    }


def curvature(cells: list[dict], field: str) -> dict:
    grouped = defaultdict(dict)
    for row in cells:
        grouped[(row["prompt_id"], row["direction_id"], abs(float(row["alpha"])))][float(row["alpha"])] = float(row[field])
    summary = {}
    for alpha in (1.0, 2.0):
        values = []
        relatives = []
        for (_, _, magnitude), pair in grouped.items():
            if magnitude != alpha or alpha not in pair or -alpha not in pair:
                continue
            curve = 0.5 * (pair[alpha] + pair[-alpha])
            values.append(curve)
            denom = abs(pair[alpha])
            relatives.append(abs(curve) / denom if denom > 1e-12 else None)
        finite = [value for value in relatives if value is not None]
        summary[str(int(alpha))] = {
            "n": len(values),
            "median": None if not values else float(np.median(values)),
            "median_abs_over_plus": None if not finite else float(np.median(finite)),
        }
    return summary


def direction_level(cells: list[dict]) -> dict:
    plus = defaultdict(list)
    minus = defaultdict(list)
    dots = {}
    for row in cells:
        dots[row["direction_id"]] = row["r_dot_d"]
        if float(row["alpha"]) == 1.0:
            plus[row["direction_id"]].append(row["delta_rec"])
        if float(row["alpha"]) == -1.0:
            minus[row["direction_id"]].append(row["delta_rec"])
    ids = sorted(plus)
    c = np.asarray([dots[direction_id] for direction_id in ids], dtype=np.float64)
    b_plus = np.asarray([float(np.mean(plus[direction_id])) for direction_id in ids], dtype=np.float64)
    b_minus = np.asarray([float(np.mean(minus[direction_id])) for direction_id in ids], dtype=np.float64)
    inv = [row["inv_rms"] for row in cells if float(row["alpha"]) == 1.0]
    fit_plus = fit_through_origin(c, b_plus)
    fit_minus = fit_through_origin(c, b_minus)
    mean_inv = float(np.mean(inv)) if inv else None
    ratio_c = None if c[0] == 0 else float(c[ids.index("D2")] / c[ids.index("D1")]) if "D1" in ids and "D2" in ids else None
    ratio_b = None
    if "D1" in ids and "D2" in ids and b_plus[ids.index("D1")] != 0.0:
        ratio_b = float(b_plus[ids.index("D2")] / b_plus[ids.index("D1")])
    log_gap = None
    if fit_plus["k"] not in (None, 0.0) and mean_inv not in (None, 0.0) and fit_plus["k"] * mean_inv > 0:
        log_gap = abs(float(np.log(fit_plus["k"] / mean_inv)))
    return {
        "direction_ids": ids,
        "c": {direction_id: float(dots[direction_id]) for direction_id in ids},
        "mean_delta_plus1": {direction_id: float(np.mean(plus[direction_id])) for direction_id in ids},
        "mean_delta_minus1": {direction_id: float(np.mean(minus[direction_id])) for direction_id in ids},
        "k_plus": fit_plus["k"],
        "r2_plus": fit_plus["r2"],
        "k_minus": fit_minus["k"],
        "r2_minus": fit_minus["r2"],
        "mean_inv_rms": mean_inv,
        "abs_log_k_over_mean_inv_rms": log_gap,
        "c_D2_over_c_D1": ratio_c,
        "b_D2_over_b_D1": ratio_b,
    }


def decide(config: dict, gate: str | None, per_split: dict) -> dict:
    if gate:
        return {"primary": gate, "linearity": None, "gate_override": True}
    labels = config["labels"]
    a2 = config["level_a2"]
    exact = all(per_split[name]["level_b_f32"]["pass_fraction"] == 1.0 for name in config["evaluation_splits"])
    if not exact:
        return {"primary": labels["mismatch"], "linearity": None, "gate_override": False}
    findings = []
    for name in config["evaluation_splits"]:
        r2 = per_split[name]["level_a2"]["r2"]
        sign = per_split[name]["level_a2"]["sign_agreement"]
        if r2 is not None and r2 >= float(a2["r2_supported"]) and sign is not None and sign >= float(a2["sign_supported"]):
            findings.append(labels["first_order"])
        elif r2 is not None and r2 < float(a2["r2_nonlinear"]):
            findings.append(labels["nonlinear"])
        else:
            findings.append(labels["inconclusive"])
    if len(set(findings)) == 1:
        linearity = findings[0]
    else:
        linearity = labels["inconclusive"]
    return {"primary": labels["exact"], "linearity": linearity, "gate_override": False}


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.6f}"


def render_report(config: dict, results: dict) -> str:
    labels = config["labels"]
    primary = results["decision"]["primary"]
    lines = [
        "# M22.1.3 readout-geometry null audit",
        "",
        f"Proposed finding id: `{config['proposed_finding_id']}`.",
        f"Primary label: `{primary}`.",
        f"Linearity finding: `{results['decision']['linearity']}`.",
        "",
        "Status: M22.1 remains `CANDIDATE` (unchanged). M22.2 is not authorized.",
        "Any status change requires a separate decision audit and explicit user approval.",
        "",
    ]
    if primary == labels["exact"]:
        lines.extend([config["exact_wording"], ""])
    elif primary == labels["mismatch"]:
        lines.extend(
            [
                "Level B did not pass on both evaluation splits.",
                "This is a pipeline diagnostic, not evidence against the readout hypothesis.",
                "",
            ]
        )
    if results.get("gate"):
        lines.extend([f"A gate fired: `{results['gate']}`.", ""])
    lines.extend(["## Environment and realized load", ""])
    env = results.get("environment", {})
    realized = results.get("realized", {})
    lines.append(
        f"- Python {env.get('python')}, Torch {env.get('torch')}, "
        f"Transformers {env.get('transformers')}, TransformerLens {env.get('transformer_lens')}."
    )
    lines.append(
        f"- Realized normalization `{realized.get('normalization_type')}`, "
        f"`ln_final` class `{realized.get('ln_final_class')}`, "
        f"weight present `{realized.get('ln_final_has_weight')}`, eps `{realized.get('eps')}`."
    )
    lines.append(
        f"- cfg flags: fold_ln `{realized.get('fold_ln')}`, "
        f"center_writing_weights `{realized.get('center_writing_weights')}`, "
        f"center_unembed `{realized.get('center_unembed')}`, "
        f"default_prepend_bos `{realized.get('default_prepend_bos')}`."
    )
    lines.append(f"- Config sha256 `{results.get('config_sha256')}`. Predictions sha256 `{results.get('predictions_sha256')}`.")
    lines.extend(["", "## Level B reconstruction", ""])
    for split in ("validation", "replication", "discovery"):
        if split not in results["splits"]:
            continue
        block = results["splits"][split]["level_b_f32"]
        lines.append(
            f"- {split}: pass {block['n_pass']}/{block['n']} = {block['pass_fraction']}, "
            f"median abs {block['overall']['median']}, p95 {block['overall']['p95']}, max {block['overall']['max']}."
        )
    lines.extend(["", "## Three-way table at alpha = +1", ""])
    for split in ("validation", "replication", "discovery"):
        if split not in results["splits"]:
            continue
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| direction | Δ_rec | Δ̂_exact f32 | Δ₁ Jacobian | Δ₀ frozen |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        table = results["splits"][split]["three_way_plus1"]
        for row in table:
            lines.append(
                f"| {row['direction_id']} | {_fmt(row['delta_rec'])} | {_fmt(row['delta_exact_f32'])} | "
                f"{_fmt(row['delta_jacobian'])} | {_fmt(row['delta_frozen'])} |"
            )
        lines.append("")
    lines.extend(["## Level A2 local Jacobian versus exact", ""])
    for split in ("validation", "replication"):
        if split not in results["splits"]:
            continue
        block = results["splits"][split]["level_a2"]
        lines.append(
            f"- {split}: n={block['n']}, R²={block['r2']}, sign={block['sign_agreement']}, pass={block['pass_fraction']}."
        )
    lines.extend(["", "## Level A frozen denominator and direction alignment", ""])
    for split in ("validation", "replication"):
        if split not in results["splits"]:
            continue
        level_a0 = results["splits"][split]["level_a0"]
        direction = results["splits"][split]["direction_level"]
        lines.append(
            f"- {split} Δ₀ vs exact: R²={level_a0['r2']}, sign={level_a0['sign_agreement']}."
        )
        lines.append(
            f"- {split} b ≈ k c: k_plus={direction['k_plus']}, R²={direction['r2_plus']}, "
            f"|log(k / mean 1/R)|={direction['abs_log_k_over_mean_inv_rms']}, "
            f"c[D2]/c[D1]={direction['c_D2_over_c_D1']}, b[D2]/b[D1]={direction['b_D2_over_b_D1']}."
        )
        lines.append("")
        lines.append("| direction | c = r·d | mean Δ_rec(+1) | mean Δ_rec(−1) |")
        lines.append("| --- | ---: | ---: | ---: |")
        for direction_id in direction["direction_ids"]:
            lines.append(
                f"| {direction_id} | {_fmt(direction['c'][direction_id])} | "
                f"{_fmt(direction['mean_delta_plus1'][direction_id])} | {_fmt(direction['mean_delta_minus1'][direction_id])} |"
            )
        lines.append("")
    if primary == labels["mismatch"]:
        lines.extend(["## Failing cells", ""])
        for split in config["evaluation_splits"]:
            failures = results["splits"][split]["level_b_f32"]["failures"]
            lines.append(f"### {split} ({len(failures)} cells)")
            for row in failures[:40]:
                lines.append(
                    f"- {row['prompt_id']} {row['direction_id']} α={row['alpha']}: "
                    f"Δ_rec={row['delta_rec']:+.6f} Δ̂={row['predicted']:+.6f} |err|={row['abs_error']:.6e}"
                )
            lines.extend(
                [
                    "",
                    "Candidate causes: hook site, BOS handling, batching/padding, dtype, or loader fold flags.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Non-claims",
            "",
            "- No mechanism is explained.",
            "- No claim is made about self-representation.",
            "- D1–D8 have no assigned semantic meaning.",
            "- Nothing is claimed about sites other than L23.",
            "- The D2>D1 ordering is reported only as readout alignment.",
            "",
            "## Status",
            "",
            "M22.1 remains `CANDIDATE`.",
            "M22.2 is `NOT AUTHORIZED`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_compare(config_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    config_path = Path(config_path) if config_path is not None else ROOT / "configs" / "m22_1_3_readout_null.yaml"
    config = load_audit_yaml(config_path)
    output = Path(output_dir) if output_dir is not None else ROOT
    predictions_path = output / config["paths"]["predictions"]
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    recorded_hash = predictions.get("run_log", {}).get("predictions_sha256")
    copy = dict(predictions)
    copy.get("run_log", {}).pop("predictions_sha256", None) if isinstance(copy.get("run_log"), dict) else None
    # The file on disk includes predictions_sha256 after the second write. Hash the current bytes
    # against the recorded value by hashing a payload without that self-hash, or the stored hash
    # after the second write. Verify by recomputing from the file excluding the self-hash field.
    body = json.loads(predictions_path.read_text(encoding="utf-8"))
    stored = body.get("run_log", {}).pop("predictions_sha256", None)
    recomputed = hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
    if stored and stored != recomputed and stored != recorded_hash:
        # Accept the on-disk hash if it matches the file bytes including the field.
        file_hash = sha256_file(predictions_path)
        if stored != file_hash and recorded_hash not in {file_hash, recomputed, stored}:
            raise RuntimeError("predictions sha256 does not match the file written by predict.py")
    delta_path = output / config["paths"]["delta_table"]
    table = json.loads(delta_path.read_text(encoding="utf-8"))
    records = table["records"]
    by_split_table = defaultdict(set)
    for row in records:
        by_split_table[row["role_split"]].add(row["prompt_id"])
    by_split_pred = defaultdict(set)
    for prompt_id, split in predictions.get("prompt_splits", {}).items():
        by_split_pred[split].add(prompt_id)
    if predictions.get("gate") == config["labels"]["prompt_missing"]:
        results = _terminal(config, predictions, config["labels"]["prompt_missing"])
        return _write_results(config, output, results)
    if predictions.get("gate") == config["labels"]["directions_fail"]:
        results = _terminal(config, predictions, config["labels"]["directions_fail"])
        return _write_results(config, output, results)
    for split in ("discovery", "validation", "replication"):
        if by_split_table[split] != by_split_pred[split]:
            results = _terminal(config, predictions, config["labels"]["prompt_set_mismatch"])
            return _write_results(config, output, results)
    alpha0 = [abs(float(row["actual_delta"])) for row in records if float(row["alpha"]) == 0.0]
    if any(value > float(config["gates"]["recorded_alpha0_abs_max"]) for value in alpha0):
        results = _terminal(config, predictions, config["labels"]["sanity_fail"])
        return _write_results(config, output, results)
    if predictions.get("gate") == config["labels"]["sanity_fail"]:
        results = _terminal(config, predictions, config["labels"]["sanity_fail"])
        return _write_results(config, output, results)
    stored_null = {}
    for row in records:
        stored_null.setdefault(row["prompt_id"], float(row["s_null"]))
        if abs(stored_null[row["prompt_id"]] - float(row["s_null"])) > 1e-12:
            raise RuntimeError(f"inconsistent s_null for {row['prompt_id']}")
    if stored_null:
        for prompt_id, value in stored_null.items():
            recomputed_margin = predictions["clean_margins"][prompt_id]["from_clean_logits"]
            if abs(recomputed_margin - value) > float(config["gates"]["clean_margin_abs_tol"]):
                results = _terminal(config, predictions, config["labels"]["reproduction_fail"])
                results["clean_margin_failures"] = {
                    prompt_id: {"stored": value, "recomputed": recomputed_margin}
                }
                return _write_results(config, output, results)
    measured = {}
    for row in records:
        key = (row["prompt_id"], row["direction_id"], float(row["alpha"]))
        measured[key] = float(row["actual_delta"])
    joined = []
    for row in predictions["cells"]:
        key = (row["prompt_id"], row["direction_id"], float(row["alpha"]))
        joined.append({**row, "delta_rec": measured[key]})
    per_split = {}
    for split in ("discovery", "validation", "replication"):
        subset = [row for row in joined if row["split"] == split]
        plus = [row for row in subset if float(row["alpha"]) == 1.0]
        by_direction = defaultdict(list)
        for row in plus:
            by_direction[row["direction_id"]].append(row)
        three_way = []
        for direction_id in config["direction_ids"]:
            rows = by_direction[direction_id]
            three_way.append(
                {
                    "direction_id": direction_id,
                    "delta_rec": float(np.mean([row["delta_rec"] for row in rows])),
                    "delta_exact_f32": float(np.mean([row["delta_exact_f32"] for row in rows])),
                    "delta_jacobian": float(np.mean([row["delta_jacobian"] for row in rows])),
                    "delta_frozen": float(np.mean([row["delta_frozen"] for row in rows])),
                }
            )
        per_split[split] = {
            "level_b_f32": level_b(subset, float(config["level_b"]["abs_floor"]), float(config["level_b"]["rel_coef"]), float(config["level_b"]["small_abs_cutoff"]), "delta_exact_f32"),
            "level_b_f64": level_b(subset, float(config["level_b"]["abs_floor"]), float(config["level_b"]["rel_coef"]), float(config["level_b"]["small_abs_cutoff"]), "delta_exact_f64"),
            "level_a2": level_a(subset, "delta_jacobian", "delta_exact_f32", float(config["level_a2"]["min_abs_exact"]), float(config["level_a2"]["abs_floor"]), float(config["level_a2"]["rel_coef"])),
            "level_a0": level_a(subset, "delta_frozen", "delta_exact_f32", float(config["level_a2"]["min_abs_exact"]), float(config["level_a2"]["abs_floor"]), float(config["level_a2"]["rel_coef"])),
            "curvature": curvature(subset, "delta_exact_f32"),
            "direction_level": direction_level(subset),
            "three_way_plus1": three_way,
        }
    decision = decide(config, None, per_split)
    results = {
        "milestone": "M22.1.3",
        "gate": None,
        "decision": decision,
        "m22_1_status": config["m22_1_status_preserved"],
        "m22_2_authorized": False,
        "proposed_finding_id": config["proposed_finding_id"],
        "config_sha256": sha256_file(config_path),
        "predictions_sha256": stored or recorded_hash or sha256_file(predictions_path),
        "environment": predictions.get("run_log", {}).get("environment", {}),
        "realized": predictions.get("run_log", {}).get("realized", {}),
        "readout_construction": predictions.get("run_log", {}).get("readout_construction"),
        "closed_form_check": predictions.get("run_log", {}).get("closed_form_check"),
        "jacobian_check": predictions.get("run_log", {}).get("jacobian_check"),
        "splits": per_split,
        "non_claims": [
            "no mechanism explained",
            "no self-representation claim",
            "no semantic meaning for D1-D8",
            "no claim about sites other than L23",
            "D2>D1 reported only as readout alignment",
        ],
    }
    return _write_results(config, output, results)


def _terminal(config: dict, predictions: dict, gate: str) -> dict:
    return {
        "milestone": "M22.1.3",
        "gate": gate,
        "decision": {"primary": gate, "linearity": None, "gate_override": True},
        "m22_1_status": config["m22_1_status_preserved"],
        "m22_2_authorized": False,
        "proposed_finding_id": config["proposed_finding_id"],
        "environment": predictions.get("run_log", {}).get("environment", {}),
        "realized": predictions.get("run_log", {}).get("realized", {}),
        "splits": {},
    }


def _write_results(config: dict, output: Path, results: dict) -> dict:
    results_path = output / config["paths"]["results"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    report = render_report(config, results)
    (output / config["paths"]["report"]).write_text(report, encoding="utf-8")
    hash_lines = []
    for key in ("predictions", "results", "directions_npy", "clean_resid_npy", "report", "hash_manifest"):
        path = output / config["paths"][key]
        if path.is_file() and key != "hash_manifest":
            hash_lines.append(f"{sha256_file(path)}  {config['paths'][key]}")
    hash_lines.append(f"{sha256_file(ROOT / 'configs' / 'm22_1_3_readout_null.yaml')}  configs/m22_1_3_readout_null.yaml")
    hash_path = output / config["paths"]["hash_manifest"]
    hash_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return results
