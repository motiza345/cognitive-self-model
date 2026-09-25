"""Held-out response-space audit on frozen measurements.

Manifest construction does not read deltas. Fitting starts only after the
manifest hash is on disk.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from .data import (
    canonical_bytes,
    load_config,
    load_measurement_payload,
    prompts_by_split,
    response_matrix,
    sha256_bytes,
)
from .decision import assign_label
from .masking import held_out_mask
from .metrics import beats, bootstrap_mean_interval, cell_metrics
from .models import PREDICTORS, predict_imputed_svd, predict_ridge_als

METHOD_NAMES = (
    "B0_global_mean",
    "B1_prompt_mean",
    "B2_direction_mean",
    "B3_additive",
    "B4_rank1",
    "B5_rank2",
    "B6_rank3",
)


def _fit_kwargs(config: dict, method: str) -> dict:
    if method == "B3_additive":
        return {"ridge_lambda": float(config["ridge_lambda"])}
    if method in {"B4_rank1", "B5_rank2", "B6_rank3"}:
        return {
            "rank": {"B4_rank1": 1, "B5_rank2": 2, "B6_rank3": 3}[method],
            "ridge_lambda": float(config["ridge_lambda"]),
            "als_iterations": int(config["als_iterations"]),
            "als_tolerance": float(config["als_tolerance"]),
            "unseen_prompt_factor": config["unseen_prompt_factor"],
        }
    return {}


def score_held_out(matrix: np.ndarray, held: np.ndarray, config: dict) -> dict[str, dict]:
    train = ~held
    scored = {}
    for method in METHOD_NAMES:
        prediction = PREDICTORS[method](matrix, train, **_fit_kwargs(config, method))
        scored[method] = cell_metrics(matrix[held], prediction[held])
    return scored


def _mean_mae(per_seed: list[dict]) -> dict[str, float]:
    return {
        method: float(np.mean([seed[method]["mae"] for seed in per_seed]))
        for method in METHOD_NAMES
    }


def _permute_within_prompt(matrix: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    shuffled = np.empty_like(matrix)
    for prompt in range(matrix.shape[0]):
        order = generator.permutation(matrix.shape[1])
        shuffled[prompt] = matrix[prompt, order]
    return shuffled


def _permute_cells(matrix: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    flat = np.array(matrix, dtype=np.float64, copy=True).ravel()
    generator.shuffle(flat)
    return flat.reshape(matrix.shape)


def _sign_flip(matrix: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    signs = generator.choice(np.array([-1.0, 1.0]), size=matrix.shape[1])
    return matrix * signs


def blocked_replication(matrix: np.ndarray, masks: list[np.ndarray], config: dict) -> dict:
    per_seed = []
    prompt_errors: list[list[float]] = [[] for _ in range(matrix.shape[0])]
    direction_errors: list[list[float]] = [[] for _ in range(matrix.shape[1])]
    intervals = []
    for index, held in enumerate(masks):
        scored = score_held_out(matrix, held, config)
        per_seed.append(scored)
        train = ~held
        prediction = predict_ridge_als(matrix, train, **_fit_kwargs(config, "B4_rank1"))
        global_prediction = PREDICTORS["B0_global_mean"](matrix, train)
        direction_prediction = PREDICTORS["B2_direction_mean"](matrix, train)
        rows, columns = np.where(held)
        absolute = np.abs(matrix[held] - prediction[held])
        paired_global = np.abs(matrix[held] - global_prediction[held]) - absolute
        paired_direction = np.abs(matrix[held] - direction_prediction[held]) - absolute
        for row, column, error in zip(rows, columns, absolute):
            prompt_errors[int(row)].append(float(error))
            direction_errors[int(column)].append(float(error))
        intervals.append(
            {
                "seed": int(config["mask_seeds"][index]),
                "rank1_mae": bootstrap_mean_interval(absolute, int(config["bootstrap_seed"]) + index, int(config["n_bootstrap"])),
                "paired_improvement_vs_global": bootstrap_mean_interval(
                    paired_global, int(config["bootstrap_seed"]) + 100 + index, int(config["n_bootstrap"])
                ),
                "paired_improvement_vs_direction": bootstrap_mean_interval(
                    paired_direction, int(config["bootstrap_seed"]) + 200 + index, int(config["n_bootstrap"])
                ),
            }
        )
    mae_mean = _mean_mae(per_seed)
    seed_flags = []
    for seed in per_seed:
        seed_flags.append(
            beats(
                seed["B4_rank1"]["mae"],
                seed["B0_global_mean"]["mae"],
                float(config["min_relative_mae_reduction"]),
                float(config["baseline_mae_floor"]),
            )
            and beats(
                seed["B4_rank1"]["mae"],
                seed["B2_direction_mean"]["mae"],
                float(config["min_relative_mae_reduction"]),
                float(config["baseline_mae_floor"]),
            )
        )
    secondary = []
    for held in masks:
        row = {}
        for rank in (1, 2, 3):
            prediction = predict_imputed_svd(matrix, ~held, rank=rank)
            row[f"svd_rank{rank}"] = cell_metrics(matrix[held], prediction[held])
        secondary.append(row)
    return {
        "per_seed": per_seed,
        "blocked_mae_mean": mae_mean,
        "blocked_mae_median": {
            method: float(np.median([seed[method]["mae"] for seed in per_seed])) for method in METHOD_NAMES
        },
        "blocked_mae_std": {
            method: float(np.std([seed[method]["mae"] for seed in per_seed], ddof=1)) for method in METHOD_NAMES
        },
        "seeds_beating_both_baselines": int(sum(seed_flags)),
        "normalized_mae_vs_global": mae_mean["B4_rank1"] / mae_mean["B0_global_mean"],
        "improvement_over_additive": mae_mean["B3_additive"] - mae_mean["B4_rank1"],
        "improvement_over_direction": mae_mean["B2_direction_mean"] - mae_mean["B4_rank1"],
        "per_prompt_mae": [float(np.mean(errors)) for errors in prompt_errors],
        "per_direction_mae": [float(np.mean(errors)) for errors in direction_errors],
        "bootstrap": intervals,
        "secondary_imputed_svd_mae_mean": {
            name: float(np.mean([seed[name]["mae"] for seed in secondary]))
            for name in ("svd_rank1", "svd_rank2", "svd_rank3")
        },
    }


def _control_means(matrix: np.ndarray, masks: list[np.ndarray], config: dict) -> dict[str, dict[str, float]]:
    cell = blocked_replication(_permute_within_prompt(matrix, int(config["cell_permutation_seed"])), masks, config)
    pairing = blocked_replication(_permute_cells(matrix, int(config["global_permutation_seed"])), masks, config)
    signed = blocked_replication(_sign_flip(matrix, int(config["sign_permutation_seed"])), masks, config)
    magnitude = []
    for held in masks:
        prediction = predict_ridge_als(np.abs(matrix), ~held, **_fit_kwargs(config, "B4_rank1"))
        magnitude.append(cell_metrics(np.abs(matrix[held]), prediction[held])["mae"])
    return {
        "cell_permutation": cell["blocked_mae_mean"],
        "global_permutation": pairing["blocked_mae_mean"],
        "sign_permutation": signed["blocked_mae_mean"],
        "magnitude_only_rank1_mae": float(np.mean(magnitude)),
    }


def leave_one_prompt_out(matrix: np.ndarray, config: dict) -> dict:
    per_prompt = []
    methods = ("B0_global_mean", "B2_direction_mean", "B3_additive", "B4_rank1")
    for prompt in range(matrix.shape[0]):
        train = np.ones(matrix.shape, dtype=bool)
        train[prompt, :] = False
        scores = {}
        for method in methods:
            prediction = PREDICTORS[method](matrix, train, **_fit_kwargs(config, method))
            scores[method] = cell_metrics(matrix[prompt], prediction[prompt])["mae"]
        per_prompt.append(scores)
    return {
        "per_prompt_mae": per_prompt,
        "mae_mean": {method: float(np.mean([row[method] for row in per_prompt])) for method in methods},
    }


def leave_one_regime_out(matrix: np.ndarray, regimes: list[str], config: dict) -> dict:
    methods = ("B0_global_mean", "B2_direction_mean", "B3_additive", "B4_rank1")
    per_regime = {}
    held_errors = {method: [] for method in methods}
    for regime in sorted(set(regimes)):
        train = np.ones(matrix.shape, dtype=bool)
        rows = [index for index, name in enumerate(regimes) if name == regime]
        train[rows, :] = False
        per_regime[regime] = {}
        for method in methods:
            prediction = PREDICTORS[method](matrix, train, **_fit_kwargs(config, method))
            error = np.abs(matrix[rows] - prediction[rows]).ravel()
            held_errors[method].extend(float(value) for value in error)
            per_regime[regime][method] = float(np.mean(error))
    return {
        "per_regime_mae": per_regime,
        "mae_mean": {method: float(np.mean(held_errors[method])) for method in methods},
    }


def calibrate_direction(matrix: np.ndarray, calibration_prompt: int, config: dict) -> dict:
    rank_errors = []
    constant_errors = []
    global_errors = []
    per_direction = []
    for direction in range(matrix.shape[1]):
        train = np.ones(matrix.shape, dtype=bool)
        train[:, direction] = False
        train[calibration_prompt, direction] = True
        held = ~train[:, direction]
        prediction = predict_ridge_als(matrix, train, **_fit_kwargs(config, "B4_rank1"))
        actual = matrix[held, direction]
        constant = np.full(actual.shape, matrix[calibration_prompt, direction])
        global_value = float(np.mean(matrix[train]))
        rank_absolute = np.abs(actual - prediction[held, direction])
        constant_absolute = np.abs(actual - constant)
        global_absolute = np.abs(actual - global_value)
        rank_errors.extend(rank_absolute.tolist())
        constant_errors.extend(constant_absolute.tolist())
        global_errors.extend(global_absolute.tolist())
        per_direction.append(
            {
                "rank1_mae": float(np.mean(rank_absolute)),
                "constant_mae": float(np.mean(constant_absolute)),
                "global_mae": float(np.mean(global_absolute)),
            }
        )
    return {
        "rank1_mae": float(np.mean(rank_errors)),
        "constant_mae": float(np.mean(constant_errors)),
        "global_mae": float(np.mean(global_errors)),
        "directions_not_worse_than_constant": int(
            sum(row["rank1_mae"] <= row["constant_mae"] + 1e-12 for row in per_direction)
        ),
        "per_direction": per_direction,
    }


def calibration_audit(matrix: np.ndarray, calibration_prompt: int, config: dict) -> dict:
    real = calibrate_direction(matrix, calibration_prompt, config)
    permuted = calibrate_direction(
        _permute_within_prompt(matrix, int(config["calibration_permutation_seed"])),
        calibration_prompt,
        config,
    )
    real["permuted_rank1_mae"] = permuted["rank1_mae"]
    real["permuted_constant_mae"] = permuted["constant_mae"]
    real["permuted_global_mae"] = permuted["global_mae"]
    return real


def symmetry_mae(plus: np.ndarray, minus: np.ndarray) -> float:
    return float(np.mean(np.abs(plus + minus)))


def build_manifest(config: dict) -> dict:
    grouped, prompt_hash = prompts_by_split()
    if prompt_hash != config["prompt_manifest_sha256"]:
        raise RuntimeError("Prompt manifest hash does not match the frozen audit config")
    direction_ids = list(config["direction_ids"])
    schedule = {}
    calibration = {}
    for split, prompts in grouped.items():
        prompt_ids = [prompt.prompt_id for prompt in prompts]
        calibration[split] = prompt_ids[0]
        schedule[split] = []
        for seed in config["mask_seeds"]:
            mask = held_out_mask(len(prompt_ids), len(direction_ids), int(seed), config["base_mask_pairs"])
            cells = [
                {"prompt_id": prompt_ids[int(row)], "direction_id": direction_ids[int(column)]}
                for row, column in zip(*np.where(mask))
            ]
            schedule[split].append({"seed": int(seed), "held_out_cells": cells})
    body = {
        "milestone": "M22.1.2",
        "frozen_before_benchmark": True,
        "model_forward_passes": 0,
        "prompt_manifest_sha256": prompt_hash,
        "direction_ids": direction_ids,
        "calibration_prompt_id": calibration,
        "calibration_prompt_rule": config["calibration_prompt_rule"],
        "mask_schedule": schedule,
        "primary_estimator": config["primary_estimator"],
        "decision_model": config["decision_model"],
        "decision_rank": config["decision_rank"],
        "decision_rule": config["decision_rule"],
        "mask_seeds": config["mask_seeds"],
        "base_mask_pairs": config["base_mask_pairs"],
        "ridge_lambda": config["ridge_lambda"],
        "secondary_closed_form_estimator": config["secondary_closed_form_estimator"],
        "secondary_estimator_is_not_used_for_the_label": True,
    }
    body["manifest_sha256"] = sha256_bytes(canonical_bytes(body))
    return body


def masks_from_schedule(schedule: list[dict], prompt_ids: list[str], direction_ids: list[str]) -> list[np.ndarray]:
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    direction_index = {direction_id: index for index, direction_id in enumerate(direction_ids)}
    masks = []
    for entry in schedule:
        mask = np.zeros((len(prompt_ids), len(direction_ids)), dtype=bool)
        for cell in entry["held_out_cells"]:
            mask[prompt_index[cell["prompt_id"]], direction_index[cell["direction_id"]]] = True
        masks.append(mask)
    return masks


def _package_leakage(package_dir: Path) -> list[str]:
    offenders = []
    forbidden_modules = ("m20", "m21", "benchmark", "legacy_import")
    forbidden_text = (
        "q_" + "invalid",
        "self_model_" + "invalid",
        "s_" + "intervened",
        "torch." + "autograd",
        "prompt_" + "embedding",
    )
    for path in sorted(package_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = " ".join(alias.name for alias in node.names)
            else:
                continue
            if any(token in module for token in forbidden_modules):
                offenders.append(f"{path.name}: {module}")
        for token in forbidden_text:
            if token in source:
                offenders.append(f"{path.name}: {token}")
    return offenders


def _scramble_invariant(matrix: np.ndarray, held: np.ndarray, config: dict) -> bool:
    contaminated = np.array(matrix, dtype=np.float64, copy=True)
    contaminated[held] = 1e9
    for method in ("B0_global_mean", "B2_direction_mean", "B3_additive", "B4_rank1"):
        clean = PREDICTORS[method](matrix, ~held, **_fit_kwargs(config, method))
        dirty = PREDICTORS[method](contaminated, ~held, **_fit_kwargs(config, method))
        if not np.allclose(clean, dirty, atol=1e-10, rtol=1e-8):
            return False
    return True


def evaluate_split(matrix: np.ndarray, minus: np.ndarray, regimes: list[str], masks: list[np.ndarray], calibration_prompt: int, config: dict) -> dict:
    blocked = blocked_replication(matrix, masks, config)
    controls = _control_means(matrix, masks, config)
    summary = {
        "blocked_mae_mean": blocked["blocked_mae_mean"],
        "blocked_mae_median": blocked["blocked_mae_median"],
        "blocked_mae_std": blocked["blocked_mae_std"],
        "seeds_beating_both_baselines": blocked["seeds_beating_both_baselines"],
        "normalized_mae_vs_global": blocked["normalized_mae_vs_global"],
        "improvement_over_additive": blocked["improvement_over_additive"],
        "improvement_over_direction": blocked["improvement_over_direction"],
        "per_prompt_blocked_mae": blocked["per_prompt_mae"],
        "per_direction_blocked_mae": blocked["per_direction_mae"],
        "bootstrap": blocked["bootstrap"],
        "secondary_imputed_svd_mae_mean": blocked["secondary_imputed_svd_mae_mean"],
        "control_mae_mean": {
            "cell_permutation": controls["cell_permutation"],
            "global_permutation": controls["global_permutation"],
            "sign_permutation": controls["sign_permutation"],
        },
        "magnitude_only_rank1_mae": controls["magnitude_only_rank1_mae"],
        "leave_one_prompt": leave_one_prompt_out(matrix, config),
        "leave_one_regime_mae": leave_one_regime_out(matrix, regimes, config)["mae_mean"]["B4_rank1"],
        "leave_one_regime": leave_one_regime_out(matrix, regimes, config),
        "calibration": calibration_audit(matrix, calibration_prompt, config),
        "symmetry_mae": symmetry_mae(matrix, minus),
        "per_seed_mae": [
            {method: seed[method]["mae"] for method in METHOD_NAMES} | {"rmse_rank1": seed["B4_rank1"]["rmse"], "pearson_rank1": seed["B4_rank1"]["pearson"], "spearman_rank1": seed["B4_rank1"]["spearman"]}
            for seed in blocked["per_seed"]
        ],
    }
    return summary


def render_report(config: dict, manifest: dict, results: dict) -> str:
    label = results["label"]["label"]
    lines = [
        "# M22.1.2 response-space identifiability",
        "",
        f"Descriptive label: `{label}`.",
        "",
        "The audit reuses the frozen M22.1.1 measurements. It does not run a new model forward.",
        f"M22.1 remains `{results['m22_1_status']}`. M22.2 remains not authorized.",
        "",
        "## Frozen estimator",
        "",
        f"- Primary estimator: `{config['primary_estimator']}` rank {config['decision_rank']}, ridge `{config['ridge_lambda']}`.",
        f"- Decision model: `{config['decision_model']}`.",
        f"- Secondary SVD is reported and is not used for the label.",
        f"- Manifest: `{manifest['manifest_sha256']}`.",
        "",
        "## Blocked-cell MAE",
        "",
        "Means are over the five pre-registered mask seeds. Lower is better.",
        "",
    ]
    header = "| model | " + " | ".join(results["evaluation_splits"]) + " |"
    lines.extend([header, "| --- | " + " | ".join(["---:"] * len(results["evaluation_splits"])) + " |"])
    for method in list(METHOD_NAMES) + ["B7_cell_permutation_rank1"]:
        cells = []
        for split in results["evaluation_splits"]:
            summary = results["splits"][split]
            if method == "B7_cell_permutation_rank1":
                value = summary["control_mae_mean"]["cell_permutation"]["B4_rank1"]
            else:
                value = summary["blocked_mae_mean"][method]
            cells.append(f"{value:.6f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "Median and standard deviation of masked-cell MAE across mask seeds:", ""])
    lines.append("| model | " + " | ".join(f"{split} median | {split} std" for split in results["evaluation_splits"]) + " |")
    lines.append("| --- | " + " | ".join(["---: | ---:"] * len(results["evaluation_splits"])) + " |")
    for method in ("B0_global_mean", "B2_direction_mean", "B4_rank1"):
        cells = []
        for split in results["evaluation_splits"]:
            summary = results["splits"][split]
            cells.append(f"{summary['blocked_mae_median'][method]:.6f} | {summary['blocked_mae_std'][method]:.6f}")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Negative controls and secondary audits", ""])
    for split in results["evaluation_splits"]:
        summary = results["splits"][split]
        prompt_ids = results["matrices"][split]["prompt_ids"]
        lines.append(f"### {split}")
        lines.append("")
        lines.append(
            f"- Seeds on which rank-1 beats both baselines by the frozen margin: {summary['seeds_beating_both_baselines']} of 5."
        )
        lines.append(f"- Normalized MAE versus global mean: {summary['normalized_mae_vs_global']:.4f}.")
        lines.append(f"- Improvement over direction mean (baseline minus rank-1): {summary['improvement_over_direction']:.6f}.")
        lines.append(f"- Improvement over additive model: {summary['improvement_over_additive']:.6f}.")
        lines.append(f"- Global-permutation rank-1 MAE: {summary['control_mae_mean']['global_permutation']['B4_rank1']:.6f}.")
        lines.append(f"- Sign-permutation rank-1 MAE: {summary['control_mae_mean']['sign_permutation']['B4_rank1']:.6f}.")
        lines.append(f"- Magnitude-only rank-1 MAE: {summary['magnitude_only_rank1_mae']:.6f}.")
        lines.append(f"- Leave-one-prompt rank-1 MAE: {summary['leave_one_prompt']['mae_mean']['B4_rank1']:.6f}.")
        lines.append(f"- Leave-one-regime rank-1 MAE: {summary['leave_one_regime_mae']:.6f}.")
        lines.append(
            f"- Calibration rank-1 / constant / global MAE: {summary['calibration']['rank1_mae']:.6f} / {summary['calibration']['constant_mae']:.6f} / {summary['calibration']['global_mae']:.6f}."
        )
        lines.append(f"- Symmetry mean |Δ(+1)+Δ(-1)|: {summary['symmetry_mae']:.6f}.")
        lines.append(
            "- Per-prompt blocked rank-1 MAE: "
            + ", ".join(
                f"{prompt_id} {error:.6f}"
                for prompt_id, error in zip(prompt_ids, summary["per_prompt_blocked_mae"])
            )
            + "."
        )
        lines.append(
            "- Per-direction blocked rank-1 MAE: "
            + ", ".join(
                f"D{index + 1} {error:.6f}" for index, error in enumerate(summary["per_direction_blocked_mae"])
            )
            + "."
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation limit",
            "",
            "Held-out cell prediction tests whether the low-rank description is useful inside this frozen panel.",
            "It does not identify a causal circuit, a direction's semantic meaning, or a SelfModel.",
            "An unseen direction without a calibration cell is outside the claim.",
            "",
        ]
    )
    return "\n".join(lines)


def render_readme(results: dict) -> str:
    return "\n".join(
        [
            "# M22.1.2 response-space audit",
            "",
            "This note records the held-out prediction audit of the frozen M22.1.1 response matrices.",
            "",
            "## Protocol",
            "",
            "- No new model forward pass.",
            "- Primary target: alpha `+1` prompt-by-direction deltas on validation and replication separately.",
            "- Alpha `-1` is a symmetry check and is not a training target.",
            "- Masks hold out two cells per prompt and are stored in the manifest before fitting.",
            "- The label uses ridge alternating least squares of rank 1.",
            "- Mean-imputed SVD is a secondary closed form and cannot change the label.",
            "- Few-shot calibration reveals the lexicographic first prompt of a held-out direction.",
            "",
            "## Result",
            "",
            f"- Label: `{results['label']['label']}`.",
            f"- Manifest: `{results['manifest_sha256']}`.",
            f"- M22.1 status: `{results['m22_1_status']}`.",
            "- M22.2: not authorized.",
            "",
            "## Non-claims",
            "",
            "The result does not establish a causal mechanism, semantic direction identity, self-awareness, or a causal SelfModel.",
            "",
        ]
    )


def run_audit(config_path: str | Path, output_dir: str | Path) -> dict:
    config = load_config(config_path)
    if config["primary_estimator"] != "ridge_als" or config["decision_model"] != "B4_rank1":
        raise RuntimeError("Refusing to run with an estimator that was not the frozen primary")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config)
    manifest_path = output / "m22_1_2_response_space_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_hash = reloaded.pop("manifest_sha256")
    if sha256_bytes(canonical_bytes(reloaded)) != recorded_hash:
        raise RuntimeError("Manifest hash changed while it was being written")
    if recorded_hash != manifest["manifest_sha256"]:
        raise RuntimeError("Manifest hash does not match the pre-benchmark manifest")
    payload = load_measurement_payload(config["measurement_results"], config)
    grouped, _ = prompts_by_split()
    direction_ids = list(config["direction_ids"])
    package_offenders = _package_leakage(Path(__file__).resolve().parent)
    split_summaries = {}
    matrices = {}
    scramble_ok = True
    for split, prompts in grouped.items():
        prompt_ids = [prompt.prompt_id for prompt in prompts]
        plus, regimes = response_matrix(
            payload["records"],
            split=split,
            alpha=float(config["primary_alpha"]),
            prompt_ids=prompt_ids,
            direction_ids=direction_ids,
        )
        minus, _ = response_matrix(
            payload["records"],
            split=split,
            alpha=float(config["symmetry_alpha"]),
            prompt_ids=prompt_ids,
            direction_ids=direction_ids,
        )
        masks = masks_from_schedule(manifest["mask_schedule"][split], prompt_ids, direction_ids)
        for seed, mask in zip(config["mask_seeds"], masks):
            expected = held_out_mask(len(prompt_ids), len(direction_ids), int(seed), config["base_mask_pairs"])
            if not np.array_equal(mask, expected):
                raise RuntimeError("Stored mask does not match the pre-registered generator")
        calibration_prompt = prompt_ids.index(manifest["calibration_prompt_id"][split])
        if calibration_prompt != 0:
            raise RuntimeError("Calibration prompt is not the lexicographic first prompt")
        if not _scramble_invariant(plus, masks[0], config):
            scramble_ok = False
        split_summaries[split] = evaluate_split(plus, minus, regimes, masks, calibration_prompt, config)
        matrices[split] = {"alpha_plus1": plus.tolist(), "alpha_minus1": minus.tolist(), "prompt_ids": prompt_ids, "regimes": regimes}
    leakage_pass = scramble_ok and not package_offenders
    label = assign_label(
        split_summaries["validation"],
        split_summaries["replication"],
        leakage_pass,
        config,
    )
    results = {
        "milestone": "M22.1.2",
        "manifest_sha256": recorded_hash,
        "model_forward_passes": 0,
        "m22_1_status": config["m22_1_status_preserved"],
        "m22_2_authorized": False,
        "evaluation_splits": list(config["evaluation_splits"]),
        "discovery_used_for_decision": False,
        "label": label,
        "leakage": {"scramble_invariant": scramble_ok, "offenders": package_offenders},
        "splits": split_summaries,
        "matrices": matrices,
        "non_claims": [
            "not a causal circuit",
            "not a mechanism identity",
            "not semantic direction meaning",
            "not a SelfModel",
            "not self-awareness",
            "not authorization for M22.2",
        ],
    }
    (output / "m22_1_2_response_space_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = render_report(config, manifest, results)
    readme = render_readme(results)
    (output / "m22_1_2_response_space_report.md").write_text(report, encoding="utf-8")
    (output / "m22_1_2_response_space_README.md").write_text(readme, encoding="utf-8")
    return results
