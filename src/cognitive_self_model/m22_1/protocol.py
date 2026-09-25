"""Bounded discovery, then frozen validation. No second-look selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .config import hook_name
from .metrics import (
    bootstrap_mean_interval,
    by_regime,
    classify_directionality,
    effect_floor,
    fraction_beyond_floor,
    spearman_alpha_response,
    summarize_deltas,
    wilcoxon_versus_zero,
)
from .prompts import PromptRecord, prompts_by_role
from .status import classify_status

Measure = Callable[..., dict[str, Any]]


def candidate_layers(n_layers: int, n_candidates: int) -> list[int]:
    if n_layers < 1:
        raise ValueError("Model has no layers.")
    if n_candidates < 1:
        raise ValueError("Candidate count must be positive.")
    if n_layers <= n_candidates:
        return list(range(n_layers))
    grid = np.linspace(0, n_layers - 1, int(n_candidates))
    return sorted({int(np.rint(value)) for value in grid})


def select_layer(discovery_means: dict[int, float]) -> int:
    if not discovery_means:
        raise ValueError("Discovery produced no layer scores.")
    return min(discovery_means, key=lambda layer: (-float(discovery_means[layer]), int(layer)))


def _require_role(prompts: list[PromptRecord], role: str) -> None:
    roles = {prompt.role for prompt in prompts}
    if roles != {role}:
        raise RuntimeError(f"{role} measurement received roles {sorted(roles)}.")


def _row(
    prompt: PromptRecord,
    *,
    layer: int | None,
    alpha: float,
    direction_id: str,
    s_null: float,
    measured: dict[str, Any],
) -> dict[str, Any]:
    score = float(measured["s"])
    return {
        "prompt_id": prompt.prompt_id,
        "prompt_sha256": prompt.prompt_sha256,
        "regime_id": prompt.regime_id,
        "role": prompt.role,
        "layer": layer,
        "alpha": float(alpha),
        "direction_id": direction_id,
        "s_null": float(s_null),
        "s_intervened": score,
        "actual_delta": float(score - s_null),
        "hook_fired": bool(measured["hook_fired"]),
        "last_modified": bool(measured["last_modified"]),
        "other_unchanged": bool(measured["other_unchanged"]),
        "hook_name": measured.get("hook_name"),
    }


def _hook_ok(row: dict[str, Any], expected_name: str | None) -> bool:
    if row["direction_id"] == "none":
        return (not row["hook_fired"]) and (not row["last_modified"]) and row["other_unchanged"]
    if row["hook_name"] != expected_name or not row["hook_fired"] or not row["other_unchanged"]:
        return False
    if float(row["alpha"]) == 0.0:
        return not row["last_modified"]
    return bool(row["last_modified"])


def _mean(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([row["actual_delta"] for row in rows]))


def _evaluate_grid(
    prompts: list[PromptRecord],
    *,
    role: str,
    layer: int,
    grid: list[float],
    measure: Measure,
    s_null: dict[str, float],
    primary_alpha: float,
) -> dict[str, Any]:
    _require_role(prompts, role)
    name = hook_name(layer)
    rows: list[dict[str, Any]] = []
    for alpha in grid:
        for prompt in prompts:
            measured = measure(
                prompt,
                hook=True,
                layer=layer,
                alpha=float(alpha),
                direction_id="primary",
            )
            rows.append(
                _row(
                    prompt,
                    layer=layer,
                    alpha=float(alpha),
                    direction_id="primary",
                    s_null=s_null[prompt.prompt_id],
                    measured=measured,
                )
            )
    control_rows = []
    for prompt in prompts:
        measured = measure(
            prompt,
            hook=True,
            layer=layer,
            alpha=float(primary_alpha),
            direction_id="control",
        )
        control_rows.append(
            _row(
                prompt,
                layer=layer,
                alpha=float(primary_alpha),
                direction_id="control",
                s_null=s_null[prompt.prompt_id],
                measured=measured,
            )
        )
    integrity = all(_hook_ok(row, name) for row in rows + control_rows)
    by_alpha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_alpha[f"{row['alpha']:.8f}"].append(row)
    alpha_means = [(float(alpha), _mean(by_alpha[f"{float(alpha):.8f}"])) for alpha in grid]
    primary_rows = by_alpha[f"{float(primary_alpha):.8f}"]
    negative_rows = by_alpha[f"{-float(primary_alpha):.8f}"]
    primary_by_prompt = {row["prompt_id"]: row["actual_delta"] for row in primary_rows}
    control_by_prompt = {row["prompt_id"]: row["actual_delta"] for row in control_rows}
    paired = [primary_by_prompt[prompt.prompt_id] - control_by_prompt[prompt.prompt_id] for prompt in prompts]
    return {
        "rows": rows,
        "control_rows": control_rows,
        "hook_integrity_pass": integrity,
        "positive_deltas": [row["actual_delta"] for row in primary_rows],
        "negative_deltas": [row["actual_delta"] for row in negative_rows],
        "control_deltas": [row["actual_delta"] for row in control_rows],
        "paired_difference_deltas": paired,
        "primary_mean": _mean(primary_rows),
        "control_mean": _mean(control_rows),
        "paired_difference_mean": float(np.mean(paired)),
        "alpha_means": alpha_means,
        "per_prompt_primary": primary_rows,
        "per_regime_primary": by_regime(primary_rows),
        "per_regime_control": by_regime(control_rows),
    }


def execute_protocol(
    prompts: list[PromptRecord],
    *,
    n_layers: int,
    measure: Measure,
    config: dict[str, Any],
    revision_pinned: bool,
    definition_reproducible: bool,
    leakage_pass: bool,
    on_freeze: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run discovery, freeze one layer, then score validation and replication.

    `measure` is called with keyword arguments:
    prompt, hook, layer, alpha, direction_id.
    It must return s, hook_fired, last_modified, other_unchanged, hook_name.
    """
    grouped = prompts_by_role(prompts)
    for role, group in grouped.items():
        _require_role(group, role)
    layers = candidate_layers(int(n_layers), int(config["n_candidate_layers"]))
    grid = [float(value) for value in config["magnitude_grid"]]
    primary = float(config["primary_alpha"])
    s_null: dict[str, float] = {}
    null_rows = []
    finite = True
    for prompt in prompts:
        measured = measure(prompt, hook=False, layer=None, alpha=0.0, direction_id="none")
        if not np.isfinite(measured["s"]):
            finite = False
        s_null[prompt.prompt_id] = float(measured["s"])
        null_rows.append(
            _row(
                prompt,
                layer=None,
                alpha=0.0,
                direction_id="none",
                s_null=float(measured["s"]),
                measured=measured,
            )
        )
    null_instrument_rows = []
    null_abs = []
    for layer in layers:
        for prompt in grouped["discovery"]:
            measured = measure(
                prompt,
                hook=True,
                layer=layer,
                alpha=0.0,
                direction_id="primary",
            )
            row = _row(
                prompt,
                layer=layer,
                alpha=0.0,
                direction_id="primary",
                s_null=s_null[prompt.prompt_id],
                measured=measured,
            )
            null_instrument_rows.append(row)
            null_abs.append(abs(row["actual_delta"]))
            if not np.isfinite(measured["s"]):
                finite = False
    max_null_abs = float(np.max(null_abs)) if null_abs else 0.0
    null_pass = bool(finite and max_null_abs <= float(config["null_bug_threshold"]))
    null_hook_ok = all(_hook_ok(row, hook_name(int(row["layer"]))) for row in null_instrument_rows)
    no_hook_ok = all(_hook_ok(row, None) for row in null_rows)
    discovery_rows = []
    discovery_means: dict[int, float] = {}
    selected_layer = None
    validation = None
    replication = None
    if null_pass and null_hook_ok and no_hook_ok and leakage_pass:
        called_ids: list[str] = []
        for layer in layers:
            layer_rows = []
            for prompt in grouped["discovery"]:
                called_ids.append(prompt.prompt_id)
                measured = measure(
                    prompt,
                    hook=True,
                    layer=layer,
                    alpha=primary,
                    direction_id="primary",
                )
                if not np.isfinite(measured["s"]):
                    finite = False
                layer_rows.append(
                    _row(
                        prompt,
                        layer=layer,
                        alpha=primary,
                        direction_id="primary",
                        s_null=s_null[prompt.prompt_id],
                        measured=measured,
                    )
                )
            discovery_rows.extend(layer_rows)
            discovery_means[layer] = _mean(layer_rows)
        discovery_ids = {prompt.prompt_id for prompt in grouped["discovery"]}
        if set(called_ids) - discovery_ids:
            raise RuntimeError("Candidate selection observed a non-discovery prompt.")
        selected_layer = select_layer(discovery_means)
        if on_freeze is not None:
            on_freeze(
                {
                    "selected_on": "discovery",
                    "selected_layer": int(selected_layer),
                    "selection_alpha": float(primary),
                    "selection_direction": "primary",
                    "discovery_means": {
                        str(layer): float(discovery_means[layer]) for layer in layers
                    },
                    "candidate_layers": list(layers),
                }
            )
        validation = _evaluate_grid(
            grouped["validation"],
            role="validation",
            layer=selected_layer,
            grid=grid,
            measure=measure,
            s_null=s_null,
            primary_alpha=primary,
        )
        replication = _evaluate_grid(
            grouped["replication"],
            role="replication",
            layer=selected_layer,
            grid=grid,
            measure=measure,
            s_null=s_null,
            primary_alpha=primary,
        )
        finite = finite and all(
            np.isfinite(row["actual_delta"])
            for block in (validation, replication)
            for row in block["rows"] + block["control_rows"]
        )
    if validation is not None and replication is not None:
        for block in (validation, replication):
            for row in block["rows"]:
                if float(row["alpha"]) == 0.0:
                    null_abs.append(abs(float(row["actual_delta"])))
        max_null_abs = float(np.max(null_abs))
        if max_null_abs > float(config["null_bug_threshold"]):
            null_pass = False
    floor = effect_floor(max_null_abs, float(config["effect_floor_abs"]), float(config["effect_floor_mult"]))
    hook_integrity_pass = bool(no_hook_ok and null_hook_ok)
    if validation is not None and replication is not None:
        hook_integrity_pass = hook_integrity_pass and validation["hook_integrity_pass"] and replication["hook_integrity_pass"]
        discovery_mean = float(discovery_means[selected_layer])
        validation_mean = float(validation["primary_mean"])
        replication_mean = float(replication["primary_mean"])
        validation_control_mean = float(validation["control_mean"])
        replication_control_mean = float(replication["control_mean"])
        validation_paired = float(validation["paired_difference_mean"])
        replication_paired = float(replication["paired_difference_mean"])
    else:
        discovery_mean = 0.0
        validation_mean = 0.0
        replication_mean = 0.0
        validation_control_mean = 0.0
        replication_control_mean = 0.0
        validation_paired = 0.0
        replication_paired = 0.0
        finite = False if not null_pass else finite
    evidence = {
        "null_pass": null_pass and null_hook_ok and no_hook_ok,
        "hook_integrity_pass": hook_integrity_pass,
        "split_integrity_pass": True,
        "leakage_pass": bool(leakage_pass),
        "definition_reproducible": bool(definition_reproducible),
        "scores_finite": bool(finite),
        "revision_pinned": bool(revision_pinned),
        "floor": floor,
        "discovery_mean": discovery_mean,
        "validation_mean": validation_mean,
        "replication_mean": replication_mean,
        "validation_control_mean": validation_control_mean,
        "replication_control_mean": replication_control_mean,
        "validation_paired_difference_mean": validation_paired,
        "replication_paired_difference_mean": replication_paired,
    }
    status = classify_status(evidence)
    directionality = None
    dose = None
    detectability = None
    if validation is not None:
        directionality = {
            "validation": classify_directionality(
                validation["positive_deltas"],
                validation["negative_deltas"],
                floor,
                float(config["directionality"]["clear_min_opposite_fraction"]),
                float(config["directionality"]["partial_min_opposite_fraction"]),
            ),
            "replication": classify_directionality(
                replication["positive_deltas"],
                replication["negative_deltas"],
                floor,
                float(config["directionality"]["clear_min_opposite_fraction"]),
                float(config["directionality"]["partial_min_opposite_fraction"]),
            ),
        }
        dose = {
            "validation": spearman_alpha_response(validation["alpha_means"]),
            "replication": spearman_alpha_response(replication["alpha_means"]),
        }
        detectability = {
            "validation": {
                **bootstrap_mean_interval(
                    validation["positive_deltas"],
                    int(config["bootstrap_draws"]),
                    int(config["bootstrap_seed"]),
                ),
                "fraction_beyond_floor": fraction_beyond_floor(validation["positive_deltas"], floor),
                "per_regime": validation["per_regime_primary"],
                "per_prompt": validation["per_prompt_primary"],
                "wilcoxon": wilcoxon_versus_zero(validation["positive_deltas"]),
            },
            "replication": {
                **bootstrap_mean_interval(
                    replication["positive_deltas"],
                    int(config["bootstrap_draws"]),
                    int(config["bootstrap_seed"]) + 1,
                ),
                "fraction_beyond_floor": fraction_beyond_floor(replication["positive_deltas"], floor),
                "per_regime": replication["per_regime_primary"],
                "per_prompt": replication["per_prompt_primary"],
                "wilcoxon": wilcoxon_versus_zero(replication["positive_deltas"]),
            },
            "discovery": {
                "means_by_layer": {str(layer): float(discovery_means[layer]) for layer in layers},
                "selected_layer": selected_layer,
                "selected_mean": discovery_mean,
                "per_prompt": [row for row in discovery_rows if row["layer"] == selected_layer],
            },
        }
    return {
        "status": status,
        "evidence": evidence,
        "candidate_layers": layers,
        "selected_layer": selected_layer,
        "discovery_means": {str(layer): float(value) for layer, value in discovery_means.items()},
        "null": {
            "max_abs": max_null_abs,
            "mean_abs": float(np.mean(null_abs)) if null_abs else None,
            "n": int(len(null_abs)),
            "threshold": float(config["null_bug_threshold"]),
            "pass": bool(evidence["null_pass"]),
            "rows": null_instrument_rows,
        },
        "discovery_rows": discovery_rows,
        "validation": validation,
        "replication": replication,
        "detectability": detectability,
        "dose_response": dose,
        "directionality": directionality,
        "floor": floor,
    }
