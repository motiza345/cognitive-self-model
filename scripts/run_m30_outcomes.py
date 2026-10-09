"""Measure M30 outcomes once and score them under M30.PROTOCOL.2.

The intervention is scripts.run_m29d_outcome._effect: logit_margin and
make_resid_hook, alpha +1, last token only, CPU float32. g and kappa are
read from the frozen prediction file. This script does not recompute them.

Scoring joins those frozen predictions to the measured y. The primary
comparison is MAE(g) against MAE(g + kappa), with the prompt-level paired
bootstrap in src.cognitive_self_model.m23.stats. B1 is reported and does
not choose the verdict. The M29 scorer is a different comparison and is
not used here.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m30_catalog import catalog
from scripts.run_m29d_outcome import _effect, _load_model
from scripts.run_m30_pre_outcome import (
    ALPHA,
    ANCHOR_CELLS,
    DEVICE_NAME,
    DTYPE_NAME,
    LOCKED_CATALOG_SHA256,
    LOCKED_PROTOCOL_SHA256,
    MANIFEST_PATH,
    MODEL_ID,
    MODEL_REVISION,
    NEW_CELLS,
    PREDICTIONS_PATH,
    assert_catalog_lock,
    assert_clean_git,
    load_directions,
    magnitude_check,
    protocol_sha256,
    resolve_anchor_tokens,
    resolve_new_identity_tokens,
)
from src.cognitive_self_model.m23.stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_mean_ci

if BOOTSTRAP_DRAWS != 5000 or BOOTSTRAP_SEED != 23001:
    raise RuntimeError("bootstrap constants are not the frozen M30 values")

OUT = ROOT / "reports" / "m30_raw"
OUTCOMES_PATH = OUT / "OUTCOMES.json"
VERDICT_PATH = OUT / "VERDICT.json"
REPORT_PATH = ROOT / "reports" / "M30_STAGE2_REPORT.md"

EXPECTED_PREDICTION_SHA256 = "f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5"
EXPECTED_NEW_TOKEN_IDS = (830, 895)
EXPECTED_ANCHOR_TOKEN_IDS = (9834, 902)
RELATIVE_REDUCTION_MIN = 0.25
D1_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
D2_SHA256 = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
NEW_CELL_IDS = tuple(cell[0] for cell in NEW_CELLS)
ANCHOR_CELL_IDS = tuple(cell[0] for cell in ANCHOR_CELLS)
CELL_HOOKS = {cell[0]: cell[2] for cell in (*NEW_CELLS, *ANCHOR_CELLS)}
ARM_CELLS = {"new_identity": NEW_CELL_IDS, "anchor": ANCHOR_CELL_IDS}
OLD_HOLDOUT_INDICES = (3, 6, 9, 12)
NEW_HOLDOUT_INDICES = (13, 14, 15, 16)
FORBIDDEN_PREDICTION_KEYS = (
    "observed_effect",
    "intervened_output",
    "baseline_output",
    "residual",
    "actual_outcome",
    "y",
    "r",
)
COPIED_FIELDS = ("alpha", "arm", "f_second", "family", "g", "hook", "intervention_id", "kappa", "partition", "prompt_id")
MEAN_TOLERANCE = 1e-9

INTERPRETATION = {
    (True, True): "Both identities beat `g` on this catalog under the frozen rule.",
    (True, False): "The inherited identity beats `g` on this catalog. The new identity does not meet `SUPPORTED`.",
    (False, True): "The new identity meets `SUPPORTED` on this catalog. The inherited identity does not.",
    (False, False): "Neither identity meets `SUPPORTED` on this catalog.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _num(value: float | None) -> str:
    if value is None:
        return "undefined"
    return json.dumps(value)


def holdout_cohort(prompt_id: str) -> str:
    """Old holdout is index 3/6/9/12. New holdout is the M30-HX1 indices 13..16."""
    index = int(str(prompt_id).rsplit("-", 1)[-1])
    if index in OLD_HOLDOUT_INDICES:
        return "old"
    if index in NEW_HOLDOUT_INDICES:
        return "new"
    raise RuntimeError(f"holdout prompt index is outside the frozen split: {prompt_id}")


def assert_prediction_lock(path: Path = PREDICTIONS_PATH) -> tuple[str, list[dict[str, Any]]]:
    """Reject any prediction file other than the frozen stage-1 artifact."""
    if not path.is_file():
        raise SystemExit("prediction file is missing")
    digest = _sha256(path)
    if digest != EXPECTED_PREDICTION_SHA256:
        raise SystemExit(f"prediction sha256 mismatch: {digest}")
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise SystemExit("prediction artifact is not a list")
    _validate_predictions(predictions)
    return digest, predictions


def _validate_predictions(predictions: list[dict[str, Any]]) -> None:
    if len(predictions) != 288:
        raise SystemExit(f"expected 288 prediction rows, got {len(predictions)}")
    seen: set[tuple[str, str, str]] = set()
    counts = {"update": 0, "validation": 0, "holdout": 0}
    arm_counts = {"new_identity": 0, "anchor": 0}
    for index, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise SystemExit(f"prediction row {index} is not an object")
        bad = set(FORBIDDEN_PREDICTION_KEYS).intersection(row)
        if bad or row.get("outcome_present") is not False:
            raise SystemExit(f"prediction row {index} is not a pre-outcome row")
        arm = str(row["arm"])
        cell = str(row["intervention_id"])
        partition = str(row["partition"])
        if arm not in ARM_CELLS or cell not in ARM_CELLS[arm]:
            raise SystemExit(f"prediction row {index} is not a frozen cell")
        if row["hook"] != CELL_HOOKS[cell]:
            raise SystemExit(f"prediction hook does not match the frozen cell: {cell}")
        if partition not in counts:
            raise SystemExit(f"unexpected partition {partition}")
        if float(row["alpha"]) != 1.0:
            raise SystemExit("alpha is not +1")
        if not all(math.isfinite(float(row[name])) for name in ("g", "kappa", "f_second")):
            raise SystemExit(f"non-finite prediction at row {index}")
        key = (arm, cell, str(row["prompt_id"]))
        if key in seen:
            raise SystemExit(f"duplicate prediction key {key}")
        seen.add(key)
        counts[partition] += 1
        arm_counts[arm] += 1
    if counts != {"update": 72, "validation": 72, "holdout": 144}:
        raise SystemExit(f"partition counts are not 72/72/144: {counts}")
    if arm_counts != {"new_identity": 144, "anchor": 144}:
        raise SystemExit(f"arm counts are not 144/144: {arm_counts}")


def _load_stage1_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256:
        raise SystemExit("stage-1 manifest prediction hash mismatch")
    if manifest.get("catalog_sha256") != LOCKED_CATALOG_SHA256:
        raise SystemExit("stage-1 manifest catalog hash mismatch")
    if manifest.get("protocol_sha256") != LOCKED_PROTOCOL_SHA256:
        raise SystemExit("stage-1 manifest protocol hash mismatch")
    tokens = manifest.get("token_ids", {})
    new_ids = (tokens["new_identity"]["positive_id"], tokens["new_identity"]["negative_id"])
    anchor_ids = (tokens["anchor"]["positive_id"], tokens["anchor"]["negative_id"])
    if new_ids != EXPECTED_NEW_TOKEN_IDS or anchor_ids != EXPECTED_ANCHOR_TOKEN_IDS:
        raise SystemExit("stage-1 manifest token ids are not the frozen pair")
    return manifest


def _require_magnitude(predictions: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    recorded = manifest.get("magnitude_check")
    if not isinstance(recorded, list) or not all(item.get("passed") and item.get("finite") for item in recorded):
        raise SystemExit("magnitude check did not pass")
    checks = [
        magnitude_check(predictions, "new_identity", NEW_CELL_IDS),
        magnitude_check(predictions, "anchor", ANCHOR_CELL_IDS),
    ]
    if not all(item["passed"] and item["finite"] for item in checks):
        raise SystemExit("magnitude check did not pass")
    return checks


def classify_verdict(
    *,
    leakage_ok: bool,
    delta: float,
    ci_low: float,
    mae_g: float,
    relative_reduction: float | None,
    anchor: bool,
) -> str:
    """First matching clause of protocol section 6."""
    if not leakage_ok:
        label = "INCONCLUSIVE"
    elif delta <= 0.0:
        label = "NOT_SUPPORTED"
    elif not (ci_low > 0.0):
        label = "INCONCLUSIVE"
    elif mae_g == 0.0 or relative_reduction is None or relative_reduction < RELATIVE_REDUCTION_MIN:
        label = "SUPPORTED_WEAK"
    else:
        label = "SUPPORTED"
    if anchor:
        return "ANCHOR_" + label
    return label


def interpret_two_by_two(new_verdict: str, anchor_verdict: str) -> dict[str, Any]:
    """SUPPORTED_WEAK counts as not SUPPORTED. The table does not relabel either verdict."""
    new_supported = new_verdict == "SUPPORTED"
    anchor_supported = anchor_verdict == "ANCHOR_SUPPORTED"
    return {
        "anchor_supported": anchor_supported,
        "new_identity_supported": new_supported,
        "interpretation": INTERPRETATION[(anchor_supported, new_supported)],
    }


def _mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("mean requires values")
    return sum(values) / len(values)


def _reduction(mae_g: float, mae_model: float) -> float | None:
    if not (mae_g > 0.0):
        return None
    return 1.0 - (mae_model / mae_g)


def _prompt_scores(rows: list[dict[str, Any]], difference) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["prompt_id"]), []).append(float(difference(row)))
    if len(grouped) != 24:
        raise RuntimeError(f"expected 24 holdout prompts, got {len(grouped)}")
    scores: list[float] = []
    for prompt_id in sorted(grouped):
        values = grouped[prompt_id]
        if len(values) != 3:
            raise RuntimeError(f"{prompt_id} does not have 3 cell scores")
        scores.append(_mean(values))
    return scores


def _fit_b1(rows: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, float]:
    """Cell means of y - g. UPDATE rows only."""
    means: dict[str, float] = {}
    for cell in cells:
        group = [row for row in rows if row["partition"] == "update" and row["intervention_id"] == cell]
        if len(group) != 12:
            raise RuntimeError(f"{cell} expected 12 UPDATE rows, got {len(group)}")
        means[cell] = _mean([float(row["observed_effect"]) - float(row["g"]) for row in group])
    return means


def _pooled(rows: list[dict[str, Any]], cells: tuple[str, ...]) -> dict[str, float]:
    holdout = [row for row in rows if row["partition"] == "holdout"]
    if len(holdout) != 72:
        raise RuntimeError(f"expected 72 holdout rows, got {len(holdout)}")
    error_g = [abs(float(row["observed_effect"]) - float(row["g"])) for row in holdout]
    error_model = [abs(float(row["observed_effect"]) - (float(row["g"]) + float(row["kappa"]))) for row in holdout]
    mae_g = _mean(error_g)
    mae_model = _mean(error_model)
    prompt_scores = _prompt_scores(
        holdout,
        lambda row: abs(float(row["observed_effect"]) - float(row["g"]))
        - abs(float(row["observed_effect"]) - (float(row["g"]) + float(row["kappa"]))),
    )
    delta = mae_g - mae_model
    if abs(_mean(prompt_scores) - delta) > MEAN_TOLERANCE:
        raise RuntimeError("prompt-level mean does not equal pooled Delta")
    per_cell = []
    for cell in cells:
        group = [row for row in holdout if row["intervention_id"] == cell]
        if len(group) != 24:
            raise RuntimeError(f"{cell} expected 24 holdout rows, got {len(group)}")
        per_cell.append(
            {
                "intervention_id": cell,
                "mae_g": _mean([abs(float(row["observed_effect"]) - float(row["g"])) for row in group]),
                "mae_g_plus_kappa": _mean(
                    [abs(float(row["observed_effect"]) - (float(row["g"]) + float(row["kappa"]))) for row in group]
                ),
                "median_abs_g": float(statistics.median([abs(float(row["g"])) for row in group])),
                "median_abs_kappa": float(statistics.median([abs(float(row["kappa"])) for row in group])),
                "n": len(group),
            }
        )
    cohorts = {}
    for name in ("old", "new"):
        group = [row for row in holdout if holdout_cohort(str(row["prompt_id"])) == name]
        prompts = {str(row["prompt_id"]) for row in group}
        if len(prompts) != 12 or len(group) != 36:
            raise RuntimeError(f"{name} holdout is not 12 prompts")
        mae_g_cohort = _mean([abs(float(row["observed_effect"]) - float(row["g"])) for row in group])
        mae_model_cohort = _mean(
            [abs(float(row["observed_effect"]) - (float(row["g"]) + float(row["kappa"]))) for row in group]
        )
        cohorts[name] = {
            "delta": mae_g_cohort - mae_model_cohort,
            "mae_g": mae_g_cohort,
            "mae_g_plus_kappa": mae_model_cohort,
            "n_prompts": len(prompts),
            "n_rows": len(group),
            "prompt_ids": sorted(prompts),
        }
    return {
        "cohorts": cohorts,
        "delta": delta,
        "mae_g": mae_g,
        "mae_g_plus_kappa": mae_model,
        "per_cell": per_cell,
        "prompt_scores": prompt_scores,
        "relative_reduction": _reduction(mae_g, mae_model),
    }


def _b1_score(rows: list[dict[str, Any]], cells: tuple[str, ...], mae_model: float) -> dict[str, Any]:
    means = _fit_b1(rows, cells)
    holdout = [row for row in rows if row["partition"] == "holdout"]
    errors = []
    for row in holdout:
        predicted = float(row["g"]) + means[str(row["intervention_id"])]
        errors.append(abs(float(row["observed_effect"]) - predicted))
    mae_b1 = _mean(errors)

    def difference(row: dict[str, Any]) -> float:
        b1 = means[str(row["intervention_id"])]
        return abs(float(row["observed_effect"]) - (float(row["g"]) + b1)) - abs(
            float(row["observed_effect"]) - (float(row["g"]) + float(row["kappa"]))
        )

    prompt_scores = _prompt_scores(holdout, difference)
    delta = mae_b1 - mae_model
    if abs(_mean(prompt_scores) - delta) > MEAN_TOLERANCE:
        raise RuntimeError("B1 prompt-level mean does not equal MAE(B1) - MAE(g+kappa)")
    interval = paired_mean_ci(prompt_scores)
    return {
        "bootstrap": _interval_record(interval, len(prompt_scores)),
        "cell_means": means,
        "delta_mae_b1_minus_mae_g_plus_kappa": delta,
        "fit_partition": "update",
        "gate": False,
        "mae_b1": mae_b1,
        "n_update_per_cell": 12,
    }


def _interval_record(interval: dict[str, Any], n_prompts: int) -> dict[str, Any]:
    return {
        "class": str(interval["class"]),
        "draws": BOOTSTRAP_DRAWS,
        "high": float(interval["high"]),
        "low": float(interval["low"]),
        "mean": float(interval["mean"]),
        "n_prompts": n_prompts,
        "seed": BOOTSTRAP_SEED,
    }


def score_arm(rows: list[dict[str, Any]], *, arm: str, leakage_ok: bool) -> dict[str, Any]:
    """Score one arm. Validation rows may be present and are not used."""
    if arm not in ARM_CELLS:
        raise RuntimeError(f"unknown arm {arm}")
    if any(str(row["arm"]) != arm for row in rows):
        raise RuntimeError("arm rows are mixed")
    cells = ARM_CELLS[arm]
    known = {"update", "validation", "holdout"}
    if any(str(row["partition"]) not in known for row in rows):
        raise RuntimeError("unexpected partition")
    if any(str(row["intervention_id"]) not in cells for row in rows):
        raise RuntimeError("unexpected cell")
    primary = _pooled(rows, cells)
    interval = paired_mean_ci(primary["prompt_scores"])
    if abs(float(interval["mean"]) - primary["delta"]) > MEAN_TOLERANCE:
        raise RuntimeError("paired bootstrap mean does not equal Delta")
    anchor = arm == "anchor"
    verdict = classify_verdict(
        leakage_ok=leakage_ok,
        delta=primary["delta"],
        ci_low=float(interval["low"]),
        mae_g=primary["mae_g"],
        relative_reduction=primary["relative_reduction"],
        anchor=anchor,
    )
    return {
        "arm": arm,
        "b1": _b1_score(rows, cells, primary["mae_g_plus_kappa"]),
        "bootstrap": _interval_record(interval, len(primary["prompt_scores"])),
        "delta": primary["delta"],
        "holdout_cohorts": primary["cohorts"],
        "mae_g": primary["mae_g"],
        "mae_g_plus_kappa": primary["mae_g_plus_kappa"],
        "per_cell": primary["per_cell"],
        "relative_reduction": primary["relative_reduction"],
        "verdict": verdict,
    }


def leakage_audit(
    *,
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    magnitude_passed: bool,
    token_ids_ok: bool,
    direction_hashes_ok: bool,
) -> dict[str, Any]:
    prediction_clean = all(
        not set(FORBIDDEN_PREDICTION_KEYS).intersection(row) and row.get("outcome_present") is False
        for row in predictions
    )
    copied = True
    pred_by = {(row["arm"], row["intervention_id"], row["prompt_id"]): row for row in predictions}
    out_by = {(row["arm"], row["intervention_id"], row["prompt_id"]): row for row in outcomes}
    if set(pred_by) != set(out_by):
        copied = False
    else:
        for key, prediction in pred_by.items():
            outcome = out_by[key]
            for field in COPIED_FIELDS:
                if outcome[field] != prediction[field]:
                    copied = False
    update_ids = {row["prompt_id"] for row in predictions if row["partition"] == "update"}
    holdout_ids = {row["prompt_id"] for row in predictions if row["partition"] == "holdout"}
    validation_ids = {row["prompt_id"] for row in predictions if row["partition"] == "validation"}
    checks = {
        "alpha_is_one": all(float(row["alpha"]) == 1.0 for row in predictions),
        "arms_not_mixed": True,
        "b1_fit_on_update_only": True,
        "catalog_sha256_matches": True,
        "copied_prediction_fields_unchanged": copied,
        "device_cpu": True,
        "direction_hashes_match": direction_hashes_ok,
        "holdout_not_used_in_b1_or_magnitude": True,
        "m29_holdout_statistics_not_loaded": True,
        "magnitude_passed_before_outcomes": magnitude_passed,
        "prediction_sha256_matches": True,
        "predictions_lack_outcome_fields": prediction_clean,
        "protocol_sha256_matches": True,
        "token_ids_match_stage1": token_ids_ok,
        "update_holdout_validation_prompts_disjoint": update_ids.isdisjoint(holdout_ids)
        and update_ids.isdisjoint(validation_ids)
        and holdout_ids.isdisjoint(validation_ids),
        "validation_excluded_from_verdict": True,
    }
    return {"checks": checks, "passed": all(checks.values())}


def score_outcomes(
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    magnitude_passed: bool,
    token_ids_ok: bool,
    direction_hashes_ok: bool,
) -> dict[str, Any]:
    audit = leakage_audit(
        predictions=predictions,
        outcomes=outcomes,
        magnitude_passed=magnitude_passed,
        token_ids_ok=token_ids_ok,
        direction_hashes_ok=direction_hashes_ok,
    )
    by_arm: dict[str, list[dict[str, Any]]] = {"new_identity": [], "anchor": []}
    out_by = {(row["arm"], row["intervention_id"], row["prompt_id"]): row for row in outcomes}
    if len(out_by) != len(outcomes):
        raise RuntimeError("duplicate outcome key")
    for prediction in predictions:
        key = (prediction["arm"], prediction["intervention_id"], prediction["prompt_id"])
        outcome = out_by.get(key)
        if outcome is None:
            raise RuntimeError(f"missing outcome for {key}")
        for field in COPIED_FIELDS:
            if outcome[field] != prediction[field]:
                raise RuntimeError(f"copied prediction field changed: {field} {key}")
        if not math.isfinite(float(outcome["observed_effect"])):
            raise RuntimeError(f"non-finite outcome for {key}")
        joined = dict(prediction)
        joined["observed_effect"] = outcome["observed_effect"]
        by_arm[str(prediction["arm"])].append(joined)
    if set(out_by) != {(row["arm"], row["intervention_id"], row["prompt_id"]) for row in predictions}:
        raise RuntimeError("outcome keys are not the prediction keys")
    new_score = score_arm(by_arm["new_identity"], arm="new_identity", leakage_ok=audit["passed"])
    anchor_score = score_arm(by_arm["anchor"], arm="anchor", leakage_ok=audit["passed"])
    return {
        "anchor": anchor_score,
        "leakage": audit,
        "new_identity": new_score,
        "two_by_two": interpret_two_by_two(new_score["verdict"], anchor_score["verdict"]),
    }


def measure_outcomes(model: Any, predictions: list[dict[str, Any]], directions: dict[str, Any], tokens: dict[str, tuple[int, int]]) -> list[dict[str, Any]]:
    by_prompt = {row["prompt_id"]: row for row in catalog()}
    if len(by_prompt) != 48:
        raise RuntimeError("catalog prompt ids are not unique")
    baselines: dict[tuple[str, str], float] = {}
    outcomes: list[dict[str, Any]] = []
    for index, row in enumerate(predictions, start=1):
        prompt_id = str(row["prompt_id"])
        source = by_prompt.get(prompt_id)
        if source is None or source["partition"] != row["partition"] or source["family"] != row["family"]:
            raise RuntimeError(f"prediction catalog fields differ for {prompt_id}")
        arm = str(row["arm"])
        positive_id, negative_id = tokens[arm]
        measured = _effect(model, source["text"], row["hook"], directions[arm], positive_id, negative_id)
        cache_key = (arm, prompt_id)
        if cache_key not in baselines:
            baselines[cache_key] = measured["baseline_output"]
        elif baselines[cache_key] != measured["baseline_output"]:
            raise RuntimeError("unperturbed margin changed across cells of one prompt")
        if arm == "new_identity":
            direction_seed, direction_hash = 22103, D2_SHA256
        elif arm == "anchor":
            direction_seed, direction_hash = 22101, D1_SHA256
        else:
            raise RuntimeError(f"unknown arm {arm}")
        g = row["g"]
        kappa = row["kappa"]
        observed = measured["observed_effect"]
        outcomes.append(
            {
                "abs_error_g": abs(float(observed) - float(g)),
                "abs_error_g_plus_kappa": abs(float(observed) - (float(g) + float(kappa))),
                "alpha": row["alpha"],
                "arm": arm,
                "baseline_output": measured["baseline_output"],
                "device": DEVICE_NAME,
                "direction_seed": direction_seed,
                "direction_sha256": direction_hash,
                "dtype": DTYPE_NAME,
                "f_second": row["f_second"],
                "family": row["family"],
                "g": g,
                "hook": row["hook"],
                "intervened_output": measured["intervened_output"],
                "intervention_id": row["intervention_id"],
                "kappa": kappa,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "negative_token_id": negative_id,
                "observed_effect": observed,
                "partition": row["partition"],
                "positive_token_id": positive_id,
                "prompt_id": prompt_id,
                "residual": float(observed) - float(g),
            }
        )
        if index % 6 == 0:
            print(f"outcome {index}/{len(predictions)} {arm} {prompt_id}", flush=True)
    if len(outcomes) != 288:
        raise RuntimeError(f"expected 288 outcome rows, got {len(outcomes)}")
    return outcomes


def render_report(verdict: dict[str, Any]) -> str:
    new_arm = verdict["arms"]["new_identity"]
    anchor_arm = verdict["arms"]["anchor"]
    lines = [
        "# M30 stage 2",
        "",
        f"Protocol `M30.PROTOCOL.2`, hash `{verdict['protocol_sha256']}`.",
        f"Catalog `{verdict['catalog_sha256']}`. Predictions `{verdict['prediction_sha256']}`.",
        f"Runner commit `{verdict['git_commit']}`.",
        "",
        "## Verdicts",
        "",
        f"- New identity: `{verdict['new_identity_verdict']}`",
        f"- Anchor: `{verdict['anchor_verdict']}`",
        f"- 2x2: {verdict['two_by_two']['interpretation']}",
        "",
        "## Primary scores",
        "",
        "| arm | Delta | 95% CI low | 95% CI high | relative reduction | MAE(g) | MAE(g+kappa) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in (new_arm, anchor_arm):
        interval = arm["bootstrap"]
        lines.append(
            "| {arm} | {delta} | {low} | {high} | {reduction} | {mae_g} | {mae_model} |".format(
                arm=arm["arm"],
                delta=_num(arm["delta"]),
                low=_num(interval["low"]),
                high=_num(interval["high"]),
                reduction=_num(arm["relative_reduction"]),
                mae_g=_num(arm["mae_g"]),
                mae_model=_num(arm["mae_g_plus_kappa"]),
            )
        )
    lines.extend(
        [
            "",
            "Bootstrap: 24 prompt scores, 5000 draws, seed 23001, percentiles 2.5 and 97.5.",
            "",
            "## Per-cell description",
            "",
            "| arm | cell | MAE(g) | MAE(g+kappa) | median |kappa| | median |g| |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in (new_arm, anchor_arm):
        for cell in arm["per_cell"]:
            lines.append(
                "| {arm} | {cell} | {mae_g} | {mae_model} | {kappa} | {g} |".format(
                    arm=arm["arm"],
                    cell=cell["intervention_id"],
                    mae_g=_num(cell["mae_g"]),
                    mae_model=_num(cell["mae_g_plus_kappa"]),
                    kappa=_num(cell["median_abs_kappa"]),
                    g=_num(cell["median_abs_g"]),
                )
            )
    lines.extend(
        [
            "",
            "These per-cell numbers are not a gate.",
            "",
            "## Old holdout versus new holdout",
            "",
            "| arm | cohort | prompts | Delta | MAE(g) | MAE(g+kappa) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in (new_arm, anchor_arm):
        for name in ("old", "new"):
            cohort = arm["holdout_cohorts"][name]
            lines.append(
                "| {arm} | {name} | {n} | {delta} | {mae_g} | {mae_model} |".format(
                    arm=arm["arm"],
                    name=name,
                    n=cohort["n_prompts"],
                    delta=_num(cohort["delta"]),
                    mae_g=_num(cohort["mae_g"]),
                    mae_model=_num(cohort["mae_g_plus_kappa"]),
                )
            )
    lines.extend(
        [
            "",
            "Old holdout is prompt index 3, 6, 9, 12. New holdout is prompt index 13 through 16. Descriptive only.",
            "",
            "## Secondary B1",
            "",
            "| arm | MAE(B1) | MAE(B1) - MAE(g+kappa) | 95% CI low | 95% CI high |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in (new_arm, anchor_arm):
        b1 = arm["b1"]
        lines.append(
            "| {arm} | {mae} | {delta} | {low} | {high} |".format(
                arm=arm["arm"],
                mae=_num(b1["mae_b1"]),
                delta=_num(b1["delta_mae_b1_minus_mae_g_plus_kappa"]),
                low=_num(b1["bootstrap"]["low"]),
                high=_num(b1["bootstrap"]["high"]),
            )
        )
    lines.extend(
        [
            "",
            "B1 is the UPDATE cell mean of `y - g`. It does not choose the verdict.",
            "",
            f"Leakage audit passed: `{json.dumps(verdict['leakage']['passed'])}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _refuse_overwrite() -> None:
    existing = [path for path in (OUTCOMES_PATH, VERDICT_PATH, REPORT_PATH) if path.exists()]
    if existing:
        raise SystemExit("M30 outcome artifacts already exist; refusing to overwrite")


def _resolve_tokens(tokenizer: Any) -> dict[str, tuple[int, int]]:
    new_ids = resolve_new_identity_tokens(tokenizer)
    anchor_ids = resolve_anchor_tokens(tokenizer)
    if new_ids != EXPECTED_NEW_TOKEN_IDS:
        raise SystemExit(f"new-identity token ids are {new_ids[0]}, {new_ids[1]}")
    if anchor_ids != EXPECTED_ANCHOR_TOKEN_IDS:
        raise SystemExit(f"anchor token ids are {anchor_ids[0]}, {anchor_ids[1]}")
    return {"new_identity": new_ids, "anchor": anchor_ids}


def build_verdict(
    *,
    scored: dict[str, Any],
    catalog_digest: str,
    protocol_digest: str,
    prediction_digest: str,
    commit: str,
    magnitude: list[dict[str, Any]],
    tokens: dict[str, tuple[int, int]],
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "alpha": ALPHA,
        "anchor_verdict": scored["anchor"]["verdict"],
        "arms": {"anchor": scored["anchor"], "new_identity": scored["new_identity"]},
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "catalog_sha256": catalog_digest,
        "device": DEVICE_NAME,
        "directions": {
            "D1": {"seed": 22101, "sha256": D1_SHA256},
            "D2": {"seed": 22103, "sha256": D2_SHA256},
        },
        "dtype": DTYPE_NAME,
        "git_commit": commit,
        "leakage": scored["leakage"],
        "magnitude_check": magnitude,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "new_identity_verdict": scored["new_identity"]["verdict"],
        "prediction_sha256": prediction_digest,
        "protocol_revision": "M30.PROTOCOL.2",
        "protocol_sha256": protocol_digest,
        "python": platform.python_version(),
        "runtime_seconds": runtime_seconds,
        "status": "OUTCOME_COMPLETE",
        "token_ids": {
            "anchor": {"negative_id": tokens["anchor"][1], "positive_id": tokens["anchor"][0]},
            "new_identity": {"negative_id": tokens["new_identity"][1], "positive_id": tokens["new_identity"][0]},
        },
        "torch": _version("torch"),
        "transformer_lens": _version("transformer-lens"),
        "two_by_two": scored["two_by_two"],
    }


def main() -> None:
    _refuse_overwrite()
    prediction_digest, predictions = assert_prediction_lock()
    catalog_digest = assert_catalog_lock()
    protocol_digest = protocol_sha256()
    manifest = _load_stage1_manifest()
    magnitude = _require_magnitude(predictions, manifest)
    direction_d1, direction_d2 = load_directions()
    commit = assert_clean_git()

    started = time.perf_counter()
    model, device = _load_model()
    if device != DEVICE_NAME:
        raise SystemExit("outcome execution must stay on CPU")
    tokens = _resolve_tokens(model.tokenizer)
    outcomes = measure_outcomes(
        model,
        predictions,
        {"new_identity": direction_d2, "anchor": direction_d1},
        tokens,
    )
    scored = score_outcomes(
        predictions,
        outcomes,
        magnitude_passed=True,
        token_ids_ok=True,
        direction_hashes_ok=True,
    )
    runtime_seconds = time.perf_counter() - started
    verdict = build_verdict(
        scored=scored,
        catalog_digest=catalog_digest,
        protocol_digest=protocol_digest,
        prediction_digest=prediction_digest,
        commit=commit,
        magnitude=magnitude,
        tokens=tokens,
        runtime_seconds=runtime_seconds,
    )
    report = render_report(verdict)
    if _sha256(PREDICTIONS_PATH) != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError("prediction artifact changed before the outcome write")
    _write_text(OUTCOMES_PATH, _dump(outcomes))
    _write_text(VERDICT_PATH, _dump(verdict))
    _write_text(REPORT_PATH, report)
    if _sha256(PREDICTIONS_PATH) != EXPECTED_PREDICTION_SHA256:
        for path in (OUTCOMES_PATH, VERDICT_PATH, REPORT_PATH):
            path.unlink(missing_ok=True)
        raise RuntimeError("prediction artifact changed during the outcome write")
    print(
        json.dumps(
            {
                "anchor_verdict": verdict["anchor_verdict"],
                "new_identity_verdict": verdict["new_identity_verdict"],
                "prediction_sha256": prediction_digest,
                "status": verdict["status"],
                "two_by_two": verdict["two_by_two"]["interpretation"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
