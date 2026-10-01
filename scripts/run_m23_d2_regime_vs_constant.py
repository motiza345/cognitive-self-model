"""Frozen M23-D2 comparison: constant belief versus regime mean.

Evaluation forwards start only after predictions are written.
Replication prompts are not read.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.m22_reuse import frozen_prompts
from src.cognitive_self_model.m23.stats import paired_mean_ci

FROZEN_HASHES = {
    "reports/M23_D2_PREREGISTRATION.md": "b81fbc4b826db8031b08eea20aa5d2ff41c193854611f874cf5c994416abe702",
    "reports/M23_D2_SPLIT_MANIFEST.json": "e456bedce024e4632aa698a74002262a2afec4a134c479e69ee5add60afe2e07",
    "reports/M23_D2_LEAKAGE_AUDIT.md": "8224a98146bf91adde47022ffed427762df0bdd07ad11084d82325872d58ed16",
}
PREDICTION_A = 0.028189738591512043
PREDICTION_B = {
    "completion": 0.028961896896362305,
    "instruction": 0.02886199951171875,
    "syntax": 0.026745319366455078,
}
EVAL_IDS = (
    "completion-01",
    "completion-04",
    "instruction-01",
    "instruction-04",
    "syntax-01",
    "syntax-04",
)
EXCLUDED_IDS = {
    "completion-03",
    "completion-06",
    "instruction-03",
    "instruction-06",
    "syntax-03",
    "syntax-06",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_frozen() -> None:
    for relative, expected in FROZEN_HASHES.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"frozen hash mismatch for {relative}: {actual}")


def _train_predictions() -> None:
    payload = json.loads((ROOT / "reports" / "m23_raw" / "validation_outcomes.json").read_text(encoding="utf-8"))
    rows = payload["d1"]
    if any(row["prompt_id"] in EXCLUDED_IDS or row["prompt_id"] in EVAL_IDS for row in rows):
        raise SystemExit("train file contains evaluation or replication prompts")
    values = [float(row["observed_delta"]) for row in rows]
    constant = sum(values) / len(values)
    if abs(constant - PREDICTION_A) > 1e-15:
        raise SystemExit(f"constant belief drifted: {constant}")
    for regime, expected in PREDICTION_B.items():
        group = [float(row["observed_delta"]) for row in rows if row["regime_id"] == regime]
        if len(group) != 2:
            raise SystemExit(f"regime {regime} train count is {len(group)}")
        got = sum(group) / len(group)
        if abs(got - expected) > 1e-15:
            raise SystemExit(f"regime mean drifted for {regime}: {got}")


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _label(interval_class: str, leakage_ok: bool) -> str:
    if not leakage_ok:
        return "INCONCLUSIVE"
    if interval_class == "CI_POSITIVE":
        return "REGIME_INFORMATION_SUPPORTED"
    if interval_class == "CI_NEGATIVE":
        return "REGIME_FEATURE_HURTS"
    return "INCONCLUSIVE"


def main() -> None:
    _check_frozen()
    _train_predictions()
    out = ROOT / "reports" / "m23_d2_raw"
    prediction_path = out / "eval_predictions.json"
    outcome_path = out / "eval_outcomes.json"
    outcome_path.unlink(missing_ok=True)
    if outcome_path.exists():
        raise SystemExit("evaluation outcomes exist before predictions")

    prompts = {record.prompt_id: record for record in frozen_prompts()}
    rows: list[dict[str, Any]] = []
    for index, prompt_id in enumerate(EVAL_IDS):
        prompt = prompts[prompt_id]
        if prompt.role != "discovery":
            raise SystemExit(f"{prompt_id} is not in the discovery role")
        prediction_b = PREDICTION_B[prompt.regime_id]
        rows.append(
            {
                "episode_id": f"d2-eval-{index:02d}",
                "regime": prompt.regime_id,
                "prompt_id": prompt.prompt_id,
                "seed": 22101,
                "intervention_id": "M22.1-D1-L23",
                "alpha": 1.0,
                "prediction_A": PREDICTION_A,
                "prediction_B": prediction_b,
            }
        )
    by_regime: dict[str, set[float]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], set()).add(row["prediction_B"])
    leakage = {
        "predictions_before_outcomes": True,
        "prediction_B_constant_within_regime": all(len(values) == 1 for values in by_regime.values()),
        "prediction_A_constant_across_regimes": len({row["prediction_A"] for row in rows}) == 1,
        "replication_prompts_absent": not any(row["prompt_id"] in EXCLUDED_IDS for row in rows),
        "no_outcome_field_in_prediction_rows": all("observed_effect" not in row for row in rows),
        "evaluation_prompt_ids_match_manifest": [row["prompt_id"] for row in rows] == list(EVAL_IDS),
    }
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    if outcome_path.exists():
        raise SystemExit("outcomes appeared before the forward")

    runner = _runner()
    model, _device = runner.load_model()
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise SystemExit(f"token ids changed: {tokens['token_ids']}")
    direction = runner.primary_direction(int(model.cfg.d_model), runner.DIRECTION_SEED)
    outcomes = []
    for row in rows:
        prompt = prompts[row["prompt_id"]]
        baseline, _ = runner.forward_margin(
            model, prompt.text, tokens["positive_token_id"], tokens["negative_token_id"], None
        )
        intervened, _recorder = runner.forward_margin(
            model,
            prompt.text,
            tokens["positive_token_id"],
            tokens["negative_token_id"],
            {"alpha": 1.0, "direction": direction},
        )
        observed = float(intervened - baseline)
        outcomes.append({"prompt_id": row["prompt_id"], "observed_effect": observed})
        print(f"{row['prompt_id']} {row['regime']} {observed:.6f}", flush=True)
    outcome_path.write_text(json.dumps({"rows": outcomes}, indent=2, sort_keys=True), encoding="utf-8")
    if prediction_path.stat().st_mtime_ns > outcome_path.stat().st_mtime_ns:
        leakage["predictions_before_outcomes"] = False

    observed_by_id = {row["prompt_id"]: float(row["observed_effect"]) for row in outcomes}
    episodes = []
    differences = []
    for row in rows:
        observed = observed_by_id[row["prompt_id"]]
        error_a = abs(observed - float(row["prediction_A"]))
        error_b = abs(observed - float(row["prediction_B"]))
        difference = error_a - error_b
        differences.append(difference)
        episodes.append({**row, "observed_effect": observed, "abs_error_A": error_a, "abs_error_B": error_b, "paired_difference": difference})

    interval = paired_mean_ci(differences)
    leakage_ok = all(bool(value) for value in leakage.values())
    per_regime: dict[str, dict[str, Any]] = {}
    for regime in ("completion", "instruction", "syntax"):
        group = [episode for episode in episodes if episode["regime"] == regime]
        effects = [float(episode["observed_effect"]) for episode in group]
        center = sum(effects) / len(effects)
        per_regime[regime] = {
            "n": len(group),
            "mean_observed": center,
            "prediction_B": PREDICTION_B[regime],
            "effects": effects,
        }
    effects = [float(episode["observed_effect"]) for episode in episodes]
    effect_mean = sum(effects) / len(effects)
    effect_var = sum((value - effect_mean) ** 2 for value in effects) / (len(effects) - 1)
    b_values = [float(episode["prediction_B"]) for episode in episodes]
    b_mean = sum(b_values) / len(b_values)
    b_var = sum((value - b_mean) ** 2 for value in b_values) / (len(b_values) - 1)
    result_name = _label(str(interval["class"]), leakage_ok)
    payload = {
        "historical_m23_verdict": "INCONCLUSIVE",
        "result": result_name,
        "leakage_ok": leakage_ok,
        "leakage_note": None if leakage_ok else "LEAKAGE",
        "n": len(episodes),
        "per_regime_n": {regime: stats["n"] for regime, stats in per_regime.items()},
        "mean_abs_error_A": sum(episode["abs_error_A"] for episode in episodes) / len(episodes),
        "mean_abs_error_B": sum(episode["abs_error_B"] for episode in episodes) / len(episodes),
        "mean_paired_difference": interval["mean"],
        "interval_low": interval["low"],
        "interval_high": interval["high"],
        "interval_class": interval["class"],
        "interval_width": float(interval["high"]) - float(interval["low"]),
        "paired_differences": differences,
        "effect_sample_variance": effect_var,
        "prediction_B_sample_variance": b_var,
        "prediction_A_variance": 0.0,
        "per_regime": per_regime,
        "leakage": leakage,
        "small_sample": True,
    }
    (ROOT / "reports" / "M23_D2_RESULTS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (ROOT / "reports" / "M23_D2_EPISODES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    print(result_name, interval["class"], flush=True)


if __name__ == "__main__":
    main()
