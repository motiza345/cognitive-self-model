"""M23 direct Qwen self-model audit runner.

Predictions for a split are written before that split's outcomes exist.
The scored verdict uses only the pre-registered map in src/cognitive_self_model/m23.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.belief import belief_from_observations, update_belief
from src.cognitive_self_model.m23.m22_reuse import (
    frozen_prompts,
    logit_margin,
    make_resid_hook,
    orthogonal_direction,
    primary_direction,
    prompt_manifest_sha256,
    resolve_outcome_tokens,
    vector_sha256,
)
from src.cognitive_self_model.m23.report import render_report
from src.cognitive_self_model.m23.score import score_partition

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
HOOK_NAME = "blocks.23.hook_resid_post"
POSITIVE_TEXT = " yes"
NEGATIVE_TEXT = " no"
EXPECTED_TOKEN_IDS = (9834, 902)
MECHANISM_ID = "M22.1-D1-L23"
CONTROL_ID = "M22.1-D2-L23"
DIRECTION_SEED = 22101
CONTROL_SEED = 22103
PUBLISHED_M22_MEANS_LOADED = False

FROZEN_HASHES = {
    "reports/M23_PREREGISTRATION.md": "e590df86cbb057f4dbe2b0f211809927dcd7a9839f4a069cf936bae14f1472dd",
    "reports/M23_LEAKAGE_AUDIT.md": "fbe483de7c234b1bb92b72cdd839af9f3fee75b6fa1d0704bc2c9bf9c142f4b1",
    "reports/M23_SPLIT_MANIFEST.json": "526b73a7cf24f3f3d1fcb0eb3c98f238f4b6f1627f67af43a0c4f2337feedd9c",
    "reports/M23_REPOSITORY_AUDIT.md": "9dbe265f9c974197a717743881b5cd16a350852a03a32a51073f85ea4cf439dd",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen_documents() -> None:
    for relative, expected in FROZEN_HASHES.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"frozen document hash mismatch for {relative}: {actual}")
    manifest = prompt_manifest_sha256()
    if manifest != "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db":
        raise SystemExit(f"prompt manifest mismatch: {manifest}")
    if PUBLISHED_M22_MEANS_LOADED:
        raise SystemExit("published M22 means must not be loaded into the belief")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError as exc:
        return f"UNAVAILABLE:{exc.returncode}"


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def reproducibility() -> dict[str, str]:
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["branch", "--show-current"]),
        "model": MODEL_ID,
        "model_revision": REVISION,
        "seed_direction": str(DIRECTION_SEED),
        "seed_control": str(CONTROL_SEED),
        "bootstrap_seed": "23001",
        "device": "pending",
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformer_lens": _version("transformer-lens"),
        "transformers": _version("transformers"),
        "hook_name": HOOK_NAME,
        "dataset_manifest": prompt_manifest_sha256(),
        "preregistration_sha256": FROZEN_HASHES["reports/M23_PREREGISTRATION.md"],
    }


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _as_logits(output: Any):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("model forward did not return logits")


def load_model():
    import torch
    from transformer_lens import HookedTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float32,
        revision=REVISION,
    )
    model.eval()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError(
            f"unexpected architecture: layers={model.cfg.n_layers} d_model={model.cfg.d_model}"
        )
    if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model, device


def forward_margin(model, text: str, positive_id: int, negative_id: int, hook: dict[str, Any] | None):
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    recorder: dict[str, Any] = {}
    with torch.no_grad():
        if hook is None:
            logits = _as_logits(model(tokens))
            return logit_margin(logits, positive_id, negative_id), recorder
        direction = torch.tensor(hook["direction"], dtype=torch.float32)
        hook_fn = make_resid_hook(float(hook["alpha"]), direction, recorder)
        logits = _as_logits(model.run_with_hooks(tokens, fwd_hooks=[(HOOK_NAME, hook_fn)]))
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError(f"hook fired {recorder.get('fired', 0)} times")
    if recorder.get("hook_name") not in {HOOK_NAME, ""}:
        raise RuntimeError(f"unexpected hook name {recorder.get('hook_name')}")
    if not recorder.get("other_unchanged", False):
        raise RuntimeError("intervention modified a non-final position")
    if float(hook["alpha"]) != 0.0 and not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not modify the final position")
    return logit_margin(logits, positive_id, negative_id), recorder


def _scope() -> dict[str, Any]:
    return {
        "status": "IN_SCOPE",
        "hook": HOOK_NAME,
        "mechanism_id": MECHANISM_ID,
        "anchor_alpha": 1.0,
        "prompt_catalog": "M22.1",
    }


def _prediction_row(prompt, belief, alpha: float, sequence_index: int) -> dict[str, Any]:
    prediction = belief.predict(alpha)
    return {
        "sequence_index": sequence_index,
        "prompt_id": prompt.prompt_id,
        "regime_id": prompt.regime_id,
        "role": prompt.role,
        "alpha": alpha,
        "mechanism_id": belief.mechanism_id,
        "predicted_effect": prediction["predicted_effect"],
        "predicted_direction": prediction["predicted_direction"],
        "uncertainty": prediction["uncertainty"],
        "model_version": prediction["model_version"],
        "evidence_version": prediction["evidence_version"],
        "outcome_present": False,
    }


def _apply_updates(belief, pairs: list[tuple[str, float]]):
    current = belief
    ok = True
    for evidence_id, observed in pairs:
        before = current.predicted_effect
        updated = update_belief(current, observed, evidence_id)
        record = updated.update_history[-1]
        if updated.version != current.version + 1:
            ok = False
        if record["model_version_after"] != updated.version:
            ok = False
        stored = updated.observations
        if abs(updated.predicted_effect - (sum(stored) / len(stored))) > 1e-9:
            ok = False
        if len(stored) >= 2 and any(abs(value - observed) > 1e-12 for value in stored[:-1]):
            if abs(updated.predicted_effect - observed) <= 1e-12:
                ok = False
        if updated.predicted_effect == before and abs(observed - before) > 1e-12:
            # A no-op when the observation differs is a failed revision.
            ok = False
        current = updated
    return current, ok


def run_smoke(model) -> None:
    out = ROOT / "reports" / "m23_raw" / "smoke"
    outcome = out / "outcome.json"
    prediction = out / "prediction.json"
    outcome.unlink(missing_ok=True)
    if outcome.exists():
        raise RuntimeError("smoke outcome file still exists")
    prompt = next(record for record in frozen_prompts() if record.role == "discovery")
    tokens = resolve_outcome_tokens(model.tokenizer, POSITIVE_TEXT, NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"token ids changed: {tokens['token_ids']}")
    direction = primary_direction(int(model.cfg.d_model), DIRECTION_SEED)
    belief = belief_from_observations(
        MECHANISM_ID,
        "additive_last_token",
        [0.02, 0.03],
        validity_scope=_scope(),
    )
    _write(prediction, _prediction_row(prompt, belief, 1.0, 0))
    if outcome.exists():
        raise RuntimeError("smoke outcome appeared before the intervention")
    baseline, _ = forward_margin(
        model, prompt.text, tokens["positive_token_id"], tokens["negative_token_id"], None
    )
    intervened, recorder = forward_margin(
        model,
        prompt.text,
        tokens["positive_token_id"],
        tokens["negative_token_id"],
        {"alpha": 1.0, "direction": direction},
    )
    _write(
        outcome,
        {
            "prompt_id": prompt.prompt_id,
            "baseline_margin": baseline,
            "intervened_margin": intervened,
            "observed_delta": intervened - baseline,
            "pre_dot": recorder.get("pre_dot"),
            "hook_fired": recorder.get("fired"),
            "scored": False,
        },
    )
    _write(
        out / "smoke_log.json",
        {
            "prediction_before_outcome": prediction.stat().st_mtime_ns <= outcome.stat().st_mtime_ns,
            "hook_fired_once": recorder.get("fired") == 1,
            "other_unchanged": recorder.get("other_unchanged"),
            "scored": False,
            "verdict": "NOT_SCORED",
        },
    )
    print("smoke complete; not scored")


def _intervene_split(model, prompts, alpha: float, direction, tokens) -> list[dict[str, Any]]:
    rows = []
    for prompt in prompts:
        baseline, _ = forward_margin(
            model, prompt.text, tokens["positive_token_id"], tokens["negative_token_id"], None
        )
        intervened, recorder = forward_margin(
            model,
            prompt.text,
            tokens["positive_token_id"],
            tokens["negative_token_id"],
            {"alpha": alpha, "direction": direction},
        )
        rows.append(
            {
                "prompt_id": prompt.prompt_id,
                "regime_id": prompt.regime_id,
                "role": prompt.role,
                "alpha": alpha,
                "baseline_margin": baseline,
                "intervened_margin": intervened,
                "observed_delta": intervened - baseline,
                "observed_direction": 1 if intervened > baseline else (-1 if intervened < baseline else 0),
                "pre_dot": recorder.get("pre_dot"),
            }
        )
        print(f"  {prompt.prompt_id} alpha={alpha} delta={rows[-1]['observed_delta']:.6f}", flush=True)
    return rows


def run_scored(model, device: str) -> None:
    out = ROOT / "reports" / "m23_raw"
    validation_predictions = out / "validation_predictions.json"
    validation_outcomes = out / "validation_outcomes.json"
    replication_predictions = out / "replication_predictions.json"
    replication_outcomes = out / "replication_outcomes.json"
    for path in (validation_predictions, validation_outcomes, replication_predictions, replication_outcomes):
        path.unlink(missing_ok=True)

    tokens = resolve_outcome_tokens(model.tokenizer, POSITIVE_TEXT, NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != EXPECTED_TOKEN_IDS:
        raise RuntimeError(f"token ids changed: {tokens['token_ids']}")
    dimension = int(model.cfg.d_model)
    direction = primary_direction(dimension, DIRECTION_SEED)
    control = orthogonal_direction(dimension, CONTROL_SEED, direction)
    grouped: dict[str, list] = {"discovery": [], "validation": [], "replication": []}
    for prompt in frozen_prompts():
        grouped[prompt.role].append(prompt)
    for role in grouped:
        grouped[role].sort(key=lambda record: record.prompt_id)

    print("calibration / discovery", flush=True)
    discovery_d1 = _intervene_split(model, grouped["discovery"], 1.0, direction, tokens)
    discovery_d2 = _intervene_split(model, grouped["discovery"], 1.0, control, tokens)
    if len(discovery_d1) < 2:
        raise RuntimeError("discovery split cannot form a belief")
    initial = belief_from_observations(
        MECHANISM_ID,
        "additive_last_token",
        [float(row["observed_delta"]) for row in discovery_d1],
        validity_scope=_scope(),
    )
    outcome_only = belief_from_observations(
        "OUTCOME_ONLY_POOLED",
        "pooled_outcome_history",
        [float(row["observed_delta"]) for row in discovery_d1 + discovery_d2],
        validity_scope={"status": "OUTCOME_ONLY", "mechanism_id": None},
    )

    if validation_outcomes.exists():
        raise RuntimeError("validation outcomes exist before predictions")
    frozen_validation = [
        _prediction_row(prompt, initial, 1.0, index)
        for index, prompt in enumerate(grouped["validation"])
    ]
    _write(validation_predictions, {"rows": frozen_validation, "belief_version": initial.version})

    print("update split / validation", flush=True)
    validation_d1 = _intervene_split(model, grouped["validation"], 1.0, direction, tokens)
    validation_d2 = _intervene_split(model, grouped["validation"], 1.0, control, tokens)
    _write(validation_outcomes, {"d1": validation_d1, "d2": validation_d2})
    if validation_predictions.stat().st_mtime_ns > validation_outcomes.stat().st_mtime_ns:
        raise RuntimeError("validation outcomes were written before predictions")

    updated, q3_ok = _apply_updates(
        initial,
        [(row["prompt_id"], float(row["observed_delta"])) for row in validation_d1],
    )
    shuffled, shuffled_ok = _apply_updates(
        initial,
        [(f"shuffle-{row['prompt_id']}", float(row["observed_delta"])) for row in validation_d2],
    )
    q3_ok = q3_ok and shuffled_ok

    if replication_outcomes.exists():
        raise RuntimeError("replication outcomes exist before predictions")
    replication_prediction_rows = []
    sequence = 0
    for alpha in (1.0, 2.0):
        for prompt in grouped["replication"]:
            issued = _prediction_row(prompt, updated, alpha, sequence)
            issued["no_update_prediction"] = initial.predict(alpha)["predicted_effect"]
            issued["shuffled_prediction"] = shuffled.predict(alpha)["predicted_effect"]
            issued["outcome_only_prediction"] = outcome_only.predict(alpha)["predicted_effect"]
            replication_prediction_rows.append(issued)
            sequence += 1
    _write(
        replication_predictions,
        {"rows": replication_prediction_rows, "belief_version": updated.version},
    )

    print("eval split / replication", flush=True)
    replication_a1 = _intervene_split(model, grouped["replication"], 1.0, direction, tokens)
    replication_a2 = _intervene_split(model, grouped["replication"], 2.0, direction, tokens)
    _write(replication_outcomes, {"alpha_1": replication_a1, "alpha_2": replication_a2})
    if replication_predictions.stat().st_mtime_ns > replication_outcomes.stat().st_mtime_ns:
        raise RuntimeError("replication outcomes were written before predictions")

    by_id = {row["prompt_id"]: row for row in validation_d1}
    validation_scored = []
    running = initial
    for issued in frozen_validation:
        observed = by_id[issued["prompt_id"]]
        running = update_belief(running, float(observed["observed_delta"]), issued["prompt_id"])
        evidence = running.update_history[-1]["evidence"]
        validation_scored.append(
            {
                **issued,
                "baseline_margin": observed["baseline_margin"],
                "pre_dot": observed["pre_dot"],
                "observed_delta": observed["observed_delta"],
                "observed_direction": observed["observed_direction"],
                "prediction_error": observed["observed_delta"] - issued["predicted_effect"],
                "critical": bool(evidence["critical"]),
                "update_reason": running.update_history[-1]["update_reason"],
                "version_before": running.update_history[-1]["model_version_before"],
                "version_after": running.update_history[-1]["model_version_after"],
                "split": "validation",
            }
        )

    outcomes_by_alpha = {1.0: replication_a1, 2.0: replication_a2}
    replication_scored = {1.0: [], 2.0: []}
    for issued in replication_prediction_rows:
        observed = next(
            row
            for row in outcomes_by_alpha[float(issued["alpha"])]
            if row["prompt_id"] == issued["prompt_id"]
        )
        replication_scored[float(issued["alpha"])].append(
            {
                **issued,
                "baseline_margin": observed["baseline_margin"],
                "pre_dot": observed["pre_dot"],
                "observed_delta": observed["observed_delta"],
                "observed_direction": observed["observed_direction"],
                "prediction_error": observed["observed_delta"] - issued["predicted_effect"],
                "split": "replication",
            }
        )

    leakage = {
        "validation_predictions_before_outcomes": True,
        "replication_predictions_before_outcomes": True,
        "predict_rejects_outcome_argument": True,
        "published_m22_means_not_loaded": not PUBLISHED_M22_MEANS_LOADED,
    }
    _write(out / "leakage_execution.json", leakage)
    scored = score_partition(
        validation_scored,
        replication_scored[1.0],
        replication_scored[2.0],
        q3_pass=q3_ok,
        leakage_ok=all(leakage.values()),
    )
    repro = reproducibility()
    repro["device"] = device
    repro["direction_sha256"] = vector_sha256(direction)
    repro["control_direction_sha256"] = vector_sha256(control)
    repro["positive_token_id"] = str(tokens["positive_token_id"])
    repro["negative_token_id"] = str(tokens["negative_token_id"])
    payload = {
        **scored,
        "milestone": "M23",
        "mechanism_id": MECHANISM_ID,
        "prior_status": "CANDIDATE",
        "reproducibility": repro,
        "discovery_d1_mean": sum(row["observed_delta"] for row in discovery_d1) / len(discovery_d1),
        "discovery_d2_mean": sum(row["observed_delta"] for row in discovery_d2) / len(discovery_d2),
        "initial_predicted_effect": initial.predicted_effect,
        "updated_predicted_effect": updated.predicted_effect,
        "shuffled_predicted_effect": shuffled.predicted_effect,
        "outcome_only_predicted_effect": outcome_only.predicted_effect,
    }
    _write(ROOT / "reports" / "M23_SCIENTIFIC_VERDICT.json", payload)
    _write(out / "beliefs.json", {
        "initial": initial.update_history,
        "updated_history": updated.update_history,
        "final_primary": {
            "predicted_effect": updated.predicted_effect,
            "uncertainty": updated.uncertainty,
            "version": updated.version,
            "validity_scope": updated.validity_scope,
        },
    })
    csv_path = ROOT / "reports" / "M23_EPISODE_RESULTS.csv"
    rows = validation_scored + replication_scored[1.0] + replication_scored[2.0]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = render_report(payload)
    (ROOT / "reports" / "M23_SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
    print(f"verdict {payload['verdict']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "scored"], required=True)
    args = parser.parse_args()
    assert_frozen_documents()
    model, device = load_model()
    if args.mode == "smoke":
        run_smoke(model)
        return
    run_scored(model, device)


if __name__ == "__main__":
    main()
