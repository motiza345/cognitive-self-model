"""Execute the frozen M29-D intervention outcomes.

Consumes reports/m29d_raw/PRE_OUTCOME_PREDICTIONS.json and writes observed
effects. It does not recompute g or kappa, does not fit B0/B1, and does not
score HOLDOUT.

The intervention and margin are the M24/M26 implementation:
logit_margin and make_resid_hook from m22_reuse, alpha +1, last token only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m29d_catalog import CATALOG_SHA256, catalog
from scripts.m29d_measurement import (
    ALPHA,
    DIRECTION_SEED,
    DIRECTION_SHA256,
    HOOKS,
    MODEL_ID,
    NEGATIVE_ID,
    POSITIVE_ID,
    REVISION,
    _direction,
)
from scripts.m29d_score import EXPECTED_PREDICTION_SHA256, assert_frozen_predictions
from src.cognitive_self_model.m23.m22_reuse import logit_margin, make_resid_hook

OUT = ROOT / "reports" / "m29d_raw"
PREDICTIONS = OUT / "PRE_OUTCOME_PREDICTIONS.json"
OUTCOMES = OUT / "OUTCOMES.json"
MANIFEST = OUT / "OUTCOME_MANIFEST.json"
CATALOG_LOCK = "d6ecab612a2214d947be11c82671612f883c0d13e479b07e8cbaa09b635d4d77"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "UNAVAILABLE"


def _as_logits(output: Any):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("model forward did not return logits")


def _load_model():
    """CPU float32, matching the frozen prediction artifact's device."""
    import torch
    from transformer_lens import HookedTransformer

    device = "cpu"
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float32,
        revision=REVISION,
    )
    model.eval()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("unexpected model architecture")
    if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model, device


def _effect(model, text: str, hook_name: str, direction, positive_id: int, negative_id: int) -> dict[str, float]:
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        baseline = logit_margin(_as_logits(model(tokens)), positive_id, negative_id)
        recorder: dict[str, Any] = {}
        vector = torch.tensor(direction, dtype=torch.float32)
        hook_fn = make_resid_hook(float(ALPHA), vector, recorder)
        intervened = logit_margin(
            _as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])),
            positive_id,
            negative_id,
        )
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError("intervention hook did not fire once")
    if not recorder.get("other_unchanged", False) or not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not stay on the last token")
    if not (baseline == baseline and intervened == intervened):
        raise RuntimeError("non-finite margin")
    return {
        "baseline_output": float(baseline),
        "intervened_output": float(intervened),
        "observed_effect": float(intervened - baseline),
    }


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if OUTCOMES.exists() or MANIFEST.exists():
        raise SystemExit("M29-D outcome artifacts already exist; refusing to overwrite")
    if CATALOG_SHA256 != CATALOG_LOCK or len(catalog()) != 36:
        raise SystemExit("catalog freeze mismatch")
    before = _sha(PREDICTIONS)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise SystemExit("prediction artifact is not a list")
    assert_frozen_predictions(predictions, digest=before)

    by_prompt = {row["prompt_id"]: row for row in catalog()}
    if len(by_prompt) != 36:
        raise SystemExit("catalog prompt ids are not unique")
    for row in predictions:
        source = by_prompt.get(row["prompt_id"])
        if source is None:
            raise SystemExit(f"prediction prompt is not in the frozen catalog: {row['prompt_id']}")
        if source["partition"] != row["partition"] or source["family"] != row["family"]:
            raise SystemExit(f"prediction catalog fields differ for {row['prompt_id']}")
        if HOOKS[row["intervention_id"]] != row["hook"]:
            raise SystemExit(f"hook does not match the frozen cell: {row['intervention_id']}")

    model, device = _load_model()
    if device != "cpu":
        raise SystemExit("outcome execution must stay on CPU")
    direction = _direction()
    tokenizer = model.tokenizer
    positive_ids = tokenizer.encode(" yes", add_special_tokens=False)
    negative_ids = tokenizer.encode(" no", add_special_tokens=False)
    if positive_ids != [POSITIVE_ID] or negative_ids != [NEGATIVE_ID]:
        raise SystemExit(f"frozen token ids changed: {positive_ids}, {negative_ids}")

    outcomes = []
    baseline_cache: dict[str, float] = {}
    for index, row in enumerate(predictions, start=1):
        text = by_prompt[row["prompt_id"]]["text"]
        measured = _effect(model, text, row["hook"], direction, POSITIVE_ID, NEGATIVE_ID)
        if row["prompt_id"] not in baseline_cache:
            baseline_cache[row["prompt_id"]] = measured["baseline_output"]
        elif baseline_cache[row["prompt_id"]] != measured["baseline_output"]:
            raise RuntimeError("unperturbed margin changed across cells of one prompt")
        g = float(row["g"])
        observed = measured["observed_effect"]
        outcomes.append(
            {
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "partition": row["partition"],
                "intervention_id": row["intervention_id"],
                "mechanism_id": row["intervention_id"],
                "hook": row["hook"],
                "alpha": row["alpha"],
                "g": row["g"],
                "kappa": row["kappa"],
                "f_second": row["f_second"],
                "r_hat_m29": row["r_hat_m29"],
                "y_hat_m29": row["y_hat_m29"],
                "baseline_output": measured["baseline_output"],
                "intervened_output": measured["intervened_output"],
                "observed_effect": observed,
                "residual": float(observed - g),
                "model": MODEL_ID,
                "model_revision": REVISION,
                "direction_seed": DIRECTION_SEED,
                "direction_sha256": DIRECTION_SHA256,
                "positive_token_id": POSITIVE_ID,
                "negative_token_id": NEGATIVE_ID,
                "dtype": "float32",
                "device": device,
            }
        )
        if index % 3 == 0:
            print(f"outcome {row['partition']} {row['prompt_id']}", flush=True)

    if len(outcomes) != 108:
        raise RuntimeError(f"expected 108 outcome rows, got {len(outcomes)}")
    for prediction, outcome in zip(predictions, outcomes):
        for field in ("prompt_id", "partition", "intervention_id", "g", "kappa", "y_hat_m29", "alpha", "hook"):
            if outcome[field] != prediction[field]:
                raise RuntimeError(f"copied prediction field changed: {field}")
        if outcome["mechanism_id"] != prediction["intervention_id"]:
            raise RuntimeError("mechanism id does not match the prediction cell")

    if _sha(PREDICTIONS) != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError("prediction artifact changed before the outcome write")
    _write(OUTCOMES, outcomes)
    manifest = {
        "status": "OUTCOME_COMPLETE",
        "model": MODEL_ID,
        "model_revision": REVISION,
        "device": device,
        "dtype": "float32",
        "python": platform.python_version(),
        "torch": _version("torch"),
        "transformer_lens": _version("transformer-lens"),
        "alpha": ALPHA,
        "direction_seed": DIRECTION_SEED,
        "direction_sha256": DIRECTION_SHA256,
        "positive_token_id": POSITIVE_ID,
        "negative_token_id": NEGATIVE_ID,
        "catalog_sha256": CATALOG_SHA256,
        "prediction_sha256": EXPECTED_PREDICTION_SHA256,
        "prediction_sha256_after_write": _sha(PREDICTIONS),
        "outcome_rows": len(outcomes),
        "partition_counts": {
            name: sum(row["partition"] == name for row in outcomes)
            for name in ("update", "validation", "holdout")
        },
        "outcome_sha256": _sha(OUTCOMES),
        "baselines_fit": False,
        "verdict_executed": False,
        "prediction_modified": False,
    }
    if manifest["prediction_sha256_after_write"] != EXPECTED_PREDICTION_SHA256:
        OUTCOMES.unlink(missing_ok=True)
        raise RuntimeError("prediction artifact changed during the outcome write")
    _write(MANIFEST, manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "outcome_rows", "partition_counts", "prediction_sha256", "outcome_sha256")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
