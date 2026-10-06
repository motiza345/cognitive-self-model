"""M29-D pre-outcome prediction runner.

This script intentionally stops after computing frozen g/kappa predictions.
It never runs the scored intervention and never reads an outcome/residual.

Run only after the M29-D execution-freeze report is committed.
"""

from __future__ import annotations

import hashlib
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
    HOOKS,
    LAYERS,
    MODEL_ID,
    REVISION,
    _direction,
    measure_kappa,
    measurement_manifest,
)

OUT = ROOT / "reports" / "m29d_raw"
PREDICTIONS = OUT / "PRE_OUTCOME_PREDICTIONS.json"
MANIFEST = OUT / "PRE_OUTCOME_MANIFEST.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_model():
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
        raise RuntimeError("unexpected model architecture")
    if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model, device


def _as_logits(output: Any):
    import torch
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("unexpected model output")


def main() -> None:
    if PREDICTIONS.exists() or MANIFEST.exists():
        raise SystemExit("M29-D pre-outcome artifacts already exist; refusing to overwrite")

    rows = catalog()
    if len(rows) != 36 or CATALOG_SHA256 != "d6ecab612a2214d947be11c82671612f883c0d13e479b07e8cbaa09b635d4d77":
        raise SystemExit("catalog freeze mismatch")

    model, device = _load_model()
    direction = _direction()

    tokenizer = model.tokenizer
    positive_ids = tokenizer.encode(" yes", add_special_tokens=False)
    negative_ids = tokenizer.encode(" no", add_special_tokens=False)
    if positive_ids != [9834] or negative_ids != [902]:
        raise SystemExit(f"frozen token ids changed: {positive_ids}, {negative_ids}")

    predictions = []
    for row in rows:
        tokens = model.to_tokens(row["text"], prepend_bos=True)
        for layer in LAYERS:
            values = measure_kappa(model, tokens, layer, direction)
            intervention_id = f"M22.1-D1-L{layer}"
            predictions.append({
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "partition": row["partition"],
                "intervention_id": intervention_id,
                "hook": HOOKS[intervention_id],
                "alpha": 1.0,
                **values,
                "outcome_present": False,
            })

    if len(predictions) != 108:
        raise RuntimeError(f"expected 108 prediction rows, got {len(predictions)}")

    # Hard leakage boundary: these names are forbidden in the prediction artifact.
    serialized = json.dumps(predictions, sort_keys=True)
    forbidden = (
        "observed_effect",
        "intervened_output",
        "baseline_output",
        "residual",
    )
    if any(key in serialized for key in forbidden):
        raise RuntimeError("prediction artifact contains forbidden outcome fields")

    OUT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_text(json.dumps(predictions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "status": "PRE_OUTCOME_COMPLETE",
        "model": MODEL_ID,
        "model_revision": REVISION,
        "device": device,
        "python": platform.python_version(),
        "catalog_sha256": CATALOG_SHA256,
        "measurement_manifest": measurement_manifest(),
        "prediction_rows": len(predictions),
        "partition_counts": {
            p: sum(r["partition"] == p for r in predictions)
            for p in ("update", "validation", "holdout")
        },
        "outcome_fields_present": False,
        "prediction_sha256": _sha(PREDICTIONS),
        "scored_intervention_executed": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
