"""M30 stage 1: pre-outcome predictions only.

This script computes g and kappa for the frozen M30 catalog and stops.
It does not run a scored intervention, and it does not read or write an
outcome, a residual, or a holdout score.

g, F''(0), and kappa come from scripts/m29d_measurement.py:measure_kappa.
That function is not modified. It looks up the hook as
HOOKS[f"M22.1-D1-L{layer}"] and the margin token ids as module globals.
This runner installs the requested hook and token ids for the duration of
one call, then restores the module. The autograd body is the M29-D function:
float32, Hessian-vector product, alpha +1. The model is loaded on CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.m29d_measurement as measurement
from scripts.m30_catalog import CATALOG_SHA256, catalog, catalog_sha256
from src.cognitive_self_model.m23.m22_reuse import (
    orthogonal_direction,
    primary_direction,
    vector_sha256,
)

PROTOCOL_PATH = ROOT / "docs" / "M30_PROTOCOL.md"
AUDIT_PATH = ROOT / "reports" / "M30_CATALOG_AUDIT.md"
OUT = ROOT / "reports" / "m30_raw"
PREDICTIONS_PATH = OUT / "PRE_OUTCOME_PREDICTIONS.json"
MANIFEST_PATH = OUT / "PRE_OUTCOME_MANIFEST.json"

LOCKED_CATALOG_SHA256 = "4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770"
LOCKED_PROTOCOL_SHA256 = "1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3"
D1_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
D2_SHA256 = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
MODEL_ID = measurement.MODEL_ID
MODEL_REVISION = measurement.REVISION
ALPHA = 1.0
DTYPE_NAME = "float32"
DEVICE_NAME = "cpu"

NEW_CELLS = (
    ("M30-D2-L4", 4, "blocks.4.hook_resid_post"),
    ("M30-D2-L12", 12, "blocks.12.hook_resid_post"),
    ("M30-D2-L20", 20, "blocks.20.hook_resid_post"),
)
ANCHOR_CELLS = (
    ("M22.1-D1-L0", 0, "blocks.0.hook_resid_post"),
    ("M22.1-D1-L8", 8, "blocks.8.hook_resid_post"),
    ("M22.1-D1-L15", 15, "blocks.15.hook_resid_post"),
)
FORBIDDEN_KEYS = (
    "observed_effect",
    "intervened_output",
    "baseline_output",
    "residual",
    "actual_outcome",
    "y",
    "r",
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    direction: np.ndarray
    positive_id: int
    negative_id: int
    cells: tuple[tuple[str, int, str], ...]


def assert_catalog_lock() -> str:
    digest = catalog_sha256()
    if digest != CATALOG_SHA256 or digest != LOCKED_CATALOG_SHA256 or len(catalog()) != 48:
        raise SystemExit(f"catalog hash mismatch: {digest}")
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    if digest not in protocol or digest not in audit:
        raise SystemExit("catalog hash mismatch")
    return digest


def protocol_sha256() -> str:
    raw = PROTOCOL_PATH.read_bytes()
    marker = b"\nPROTOCOL_SHA256: "
    head, separator, tail = raw.rpartition(marker)
    if not separator:
        raise SystemExit("protocol hash mismatch")
    digest = hashlib.sha256(head).hexdigest()
    if tail.strip().decode("utf-8") != digest or digest != LOCKED_PROTOCOL_SHA256:
        raise SystemExit("protocol hash mismatch")
    return digest


def assert_clean_git() -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit("work tree is not clean")
    if not commit:
        raise SystemExit("git commit id is empty")
    return commit


def load_directions() -> tuple[np.ndarray, np.ndarray]:
    if measurement.DIRECTION_SHA256 != D1_SHA256:
        raise SystemExit("D1 direction hash mismatch")
    direction_d1 = primary_direction(896, 22101)
    if vector_sha256(direction_d1) != D1_SHA256:
        raise SystemExit("D1 direction hash mismatch")
    direction_d2 = orthogonal_direction(896, 22103, direction_d1)
    if vector_sha256(direction_d2) != D2_SHA256:
        raise SystemExit("D2 direction hash mismatch")
    return direction_d1, direction_d2


def _as_id_list(encoded: Any) -> list[int]:
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(item) for item in encoded]


def resolve_single_token(tokenizer: Any, text: str) -> int:
    """Abort when text is not one token. Do not try a substitute string."""
    token_ids = _as_id_list(tokenizer.encode(text, add_special_tokens=False))
    if len(token_ids) != 1:
        raise SystemExit(f"not a single token: {text!r} -> {token_ids}")
    return token_ids[0]


def resolve_new_identity_tokens(tokenizer: Any) -> tuple[int, int]:
    positive = resolve_single_token(tokenizer, " true")
    negative = resolve_single_token(tokenizer, " false")
    return positive, negative


def resolve_anchor_tokens(tokenizer: Any) -> tuple[int, int]:
    positive = resolve_single_token(tokenizer, " yes")
    negative = resolve_single_token(tokenizer, " no")
    if (positive, negative) != (9834, 902):
        raise SystemExit(f"anchor token ids are {positive}, {negative}")
    return positive, negative


def _call_measure(
    model: Any,
    tokens: Any,
    layer: int,
    hook: str,
    direction: np.ndarray,
    positive_id: int,
    negative_id: int,
) -> dict[str, float]:
    key = f"M22.1-D1-L{layer}"
    saved_hooks = dict(measurement.HOOKS)
    saved_positive = measurement.POSITIVE_ID
    saved_negative = measurement.NEGATIVE_ID
    measurement.HOOKS[key] = hook
    measurement.POSITIVE_ID = int(positive_id)
    measurement.NEGATIVE_ID = int(negative_id)
    try:
        values = measurement.measure_kappa(model, tokens, layer, direction)
    finally:
        measurement.HOOKS.clear()
        measurement.HOOKS.update(saved_hooks)
        measurement.POSITIVE_ID = saved_positive
        measurement.NEGATIVE_ID = saved_negative
    return {
        "g": values["g"],
        "kappa": values["kappa"],
        "f_second": values["f_second"],
    }


def predict_catalog(model: Any, rows: list[dict[str, str]], arms: list[ArmSpec]) -> list[dict[str, Any]]:
    if len(rows) != 48:
        raise RuntimeError("expected 48 prompts")
    predictions: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        tokens = model.to_tokens(row["text"], prepend_bos=True)
        for arm in arms:
            for intervention_id, layer, hook in arm.cells:
                measured = _call_measure(
                    model,
                    tokens,
                    layer,
                    hook,
                    arm.direction,
                    arm.positive_id,
                    arm.negative_id,
                )
                predictions.append(
                    {
                        "alpha": ALPHA,
                        "arm": arm.name,
                        "f_second": measured["f_second"],
                        "family": row["family"],
                        "g": measured["g"],
                        "hook": hook,
                        "intervention_id": intervention_id,
                        "kappa": measured["kappa"],
                        "outcome_present": False,
                        "partition": row["partition"],
                        "prompt_id": row["prompt_id"],
                    }
                )
        print(f"predicted {row_index + 1}/{len(rows)} {row['prompt_id']}", flush=True)
    if len(predictions) != 288:
        raise RuntimeError(f"expected 288 prediction rows, got {len(predictions)}")
    reject_outcome_fields(predictions)
    return predictions


def reject_outcome_fields(predictions: list[dict[str, Any]]) -> None:
    for index, row in enumerate(predictions):
        bad = set(FORBIDDEN_KEYS).intersection(row)
        if bad or row.get("outcome_present") is not False:
            raise RuntimeError(f"prediction row {index} is not a pre-outcome row")


def magnitude_check(predictions: list[dict[str, Any]], arm: str, cells: tuple[str, ...]) -> dict[str, Any]:
    """Protocol section 7. UPDATE rows of one arm only."""
    update = [row for row in predictions if row["arm"] == arm and row["partition"] == "update"]
    finite = all(math.isfinite(row["g"]) and math.isfinite(row["kappa"]) for row in update)
    cell_reports = []
    for cell in cells:
        group = [row for row in update if row["intervention_id"] == cell]
        if len(group) != 12:
            raise RuntimeError(f"{arm} {cell} expected 12 UPDATE rows, got {len(group)}")
        median_abs_kappa = float(statistics.median([abs(float(row["kappa"])) for row in group]))
        median_abs_g = float(statistics.median([abs(float(row["g"])) for row in group]))
        ratio = None if median_abs_g == 0.0 else median_abs_kappa / median_abs_g
        cell_reports.append(
            {
                "intervention_id": cell,
                "median_abs_g": median_abs_g,
                "median_abs_kappa": median_abs_kappa,
                "ratio": ratio,
            }
        )
    passed = finite and all(
        item["median_abs_kappa"] > 0.0 and item["median_abs_g"] > 0.0 for item in cell_reports
    )
    return {"arm": arm, "cells": cell_reports, "finite": finite, "passed": passed}


def _load_model():
    import torch
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=DEVICE_NAME,
        dtype=torch.float32,
        revision=MODEL_REVISION,
    )
    model.eval()
    if int(model.cfg.n_layers) != 24 or int(model.cfg.d_model) != 896:
        raise RuntimeError("unexpected model architecture")
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32 or str(parameter.device) != DEVICE_NAME:
        raise RuntimeError("model is not float32 on CPU")
    if getattr(model, "tokenizer", None) is not None and model.tokenizer.pad_token_id is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    return model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if PREDICTIONS_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit("M30 pre-outcome artifacts already exist; refusing to overwrite")
    catalog_digest = assert_catalog_lock()
    protocol_digest = protocol_sha256()
    commit = assert_clean_git()
    direction_d1, direction_d2 = load_directions()

    started = time.perf_counter()
    model = _load_model()
    new_positive, new_negative = resolve_new_identity_tokens(model.tokenizer)
    anchor_positive, anchor_negative = resolve_anchor_tokens(model.tokenizer)
    arms = [
        ArmSpec("new_identity", direction_d2, new_positive, new_negative, NEW_CELLS),
        ArmSpec("anchor", direction_d1, anchor_positive, anchor_negative, ANCHOR_CELLS),
    ]
    predictions = predict_catalog(model, catalog(), arms)
    OUT.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(
        json.dumps(predictions, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    prediction_digest = _sha256(PREDICTIONS_PATH)
    checks = [
        magnitude_check(predictions, "new_identity", tuple(cell[0] for cell in NEW_CELLS)),
        magnitude_check(predictions, "anchor", tuple(cell[0] for cell in ANCHOR_CELLS)),
    ]
    runtime_seconds = time.perf_counter() - started
    manifest = {
        "alpha": ALPHA,
        "catalog_sha256": catalog_digest,
        "device": DEVICE_NAME,
        "directions": {
            "D1": {"seed": 22101, "sha256": D1_SHA256},
            "D2": {"seed": 22103, "sha256": D2_SHA256},
        },
        "dtype": DTYPE_NAME,
        "git_commit": commit,
        "magnitude_check": checks,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "outcome_fields_present": False,
        "prediction_rows": len(predictions),
        "prediction_sha256": prediction_digest,
        "protocol_sha256": protocol_digest,
        "runtime_seconds": runtime_seconds,
        "scored_intervention_executed": False,
        "status": "PRE_OUTCOME_COMPLETE",
        "token_ids": {
            "anchor": {
                "negative_id": anchor_negative,
                "negative_text": " no",
                "positive_id": anchor_positive,
                "positive_text": " yes",
            },
            "new_identity": {
                "negative_id": new_negative,
                "negative_text": " false",
                "positive_id": new_positive,
                "positive_text": " true",
            },
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
