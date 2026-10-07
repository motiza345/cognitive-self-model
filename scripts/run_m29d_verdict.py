"""Score the frozen M29-D outcomes once.

Reads the raw outcome artifact. Does not modify predictions or outcomes
and does not refit kappa. VALIDATION is reported after the verdict and
is not an input to it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m29d_measurement import DIRECTION_SHA256, MODEL_ID, REVISION
from scripts.m29d_score import (
    EXPECTED_PREDICTION_SHA256,
    FAMILY,
    assert_frozen_predictions,
    evaluate_rows,
    fit_baselines,
    score_m29,
)

OUT = ROOT / "reports" / "m29d_raw"
PREDICTIONS = OUT / "PRE_OUTCOME_PREDICTIONS.json"
OUTCOMES = OUT / "OUTCOMES.json"
OUTCOME_MANIFEST = OUT / "OUTCOME_MANIFEST.json"
VERDICT = OUT / "VERDICT.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _select(rows: list[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["partition"] == partition]


def _leakage(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]], prediction_digest: str) -> dict[str, bool]:
    checks = {
        "prediction_sha256": prediction_digest == EXPECTED_PREDICTION_SHA256,
        "prediction_count": len(predictions) == 108,
        "outcome_count": len(outcomes) == 108,
        "prediction_outcome_fields_absent": True,
        "copied_prediction_fields": True,
        "keys_match": True,
        "update_holdout_disjoint": True,
        "frozen_identity": True,
        "cell_counts": True,
    }
    try:
        assert_frozen_predictions(predictions, digest=prediction_digest)
    except ValueError:
        checks["prediction_outcome_fields_absent"] = False
        checks["prediction_count"] = False
    prediction_by = {(row["intervention_id"], row["prompt_id"]): row for row in predictions}
    outcome_by = {(row["intervention_id"], row["prompt_id"]): row for row in outcomes}
    checks["keys_match"] = set(prediction_by) == set(outcome_by) and len(outcome_by) == 108
    copied_fields = ("g", "kappa", "y_hat_m29", "partition", "alpha", "hook", "family", "r_hat_m29")
    for key, prediction in prediction_by.items():
        outcome = outcome_by.get(key)
        if outcome is None:
            checks["copied_prediction_fields"] = False
            continue
        if any(outcome.get(field) != prediction.get(field) for field in copied_fields):
            checks["copied_prediction_fields"] = False
        if outcome.get("mechanism_id") != prediction["intervention_id"]:
            checks["frozen_identity"] = False
        if outcome.get("model") != MODEL_ID or outcome.get("model_revision") != REVISION:
            checks["frozen_identity"] = False
        if outcome.get("direction_sha256") != DIRECTION_SHA256 or float(outcome.get("alpha", -1)) != 1.0:
            checks["frozen_identity"] = False
    update_ids = {row["prompt_id"] for row in predictions if row["partition"] == "update"}
    holdout_ids = {row["prompt_id"] for row in predictions if row["partition"] == "holdout"}
    checks["update_holdout_disjoint"] = update_ids.isdisjoint(holdout_ids)
    for partition in ("update", "validation", "holdout"):
        for cell in FAMILY:
            count = sum(
                1
                for row in outcomes
                if row["partition"] == partition and row["intervention_id"] == cell
            )
            if count != 12:
                checks["cell_counts"] = False
    checks["ok"] = all(checks.values())
    return checks


def main() -> None:
    if VERDICT.exists():
        raise SystemExit("M29-D verdict already exists; refusing to overwrite")
    if not OUTCOMES.exists() or not OUTCOME_MANIFEST.exists():
        raise SystemExit("raw outcome artifact is missing")
    prediction_digest = _sha(PREDICTIONS)
    outcome_digest_before = _sha(OUTCOMES)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    outcomes = json.loads(OUTCOMES.read_text(encoding="utf-8"))
    outcome_manifest = json.loads(OUTCOME_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not isinstance(outcomes, list):
        raise SystemExit("artifacts are not lists")
    leakage = _leakage(predictions, outcomes, prediction_digest)
    if outcome_manifest.get("status") != "OUTCOME_COMPLETE":
        leakage["ok"] = False
    if outcome_manifest.get("prediction_sha256") != EXPECTED_PREDICTION_SHA256:
        leakage["ok"] = False
    if outcome_manifest.get("baselines_fit") is not False:
        leakage["ok"] = False

    update = _select(outcomes, "update")
    holdout = _select(outcomes, "holdout")
    validation = _select(outcomes, "validation")
    scored = score_m29(update, holdout, leakage_ok=bool(leakage["ok"]))
    # Validation is descriptive only. It is computed after the verdict.
    validation_report = None
    if leakage["ok"]:
        validation_report = evaluate_rows(validation, fit_baselines(update))
        validation_report = {
            "rows": validation_report["rows"],
            "mae_b0": validation_report["mae_b0"],
            "mae_b1": validation_report["mae_b1"],
            "mae_m29": validation_report["mae_m29"],
            "delta": validation_report["delta"],
            "used_for_verdict": False,
        }
    if _sha(PREDICTIONS) != prediction_digest or _sha(OUTCOMES) != outcome_digest_before:
        raise RuntimeError("artifacts changed while scoring")
    outcome_digest = outcome_digest_before
    payload = {
        "status": "VERDICT_COMPLETE",
        "verdict": scored["verdict"],
        "prediction_sha256": prediction_digest,
        "outcome_sha256": outcome_digest,
        "prediction_modified": prediction_digest != EXPECTED_PREDICTION_SHA256,
        "leakage": leakage,
        "same_holdout_rows": scored["same_rows"],
        "baselines": {
            "b0": scored["baselines"]["b0"],
            "b1": scored["baselines"]["b1"],
            "n": scored["baselines"]["n"],
            "source_partition": "update",
        },
        "holdout": {
            "rows": scored["holdout"]["rows"],
            "mae_b0": scored["holdout"]["mae_b0"],
            "mae_b1": scored["holdout"]["mae_b1"],
            "mae_m29": scored["holdout"]["mae_m29"],
            "delta": scored["holdout"]["delta"],
            "bootstrap": scored["holdout"]["bootstrap"],
            "paired_rows": scored["holdout"]["paired_rows"],
        },
        "b2": {
            "procedure": scored["b2"]["procedure"],
            "seed": scored["b2"]["seed"],
            "class": scored["b2"]["class"],
            "mae": scored["b2"]["mae"],
            "delta": scored["b2"]["delta"],
            "low": scored["b2"]["low"],
            "high": scored["b2"]["high"],
        },
        "validation_descriptive": validation_report,
        "validation_used_for_verdict": False,
    }
    if _sha(PREDICTIONS) != EXPECTED_PREDICTION_SHA256:
        raise RuntimeError("prediction artifact changed before the verdict write")
    VERDICT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _sha(PREDICTIONS) != EXPECTED_PREDICTION_SHA256 or _sha(OUTCOMES) != outcome_digest:
        VERDICT.unlink(missing_ok=True)
        raise RuntimeError("frozen artifacts changed during the verdict write")
    holdout_report = payload["holdout"]
    print("BRANCH", "research/m29-d-design")
    print("VERDICT", payload["verdict"])
    print("B0_MAE", holdout_report["mae_b0"])
    print("B1_MAE", holdout_report["mae_b1"])
    print("M29_MAE", holdout_report["mae_m29"])
    print("DELTA", holdout_report["delta"])
    print("BOOTSTRAP_CI_LOW", holdout_report["bootstrap"]["low"])
    print("BOOTSTRAP_CI_HIGH", holdout_report["bootstrap"]["high"])
    print("B2_RESULT", payload["b2"]["class"])
    print("LEAKAGE_STATUS", "PASS" if leakage["ok"] else "FAIL")


if __name__ == "__main__":
    main()
