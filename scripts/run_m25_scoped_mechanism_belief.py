"""Apply the frozen M25 bookkeeping rules to recorded M24 rows.

This script does not load Qwen and does not change the response model.
It is not part of the protocol freeze; run it only when execution is requested.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.mechanism_belief import (
    M24_CATALOG_SHA256,
    PROTOCOL_VERSION,
    apply_evidence,
    beliefs_from_m24,
    contradictory_observed,
    make_evidence,
)

RESULTS_PATH = ROOT / "reports" / "M25_SCOPED_MECHANISM_BELIEF_RESULTS.json"
EPISODES = ROOT / "reports" / "M24_CONSUMPTION_EPISODES.csv"


def _evaluation_rows(intervention_id: str) -> list[dict[str, str]]:
    rows = []
    with EPISODES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["intervention_id"] == intervention_id and row["partition"] == "evaluation":
                rows.append(row)
    rows.sort(key=lambda row: row["prompt_id"])
    if len(rows) != 12:
        raise RuntimeError(f"{intervention_id} does not have 12 evaluation rows")
    return rows


def _apply_recorded(belief, rows: list[dict[str, str]]):
    current = belief
    for row in rows:
        predicted = float(row["g"])
        if float(row["candidate_prediction"]) != predicted:
            raise RuntimeError("evaluation row changed g")
        evidence = make_evidence(
            evidence_id=f"M24:{row['intervention_id']}:{row['prompt_id']}",
            source_experiment="M24",
            artifact="reports/M24_CONSUMPTION_EPISODES.csv",
            claim=belief.validity_scope.claim,
            prompt_id=row["prompt_id"],
            predicted_value=predicted,
            observed_value=float(row["observed_effect"]),
            held_out=True,
            provenance="M24 evaluation episode",
        )
        current = apply_evidence(current, evidence)
    return current


def _snapshot(belief) -> dict[str, object]:
    return {
        "mechanism_id": belief.mechanism_id,
        "status": belief.status,
        "version": belief.version,
        "inflation": belief.uncertainty.inflation,
        "residual_sd": belief.uncertainty.residual_sd,
        "effective_uncertainty": belief.uncertainty.effective,
        "n_evidence": len(belief.evidence_history),
        "n_contradictions": len(belief.contradiction_history),
        "reasons": [item.reason for item in belief.evidence_history],
        "response_rule": belief.response_model.response_rule,
        "direction_sha256": belief.response_model.direction_sha256,
        "training_baseline_mean": belief.response_model.training_baseline_mean,
        "supported_catalog_sha256": belief.validity_scope.supported_catalog_sha256,
    }


def main() -> None:
    if RESULTS_PATH.exists():
        raise SystemExit("M25 results already exist; refusing to rerun")
    cells = []
    for belief in beliefs_from_m24():
        original = belief.response_model
        consistent = _apply_recorded(belief, _evaluation_rows(belief.mechanism_id))
        contradictory = apply_evidence(
            belief,
            make_evidence(
                evidence_id=f"synthetic-contradiction:{belief.mechanism_id}",
                source_experiment="M25-synthetic",
                artifact="scripts/run_m25_scoped_mechanism_belief.py",
                claim=belief.validity_scope.claim,
                prompt_id="synthetic-contradiction",
                predicted_value=0.25,
                observed_value=contradictory_observed(0.25, belief.uncertainty.effective),
                held_out=True,
                provenance="predeclared contradictory observation",
            ),
        )
        foreign_claim = belief.validity_scope.claim
        foreign_claim = type(foreign_claim)(**{**asdict(foreign_claim), "catalog_sha256": "0" * 64})
        uncertain = apply_evidence(
            belief,
            make_evidence(
                evidence_id=f"synthetic-unsupported-catalog:{belief.mechanism_id}",
                source_experiment="M25-synthetic",
                artifact="scripts/run_m25_scoped_mechanism_belief.py",
                claim=foreign_claim,
                prompt_id="synthetic-unsupported-catalog",
                predicted_value=0.25,
                observed_value=0.25,
                held_out=True,
                provenance="predeclared unsupported catalog",
            ),
        )
        if contradictory.status != "CONTRADICTED" or contradictory.response_model is not original:
            raise RuntimeError("contradictory condition did not preserve the frozen transition")
        if uncertain.status != "UNCERTAIN" or uncertain.validity_scope.supported_catalog_sha256 != M24_CATALOG_SHA256:
            raise RuntimeError("unsupported catalog changed the support claim or the expected status")
        if consistent.response_model is not original:
            raise RuntimeError("consistent condition replaced the response model")
        cells.append(
            {
                "mechanism_id": belief.mechanism_id,
                "consistent": _snapshot(consistent),
                "contradictory": _snapshot(contradictory),
                "unsupported_catalog": _snapshot(uncertain),
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "catalog_sha256": M24_CATALOG_SHA256,
        "qwen_loaded": False,
        "cells": cells,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(RESULTS_PATH), "cells": [row["mechanism_id"] for row in cells]}, indent=2))


if __name__ == "__main__":
    main()
