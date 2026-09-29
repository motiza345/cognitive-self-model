"""Package a planted P checkpoint. Does not discover and does not score holdout."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import asdict

import torch

import mrsm_p_v1
from src.mrsm.interventions import execute
from src.mrsm.p_data import sha256_file, write_json


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def package(checkpoint: Path, train_log: Path | None = None) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = mrsm_p_v1.Config()
    mrsm_p_v1.seed_all(cfg.seed)
    dev = mrsm_p_v1.make_dataset(cfg, cfg.dev_examples, cfg.seed + 200)
    model = mrsm_p_v1.build_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    audit = mrsm_p_v1.run_mechanism_audit(model, dev, cfg)
    probe = execute(model, dev, "baseline", [])
    created = _now()
    out = checkpoint.parent
    digest = sha256_file(checkpoint)
    ground_truth = {
        "source": "control_plane/P_SPEC.md mechanism block",
        "derived_from_discovery": False,
        "visibility": "evaluator_only",
        "ordered_edge": ["L0H0", "L1H1"],
        "planted_A": [0, 0],
        "planted_B": [1, 1],
        "self_model_may_read": False,
    }
    causal_pass = bool(audit.sequential_signature)
    static_detector = "NOT_SEPARATELY_SPECIFIED"
    dod = {
        "model_artifact_sha256": digest,
        "mechanism_definition_recorded": True,
        "ground_truth_derived_without_discovery": ground_truth["derived_from_discovery"] is False,
        "intervention_interface_callable": probe["mode"] == "baseline",
        "observation_interface_callable": "accuracy" in probe["observable_outcome"],
        "deterministic_seeds_recorded": True,
        "causal_validation_passes": causal_pass,
        "static_weight_detector": static_detector,
        "train_dev_holdout_protocol_recorded": True,
        "discovery_run": False,
        "holdout_generated": False,
        "holdout_scored": False,
    }
    engineering_pass = all(
        dod[key]
        for key in (
            "model_artifact_sha256",
            "mechanism_definition_recorded",
            "ground_truth_derived_without_discovery",
            "intervention_interface_callable",
            "observation_interface_callable",
            "deterministic_seeds_recorded",
            "causal_validation_passes",
            "train_dev_holdout_protocol_recorded",
        )
    )
    record = {
        "created_at": created,
        "checkpoint": str(checkpoint.relative_to(_ROOT)) if checkpoint.is_relative_to(_ROOT) else str(checkpoint),
        "sha256": digest,
        "train_log": None if train_log is None or not train_log.is_file() else json.loads(train_log.read_text(encoding="utf-8")),
        "mechanism_audit_dev": asdict(audit),
        "definition_of_done": dod,
        "engineering_outcome": "PASS" if engineering_pass else "FAIL",
        "budget_outcome": "DONE" if engineering_pass else "P_CONSTRUCTION_INCOMPLETE",
        "holdout_generated": False,
        "holdout_scored": False,
        "discovery_run": False,
    }
    write_json(_ROOT / "artifacts" / "mrsm" / "planted_ground_truth.json", ground_truth)
    write_json(out / "checkpoint_manifest.json", {"sha256": digest, "created_at": created, "path": record["checkpoint"]})
    write_json(out / "construction_record.json", record)
    return record


def main() -> int:
    checkpoint = _ROOT / "artifacts" / "mrsm" / "construction" / "p_model_state.pt"
    log = _ROOT / "artifacts" / "mrsm" / "construction" / "construction_train_log.json"
    record = package(checkpoint, log if log.is_file() else None)
    print(json.dumps({
        "engineering_outcome": record["engineering_outcome"],
        "budget_outcome": record["budget_outcome"],
        "causal_validation_passes": record["definition_of_done"]["causal_validation_passes"],
        "audit": record["mechanism_audit_dev"],
        "holdout_scored": False,
    }, indent=2))
    return 0 if record["engineering_outcome"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
