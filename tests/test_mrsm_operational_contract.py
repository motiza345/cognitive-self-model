"""Unit tests for the draft MRSM operational contract. No model load."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "mrsm_operational_contract_v0.1.yaml"
SCRIPT = ROOT / "scripts" / "validate_mrsm_operational_contract.py"


def _validator():
    spec = importlib.util.spec_from_file_location("validate_mrsm_operational_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _document():
    return yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))


def test_valid_contract_passes():
    module = _validator()
    document = _document()
    assert module.validate_document(document) == []
    assert module.main([str(CONTRACT)]) == 0


def test_q_h1_cannot_become_ground_truth_verification():
    module = _validator()
    document = _document()
    document["q_h1"]["verifies_mechanism_identity"] = True
    document["p_q_boundary"]["q_h1_is_mechanism_verification"] = True
    errors = module.validate_document(document)
    assert any("verifies_mechanism_identity" in item for item in errors)
    assert any("q_h1_is_mechanism_verification" in item for item in errors)


def test_q_h1_role_cannot_become_discovery_verification():
    module = _validator()
    document = _document()
    document["q_h1"]["role"] = "DISCOVERY_VERIFICATION"
    errors = module.validate_document(document)
    assert any("DIAGNOSTIC_TRANSFER" in item for item in errors)
    assert any("DISCOVERY_VERIFICATION" in item for item in errors)


def test_thresholds_remain_deferred():
    module = _validator()
    document = _document()
    assert document["q_h1"]["thresholds_status"] == "DEFERRED"
    assert document["h3"]["thresholds_status"] == "DEFERRED"
    document["q_h1"]["thresholds_status"] = "FROZEN"
    document["h3"]["thresholds_status"] = "FROZEN"
    errors = module.validate_document(document)
    assert any("q_h1.thresholds_status" in item for item in errors)
    assert any("h3.thresholds_status" in item for item in errors)


def test_all_h3_transformation_families_exist():
    document = _document()
    assert document["h3"]["transformation_family"] == [
        "invertible_affine",
        "coordinate_permutation",
        "orthogonal_basis_change",
    ]


def test_missing_transformation_family_fails():
    module = _validator()
    document = _document()
    document["h3"]["transformation_family"] = [
        "invertible_affine",
        "orthogonal_basis_change",
    ]
    errors = module.validate_document(document)
    assert any("coordinate_permutation" in item for item in errors)


def test_p_ground_truth_true_and_q_ground_truth_false():
    module = _validator()
    document = _document()
    assert document["p_q_boundary"]["P_ground_truth"] is True
    assert document["p_q_boundary"]["Q_ground_truth"] is False
    assert module.validate_document(document) == []
    broken = copy.deepcopy(document)
    broken["p_q_boundary"]["P_ground_truth"] = False
    broken["p_q_boundary"]["Q_ground_truth"] = True
    errors = module.validate_document(broken)
    assert any("P_ground_truth" in item for item in errors)
    assert any("Q_ground_truth" in item for item in errors)


def test_validator_rejects_numerical_threshold_insertion():
    module = _validator()
    document = _document()
    document["q_h1"]["pass_threshold"] = 0.8
    document["h3"]["identity_agreement_min"] = 0.9
    errors = module.validate_document(document)
    assert any("q_h1.pass_threshold" in item for item in errors)
    assert any("h3.identity_agreement_min" in item for item in errors)
    assert module.main([str(CONTRACT)]) == 0
