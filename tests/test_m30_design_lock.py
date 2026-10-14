"""M30 is a design lock: disjoint catalog, frozen identity, no outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_m30_catalog import audit
from scripts.m30_catalog import (
    CATALOG_SHA256,
    HOLDOUT_EXTENSION_RULE,
    _EXTENSION_ARTICLES,
    _EXTENSION_IMPLEMENTS,
    _EXTENSION_MATERIALS,
    _EXTENSION_TARGETS,
    _EXTENSION_TEMPLATES,
    catalog,
    catalog_sha256,
    holdout_extension_text,
    partition_manifest_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "M30_PROTOCOL.md"
AUDIT_REPORT = ROOT / "reports" / "M30_CATALOG_AUDIT.md"

# Manifest hashes of the parent revision's UPDATE and VALIDATION rows.
# Payload: JSON list sorted by prompt_id, object keys sorted, no whitespace.
PREVIOUS_UPDATE_SHA256 = "e6749441240e4aa97217e970241eba8381653e8d12b4e9b3dfddf4f87af5a82f"
PREVIOUS_VALIDATION_SHA256 = "329f729243d12e206ebf853236a5b77530fd6518f9d8b69da956168159068fb5"
PREVIOUS_HOLDOUT_SHA256 = "9c60ba51d2cb6cdc397a7c4b24fc3f8aa6de3b4e977f43d969ed1c313ab52434"
PROTOCOL_SHA256 = "1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3"
PARENT_PROTOCOL_SHA256 = "a2662cfae992380524725f1457a412e4aabda9662fcd597f9245ff5b911b6cf2"


def _protocol_body_sha256(raw: bytes) -> str:
    marker = b"\nPROTOCOL_SHA256: "
    head, separator, tail = raw.rpartition(marker)
    if not separator:
        raise AssertionError("protocol hash line missing")
    digest = hashlib.sha256(head).hexdigest()
    if tail.strip().decode("utf-8") != digest:
        raise AssertionError("recorded protocol hash does not match the body")
    return digest


def test_catalog_audit_passes_without_outcomes() -> None:
    result = audit()
    assert result["status"] == "PASS"
    assert result["catalog_sha256"] == CATALOG_SHA256
    assert result["n_rows"] == 48
    assert result["partitions"] == {"update": 12, "validation": 12, "holdout": 24}
    assert result["qwen_loaded"] is False
    assert result["outcome_data_loaded"] is False
    assert result["m29_holdout_statistics_loaded"] is False
    assert result["historical_catalogs_checked"] == [
        "M22.1",
        "M23-G",
        "POST-M23-SINGLE",
        "FEASIBILITY",
        "M24",
        "M26",
        "M29-D",
    ]
    assert len(catalog()) == 48
    for name, item in result["overlaps"].items():
        assert item["id_overlap"] == [], name
        assert item["text_overlap"] == [], name
        assert item["normalized_text_overlap"] == [], name
    for partition, item in result["internal_overlaps"].items():
        assert item["id_overlap"] == [], partition
        assert item["text_overlap"] == [], partition
        assert item["normalized_text_overlap"] == [], partition


def test_update_and_validation_hashes_match_the_previous_revision() -> None:
    assert partition_manifest_sha256("update") == PREVIOUS_UPDATE_SHA256
    assert partition_manifest_sha256("validation") == PREVIOUS_VALIDATION_SHA256
    rows = catalog()
    for partition, expected in (("update", 12), ("validation", 12)):
        matched = [row for row in rows if row["partition"] == partition]
        assert len(matched) == expected
        assert all(int(row["prompt_id"].rsplit("-", 1)[1]) <= 12 for row in matched)
    retained_holdout = sorted(
        (
            row
            for row in rows
            if row["partition"] == "holdout" and int(row["prompt_id"].rsplit("-", 1)[1]) <= 12
        ),
        key=lambda row: row["prompt_id"],
    )
    encoded = json.dumps(retained_holdout, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(retained_holdout) == 12
    assert hashlib.sha256(encoded).hexdigest() == PREVIOUS_HOLDOUT_SHA256


def test_new_holdout_is_the_image_of_rule_m30_hx1() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert HOLDOUT_EXTENSION_RULE == "M30-HX1"
    assert "M30-HX1" in text
    for word in (
        *_EXTENSION_MATERIALS,
        *_EXTENSION_ARTICLES,
        *_EXTENSION_IMPLEMENTS,
        *_EXTENSION_TARGETS,
    ):
        assert word in text
    for template in _EXTENSION_TEMPLATES.values():
        assert template in text
    extension = [row for row in catalog() if int(row["prompt_id"].rsplit("-", 1)[1]) >= 13]
    assert len(extension) == 12
    families = {"completion": 0, "syntax": 0, "instruction": 0}
    for row in extension:
        index = int(row["prompt_id"].rsplit("-", 1)[1])
        families[row["family"]] += 1
        assert row["partition"] == "holdout"
        assert row["text"] == holdout_extension_text(row["family"], index - 13)
    assert families == {"completion": 4, "syntax": 4, "instruction": 4}
    holdout_counts = {
        family: sum(row["family"] == family and row["partition"] == "holdout" for row in catalog())
        for family in families
    }
    assert holdout_counts == {"completion": 8, "syntax": 8, "instruction": 8}


def test_catalog_hash_matches_protocol_audit_and_module() -> None:
    digest = catalog_sha256()
    assert digest == CATALOG_SHA256
    assert digest == "4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770"
    protocol = PROTOCOL.read_text(encoding="utf-8")
    report = AUDIT_REPORT.read_text(encoding="utf-8")
    assert digest in protocol
    assert digest in report
    assert audit()["catalog_sha256"] == digest


def test_protocol_revision_hash_and_parent_diff() -> None:
    raw = PROTOCOL.read_bytes()
    text = raw.decode("utf-8")
    assert _protocol_body_sha256(raw) == PROTOCOL_SHA256
    assert f"PROTOCOL_SHA256: {PROTOCOL_SHA256}" in text
    assert PARENT_PROTOCOL_SHA256 in text
    assert "SUPPORTED_WEAK" in text
    assert "1 - MAE(g+kappa)/MAE(g) >= 0.25" in text
    assert "ANCHOR_SUPPORTED" in text
    assert "The anchor does not change the new-identity verdict." in text


def test_protocol_locks_the_new_identity_and_the_g_baseline() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "DESIGN_LOCKED" in text
    assert "M30-D2-L4" in text
    assert "M30-D2-L12" in text
    assert "M30-D2-L20" in text
    assert "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e" in text
    assert '" true"' in text
    assert '" false"' in text
    assert "MAE(g) - MAE(g + kappa)" in text
    assert "5000" in text
    assert "23001" in text
    assert "B1 does not choose the verdict." in text
    assert "No Qwen prediction" in text
    assert "0.012483114971675806" not in text
    assert "M22.1-D1-L0" in text
    assert "M22.1-D1-L8" in text
    assert "M22.1-D1-L15" in text
    assert "22101" in text
    assert "9834" in text
    assert "902" in text
    assert "median(|kappa|)" in text
    assert "median(|g|)" in text
