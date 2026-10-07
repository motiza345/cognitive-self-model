"""M30 is a design lock: disjoint catalog, frozen identity, no outcomes."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_m30_catalog import audit
from scripts.m30_catalog import CATALOG_SHA256, catalog

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "M30_PROTOCOL.md"


def test_catalog_audit_passes_without_outcomes() -> None:
    result = audit()
    assert result["status"] == "PASS"
    assert result["catalog_sha256"] == CATALOG_SHA256
    assert result["n_rows"] == 36
    assert result["qwen_loaded"] is False
    assert result["outcome_data_loaded"] is False
    assert result["m29_holdout_statistics_loaded"] is False
    assert result["historical_catalogs_checked"][-1] == "M29-D"
    assert len(catalog()) == 36


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
