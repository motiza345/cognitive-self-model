"""Schema checks for the one completed M22.1.1 run."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "M22_1_1"
LABELS = {
    "BROAD_RESIDUAL_SENSITIVITY",
    "STRUCTURED_INTERVENTION_SPACE",
    "LOW_DIMENSIONAL_SENSITIVITY",
    "REGIME_DEPENDENT_SENSITIVITY",
    "INSUFFICIENT_EVIDENCE",
}


def test_audit_artifacts_match_the_frozen_schema():
    manifest = json.loads((REPORT_DIR / "direction_panel_manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((REPORT_DIR / "direction_panel_results.json").read_text(encoding="utf-8"))
    report = json.loads((REPORT_DIR / "intervention_space_report.json").read_text(encoding="utf-8"))
    readme = (REPORT_DIR / "README.md").read_text(encoding="utf-8")
    assert manifest["frozen_before_measurement"] is True
    assert manifest["dimension"] == 896
    assert [row["direction_id"] for row in manifest["directions"]] == [f"D{i}" for i in range(1, 9)]
    assert manifest["directions"][0]["final_sha256"] == "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
    assert manifest["directions"][1]["final_sha256"] == "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
    assert all(abs(row["norm"] - 1.0) < 1e-8 for row in manifest["directions"])
    assert payload["environment"]["model_revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert payload["environment"]["d_model"] == 896
    records = payload["records"]
    assert len(records) == 8 * 5 * 18
    assert {row["hook_name_seen"] for row in records} == {"blocks.23.hook_resid_post"}
    assert {row["alpha"] for row in records} == {-2.0, -1.0, 0.0, 1.0, 2.0}
    assert {row["role_split"] for row in records} == {"discovery", "validation", "replication"}
    assert report["m22_1_status"] == "CANDIDATE"
    assert report["m22_2_authorized"] is False
    assert report["classification"] in LABELS
    assert report["classification_is_descriptive"] is True
    assert "validation" in report["per_split"]
    assert "functional_correlations_alpha_plus1" in report
    assert "not a SelfModel" in report["non_claims"]
    assert report["classification"] in readme
    assert "M22.2" in readme
    assert not (REPORT_DIR / "validated_intervention.json").exists()
