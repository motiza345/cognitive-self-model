"""The post-hoc supplement is descriptive and leaves the verdict file untouched."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.m29d_post_hoc_supplement import OUTPUT_PATH, VERDICT_PATH, render, summarize

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "m29d_raw" / "PRE_OUTCOME_PREDICTIONS.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_holdout_supplement_matches_the_committed_rows() -> None:
    rows = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))["holdout"]["paired_rows"]
    summary = summarize(rows)
    assert summary["mae_g"] == pytest.approx(0.012483114971675806)
    assert summary["delta_vs_g"] == pytest.approx(0.00796987532117378)
    assert summary["bootstrap"]["low"] == pytest.approx(0.0047422432851615464)
    assert summary["bootstrap"]["high"] == pytest.approx(0.011489202147155689)
    assert summary["bootstrap"]["draws"] == 5000
    assert summary["bootstrap"]["seed"] == 23001
    assert summary["pearson_kappa_residual"] == pytest.approx(0.8639837450934315)
    assert (summary["sign_matches"], summary["sign_rows"]) == (34, 36)
    ratios = {cell["intervention_id"]: cell["median_ratio"] for cell in summary["cells"]}
    assert ratios["M22.1-D1-L0"] == pytest.approx(0.854869480950888)
    assert ratios["M22.1-D1-L8"] == pytest.approx(1.085331999567653)
    assert ratios["M22.1-D1-L15"] == pytest.approx(0.9781572212719309)
    text = render(summary)
    assert "were not pre-registered" in text
    assert "cannot change the verdict" in text
    assert "POST_HOC" in text


def test_writing_the_supplement_does_not_touch_frozen_artifacts(tmp_path: Path, monkeypatch) -> None:
    verdict_before = _sha(VERDICT_PATH)
    prediction_before = _sha(PREDICTIONS)
    destination = tmp_path / "M29_D_POST_HOC_SUPPLEMENT.md"
    monkeypatch.setattr("scripts.m29d_post_hoc_supplement.OUTPUT_PATH", destination)
    from scripts.m29d_post_hoc_supplement import main

    main()
    assert destination.is_file()
    assert _sha(VERDICT_PATH) == verdict_before
    assert _sha(PREDICTIONS) == prediction_before
    assert OUTPUT_PATH.name == "M29_D_POST_HOC_SUPPLEMENT.md"
