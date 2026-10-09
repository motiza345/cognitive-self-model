"""M30 stage 2 locks the prediction hash and the section 6 verdict rule."""

from __future__ import annotations

from pathlib import Path

from scripts.m30_catalog import catalog
from scripts.run_m30_outcomes import (
    COPIED_FIELDS,
    EXPECTED_PREDICTION_SHA256,
    assert_prediction_lock,
    classify_verdict,
    holdout_cohort,
    interpret_two_by_two,
    leakage_audit,
    main,
    render_report,
    score_arm,
)

ROOT = Path(__file__).resolve().parents[1]
CELLS = ("M30-D2-L4", "M30-D2-L12", "M30-D2-L20")
ANCHOR_CELLS = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")


def _prompts(partition: str) -> list[dict[str, str]]:
    return [row for row in catalog() if row["partition"] == partition]


def _rows(y_of, *, arm: str = "new_identity", cells=CELLS, validation_y=None) -> list[dict]:
    rows = []
    for partition in ("update", "holdout", "validation"):
        prompts = _prompts(partition)
        for prompt in prompts:
            for cell in cells:
                if partition == "validation" and validation_y is not None:
                    observed = validation_y
                else:
                    observed = y_of(prompt, cell, partition)
                rows.append(
                    {
                        "alpha": 1.0,
                        "arm": arm,
                        "f_second": 0.0,
                        "family": prompt["family"],
                        "g": 0.0,
                        "hook": "unused",
                        "intervention_id": cell,
                        "kappa": y_of(prompt, cell, "kappa"),
                        "observed_effect": observed,
                        "partition": partition,
                        "prompt_id": prompt["prompt_id"],
                    }
                )
    return rows


def _constant(kappa: float, y: float):
    def y_of(prompt, cell, partition):
        del prompt, cell
        if partition == "kappa":
            return kappa
        return y

    return y_of


def test_prediction_lock_matches_frozen_hash() -> None:
    digest, predictions = assert_prediction_lock()
    assert digest == EXPECTED_PREDICTION_SHA256
    assert len(predictions) == 288


def _empty_outcome_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("scripts.run_m30_outcomes.OUTCOMES_PATH", tmp_path / "OUTCOMES.json")
    monkeypatch.setattr("scripts.run_m30_outcomes.VERDICT_PATH", tmp_path / "VERDICT.json")
    monkeypatch.setattr("scripts.run_m30_outcomes.REPORT_PATH", tmp_path / "REPORT.md")


def test_prediction_hash_mismatch_aborts_before_model(monkeypatch, tmp_path: Path) -> None:
    _empty_outcome_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "scripts.run_m30_outcomes.EXPECTED_PREDICTION_SHA256",
        "0" * 64,
    )

    def fail_if_called():
        raise AssertionError("model loaded")

    monkeypatch.setattr("scripts.run_m30_outcomes._load_model", fail_if_called)
    try:
        main()
    except SystemExit as exc:
        assert "prediction sha256" in str(exc)
    else:
        raise AssertionError("hash mismatch did not abort")


def test_catalog_hash_mismatch_aborts_before_model(monkeypatch, tmp_path: Path) -> None:
    _empty_outcome_paths(monkeypatch, tmp_path)
    monkeypatch.setattr("scripts.run_m30_pre_outcome.LOCKED_CATALOG_SHA256", "0" * 64)

    def fail_if_called():
        raise AssertionError("model loaded")

    monkeypatch.setattr("scripts.run_m30_outcomes._load_model", fail_if_called)
    try:
        main()
    except SystemExit as exc:
        assert "catalog hash" in str(exc)
    else:
        raise AssertionError("catalog mismatch did not abort")


def test_magnitude_mismatch_aborts_before_model(monkeypatch, tmp_path: Path) -> None:
    _empty_outcome_paths(monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        del args, kwargs
        return {"arm": "x", "cells": [], "finite": False, "passed": False}

    monkeypatch.setattr("scripts.run_m30_outcomes.magnitude_check", fail)

    def fail_if_called():
        raise AssertionError("model loaded")

    monkeypatch.setattr("scripts.run_m30_outcomes._load_model", fail_if_called)
    try:
        main()
    except SystemExit as exc:
        assert "magnitude" in str(exc)
    else:
        raise AssertionError("magnitude failure did not abort")


def test_classify_clause_order() -> None:
    supported = dict(leakage_ok=True, delta=1.0, ci_low=0.1, mae_g=1.0, relative_reduction=1.0, anchor=False)
    assert classify_verdict(**supported) == "SUPPORTED"
    assert classify_verdict(**{**supported, "anchor": True}) == "ANCHOR_SUPPORTED"
    assert classify_verdict(**{**supported, "leakage_ok": False}) == "INCONCLUSIVE"
    assert classify_verdict(**{**supported, "delta": 0.0, "leakage_ok": True}) == "NOT_SUPPORTED"
    assert classify_verdict(**{**supported, "delta": -1.0}) == "NOT_SUPPORTED"
    assert classify_verdict(**{**supported, "ci_low": 0.0}) == "INCONCLUSIVE"
    assert classify_verdict(**{**supported, "relative_reduction": 0.249}) == "SUPPORTED_WEAK"
    assert classify_verdict(**{**supported, "relative_reduction": 0.25}) == "SUPPORTED"
    assert classify_verdict(**{**supported, "mae_g": 0.0, "relative_reduction": None}) == "SUPPORTED_WEAK"
    assert classify_verdict(**{**supported, "anchor": True, "relative_reduction": 0.1}) == "ANCHOR_SUPPORTED_WEAK"


def test_two_by_two_counts_weak_as_not_supported() -> None:
    landed = interpret_two_by_two("SUPPORTED_WEAK", "ANCHOR_SUPPORTED")
    assert landed["new_identity_supported"] is False
    assert landed["anchor_supported"] is True
    assert landed["interpretation"] == (
        "The inherited identity beats `g` on this catalog. The new identity does not meet `SUPPORTED`."
    )
    assert interpret_two_by_two("SUPPORTED", "ANCHOR_NOT_SUPPORTED")["interpretation"].startswith(
        "The new identity meets `SUPPORTED`"
    )


def test_perfect_match_is_supported_at_prompt_level(monkeypatch) -> None:
    seen: list[int] = []
    real = __import__("scripts.run_m30_outcomes", fromlist=["paired_mean_ci"]).paired_mean_ci

    def wrap(differences, **kwargs):
        seen.append(len(differences))
        return real(differences, **kwargs)

    monkeypatch.setattr("scripts.run_m30_outcomes.paired_mean_ci", wrap)
    rows = _rows(_constant(kappa=1.0, y=1.0))
    poisoned = [dict(row, abs_error_g=999.0) for row in rows]
    forward = score_arm(poisoned, arm="new_identity", leakage_ok=True)
    reverse = score_arm(list(reversed(poisoned)), arm="new_identity", leakage_ok=True)
    assert forward["verdict"] == "SUPPORTED"
    assert forward["delta"] == 1.0
    assert forward["relative_reduction"] == 1.0
    assert forward["bootstrap"]["n_prompts"] == 24
    assert forward["bootstrap"]["low"] > 0.0
    assert seen[0] == 24
    assert forward["bootstrap"] == reverse["bootstrap"]
    assert forward["holdout_cohorts"]["old"]["n_prompts"] == 12
    assert forward["holdout_cohorts"]["new"]["n_prompts"] == 12


def test_small_reduction_is_supported_weak() -> None:
    scored = score_arm(_rows(_constant(kappa=0.2, y=1.0)), arm="new_identity", leakage_ok=True)
    assert scored["verdict"] == "SUPPORTED_WEAK"
    assert scored["bootstrap"]["low"] > 0.0
    assert scored["relative_reduction"] < 0.25
    assert scored["delta"] > 0.0


def test_ci_including_zero_is_inconclusive() -> None:
    def y_of(prompt, cell, partition):
        del cell
        special = prompt["prompt_id"].endswith("-03") and prompt["family"] == "completion"
        if partition == "kappa":
            return 1.0 if special else 0.0
        return 1.0 if special else 0.0

    scored = score_arm(_rows(y_of), arm="new_identity", leakage_ok=True)
    assert scored["delta"] > 0.0
    assert scored["bootstrap"]["low"] <= 0.0
    assert scored["verdict"] == "INCONCLUSIVE"


def test_worse_kappa_is_not_supported() -> None:
    scored = score_arm(_rows(_constant(kappa=1.0, y=0.0)), arm="new_identity", leakage_ok=True)
    assert scored["delta"] < 0.0
    assert scored["verdict"] == "NOT_SUPPORTED"


def test_validation_and_holdout_do_not_change_b1() -> None:
    base = score_arm(_rows(_constant(kappa=1.0, y=1.0)), arm="new_identity", leakage_ok=True)
    shifted = score_arm(
        _rows(_constant(kappa=1.0, y=1.0), validation_y=1000.0),
        arm="new_identity",
        leakage_ok=True,
    )
    assert shifted["verdict"] == base["verdict"]
    assert shifted["b1"]["cell_means"] == base["b1"]["cell_means"]
    assert shifted["delta"] == base["delta"]


def test_b1_uses_update_only() -> None:
    def y_of(prompt, cell, partition):
        del prompt, cell
        if partition == "kappa":
            return 0.0
        if partition == "update":
            return 2.0
        return 50.0

    scored = score_arm(_rows(y_of), arm="new_identity", leakage_ok=True)
    assert set(scored["b1"]["cell_means"].values()) == {2.0}
    assert scored["b1"]["gate"] is False


def test_holdout_split_matches_catalog() -> None:
    holdout = _prompts("holdout")
    old = [row for row in holdout if holdout_cohort(row["prompt_id"]) == "old"]
    new = [row for row in holdout if holdout_cohort(row["prompt_id"]) == "new"]
    assert len(old) == 12
    assert len(new) == 12
    for family in ("completion", "syntax", "instruction"):
        assert sum(row["family"] == family for row in old) == 4
        assert sum(row["family"] == family for row in new) == 4


def test_runner_does_not_score_with_the_m29_rule() -> None:
    source = (ROOT / "scripts" / "run_m30_outcomes.py").read_text(encoding="utf-8")
    assert "score_m29" not in source
    assert "verdict_from_criteria" not in source
    assert "m29d_raw" not in source
    assert "measure_kappa" not in source


def test_leakage_audit_passes_for_copied_predictions() -> None:
    _, predictions = assert_prediction_lock()
    outcomes = []
    for row in predictions:
        outcome = {field: row[field] for field in COPIED_FIELDS}
        outcome["observed_effect"] = 0.0
        outcomes.append(outcome)
    audit = leakage_audit(
        predictions=predictions,
        outcomes=outcomes,
        magnitude_passed=True,
        token_ids_ok=True,
        direction_hashes_ok=True,
    )
    assert audit["passed"] is True


def test_report_quotes_the_verdict_dict() -> None:
    scored = score_arm(_rows(_constant(kappa=1.0, y=1.0)), arm="new_identity", leakage_ok=True)
    anchor = score_arm(
        _rows(_constant(kappa=0.2, y=1.0), arm="anchor", cells=ANCHOR_CELLS),
        arm="anchor",
        leakage_ok=True,
    )
    verdict = {
        "anchor_verdict": anchor["verdict"],
        "arms": {"anchor": anchor, "new_identity": scored},
        "catalog_sha256": "catalog",
        "git_commit": "commit",
        "leakage": {"passed": True},
        "new_identity_verdict": scored["verdict"],
        "prediction_sha256": "prediction",
        "protocol_sha256": "protocol",
        "two_by_two": interpret_two_by_two(scored["verdict"], anchor["verdict"]),
    }
    report = render_report(verdict)
    assert "`SUPPORTED`" in report
    assert "`ANCHOR_SUPPORTED_WEAK`" in report
    assert verdict["two_by_two"]["interpretation"] in report
    assert "12" in report
