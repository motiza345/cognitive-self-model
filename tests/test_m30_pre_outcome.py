"""M30 stage 1 stays on the pre-outcome side of the frozen measurement."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

import scripts.m29d_measurement as measurement
from scripts.m30_catalog import catalog
from scripts.run_m30_pre_outcome import (
    ANCHOR_CELLS,
    NEW_CELLS,
    ArmSpec,
    assert_catalog_lock,
    load_directions,
    magnitude_check,
    main,
    predict_catalog,
    resolve_anchor_tokens,
    resolve_new_identity_tokens,
    resolve_single_token,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_m30_pre_outcome.py"


class _Tokenizer:
    def __init__(self, table: dict[str, list[int]]) -> None:
        self.table = table

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.table[text])


class _MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))
        self.hook_names: list[str] = []
        self.token_calls = 0
        self.outcome_calls = 0

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        del text, prepend_bos
        self.token_calls += 1
        return torch.ones(1, 3, dtype=torch.long)

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: list) -> torch.Tensor:
        del tokens
        resid = torch.zeros(1, 3, 896, dtype=torch.float32) + self.anchor
        for name, fn in fwd_hooks:
            self.hook_names.append(name)
            resid = fn(resid, None)
        base = resid.reshape(-1).sum()
        curved = base * base
        columns = [curved * float(index + 1) for index in range(4)]
        return torch.stack(columns).view(1, 1, 4).expand(1, 3, 4).contiguous()

    def score_outcome(self) -> None:
        self.outcome_calls += 1
        raise AssertionError("outcome path called")


def test_runner_source_has_no_outcome_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "run_m29d_outcome" not in source
    assert "y_hat" not in source
    assert "score_outcome" not in source
    assert "def measure_outcome" not in source
    assert "HookedTransformer.from_pretrained" not in source or "device=DEVICE_NAME" in source


def test_mock_model_predictions_do_not_compute_outcomes() -> None:
    saved_hooks = dict(measurement.HOOKS)
    saved_positive = measurement.POSITIVE_ID
    saved_negative = measurement.NEGATIVE_ID
    direction_d1, direction_d2 = load_directions()
    model = _MockModel()
    arms = [
        ArmSpec("new_identity", direction_d2, 1, 0, NEW_CELLS),
        ArmSpec("anchor", direction_d1, 1, 0, ANCHOR_CELLS),
    ]
    rows = predict_catalog(model, catalog(), arms)
    assert model.outcome_calls == 0
    assert model.token_calls == 48
    assert len(model.hook_names) == 288
    assert len(rows) == 288
    assert {row["outcome_present"] for row in rows} == {False}
    assert {row["arm"] for row in rows} == {"new_identity", "anchor"}
    forbidden = {"observed_effect", "intervened_output", "baseline_output", "residual", "y", "r"}
    for row in rows:
        assert forbidden.isdisjoint(row)
        assert math.isfinite(row["g"])
        assert math.isfinite(row["kappa"])
        assert math.isfinite(row["f_second"])
    assert measurement.HOOKS == saved_hooks
    assert measurement.POSITIVE_ID == saved_positive
    assert measurement.NEGATIVE_ID == saved_negative
    new_hooks = [name for name in model.hook_names if name.startswith("blocks.4.")]
    assert new_hooks == ["blocks.4.hook_resid_post"] * 48


def test_catalog_hash_mismatch_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.run_m30_pre_outcome.CATALOG_SHA256", "0" * 64)

    def fail_if_called():
        raise AssertionError("model loaded after a catalog mismatch")

    monkeypatch.setattr("scripts.run_m30_pre_outcome._load_model", fail_if_called)
    with pytest.raises(SystemExit, match="catalog hash mismatch"):
        main()


def test_catalog_lock_accepts_the_frozen_hash() -> None:
    assert assert_catalog_lock() == "4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770"


def test_multi_token_text_aborts_without_substitution() -> None:
    tokenizer = _Tokenizer({" true": [4, 5], " false": [6]})
    with pytest.raises(SystemExit, match="not a single token"):
        resolve_new_identity_tokens(tokenizer)
    with pytest.raises(SystemExit, match="not a single token"):
        resolve_single_token(tokenizer, " true")


def test_anchor_token_ids_must_be_the_frozen_pair() -> None:
    tokenizer = _Tokenizer({" yes": [111], " no": [902]})
    with pytest.raises(SystemExit, match="anchor token ids"):
        resolve_anchor_tokens(tokenizer)
    tokenizer = _Tokenizer({" yes": [9834], " no": [902]})
    assert resolve_anchor_tokens(tokenizer) == (9834, 902)


def test_magnitude_check_uses_update_rows_only() -> None:
    rows = []
    for index in range(12):
        for cell in ("M30-D2-L4", "M30-D2-L12", "M30-D2-L20"):
            rows.append(
                {
                    "arm": "new_identity",
                    "partition": "update",
                    "intervention_id": cell,
                    "prompt_id": f"u-{index}",
                    "g": 2.0,
                    "kappa": 1.0,
                }
            )
    rows.append(
        {
            "arm": "new_identity",
            "partition": "holdout",
            "intervention_id": "M30-D2-L4",
            "prompt_id": "holdout-nan",
            "g": float("nan"),
            "kappa": 0.0,
        }
    )
    result = magnitude_check(rows, "new_identity", ("M30-D2-L4", "M30-D2-L12", "M30-D2-L20"))
    assert result["finite"] is True
    assert result["passed"] is True
    assert result["cells"][0]["median_abs_g"] == 2.0
    assert result["cells"][0]["median_abs_kappa"] == 1.0
    assert result["cells"][0]["ratio"] == 0.5
    rows[0]["kappa"] = 0.0
    for row in rows:
        if row["partition"] == "update" and row["intervention_id"] == "M30-D2-L4":
            row["kappa"] = 0.0
    failed = magnitude_check(rows, "new_identity", ("M30-D2-L4", "M30-D2-L12", "M30-D2-L20"))
    assert failed["passed"] is False
    assert failed["cells"][0]["median_abs_kappa"] == 0.0
