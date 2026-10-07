"""One mock-model smoke test for the pre-registered S1 formulas."""

from __future__ import annotations

import math

import torch
from torch import nn

from scripts.m30_catalog import catalog
from scripts.run_s1_steering import (
    assert_prediction_lock,
    beats_m1,
    clip_alpha,
    contrast,
    decide,
    gbar_from_update,
    main,
    margin_at,
    pick_root,
    probe_sequence,
    solve_m2,
    time_forward,
    time_gradient,
    time_hvp,
)


class _Mock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))

    def to_tokens(self, text: str, prepend_bos: bool = True) -> torch.Tensor:
        del text, prepend_bos
        return torch.ones(1, 3, dtype=torch.long)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        del tokens
        resid = torch.zeros(1, 3, 896, dtype=torch.float32) + self.anchor
        return self._logits(resid)

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks: list) -> torch.Tensor:
        del tokens
        resid = torch.zeros(1, 3, 896, dtype=torch.float32) + self.anchor
        for _name, fn in fwd_hooks:
            resid = fn(resid, hook=None)
        return self._logits(resid)

    def _logits(self, resid: torch.Tensor) -> torch.Tensor:
        signal = resid[:, -1, 0]
        curved = signal * signal
        columns = torch.stack((curved * 0.0, signal + curved, curved, curved), dim=-1)
        return columns.view(1, 1, 4).expand(1, resid.shape[1], 4).contiguous()


def test_s1_smoke_mock_model(monkeypatch, tmp_path) -> None:
    assert solve_m2(1.0, 0.0, 4) == 4.0
    assert solve_m2(1.0, -1.0, 1) == 0.5
    assert abs(solve_m2(1.0, -0.16, 1) - 1.25) < 1e-9
    assert clip_alpha(100.0, 1) == 4.0
    assert clip_alpha(-9.0, 2) == -8.0
    assert pick_root(5.0, -1.0, 2.0) == -1.0

    calls = {"n": 0}

    def quadratic(alpha: float) -> float:
        calls["n"] += 1
        return alpha * alpha

    probes, estimates = probe_sequence(quadratic, 4.0, 1.0, 3)
    assert calls["n"] == 3
    assert len(probes) == 3
    assert estimates[0] == 1.0
    assert estimates[1] == 4.0
    assert estimates[2] == 1.0
    assert probes[1]["alpha"] == estimates[0]

    update = []
    for index in range(12):
        update.append(
            {
                "arm": "new_identity",
                "intervention_id": "M30-D2-L4",
                "observed_effect": 1.0,
                "partition": "update",
                "prompt_id": f"u-{index}",
            }
        )
    for arm, cells in (
        ("new_identity", ("M30-D2-L12", "M30-D2-L20")),
        ("anchor", ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")),
    ):
        for cell in cells:
            for index in range(12):
                update.append(
                    {
                        "arm": arm,
                        "intervention_id": cell,
                        "observed_effect": 2.0,
                        "partition": "update",
                        "prompt_id": f"{arm}-{cell}-{index}",
                    }
                )
    update.append(
        {
            "arm": "new_identity",
            "intervention_id": "M30-D2-L4",
            "observed_effect": 1000.0,
            "partition": "holdout",
            "prompt_id": "holdout-must-not-enter",
        }
    )
    assert gbar_from_update(update)[("new_identity", "M30-D2-L4")] == 1.0

    model = _Mock()
    tokens = model.to_tokens("unused")
    direction = torch.zeros(896).numpy()
    direction[0] = 1.0
    base = margin_at(model, tokens, "blocks.4.hook_resid_post", direction, 0.0, 1, 0)
    shifted = margin_at(model, tokens, "blocks.4.hook_resid_post", direction, 1.0, 1, 0)
    assert math.isfinite(base) and math.isfinite(shifted) and shifted != base
    assert time_forward(model, tokens, "blocks.4.hook_resid_post", direction, 1, 0) > 0.0
    assert time_gradient(model, tokens, "blocks.4.hook_resid_post", direction, 1, 0) > 0.0
    assert time_hvp(model, tokens, 4, "blocks.4.hook_resid_post", direction, 1, 0) > 0.0

    ones = [1.0] * 24
    half = [0.5] * 24
    low = [0.4] * 24
    mild = [0.9] * 24
    strong = contrast(ones, half)
    weak = contrast(ones, mild)
    assert beats_m1(strong) is True
    assert beats_m1(weak) is False
    assert decide(
        [
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
        ]
    ) == "GO"
    assert decide(
        [
            {"beat": True, "median_m2": 0.5, "median_bb": 0.4},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.4},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.4},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.4},
        ]
    ) == "PIVOT"
    assert decide(
        [
            {"beat": False, "median_m2": 0.9, "median_bb": 0.4},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
            {"beat": True, "median_m2": 0.5, "median_bb": 0.6},
        ]
    ) == "STOP"
    assert decide(
        [
            {"beat": True, "median_m2": 0.4, "median_bb": 0.5},
            {"beat": True, "median_m2": 0.4, "median_bb": 0.5},
            {"beat": True, "median_m2": 0.5, "median_bb": low[0]},
            {"beat": True, "median_m2": 0.5, "median_bb": low[0]},
        ]
    ) == "MIXED"
    holdout_ids = [row["prompt_id"] for row in catalog() if row["partition"] == "holdout"]
    assert len(holdout_ids) == 24
    digest, predictions = assert_prediction_lock()
    assert digest == "f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5"
    assert len(predictions) == 288

    monkeypatch.setattr("scripts.run_s1_steering.REPORT_PATH", tmp_path / "S1_REPORT.md")
    monkeypatch.setattr("scripts.run_s1_steering.RAW_PATH", tmp_path / "ROWS.json")
    monkeypatch.setattr("scripts.run_s1_steering.EXPECTED_PREDICTION_SHA256", "0" * 64)

    def fail_if_called():
        raise AssertionError("model loaded")

    monkeypatch.setattr("scripts.run_s1_steering._load_model", fail_if_called)
    try:
        main()
    except SystemExit as exc:
        assert "prediction sha256" in str(exc)
    else:
        raise AssertionError("hash mismatch did not abort")
