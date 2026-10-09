"""Mock-model smoke test for the pre-registered S2 finite-difference control."""

from __future__ import annotations

import math

import torch
from torch import nn

from scripts.run_s1_steering import solve_m2
from scripts.run_s2_fdq import (
    fdq_coefficients,
    margin_at,
    paired_median_ci,
    predicates,
    solve_fdq,
    summarize,
    trimmed_mean,
    verdict_from_counts,
)


class _Mock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.25))

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


def test_s2_smoke_mock_model() -> None:
    assert solve_fdq(1.0, 0.0, 4.0, 4) == solve_m2(1.0, 0.0, 4)
    assert solve_fdq(1.0, -1.0, 1.0, 1) == solve_m2(1.0, -1.0, 1)
    assert abs(solve_fdq(1.0, -0.16, 1.0, 1) - solve_m2(1.0, -0.16, 1)) < 1e-12
    linear, quad = fdq_coefficients(3.0, 3.0 + 0.5 + 0.25, 3.0 - 0.5 + 0.25)
    assert abs(linear - 0.5) < 1e-12
    assert abs(quad - 0.25) < 1e-12
    alpha = solve_fdq(linear, quad, 1.0, 2)
    predicted = linear * alpha + quad * alpha * alpha
    assert abs(predicted - 1.0) < 1e-9

    values = list(range(1, 25))
    assert trimmed_mean([float(value) for value in values]) == 12.5
    interval = paired_median_ci([0.2] * 24)
    assert interval["median"] == 0.2
    assert interval["low"] > 0.0
    try:
        predicates(0.0, 0.0, 1.0)
    except RuntimeError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("overlapping predicates did not abort")
    assert predicates(1.0, 2.0, 0.1) == (True, False)
    assert predicates(2.0, 1.0, -1.0) == (False, True)
    assert verdict_from_counts(3, 1) == "WHITE_BOX_ADVANTAGE"
    assert verdict_from_counts(4, 0) == "WHITE_BOX_ADVANTAGE"
    assert verdict_from_counts(1, 3) == "NO_ADVANTAGE"
    assert verdict_from_counts(2, 2) == "MIXED"
    try:
        verdict_from_counts(3, 3)
    except RuntimeError as exc:
        assert "both" in str(exc)
    else:
        raise AssertionError("overlapping verdicts did not abort")

    model = _Mock()
    tokens = torch.ones(1, 3, dtype=torch.long)
    direction = torch.zeros(896).numpy()
    direction[0] = 1.0
    base = margin_at(model, tokens, "blocks.4.hook_resid_post", direction, 0.0, 1, 0)
    plus = margin_at(model, tokens, "blocks.4.hook_resid_post", direction, 1.0, 1, 0)
    minus = margin_at(model, tokens, "blocks.4.hook_resid_post", direction, -1.0, 1, 0)
    assert math.isfinite(base) and plus != base and minus != base
    recovered_b, recovered_c = fdq_coefficients(base, plus, minus)
    assert math.isfinite(recovered_b) and math.isfinite(recovered_c)
    assert math.isfinite(solve_fdq(recovered_b, recovered_c, 0.1, 1))

    def grid(m1: float, m2: float, fdq: float) -> list[dict]:
        rows = []
        for arm in ("new_identity", "anchor"):
            for scale in (1, 2, 4, 8):
                for prompt_index in range(24):
                    for cell in ("c0", "c1", "c2"):
                        rows.append(
                            {
                                "arm": arm,
                                "intervention_id": cell,
                                "miss_bb_3": m1,
                                "miss_fdq": fdq,
                                "miss_m1": m1,
                                "miss_m2": m2,
                                "prompt_id": f"p{prompt_index:02d}",
                                "s": scale,
                            }
                        )
        return rows

    no_advantage = summarize(grid(1.0, 1.0, 1.0))
    assert no_advantage["verdict"] == "NO_ADVANTAGE"
    assert no_advantage["n_no_advantage"] == 4
    assert no_advantage["n_white"] == 0
    white = summarize(grid(1.0, 1.0, 2.0))
    assert white["verdict"] == "WHITE_BOX_ADVANTAGE"
    assert white["n_white"] == 4
    assert white["n_neither"] == 0
