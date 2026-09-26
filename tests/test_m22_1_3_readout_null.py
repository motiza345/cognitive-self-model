"""M22.1.3 readout-null tests. Predict must never open the measurement table."""

import ast
import builtins
from pathlib import Path

import numpy as np
import pytest
import torch

from src.cognitive_self_model.m22_1.direction import vector_sha256
from src.cognitive_self_model.m22_1_1.panel import build_direction_panel, load_audit_config
from src.cognitive_self_model.m22_1_3.predict import (
    build_cell_keys,
    closed_form_margin,
    jacobian_first_order,
    load_audit_yaml,
    load_prompt_records,
    module_margin,
    rms_scale,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cognitive_self_model" / "m22_1_3"
DELTA_NAME = "direction_panel_results.json"


def test_predict_source_never_names_the_measurement_table():
    source = (PACKAGE / "predict.py").read_text(encoding="utf-8")
    assert DELTA_NAME not in source
    assert "actual_delta" not in source
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("compare" in name for name in imported)
    assert not any("m22_1_2" in name for name in imported)


def test_predict_runtime_does_not_open_the_measurement_table(monkeypatch):
    real_open = builtins.open

    def guarded_open(path, *args, **kwargs):
        if DELTA_NAME in str(path):
            raise AssertionError("predict opened the measurement table")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    config = load_audit_yaml()
    prompts = load_prompt_records(config)
    cells = build_cell_keys(prompts, list(config["direction_ids"]), [float(value) for value in config["alphas"]])
    assert len(prompts) == 18
    assert len(cells) == 18 * 8 * 5
    assert {cell["prompt_id"] for cell in cells} == {prompt["prompt_id"] for prompt in prompts}


def test_direction_hashes_match_the_frozen_manifest():
    config = load_audit_yaml()
    panel = build_direction_panel(896, load_audit_config(ROOT / config["paths"]["m22_1_1_config"]))
    observed = {row["direction_id"]: row["final_sha256"] for row in panel["directions"]}
    assert observed == config["direction_hashes"]
    for row in panel["directions"]:
        assert vector_sha256(row["vector"]) == row["final_sha256"]
        assert abs(float(np.linalg.norm(row["vector"])) - 1.0) < 1e-12


def test_closed_form_matches_rms_and_linear_modules():
    torch.manual_seed(0)
    d_model = 16
    eps = 1e-6
    residual = torch.randn(1, 1, d_model)
    weight = torch.randn(d_model)
    unembed = torch.randn(d_model, 7)
    bias = torch.zeros(7)
    positive_id, negative_id = 3, 1

    class _Norm(torch.nn.Module):
        def forward(self, x):
            return x / rms_scale(x, eps) * weight

    class _Unembed(torch.nn.Module):
        def forward(self, x):
            return x @ unembed + bias

    readout = weight * (unembed[:, positive_id] - unembed[:, negative_id])
    closed = closed_form_margin(residual, readout, bias[positive_id] - bias[negative_id], eps)
    module = module_margin(residual, _Norm(), _Unembed(), positive_id, negative_id)
    rel = abs(float(closed.item()) - float(module.item())) / max(abs(float(module.item())), 1e-8)
    assert rel <= 1e-5


def test_jacobian_matches_autograd_and_alpha_zero_is_exactly_zero():
    torch.manual_seed(1)
    d_model = 32
    eps = 1e-6
    residual = torch.randn(d_model, dtype=torch.float64, requires_grad=True)
    direction = torch.randn(d_model, dtype=torch.float64)
    direction = direction / torch.linalg.norm(direction)
    readout = torch.randn(d_model, dtype=torch.float64)
    value = closed_form_margin(residual, readout, torch.zeros((), dtype=torch.float64), eps)
    gradient = torch.autograd.grad(value, residual)[0]
    automatic = float(torch.dot(gradient, direction))
    closed = float(jacobian_first_order(residual.detach(), direction, readout, eps))
    assert abs(closed - automatic) / max(abs(automatic), 1e-12) <= 1e-6
    residual_det = residual.detach()
    zero = float(
        closed_form_margin(residual_det + 0.0 * direction, readout, torch.zeros((), dtype=torch.float64), eps)
        - closed_form_margin(residual_det, readout, torch.zeros((), dtype=torch.float64), eps)
    )
    assert zero == 0.0
