"""M22.1.1 panel and classification tests. No model and no outcome-dependent generation."""

import ast
import inspect
from pathlib import Path

import pytest

from src.cognitive_self_model.m22_1.direction import orthogonal_direction, primary_direction
from src.cognitive_self_model.m22_1.leakage import audit_package_imports
from src.cognitive_self_model.m22_1.prompts import frozen_prompts, prompt_sha256
from src.cognitive_self_model.m22_1_1.analyze import classify_panel, classify_split
from src.cognitive_self_model.m22_1_1.panel import (
    _new_direction,
    build_direction_panel,
    load_audit_config,
    panel_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "m22_1_1_specificity_audit.json"
PACKAGE = ROOT / "src" / "cognitive_self_model" / "m22_1_1"
LABELS = {
    "BROAD_RESIDUAL_SENSITIVITY",
    "STRUCTURED_INTERVENTION_SPACE",
    "LOW_DIMENSIONAL_SENSITIVITY",
    "REGIME_DEPENDENT_SENSITIVITY",
    "INSUFFICIENT_EVIDENCE",
}


def _config():
    return load_audit_config(CONFIG_PATH)


def _record(direction_id, prompt_id, regime, alpha, delta):
    return {
        "direction_id": direction_id,
        "role": "pre_registered_new",
        "prompt_id": prompt_id,
        "regime": regime,
        "alpha": float(alpha),
        "actual_delta": float(delta),
        "direction_norm": 1.0,
    }


def _grid(direction_means, prompt_regimes):
    """Build alpha 0/-1/+1 rows. +1 uses the supplied per-direction regime means."""
    rows = []
    for index, direction_id in enumerate("D1 D2 D3 D4 D5 D6 D7 D8".split()):
        for prompt_id, regime in prompt_regimes:
            plus = direction_means[index][regime]
            for alpha, delta in ((0.0, 0.0), (-1.0, -plus), (1.0, plus)):
                rows.append(_record(direction_id, prompt_id, regime, alpha, delta))
    return rows


PROMPTS = (
    ("completion-01", "completion"),
    ("instruction-01", "instruction"),
    ("syntax-01", "syntax"),
)


def test_frozen_contract_matches_m22_1_site_and_outcome():
    config = _config()
    assert config["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert config["model_revision"] == "060db6499f32faf8b98477b0a26969ef7d8b9987"
    assert config["hook_name"] == "blocks.23.hook_resid_post"
    assert config["layer"] == 23
    assert config["token_position"] == "last"
    assert config["expected_positive_id"] == 9834
    assert config["expected_negative_id"] == 902
    assert config["magnitude_grid"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert config["m22_1_status_preserved"] == "CANDIDATE"
    assert config["m22_2_authorized"] is False
    assert config["new_direction_seeds"] == [22111, 22112, 22113, 22114, 22115, 22116]
    assert config["frozen_before_intervention_measurement"] is True


def test_prompt_hashes_are_the_m22_1_frozen_set():
    france = next(prompt for prompt in frozen_prompts() if prompt.prompt_id == "completion-01")
    assert france.prompt_sha256 == "bbaff4d2ecd5892d4a442b0f53131641bf6e6f284761dd20fc0664bc97145762"
    assert france.prompt_sha256 == prompt_sha256(france.text)
    assert len(frozen_prompts()) == 18


def test_panel_reproduces_m22_1_directions_and_is_orthogonal():
    config = _config()
    panel = build_direction_panel(896, config)
    again = build_direction_panel(896, config)
    by_id = {row["direction_id"]: row for row in panel["directions"]}
    assert by_id["D1"]["final_sha256"] == config["m22_1_primary_sha256"]
    assert by_id["D2"]["final_sha256"] == config["m22_1_control_sha256"]
    assert by_id["D1"]["role"] == "m22_1_primary"
    assert by_id["D2"]["role"] == "m22_1_control"
    assert [row["seed"] for row in panel["directions"][2:]] == config["new_direction_seeds"]
    assert all(row["fallback_attempt"] == 0 for row in panel["directions"])
    for row in panel["directions"]:
        assert abs(row["norm"] - 1.0) < 1e-12
        assert row["final_sha256"] == next(
            other["final_sha256"] for other in again["directions"] if other["direction_id"] == row["direction_id"]
        )
    for left in panel["directions"]:
        for right in panel["directions"]:
            dot = panel["pairwise_dot"][left["direction_id"]][right["direction_id"]]
            if left["direction_id"] == right["direction_id"]:
                assert abs(dot - 1.0) < 1e-8
            else:
                assert abs(dot) < 1e-8
    manifest = panel_manifest(panel)
    assert "vector" not in manifest["directions"][0]
    assert manifest["directions"][0]["final_sha256"] == config["m22_1_primary_sha256"]


def test_direction_generation_does_not_accept_outcomes():
    signature = inspect.signature(build_direction_panel)
    assert list(signature.parameters) == ["dimension", "config"]
    source = (PACKAGE / "panel.py").read_text(encoding="utf-8")
    for forbidden in (
        "actual_delta",
        "logit",
        "validation",
        "replication",
        "q_invalid",
        "self_model_invalid",
        "latent",
    ):
        assert forbidden not in source


def test_degenerate_direction_stops_instead_of_searching_effects():
    primary = primary_direction(2, 1)
    control = orthogonal_direction(2, 2, primary)
    with pytest.raises(RuntimeError, match="degenerate"):
        _new_direction(2, 22111, [primary, control], 1e-8)


def test_package_does_not_import_m20_m21_or_benchmark_code():
    audit = audit_package_imports(PACKAGE)
    assert audit["pass"], audit["offenders"]
    offenders = []
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = " ".join(alias.name for alias in node.names)
            else:
                continue
            if any(token in module for token in ("m20", "m21", "benchmark", "legacy_import")):
                offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_manifest_is_written_before_the_model_is_loaded():
    source = (PACKAGE / "run_audit.py").read_text(encoding="utf-8")
    assert source.index("direction_panel_manifest.json") < source.index("_load_exact(config)")


def test_classification_rules_are_predeclared_and_descriptive():
    rules = _config()["classification"]
    assert rules["substantial_abs_floor"] == 0.001
    assert rules["broad_min_directions"] == 6
    assert rules["low_dimensional_min_leading_energy"] == 0.8
    assert rules["multiple_matching_labels"] == "INSUFFICIENT_EVIDENCE"
    assert rules["validation_replication_disagreement"] == "INSUFFICIENT_EVIDENCE"
    assert rules["discovery_reported_but_not_used_for_label"] is True
    same = {regime: 0.05 for regime in ("completion", "instruction", "syntax")}
    broad = _grid([same] * 8, PROMPTS)
    assert classify_split(broad, _config())["label"] == "BROAD_RESIDUAL_SENSITIVITY"
    structured_means = []
    for index in range(8):
        scale = 0.2 if index < 2 else 0.02
        structured_means.append({regime: scale for regime in ("completion", "instruction", "syntax")})
    structured = _grid(structured_means, PROMPTS)
    assert classify_split(structured, _config())["label"] == "STRUCTURED_INTERVENTION_SPACE"
    low_means = []
    for index in range(8):
        scale = 0.2 if index == 0 else 0.0
        low_means.append({regime: scale for regime in ("completion", "instruction", "syntax")})
    low = _grid(low_means, PROMPTS)
    assert classify_split(low, _config())["label"] == "LOW_DIMENSIONAL_SENSITIVITY"
    regime_means = []
    for index in range(8):
        if index < 4:
            regime_means.append({"completion": 0.3, "instruction": -0.3, "syntax": 0.05})
        else:
            regime_means.append({"completion": 0.0, "instruction": 0.0, "syntax": 0.0})
    regime = _grid(regime_means, PROMPTS)
    assert classify_split(regime, _config())["label"] == "REGIME_DEPENDENT_SENSITIVITY"
    mixed = _grid(
        [{"completion": 0.2, "instruction": -0.2, "syntax": 0.05}] * 8,
        PROMPTS,
    )
    assert classify_split(mixed, _config())["label"] == "INSUFFICIENT_EVIDENCE"
    noise = _grid([{regime: 0.0 for regime in ("completion", "instruction", "syntax")}] * 8, PROMPTS)
    assert classify_split(noise, _config())["label"] == "INSUFFICIENT_EVIDENCE"
    disagreed = classify_panel(
        {"discovery": noise, "validation": broad, "replication": low},
        _config(),
    )
    assert disagreed["label"] == "INSUFFICIENT_EVIDENCE"
    agreed = classify_panel(
        {"discovery": noise, "validation": broad, "replication": broad},
        _config(),
    )
    assert agreed["label"] == "BROAD_RESIDUAL_SENSITIVITY"
    assert agreed["label"] in LABELS
