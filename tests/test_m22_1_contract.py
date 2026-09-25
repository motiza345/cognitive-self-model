"""Frozen-contract tests for M22.1. These lock the preflight before any model run."""

import hashlib

import numpy as np
import pytest

from src.cognitive_self_model.m22_1.config import config_sha256, load_config
from src.cognitive_self_model.m22_1.direction import orthogonal_direction, primary_direction, vector_sha256
from src.cognitive_self_model.m22_1.leakage import audit_package_imports
from src.cognitive_self_model.m22_1.prompts import (
    assert_split_integrity,
    frozen_prompts,
    prompt_manifest_sha256,
    prompt_sha256,
)
from src.cognitive_self_model.m22_1.protocol import candidate_layers, select_layer
from src.cognitive_self_model.m22_1.runtime_info import sanitize_origin


def test_outcome_and_grid_are_frozen():
    config = load_config()
    assert config["outcome"]["type"] == "logit_margin"
    assert config["outcome"]["positive_text"] == " yes"
    assert config["outcome"]["negative_text"] == " no"
    assert config["magnitude_grid"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert config["primary_alpha"] == 1.0
    assert config["direction_seed"] == 22101
    assert config["control_direction_seed"] == 22103
    assert config["n_candidate_layers"] == 4
    assert config["null_bug_threshold"] == 0.0001
    assert config["effect_floor_abs"] == 0.0001
    assert config["effect_floor_mult"] == 10
    assert config["directionality"]["clear_min_opposite_fraction"] == 0.75
    assert config["directionality"]["partial_min_opposite_fraction"] == 0.5
    assert config["dtype"] == "float32"
    assert config["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert config["token_position"] == "last"
    assert config["hook_template"] == "blocks.{layer}.hook_resid_post"
    assert config["gpu_required"] is False
    assert config_sha256(config) == hashlib.sha256(
        __import__("json").dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_layer_rule_does_not_preselect_layer_13_for_qwen_depth():
    layers = candidate_layers(24, 4)
    assert layers == [0, 8, 15, 23]
    assert 13 not in layers


def test_prompt_manifest_is_disjoint_and_hashed():
    prompts = frozen_prompts()
    assert_split_integrity(prompts)
    by_role = {}
    for prompt in prompts:
        by_role.setdefault(prompt.role, []).append(prompt)
    assert set(by_role) == {"discovery", "validation", "replication"}
    assert all(len(group) == 6 for group in by_role.values())
    france = next(prompt for prompt in prompts if prompt.prompt_id == "completion-01")
    assert france.role == "discovery"
    assert france.prompt_sha256 == "bbaff4d2ecd5892d4a442b0f53131641bf6e6f284761dd20fc0664bc97145762"
    assert france.prompt_sha256 == prompt_sha256(france.text)
    assert len(prompt_manifest_sha256(prompts)) == 64
    overlaps = []
    roles = list(by_role)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlaps.append(
                {prompt.prompt_sha256 for prompt in by_role[left]}
                & {prompt.prompt_sha256 for prompt in by_role[right]}
            )
    assert overlaps == [set(), set(), set()]


def test_directions_are_deterministic_unit_and_orthogonal():
    primary = primary_direction(32, 22101)
    again = primary_direction(32, 22101)
    control = orthogonal_direction(32, 22103, primary)
    assert vector_sha256(primary) == vector_sha256(again)
    assert np.allclose(np.linalg.norm(primary), 1.0)
    assert np.allclose(np.linalg.norm(control), 1.0)
    assert abs(float(np.dot(primary, control))) < 1e-6


def test_select_layer_tie_breaks_to_the_lowest_index():
    assert select_layer({8: 0.2, 0: 0.2, 15: -1.0}) == 0
    assert select_layer({0: 0.1, 15: 0.4, 23: 0.4}) == 15


def test_origin_url_does_not_keep_credentials():
    raw = "https://x-access-token:secret@github.com/motiza345/cognitive-self-model"
    assert sanitize_origin(raw) == "https://github.com/motiza345/cognitive-self-model"


def test_package_does_not_import_epistemic_benchmark_code():
    audit = audit_package_imports()
    assert audit["pass"] is True
    assert audit["offenders"] == []


def test_loader_revision_helper_does_not_invent_a_sha(monkeypatch):
    import sys
    import types

    import src.cognitive_self_model.m22_1.loader as loader

    fake = types.ModuleType("huggingface_hub")

    def model_info(model_id):
        raise RuntimeError("offline")

    fake.model_info = model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    result = loader.resolve_hf_revision("Qwen/Qwen2.5-0.5B")
    assert result["model_revision"] is None
    assert result["revision_pinned"] is False
    assert "offline" in result["revision_error"]
