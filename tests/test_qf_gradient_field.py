"""Gradient-field rules. No model download and no MRSM gate."""

import torch

from src.cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role
from src.mrsm.qf_gradient_field import (
    FIELD_NOT,
    FIELD_NOT_PREDICTABLE,
    FIELD_NO_POWER,
    FIELD_PARTIAL,
    FIELD_PARTIAL_OVERALL,
    FIELD_PREDICTABLE,
    FIELD_SUPPORTED,
    FIELD_TIED,
    INCONCLUSIVE,
    LINEAR_DIFFERS,
    LINEAR_MATCH,
    LINEAR_MISS,
    LINEAR_SHRINKS,
    LINEAR_UNDEFINED,
    SMALL_ALPHAS,
    alpha_key,
    alpha0_passes,
    assemble_decision,
    classify_field_splits,
    classify_linear,
    classify_model_linear,
    compare_prediction,
    field_null,
    gradient_check_passes,
    last_position_margin_gradient,
    linear_prediction,
    preregistration_document,
    smallest_alpha_keys,
)


def _grid(relation="AGREE", override=None, extra=None):
    rows = []
    for alpha in SMALL_ALPHAS:
        used = relation
        if override and alpha_key(alpha) in override:
            used = override[alpha_key(alpha)]
        compared = compare_prediction(alpha, 1.0, -alpha if used == "AGREE" else alpha)
        compared["relation"] = used
        rows.append(compared)
    if extra:
        rows.extend(extra)
    return rows


def test_linear_term_is_opposite_the_projection_for_positive_alpha():
    assert linear_prediction(0.01, 2.0) == -0.02
    cell = compare_prediction(0.01, 2.0, -0.02)
    assert cell["relation"] == "AGREE"
    assert cell["linear_sign"] == "NEGATIVE"
    assert cell["projection_sign"] == "POSITIVE"
    assert compare_prediction(0.01, 2.0, 0.02)["relation"] == "DISAGREE"
    assert compare_prediction(-0.01, 2.0, 0.02)["relation"] == "AGREE"
    assert compare_prediction(0.01, 2.0, 0.0)["relation"] == "NEAR_ZERO"


def test_linear_classes_use_only_the_small_grid():
    assert smallest_alpha_keys() == ("-0.01", "0.01")
    assert classify_linear(_grid(), gradient_ok=True, alpha0_ok=True) == LINEAR_MATCH
    assert classify_linear(_grid(override={"0.25": "DISAGREE"}), gradient_ok=True, alpha0_ok=True) == LINEAR_SHRINKS
    assert classify_linear(_grid(override={"0.01": "DISAGREE"}), gradient_ok=True, alpha0_ok=True) == LINEAR_MISS
    assert classify_linear(_grid("NEAR_ZERO"), gradient_ok=True, alpha0_ok=True) == LINEAR_UNDEFINED
    historical = compare_prediction(1.0, 1.0, 1.0)
    historical["relation"] = "DISAGREE"
    assert classify_linear(_grid(extra=[historical]), gradient_ok=True, alpha0_ok=True) == LINEAR_MATCH
    assert classify_linear(_grid(), gradient_ok=False, alpha0_ok=True) == INCONCLUSIVE
    assert classify_linear(_grid(), gradient_ok=True, alpha0_ok=False) == INCONCLUSIVE


def test_held_out_labels_are_not_pooled_or_ranked():
    same = {split: LINEAR_MATCH for split in ("discovery", "validation", "replication")}
    assert classify_model_linear(same) == LINEAR_MATCH
    mixed = dict(same)
    mixed["validation"] = LINEAR_SHRINKS
    assert classify_model_linear(mixed) == LINEAR_DIFFERS
    broken = dict(same)
    broken["replication"] = INCONCLUSIVE
    assert classify_model_linear(broken) == INCONCLUSIVE


def test_instrument_checks_reuse_the_existing_absolute_floor():
    assert gradient_check_passes(1.0, 1.0 + 5e-5)
    assert gradient_check_passes(1.0, 1.009)
    assert not gradient_check_passes(1.0, 1.02)
    assert not gradient_check_passes(0.0, 1e-3)
    assert alpha0_passes(0.0)
    assert alpha0_passes(1e-4)
    assert not alpha0_passes(1.1e-4)


def _basis(index, dimension=8, scale=1.0):
    vector = [0.0] * dimension
    vector[index] = scale
    return vector


def test_field_null_is_the_exhaustive_split_permutation():
    matched = [_basis(index) for index in range(6)]
    supported = field_null(matched, matched)
    assert supported["status"] == FIELD_SUPPORTED
    assert supported["n_permutations"] == 720
    assert supported["n_strictly_better"] == 0
    rotated = matched[1:] + matched[:1]
    assert field_null(matched, rotated)["status"] == FIELD_NOT
    partial_gpt = [_basis(0, scale=-0.5)]
    partial_gpt[0][6] = (1 - 0.25) ** 0.5
    partial_gpt.extend(_basis(index) for index in range(1, 6))
    partial = field_null(matched, partial_gpt)
    assert partial["status"] == FIELD_PARTIAL
    assert partial["identity_cosines"][0] < 0
    tied_qwen = [_basis(0), _basis(0)] + [_basis(index) for index in range(2, 6)]
    assert field_null(tied_qwen, tied_qwen)["status"] == FIELD_TIED
    constant = [_basis(0) for _ in range(6)]
    assert field_null(constant, constant)["status"] == FIELD_NO_POWER
    assert field_null(matched, [[0.0] * 8 for _ in range(6)])["status"] == INCONCLUSIVE


def test_field_decision_does_not_treat_a_negative_as_nonexistence():
    supported = {split: {"status": FIELD_SUPPORTED} for split in ("discovery", "validation", "replication")}
    assert classify_field_splits(supported) == FIELD_PREDICTABLE
    absent = {split: {"status": FIELD_NOT} for split in ("discovery", "validation", "replication")}
    assert classify_field_splits(absent) == FIELD_NOT_PREDICTABLE
    mixed = dict(supported)
    mixed["replication"] = {"status": FIELD_NOT}
    assert classify_field_splits(mixed) == FIELD_PARTIAL_OVERALL
    powerless = dict(supported)
    powerless["discovery"] = {"status": FIELD_NO_POWER}
    assert classify_field_splits(powerless) == INCONCLUSIVE


def _measurement(label_relation="AGREE", field=None, gradient_ok=True, alpha0_ok=True):
    cells = _grid(label_relation)
    rows = field if field is not None else [_basis(index) for index in range(6)]
    payload = {
        "cells": cells,
        "field": [{"prompt_id": f"p{index}", "vector": row} for index, row in enumerate(rows)],
        "gradient_ok": gradient_ok,
        "alpha0_ok": alpha0_ok,
    }
    return {split: payload for split in ("discovery", "validation", "replication")}


def test_assemble_keeps_the_sign_correction_and_refuses_m3():
    decision = assemble_decision({"qwen": _measurement(), "gpt2": _measurement()})
    assert decision["m3_executed"] is False
    assert decision["mechanism_identified"] is False
    assert decision["models_ranked"] is False
    assert decision["linear"]["qwen"]["label"] == LINEAR_MATCH
    assert decision["field_decision"] == FIELD_PREDICTABLE
    text = decision["primary_interpretation"]
    assert "not ranked" in text
    assert "cannot exist" in text
    assert "M3 was not run" in text
    assert "better" not in text
    document = preregistration_document()
    assert document["m3"] is False
    assert document["map"]["learned_rotation"] is False
    assert document["map"]["layers_15_and_23_on_gpt2"] == "NOT_COMPARABLE"
    assert document["sign_correction"].startswith("For alpha > 0")
    assert document["gradient_check_epsilon"] == 1e-2
    assert "2606.27510" in document["related_not_this_test"][0]["arxiv"]


def test_frozen_prompt_splits_stay_at_six_without_a_new_holdout():
    grouped = prompts_by_role(frozen_prompts())
    assert set(grouped) == {"discovery", "validation", "replication"}
    assert all(len(grouped[split]) == 6 for split in grouped)
    discovery = {record.prompt_id for record in grouped["discovery"]}
    assert discovery.isdisjoint({record.prompt_id for record in grouped["validation"]})
    assert discovery.isdisjoint({record.prompt_id for record in grouped["replication"]})


def test_direct_gradient_matches_an_analytic_margin():
    def run_with_hooks(tokens, fwd_hooks):
        residual = torch.zeros(1, 1, 3)
        for _name, hook in fwd_hooks:
            residual = hook(residual, None)
        hidden = residual[0, -1]
        logits = torch.zeros(1, 1, 2)
        logits[0, 0, 0] = 3.0 * hidden[0] - hidden[1]
        logits[0, 0, 1] = -2.0 * hidden[2]
        return logits

    gradient = last_position_margin_gradient(run_with_hooks, None, 3, 0, 1)
    assert gradient.tolist() == [3.0, -1.0, 2.0]
