"""Protocol and status tests on a synthetic instrument. No Qwen weights."""

from src.cognitive_self_model.m22_1.config import load_config
from src.cognitive_self_model.m22_1.metrics import classify_directionality
from src.cognitive_self_model.m22_1.prompts import frozen_prompts
from src.cognitive_self_model.m22_1.protocol import execute_protocol
from src.cognitive_self_model.m22_1.status import classify_status


def _flags(hook, alpha, layer):
    if not hook:
        return {
            "hook_fired": False,
            "last_modified": False,
            "other_unchanged": True,
            "hook_name": None,
        }
    return {
        "hook_fired": True,
        "last_modified": float(alpha) != 0.0,
        "other_unchanged": True,
        "hook_name": f"blocks.{layer}.hook_resid_post",
    }


def _measure_factory(effect_layer, *, null_bias=0.0, validation_layer_bait=None):
    calls = []

    def measure(prompt, *, hook, layer, alpha, direction_id):
        calls.append(
            {
                "prompt_id": prompt.prompt_id,
                "role": prompt.role,
                "hook": hook,
                "layer": layer,
                "alpha": float(alpha),
                "direction_id": direction_id,
            }
        )
        base = 0.2
        flags = _flags(hook, alpha, layer)
        if not hook or float(alpha) == 0.0:
            return {"s": base + null_bias if hook else base, **flags}
        bait = validation_layer_bait is not None and prompt.role != "discovery" and layer == validation_layer_bait
        if bait:
            delta = 5.0 * float(alpha)
        elif direction_id == "primary" and layer == effect_layer:
            delta = 0.4 * float(alpha)
        elif direction_id == "control":
            delta = 0.01 * float(alpha)
        else:
            delta = 1e-5 * float(alpha)
        return {"s": base + delta, **flags}

    return measure, calls


def test_status_labels_follow_the_frozen_rule():
    base = {
        "null_pass": True,
        "hook_integrity_pass": True,
        "split_integrity_pass": True,
        "leakage_pass": True,
        "definition_reproducible": True,
        "scores_finite": True,
        "revision_pinned": True,
        "floor": 1e-4,
        "discovery_mean": 0.4,
        "validation_mean": 0.35,
        "replication_mean": 0.33,
        "validation_control_mean": 0.01,
        "replication_control_mean": 0.01,
        "validation_paired_difference_mean": 0.34,
        "replication_paired_difference_mean": 0.32,
    }
    assert classify_status(base) == "VALIDATED_FOR_M22"
    unpinned = dict(base, revision_pinned=False)
    assert classify_status(unpinned) == "CANDIDATE"
    weak_control = dict(base, validation_control_mean=0.5, validation_paired_difference_mean=0.0)
    assert classify_status(weak_control) == "CANDIDATE"
    no_validation = dict(base, validation_mean=0.0)
    assert classify_status(no_validation) == "REJECTED"
    broken_null = dict(base, null_pass=False)
    assert classify_status(broken_null) == "REJECTED"


def test_directionality_thresholds_are_not_status_gates():
    clear = classify_directionality([0.4, 0.4, 0.4, 0.4], [-0.4, -0.4, -0.4, -0.4], 1e-4, 0.75, 0.5)
    assert clear["label"] == "CLEAR_DIRECTIONAL"
    assert clear["gate"] is False
    partial_inputs_pos = [0.5, 0.5, 0.5, 0.5]
    partial_inputs_neg = [-0.5, -0.5, 0.2, 0.2]
    partial = classify_directionality(partial_inputs_pos, partial_inputs_neg, 1e-4, 0.75, 0.5)
    assert partial["label"] == "PARTIAL_DIRECTIONAL"
    unidentified = classify_directionality([1e-8, 1e-8], [1e-8, 1e-8], 1e-4, 0.75, 0.5)
    assert unidentified["label"] == "UNIDENTIFIABLE"


def test_discovery_freeze_precedes_validation_and_ignores_validation_bait():
    config = load_config()
    measure, calls = _measure_factory(15, validation_layer_bait=0)
    order = []

    def on_freeze(frozen):
        order.append("freeze")
        assert frozen["selected_layer"] == 15
        assert frozen["selected_on"] == "discovery"
        assert not any(call["role"] != "discovery" and call["hook"] and call["alpha"] != 0.0 for call in calls)

    result = execute_protocol(
        frozen_prompts(),
        n_layers=24,
        measure=measure,
        config=config,
        revision_pinned=True,
        definition_reproducible=True,
        leakage_pass=True,
        on_freeze=on_freeze,
    )
    order.append("done")
    assert order == ["freeze", "done"]
    assert result["status"] == "VALIDATED_FOR_M22"
    assert result["selected_layer"] == 15
    validation_layers = {call["layer"] for call in calls if call["role"] == "validation" and call["hook"]}
    assert validation_layers == {15}


def test_null_mismatch_stops_before_nonzero_alpha():
    config = load_config()
    measure, calls = _measure_factory(15, null_bias=1.0)
    result = execute_protocol(
        frozen_prompts(),
        n_layers=24,
        measure=measure,
        config=config,
        revision_pinned=True,
        definition_reproducible=True,
        leakage_pass=True,
        on_freeze=lambda frozen: (_ for _ in ()).throw(AssertionError("freeze must not run")),
    )
    assert result["status"] == "REJECTED"
    assert result["selected_layer"] is None
    assert not any(call["hook"] and call["alpha"] != 0.0 for call in calls)


def test_unpinned_revision_cannot_be_validated():
    config = load_config()
    measure, _calls = _measure_factory(15)
    result = execute_protocol(
        frozen_prompts(),
        n_layers=24,
        measure=measure,
        config=config,
        revision_pinned=False,
        definition_reproducible=True,
        leakage_pass=True,
    )
    assert result["status"] == "CANDIDATE"
    assert result["selected_layer"] == 15


def test_no_effect_is_rejected_without_relabeling():
    config = load_config()

    def measure(prompt, *, hook, layer, alpha, direction_id):
        flags = _flags(hook, alpha, layer)
        return {"s": 0.2, **flags}

    result = execute_protocol(
        frozen_prompts(),
        n_layers=24,
        measure=measure,
        config=config,
        revision_pinned=True,
        definition_reproducible=True,
        leakage_pass=True,
    )
    assert result["status"] == "REJECTED"
