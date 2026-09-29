"""Decision rules for the Qwen bridge diagnostic. No model load."""

from src.mrsm.qwen_bridge import classify_identity, decide, is_context_cancellation, select_bridge


def test_identity_categories_are_not_ranked():
    same = {
        "model_family": "yes",
        "model_revision": "yes",
        "tokenizer": "yes",
        "dtype": "yes",
        "activation_extraction": "yes",
        "indexing": "yes",
        "prompt_distribution": "yes",
        "intervention_location": "yes",
        "intervention_type": "yes",
        "intervention_magnitude": "yes",
    }
    assert classify_identity(same) == "IDENTICAL_SETUP"
    different_site = dict(same)
    different_site["intervention_location"] = "no"
    assert classify_identity(different_site) == "SAME_MODEL_DIFFERENT_SETUP"
    other_revision = dict(same)
    other_revision["model_revision"] = "no"
    assert classify_identity(other_revision) == "SAME_FAMILY_DIFFERENT_REVISION"
    unknown = dict(same)
    unknown["tokenizer"] = "unknown"
    assert classify_identity(unknown) == "NOT_COMPARABLE"
    unknown_but_different_site = dict(unknown)
    unknown_but_different_site["intervention_location"] = "no"
    assert classify_identity(unknown_but_different_site) == "SAME_MODEL_DIFFERENT_SETUP"


def test_bridge_selection_prefers_the_direct_experiment_with_fewer_dependencies():
    records = [
        {
            "experiment_id": "later",
            "direct_causal_intervention": True,
            "negative_control_present": True,
            "effect_size": 0.1,
            "reproducible_code": True,
            "external_dependency_count": 2,
        },
        {
            "experiment_id": "M22.1",
            "direct_causal_intervention": True,
            "negative_control_present": True,
            "effect_size": 0.02,
            "reproducible_code": True,
            "external_dependency_count": 0,
        },
        {
            "experiment_id": "reanalysis",
            "direct_causal_intervention": False,
            "negative_control_present": True,
            "effect_size": 1.0,
            "reproducible_code": True,
            "external_dependency_count": 0,
        },
    ]
    chosen = select_bridge(records)
    assert chosen["experiment_id"] == "M22.1"


def test_cancellation_requires_opposite_regimes_and_a_smaller_pool():
    assert is_context_cancellation(
        {"completion": 0.09, "instruction": -0.08, "syntax": -0.04},
        pooled=-0.007,
    )
    assert not is_context_cancellation(
        {"completion": -0.03, "instruction": -0.02, "syntax": -0.03},
        pooled=-0.027,
    )


def test_decision_maps_reproduction_and_cancellation():
    cancelled = decide(
        historical_reproduced=True,
        failure_locus=None,
        mrsm_signal_present=True,
        context_cancellation=True,
    )
    assert cancelled["bridge_result"] == "BRIDGE_SIGNAL_PRESENT_BUT_CONTEXT_DEPENDENT"
    assert cancelled["primary_diagnosis"] == "INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED"
    lost = decide(
        historical_reproduced=True,
        failure_locus=None,
        mrsm_signal_present=False,
        context_cancellation=False,
    )
    assert lost["bridge_result"] == "BRIDGE_SIGNAL_LOST_IN_MRSM_SETUP"
    assert lost["primary_diagnosis"] == "SETUP_MISMATCH_BOTTLENECK"
    missing = decide(
        historical_reproduced=False,
        failure_locus="IMPLEMENTATION",
        mrsm_signal_present=False,
        context_cancellation=False,
    )
    assert missing["bridge_result"] == "HISTORICAL_RESULT_NOT_REPRODUCIBLE"
    assert missing["failure_locus"] == "IMPLEMENTATION"
