"""IG-0: initial go/no-go for a mechanism self-model.

The project history says a complete self-model does not exist yet, and that a
good predictor is not one. This module tests the smallest claim that has to be
true before that roadmap continues:

an interventional search over a small mechanism library can identify the hidden
mechanism, re-identify it when its components move, and separate novelty from
invalidity. A linear predictor and an observational fit are run beside it so a
good prediction cannot be mistaken for that result.

A passing run is evidence about this controlled world only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "initial_gate.json"

Hypothesis = Tuple[str, int, int]
Kind = Tuple[str, int, int]

ALLOWED_CLAIMS = (
    "On this 4-component world, interventional search over the declared library identifies the generating mechanism and its roles.",
    "When addition and copy are observationally identical, only interventions separate them.",
    "A frozen mechanism is marked invalid inside its training range when the mechanism changes, and novel but still valid when only the input range changes.",
    "A position-bound linear predictor can fit in-distribution addition and still fail after rebinding or a mechanism change.",
    "Shuffling intervention outcomes destroys identification.",
)

DISALLOWED_CLAIMS = (
    "A general self-model of a language model.",
    "Mechanism identity outside this library.",
    "Calibration of an epistemic estimator on the M20.6.3.2 streams.",
    "Safe self-modification or self-improvement.",
    "That this gate validates milestone M21.2.4.2.",
)


def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _apply(family: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if family == "add":
        return a + b
    if family == "sub":
        return a - b
    if family == "mul":
        return a * b
    raise ValueError(f"Unsupported binary family: {family}")


def structural(family: str, state: np.ndarray, left: int, right: int) -> np.ndarray:
    if family == "copy":
        return state[..., left]
    return _apply(family, state[..., left], state[..., right])


def covers_same_relation(first: Hypothesis, second: Hypothesis) -> bool:
    family_a, left_a, right_a = first
    family_b, left_b, right_b = second
    if family_a != family_b:
        return False
    if family_a == "copy":
        return left_a == left_b
    if family_a in ("add", "mul"):
        return {left_a, right_a} == {left_b, right_b}
    return (left_a, right_a) == (left_b, right_b)


def matches_truth(hypothesis: Hypothesis, truth: Kind) -> bool:
    return covers_same_relation(hypothesis, truth)


def hypotheses() -> List[Hypothesis]:
    pairs = [(left, right) for left in range(4) for right in range(4) if left != right]
    found: List[Hypothesis] = []
    for left, right in pairs:
        found.append(("add", left, right))
        found.append(("sub", left, right))
        found.append(("mul", left, right))
    for index in range(4):
        found.append(("copy", index, -1))
    return found


HYPOTHESES = hypotheses()


def predict_hypothesis(hypothesis: Hypothesis, state: np.ndarray) -> np.ndarray:
    family, left, right = hypothesis
    return structural(family, state, left, right)


def score_hypotheses(
    state: np.ndarray,
    outcome: np.ndarray,
    allowed_families: Optional[Sequence[str]] = None,
) -> List[Tuple[float, Hypothesis]]:
    scored: List[Tuple[float, Hypothesis]] = []
    for hypothesis in HYPOTHESES:
        if allowed_families is not None and hypothesis[0] not in allowed_families:
            continue
        prediction = predict_hypothesis(hypothesis, state)
        scored.append((float(np.mean(np.abs(prediction - outcome))), hypothesis))
    scored.sort(key=lambda item: item[0])
    return scored


def select_hypothesis(
    state: np.ndarray,
    outcome: np.ndarray,
    allowed_families: Optional[Sequence[str]] = None,
) -> Tuple[Hypothesis, float, float]:
    """Return the best hypothesis, its error, and the gap to the next distinct one."""
    scored = score_hypotheses(state, outcome, allowed_families)
    best_error, best = scored[0]
    margin = float("inf")
    for error, hypothesis in scored[1:]:
        if not covers_same_relation(hypothesis, best):
            margin = float(error - best_error)
            break
    return best, best_error, margin


def plant_state(
    rng: np.random.Generator,
    truth: Kind,
    low: float,
    high: float,
) -> np.ndarray:
    """Build a state in which two different mechanisms explain the same observation."""
    state = rng.uniform(low, high, size=4)
    family, left, right = truth
    if family == "copy":
        others = [index for index in range(4) if index != left]
        state[others[0]] = rng.uniform(low, high)
        state[others[1]] = state[left] - state[others[0]]
        return state

    clean = structural(family, state, left, right)
    distractors = [index for index in range(4) if index not in (left, right)]
    state[distractors[0]] = clean
    return state


def sample_interventions(
    rng: np.random.Generator,
    truth: Kind,
    count: int,
    config: Mapping[str, float],
    low: Optional[float] = None,
    high: Optional[float] = None,
    operand_span: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    value_low = float(config["value_low"] if low is None else low)
    value_high = float(config["value_high"] if high is None else high)
    noise = float(config["noise_std"])
    states = np.zeros((count, 4), dtype=np.float64)
    outcomes = np.zeros(count, dtype=np.float64)
    indices = np.zeros(count, dtype=np.int64)
    family, left, right = truth

    for row in range(count):
        state = plant_state(rng, truth, value_low, value_high)
        if operand_span is not None:
            span_low, span_high = operand_span
            if family == "copy":
                sign = float(rng.choice(np.array([-1.0, 1.0])))
                state[left] = sign * float(rng.uniform(span_low, span_high))
            else:
                for index in (left, right):
                    sign = float(rng.choice(np.array([-1.0, 1.0])))
                    state[index] = sign * float(rng.uniform(span_low, span_high))
        index = int(rng.integers(0, 4))
        state[index] = float(rng.uniform(value_low, value_high))
        if operand_span is not None and family != "copy" and index in (left, right):
            sign = float(rng.choice(np.array([-1.0, 1.0])))
            state[index] = sign * float(rng.uniform(*operand_span))
        states[row] = state
        outcomes[row] = float(structural(family, state, left, right) + rng.normal(0.0, noise))
        indices[row] = index
    return states, outcomes, indices


def sample_observational(
    rng: np.random.Generator,
    truth: Kind,
    count: int,
    config: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    noise = float(config["noise_std"])
    low = float(config["value_low"])
    high = float(config["value_high"])
    family, left, right = truth
    states = np.zeros((count, 4), dtype=np.float64)
    outcomes = np.zeros(count, dtype=np.float64)
    for row in range(count):
        state = plant_state(rng, truth, low, high)
        clean = float(structural(family, state, left, right))
        states[row] = state
        outcomes[row] = clean + float(rng.normal(0.0, noise))
    return states, outcomes


def fit_ridge(state: np.ndarray, outcome: np.ndarray, penalty: float = 1e-2) -> np.ndarray:
    design = np.column_stack([np.ones(len(outcome)), state])
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[0, 0] = 0.0
    gram = design.T @ design + regularizer
    return np.linalg.solve(gram, design.T @ outcome)


def predict_ridge(coefficients: np.ndarray, state: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(state)), state])
    return design @ coefficients


def fit_component_model(
    state: np.ndarray,
    outcome: np.ndarray,
    intervention_index: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    coefficients: Dict[int, Tuple[float, float]] = {}
    for index in range(4):
        mask = intervention_index == index
        if int(np.sum(mask)) < 2:
            coefficients[index] = (float(np.mean(outcome)), 0.0)
            continue
        design = np.column_stack([np.ones(int(np.sum(mask))), state[mask, index]])
        beta, *_ = np.linalg.lstsq(design, outcome[mask], rcond=None)
        coefficients[index] = (float(beta[0]), float(beta[1]))
    return coefficients


def predict_component_model(
    coefficients: Mapping[int, Tuple[float, float]],
    state: np.ndarray,
    intervention_index: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros(len(state), dtype=np.float64)
    for index in range(4):
        mask = intervention_index == index
        intercept, slope = coefficients[index]
        prediction[mask] = intercept + slope * state[mask, index]
    return prediction


def mean_absolute_error(prediction: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - outcome)))


def observational_correlations(state: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    scores = np.zeros(4, dtype=np.float64)
    for index in range(4):
        if float(np.std(state[:, index])) < 1e-8 or float(np.std(outcome)) < 1e-8:
            continue
        scores[index] = abs(float(np.corrcoef(state[:, index], outcome)[0, 1]))
    return scores


def interventional_effects(
    rng: np.random.Generator,
    truth: Kind,
    config: Mapping[str, float],
) -> Tuple[np.ndarray, float]:
    """Mean absolute effect of setting each component, plus the placebo false-alarm rate."""
    effects = np.zeros(4, dtype=np.float64)
    samples = int(config["effect_samples"])
    low = float(config["value_low"])
    high = float(config["value_high"])
    noise = float(config["noise_std"])
    alarm = float(config["placebo_alarm_level"])
    family, left, right = truth
    causal = {left} if family == "copy" else {left, right}
    placebo_alarms = []

    for index in range(4):
        deltas = []
        for _ in range(samples):
            state = plant_state(rng, truth, low, high)
            before = float(structural(family, state, left, right))
            state[index] = float(rng.uniform(low, high))
            after = float(structural(family, state, left, right))
            deltas.append(abs(after - before))
            if index not in causal:
                noisy_before = before + float(rng.normal(0.0, noise))
                noisy_after = after + float(rng.normal(0.0, noise))
                placebo_alarms.append(abs(noisy_after - noisy_before) > alarm)
        effects[index] = float(np.mean(deltas))
    false_alarm = float(np.mean(placebo_alarms))
    return effects, false_alarm


def rotate_operands(state: np.ndarray) -> np.ndarray:
    """Phase-style coordinate change: (x0, x1) -> (x1, -x0). Addition becomes subtraction."""
    rotated = np.array(state, copy=True)
    original_left = np.array(state[:, 0], copy=True)
    rotated[:, 0] = state[:, 1]
    rotated[:, 1] = -original_left
    return rotated


def operand_magnitude(hypothesis: Hypothesis, state: np.ndarray) -> float:
    family, left, right = hypothesis
    if family == "copy":
        return float(np.max(np.abs(state[:, left])))
    return float(np.max(np.abs(state[:, [left, right]])))


def run_seed(seed: int, config: Mapping) -> Dict:
    streams = np.random.SeedSequence(seed).spawn(16)
    train_rng = np.random.default_rng(streams[0])
    truth: Kind = ("add", 0, 1)
    train_state, train_outcome, train_index = sample_interventions(
        train_rng,
        truth,
        int(config["train_interventions"]),
        config,
    )
    hypothesis, train_mae, train_margin = select_hypothesis(train_state, train_outcome)
    scope_max = operand_magnitude(hypothesis, train_state)
    ridge = fit_ridge(train_state, train_outcome)
    component = fit_component_model(train_state, train_outcome, train_index)

    holdout_rng = np.random.default_rng(streams[1])
    hold_state, hold_outcome, hold_index = sample_interventions(
        holdout_rng,
        truth,
        80,
        config,
    )
    predictor_mae = mean_absolute_error(predict_ridge(ridge, hold_state), hold_outcome)
    component_mae = mean_absolute_error(
        predict_component_model(component, hold_state, hold_index),
        hold_outcome,
    )
    mechanism_mae = mean_absolute_error(predict_hypothesis(hypothesis, hold_state), hold_outcome)

    trap_rng = np.random.default_rng(streams[2])
    observed_state, observed_outcome = sample_observational(
        trap_rng,
        truth,
        int(config["observational_rows"]),
        config,
    )
    correlations = observational_correlations(observed_state, observed_outcome)
    effects, placebo_false_alarm = interventional_effects(trap_rng, truth, config)
    observational_top = int(np.argmax(correlations))
    interventional_top = int(np.argmax(effects))
    causal_effect = float(np.max(effects[[0, 1]]))
    placebo_effect = float(np.max(effects[[2, 3]]))

    swap_rng = np.random.default_rng(streams[3])
    swap_state, swap_outcome, _ = sample_interventions(swap_rng, ("sub", 0, 1), 80, config)
    predictor_swap_mae = mean_absolute_error(predict_ridge(ridge, swap_state), swap_outcome)
    frozen_swap_mae = mean_absolute_error(predict_hypothesis(hypothesis, swap_state), swap_outcome)

    rebind_rng = np.random.default_rng(streams[4])
    rebound_truth: Kind = ("add", 2, 3)
    rebind_state, rebind_outcome, _ = sample_interventions(
        rebind_rng,
        rebound_truth,
        int(config["few_shot"]),
        config,
    )
    zero_shot_rebind_mae = mean_absolute_error(
        predict_hypothesis(hypothesis, rebind_state),
        rebind_outcome,
    )
    predictor_rebind_mae = mean_absolute_error(predict_ridge(ridge, rebind_state), rebind_outcome)
    transfer_hypothesis, transfer_mae, _ = select_hypothesis(
        rebind_state,
        rebind_outcome,
        allowed_families=("add",),
    )
    cold_hypothesis, cold_mae, _ = select_hypothesis(rebind_state, rebind_outcome)

    rotation_rng = np.random.default_rng(streams[5])
    rotation_state, rotation_outcome, _ = sample_interventions(
        rotation_rng,
        truth,
        int(config["few_shot"]),
        config,
    )
    rotated = rotate_operands(rotation_state)
    rotation_zero_shot_mae = mean_absolute_error(
        predict_hypothesis(hypothesis, rotated),
        rotation_outcome,
    )
    rotation_hypothesis, rotation_mae, _ = select_hypothesis(rotated, rotation_outcome)

    four_rng = np.random.default_rng(streams[6])
    cells = {
        "known_valid": ((False, False), ("add", 0, 1), None),
        "known_invalid": ((False, True), ("sub", 0, 1), None),
        "novel_valid": ((True, False), ("add", 0, 1), (float(config["novel_low"]), float(config["novel_high"]))),
        "novel_invalid": ((True, True), ("sub", 0, 1), (float(config["novel_low"]), float(config["novel_high"]))),
    }
    cell_hits = {name: 0 for name in cells}
    novel_valid_invalid_flags = 0
    known_invalid_hits = 0
    episodes = int(config["episodes_per_cell"])
    for name, (expected, cell_truth, span) in cells.items():
        for _ in range(episodes):
            probe_state, probe_outcome, _ = sample_interventions(
                four_rng,
                cell_truth,
                int(config["probe_rows"]),
                config,
                operand_span=span,
            )
            probe_mae = mean_absolute_error(
                predict_hypothesis(hypothesis, probe_state),
                probe_outcome,
            )
            novel = operand_magnitude(hypothesis, probe_state) > scope_max + float(config["scope_margin"])
            invalid = probe_mae > float(config["invalid_mae"])
            if (novel, invalid) == expected:
                cell_hits[name] += 1
            if name == "novel_valid" and invalid:
                novel_valid_invalid_flags += 1
            if name == "known_invalid" and invalid:
                known_invalid_hits += 1
    four_state_accuracy = float(np.mean([hits / episodes for hits in cell_hits.values()]))
    novel_valid_false_invalid = novel_valid_invalid_flags / episodes
    known_invalid_detection = known_invalid_hits / episodes

    identity_rng = np.random.default_rng(streams[7])
    observational_hits = 0
    interventional_hits = 0
    identity_episodes = int(config["identity_episodes"])
    margin_required = float(config["identification_margin"])
    for episode in range(identity_episodes):
        if episode % 2 == 0:
            identity_truth: Kind = ("add", 0, 1)
        else:
            identity_truth = ("copy", 2, -1)
        observed_x, observed_y = sample_observational(identity_rng, identity_truth, 40, config)
        observed_choice, _, observed_margin = select_hypothesis(observed_x, observed_y)
        if observed_margin >= margin_required and matches_truth(observed_choice, identity_truth):
            observational_hits += 1
        intervened_x, intervened_y, _ = sample_interventions(identity_rng, identity_truth, 24, config)
        intervened_choice, _, intervened_margin = select_hypothesis(intervened_x, intervened_y)
        if intervened_margin >= margin_required and matches_truth(intervened_choice, identity_truth):
            interventional_hits += 1
    observational_accuracy = observational_hits / identity_episodes
    interventional_accuracy = interventional_hits / identity_episodes

    shuffle_rng = np.random.default_rng(streams[8])
    shuffle_trials = 20
    shuffle_hits = 0
    for _ in range(shuffle_trials):
        shuffled = shuffle_rng.permutation(train_outcome)
        shuffled_choice, _, shuffled_margin = select_hypothesis(train_state, shuffled)
        if shuffled_margin >= margin_required and matches_truth(shuffled_choice, truth):
            shuffle_hits += 1
    shuffle_accuracy = shuffle_hits / shuffle_trials

    mul_rng = np.random.default_rng(streams[9])
    mul_truth: Kind = ("mul", 0, 1)
    mul_state, mul_outcome, _ = sample_interventions(
        mul_rng,
        mul_truth,
        int(config["train_interventions"]),
        config,
    )
    mul_hypothesis, mul_mae, mul_margin = select_hypothesis(mul_state, mul_outcome)
    mul_hold_state, mul_hold_outcome, _ = sample_interventions(mul_rng, mul_truth, 80, config)
    mul_predictor = fit_ridge(mul_state, mul_outcome)
    mul_predictor_mae = mean_absolute_error(predict_ridge(mul_predictor, mul_hold_state), mul_hold_outcome)
    mul_mechanism_mae = mean_absolute_error(
        predict_hypothesis(mul_hypothesis, mul_hold_state),
        mul_hold_outcome,
    )

    metrics = {
        "train_mae": train_mae,
        "train_margin": train_margin,
        "holdout_mechanism_mae": mechanism_mae,
        "predictor_indistribution_mae": predictor_mae,
        "component_indistribution_mae": component_mae,
        "predictor_swap_mae": predictor_swap_mae,
        "frozen_swap_mae": frozen_swap_mae,
        "observational_top": observational_top,
        "interventional_top": interventional_top,
        "observational_top_correlation": float(correlations[observational_top]),
        "causal_effect": causal_effect,
        "placebo_effect": placebo_effect,
        "placebo_false_alarm": placebo_false_alarm,
        "zero_shot_rebind_mae": zero_shot_rebind_mae,
        "predictor_rebind_mae": predictor_rebind_mae,
        "transfer_mae": transfer_mae,
        "cold_start_mae": cold_mae,
        "rotation_zero_shot_mae": rotation_zero_shot_mae,
        "rotation_few_shot_mae": rotation_mae,
        "four_state_accuracy": four_state_accuracy,
        "novel_valid_false_invalid": novel_valid_false_invalid,
        "known_invalid_detection": known_invalid_detection,
        "observational_identification": observational_accuracy,
        "interventional_identification": interventional_accuracy,
        "shuffle_accuracy": shuffle_accuracy,
        "mul_mechanism_mae": mul_mechanism_mae,
        "mul_predictor_mae": mul_predictor_mae,
        "mul_margin": mul_margin,
        "scope_max": scope_max,
    }
    gates = {
        "trap_exists": (
            observational_top in (2, 3)
            and interventional_top in (0, 1)
            and placebo_effect <= float(config["placebo_effect_max"])
            and causal_effect >= float(config["causal_effect_min"])
            and placebo_false_alarm <= float(config["placebo_false_alarm_max"])
        ),
        "identify_training_mechanism": (
            matches_truth(hypothesis, truth)
            and train_margin >= margin_required
            and train_mae <= float(config["fitted_mae_max"])
        ),
        "prediction_is_not_identity": (
            predictor_mae <= float(config["predictor_indistribution_mae_max"])
            and predictor_swap_mae >= float(config["predictor_failure_mae_min"])
            and component_mae >= float(config["component_mae_min"])
            and frozen_swap_mae >= float(config["predictor_failure_mae_min"])
        ),
        "rebind_with_new_evidence": (
            zero_shot_rebind_mae >= float(config["predictor_failure_mae_min"])
            and predictor_rebind_mae >= float(config["predictor_failure_mae_min"])
            and transfer_mae <= float(config["transfer_mae_max"])
            and transfer_mae <= cold_mae + float(config["transfer_cold_start_tolerance"])
            and matches_truth(transfer_hypothesis, rebound_truth)
        ),
        "rotation_needs_reidentification": (
            rotation_zero_shot_mae >= float(config["predictor_failure_mae_min"])
            and rotation_mae <= float(config["transfer_mae_max"])
            and matches_truth(rotation_hypothesis, ("sub", 0, 1))
        ),
        "novelty_is_not_invalidity": (
            four_state_accuracy >= float(config["four_state_accuracy_min"])
            and novel_valid_false_invalid <= float(config["novel_valid_false_invalid_max"])
            and known_invalid_detection >= float(config["known_invalid_detection_min"])
        ),
        "observation_cannot_identify": (
            observational_accuracy <= float(config["observational_identification_max"])
            and interventional_accuracy >= float(config["interventional_identification_min"])
        ),
        "shuffle_collapses": shuffle_accuracy <= float(config["shuffle_accuracy_max"]),
        "nonlinear_mechanism": (
            matches_truth(mul_hypothesis, mul_truth)
            and mul_margin >= margin_required
            and mul_mechanism_mae <= float(config["fitted_mae_max"])
            and mul_mechanism_mae + 0.05 < mul_predictor_mae
        ),
    }
    return {
        "seed": seed,
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "selected_hypothesis": list(hypothesis),
        "transfer_hypothesis": list(transfer_hypothesis),
        "rotation_hypothesis": list(rotation_hypothesis),
        "cold_start_hypothesis": list(cold_hypothesis),
        "mul_hypothesis": list(mul_hypothesis),
    }


def _metric_mean(seed_reports: Sequence[Mapping], name: str) -> float:
    return float(np.mean([report["metrics"][name] for report in seed_reports]))


def run_gate(seeds: Optional[Iterable[int]] = None, config: Optional[Mapping] = None) -> Dict:
    config = dict(load_config() if config is None else config)
    if seeds is None:
        seeds = range(int(config["seeds"]))
    reports = [run_seed(int(seed), config) for seed in seeds]
    passes = sum(1 for report in reports if report["passed"])
    gate_names = list(reports[0]["gates"])
    gate_passes = {
        name: sum(1 for report in reports if report["gates"][name])
        for name in gate_names
    }
    metric_names = list(reports[0]["metrics"])
    metric_means = {name: _metric_mean(reports, name) for name in metric_names}
    passed = passes >= int(config["minimum_seed_passes"])
    return {
        "name": config["name"],
        "question": config["question"],
        "passed": passed,
        "seed_passes": passes,
        "seed_count": len(reports),
        "minimum_seed_passes": int(config["minimum_seed_passes"]),
        "gate_passes": gate_passes,
        "metric_means": metric_means,
        "allowed_claims": list(ALLOWED_CLAIMS),
        "disallowed_claims": list(DISALLOWED_CLAIMS),
        "seeds": reports,
        "leakage_audit": {
            "fit_inputs": ["post-intervention state", "outcome"],
            "forbidden_inputs": [
                "family label",
                "role label",
                "novelty label",
                "invalidity label",
                "future outcome",
            ],
            "negative_controls": [
                "placebo intervention on the correlated component",
                "observational fit when two mechanisms are identical",
                "shuffled intervention outcomes",
            ],
        },
    }


def format_report(result: Mapping) -> str:
    verdict = "PASS" if result["passed"] else "FAIL"
    lines = [
        f"{result['name']} {verdict}",
        result["question"],
        f"Seeds passed: {result['seed_passes']} / {result['seed_count']} "
        f"(required {result['minimum_seed_passes']})",
        "",
        "Gates (seeds passed):",
    ]
    for name, count in result["gate_passes"].items():
        lines.append(f"  {name}: {count}/{result['seed_count']}")
    lines.append("")
    lines.append("Mean metrics:")
    for name, value in result["metric_means"].items():
        lines.append(f"  {name}: {value:.4f}")
    lines.append("")
    lines.append("Allowed claims if the gate passed:")
    for claim in result["allowed_claims"]:
        lines.append(f"  - {claim}")
    lines.append("")
    lines.append("Claims this result does not support:")
    for claim in result["disallowed_claims"]:
        lines.append(f"  - {claim}")
    return "\n".join(lines)
