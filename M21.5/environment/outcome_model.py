"""Evaluator-only outcome probabilities. Agents do not import this module."""

from __future__ import annotations

from fractions import Fraction


def parse_probability(value) -> Fraction:
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, (int, float)):
        return Fraction(str(value))
    raise TypeError(f"unsupported probability {value!r}")


def as_float_map(raw: dict) -> dict[str, float]:
    parsed = {key: parse_probability(value) for key, value in raw.items()}
    total = sum(parsed.values())
    if total != 1:
        raise ValueError(f"probabilities sum to {total}, not 1")
    return {key: float(value) for key, value in parsed.items()}


def optimal_action(probabilities: dict[str, float]) -> str:
    best_name = None
    best_value = None
    for name, value in probabilities.items():
        if best_value is None or value > best_value:
            best_name = name
            best_value = value
        elif value == best_value:
            raise ValueError(f"tied optimal probabilities in {probabilities}")
    if best_name is None:
        raise ValueError("empty probability row")
    return best_name


def success_probability(matrix: dict, state: str, regime: str, action: str) -> float:
    return float(matrix[state][regime][action])
