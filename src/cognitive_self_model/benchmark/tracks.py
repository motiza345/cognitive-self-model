"""
Benchmark track taxonomy and observation/label contracts.

The seven environment tracks are the canonical M21.2.4.x taxonomy. Their
comments record the intended ground-truth semantics relative to the frozen
reference self-model ``y_pred = 0.6 * x + 0.5 * u``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class EnvironmentTrack(str, Enum):
    """The seven canonical benchmark tracks."""

    KNOWN_VALID = "KNOWN_VALID"                          # reference valid, no change
    CAUSAL_BREAK = "CAUSAL_BREAK"                        # structural break -> invalid
    REGIME_CHANGE_IN_SCOPE = "REGIME_CHANGE_IN_SCOPE"    # regime shift, still valid
    REGIME_CHANGE_OUT_OF_SCOPE = "REGIME_CHANGE_OUT_OF_SCOPE"  # regime shift, invalid
    FALSE_ALARM = "FALSE_ALARM"                          # transient noise burst, valid
    NOVEL_BUT_VALID = "NOVEL_BUT_VALID"                  # novel mechanism, still valid
    NOVEL_AND_INVALID = "NOVEL_AND_INVALID"              # novel mechanism, invalid


@dataclass(frozen=True)
class AgentObservation:
    """Everything the estimator is allowed to observe at a single timestep.

    Deliberately excludes track identity, hidden mechanism, and ground-truth
    labels; those live only in :class:`GroundTruthLabels` produced by the oracle.
    """

    t: int
    x: float
    u: float
    y_obs: float
    y_pred_model: float
    observable_context_features: Dict[str, float]
    probe_observations: Dict[str, float]


@dataclass(frozen=True)
class GroundTruthLabels:
    """Oracle-only ground truth generated from the true causal mechanism.

    ``self_model_invalid`` is defined as ``is_contextually_invalid`` (the
    reference self-model is invalid for the current context). The remaining
    flags are latent descriptors used for identifiability analysis and must
    never be fed to the estimator.
    """

    is_structurally_invalid: int
    is_contextually_invalid: int
    is_regime_changed: int
    is_scope_violated: int
    is_novel: int
    true_active_mechanism: str

    @property
    def self_model_invalid(self) -> int:
        """Canonical binary target: contextual invalidity of the reference model."""
        return int(self.is_contextually_invalid)
