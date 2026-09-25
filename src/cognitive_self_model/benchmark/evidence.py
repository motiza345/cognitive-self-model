"""
Frozen 16-dimensional endogenous evidence encoder.

The encoder maps a single :class:`AgentObservation` (plus its own internal
rolling history) to a fixed 16-D feature vector ``phi``. It is *endogenous*: it
consumes only residuals, probe responses, and observable context. It never sees
track identity, hidden mechanism, or ground-truth labels, which is what makes it
leakage-safe (see the label-agnostic tests).

Feature groups:
    RR (6) residual statistics, PT (2) temporal persistence,
    DD (2) causal probe discrepancy, CC (2) causal structure,
    HH (2) historical reliability, XX (2) observable context.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .reference_model import frozen_reference_prediction
from .tracks import AgentObservation

FEATURE_NAMES = [
    "R_abs_residual_t",
    "R_squared_residual_t",
    "R_mean_abs_residual_window",
    "R_residual_variance_window",
    "R_abs_residual_p90_window",
    "R_residual_zscore_t",
    "P_failure_run_length",
    "P_extreme_residual_fraction",
    "D_mean_probe_discrepancy",
    "D_max_probe_discrepancy",
    "C_probe_discrepancy_variance",
    "C_probe_coverage_fraction",
    "H_residual_anomaly_fraction",
    "H_mean_probe_discrepancy_history",
    "X_ambient_vibration",
    "X_input_energy",
]
FEATURE_DIM = len(FEATURE_NAMES)
assert FEATURE_DIM == 16

# Maximum number of probes assumed for the coverage feature normalization.
MAX_PROBES = 5


class ScientificFrozenEncoder:
    """Stateful, frozen encoder producing a 16-D evidence vector per timestep."""

    def __init__(self, window_size: int = 15) -> None:
        self.window_size = int(window_size)
        self.residual_history: deque = deque(maxlen=self.window_size)
        self.probe_disc_history: deque = deque(maxlen=self.window_size)
        self.failure_run_length = 0

    def reset(self) -> None:
        self.residual_history.clear()
        self.probe_disc_history.clear()
        self.failure_run_length = 0

    def encode(self, obs: AgentObservation) -> np.ndarray:
        residual_t = obs.y_obs - obs.y_pred_model
        self.residual_history.append(residual_t)

        residuals = np.asarray(self.residual_history, dtype=float)
        abs_residuals = np.abs(residuals)

        mean_residual = float(np.mean(residuals))
        residual_std = float(np.std(residuals) + 1e-6)
        residual_z = float((residual_t - mean_residual) / residual_std)

        residual_features = [
            float(abs_residuals[-1]),
            float(residual_t**2),
            float(np.mean(abs_residuals)),
            float(np.var(residuals)) if len(residuals) > 1 else 0.0,
            float(np.percentile(abs_residuals, 90)),
            residual_z,
        ]

        if abs(residual_z) > 2.0:
            self.failure_run_length += 1
        else:
            self.failure_run_length = 0

        persistence_features = [
            float(self.failure_run_length),
            float(np.mean(abs_residuals > (np.mean(abs_residuals) + 2.0 * residual_std))),
        ]

        causal_discrepancies = []
        for probe_key, probe_value in obs.probe_observations.items():
            probe_u = float(probe_key.split("_", maxsplit=1)[1])
            expected_probe_delta = (
                frozen_reference_prediction(obs.x, probe_u) - obs.y_pred_model
            )
            observed_probe_delta = probe_value - obs.y_obs
            causal_discrepancies.append(
                abs(observed_probe_delta - expected_probe_delta)
            )

        mean_probe_disc = (
            float(np.mean(causal_discrepancies)) if causal_discrepancies else 0.0
        )
        max_probe_disc = (
            float(np.max(causal_discrepancies)) if causal_discrepancies else 0.0
        )
        self.probe_disc_history.append(mean_probe_disc)

        discrepancy_features = [mean_probe_disc, max_probe_disc]

        causal_structure_features = [
            float(np.var(causal_discrepancies))
            if len(causal_discrepancies) > 1
            else 0.0,
            float(len(causal_discrepancies)) / float(MAX_PROBES),
        ]

        historical_features = [
            float(np.mean(abs_residuals > (np.mean(abs_residuals) + 1.5 * residual_std))),
            float(np.mean(self.probe_disc_history)),
        ]

        context_features = [
            float(obs.observable_context_features.get("ambient_vibration", 0.0)),
            float(obs.observable_context_features.get("input_energy", 0.0)),
        ]

        phi = np.asarray(
            residual_features
            + persistence_features
            + discrepancy_features
            + causal_structure_features
            + historical_features
            + context_features,
            dtype=float,
        )
        phi = np.nan_to_num(phi, nan=0.0, posinf=1e6, neginf=-1e6)
        assert phi.shape == (FEATURE_DIM,)
        return phi
