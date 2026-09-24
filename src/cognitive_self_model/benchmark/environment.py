"""
Frozen benchmark environment (oracle + true causal data-generating process).

The environment produces :class:`AgentObservation` (what the estimator may see)
and :class:`GroundTruthLabels` (oracle-only truth). The oracle uses the true
mechanism coefficients internally solely to emit labels; those coefficients are
never placed into the observation.

True mechanism:   ``y_true = true_h1 * x + true_h4 * u + shared_disturbance``
Frozen reference: ``y_pred = 0.6 * x + 0.5 * u``  (see :mod:`reference_model`)

Track behavior activates after ``onset_step`` (default 20):

- CAUSAL_BREAK:              ``true_h4 = -0.5`` -> structurally & contextually invalid
- REGIME_CHANGE_IN_SCOPE:    regime flag only; reference still valid
- REGIME_CHANGE_OUT_OF_SCOPE:``true_h4 = 0.0`` -> scope violated & contextually invalid
- FALSE_ALARM:               5x noise burst for ``onset-2 <= t <= onset+2`` but valid
- NOVEL_BUT_VALID:           novel mechanism tag; reference still valid
- NOVEL_AND_INVALID:         ``true_h4 = -0.3`` + novel -> contextually invalid
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .reference_model import REFERENCE_H1, REFERENCE_H4, frozen_reference_prediction
from .tracks import AgentObservation, EnvironmentTrack, GroundTruthLabels


class ScientificFrozenEnvironment:
    """Deterministic, seed-controlled benchmark environment for one episode."""

    def __init__(
        self,
        track: EnvironmentTrack,
        noise_std: float = 0.05,
        probe_measurement_std: float = 0.01,
        seed: int = 42,
        onset_step: int = 20,
    ) -> None:
        self.track = track
        self.noise_std = float(noise_std)
        self.probe_measurement_std = float(probe_measurement_std)
        self.seed = int(seed)
        self.onset_step = int(onset_step)
        self.rng = np.random.RandomState(self.seed)
        self.t = 0

    def reset(self, seed: Optional[int] = None) -> "ScientificFrozenEnvironment":
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.RandomState(self.seed)
        self.t = 0
        return self

    def step(
        self,
        x: float,
        u: float,
        probe_inputs: List[float],
    ) -> Tuple[AgentObservation, GroundTruthLabels]:
        self.t += 1
        onset = self.t > self.onset_step

        structurally_invalid = 0
        contextually_invalid = 0
        regime_changed = 0
        scope_violated = 0
        novel = 0
        active_mechanism = "M1"

        if self.track == EnvironmentTrack.CAUSAL_BREAK and onset:
            structurally_invalid = 1
            contextually_invalid = 1
        elif self.track == EnvironmentTrack.REGIME_CHANGE_IN_SCOPE and onset:
            regime_changed = 1
        elif self.track == EnvironmentTrack.REGIME_CHANGE_OUT_OF_SCOPE and onset:
            regime_changed = 1
            scope_violated = 1
            contextually_invalid = 1
        elif self.track == EnvironmentTrack.NOVEL_BUT_VALID and onset:
            novel = 1
            active_mechanism = "M_NEW"
        elif self.track == EnvironmentTrack.NOVEL_AND_INVALID and onset:
            novel = 1
            active_mechanism = "M_NEW"
            structurally_invalid = 1
            contextually_invalid = 1

        z = self.rng.normal(0.0, 1.0) if self.noise_std > 0.0 else 0.0

        disturbance_scale = self.noise_std
        if self.track == EnvironmentTrack.FALSE_ALARM and (
            self.onset_step - 2 <= self.t <= self.onset_step + 2
        ):
            disturbance_scale = self.noise_std * 5.0

        shared_disturbance = z * disturbance_scale

        true_h1 = REFERENCE_H1
        true_h4 = REFERENCE_H4
        if self.track == EnvironmentTrack.CAUSAL_BREAK and onset:
            true_h4 = -0.5
        elif self.track == EnvironmentTrack.REGIME_CHANGE_OUT_OF_SCOPE and onset:
            true_h4 = 0.0
        elif self.track == EnvironmentTrack.NOVEL_AND_INVALID and onset:
            true_h4 = -0.3

        y_true = true_h1 * x + true_h4 * u + shared_disturbance
        y_pred = frozen_reference_prediction(x, u)

        probe_observations = {}
        for probe_u in probe_inputs:
            p_noise = (
                self.rng.normal(0.0, self.probe_measurement_std)
                if self.probe_measurement_std > 0.0
                else 0.0
            )
            probe_observations[f"probe_{probe_u:.8f}"] = (
                true_h1 * x + true_h4 * probe_u + shared_disturbance + p_noise
            )

        observation = AgentObservation(
            t=self.t,
            x=float(x),
            u=float(u),
            y_obs=float(y_true),
            y_pred_model=float(y_pred),
            observable_context_features={
                "ambient_vibration": float(abs(shared_disturbance)),
                "input_energy": float(x**2 + u**2),
            },
            probe_observations=probe_observations,
        )

        ground_truth = GroundTruthLabels(
            is_structurally_invalid=structurally_invalid,
            is_contextually_invalid=contextually_invalid,
            is_regime_changed=regime_changed,
            is_scope_violated=scope_violated,
            is_novel=novel,
            true_active_mechanism=active_mechanism,
        )

        return observation, ground_truth
