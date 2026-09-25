"""
Exact weighted PAVA isotonic calibrator.

Pool-Adjacent-Violators isotonic regression on raw scores. Mapping is a
monotonic step function determined by the fitted pooled blocks. Exact ``p=0`` and
``p=1`` outputs are intentionally retained as the standard PAVA result and must
not be interpreted as epistemic certainty.

Note: within M21.2.4.3.1 this calibrator is applied *after* the raw estimator,
as a downstream stage; the raw ``q_invalid`` written to the NPZ bundle is
uncalibrated by design so the M21.2.4.3.3 audit can perform calibration itself.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class IsotonicCalibrator:
    def __init__(self) -> None:
        self.block_left_edges: Optional[np.ndarray] = None
        self.block_right_edges: Optional[np.ndarray] = None
        self.block_weights: Optional[np.ndarray] = None
        self.block_positive_masses: Optional[np.ndarray] = None
        self.block_values: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=float).reshape(-1)
        labels = np.asarray(labels, dtype=float).reshape(-1)

        if len(scores) == 0:
            raise ValueError("Calibration set is empty.")
        if len(scores) != len(labels):
            raise ValueError("scores and labels length mismatch.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Scores must be finite.")
        if not np.all(np.isin(labels, [0.0, 1.0])):
            raise ValueError("Labels must be binary.")

        order = np.argsort(scores, kind="mergesort")
        scores = scores[order]
        labels = labels[order]

        unique_scores, inverse = np.unique(scores, return_inverse=True)
        grouped_weight = np.zeros(len(unique_scores), dtype=float)
        grouped_positive_mass = np.zeros(len(unique_scores), dtype=float)
        np.add.at(grouped_weight, inverse, 1.0)
        np.add.at(grouped_positive_mass, inverse, labels)

        blocks: List[Dict[str, float]] = []
        for i, score in enumerate(unique_scores):
            blocks.append(
                {
                    "left": float(score),
                    "right": float(score),
                    "weight": float(grouped_weight[i]),
                    "positive_mass": float(grouped_positive_mass[i]),
                }
            )
            while len(blocks) >= 2:
                left = blocks[-2]
                right = blocks[-1]
                left_value = left["positive_mass"] / left["weight"]
                right_value = right["positive_mass"] / right["weight"]
                if left_value <= right_value:
                    break
                blocks[-2:] = [
                    {
                        "left": left["left"],
                        "right": right["right"],
                        "weight": left["weight"] + right["weight"],
                        "positive_mass": left["positive_mass"] + right["positive_mass"],
                    }
                ]

        self.block_left_edges = np.asarray([b["left"] for b in blocks], dtype=float)
        self.block_right_edges = np.asarray([b["right"] for b in blocks], dtype=float)
        self.block_weights = np.asarray([b["weight"] for b in blocks], dtype=float)
        self.block_positive_masses = np.asarray(
            [b["positive_mass"] for b in blocks], dtype=float
        )
        self.block_values = self.block_positive_masses / self.block_weights
        self.is_fitted = True

        assert np.all(np.diff(self.block_values) >= -1e-12)
        return self

    def calibrate(self, scores: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted.")
        assert self.block_right_edges is not None
        assert self.block_values is not None

        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            raise ValueError("Scores must be finite.")

        indices = np.searchsorted(self.block_right_edges, scores, side="left")
        indices = np.clip(indices, 0, len(self.block_values) - 1)
        return self.block_values[indices]
