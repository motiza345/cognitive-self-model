"""
Raw self-model invalidity estimator.

A standardized L2-regularized logistic regression that maps the 16-D evidence
vector to a raw invalidity probability ``q_invalid`` in ``[0, 1]``. It is trained
only on ``(phi, self_model_invalid)`` pairs and never receives track identity,
mechanism, future information, or calibration output. Training is deterministic
(zero initialization, full-batch gradient descent).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SelfModelInvalidityEstimator:
    def __init__(self, feature_dim: int, l2_reg: float = 0.01) -> None:
        self.feature_dim = int(feature_dim)
        self.l2_reg = float(l2_reg)
        self.weights = np.zeros(self.feature_dim, dtype=float)
        self.bias = 0.0
        self.mean_X: Optional[np.ndarray] = None
        self.std_X: Optional[np.ndarray] = None
        self.is_trained = False

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 800,
        lr: float = 0.1,
    ) -> "SelfModelInvalidityEstimator":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError("Invalid X shape.")
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("Empty or inconsistent training data.")
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("Training labels must be binary.")

        self.mean_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean_X) / self.std_X

        for _ in range(int(epochs)):
            logits = X_norm @ self.weights + self.bias
            probabilities = self._sigmoid(logits)
            error = probabilities - y
            gradient_w = (X_norm.T @ error) / len(X_norm) + self.l2_reg * self.weights
            gradient_b = float(np.mean(error))
            self.weights -= float(lr) * gradient_w
            self.bias -= float(lr) * gradient_b

        self.is_trained = True
        return self

    def predict_logit(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.mean_X is None or self.std_X is None:
            raise RuntimeError("Estimator must be fitted first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.mean_X) / self.std_X
        return X_norm @ self.weights + self.bias

    def predict_raw_probability(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(self.predict_logit(X))
