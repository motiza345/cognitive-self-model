"""Calibration and discrimination statistics. Pure functions, numpy only
(plus scipy for the exact binomial interval).

- fit_platt:  weighted logistic recalibration on logit(confidence), with a
              Gaussian prior centred on the identity map (a=1, b=0). With no
              data the calibrated value equals the raw confidence.
- auroc:      Mann-Whitney AUROC with Hanley-McNeil standard error.
- ece:        expected calibration error, equal-width bins.
- clopper_pearson: exact binomial interval (from real_claim_self_model).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, norm

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, dtype=np.float64), -500.0, 500.0)))


@dataclass(frozen=True)
class PlattFit:
    a: float
    b: float

    def apply(self, confidence: float) -> float:
        return float(sigmoid(self.a * logit(confidence) + self.b))


IDENTITY = PlattFit(1.0, 0.0)


def _neg_log_posterior(theta, X, y, w, mu, prior_strength):
    z = X @ theta
    return float(np.sum(w * (np.logaddexp(0.0, z) - y * z)) + 0.5 * prior_strength * np.sum((theta - mu) ** 2))


def fit_platt(confidence, success, weight=None, prior_strength: float = 2.0, iters: int = 100) -> PlattFit:
    """Damped Newton MAP fit of p = sigmoid(a * logit(c) + b).

    v0.2.1: backtracking line search. The undamped v0.2 iteration diverged on real
    log-probability confidences (many values at the clip boundary, all correct),
    returning a = 2174 with a negative log posterior 30000 times worse than the optimum.
    """
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(success, dtype=np.float64)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
    if c.size == 0:
        return IDENTITY
    X = np.column_stack([logit(c), np.ones_like(c)])
    mu = np.array([1.0, 0.0])
    theta = mu.copy()
    f = _neg_log_posterior(theta, X, y, w, mu, prior_strength)
    for _ in range(iters):
        p = sigmoid(X @ theta)
        grad = X.T @ (w * (p - y)) + prior_strength * (theta - mu)
        hess = (X * (w * p * (1.0 - p))[:, None]).T @ X + prior_strength * np.eye(2)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        accepted = False
        while t > 1e-12:
            cand = theta - t * step
            fc = _neg_log_posterior(cand, X, y, w, mu, prior_strength)
            if fc <= f:
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break
        moved = float(np.max(np.abs(cand - theta)))
        theta, f = cand, fc
        if moved < 1e-10:
            break
    return PlattFit(float(theta[0]), float(theta[1]))


@dataclass(frozen=True)
class Discrimination:
    auroc: float | None
    lower: float | None
    upper: float | None
    n_pos: int
    n_neg: int


def auroc(confidence, success, alpha: float = 0.05) -> Discrimination:
    c = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(success, dtype=bool)
    pos, neg = c[y], c[~y]
    n1, n0 = int(pos.size), int(neg.size)
    if n1 == 0 or n0 == 0:
        return Discrimination(None, None, None, n1, n0)
    # Mann-Whitney with ties counted as one half.
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    a = float((greater + 0.5 * ties) / (n1 * n0))
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    se = math.sqrt(max(var, 1e-12))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return Discrimination(a, max(0.0, a - z * se), min(1.0, a + z * se), n1, n0)


def ece(prob, success, bins: int = 10) -> float | None:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(success, dtype=np.float64)
    if p.size == 0:
        return None
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / p.size)


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total <= 0 or not (0 <= successes <= total):
        raise ValueError("binomial counts are outside the evidence set")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return lower, upper
