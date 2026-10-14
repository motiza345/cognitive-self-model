"""Regression test for the v0.2 Platt divergence on real log-probability confidences.

The fixture holds the 600 calibration pairs of M34a v2 (-(sum of token log probabilities), correct).
Confidence is exp(sum of token log probabilities); about 18% of the items sit at the clip boundary
(conf >= 1 - 1e-6) and are all correct. The undamped Newton iteration of v0.2 diverged here
(a = 2174, negative log posterior 5.5e6 against an optimum of 166). The fit must reach the MAP optimum.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from cognitive_layer.calibration import fit_platt, logit, sigmoid

FIXTURE = Path(__file__).parent / "fixtures" / "m34a_v2_cal_logprob_pairs.json"


def test_platt_reaches_map_optimum_on_real_logprob_confidences():
    pairs = np.array(json.loads(FIXTURE.read_text())["pairs"], dtype=float)
    conf = np.exp(-pairs[:, 0])
    y = pairs[:, 1]
    ps = 2.0
    X = np.column_stack([logit(conf), np.ones_like(conf)])
    mu = np.array([1.0, 0.0])

    def nlp(th):
        z = X @ th
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * ps * np.sum((th - mu) ** 2))

    def grad(th):
        return X.T @ (sigmoid(X @ th) - y) + ps * (th - mu)

    opt = minimize(nlp, mu, jac=grad, method="L-BFGS-B")
    fit = fit_platt(conf, y.astype(bool), prior_strength=ps)
    assert nlp(np.array([fit.a, fit.b])) <= opt.fun * 1.001 + 1e-6
    assert abs(fit.a) < 20.0
