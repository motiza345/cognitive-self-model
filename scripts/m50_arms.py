"""M50 arms, probes, bootstrap and labels (offline). Protocol: docs/M50_PREREG.md."""

from __future__ import annotations

import random

import numpy as np
from scipy.optimize import minimize

from scripts.m50_common import (CORRUPTED_TASK, CORRUPTION_PROBABILITY, SEED_BOOT, SEED_CORRUPT, SEED_RANDOM_VERIFY,
                                add_layer_to_path)

add_layer_to_path()
from cognitive_layer import (CalibrationRegistry, CalibrationRule, CognitiveLayer, ExpectedUtilityPolicy, Scope,  # noqa: E402
                             UtilityTable)
from cognitive_layer.core import OBSERVING_ACTIONS  # noqa: E402

U = UtilityTable()
DRAWS = 5000


# ----------------------------------------------------------------------------- conditions
def apply_condition(items: list[dict], condition: str, seed: int = SEED_CORRUPT) -> list[dict]:
    """natural: unchanged. implanted: every item of the corrupted task is, with probability 0.3, wrong
    (the corruption flag is drawn in problem_id order from the seeded generator)."""
    if condition == "natural":
        return [dict(r) for r in items]
    if condition != "implanted":
        raise ValueError(condition)
    rng = random.Random(seed)
    flags = {}
    for r in sorted((r for r in items if r["task"] == CORRUPTED_TASK), key=lambda r: r["problem_id"]):
        flags[r["problem_id"]] = rng.random() < CORRUPTION_PROBABILITY
    out = []
    for r in items:
        copy = dict(r)
        if flags.get(r["problem_id"], False):
            copy["correct"] = False
        out.append(copy)
    return out


# ----------------------------------------------------------------------------- simple arms
def util_from_p(p: float, correct: bool) -> float:
    """Act when the expected utility 2p - 1 exceeds the verify value, else verify (abstain never wins)."""
    if 2.0 * p - 1.0 > U.verify:
        return 1.0 if correct else -1.0
    return U.verify


def arm_always_answer(test: list[dict]) -> np.ndarray:
    return np.array([1.0 if r["correct"] else -1.0 for r in test])


def arm_always_verify(test: list[dict]) -> np.ndarray:
    return np.full(len(test), U.verify)


def arm_from_probabilities(test: list[dict], probs) -> np.ndarray:
    return np.array([util_from_p(float(p), r["correct"]) for p, r in zip(probs, test)])


def arm_oracle_instance(test: list[dict]) -> np.ndarray:
    return np.array([max(1.0 if r["correct"] else -1.0, U.verify, 0.0) for r in test])


def arm_random_verify(test: list[dict], q: float, seed: int = SEED_RANDOM_VERIFY) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([U.verify if rng.random() < q else (1.0 if r["correct"] else -1.0) for r in test])


# ----------------------------------------------------------------------------- layer arms
def run_layer(cal: list[dict], test: list[dict], scopes: list[tuple[str, int]], seed: int, mode: str = "scoped",
              scoped: bool = True, epsilon: float = 0.05):
    reg = CalibrationRegistry(CalibrationRule(min_each=15))
    if scoped:
        for task, level in scopes:
            reg.declare(f"{task}:{level}", f"confidence informative for {task} at level {level}", Scope.of(task=task, level=level))
    layer = CognitiveLayer(reg, ExpectedUtilityPolicy(U, epsilon=epsilon, seed=seed), mode=mode, seed=seed)
    stream = cal[:]
    random.Random(seed).shuffle(stream)
    learn = []
    for r in stream:
        pred = layer.before({"task": r["task"], "level": r["level"]}, r["conf"])
        layer.after(pred.prediction_id, r["correct"] if pred.action in OBSERVING_ACTIONS else None, provenance="m50:cal")
        learn.append(U.realized(pred.action, r["correct"]))
    layer.freeze()
    util, verified = [], []
    for r in test:
        pred = layer.before({"task": r["task"], "level": r["level"]}, r["conf"])
        layer.after(pred.prediction_id, r["correct"] if pred.action in OBSERVING_ACTIONS else None, provenance="m50:test")
        util.append(U.realized(pred.action, r["correct"]))
        verified.append(1.0 if pred.action.value == "VERIFY" else 0.0)
    return np.array(util), np.array(verified), float(np.mean(learn)) if learn else float("nan")


def run_layer_avg(cal, test, scopes, seeds, **kw):
    utils, vers, learns = [], [], []
    for s in seeds:
        u, v, l = run_layer(cal, test, scopes, s, **kw)
        utils.append(u), vers.append(v), learns.append(l)
    return np.mean(utils, axis=0), np.mean(vers, axis=0), float(np.mean(learns))


# ----------------------------------------------------------------------------- probes
class Probe:
    def __init__(self, mu, sd, weights, n_scopes):
        self.mu, self.sd, self.weights, self.n_scopes = mu, sd, weights, n_scopes

    def predict(self, X, scope_idx):
        Z = _design(X, scope_idx, self.n_scopes, self.mu, self.sd)
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.weights, -500, 500)))


def _design(X, scope_idx, n_scopes, mu, sd):
    Z = (np.asarray(X, dtype=np.float64) - mu) / sd
    onehot = np.eye(n_scopes)[np.asarray(scope_idx, dtype=int)]
    return np.hstack([Z, onehot, np.ones((Z.shape[0], 1))])


def _fit_ridge_logistic(Z, y, lam):
    n, d = Z.shape
    penal = np.ones(d)
    penal[-1] = 0.0  # no penalty on the intercept

    def fun(w):
        z = Z @ w
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * np.sum(penal * w * w))

    def jac(w):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -500, 500)))
        return Z.T @ (p - y) + lam * penal * w

    return minimize(fun, np.zeros(d), jac=jac, method="L-BFGS-B", options={"maxiter": 300}).x


def fit_probe(X, y, scope_idx, n_scopes, lambdas=(1e-1, 1.0, 10.0, 100.0, 1000.0), folds: int = 5, seed: int = 0) -> Probe:
    """L2 logistic regression on standardised hidden states plus scope one-hot, penalty chosen by k-fold
    cross-validation of the log loss on the calibration pool only."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = _design(X, scope_idx, n_scopes, mu, sd)
    order = np.random.default_rng(seed).permutation(len(y))
    fold_of = np.empty(len(y), dtype=int)
    fold_of[order] = np.arange(len(y)) % folds
    best_lam, best_loss = lambdas[0], float("inf")
    for lam in lambdas:
        losses = []
        for k in range(folds):
            tr, va = fold_of != k, fold_of == k
            if y[tr].min() == y[tr].max():
                continue
            w = _fit_ridge_logistic(Z[tr], y[tr], lam)
            p = np.clip(1.0 / (1.0 + np.exp(-np.clip(Z[va] @ w, -500, 500))), 1e-6, 1 - 1e-6)
            losses.append(float(-np.mean(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))))
        if losses and np.mean(losses) < best_loss:
            best_lam, best_loss = lam, float(np.mean(losses))
    return Probe(mu, sd, _fit_ridge_logistic(Z, y, best_lam), n_scopes)


# ----------------------------------------------------------------------------- bootstrap and labels
def bootstrap_diff(a, b, draws: int = DRAWS, seed: int = SEED_BOOT) -> dict:
    """Paired percentile bootstrap over items; also a two-sided bootstrap p-value (for Holm)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (draws, len(d)))
    means = d[idx].mean(axis=1)
    p_two = 2.0 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0)))
    return {"mean": float(d.mean()), "lower": float(np.percentile(means, 2.5)), "upper": float(np.percentile(means, 97.5)),
            "p": min(1.0, max(p_two, 1.0 / draws))}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni: returns, per test name, whether it is significant after correction."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still = {}, True
    for rank, (name, p) in enumerate(items):
        if still and p <= alpha / (m - rank):
            out[name] = True
        else:
            still = False
            out[name] = False
    return out


def label_floor(prefix: str, delta: float, lower: float, floor: float) -> str:
    if lower > 0 and delta >= floor:
        return f"{prefix}_SUPPORTED"
    if lower > 0:
        return f"{prefix}_WEAK"
    return f"{prefix}_NOT_SUPPORTED"


def label_probe(delta: float, lower: float, upper: float) -> str:
    if lower > 0 and delta >= 0.02:
        return "PROBE_BETTER"
    if upper < 0 and delta <= -0.02:
        return "LOGPROB_BETTER"
    return "UNRESOLVED"


def condition_labels(condition: str, gate_ok: bool, comps: dict) -> dict:
    """comps: dict with keys A, B, C, D each {'mean','lower','upper'}. Every input maps to a label."""
    if not gate_ok:
        return {"A": "NOT_INFORMATIVE", "B": "NOT_INFORMATIVE", "C": "NOT_INFORMATIVE", "D": "NOT_INFORMATIVE",
                "status": "NOT_INFORMATIVE"}
    labels = {
        "A": label_floor("A", comps["A"]["mean"], comps["A"]["lower"], 0.03),
        "B": label_floor("B", comps["B"]["mean"], comps["B"]["lower"], 0.02),
        "C": label_floor("C", comps["C"]["mean"], comps["C"]["lower"], 0.02),
        "D": label_probe(comps["D"]["mean"], comps["D"]["lower"], comps["D"]["upper"]),
        "status": "OK",
    }
    if condition == "implanted" and labels["C"] != "C_SUPPORTED":
        labels["status"] = "POSITIVE_CONTROL_FAILED"
    return labels
