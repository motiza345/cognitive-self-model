"""Acceptance replay: run the cognitive layer on a cached M34a data set (no new model calls).

Learning stream = CAL pool (shuffled), then freeze, then score the TEST pool once.
Scopes = difficulty level. Confidence = exp(sum of token log probabilities) (M34a definition).
Baselines are recomputed from the same cache with the M34a definitions:
  global      p = pooled CAL accuracy
  level_cal   p = (correct + 1) / (n + 2) per level from CAL
  terciles    p per (level, within-level tercile of answer_logprob) calibrated on CAL
Utilities: act +1 / -1, abstain 0, verify 0.7.

Usage: PYTHONPATH=. python scripts/replay_m34a.py <responses.jsonl> [seeds=3] [mode ...]
modes: scoped (default), raw, base_rate_only, global_only
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from cognitive_layer import (CalibrationRegistry, CalibrationRule, CognitiveLayer, ExpectedUtilityPolicy, Scope,
                             UtilityTable, from_token_logprobs)
from cognitive_layer.core import OBSERVING_ACTIONS
from cognitive_layer.evaluation import paired_bootstrap


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return ([r for r in rows if r["pool"] == "cal"], [r for r in rows if r["pool"] == "test"],
            sorted({r["level"] for r in rows}))


def baselines(cal, test, levels, U):
    def act_if(p, ok):
        return (1.0 if ok else -1.0) if 2 * p - 1 > U.verify else U.verify
    gp = (sum(r["correct"] for r in cal) + 1) / (len(cal) + 2)
    lc = {L: (sum(r["correct"] for r in cal if r["level"] == L) + 1) / (sum(1 for r in cal if r["level"] == L) + 2)
          for L in levels}
    edges = {L: (lambda s: (s[len(s) // 3], s[2 * len(s) // 3]))(sorted(r["answer_logprob"] for r in cal if r["level"] == L))
             for L in levels}
    tr = lambda L, v: 0 if v < edges[L][0] else (1 if v < edges[L][1] else 2)
    tp = {(L, t): (sum(r["correct"] for r in cal if r["level"] == L and tr(L, r["answer_logprob"]) == t) + 1) /
                  (sum(1 for r in cal if r["level"] == L and tr(L, r["answer_logprob"]) == t) + 2)
          for L in levels for t in range(3)}
    return {
        "naive_always_answer": np.array([1.0 if r["correct"] else -1.0 for r in test]),
        "global": np.array([act_if(gp, r["correct"]) for r in test]),
        "level_cal": np.array([act_if(lc[r["level"]], r["correct"]) for r in test]),
        "terciles": np.array([act_if(tp[(r["level"], tr(r["level"], r["answer_logprob"]))], r["correct"]) for r in test]),
        "oracle_instance": np.array([max(1.0 if r["correct"] else -1.0, U.verify, 0.0) for r in test]),
    }


def run(cal, test, levels, U, seed, mode="scoped", scoped=True):
    reg = CalibrationRegistry(CalibrationRule(min_each=15))
    if scoped:
        for L in levels:
            reg.declare(f"level{L}", f"confidence informative at level {L}", Scope.of(level=L))
    layer = CognitiveLayer(reg, ExpectedUtilityPolicy(U, epsilon=0.05, seed=seed), mode=mode, seed=seed)
    stream = cal[:]
    random.Random(seed).shuffle(stream)
    for r in stream:
        pred = layer.before({"level": r["level"]}, from_token_logprobs(r["token_logprobs"]))
        layer.after(pred.prediction_id, r["correct"] if pred.action in OBSERVING_ACTIONS else None, provenance="replay:cal")
    layer.freeze()
    out, acts = [], []
    for r in test:
        pred = layer.before({"level": r["level"]}, from_token_logprobs(r["token_logprobs"]))
        layer.after(pred.prediction_id, r["correct"] if pred.action in OBSERVING_ACTIONS else None, provenance="replay:test")
        out.append(U.realized(pred.action, r["correct"]))
        acts.append(pred.action.value)
    return np.array(out), acts


def main(argv):
    path, nseeds = argv[0], int(argv[1]) if len(argv) > 1 else 3
    modes = argv[2:] or ["scoped"]
    U = UtilityTable()
    cal, test, levels = load(path)
    base = baselines(cal, test, levels, U)
    print(Path(path).parent.name, f"TEST n={len(test)}", {k: round(float(v.mean()), 4) for k, v in base.items()})
    for m in modes:
        kw = {"mode": m} if m in ("raw", "base_rate_only") else ({"scoped": False} if m == "global_only" else {})
        us = []
        for s in range(nseeds):
            u, acts = run(cal, test, levels, U, s, **kw)
            us.append(u)
        mean = np.mean(us, axis=0)
        b = paired_bootstrap(mean, base["terciles"])
        print(f"  {m:15s} TEST utility {mean.mean():.4f}  minus terciles {b['mean']:+.4f} [{b['lower']:+.4f}, {b['upper']:+.4f}]  actions {dict(Counter(acts))}")


if __name__ == "__main__":
    main(sys.argv[1:])
