"""S-SIM: simulations pre-registered in SIM_PREREG.md. One run, seeds 1..30."""
import json
import os
import numpy as np

SEEDS = list(range(1, 31))


def boot_ci(x, n=5000, seed=23001):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------------------------------------------------------- E1
W = np.array([.4, .3, .2, .1])
XSTAR = W ** 2 / (W ** 2).sum()


def util(x):
    return float((W * np.sqrt(x)).sum() - 4.0 * max(0.0, x.sum() - 1.0))


def e1(beta, arm, seed, T=30):
    rng = np.random.default_rng(1000 + seed)
    cat = int(rng.integers(4))
    mag = rng.uniform(.1, .3)
    b = np.zeros(4)
    b[cat] = beta * mag
    shift = int(rng.integers(1, 4))
    noise = rng.normal(0, .03, (T, 4))
    obs, U = [], []
    for t in range(T):
        est = np.mean(obs, 0) if obs else np.zeros(4)
        if arm == 'naive':
            bh = np.zeros(4)
        elif arm == 'self':
            bh = est
        elif arm == 'wrong':
            bh = np.roll(est, shift)
        else:
            bh = b
        p = np.clip(XSTAR - bh, 0, None)
        x = np.clip(p + b + noise[t], 0, None)
        U.append(util(x))
        obs.append(x - p)
    return float(np.mean(U))


# ---------------------------------------------------------------- E1b
def odr(u, beta=0.5, delta=1.0):
    T = len(u)

    def val(s, t):
        return u[s] if t == s else beta * delta ** (t - s) * u[t]
    naive = T - 1
    for s in range(T):
        best = max(range(s, T), key=lambda t: val(s, t))
        if best == s:
            naive = s
            break
    d = [None] * T
    d[T - 1] = T - 1
    for s in range(T - 2, -1, -1):
        wait = beta * delta ** (d[s + 1] - s) * u[d[s + 1]]
        d[s] = s if u[s] >= wait else d[s + 1]
    soph = d[0]
    return {'naive_period': naive + 1, 'naive_welfare': delta ** naive * u[naive],
            'soph_period': soph + 1, 'soph_welfare': delta ** soph * u[soph]}


# ---------------------------------------------------------------- E2
def onehot(i):
    e = np.zeros(3)
    e[i] = 1
    return e


def h_est(n_hab, n_rest):
    return min(0.33, (n_hab + 1) / (n_rest + 5))


class Hider:
    """Agent that hides with habit h and optional self-model / opponent model."""

    def __init__(self, arm, rng, h=0.3):
        self.arm, self.rng, self.h = arm, rng, h
        self.prev = int(rng.integers(3))
        self.prev_int = self.prev
        self.nh = self.nr = 0
        self.nhw = self.nrw = 0
        self.ph = np.full((3, 3), 1e-3)

    def act(self, pred_ctx_hint=None):
        arm, prev = self.arm, self.prev
        if arm in ('opp', 'both'):
            pp = self.ph[prev] / self.ph[prev].sum()
            d = np.exp(-pp / 0.15)
            d /= d.sum()
        else:
            d = np.ones(3) / 3
        if arm in ('self', 'both', 'oracle'):
            hh = self.h if arm == 'oracle' else h_est(self.nh, self.nr)
            q = np.clip(d - hh * onehot(prev), 0, None)
        elif arm == 'wrong':
            hh = h_est(self.nhw, self.nrw)
            q = np.clip(d - hh * onehot(self.prev_int), 0, None)
        else:
            q = d
        q = q / q.sum()
        self.a_int = int(self.rng.choice(3, p=q))
        habit = self.rng.random() < self.h
        self.a = prev if habit else self.a_int
        return self.a

    def update(self, pred=None):
        prev = self.prev
        if self.a_int != prev:
            self.nr += 1
            self.nh += int(self.a == prev)
        if self.a_int != self.prev_int:
            self.nrw += 1
            self.nhw += int(self.a == self.prev_int)
        if pred is not None:
            self.ph[prev] *= 0.98
            self.ph[prev][pred] += 1
        self.prev_int = self.a_int
        self.prev = self.a


class Predictor:
    def __init__(self, kind, rng):
        self.kind, self.rng = kind, rng
        self.cnt = np.zeros((3, 3))

    def predict(self, ctx):
        if self.kind == 'static':
            return 0
        row = self.cnt[ctx]
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def learn(self, ctx, a):
        self.cnt[ctx] *= 0.98
        self.cnt[ctx][a] += 1


def e2(arm, opp, seed, T=600):
    rng = np.random.default_rng(2000 + seed)
    hd = Hider(arm, np.random.default_rng(2500 + seed))
    pr = Predictor(opp, rng)
    hits = []
    for t in range(T):
        ctx = hd.prev
        pred = pr.predict(ctx)
        a = hd.act()
        hits.append(int(a == pred))
        pr.learn(ctx, a)
        hd.update(pred)
    return float(np.mean(hits[100:]))


def e2m(armA, armB, seed, T=600):
    rngp = np.random.default_rng(4000 + seed)
    A = Hider(armA, np.random.default_rng(4100 + seed))
    B = Hider(armB, np.random.default_rng(4200 + seed))
    PA = Predictor('adaptive', rngp)  # A's predictor of B
    PB = Predictor('adaptive', rngp)  # B's predictor of A
    score = []
    for t in range(T):
        # A hides, B seeks
        ctxA = A.prev
        predB = PB.predict(ctxA)
        a = A.act()
        hitB = int(a == predB)
        PB.learn(ctxA, a)
        A.update(None)
        # B hides, A seeks
        ctxB = B.prev
        predA = PA.predict(ctxB)
        b = B.act()
        hitA = int(b == predA)
        PA.learn(ctxB, b)
        B.update(None)
        score.append(hitA - hitB)
    return float(np.mean(score[100:]))


# ---------------------------------------------------------------- E3
def e3(variant, arm, seed, N=400):
    rng = np.random.default_rng(3000 + seed)
    d = rng.integers(1, 9, N)
    perr = 1 / (1 + np.exp(-1.2 * (d - 5)))
    wrong = rng.random(N) < perr
    err, tot = np.ones(9), 2 * np.ones(9)
    ge, gt = 1.0, 2.0
    U = []
    for i in range(N):
        if arm == 'naive':
            act = 'answer'
        else:
            if arm == 'oracle':
                p = perr[i]
            elif arm == 'self':
                p = err[d[i]] / tot[d[i]]
            else:
                p = ge / gt
            opts = {'answer': 1 - 2 * p}
            if variant >= 2:
                opts['abstain'] = 0.0
            if variant >= 3:
                opts['verify'] = 0.7
            act = max(opts, key=opts.get)
        if act == 'answer':
            U.append(-1.0 if wrong[i] else 1.0)
            err[d[i]] += wrong[i]
            tot[d[i]] += 1
            ge += wrong[i]
            gt += 1
        elif act == 'abstain':
            U.append(0.0)
        else:
            U.append(0.7)
    return float(np.mean(U))


def diff_stats(a, b):
    d = np.array(a) - np.array(b)
    return boot_ci(d)


def closure(base, arm, oracle):
    nb, na, no = np.mean(base), np.mean(arm), np.mean(oracle)
    return float((na - nb) / (no - nb)) if no != nb else float('nan')


def main():
    R = {}
    # E1
    for beta in (0.0, 1.0):
        res = {arm: [e1(beta, arm, s) for s in SEEDS] for arm in ('naive', 'self', 'wrong', 'oracle')}
        R[f'E1_beta{beta}'] = {
            'means': {k: float(np.mean(v)) for k, v in res.items()},
            'self_minus_naive': diff_stats(res['self'], res['naive']),
            'closure_self': closure(res['naive'], res['self'], res['oracle']),
            'wrong_minus_naive_mean': float(np.mean(res['wrong']) - np.mean(res['naive']))}
    # E1b
    R['E1b_costs'] = odr([-3, -5, -8, -13])
    R['E1b_rewards'] = odr([3, 5, 8, 13])
    # E2
    for opp in ('static', 'adaptive'):
        res = {arm: [e2(arm, opp, s) for s in SEEDS] for arm in ('base', 'self', 'wrong', 'opp', 'both', 'oracle')}
        R[f'E2_{opp}'] = {
            'hit_rate_means': {k: float(np.mean(v)) for k, v in res.items()},
            'base_minus_self': diff_stats(res['base'], res['self']),
            'self_minus_base_abs': abs(float(np.mean(res['self']) - np.mean(res['base']))),
            'closure_self': closure(res['base'], res['self'], res['oracle']) if False else float(
                (np.mean(res['base']) - np.mean(res['self'])) / (np.mean(res['base']) - np.mean(res['oracle']))),
            'wrong_minus_self': float(np.mean(res['wrong']) - np.mean(res['self'])),
            'wrong_minus_base': float(np.mean(res['wrong']) - np.mean(res['base'])),
            'base_minus_opp': diff_stats(res['base'], res['opp']),
            'base_minus_both': diff_stats(res['base'], res['both']),
            'interaction_both_opp_minus_self_base': float(
                (np.mean(res['opp']) - np.mean(res['both'])) - (np.mean(res['base']) - np.mean(res['self'])))}
    # E2m
    for a, b in (('self', 'base'), ('base', 'base'), ('self', 'self'), ('wrong', 'base')):
        sc = [e2m(a, b, s) for s in SEEDS]
        R[f'E2m_{a}_vs_{b}'] = boot_ci(sc)
    # E3
    for v in (1, 2, 3):
        res = {arm: [e3(v, arm, s) for s in SEEDS] for arm in ('naive', 'self', 'global', 'oracle')}
        R[f'E3_V{v}'] = {
            'means': {k: float(np.mean(x)) for k, x in res.items()},
            'self_minus_naive': diff_stats(res['self'], res['naive']),
            'self_minus_global': diff_stats(res['self'], res['global']),
            'closure_self': closure(res['naive'], res['self'], res['oracle'])}
    json.dump(R, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_results.json'), 'w'), indent=1)
    return R


if __name__ == '__main__':
    R = main()
    print(json.dumps(R, indent=1))
