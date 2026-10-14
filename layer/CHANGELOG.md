# Changelog

## 0.2.1 (review fix)

- **Bug fix in `calibration.fit_platt`.** The undamped Newton iteration diverged on real log-probability
  confidences (M34a v2 calibration pool: 18.5% of items at the clip boundary, all correct): it returned
  `a = 2174, b = 289` with a negative log posterior of 5.5e6 against an optimum of 166, so the calibrated
  probabilities were saturated and the layer mostly answered (test utility 0.572). The fit now uses a damped
  Newton iteration with backtracking line search and reaches the MAP optimum (`a = 0.53, b = -1.66`).
- Added `tests/test_platt_extreme_confidences.py` with a fixture of the real M34a v2 calibration pairs
  (fails on 0.2.0, passes on 0.2.1).
- Added `scripts/replay_m34a.py`: acceptance replay on cached M34a data (no model calls).

## Acceptance replay (CAL pool as the learning stream, TEST scored once; utilities act +1/-1, abstain 0, verify 0.7)

| data | layer (scoped) | terciles (M34a method) | layer minus terciles | raw confidence | global only | base rate only |
|---|---:|---:|---|---:|---:|---:|
| M34a v2 (multiplication, diagnosis set) | 0.8667 | 0.8471 | +0.0196 [+0.0038, +0.0404] | 0.7746 | 0.8600 | 0.7000 |
| M34a v3 S1 (addition, untouched validation set) | 0.9104 | 0.8833 | +0.0271 [-0.0004, +0.0546] | 0.9012 | 0.9075 | 0.8167 |

Version 0.2.0 on the v2 data: 0.5721 (231 answers, 9 verifications).
