# Review of the Cognitive Layer library v0.2

Scope: the uploaded `cognitive_layer_v02` package (about 1100 lines, 15 tests) and its description document. Everything below was run in an offline sandbox; no new model calls.

## Verdict

The design is sound and small enough to finish. **The library as delivered had one critical numerical bug that its synthetic tests could not see**; with a 20-line fix it passes the acceptance test defined for it (replaying the cached M34a data). The description document's simulation results were not re-run by me, and all of them are on synthetic data.

## What I checked

| item | result |
|---|---|
| Tests | 15 of 15 pass on the patched library (13 fast tests in 6 s, the two simulation tests in 24 s and 210 s). On the original, the tests I ran pass, which is exactly why the bug was not caught |
| Design of exploration and weights | Inverse-propensity weights are correct for the observation probability (instances observed with probability epsilon get weight 1/epsilon; greedy observing actions get weight 1). Forced learning mode (only outcome-revealing actions while the most specific claim is uncertain) addresses the exploration trap from S-SIM |
| Ledger and replay | Hash chain and state reconstruction from the event ledger are covered by tests and pass |
| Acceptance test (not in the delivered package) | Replay of M34a v2 and v3 caches through the layer: **failed on v0.2, passes on v0.2.1** (below) |
| Simulation numbers in the document | **Not re-run** (the sensitivity script takes many minutes); the ordering test passes |

## The bug

`calibration.fit_platt` used an undamped Newton iteration. On real answer confidences (`exp` of the sum of token log probabilities) 18.5% of the M34a v2 calibration items sit at the clip boundary (conf at least 1 - 1e-6) and are all correct. The iteration diverged: `a = 2174, b = 289`, negative log posterior 5.5 million against an optimum of 166 (the proper fit is `a = 0.53, b = -1.66`). The calibrated probabilities saturated at 1, the policy answered 231 of 240 problems and the test utility was **0.572**, below always-verify (0.700). Fix: damped Newton with backtracking line search (`fit_platt_fix.diff`). A regression test with the real calibration pairs as a fixture fails on 0.2.0 and passes on 0.2.1.

Process note: before finding the cause I tried three sensor and prior variants on the v2 test pool (none helped); v2 should therefore be read as the diagnosis set and v3 S1, untouched until then, as the validation set.

## Acceptance replay results (CAL pool as the learning stream, then frozen, TEST scored once)

Utilities: answer +1 or -1, abstain 0, verify 0.7. "Terciles" is the M34a method (calibration within level and within-level log-probability tercile).

| data | layer (scoped) | terciles | layer minus terciles | raw confidence | single global claim | base rate only |
|---|---:|---:|---|---:|---:|---:|
| M34a v2, multiplication (diagnosis) | 0.8667 | 0.8471 | +0.020 [+0.004, +0.040] | 0.7746 | 0.8600 | 0.7000 |
| M34a v3 S1, addition (validation) | 0.9104 | 0.8833 | +0.027 [-0.0004, +0.055] | 0.9012 | 0.9075 | 0.8167 |

Replay references: always answer 0.517 (v2) and 0.742 (v3 S1); instance oracle 0.928 and 0.961.

## What this does and does not show

- **Shows:** the layer, once the fit is correct, matches or slightly exceeds the best nonparametric M34a method on two real caches (v3 S1 interval touches zero), so the engineering of "calibrated own-output probability to cost-aware decision" works. Replays are deterministic across seeds in practice (the policy only explores when the greedy action hides the outcome).
- **Does not show:** that scoped claims matter. Per-level scoping beats a single global claim by only +0.007 (v2) and +0.003 (v3 S1). The calibration step matters much more on v2 (+0.092 over raw confidence) than on v3 S1 (+0.009). The value of scoping has to be tested where the confidence-correctness relation really differs between scopes (heterogeneous tools or tasks, or an implanted agent-specific failure), otherwise the expected result is "global calibration is enough".
- The replay uses pools that were already analysed in M34a; it is an acceptance test, not new evidence.

## Remaining issues (for 0.3)

1. **Cost:** every observation triggers a refit and an AUROC computation over all stored observations (cubic overall; the long test takes 210 s for 1500 steps). Replace by incremental statistics or a window before scale.
2. **AUROC is unweighted** while the Platt fit is weighted; with epsilon 0.05 the mismatch is small but should be made consistent.
3. **Platt is monotone with two parameters**; fine for these data, not a general calibrator.
4. **Clip boundary:** logit clipping at 1e-6 discards information among the most confident items; the fixed fit copes, but a sensor that works on the log-probability scale would use it fully.
5. **Scopes are defined by hand**; automatic scope proposals are planned for 0.3.
6. **Document claims:** all published numbers are synthetic; the statement that raw confidence is the worst group and the size of the gains should be re-measured on real caches with all arms (the replay script does this).

## Recommendations

1. Adopt `cognitive_layer_v0.2.1` (patched files, regression test, replay script, changelog) and run the 15 + 1 tests.
2. Treat definition-of-done for the library as met: tests pass and the replay reproduces the M34a results on two caches.
3. Next, one decisive pre-registered evaluation of the layer, with the ceiling gate checked in a pilot first, and a **scope-heterogeneous** condition, because that is the only place where the scoped-claims design can add value over a single calibrated confidence.
