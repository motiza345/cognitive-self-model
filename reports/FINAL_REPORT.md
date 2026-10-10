# Self-model line: consolidation (v0.2, interim)

Status: interim. Supersedes the v0.1 closing recommendation: the owner chose to complete the remaining pre-specified tests (`docs/NEXT_STEPS.md`). Every number comes from a frozen report or manifest linked in `reports/CLAIM_LADDER.md`, or from `simulations/ssim/`.

## 1. Summary

1. **Supported (in scope):** a measurement computed before any outcome, the second-order curvature `kappa = alpha^2/2 * F''(0)`, predicts the residual of the first-order response `g` for `Qwen/Qwen2.5-0.5B` steering. Frozen holdout (M29-D) and a replication with a new direction, layers, observable and disjoint catalog (M30): relative error reduction versus `g` alone 0.66 to 0.86. The absolute error of `g` is already small (about 0.002 for the M30 new identity).
2. **Not supported:** privileged access. A black-box finite-difference quadratic with 3 forward passes matched the white-box method (S2: 0 of 4 conditions favour white-box). A learned external observer has not been tested.
3. **Exploratory:** abs(kappa) predicts the remaining error (Spearman 0.76 and 0.61), a usable reliability signal; a finite-difference probe gives the same quantity.
4. **Simulation (S-SIM, injected agent properties):** a self-model that learns a property of the agent from its own history closes 87% to 99% of the gap to an agent that knows it; no gain against a static opponent; no change in unaided accuracy, but gains when abstain or verify options exist; the self-model agent beats one without it by 0.198 hits per round in a match. Caveats: opponent model and self-model were substitutes when the opponent's predictions were visible; three placebo checks were badly specified; one arm never learned (exploration trap). This says nothing yet about LLM agents.
5. **Untested:** LLM-agent decision utility, cross-weights specificity, stability after the model is modified, a learned external observer.

## 2. What the mechanistic result means

The local response of this model along these directions is well approximated by a second-order expansion. It does not show the model knows something about itself that an outside observer cannot learn cheaply. Honest wording:

> A pre-outcome, mechanism-derived measurement predicts the residual of the first-order response on a held-out catalog for one small model and two directions. It offers no advantage over an equally cheap black-box probe (S2).

## 3. Results

| Milestone | Verdict | Key number |
|---|---|---|
| M24 | CONSUMPTION_SUPPORTED | MAE(g) 0.0059 vs stored cell mean 0.0627 |
| M26 | INCONCLUSIVE | residual update vs g: CI includes zero |
| M29-D | SUPPORTED | Delta 0.0157 [0.0098, 0.0223] vs B1; post hoc vs g alone 0.0080 [0.0047, 0.0115] |
| M30 new identity | SUPPORTED | relative reduction 0.858; MAE(g) 0.0022 |
| M30 anchor | ANCHOR_SUPPORTED | relative reduction 0.660 |
| S1 steering | UNDEFINED_BY_RULE | decision rule was not exhaustive; STOP label withdrawn |
| S2 FDQ control | MIXED | 0/4 white-box advantage |
| S-SIM | 9 of 13 predictions met | see `simulations/ssim/SIM_RESULTS.md` |

## 4. Methodological findings

- **M21.5** (synthetic bandit v1): the instrument could not reward a self-model (detector fired in 2 of 40 runs; ceiling about 0.019; no recurrence; DSR 0 for all; two degenerate ablations).
- **M21.6** (gate-first redesign): gates G1 to G4 passed on tuning seeds, the identifiability gate G5 failed after two pre-registered redesigns, evaluation seeds never run; an ideal observer reached NMI only 0.31. See `M21_6_RECOVERED_RECORD.md` (transcribed, not hash-locked).
- **Lessons:** exhaustive decision rules before running; "no correction" as a baseline (B1 was worse than g alone); non-degenerate metrics; ideal observer to calibrate identifiability; disjoint tuning and evaluation seeds; a placebo must be a true null (S-SIM showed a "wrong" self-model can still carry information); learning from own outcomes needs exploration.

## 5. Interim decision

Continue, bounded: run the remaining tests in `docs/NEXT_STEPS.md` in the stated order (M35, M31, M32, M33, M34, M36 optional). Checkpoints: after M31 to M33, decide whether the mechanistic claim gains privileged access or stays narrow. After M34 (real LLM agents), decide whether the project continues; the project-specific claim needs a result beyond plain calibration from a track record. Expected total: about one to two weeks.

M34a is closed after v3: v1 `NOT_INFORMATIVE`; v2 A=`SUPPORTED_WEAK`, C=`INTERNAL_VALUE` on Qwen `mul_n1` 2–7; v3 aggregate C=`C_NOT_INFORMATIVE` (S1 add_nn A=`NOT_SUPPORTED` / C=`NOT_INFORMATIVE`; S2 Phi pilot STOP, no pools). The v2 instance-level logprob claim did not generalize under the preregistered settings. Remaining M34 work (planning/games) is not started.

## 6. Reproducibility

Repository `github.com/motiza345/cognitive-self-model`; tag `v0.1-mechanistic-line` at `6193bfa`. Frozen hashes: M30 catalog `4ceb314e3...1770`, M30 predictions `f363d9386...bb5`, M29-D predictions `9ff6a11e6...33a`. Simulation: `python simulations/ssim/sim_all.py` reproduces `sim_results.json` exactly. M21.6 and M21.7 artifacts are not on the remote.
