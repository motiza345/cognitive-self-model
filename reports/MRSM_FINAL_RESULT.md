# MRSM final result

One preregistered P run and one preregistered Q run were scored. Thresholds, holdout membership, H3 transforms, baselines, and the planted identity were not changed after `p_freeze`.

## 1. Preregistration identity

`configs/mrsm_prereg.yaml`

- id: `mrsm-prereg-2026-09-29`
- status: `PREREGISTERED`
- frozen_on: `2026-09-29T16:55:00Z`
- `p_freeze` commit: `2cccafd047044332828eaf602b69f4267852cba2`
- committer time: `2026-09-29T17:58:18+00:00`
- MRSM deadline: `2026-10-27T17:58:18+00:00`
- P checkpoint sha256: `6d2bdaad604ab544b47109c262d41d68915cf2f0fdf1c6343c8e7925876e0bd9`
- Holdout seed `424242`, n `512`, hash `c6b3f1b4b144a63d0f9665a6d9a06e6e0e27423b1a29533b8faab36da43fd8de`
- P run git commit: `e5963b21d7f0eb20c951282e299b9bb00368f24f` (YAML parse only; not `p_freeze`)

The first construction attempt, commit `4793cd4`, recorded `P_CONSTRUCTION_INCOMPLETE` for checkpoint `ea40cda33a054c1c7fd9b169686c2eff6a9ada1aab9f0d75324fbdd74a8986e6` (intact accuracy `0.318359375`). That attempt used the written P_SPEC schedule (`n_steps: 1600`, learning rate `0.002`, post-softmax mask). DEC-011 stopped the holdout experiment there. The later `DONE` record supersedes that budget outcome. It does not delete the failed attempt.

Engineering correction recorded before any holdout score: `pre_softmax_two_phase_planter`. Phase 1 is 2000 steps and phase 2 is 1000 steps, learning rate `0.001`, pre-softmax mask plus a score penalty. `spec_n_steps: 1600` and `spec_lr: 0.002` were stored and not used. `P_SPEC.md` was not edited. Development audit after the correction, training mask removed: intact `0.998046875`, hop1-only `0.056640625`, hop2-only `0.302734375`, both ablated `0.052734375`, specificity gap `0.8863932291666666`, sequential signature true. Holdout was locked after that audit and was not scored during construction.

## 2. P result

Run: `artifacts/mrsm/p_run_001/`

Status: `SCORED`

Leakage: `PASS`

| gate | status |
| --- | --- |
| P.H1 | PASS |
| P.H2 | PASS |
| P.H3 | PASS |
| P.H4 | PASS |

H1 predicted `L0H0 -> L1H1`. Identity match true. Sign match true. Mechanism F1 `1.0`. Causal-edge precision `1.0`. False discovery rate `0.0`. This gate verifies the frozen planted mechanism. The evaluator loaded ground truth only at scoring.

H2 self MAE `0.2943297701680826`. Strongest baseline is B2 at `1.450227096832047` (reduction `0.797045738001288`). Sign accuracy `1.0`. Paired bootstrap of (baseline MAE − self MAE): low `1.0216825655765003`, high `1.244936469692685`.

H3 sign agreement `1.0`. Normalized prediction difference `1.1181675733888438e-16`. Mechanism abstraction and scope stayed the same under the frozen inverse readout of T1, T2, and T3. No retraining.

H4 full action `L1H2` (minimum predicted margin drop, then lowest head index). Blind action `L0H3`. Utility is holdout accuracy: full `1.0`, blind `0.8671875`, relative improvement `0.15315315315315314`. Bootstrap low `0.1015625`.

## 3. Q result

Run: `artifacts/mrsm/q_run_001/`

Artifact: `AVAILABLE`

Model: `Qwen/Qwen2.5-0.5B`

Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`

Weight sha256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`

Substitution: false. Shape check: 24 layers, hidden size 896. Device: CPU.

Status: `SCORED`

Leakage: `PASS` (`planted_ground_truth_loaded: false`, predictions written before outcomes)

| gate | status |
| --- | --- |
| Q.H1 | FAIL |
| Q.H2 | FAIL |
| Q.H3 | PASS |
| Q.H4 | FAIL |

Q.H1 is diagnostic/transfer evidence. It is not ground-truth mechanism verification. The full-discovery candidate was `L0H0 -> L0H1`. Discovery half A returned `L0H2 -> L0H3`. Half B returned `L0H0 -> L0H1`. Member effects were not positive on both halves. `verifies_mechanism_identity` is false. This FAIL retires the Q transfer claim for this benchmark. It is not a proof that mechanism discovery is impossible on real models.

## 4. H1

- P.H1: `PASS`. Verified against the frozen edge `L0H0 -> L1H1`.
- Q.H1: `FAIL`. Halves disagreed and member-effect signs were not consistent. Diagnostic only.

## 5. H2

- P.H2: `PASS`. Self MAE beat B2 by more than 20 percent, sign accuracy was 1, and the bootstrap lower bound was above 0.
- Q.H2: `FAIL`. Self MAE `0.027457769270296452` versus strongest baseline B2 `0.02828113017258821` (reduction `0.02911343702557576`, below `0.20`). Sign accuracy `0.7222222222222222`, below `0.75`. Bootstrap low `-0.0008684485046951754`, so the interval includes 0.

## 6. H3

- P.H3: `PASS` by the frozen inverse readout.
- Q.H3: `PASS` by the same inverse readout on the eight intervention slots. Sign agreement `1.0`. Normalized prediction difference `1.7325896106872708e-16`. Abstraction and scope matched. This is not a claim that a Qwen mechanism was recovered.

## 7. H4

- P.H4: `PASS`. Full `L1H2` utility `1.0` versus blind `L0H3` utility `0.8671875`.
- Q.H4: `FAIL`. Utility is mean holdout logit margin. Full action `L1H3` utility `-2.6471707820892334`. Blind `L0H3` utility `-2.73256786664327`. Relative improvement `0.03125158778176652`, below `0.10`. Bootstrap low `0.0459902286529541` is above 0, and that single condition does not pass the gate.

## 8. Baseline comparison

P holdout MAE:

| baseline | MAE |
| --- | --- |
| B0 null | 7.863636495286806 |
| B1 readout-only | 4.7821147997035744 |
| B2 linear probe | 1.450227096832047 |
| B3 cold-start | 7.863636495286806 |
| self-model | 0.2943297701680826 |

Q validation MAE:

| baseline | MAE |
| --- | --- |
| B0 null | 0.053908363536552144 |
| B1 readout-only | 0.036957356664869524 |
| B2 linear probe | 0.02828113017258821 |
| B3 cold-start | 0.053908363536552144 |
| self-model | 0.027457769270296452 |

Primary comparator on both arms is the lowest MAE among B0–B3. On both arms that comparator is B2.

## 9. Leakage audit

P: `PASS`. Ground truth was not loaded before prediction. Holdout outcomes were not loaded before prediction. Ground truth was loaded at scoring. H3 parameters match the preregistration. Source, state, and prediction audits passed.

Q: `PASS`. The planted ground-truth file was not loaded. Predictions were written with outcomes absent. The loaded revision matches the pin.

No run was marked `INVALID`.

## 10. Failures

The first planter failed its own causal predicate and stopped the program under DEC-011. That stop was an engineering failure. The correction above is what allowed construction to reach `DONE`. The scientific code was not edited after these scores to improve them.

Q.H1, Q.H2, and Q.H4 failed the frozen rules. Those failures are the scientific result on this Q transfer. They are not averaged with the passing gates.

The static-weight detector remains `NOT_SEPARATELY_SPECIFIED`.

## 11. Scope

P scope string: `P:episodic-two-hop:query-position:n_layers=2:n_heads=4`.

Q scope string: `Q:m22_1_residual:diagnostic`. Eight residual interventions on layers 0, 8, 15, and 23, primary and orthogonal directions, mapped onto the frozen eight-slot order. Discovery prompts are the M22.1 discovery role. The scored split is the M22.1 validation role.

Full runs used: 2 of 8 (P holdout read, Q validation read). Q GPU full runs used: 0 of 2.

## 12. Non-claims

- Q.H1 `FAIL` is not evidence that discovery is impossible.
- Q.H1 is not a verified mechanism identity. There is no Q ground-truth label in this benchmark.
- Q.H3 `PASS` is the algebraic inverse of the frozen 8-dimensional transforms. It does not identify a Qwen circuit.
- P.H1 `PASS` verifies only the planted L0H0 → L1H1 instrument under the frozen rule.
- The engineering correction changed the training schedule relative to the written 1600-step, learning-rate-0.002 line in `P_SPEC.md`. That deviation is recorded. It is not a silent match to the original hyperparameters.
- No threshold, holdout id, H3 matrix, baseline, or planted identity was moved after the scores.

## 13. DEC-010 terminal decision

Eight-key record:

| key | status |
| --- | --- |
| P.H1 | PASS |
| P.H2 | PASS |
| P.H3 | PASS |
| P.H4 | PASS |
| Q.H1 | FAIL |
| Q.H2 | FAIL |
| Q.H3 | PASS |
| Q.H4 | FAIL |

FIRST_MATCH: P.H1–H4 are `PASS`, and all Q.H1–H4 are evaluated, and at least one Q gate is `FAIL`.

Terminal decision: `REDEFINE_SCALE`

Under DEC-010-A1, this retires the Q transfer claim for this benchmark only. It is not `GO` and it is not `STOP`.
