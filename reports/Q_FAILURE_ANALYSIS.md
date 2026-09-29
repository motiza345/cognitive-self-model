# Q Failure Analysis

## Frozen MRSM Result

This file is a diagnostic follow-up. It does not replace `reports/MRSM_FINAL_RESULT.md`.

Q gates, unchanged:

H1 = FAIL
H2 = FAIL
H3 = PASS
H4 = FAIL

P gates remain PASS. Leakage on P and Q remains PASS. DEC-010 remains `REDEFINE_SCALE`.
`p_freeze` remains `2cccafd047044332828eaf602b69f4267852cba2`.
Q revision remains `060db6499f32faf8b98477b0a26969ef7d8b9987`.
P holdout remains seed 424242, n 512, hash `c6b3f1b4b144a63d0f9665a6d9a06e6e0e27423b1a29533b8faab36da43fd8de`.
No threshold, H3 transform, holdout id, or frozen artifact was edited.

## QF-1 Discovery Stability

Discovery prompts were split by sorted `prompt_id` round-robin into D1–D4.
The official P holdout was not read. The Q validation split was not used to choose a candidate.
Partition design is regime-confounded: `True`.
D3 and D4 are single instruction prompts. D1 and D2 contain completion and syntax only.
That imbalance is a property of the frozen ids. It was not rebalanced after measurement.

partition | candidate | effect | sign | support | contradiction | status
---|---|---|---|---|---|---
D1 | null | null | null | 0 | 0 | NOT_IDENTIFIABLE
D2 | L0H0->L0H1 | 0.0180683 | 1 | 1 | 1 | IDENTIFIED
D3 | L0H0->L0H1 | 0.0258598 | 1 | 1 | 1 | IDENTIFIED
D4 | L0H0->L0H1 | 0.0499239 | 1 | 0 | 2 | IDENTIFIED

Tracked edge `L0H0 -> L0H1` on the same partitions:

partition | effect | member L0H0 | member L0H1 | sign | support | contradiction | selected
---|---|---|---|---|---|---|---
D1 | -0.00446129 | -0.00182724 | -0.0262413 | -1 | 0 | 2 | False
D2 | 0.0180683 | 0.0551386 | -0.117063 | 1 | 1 | 1 | True
D3 | 0.0258598 | -0.00260925 | 0.0387125 | 1 | 1 | 1 | True
D4 | 0.0499239 | -0.147832 | -0.0507402 | 1 | 0 | 2 | True

```text
DISCOVERY_STABILITY:
PARTIALLY_STABLE
```

This label is diagnostic. It is not a new H1 result.

## QF-2 Intervention Signal

Real interventions are the eight frozen Q catalog slots at alpha 1.
The null is the same slot at alpha 0, already present on the M22.1 magnitude grid.
No new direction was sampled.
Floor = 0.0001 by `max(1e-4, 10 * max abs alpha=0 drop), from configs/m22_1_preflight.json`.
Measurement split: discovery prompts only.

intervention | raw effect | snr | sign consistency | separable | regime conflict
---|---|---|---|---|---
L0H0 | -0.00730308 | 0.0749872 | 0.5 | True | True
L0H1 | -0.0497726 | 0.822907 | 0.833333 | True | False
L0H2 | -0.00428947 | 0.0703487 | 0.333333 | True | True
L0H3 | 0.0162985 | 0.210474 | 0.5 | True | True
L1H0 | 0.0100431 | 0.179424 | 0.5 | True | True
L1H1 | -0.0535911 | 2.02157 | 1 | True | False
L1H2 | -0.0293859 | 10.7102 | 1 | True | False
L1H3 | -0.108719 | 10.8399 | 1 | True | False

```text
SIGNAL_CONTEXT_DEPENDENT
```

This classification is diagnostic. Separable means larger than the M22.1 null floor, not large enough to pass H2.

## QF-3 Representation Sufficiency

Status: `INCONCLUSIVE`.

R2 is the same intervention-effect table already stored by SelfModelCore. R3 activation trajectories are not in the Q observation pipeline. R4 is not formed, because R3 is absent. No synthetic features were added.

R1 leave-one-discovery-prompt-out:

- Self-Model MAE: 0.0745591
- B2 linear probe MAE: 0.0745801
- Self-Model sign accuracy: 0.615741
- Feature dimensionality: 36
- Discovery prompts: 6
- Train/test: leave-one-discovery-prompt-out; validation not used

Saved validation means, marked diagnostic_reuse, are the frozen H2 numbers:

- Self-Model MAE: 0.0274578
- B2 MAE: 0.0282811

R2 status: `SAME_AS_R1`. R3 status: `NOT_AVAILABLE`. R4 status: `NOT_AVAILABLE`.

## QF-4 Oracle Prediction

Status: `ORACLE_INVALID`.
Interpretation: `INCONCLUSIVE`.

SelfModelCore.predict returns the stored margin-drop and does not read mechanism.abstraction. P_SPEC defines the planted P edge as an attention restriction on L0H0 and L1H1, not a numeric map from the Q slot label L0H0->L0H1 onto logit-margin drops. No repository function supplies that map. An oracle predictor was not invented, and holdout outcomes were not used to build one.

```text
O0 MAE = 0.0274578
O1 MAE = null
O2 MAE = null
B2 MAE = 0.0282811
```

O0 and B2 are the frozen validation scores, read back from `q_run_001`. They are not an oracle fit.

## QF-5 H2 Decomposition

Status: `global`.
Source: saved `q_run_001` intervention means. Diagnostic reuse. The validation split was not executed again.
Regime: `NOT_AVAILABLE` — q_run_001 stores validation means, not per-regime residuals
In-distribution vs OOD: `NOT_AVAILABLE` — the frozen Q split has no OOD bucket

bucket | n | self MAE | B2 MAE | relative reduction | sign accuracy
---|---|---|---|---|---
type:single | 8 | 0.020733 | 0.020733 | 0 | 0.5
type:pair | 28 | 0.0293791 | 0.0304377 | 0.0347794 | 0.785714
sign:positive | 5 | 0.0442396 | 0.042908 | -0.0310327 | 0.2
sign:negative | 31 | 0.024751 | 0.025922 | 0.0451713 | 0.806452
magnitude:above_1e-4 | 36 | 0.0274578 | 0.0282811 | 0.0291134 | 0.722222
target:L0H0 | 8 | 0.0315123 | 0.0354726 | 0.111643 | 0.75
target:L0H1 | 8 | 0.0447558 | 0.0470329 | 0.0484131 | 0.625
target:L0H2 | 8 | 0.0291283 | 0.0286111 | -0.0180769 | 0.625
target:L0H3 | 8 | 0.0346924 | 0.0358011 | 0.0309676 | 0.5
target:L1H0 | 8 | 0.023442 | 0.023551 | 0.00462939 | 0.5
target:L1H1 | 8 | 0.0206432 | 0.0213399 | 0.0326464 | 1
target:L1H2 | 8 | 0.0208752 | 0.0208525 | -0.00109 | 1
target:L1H3 | 8 | 0.0213376 | 0.0211362 | -0.00953123 | 1

The global/local label uses disjoint buckets with n >= 4: intervention type, actual sign, and the 1e-4 magnitude split.
Overlapping target-head rows are descriptive and are not the decision.

## QF-6 Baseline Ceiling

Status: `NEAR_CEILING`.

- B2 feature dimensionality: 8
- Self-Model feature dimensionality: 36
- Discovery prompts used to estimate effects: 6
- Train split: m22 discovery role, 6 prompts
- Frozen test split: m22 validation role, diagnostic_reuse, 6 prompts aggregated to 36 intervention means
- B2 sees information unavailable to the Self-Model: `False`
- Frozen self MAE: 0.0274578
- Frozen B2 MAE: 0.0282811
- Relative reduction: 0.0291134
- B2 MAE on discovery complements: min 0.022421, max 0.0883548

On the saved validation means, B2 is within the frozen 20% band of the self-model. That is the preregistered additive probe, not a new fit.
B2 was not weakened and was not removed.

## Failure Localization Matrix

| Layer | Evidence | Status |
| --- | --- | --- |
| Discovery | QF-1 | PARTIALLY_STABLE |
| Intervention | QF-2 | SIGNAL_CONTEXT_DEPENDENT |
| Representation | QF-3 | INCONCLUSIVE |
| Prediction | QF-4 | ORACLE_INVALID |
| H2 localization | QF-5 | global |
| Baseline ceiling | QF-6 | NEAR_CEILING |

## Primary Bottleneck

INTERVENTION_BOTTLENECK

## Evidence Supporting Diagnosis

- QF-1 label is `PARTIALLY_STABLE`.
- QF-2 label is `SIGNAL_CONTEXT_DEPENDENT` under the precommitted M22.1 null floor.
- QF-3 is `INCONCLUSIVE` because R3 and R4 were not available and R2 is not a new feature family.
- QF-4 did not invent a numeric oracle. `predict()` does not read the mechanism label, so the H1 candidate is not an input to H2.
- QF-5 localization is `global` on the saved validation means.
- QF-6 is `NEAR_CEILING`: the frozen self MAE does not clear the 20% bar against B2, and B2 uses a subset of the self-model's numbers.

## Evidence Against Diagnosis

- P.H1–H4 remain PASS. This audit does not reopen that result.
- Q.H3 remains PASS and is the frozen inverse readout, not evidence for a Qwen circuit.
- D1–D4 are regime-confounded by the frozen prompt ids, so a discovery label from those bins is not a clean estimate of sampler noise.
- The signal floor is the M22.1 numerical floor. Clearing it does not mean the effect is large enough for H2.
- Six discovery prompts and six validation prompts bound every comparison. Regime means use two prompts.
- QF-5 cannot see regime or OOD structure, because those fields were not stored in `q_run_001`.

## What This Does NOT Prove

This phase does not claim that MRSM succeeds on Qwen.
It does not convert Q.H3 PASS into a recovered Qwen mechanism.
It does not convert Q.H1 FAIL into a proof that mechanism discovery is impossible.
It does not authorize a change to thresholds, the holdout, the pinned revision, or the frozen scores.
A diagnostic label is not an MRSM gate result.

## Single Recommended Next Intervention

Primary bottleneck: INTERVENTION_BOTTLENECK

Next intervention:
Do not modify the Self-Model or the MRSM contract. The next intervention is a new preregistered decision on whether these Q residual slots produce a separable causal signal. Do not rerun the frozen MRSM gates to chase a pass.

No repair and no architectural redesign was executed in this phase.

## Reproducibility

- git_commit: `aba115119f81c1737e5a964458b701ddd82be5eb`
- git_branch: `cursor/qf-q-failure-localization-4e19`
- python: `3.12.3`
- torch: `2.14.0+cpu`
- transformer_lens: `3.9.0`
- numpy: `2.4.4`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- device: `cpu`
- weight_sha256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`
- prereg_sha256: `1c83d94ba3dbc27e101f1f4c14ca4a404e6b529793814dcaf1f7062ce4483471`
- final_result_sha256: `b44940da035fb42d71eb621f495179c066150b64c03d17a78b9371db73e48c20`

## Scientific Status

DIAGNOSTIC_ONLY
