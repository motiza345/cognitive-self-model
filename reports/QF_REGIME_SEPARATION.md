# QF Regime Separation

Diagnostic characterization. Not an MRSM result and not a mechanism identification.

## 1. Executive Summary

- Regime decision: `MIXED_SIGNAL`.
- Previous diagnosis retained: `INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED`.
- Mechanism identity: `NOT_EVALUATED`.
- Primary split: `discovery`. Validation and replication are conditioning views only.
- Interpretation: Every measured discovery regime cell separates from that regime's own null, but the slots do not share one sign pattern. A single regime explanation is not supported. This does not identify a mechanism.

## 2. Frozen MRSM Status

| Item | State |
| --- | --- |
| P | H1 PASS, H2 PASS, H3 PASS, H4 PASS, Leakage PASS |
| Q | H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL, Leakage PASS |
| DEC-010 | `REDEFINE_SCALE` |
| p_freeze | `2cccafd047044332828eaf602b69f4267852cba2` |
| Qwen revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| MRSM rerun | not performed |

## 3. Frozen Regime Definitions

Regimes are the three ids in `src/cognitive_self_model/m22_1/prompts.py`.
None were dropped, merged, renamed, relabeled, reweighted, or downsampled.
Discovery, validation, and replication are splits, not regimes.

| Regime | Discovery prompts | Validation prompts | Replication prompts |
| --- | --- | --- | --- |
| completion | completion-01, completion-04 | completion-02, completion-05 | completion-03, completion-06 |
| instruction | instruction-01, instruction-04 | instruction-02, instruction-05 | instruction-03, instruction-06 |
| syntax | syntax-01, syntax-04 | syntax-02, syntax-05 | syntax-03, syntax-06 |

## 4. Experimental Contract

- Metric: MRSM margin drop, unhooked minus alpha-1 intervention.
- Candidates are the eight existing MRSM slots. A slot is not a mechanism.
- Each regime has its own alpha-0 null. There is no shared null.
- Separability reuses `effect_floor` on that regime's null sample.
- Sign zero floor is the existing `1e-8`. Unstable reuses the existing `0.75` consistency bar.
- The confidence interval reuses the frozen percentile bootstrap, 2000 draws, seed 22102.
- Pooled means are descriptive. They are not the primary result.
- Heterogeneity is labeled `EXPLORATORY` and is not a gate.
- The decision uses the discovery split only.

## 5. Regime-Specific Intervention Results

Discovery is primary.

| Slot | Regime | n | Mean | Median | Std | SE | Effect minus null | Sign | Consistency | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| L0H0 | completion | 2 | 0.0963473 | 0.0963473 | 0.0211511 | 0.0149561 | 0.0963473 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H0 | instruction | 2 | -0.0752206 | -0.0752206 | 0.0726113 | 0.051344 | -0.0752206 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H0 | syntax | 2 | -0.043036 | -0.043036 | 0.0781169 | 0.055237 | -0.043036 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H1 | completion | 2 | -0.0637145 | -0.0637145 | 0.0149808 | 0.010593 | -0.0637145 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H1 | instruction | 2 | -0.00601387 | -0.00601387 | 0.0447264 | 0.0316263 | -0.00601387 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H1 | syntax | 2 | -0.0795894 | -0.0795894 | 0.0758405 | 0.0536273 | -0.0795894 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H2 | completion | 2 | 0.0429649 | 0.0429649 | 0.0193977 | 0.0137163 | 0.0429649 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H2 | instruction | 2 | -0.069767 | -0.069767 | 0.0611982 | 0.0432737 | -0.069767 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H2 | syntax | 2 | 0.0139337 | 0.0139337 | 0.0134072 | 0.00948034 | 0.0139337 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H3 | completion | 2 | 0.0517416 | 0.0517416 | 0.0095911 | 0.00678193 | 0.0517416 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H3 | instruction | 2 | 0.04526 | 0.04526 | 0.107956 | 0.0763367 | 0.04526 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H3 | syntax | 2 | -0.048106 | -0.048106 | 0.000192404 | 0.00013605 | -0.048106 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H0 | completion | 2 | 0.0727177 | 0.0727177 | 0.00049305 | 0.000348639 | 0.0727177 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H0 | instruction | 2 | -0.00567102 | -0.00567102 | 0.048718 | 0.0344488 | -0.00567102 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L1H0 | syntax | 2 | -0.0369174 | -0.0369174 | 0.0254018 | 0.0179618 | -0.0369174 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | completion | 2 | -0.0316253 | -0.0316253 | 0.000327587 | 0.000231639 | -0.0316253 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | instruction | 2 | -0.064342 | -0.064342 | 0.0365186 | 0.0258225 | -0.064342 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | syntax | 2 | -0.064806 | -0.064806 | 0.00712204 | 0.00503604 | -0.064806 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H2 | completion | 2 | -0.0299549 | -0.0299549 | 0.0019989 | 0.00141344 | -0.0299549 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H2 | instruction | 2 | -0.0298491 | -0.0298491 | 0.00387096 | 0.00273718 | -0.0298491 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H2 | syntax | 2 | -0.0283537 | -0.0283537 | 0.0014143 | 0.00100006 | -0.0283537 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | completion | 2 | -0.10939 | -0.10939 | 0.0103681 | 0.00733136 | -0.10939 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | instruction | 2 | -0.105217 | -0.105217 | 0.0118265 | 0.00836261 | -0.105217 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | syntax | 2 | -0.11155 | -0.11155 | 0.00580359 | 0.00410375 | -0.11155 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |

## 6. Regime-Specific Null Results

Alpha is 0 inside the same slot and the same regime. The null is not global.

| Slot | Regime | Null mean | Null std | Effect floor |
| --- | --- | ---: | ---: | ---: |
| L0H0 | completion | 0 | 0 | 0.0001 |
| L0H0 | instruction | 0 | 0 | 0.0001 |
| L0H0 | syntax | 0 | 0 | 0.0001 |
| L0H1 | completion | 0 | 0 | 0.0001 |
| L0H1 | instruction | 0 | 0 | 0.0001 |
| L0H1 | syntax | 0 | 0 | 0.0001 |
| L0H2 | completion | 0 | 0 | 0.0001 |
| L0H2 | instruction | 0 | 0 | 0.0001 |
| L0H2 | syntax | 0 | 0 | 0.0001 |
| L0H3 | completion | 0 | 0 | 0.0001 |
| L0H3 | instruction | 0 | 0 | 0.0001 |
| L0H3 | syntax | 0 | 0 | 0.0001 |
| L1H0 | completion | 0 | 0 | 0.0001 |
| L1H0 | instruction | 0 | 0 | 0.0001 |
| L1H0 | syntax | 0 | 0 | 0.0001 |
| L1H1 | completion | 0 | 0 | 0.0001 |
| L1H1 | instruction | 0 | 0 | 0.0001 |
| L1H1 | syntax | 0 | 0 | 0.0001 |
| L1H2 | completion | 0 | 0 | 0.0001 |
| L1H2 | instruction | 0 | 0 | 0.0001 |
| L1H2 | syntax | 0 | 0 | 0.0001 |
| L1H3 | completion | 0 | 0 | 0.0001 |
| L1H3 | instruction | 0 | 0 | 0.0001 |
| L1H3 | syntax | 0 | 0 | 0.0001 |

## 7. Effect Consistency

Symbols are the sign of the measured regime mean. `+` positive, `-` negative, `~` inside the existing zero floor.

| Slot | completion | instruction | syntax |
| --- | --- | --- | --- |
| L0H0 | + | - | - |
| L0H1 | - | - | - |
| L0H2 | + | - | + |
| L0H3 | + | + | - |
| L1H0 | + | - | - |
| L1H1 | - | - | - |
| L1H2 | - | - | - |
| L1H3 | - | - | - |

## 8. Heterogeneity

Exploratory only. These ratios did not choose the decision.

| Slot | Between | Within | Ratio |
| --- | ---: | ---: | ---: |
| L0H0 | 0.00554435 | 0.00394067 | 1.40695 |
| L0H1 | 0.000999414 | 0.00265888 | 0.375877 |
| L0H2 | 0.00228412 | 0.00143375 | 1.59311 |
| L0H3 | 0.00208097 | 0.00391554 | 0.531465 |
| L1H0 | 0.00212678 | 0.00100631 | 2.11344 |
| L1H1 | 0.000241285 | 0.000461479 | 0.522851 |
| L1H2 | 5.34579e-07 | 6.9934e-06 | 0.0764405 |
| L1H3 | 6.90938e-06 | 9.36819e-05 | 0.0737536 |

## 9. Context Cancellation

Cancellation uses the existing QF-Bridge rule on discovery regime means.
A true flag means the effect depends on regime. It does not mean multiple mechanisms.

| Slot | Context cancellation |
| --- | --- |
| L0H0 | True |
| L0H1 | False |
| L0H2 | True |
| L0H3 | True |
| L1H0 | False |
| L1H1 | False |
| L1H2 | False |
| L1H3 | False |

## 10. Pooled vs Regime-Specific Comparison

| Representation | Role in this diagnostic |
| --- | --- |
| R0 pooled | Secondary descriptive mean within discovery. Not used to drop a regime. |
| R1 regime-specific | Primary. One estimate per slot and regime on discovery. |
| R2 regime-conditioned | Validation and replication are reported separately in the artifacts. They are not a Self-Model. |

| Slot | R0 pooled discovery mean |
| --- | ---: |
| L0H0 | -0.00730308 |
| L0H1 | -0.0497726 |
| L0H2 | -0.00428947 |
| L0H3 | 0.0162985 |
| L1H0 | 0.0100431 |
| L1H1 | -0.0535911 |
| L1H2 | -0.0293859 |
| L1H3 | -0.108719 |

## 11. Decision

`MIXED_SIGNAL`

Every measured discovery regime cell separates from that regime's own null, but the slots do not share one sign pattern. A single regime explanation is not supported. This does not identify a mechanism.

Signal existence and sign consistency were examined. Mechanism identity was not.

## 12. What This Does Not Prove

- It does not prove one mechanism.
- It does not prove several independent mechanisms.
- It does not repair Q.
- It does not change `REDEFINE_SCALE`.
- It does not select a best, worst, or representative regime.

## 13. Implication for Intervention Bottleneck

`INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED` stays in force.
This diagnostic refines it to `MIXED_SIGNAL`.
The refinement is about how the effect varies by regime, not about mechanism identity.

## 14. Single Next Intervention

Do not modify the Self-Model or rerun MRSM. Do not drop a regime. The next single step is a preregistered decision that keeps every frozen regime visible, because the intervention slots do not share one regime pattern.

## 15. Reproducibility Manifest

See `artifacts/qf_regime/manifest.sha256`.

| Frozen file | sha256 |
| --- | --- |
| configs/m22_1_preflight.json | `4c52fe878cd3582a3df873b43eab00b545623f98b25053a9c1c00801333716ca` |
| configs/mrsm_prereg.yaml | `1c83d94ba3dbc27e101f1f4c14ca4a404e6b529793814dcaf1f7062ce4483471` |
| reports/MRSM_FINAL_RESULT.md | `b44940da035fb42d71eb621f495179c066150b64c03d17a78b9371db73e48c20` |
| reports/QWEN_BRIDGE_INTERVENTION_ANALYSIS.md | `67ac03215259869fd74dcdc6735623b25a79ba8e3ae0d5e9dc2eae6f8d7dd195` |
| reports/Q_FAILURE_ANALYSIS.md | `d238f8d0e1803497f43e1e71f989c3ea2e2c90de7ca4ff1699bee8abae325423` |

Null artifact slots: 8.
