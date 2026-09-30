# QF-M2 Cross-Model Regime Comparison

QF-CROSS-MODEL-DIAGNOSTIC. Not an MRSM gate and not a mechanism claim.
M3 was not executed.

## 1. Executive Summary

- Decision: `PARTIAL_CROSS_MODEL_CONSISTENCY`.
- Level A literal slots: `NOT_SUPPORTED`.
- Level B regime profiles: `NOT_SUPPORTED`.
- Level C regime relationships: `PARTIAL`.
- Interpretation: Some higher-level regime structure appears transferable, but the evidence is insufficient for a general model-independent causal representation.
- M3: Do not run M3. Partial transfer is not sufficient to test a general model-independent representation.

## 2. Frozen Inputs

- Qwen revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Functional counterpart layers `0, 4, 7, 11`, declared in M1 before measurement.
- Qwen coordinate layers 15 and 23 remain `NOT_COMPARABLE`.
- No new forward pass was run. Effects come from the stored QF-Regime and M1 artifacts.
- MRSM was not rerun. The Self-Model was not modified.

## 3. Qwen vs GPT-2 Comparability

The comparison uses the eight functional-counterpart slots. Those slots are comparable.
The coordinate reading of layers 15 and 23 is preserved as `NOT_COMPARABLE` and is not imputed.
Slot names are residual-stream labels, not attention heads.

## 4. Raw Effect Profiles

R0 is the regime mean margin drop. The profile for a regime is the eight slot values in catalog order.

### R0

| Slot | Qwen completion | Qwen instruction | Qwen syntax | GPT-2 completion | GPT-2 instruction | GPT-2 syntax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0H0 | 0.0963473 | -0.0752206 | -0.043036 | 0.00227427 | 0.0125139 | -0.00217938 |
| L0H1 | -0.0637145 | -0.00601387 | -0.0795894 | 0.0042758 | -0.00295019 | 0.00182533 |
| L0H2 | 0.0429649 | -0.069767 | 0.0139337 | 5.96046e-06 | 0.0155349 | -0.00721765 |
| L0H3 | 0.0517416 | 0.04526 | -0.048106 | 0.00337076 | -0.0127389 | -7.41482e-05 |
| L1H0 | 0.0727177 | -0.00567102 | -0.0369174 | 0.0152352 | 0.00165081 | 0.0128024 |
| L1H1 | -0.0316253 | -0.064342 | -0.064806 | -0.00355268 | -0.00868082 | 0.00217462 |
| L1H2 | -0.0299549 | -0.0298491 | -0.0283537 | 0.00120616 | 0.00178576 | 0.00142097 |
| L1H3 | -0.10939 | -0.105217 | -0.11155 | 0.0056994 | 0.00548029 | 0.0058701 |

## 5. Null-Corrected Profiles

R1 is `mean_effect - regime_null_mean`. This is the primary profile.

### R1

| Slot | Qwen completion | Qwen instruction | Qwen syntax | GPT-2 completion | GPT-2 instruction | GPT-2 syntax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0H0 | 0.0963473 | -0.0752206 | -0.043036 | 0.00227427 | 0.0125139 | -0.00217938 |
| L0H1 | -0.0637145 | -0.00601387 | -0.0795894 | 0.0042758 | -0.00295019 | 0.00182533 |
| L0H2 | 0.0429649 | -0.069767 | 0.0139337 | 5.96046e-06 | 0.0155349 | -0.00721765 |
| L0H3 | 0.0517416 | 0.04526 | -0.048106 | 0.00337076 | -0.0127389 | -7.41482e-05 |
| L1H0 | 0.0727177 | -0.00567102 | -0.0369174 | 0.0152352 | 0.00165081 | 0.0128024 |
| L1H1 | -0.0316253 | -0.064342 | -0.064806 | -0.00355268 | -0.00868082 | 0.00217462 |
| L1H2 | -0.0299549 | -0.0298491 | -0.0283537 | 0.00120616 | 0.00178576 | 0.00142097 |
| L1H3 | -0.10939 | -0.105217 | -0.11155 | 0.0056994 | 0.00548029 | 0.0058701 |

## 6. Standardized Profiles

The M2 formula is `(effect - null_mean) / null_std`.
Status: `UNDEFINED_BECAUSE_NULL_STD_IS_ZERO`.
No substitute denominator was inserted. The existing QF-Regime standardized effect, which divides by the effect standard deviation, is stored in the artifact and is not a decision input.

## 7. Regime-to-Regime Relationships

Primary metric, declared before the comparison: `cosine_similarity_of_r1_null_corrected_slot_vectors`.
Pearson correlation is exploratory and was not used to choose the decision.

| Pair | Qwen cosine | GPT-2 cosine | Qwen Pearson (exploratory) | GPT-2 Pearson (exploratory) |
| --- | ---: | ---: | ---: | ---: |
| completion__instruction | 0.199466 | 0.13508 | 0.307878 | 0.0467377 |
| completion__syntax | 0.346491 | 0.780373 | 0.683058 | 0.769216 |
| instruction__syntax | 0.601912 | -0.251588 | 0.161462 | -0.327263 |

## 8. Cross-Model Regime Comparison

| Regime | R1 cosine | R1 Pearson (exploratory) |
| --- | ---: | ---: |
| completion | 0.200201 | 0.206809 |
| instruction | -0.602629 | -0.65567 |
| syntax | -0.520831 | -0.480107 |

## 9. Sign Agreement

Sign agreement describes intervention behavior. It is not mechanism identity.

- Agreement: `9`.
- Disagreement: `15`.
- Near zero: `0`.
- Not comparable: `12`.

| Slot | completion | instruction | syntax |
| --- | --- | --- | --- |
| L0H0 | sign_agreement | sign_disagreement | sign_agreement |
| L0H1 | sign_disagreement | sign_agreement | sign_disagreement |
| L0H2 | sign_agreement | sign_disagreement | sign_disagreement |
| L0H3 | sign_agreement | sign_disagreement | sign_agreement |
| L1H0 | sign_agreement | sign_disagreement | sign_disagreement |
| L1H1 | sign_agreement | sign_agreement | sign_disagreement |
| L1H2 | sign_disagreement | sign_disagreement | sign_disagreement |
| L1H3 | sign_disagreement | sign_disagreement | sign_disagreement |

## 10. Heterogeneity

Same exploratory QF-Regime variance split. Not a gate and not a ranking.

| Slot | Qwen between | Qwen within | Qwen ratio | GPT-2 between | GPT-2 within | GPT-2 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L0H0 | 0.00554435 | 0.00394067 | 1.40695 | 3.78418e-05 | 9.30236e-05 | 0.406798 |
| L0H1 | 0.000999414 | 0.00265888 | 0.375877 | 9.00282e-06 | 3.74375e-05 | 0.240476 |
| L0H2 | 0.00228412 | 0.00143375 | 1.59311 | 9.01117e-05 | 9.40812e-05 | 0.957807 |
| L0H3 | 0.00208097 | 0.00391554 | 0.531465 | 4.79763e-05 | 0.000134326 | 0.357162 |
| L1H0 | 0.00212678 | 0.00100631 | 2.11344 | 3.4979e-05 | 9.75007e-05 | 0.358757 |
| L1H1 | 0.000241285 | 0.000461479 | 0.522851 | 1.966e-05 | 6.83712e-06 | 2.87548 |
| L1H2 | 5.34579e-07 | 6.9934e-06 | 0.0764405 | 5.72379e-08 | 6.43325e-09 | null |
| L1H3 | 6.90938e-06 | 9.36819e-05 | 0.0737536 | 2.5456e-08 | 2.28854e-07 | 0.111232 |

## 11. Context Cancellation

The two results stay separate. Neither model is ranked.

- `QWEN_CONTEXT_CANCELLATION`: slots ['L0H0', 'L0H2', 'L0H3'].
- `GPT2_CONTEXT_CANCELLATION`: slots ['L0H1', 'L0H2', 'L0H3'].

| Slot | Qwen | GPT-2 |
| --- | --- | --- |
| L0H0 | True | False |
| L0H1 | False | True |
| L0H2 | True | True |
| L0H3 | True | True |
| L1H0 | False | False |
| L1H1 | False | False |
| L1H2 | False | False |
| L1H3 | False | False |

## 12. Cross-Model Null

GPT-2 regime labels are permuted exhaustively. There are six permutations because there are three regimes.
The count was not tuned. No numeric significance cutoff was added.

- Identity mean cosine: `-0.307753`.
- Identity strictly best: `False`.

| GPT-2 order | Identity | Mean cosine |
| --- | --- | ---: |
| completion, instruction, syntax | True | -0.307753 |
| completion, syntax, instruction | False | 0.0674208 |
| instruction, completion, syntax | False | -0.157401 |
| instruction, syntax, completion | False | -0.126152 |
| syntax, completion, instruction | False | -0.0818039 |
| syntax, instruction, completion | False | -0.425728 |

## 13. Three-Level Consistency Analysis

- Level A, literal slot consistency: `NOT_SUPPORTED`.
- Level B, regime-profile consistency: `NOT_SUPPORTED`.
- Level C, higher-order context structure: `PARTIAL`.

A asks whether the same slot has the same sign. B asks whether matched regime profiles align above the regime-label permutation null. C asks whether the three within-model regime relationships have the same shape.

- Cosine of the two within-model relationship triples: `0.242984`.
- Strongest Qwen pair: `instruction__syntax`.
- Strongest GPT-2 pair: `completion__syntax`.

## 14. Decision

`PARTIAL_CROSS_MODEL_CONSISTENCY`

Some higher-level regime structure appears transferable, but the evidence is insufficient for a general model-independent causal representation.

## 15. What This Does Not Prove

- It does not prove mechanism identity.
- It does not prove that causal mechanisms are model-specific in general.
- It does not rank Qwen and GPT-2.
- It does not repair Q or change `REDEFINE_SCALE`.
- It does not make layers 15 and 23 comparable.

## 16. Implication for Q Failure Diagnosis

`INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED` remains the Qwen diagnosis.
Qwen context dependence remains a Qwen diagnosis. GPT-2 shows only a partial regime correspondence, so the Q failure is not given a shared regime representation.

## 17. Recommendation for M3

Do not run M3. Partial transfer is not sufficient to test a general model-independent representation.

## 18. Reproducibility Manifest

See `artifacts/qf_cross_model/m2/manifest.sha256`.
M1 artifacts were read and not overwritten.
