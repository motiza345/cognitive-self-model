# CCSO — Contextual Causal State Observation

Diagnostic / Pre-Self-Model. Not an MRSM gate and not a mechanism claim.

## 1. Question

Does a richer observation supply counterfactual information about this frozen Qwen's
intervention drop that is not available from the snapshot or the local gradient?

- Decision: `INCONCLUSIVE`.
- The preregistered comparisons do not meet one decision rule. This result is local to the frozen Qwen revision and the frozen prompts. It does not say a richer observation cannot exist. No mechanism was identified. A Self-Model was not built. Pearson correlation, the alpha = ±0.25 probe, and the regime-label permutation are not decision inputs.

## 2. What was fixed before measurement

- Model `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. No GPT-2 and no fine-tuning.
- Prompts are the frozen discovery, validation, and replication split. Held-out prompts are not fit.
- Layers `0, 8, 15, 23`. Directions seeds `23101`–`23108`; `23107` and `23108` are unseen.
- Primary alphas are ±0.01, ±0.05, and ±0.10. ±0.25 is a nonlinear probe and is not a decision input.
- ±1 is outside the primary endpoint.
- O1 is `(h, g)`. The predicted drop is not a feature. O2 is a depth profile, not a trajectory.
- B2, B3, C1, and C2 share one readout width. B0 is the global mean and B1 is the regime mean.

## 3. Linear readouts

A win requires a strict MAE decrease and a strict sign-agreement increase. Pearson is reported only.

| Readout | Slice | MAE | Sign agreement | Pearson |
| --- | --- | ---: | ---: | ---: |
| B0 | S_val | 0.00309961 | 0.503472 | null |
| B0 | S_rep | 0.0027755 | 0.496528 | null |
| B0 | S_novel | 0.0033371 | 0.505226 | null |
| B0 | S_regime | 0.00293762 | 0.5 | null |
| B1 | S_val | 0.00309972 | 0.503472 | 0.00357829 |
| B1 | S_rep | 0.00277558 | 0.503472 | 0.000689879 |
| B1 | S_novel | 0.00333742 | 0.501742 | 0.00477147 |
| B1 | S_regime | 0.00293762 | 0.5 | null |
| B2 | S_val | 0.00306579 | 0.550926 | 0.11809 |
| B2 | S_rep | 0.00273245 | 0.548611 | 0.146005 |
| B2 | S_novel | 0.00343345 | 0.442509 | 0.107278 |
| B2 | S_regime | 0.00290476 | 0.571759 | null |
| B3 | S_val | 0.000101085 | 0.978009 | 0.99971 |
| B3 | S_rep | 0.000109582 | 0.979167 | 0.999341 |
| B3 | S_novel | 7.25962e-05 | 0.996516 | 0.999825 |
| B3 | S_regime | 9.7261e-05 | 0.979745 | null |
| C1 | S_val | 9.29072e-05 | 0.96412 | 0.99977 |
| C1 | S_rep | 9.38112e-05 | 0.972222 | 0.999472 |
| C1 | S_novel | 7.1991e-05 | 0.989547 | 0.999806 |
| C1 | S_regime | 0.000136698 | 0.954861 | null |
| C2 | S_val | 5.94532e-05 | 0.979167 | 0.999848 |
| C2 | S_rep | 8.22539e-05 | 0.97338 | 0.999543 |
| C2 | S_novel | 6.60185e-05 | 0.97561 | 0.999775 |
| C2 | S_regime | 7.84021e-05 | 0.976273 | null |

## 4. Small MLP

| Readout | Slice | MAE | Sign agreement |
| --- | --- | ---: | ---: |
| B2 | S_val | 0.0192648 | 0.481481 |
| B2 | S_rep | 0.0147517 | 0.490741 |
| B2 | S_novel | 0.00515901 | 0.550523 |
| B2 | S_regime | 0.0220764 | 0.495949 |
| B3 | S_val | 0.0183093 | 0.56713 |
| B3 | S_rep | 0.0200285 | 0.59375 |
| B3 | S_novel | 0.00376142 | 0.686411 |
| B3 | S_regime | 0.0255726 | 0.53588 |
| C1 | S_val | 0.0139217 | 0.597222 |
| C1 | S_rep | 0.0234223 | 0.556713 |
| C1 | S_novel | 0.00496378 | 0.651568 |
| C1 | S_regime | 0.0195455 | 0.558449 |
| C2 | S_val | 0.0304011 | 0.505787 |
| C2 | S_rep | 0.0194897 | 0.574074 |
| C2 | S_novel | 0.00322224 | 0.679443 |
| C2 | S_regime | 0.0413926 | 0.515046 |

## 5. Capacity, nulls, and the nonlinear probe

- Large MLP snapshot validation MAE `0.00314819`, replication MAE `0.00282722`.
- Null MAE on validation, linear family. A positive claim is blocked when a null MAE is at most the real MAE.

| Observation | N1 shuffled target | N2 shuffled direction | N4 shifted state | Real S_val MAE |
| --- | ---: | ---: | ---: | ---: |
| B2 | 0.00395034 | 0.00306691 | 0.00307303 | 0.00306579 |
| B3 | 0.0038411 | 0.00169209 | 0.0031069 | 0.000101085 |
| C1 | 0.00359515 | 0.00170608 | 0.0031116 | 9.29072e-05 |
| C2 | 0.00343629 | 0.00168649 | 0.00310809 | 5.94532e-05 |

- Regime-label permutation B1 validation MAE `0.0030998`. Not a decision input.

| Observation | Nonlinear-probe validation MAE | Sign agreement |
| --- | ---: | ---: |
| B2 | 0.0143558 | 0.545139 |
| B3 | 0.000565604 | 0.989583 |
| C1 | 0.000575359 | 0.979167 |
| C2 | 0.000513249 | 0.989583 |

## 6. Same state, different direction

Mean within-prompt Pearson at alpha = +0.01 on validation, across the six training directions.

| Observation | Mean Pearson | Defined groups | Groups |
| --- | ---: | ---: | ---: |
| C1 | 0.999901 | 24 | 24 |
| C2 | 0.999466 | 24 | 24 |

## 7. What this does not say

No Self-Model was built. Qwen was not modified. MRSM was not rerun.
No mechanism was identified. A trajectory encoder was not built.
The result is local to this frozen revision and these frozen prompts.
It does not say a richer observation cannot exist.

## 8. Which strict comparisons fired

A cell is true only when MAE falls and sign agreement rises. This trace does not change the decision.

| Comparison | Validation | Replication | Unseen direction | Held-out regime |
| --- | --- | --- | --- | --- |
| B3 over B2 | True | True | True | True |
| C1 over B2 | True | True | True | True |
| C1 over B3 | False | False | False | False |
| C2 over B2 | True | True | True | True |
| C2 over B3 | True | False | False | False |
| C2 over C1 | True | True | False | True |
| B2 over B0 | True | True | False | True |
| B2 over B1 | True | True | False | True |

The table uses the same `beats` rule as the decision. It does not reopen the label.
