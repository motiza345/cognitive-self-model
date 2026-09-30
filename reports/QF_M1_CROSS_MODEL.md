# QF-M1 Cross-Model Identity / Signal Diagnostic

QF-CROSS-MODEL-DIAGNOSTIC. Not an MRSM gate and not a mechanism discovery.
M2 and M3 were not executed.

## 1. Frozen State

| Item | State |
| --- | --- |
| P | H1 PASS, H2 PASS, H3 PASS, H4 PASS, Leakage PASS |
| Q | H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL, Leakage PASS |
| DEC-010 | `REDEFINE_SCALE` |
| p_freeze | `2cccafd047044332828eaf602b69f4267852cba2` |
| Qwen revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Qwen regime decision | `MIXED_SIGNAL` |
| MRSM rerun | not performed |

## 2. Model B Selection

One model was selected. Candidates were not ranked.

- Model: `gpt2`
- Revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Architecture: `GPT2LMHeadModel`
- Parameter count: `163049041`
- Parameter count definition: Sum of parameters in the loaded HookedTransformer. The source checkpoint has 12 layers, d_model 768, and 12 heads. Loading unties the token embedding, so this count is larger than the tied checkpoint.
- dtype: `float32`
- device: `cpu`
- Tokenizer: `GPT2Tokenizer:gpt2`
- Compatibility: `IDENTITY_FROZEN`
- Reason: gpt2 is an open-weight GPT-2 checkpoint with a pinned Hugging Face revision, a tokenizer, and TransformerLens residual hooks. Its family is GPT-2, not Qwen2. The checkpoint is small enough for repeated residual interventions. It is not another Qwen checkpoint. Candidates were not ranked.

## 3. Identity Freeze

The loader requested this revision and `local_files_only`. No other revision was substituted.

## 4. Task Mapping

No prompt text was changed. Batching uses left padding, the same call setting as the Qwen measurement.
Regimes kept: completion, instruction, syntax. None were dropped.
Primary split: `discovery`, the same split as the frozen Qwen matrix.

## 5. Intervention Mapping

Declared before any Model B effect was measured.
Slot names are residual-stream labels from the frozen catalog, not attention-head indices.
Alpha conversion: `NONE`. Alpha stays `1.0`.
Direction seeds: primary `22101`, orthogonal `22103`, width `768`.

| Slot | Qwen layer | Functional layer | Coordinate status |
| --- | ---: | --- | --- |
| L0H0 | 0 | 0 | comparable |
| L0H1 | 0 | 0 | comparable |
| L0H2 | 8 | 4 | comparable |
| L0H3 | 8 | 4 | comparable |
| L1H0 | 15 | 7 | NOT_COMPARABLE |
| L1H1 | 15 | 7 | NOT_COMPARABLE |
| L1H2 | 23 | 11 | NOT_COMPARABLE |
| L1H3 | 23 | 11 | NOT_COMPARABLE |

## 6. Functional-Counterpart Results

Primary comparison. Statistics use the QF-Regime definitions.

| Slot | Regime | n | Mean | Median | Std | SE | Effect minus null | Sign | Consistency | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| L0H0 | completion | 2 | 0.00227427 | 0.00227427 | 0.00598407 | 0.00423138 | 0.00227427 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H0 | instruction | 2 | 0.0125139 | 0.0125139 | 0.0142629 | 0.0100854 | 0.0125139 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H0 | syntax | 2 | -0.00217938 | -0.00217938 | 0.00631118 | 0.00446268 | -0.00217938 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H1 | completion | 2 | 0.0042758 | 0.0042758 | 0.0086627 | 0.00612545 | 0.0042758 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H1 | instruction | 2 | -0.00295019 | -0.00295019 | 0.00483894 | 0.00342165 | -0.00295019 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H1 | syntax | 2 | 0.00182533 | 0.00182533 | 0.00372219 | 0.00263199 | 0.00182533 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H2 | completion | 2 | 5.96046e-06 | 5.96046e-06 | 0.0114419 | 0.00809068 | 5.96046e-06 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_NOT_SEPARATED_FROM_NULL |
| L0H2 | instruction | 2 | 0.0155349 | 0.0155349 | 7.62939e-05 | 5.3948e-05 | 0.0155349 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H2 | syntax | 2 | -0.00721765 | -0.00721765 | 0.0123012 | 0.00869827 | -0.00721765 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H3 | completion | 2 | 0.00337076 | 0.00337076 | 0.0166602 | 0.0117805 | 0.00337076 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L0H3 | instruction | 2 | -0.0127389 | -0.0127389 | 0.00644994 | 0.00456079 | -0.0127389 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L0H3 | syntax | 2 | -7.41482e-05 | -7.41482e-05 | 0.00915504 | 0.00647359 | -7.41482e-05 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_NOT_SEPARATED_FROM_NULL |
| L1H0 | completion | 2 | 0.0152352 | 0.0152352 | 0.013149 | 0.00929776 | 0.0152352 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H0 | instruction | 2 | 0.00165081 | 0.00165081 | 0.00123549 | 0.00087362 | 0.00165081 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H0 | syntax | 2 | 0.0128024 | 0.0128024 | 0.0108664 | 0.00768371 | 0.0128024 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | completion | 2 | -0.00355268 | -0.00355268 | 0.00320554 | 0.00226666 | -0.00355268 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | instruction | 2 | -0.00868082 | -0.00868082 | 0.00224686 | 0.00158877 | -0.00868082 | NEGATIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H1 | syntax | 2 | 0.00217462 | 0.00217462 | 0.00227761 | 0.00161052 | 0.00217462 | UNSTABLE | 0.5 | CAUSAL_SIGNAL_WEAK |
| L1H2 | completion | 2 | 0.00120616 | 0.00120616 | 0.000127554 | 9.01943e-05 | 0.00120616 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H2 | instruction | 2 | 0.00178576 | 0.00178576 | 3.19481e-05 | 2.25907e-05 | 0.00178576 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H2 | syntax | 2 | 0.00142097 | 0.00142097 | 4.48227e-05 | 3.16944e-05 | 0.00142097 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | completion | 2 | 0.0056994 | 0.0056994 | 0.000617743 | 0.00043681 | 0.0056994 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | instruction | 2 | 0.00548029 | 0.00548029 | 2.90871e-05 | 2.05677e-05 | 0.00548029 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |
| L1H3 | syntax | 2 | 0.0058701 | 0.0058701 | 0.000551462 | 0.000389943 | 0.0058701 | POSITIVE | 1 | CAUSAL_SIGNAL_SUPPORTED |

## 7. Regime-Specific Nulls

Alpha 0 inside the same slot and the same regime. Not one global null.

| Slot | Regime | Null mean | Null std |
| --- | --- | ---: | ---: |
| L0H0 | completion | 0 | 0 |
| L0H0 | instruction | 0 | 0 |
| L0H0 | syntax | 0 | 0 |
| L0H1 | completion | 0 | 0 |
| L0H1 | instruction | 0 | 0 |
| L0H1 | syntax | 0 | 0 |
| L0H2 | completion | 0 | 0 |
| L0H2 | instruction | 0 | 0 |
| L0H2 | syntax | 0 | 0 |
| L0H3 | completion | 0 | 0 |
| L0H3 | instruction | 0 | 0 |
| L0H3 | syntax | 0 | 0 |
| L1H0 | completion | 0 | 0 |
| L1H0 | instruction | 0 | 0 |
| L1H0 | syntax | 0 | 0 |
| L1H1 | completion | 0 | 0 |
| L1H1 | instruction | 0 | 0 |
| L1H1 | syntax | 0 | 0 |
| L1H2 | completion | 0 | 0 |
| L1H2 | instruction | 0 | 0 |
| L1H2 | syntax | 0 | 0 |
| L1H3 | completion | 0 | 0 |
| L1H3 | instruction | 0 | 0 |
| L1H3 | syntax | 0 | 0 |

## 8. Effect Matrices

Qwen matrix is the frozen discovery matrix. It was not recomputed.
A symbol is the sign of the regime mean. UNSTABLE in the results table means the two discovery prompts disagree; the matrix still shows the mean sign.

| Slot | Qwen completion | Qwen instruction | Qwen syntax | B completion | B instruction | B syntax |
| --- | --- | --- | --- | --- | --- | --- |
| L0H0 | + | - | - | + | + | - |
| L0H1 | - | - | - | + | - | + |
| L0H2 | + | - | + | + | + | - |
| L0H3 | + | + | - | + | - | - |
| L1H0 | + | - | - | + | + | + |
| L1H1 | - | - | - | - | - | + |
| L1H2 | - | - | - | + | + | + |
| L1H3 | - | - | - | + | + | + |

Coordinate comparison, same Qwen layer index where the index exists:

| Slot | completion | instruction | syntax |
| --- | --- | --- | --- |
| L0H0 | + | + | - |
| L0H1 | + | - | + |
| L0H2 | + | + | + |
| L0H3 | + | - | - |
| L1H0 | NOT_COMPARABLE | NOT_COMPARABLE | NOT_COMPARABLE |
| L1H1 | NOT_COMPARABLE | NOT_COMPARABLE | NOT_COMPARABLE |
| L1H2 | NOT_COMPARABLE | NOT_COMPARABLE | NOT_COMPARABLE |
| L1H3 | NOT_COMPARABLE | NOT_COMPARABLE | NOT_COMPARABLE |

## 9. Decision

`MODEL_B_SIGNAL_PRESENT_BUT_DIFFERENT`

Model B has regime cells separated from their own nulls, but the pattern is not the frozen Qwen pattern. Differences: L0H0 instruction: Qwen -, Model B +; L0H1 completion: Qwen -, Model B +; L0H1 syntax: Qwen -, Model B +; L0H2 instruction: Qwen -, Model B +; L0H2 syntax: Qwen +, Model B -; L0H3 instruction: Qwen +, Model B -; L1H0 instruction: Qwen -, Model B +; L1H0 syntax: Qwen -, Model B +; L1H1 syntax: Qwen -, Model B +; L1H2 completion: Qwen -, Model B +; L1H2 instruction: Qwen -, Model B +; L1H2 syntax: Qwen -, Model B +; L1H3 completion: Qwen -, Model B +; L1H3 instruction: Qwen -, Model B +; L1H3 syntax: Qwen -, Model B +.

## 10. Observed, Inferred, Not Established

### Observed

- Model B identity, the unchanged prompts, the regime-specific effects, the regime-specific nulls, and the two matrices above.
- The Qwen matrix remains the frozen discovery matrix.

### Inferred

A regime-level signal separates from Model B's own nulls, and the sign pattern differs from Qwen. The Qwen profile may be model-dependent. This is not a claim that Model B is better.

Recorded associations, not an isolated cause:

- Architecture identity differs: GPT-2 versus Qwen2.
- Task strings do not differ.
- Intervention type and alpha do not differ. Alpha was not tuned.
- Direction width differs because the residual widths differ. The numeric Qwen vector was not reused.
- No experiment here changes only one of those factors, so a cell difference is not attributed to one of them.

### Not Established

- mechanism identity
- that one model is better
- that MRSM is wrong
- a cross-model regime pattern from M2
- a model-independent causal signature from M3

## 11. Stop

M1 is complete. M2 was not started. M3 was not started.
The Self-Model was not modified. MRSM was not rerun.
