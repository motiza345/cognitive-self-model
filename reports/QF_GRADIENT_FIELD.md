# QF Gradient Field

QF-GRADIENT-FIELD. Not an MRSM gate and not M3.

## 1. Sign correction

Δ is the margin drop `m(h) - m(h + α v)`, the same drop as the regime measurements.
The expansion fixed before this run is

`Δ ≈ -α vᵀ g(x) - (α² / 2) vᵀ H_m(x) v`.

For α > 0 the first-order term has the opposite sign of `vᵀ g(x)`.
The regime-effect sign matrix is not a gradient.
α = ±1 is reported as the historical scale and is not an input to the linear classification.

## 2. Preregistered question

The same frozen prompts, functional layers, and residual site are used.
`g(x)` is the direct derivative of the yes/no margin at `blocks.{layer}.hook_resid_post` on the last token.
The linear prediction `-α vᵀ g(x)` is compared with measured Δ on the small grid -0.25, -0.10, -0.05, -0.01, 0.01, 0.05, 0.10, 0.25.
Validation and replication are held-out repeats. They are not pooled with discovery.
The low-dimensional field is `φ_s(x) = v_sᵀ g(x)` on the eight functional slots.
The map is the M1 functional slot alignment: Qwen layers 0, 8, 15, 23 and GPT-2 layers 0, 4, 7, 11.
Directions are the frozen seeds resampled in each model width. No rotation is learned.
GPT-2 has no layers 15 or 23; those coordinate cells stay `NOT_COMPARABLE` and are not imputed.
Predictability is the mean cosine of prompt-aligned fields against all 720 GPT-2 prompt permutations inside one split.

## 3. Decisions

- Qwen linear: `INCONCLUSIVE`.
- GPT-2 linear: `INCONCLUSIVE`.
- Field: `INCONCLUSIVE`.
- Qwen linear classification is INCONCLUSIVE. GPT-2 linear classification is INCONCLUSIVE. The models are reported separately and are not ranked. The eight-dimensional field of v^T g under functional_slot_alignment is INCONCLUSIVE. This experiment does not assume a context-independent or model-independent mechanism. A negative or partial result says this structure was not found for these models, these prompts, and this declared map. It does not say the structure cannot exist. M3 was not run. No mechanism was identified. The regime-effect sign matrix was not used as a gradient, and alpha = 1 was not part of the linear test.

These labels are the preregistered instrument result. A failed gradient check blocks the linear and field classifications. The check is not loosened after the run, and the blocked readings in section 8 do not replace the labels.

## 4. Instrument audit

A directional derivative must match the central difference at ε = 0.01, and an α = 0 hook must leave Δ at most 1e-4.
A failed check makes that comparison inconclusive. It is not reclassified.

| Model | Split | Gradient check | Alpha 0 | Gradient failures | Alpha-0 failures |
| --- | --- | --- | --- | ---: | ---: |
| Qwen | discovery | False | True | 15 | 0 |
| Qwen | validation | False | True | 19 | 0 |
| Qwen | replication | False | True | 9 | 0 |
| GPT-2 | discovery | False | True | 14 | 0 |
| GPT-2 | validation | False | True | 16 | 0 |
| GPT-2 | replication | False | True | 15 | 0 |

## 5. Linear sign comparison

| Model | Split | Alpha | In linear test | Agree | Disagree | Near zero | Median Δ/linear |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen | discovery | -0.01 | True | 48 | 0 | 0 | 0.999952 |
| Qwen | discovery | -0.05 | True | 48 | 0 | 0 | 0.999538 |
| Qwen | discovery | -0.10 | True | 48 | 0 | 0 | 0.999676 |
| Qwen | discovery | -0.25 | True | 48 | 0 | 0 | 0.997819 |
| Qwen | discovery | -1.00 | False | 47 | 1 | 0 | 0.990167 |
| Qwen | discovery | 0.01 | True | 48 | 0 | 0 | 0.997683 |
| Qwen | discovery | 0.05 | True | 48 | 0 | 0 | 0.999806 |
| Qwen | discovery | 0.10 | True | 48 | 0 | 0 | 0.999936 |
| Qwen | discovery | 0.25 | True | 48 | 0 | 0 | 1.00094 |
| Qwen | discovery | 1.00 | False | 47 | 1 | 0 | 1.00446 |
| Qwen | discovery | label |  | `INCONCLUSIVE` |  |  |  |
| Qwen | validation | -0.01 | True | 48 | 0 | 0 | 1.0009 |
| Qwen | validation | -0.05 | True | 48 | 0 | 0 | 0.999984 |
| Qwen | validation | -0.10 | True | 48 | 0 | 0 | 0.999599 |
| Qwen | validation | -0.25 | True | 47 | 1 | 0 | 0.999042 |
| Qwen | validation | -1.00 | False | 46 | 2 | 0 | 0.99725 |
| Qwen | validation | 0.01 | True | 48 | 0 | 0 | 0.999998 |
| Qwen | validation | 0.05 | True | 48 | 0 | 0 | 0.999981 |
| Qwen | validation | 0.10 | True | 48 | 0 | 0 | 1.00038 |
| Qwen | validation | 0.25 | True | 47 | 1 | 0 | 1.00055 |
| Qwen | validation | 1.00 | False | 43 | 5 | 0 | 1.00259 |
| Qwen | validation | label |  | `INCONCLUSIVE` |  |  |  |
| Qwen | replication | -0.01 | True | 48 | 0 | 0 | 0.999349 |
| Qwen | replication | -0.05 | True | 48 | 0 | 0 | 0.998335 |
| Qwen | replication | -0.10 | True | 48 | 0 | 0 | 0.998582 |
| Qwen | replication | -0.25 | True | 47 | 1 | 0 | 0.997767 |
| Qwen | replication | -1.00 | False | 45 | 3 | 0 | 0.992497 |
| Qwen | replication | 0.01 | True | 47 | 1 | 0 | 1.00216 |
| Qwen | replication | 0.05 | True | 47 | 1 | 0 | 1.00166 |
| Qwen | replication | 0.10 | True | 47 | 1 | 0 | 1.00167 |
| Qwen | replication | 0.25 | True | 47 | 1 | 0 | 1.00232 |
| Qwen | replication | 1.00 | False | 45 | 3 | 0 | 1.00579 |
| Qwen | replication | label |  | `INCONCLUSIVE` |  |  |  |
| GPT-2 | discovery | -0.01 | True | 48 | 0 | 0 | 1.00457 |
| GPT-2 | discovery | -0.05 | True | 48 | 0 | 0 | 0.998046 |
| GPT-2 | discovery | -0.10 | True | 48 | 0 | 0 | 0.999805 |
| GPT-2 | discovery | -0.25 | True | 48 | 0 | 0 | 0.999637 |
| GPT-2 | discovery | -1.00 | False | 48 | 0 | 0 | 1.00033 |
| GPT-2 | discovery | 0.01 | True | 46 | 2 | 0 | 0.993922 |
| GPT-2 | discovery | 0.05 | True | 47 | 1 | 0 | 1.00007 |
| GPT-2 | discovery | 0.10 | True | 48 | 0 | 0 | 1.00078 |
| GPT-2 | discovery | 0.25 | True | 48 | 0 | 0 | 0.999988 |
| GPT-2 | discovery | 1.00 | False | 48 | 0 | 0 | 0.999346 |
| GPT-2 | discovery | label |  | `INCONCLUSIVE` |  |  |  |
| GPT-2 | validation | -0.01 | True | 47 | 0 | 1 | 0.999637 |
| GPT-2 | validation | -0.05 | True | 48 | 0 | 0 | 0.999044 |
| GPT-2 | validation | -0.10 | True | 48 | 0 | 0 | 0.999386 |
| GPT-2 | validation | -0.25 | True | 48 | 0 | 0 | 0.999217 |
| GPT-2 | validation | -1.00 | False | 47 | 1 | 0 | 0.999225 |
| GPT-2 | validation | 0.01 | True | 48 | 0 | 0 | 1.0047 |
| GPT-2 | validation | 0.05 | True | 48 | 0 | 0 | 1.00135 |
| GPT-2 | validation | 0.10 | True | 48 | 0 | 0 | 1.00058 |
| GPT-2 | validation | 0.25 | True | 48 | 0 | 0 | 1.00145 |
| GPT-2 | validation | 1.00 | False | 48 | 0 | 0 | 1.00064 |
| GPT-2 | validation | label |  | `INCONCLUSIVE` |  |  |  |
| GPT-2 | replication | -0.01 | True | 48 | 0 | 0 | 1.00436 |
| GPT-2 | replication | -0.05 | True | 48 | 0 | 0 | 0.999971 |
| GPT-2 | replication | -0.10 | True | 48 | 0 | 0 | 1.00025 |
| GPT-2 | replication | -0.25 | True | 48 | 0 | 0 | 1 |
| GPT-2 | replication | -1.00 | False | 48 | 0 | 0 | 1.00125 |
| GPT-2 | replication | 0.01 | True | 48 | 0 | 0 | 0.997213 |
| GPT-2 | replication | 0.05 | True | 48 | 0 | 0 | 1.00054 |
| GPT-2 | replication | 0.10 | True | 48 | 0 | 0 | 0.999378 |
| GPT-2 | replication | 0.25 | True | 48 | 0 | 0 | 0.999771 |
| GPT-2 | replication | 1.00 | False | 48 | 0 | 0 | 0.998872 |
| GPT-2 | replication | label |  | `INCONCLUSIVE` |  |  |  |

## 6. Cross-model field

Each row is one split of six prompts. The null is exhaustive inside that split.
A within-regime permutation is not used, because a regime has two discovery prompts.

| Split | Status | Identity mean cosine | Permutations strictly better | Permutations tied with identity |
| --- | --- | ---: | ---: | ---: |
| discovery | `INCONCLUSIVE` | null | null | null |
| validation | `INCONCLUSIVE` | null | null | null |
| replication | `INCONCLUSIVE` | null | null | null |

## 7. What this does not say

This experiment does not assume a context-independent or model-independent mechanism.
A negative or partial result says the declared field was not predictable for these models, these prompts, and this map.
It does not say that no such structure exists.
DAS and the Curse of Multiple Mediators paper are related citations, not this protocol.
M3 was not run. MRSM was not rerun. The Self-Model was not modified.
No model is ranked.

## 8. Descriptive readings blocked by the instrument check

The gradient check compares `vᵀ g` with a central difference at ε = 0.01. It fails when the absolute discrepancy exceeds 1e-4 and the relative discrepancy exceeds 1e-2. Alpha 0 is a separate audit. Section 5 is the sign and magnitude comparison. Neither table reopens the classification.

| Model | Median relative discrepancy | Median absolute discrepancy | Maximum relative discrepancy | Failed cells |
| --- | ---: | ---: | ---: | ---: |
| qwen | 0.0038577 | 0.00023204 | 1.28001 | 43 |
| gpt2 | 0.0114552 | 7.27028e-05 | 0.764652 | 45 |

The field permutation below uses the recorded projections. `decision_input` is false.

| Split | Blocked status | Identity mean cosine | Permutations strictly better |
| --- | --- | ---: | ---: |
| discovery | `NOT_SUPPORTED` | -0.215768 | 624 |
| validation | `NOT_SUPPORTED` | -0.102475 | 231 |
| replication | `NOT_SUPPORTED` | -0.311621 | 641 |

Where a small-grid sign disagrees, the linear term is small. Those cells are listed in `descriptive_not_decision.json`. They are not a new threshold and not a repaired label.

The blocked field reading is about these models, these prompts, and the functional slot map. It does not say the structure cannot exist.
