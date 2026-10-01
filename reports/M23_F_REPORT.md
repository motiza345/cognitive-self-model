# M23-F report

Historical M23 remains `INCONCLUSIVE`. Diagnostic labels, M23-D2, and M23-E are unchanged.

## Status

`audit_status = NO_SUITABLE_EXISTING_INTERVENTION`

`current_intervention_status = CURRENT_INTERVENTION_INSUFFICIENT`

No predictor was fit. Candidates were not ranked. Validation and replication prompts were not forwarded. Every repeat matched exactly.

## 1. Scientific question

The loop under test is prediction, then a contradiction, then belief revision, then a better prediction on a new case. That loop needs an intervention a pre-intervention belief can get wrong. A shared response to alpha does not supply that contradiction.

## 2. What was screened

Six discovery prompts. Fourteen existing cells: D1 and D2 at layers 0, 8, 15, and 23, plus D3–D8 at layer 23. Alphas `+1` and `+2`. Two repeats. The forward is the same one used in M23. D1 at layer 23 matches the M23-E discovery effects exactly (maximum absolute difference `0`).

The older M22.1 preflight table is not bit-identical to this forward. Two layer-0 prompt signs differ between that table and this screen. The screen's own signs are the ones used below. The preflight table is not used to pick a layer.

## 3. Current intervention, D1 at layer 23

On the six prompts at alpha `+1`: mean `0.028900`, sample standard deviation `0.003592`, range `0.008759`, signs `{+1}` only.

Alpha `+2` / alpha `+1`: mean ratio `2.018578`, sample standard deviation `0.020350`, ratios from `1.991027` to `2.046070`.

Repeats were identical. This is the shared-scale pattern already measured on all 18 prompts in M23-E. A belief that names one positive effect near `0.029 * alpha` is not contradicted by prompt identity on this screen.

| question | answer |
| --- | --- |
| Stable response? | One shared positive sign. The prompt range is small. |
| Meaningful response variation? | A nonzero range exists. It does not change sign. |
| Repeatable? | Yes. The two forwards were equal. This does not estimate stochastic noise. |
| Pre-intervention information? | Not refit. M23-E already found no held-out gain for `pre_dot` or baseline margin. |
| Can a prediction be falsified? | Not by which discovery prompt is used. Alpha changes the size in a tight ratio. |
| Already inspected? | Yes. Discovery, validation, and replication. |
| Clean future holdout on this catalog? | No. |

## 4. Cells where the sign is not shared

These contradict a prediction that the intervention has one sign. They are not ordered by range. None is selected.

| intervention | alpha `+1` mean | sample sd | range | signs | mean alpha ratio | ratio sample sd |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| D1 layer 0 | -0.027426 | 0.105182 | 0.274409 | both | 3.888396 | 2.145635 |
| D1 layer 8 | 0.006429 | 0.052125 | 0.134738 | both | 2.250301 | 0.328123 |
| D1 layer 15 | 0.010673 | 0.041288 | 0.103331 | both | 2.044520 | 0.299286 |
| D2 layer 0 | 0.053598 | 0.069112 | 0.170249 | both | -1.056339 | 3.807451 |
| D2 layer 8 | 0.006291 | 0.068769 | 0.184169 | both | 1.968277 | 0.478352 |

D2 at layer 0 has an alpha ratio of `-6.645192` on one prompt and `2.509814` on another. Doubling alpha does not keep a single scale there. That description is not a score.

For each of these five:

| question | answer |
| --- | --- |
| Stable response? | No. Both signs appear at alpha `+1`. |
| Meaningful response variation? | Yes, in the structural sense that a shared sign is false. No cutoff was applied to the range. |
| Repeatable? | Yes. Repeats matched exactly. |
| Pre-intervention information? | Not tested. `pre_dot` was stored and not correlated. |
| Can a prediction be falsified? | A one-sign prediction is contradicted on this discovery screen. |
| Already inspected? | Discovery was already used for layer selection or for MRSM Q. MRSM Q also used validation. |
| Clean future holdout? | No. Replication was never reserved before those inspections. |

## 5. Layer-23 directions other than the current one

Each has one sign. Each differs from D1. Within a direction the range stays small and the alpha ratio stays near 2, except where noted.

| intervention | mean | sample sd | range | signs | mean ratio | ratio sample sd |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| D2 layer 23 | 0.106638 | 0.012992 | 0.027853 | positive | 2.004473 | 0.004670 |
| D3 layer 23 | 0.037230 | 0.003974 | 0.009222 | positive | 2.013467 | 0.016607 |
| D4 layer 23 | -0.027748 | 0.003434 | 0.007996 | negative | 1.980463 | 0.021122 |
| D5 layer 23 | 0.009019 | 0.002542 | 0.006583 | positive | 2.070419 | 0.092458 |
| D6 layer 23 | 0.035609 | 0.003672 | 0.008910 | positive | 2.015315 | 0.017380 |
| D7 layer 23 | 0.016464 | 0.002359 | 0.006581 | positive | 2.029824 | 0.035304 |
| D8 layer 23 | -0.014439 | 0.001501 | 0.003674 | negative | 1.963099 | 0.041940 |

D2 at layer 15 is the remaining cell: one positive sign, mean `0.058856`, range `0.100332`, mean ratio `2.198938` (sample sd `0.437422`). It does not change sign. It is not D1.

For this group:

| question | answer |
| --- | --- |
| Stable response? | One sign inside each direction. |
| Meaningful response variation? | Prompt ranges are small beside the gap between directions. D4 and D8 are negative while D1 is positive, so direction identity changes the sign. |
| Repeatable? | Yes. Repeats matched. |
| Pre-intervention information? | Not tested, and not a basis for ranking. |
| Can a prediction be falsified? | A belief that every direction equals D1 is false on this screen. That contrast was already published for D2–D8 on all three splits. |
| Already inspected? | Yes, including validation and replication. |
| Clean future holdout? | No. |

## 6. Which structural criteria hold

Same mechanism, different prompt, different sign: D1 at layers 0, 8, and 15, and D2 at layers 0 and 8, on the discovery screen only.

Different mechanisms, distinguishable effects: the layer-23 panel. The effects are distinguishable and already known.

Same mechanism, controlled alpha change, predictable scale: D1 at layer 23, and the other layer-23 directions with ratios near 2. Predictable scaling is the opposite of a contradiction. D2 at layer 0 does not have that stable scale.

None of this is a self-model. No belief was updated, and no held-out case was scored.

## 7. Contamination

Every cell that shows a sign disagreement, and every direction that differs from D1, already has a confirmatory split inspected. The clean set is empty. Leftover replication prompts for the early layers are not a holdout that was reserved in advance.

## 8. Remaining uncertainty

The discovery screen has six prompts. Early-layer alpha `+2` behavior was not in the original preflight, and this screen does not match that preflight prompt by prompt. Repeats show the runtime is deterministic, not that a second stochastic draw would match. Whether any pre-intervention measurement tracks the early-layer sign changes was deliberately not fit.

## 9. Recommendation

Do not run a self-model test on D1 at layer 23. Do not choose layer 0, or D2 at layer 0, because the range or the alpha ratio looks large.

No existing cell can support a clean predict-then-holdout experiment on this 18-prompt catalog. A later experiment would need new prompts, and it would have to precommit to the whole set of sign-disagreement cells before seeing their outcomes. That experiment was not designed or run.
