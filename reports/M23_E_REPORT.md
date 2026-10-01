# M23-E report

Historical M23 remains `INCONCLUSIVE`. The diagnostic labels remain `INTERVENTION_DESIGN_FAILURE = INCONCLUSIVE`, `IDENTIFIABILITY_FAILURE = INCONCLUSIVE`, `REPRESENTATION_BOTTLENECK = INCONCLUSIVE`, and `BELIEF_UPDATE_INFORMATION_FAILURE = NOT_SUPPORTED`. M23-D2 remains `INCONCLUSIVE`.

## 1. Scientific question

Can pre-intervention conditions be associated with materially different responses to the current intervention?

The question is not whether a belief predicts the effect, and it is not whether the system has a self-model.

## 2. Frozen setup

Model `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`. Hook `blocks.23.hook_resid_post`. Direction seed `22101`. Catalog of 18 M22.1 prompts. Alphas `+1` and `+2` only. Two identical repeats. Train and evaluation are the even and odd positions in sorted `prompt_id` order, nine prompts each. Replication ids appear on both sides.

`baseline_output` and `baseline_margin` are the same yes/no logit margin. `pre_dot` is the last-token residual dotted with the direction before the add. Regime was not a feature.

## 3. Intervention definition

Last-token residual update `h <- h + alpha * d`. Effect is the intervened yes/no margin minus the baseline margin. Token ids were `9834` and `902`.

## 4. Alpha +1 results

Eighteen repeat-0 effects. Mean `0.028635210461086698`. Sample standard deviation `0.003194735307400828`. Minimum `0.024469375610351562`. Maximum `0.03369140625`. Range `0.009222030639648438`.

`pre_dot` on the same rows has sample standard deviation `1.8998575552355634` and range `7.52262556552887`. Baseline margin has sample standard deviation `2.9484035441639946` and range `10.314473152160645`.

Correlations with the effect:

| split | pre_dot | baseline margin |
| --- | ---: | ---: |
| train, n = 9 | 0.5290199328984694 | 0.3171901835964982 |
| evaluation, n = 9 | -0.05245642883730313 | 0.46246116996295566 |
| all 18, descriptive only | 0.1506488352668959 | 0.35062809449257876 |

Held-out mean absolute error: constant `0.002999264516948182`, `pre_dot` line `0.0033368054305972054`, baseline-margin line `0.002772879667704526`.

The train `pre_dot` correlation does not appear on the evaluation prompts.

## 5. Alpha +2 results

Mean `0.05773798624674479`. Sample standard deviation `0.006294624879313493`. Minimum `0.04981803894042969`. Maximum `0.06708049774169922`. Range `0.01726245880126953`.

Correlations with the effect:

| split | pre_dot | baseline margin |
| --- | ---: | ---: |
| train, n = 9 | 0.5340292066522973 | 0.23784002748828656 |
| evaluation, n = 9 | -0.02890261949827802 | 0.4210560594992011 |
| all 18, descriptive only | 0.16550656869292782 | 0.28518067198256597 |

Held-out mean absolute error: constant `0.0059486789467894`, `pre_dot` line `0.006613493834232568`, baseline-margin line `0.005613836079526216`.

The same pattern appears at both alphas: `pre_dot` correlates on the train half and is about zero on the evaluation half.

## 6. Scaling analysis

No ratio was undefined. Across 18 prompts, `effect(+2) / effect(+1)` has mean `2.016878657674608`, median `2.0188682903223416`, and sample standard deviation `0.01582312022402805`. The smallest ratio is `1.991026947463768` (`instruction-04`). The largest is `2.046069723883753` (`syntax-01`).

The earlier replication observation was about 2.016. The full-catalog mean ratio is in that same neighborhood. This is a description, not a pass.

## 7. Within-prompt scaling residual

`scaling_residual = effect(+2) - 2 * effect(+1)`.

Mean `0.00046756532457139756`. Sample standard deviation `0.00046046842944423105`. Minimum `-0.00030231475830078125`. Maximum `0.0014543533325195312`.

The residual mean is about 0.8% of the alpha `+2` mean effect. Doubling alpha almost doubles the effect on every prompt. The departure from exact doubling is small, reproducible, and slightly positive on average. That pattern is a nearly linear response to alpha, not a large context-dependent bend.

## 8. Pre-intervention identifiability analysis

Paired difference is constant absolute error minus feature absolute error on the nine evaluation prompts. Positive would mean the feature is closer. Interval: 5000 draws, seed `23001`.

| alpha | feature | mean paired difference | interval | result |
| --- | --- | ---: | --- | --- |
| +1 | pre_dot | -0.00033754091364902373 | [-0.0019019673746615156, 0.0012267150105604527] | `INCONCLUSIVE` |
| +1 | baseline margin | 0.00022638484924365572 | [-0.00035256643147935237, 0.0008100369557046016] | `INCONCLUSIVE` |
| +2 | pre_dot | -0.000664814887443168 | [-0.003784344748839971, 0.0024580912785688905] | `INCONCLUSIVE` |
| +2 | baseline margin | 0.0003348428672631844 | [-0.0005214763939005224, 0.0011981150582446541] | `INCONCLUSIVE` |

No contrast was selected after seeing the errors. The baseline-margin line has a smaller point error than the constant at both alphas, and the interval still includes zero. The `pre_dot` line has a larger point error than the constant at both alphas, and that interval also includes zero.

No evidence that the tested pre-intervention measurements identify variation in intervention response.

## 9. Repeatability and noise

Repeat 0 and repeat 1 matched exactly. Maximum absolute difference on `pre_dot`, baseline output, intervened output, and effect was `0`. Mean within-condition variance was `0`. Between-prompt sample variance was `1.0206333684353464e-05` at alpha `+1` and `3.9622302371272404e-05` at alpha `+2`.

The runtime is deterministic. The second pass does not estimate stochastic noise. The between-prompt spread is a property of the prompts under this intervention, not a draw-to-draw fluctuation.

A read-only comparison with the 60 previously saved cells (validation alpha `+1`, replication alpha `+1` and `+2`, discovery alpha `+1` effects) had maximum absolute difference `0`. Those files were not fit inputs.

## 10. Evidence on intervention-design failure

`INTERVENTION_DESIGN_FAILURE` stays `INCONCLUSIVE`. No cutoff for "effectively stable" was frozen, and this run does not add one.

What the measurements show, without that label:

- Every alpha `+1` effect sits in a band of width `0.009222030639648438` around `0.028635210461086698`.
- Every alpha `+2` effect sits in a band of width `0.01726245880126953` around `0.05773798624674479`.
- That spread is larger than the repeat noise, which is zero, so the prompts are not literally interchangeable.
- The spread is small next to the pre-intervention measurements. The `pre_dot` sample standard deviation is about 595 times the alpha `+1` effect sample standard deviation.
- Changing alpha from `+1` to `+2` moves the effect in nearly the same ratio on every prompt.

The intervention behaves as one shared response to alpha, plus a small prompt-level remainder. It does not behave as a response that swings with the measured pre-intervention state.

## 11. Evidence on identifiability failure

`IDENTIFIABILITY_FAILURE` stays `INCONCLUSIVE`.

The four held-out intervals include zero. Train `pre_dot` correlation is about `0.53` at both alphas and about zero, with a negative sign, on the evaluation half. That is not identification. It is also not a demonstration that every possible pre-intervention measurement would fail. The tested ones did not beat the constant under the frozen interval rule.

## 12. Remaining uncertainty

Nine evaluation prompts cannot separate a small true association from none. The full-catalog correlations are not a substitute. The baseline-margin point estimate favors the feature by a few ten-thousandths, inside an interval that includes zero. Stochastic noise was not estimated, because the forwards repeat exactly. Whether the prompt-level remainder would matter for a belief update was not given a cutoff.

These statements are separate:

- The intervention effect is nearly stable across these prompts and scales almost exactly with alpha.
- The tested pre-intervention measurements do not identify the remaining variation on held-out prompts.
- Nothing here shows that the system has a self-model of the intervention.

## 13. Recommendation for the next experiment

Do not fit another predictor on this catalog and this D1 intervention. The response is already described: about `0.0286` at alpha `+1`, about twice that at alpha `+2`, ratio mean `2.016878657674608` with sample standard deviation `0.01582312022402805`, and no held-out gain from `pre_dot` or baseline margin.

A later experiment would need a different pre-registered intervention or condition set, frozen before its outcomes are inspected. Adding features, a neural predictor, or another split of these 18 prompts would not answer a new question. That experiment is not part of this task.

## Leakage

`leakage_ok = true`. The split file preceded the measurements. Coefficients preceded the evaluation scores. The fit used the nine train ids. Evaluation is not the replication set alone. Regime and outcomes were not features. Repeats kept the same prompt, seed, and alpha. Historical files were not copied into the fit.
