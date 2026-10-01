# M23-E preregistration

Status: frozen before any M23-E forward.

Historical M23 remains `INCONCLUSIVE`. The M23 diagnostic labels remain unchanged. M23-D2 remains `INCONCLUSIVE`. This file does not rewrite those artifacts.

## Question

Can pre-intervention conditions be associated with materially different responses to the current Qwen intervention?

This is not a test of whether a belief can predict the effect, and it is not a test of a self-model. A stable response would make this intervention unsuitable for a later belief-revision test. A response that varies with pre-intervention measurements would make a later test identifiable. Neither result establishes a self-model.

## Frozen intervention

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Hook: `blocks.23.hook_resid_post`
- Operation: last-token `h <- h + alpha * d`. Other positions stay unchanged.
- Direction: unit Gaussian seed `22101`. Id `M22.1-D1-L23`.
- Alphas: `+1` and `+2` only. Alpha is not changed after results.
- Outcome tokens: `9834` (`" yes"`) minus `902` (`" no"`). The run aborts if tokenization differs.
- Prompt catalog: the existing M22.1 catalog, manifest `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`. All 18 prompts. None added or dropped.
- Bootstrap, when a paired interval is used: 5000 draws, seed `23001`, percentiles 2.5 and 97.5, from `paired_mean_ci`.

No second direction, no new hook, and no regime feature.

## Measurements

For every prompt, alpha, and repeat:

- `baseline_output`: last-token yes/no logit margin with no hook. This is the repository's existing baseline-margin quantity. It is not a second feature.
- `pre_dot`: dot product of the last-token residual with `d`, recorded in the hook before the add.
- `intervened_output`: yes/no logit margin after the add.
- `observed_effect`: `intervened_output - baseline_output`.

`pre_intervention_observation` is the pair (`pre_dot`, `baseline_margin`). `baseline_margin` is the same number as `baseline_output`.

Not used as predictors: observed effect, intervened output, post-intervention activation, future residual, token label, regime, prompt text, prompt id.

## Partition

Sorted `prompt_id`, zero-based index. Even index trains. Odd index evaluates. This rule uses the catalog order only. It was not chosen from effect sizes.

Train:

- `completion-01`, `completion-03`, `completion-05`
- `instruction-01`, `instruction-03`, `instruction-05`
- `syntax-01`, `syntax-03`, `syntax-05`

Evaluation:

- `completion-02`, `completion-04`, `completion-06`
- `instruction-02`, `instruction-04`, `instruction-06`
- `syntax-02`, `syntax-04`, `syntax-06`

Replication prompts are `*-03` and `*-06`. They fall on both sides, so replication is not the evaluation set. No prompt is moved after a forward.

## Repeats

Two forwards for every prompt at baseline, alpha `+1`, and alpha `+2`. Same prompt, seed, model, hook, direction, and alpha. Nothing is noised on purpose.

Primary tables use repeat 0. Repeat 1 is the determinism check. If the two repeats are exactly equal, the runtime is deterministic and the repeat does not estimate stochastic noise.

## Predictors

Fit on repeat 0 of the train prompts only, separately at each alpha. Closed-form line with an intercept. No penalty, no feature search, no neural model.

- A, constant: train mean of `observed_effect`.
- B, `pre_dot`: `effect = a + b * pre_dot`.
- C, baseline margin: `effect = a + b * baseline_margin`.

Coefficients are written before evaluation absolute errors are computed. They are not refit after those errors.

## What is decisive

For each alpha and each of B and C, on the nine evaluation prompts:

```
paired_difference = abs(effect - prediction_A) - abs(effect - prediction_feature)
```

Positive means the feature is closer than the constant.

- Interval entirely above zero: `SUPPORTED`
- Interval entirely below zero: `NOT_SUPPORTED`
- Otherwise: `INCONCLUSIVE`

The four contrasts stay separate. The better point estimate is not selected after the fact.

Pearson correlations on train, evaluation, and the full catalog are descriptive. A full-catalog correlation is not evidence.

If none of the four contrasts is `SUPPORTED`, the result sentence is:

"No evidence that the tested pre-intervention measurements identify variation in intervention response."

A `SUPPORTED` contrast says only that the named linear measurement reduced held-out absolute error for that alpha. It does not say the system has a self-model.

## Scaling

On repeat 0 of all 18 prompts, descriptive only. No pass threshold.

- ratio `effect(+2) / effect(+1)`
- mean, median, sample standard deviation, and the 18 ratios
- residual `effect(+2) - 2 * effect(+1)`, with mean, sample standard deviation, minimum, maximum, and the 18 values

If an alpha `+1` effect is zero, that ratio is undefined and is excluded from the ratio summaries. The count of exclusions is reported. The multiplier 2 is fixed. It is not estimated.

## Design-failure statement

No new cutoff defines "effectively stable." The report gives, at each alpha, the mean, sample standard deviation, minimum, maximum, and range of the 18 repeat-0 effects, plus the sample variances of `pre_dot` and baseline margin.

`INTERVENTION_DESIGN_FAILURE` for this experiment is `INCONCLUSIVE`. The historical diagnostic label is not changed.

## Labels that stay fixed

| artifact | label |
| --- | --- |
| M23 verdict | `INCONCLUSIVE` |
| `INTERVENTION_DESIGN_FAILURE` | `INCONCLUSIVE` |
| `IDENTIFIABILITY_FAILURE` | `INCONCLUSIVE` |
| `REPRESENTATION_BOTTLENECK` | `INCONCLUSIVE` |
| `BELIEF_UPDATE_INFORMATION_FAILURE` | `NOT_SUPPORTED` |
| M23-D2 | `INCONCLUSIVE` |

## Prior looks

Alpha `+1` effects for these 18 prompts, and alpha `+2` effects for the six replication prompts, were already written by earlier runs. Validation and replication also have saved `pre_dot` and baseline margin. This run measures every cell again. Earlier files are not inputs to the fit. A read-only comparison with those files is recorded after the forwards and does not replace a measurement.
