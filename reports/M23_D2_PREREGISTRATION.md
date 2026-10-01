# M23-D2 preregistration

Status: frozen before any M23-D2 evaluation forward.

Historical M23 stays `INCONCLUSIVE`. The M23 diagnostic labels stay as recorded. This file does not change them.

## Question

Does adding regime as the only pre-intervention feature improve held-out prediction of the D1 intervention effect, relative to the current constant belief?

## Intervention

Unchanged from M23.

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Hook: `blocks.23.hook_resid_post`, last token
- Direction: unit Gaussian seed `22101`
- Intervention id: `M22.1-D1-L23`
- Alpha: `+1` only
- Effect: intervened yes/no logit margin minus baseline margin
- Token ids must be `9834` and `902`, or the run aborts

No new direction, magnitude, feature, or mechanism search.

## Regime

The regime is the existing M22.1 `regime_id`: `completion`, `instruction`, or `syntax`. It is a property of the frozen prompt text, assigned before any outcome. No new regime is defined.

## Predictors

Both predictions are functions of the M23 validation D1 rows only. Those rows were already saved. They are not recomputed from a new forward. Replication prompts are not used.

Predictor A, current constant belief:

```
prediction_A = 0.028189738591512043
```

for every evaluation prompt. This is the mean of the six saved validation D1 effects. It does not depend on prompt or regime. That is the M23 constant-belief behavior: one scalar for every case.

Predictor B, regime-mean belief:

| regime | n in train | prediction_B |
| --- | ---: | ---: |
| completion | 2 | 0.028961896896362305 |
| instruction | 2 | 0.02886199951171875 |
| syntax | 2 | 0.026745319366455078 |

`prediction_B` depends only on `regime_id`. It does not use activation, `pre_dot`, baseline margin, prompt text, or any outcome.

## Split

A three-way train/calibration/evaluation split is not used. Each regime has only two prompts in the untouched pool, so a third partition would leave regimes empty or singletons. No calibration split is required: both predictors are means, not tuned models.

| partition | source | n | role in this comparison |
| --- | --- | ---: | --- |
| train | M23 validation role | 6 | estimate A and B |
| evaluation | M23 discovery role | 6 | held-out forwards |
| excluded | M23 replication role | 6 | not used |

The replication prompts are excluded because the previous diagnostic already reported their regime means and their paired error against a constant. Discovery per-prompt effects were not saved and were not used to choose regime. Validation regime means were already printed in the diagnostic report. That is a limitation of the training estimates, not of the evaluation outcomes: the evaluation outcomes do not exist yet as per-prompt records.

Evaluation prompt ids, sorted:

- `completion-01`
- `completion-04`
- `instruction-01`
- `instruction-04`
- `syntax-01`
- `syntax-04`

Two prompts per regime.

## Ordering

1. Write the six evaluation predictions from the table above.
2. Confirm the evaluation outcome file does not exist.
3. Run baseline and D1 alpha `+1` forwards on the evaluation prompts only.
4. Score.

## Primary comparison

For each evaluation prompt:

```
paired_difference = abs(observed - prediction_A) - abs(observed - prediction_B)
```

Positive means B is closer than A.

Report the six differences, both mean absolute errors, and the mean paired difference.

## Interval rule

Reuse `paired_mean_ci` from `src/cognitive_self_model/m23/stats.py`: 5000 draws, seed `23001`, percentiles 2.5 and 97.5.

- Interval entirely above zero: `REGIME_INFORMATION_SUPPORTED`
- Interval entirely below zero: `REGIME_FEATURE_HURTS`
- Otherwise: `INCONCLUSIVE`

No other threshold. n = 6 is reported as small. A supported result is only the narrow statement that regime contained predictive information about this D1 effect beyond the constant. It is not a self-model claim.

## Allowed statements

If supported:

"Regime contains predictive information about D1 intervention effect that is not captured by the constant belief."

If inconclusive or if regime hurts:

"No evidence was obtained in this experiment that regime information improves held-out prediction of the tested D1 intervention effect."

The second sentence is also used when regime hurts, with the additional label `REGIME_FEATURE_HURTS`.
