# M23 preregistration

Status: frozen before any scored M23 forward. This file is hashed by the runner. A mismatch aborts the run.

## Hypothesis

An explicit belief about the frozen candidate intervention `M22.1-D1-L23` can predict that intervention's logit-margin effect before the intervention is run, can be contradicted by the resulting evidence, can update without copying the observation, and the updated belief can then predict a held-out replication case better than the same belief with updates disabled.

This is not a hypothesis that Qwen has a verified circuit at layer 23. M22.1 left that intervention as `CANDIDATE`.

## Mechanism list

Frozen before evaluation. Not chosen from M23 outcomes.

| id | component | direction | role in M23 | prior status |
| --- | --- | --- | --- | --- |
| `M22.1-D1-L23` | `blocks.23.hook_resid_post`, last token | unit Gaussian seed `22101` | belief target | `CANDIDATE` |
| `M22.1-D2-L23` | same site | seed `22103`, then orthogonalized to D1 | shuffled-evidence control and outcome-only pool only | pre-declared control, not a mechanism claim |

Intervention type for both: additive `h <- h + alpha * direction`.

Expected direction for D1 at alpha `+1`, inherited from the M22.1 descriptive means and not re-estimated here: positive logit-margin delta. The belief's numeric mean is computed from this run's discovery split, not copied from the published mean.

## Intervention list

Anchor magnitude: `+1.0`, the M22.1 `primary_alpha`.

| partition | prompts | direction | alpha |
| --- | --- | --- | --- |
| calibration | discovery | D1 and D2 | `+1` |
| update | validation | D1 | `+1` |
| shuffle evidence | validation | D2 | `+1` |
| eval context | replication | D1 | `+1` |
| eval magnitude | replication | D1 | `+2` |

No other alpha is scored. Alpha `0` is not an intervention. The full M22.1 grid is not rerun.

## Partitions

Inherited from `assign_role(index_within_regime) = index % 3` in the M22.1 prompt file. Manifest hash `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`.

- discovery, calibration, n = 6: `completion-01`, `completion-04`, `instruction-01`, `instruction-04`, `syntax-01`, `syntax-04`
- validation, update, n = 6: `completion-02`, `completion-05`, `instruction-02`, `instruction-05`, `syntax-02`, `syntax-05`
- replication, eval, n = 6: `completion-03`, `completion-06`, `instruction-03`, `instruction-06`, `syntax-03`, `syntax-06`

Order inside a split is sorted `prompt_id`. If discovery n < 2, the run stops and the verdict is `INCONCLUSIVE`.

## Prediction definition

`predict` accepts only the belief and an alpha. It does not accept activations, baseline margin, or outcomes.

```
predicted_effect(alpha) = alpha * belief.predicted_effect
predicted_direction = sign of that value
uncertainty(alpha) = abs(alpha) * belief.uncertainty
```

`belief.predicted_effect` is the mean effect at alpha `+1`. The scale factor is the inherited readout assumption, not a fitted slope.

The initial belief's mean and sample standard deviation (ddof = 1) are computed only from discovery D1 deltas at alpha `+1`.

## Intervention definition

One prompt per forward. `prepend_bos` true. Dtype float32. Model `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

Observed effect = intervened logit margin minus baseline logit margin. Baseline is a forward with no hook. Outcome tokens are the first ids of `" yes"` and `" no"`. If those ids are not `9834` and `902`, the run aborts before scoring.

## Belief-update rule

After the entire validation prediction file is frozen, validation D1 outcomes are applied in `prompt_id` order.

Let the stored alpha-`+1` observations be `x_1..x_n` and the new observation be `x`.

```
mean' = mean(x_1..x_n, x)
sd' = sample_sd(x_1..x_n, x)
```

Contradiction test, using the pre-update mean `m` and pre-update uncertainty `u`:

- sign mismatch: `m` and `x` have opposite strict signs
- outside interval: `abs(x - m) > 1.96 * u`
- confident: `abs(m) >= 1.96 * u`
- critical epistemic case: confident and (sign mismatch or outside interval)

`predicted_effect` becomes `mean'`. It is not replaced by `x`.

`uncertainty` becomes `sd' * inflation`. `inflation` starts at 1. On a critical case it becomes 2 and stays capped at 2.

`validity_scope.status` becomes `CONTRADICTED` on any sign mismatch or outside-interval case, otherwise stays `IN_SCOPE`.

`version` increments by 1. `evidence_history` and `update_history` append the before/after snapshot, the evidence id, and the reason:

- `CONFIDENT_CONTRADICTION` if critical
- `EVIDENCE_OUTSIDE_OR_SIGN` if contradicted but not confident
- `PRECISION_WEIGHTED_MEAN` otherwise

The shuffled control starts from the same discovery D1 belief and applies this rule to validation D2 deltas. The primary belief never reads D2.

## Controls

- Control A, persistence: replication predictions use the discovery-only D1 belief.
- Control D, no-update: same numbers as Control A. The report states that identity instead of treating it as extra evidence.
- Control B, outcome-only: mean of discovery D1 and discovery D2 deltas at alpha `+1`. No mechanism id. No validation update. Scaled by alpha the same way.
- Control C, shuffled evidence: D1 belief updated with D2 validation deltas, then used to predict D1 replication.

## Error metrics

For one case, `prediction_error = observed_effect - predicted_effect`. Magnitude error is the absolute value. Sign agreement ignores cases where either sign is zero; a zero sign is reported and is not counted as agreement.

Split MAE is the mean absolute error over the six prompts.

Uncertainty coverage is the fraction of cases with `abs(error) <= 1.96 * uncertainty`. It is descriptive. It is not a pass gate. No Brier score is computed.

## Statistics

Paired percentile bootstrap, 5000 draws, seed `23001`, over the six prompt-level absolute errors. The 95% interval is the 2.5 and 97.5 percentiles.

- interval entirely above 0: `CI_POSITIVE`
- entirely below 0: `CI_NEGATIVE`
- otherwise: `CI_INCLUDES_ZERO`

Sign accuracy uses a two-sided Clopper-Pearson 95% interval. The pass comparison is the lower bound against 0.5. This avoids a degenerate bootstrap that collapses to `[1, 1]` when every sign matches.

## Decision rules

These are the only pass/fail rules. No threshold is added after the run.

Q1, prediction, scored on validation D1 before those outcomes update the belief.

- `PASS` if the MAE interval of `(AE_predict_zero - AE_belief)` is `CI_POSITIVE` and the sign-accuracy lower bound is above 0.5.
- `FAIL` if that MAE interval is `CI_NEGATIVE`, or the sign-accuracy upper bound is below 0.5.
- otherwise `INCONCLUSIVE`.

Q2, falsifiability, scored on validation D1.

- `PASS` if at least one case is a critical epistemic case.
- otherwise `INCONCLUSIVE`. Absence of a contradiction is not converted to `FAIL` and is not converted to `PASS`.

Q3, revision.

- `PASS` if every update increments `version`, stores before/after, and sets `predicted_effect` to the mean of the stored observations rather than to the new observation alone, except in the algebraic case where every stored observation already equals the new one.
- `FAIL` otherwise.

Q4, learning, scored on replication D1 at alpha `+1`.

- `PASS` if `(AE_no_update - AE_updated)` is `CI_POSITIVE` and `(AE_no_update - AE_shuffled)` is not `CI_POSITIVE`.
- `FAIL` if `(AE_no_update - AE_updated)` is `CI_NEGATIVE`, or both the updated arm and the shuffled arm are `CI_POSITIVE`.
- otherwise `INCONCLUSIVE`.

Q5, specificity, same replication cases.

- `PASS` if `(AE_outcome_only - AE_updated)` is `CI_POSITIVE`.
- `FAIL` if that interval is `CI_NEGATIVE`.
- otherwise `INCONCLUSIVE`.

Q6, generalization, replication D1 at alpha `+2`, predictions scaled by 2.

- Same rule as Q4, applied to the alpha `+2` errors.
- A separate linearity diagnostic compares scaled no-update MAE with a predictor that uses the alpha-`+1` mean unchanged at alpha `+2`. The diagnostic is reported and does not change Q6.

Overall:

- `PASS` only if Q1, Q2, Q3, Q4, Q5, and Q6 are all `PASS`, and the leakage execution check passes.
- `FAIL` if any question is `FAIL`, or leakage fails.
- `INCONCLUSIVE` if no question is `FAIL`, leakage passes, and at least one question is `INCONCLUSIVE`.

Failure classes, assigned only from the rules above:

| condition | class |
| --- | --- |
| Q1 `FAIL` | `PREDICTION_FAILURE` |
| Q3 `FAIL` | `BELIEF_UPDATE_FAILURE` |
| Q4 `FAIL` because the update is worse | `BELIEF_UPDATE_FAILURE` |
| Q4 `FAIL` because shuffled also improves | `BASELINE_MATCH` |
| Q5 `FAIL` | `BASELINE_MATCH` |
| Q6 `FAIL` | `GENERALIZATION_FAILURE` |
| leakage execution check fails | `LEAKAGE` |

`PREDICTION_FAILURE` here means the frozen candidate belief did not beat predicting zero. It is not a new mechanism search.

## Leakage rules

Listed in `reports/M23_LEAKAGE_AUDIT.md`. A scored run that violates ordering does not receive `PASS`.

## Stopping rules

Stop after the verdict. Do not start M24. Do not add a model, a new direction, or a new prompt because a question is `INCONCLUSIVE` or `FAIL`.

Abort before scoring if the model is not Qwen2.5-0.5B with 24 layers and `d_model` 896, if the token ids differ from the frozen pair, if discovery has fewer than two finite D1 deltas, or if a preregistered file hash mismatches.

## Allowed conclusion if the overall verdict is PASS

"A limited operational self-model loop was demonstrated on the tested Qwen mechanism/intervention setting."

That setting remains the M22.1 candidate intervention, not a verified circuit. A pass still does not authorize a general self-model claim.
