# M23 diagnostic audit

This file describes the frozen M23 experiment. It does not rescore it. `reports/M23_SCIENTIFIC_VERDICT.json` remains `INCONCLUSIVE`. No M23 source file, preregistration, split, or result file is an input that this diagnostic is allowed to rewrite.

Source of the description: `scripts/run_m23_direct_qwen_self_model_audit.py`, `src/cognitive_self_model/m23/belief.py`, `reports/M23_PREREGISTRATION.md`, `reports/M23_SPLIT_MANIFEST.json`, and the raw JSON under `reports/m23_raw/`.

## Exact intervention

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Hook: `blocks.23.hook_resid_post`.
- Operation: `h <- h + alpha * direction` at the last token only. Other positions are unchanged.
- Primary direction: unit Gaussian, seed `22101`. Mechanism id `M22.1-D1-L23`.
- Control direction: seed `22103`, then orthogonalized to the primary. Id `M22.1-D2-L23`. Used as shuffled evidence and as part of the pooled outcome-only mean. Not a second mechanism claim.
- Anchor alpha: `+1`. Held-out magnitude: `+2` on replication only.
- Outcome: last-token logit of token `9834` (`" yes"`) minus token `902` (`" no"`).
- Observed effect: intervened margin minus baseline margin.
- Prior status inherited from M22.1: `CANDIDATE`. Not a verified circuit.

Prompt catalog and roles are the M22.1 manifest `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`. Six prompts per role. Order inside a role is sorted `prompt_id`.

| role | use in M23 | alphas saved per prompt |
| --- | --- | --- |
| discovery | initial belief only | D1 and D2 at `+1` were run; per-prompt rows were not written to disk |
| validation | issued prediction, then update | D1 and D2 at `+1`, including `pre_dot` and baseline margin |
| replication | held-out evaluation | D1 at `+1` and `+2` |

The missing discovery rows are a limit on this diagnostic. State-conditioned fits cannot be trained on discovery. They use validation as the only saved pre-intervention feature set, and replication as the only untouched test set.

## Exact prediction representation

`SelfModelBelief.predict(alpha)` reads:

- `predicted_effect`, a scalar at alpha `+1`
- `uncertainty`, a scalar
- `anchor_alpha`, fixed at `+1`

It returns `alpha * predicted_effect`, the sign of that value, and `abs(alpha) * uncertainty`.

It does not read prompt text, regime, baseline margin, `pre_dot`, or the outcome. Every validation prompt received the same prediction: the discovery D1 mean `0.028899987538655598`.

## Exact belief representation

Fields: `mechanism_id`, `intervention_type`, `predicted_effect`, `uncertainty`, `validity_scope`, `evidence_history`, `update_history`, `version`, `observations`, `inflation`.

The initial belief's `observations` are the six discovery D1 deltas. `predicted_effect` is their mean. `uncertainty` is their sample standard deviation. `inflation` starts at 1. Scope status starts as `IN_SCOPE`.

## Exact update rule

After all validation predictions were written, the six validation D1 deltas were applied in `prompt_id` order.

```
mean' = mean(previous observations, new observation)
sd' = sample_sd(previous observations, new observation)
predicted_effect = mean'
uncertainty = sd' * inflation
```

A case is critical only if `abs(prior_mean) >= 1.96 * prior_uncertainty` and either the sign flips or `abs(observation - prior_mean) > 1.96 * prior_uncertainty`. That event sets `inflation` to 2, capped at 2. No validation case was critical. Every update reason was `PRECISION_WEIGHTED_MEAN`.

The final primary mean was `0.02854486306508382`. The final mean is the mean of the six discovery values and the six validation values. Permuting those validation values does not change it. Replacing them with D2 values does.

## Exact holdout

- Calibration: discovery, alpha `+1`, D1.
- Update evidence: validation, alpha `+1`, D1.
- Shuffle / control evidence: validation, alpha `+1`, D2.
- Held-out context: replication, alpha `+1`, D1.
- Held-out magnitude: replication, alpha `+2`, D1, prediction scaled by 2.

Replication predictions were written before replication outcomes existed. Leakage execution recorded all four checks true.

## What was known before the intervention

For a scored prediction, the runner knew:

- the prompt text and its frozen regime and role
- which direction it was about to add, because the runner chose it
- alpha
- the belief scalar estimated from earlier splits
- after the baseline forward, the baseline margin
- after the hook ran but before the add, `pre_dot` of the last residual with the chosen direction

`predict` used only the belief scalar and alpha. Baseline margin and `pre_dot` were logged and then ignored.

## What became known after the intervention

- intervened margin
- observed delta and its sign
- prediction error against the already frozen prediction
- the updated mean, uncertainty, version, and update reason

The final verdict was computed only after replication outcomes existed. It was not an input to prediction.

## Historical result, unchanged

| question | status |
| --- | --- |
| Q1 | PASS |
| Q2 | INCONCLUSIVE |
| Q3 | PASS |
| Q4 | INCONCLUSIVE |
| Q5 | PASS |
| Q6 | INCONCLUSIVE |
| overall | INCONCLUSIVE |

Q1 compared the constant discovery mean with predicting zero. Q4 compared the updated mean with that same discovery mean on replication. The paired interval for the update's absolute-error change included zero. Q5 compared the D1 mean with a pool of D1 and D2. This diagnostic does not reopen those gates.
