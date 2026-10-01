# M23-G preregistration

Status: frozen before any M23-G Qwen forward. This task does not run those forwards.

Historical M23 remains `INCONCLUSIVE`. M23-F remains `NO_SUITABLE_EXISTING_INTERVENTION`. D1 at layer 23 remains `CURRENT_INTERVENTION_INSUFFICIENT`. None of those labels is reopened here.

## A. Scientific hypothesis

A separate belief for each cell in the frozen family can predict that cell's alpha-`+1` logit-margin effect before the intervention, can be contradicted by the validation evidence under the existing M23 rule, can update without replacing the prediction by the observation, and the updated belief can then have lower absolute error than the untouched train belief on the new evaluation prompts.

This is not a hypothesis that any cell is a verified mechanism, and it is not a hypothesis that Qwen has a self-model. A positive interval would say only that this update reduced held-out absolute error relative to persistence.

## B. Intervention family

The family is every already-implemented D1 and D2 cell at the preflight layers that were not adopted as the current intervention. Layers `0`, `8`, and `15` are the other layers in the M22.1 candidate list `{0, 8, 15, 23}`. D1 and D2 are the two directions already constructed for those layers. No cell is dropped, and no cell is added.

| intervention_id | layer | direction | alpha | historical status | why it is included |
| --- | ---: | --- | ---: | --- | --- |
| `M22.1-D1-L0` | 0 | D1, seed `22101` | `+1` | not selected, not verified | remaining preflight layer, existing direction |
| `M22.1-D2-L0` | 0 | D2, seed `22103` orthogonal to D1 | `+1` | not verified | same layer family, existing second direction |
| `M22.1-D1-L8` | 8 | D1 | `+1` | not selected, not verified | remaining preflight layer |
| `M22.1-D2-L8` | 8 | D2 | `+1` | not verified | same layer family |
| `M22.1-D1-L15` | 15 | D1 | `+1` | not selected, not verified | remaining preflight layer |
| `M22.1-D2-L15` | 15 | D2 | `+1` | not verified | same layer family; not removed for its screen |

Excluded, and not available as substitutes after a result:

- `M22.1-D1-L23`: the current intervention. All three old splits are consumed. M23-F called it insufficient for this loop.
- `M22.1-D2-L23` and `M22.1-D3-L23` through `M22.1-D8-L23`: layer-23 panel only. Discovery, validation, and replication outcomes already exist.
- D3–D8 at layers 0, 8, or 15: those cells were never implemented. This protocol does not create them.

Operation, unchanged: last-token `h <- h + alpha * direction` on `blocks.{layer}.hook_resid_post`. Model `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, dtype float32, `prepend_bos` true. Outcome tokens must be `9834` and `902` or the run aborts. Observed effect is the intervened yes/no margin minus the baseline margin.

Alpha is frozen at `+1` only. Alpha `+2` is not scored.

Historical screens motivate why the current cell is not reused. They are not confirmation of these cells on the new prompts.

## C. Prompt catalog

Thirty-six new prompts. None of the eighteen M22.1 texts or ids are reused. Surface families are balance strata. `predict` does not receive the family label.

Completion:

| id | partition | text |
| --- | --- | --- |
| `g-completion-01` | train | The largest planet in the solar system is |
| `g-completion-02` | validation | A bicycle has two |
| `g-completion-03` | evaluation | The author of Hamlet is |
| `g-completion-04` | train | Ice is the solid form of |
| `g-completion-05` | validation | The Pacific is an |
| `g-completion-06` | evaluation | A triangle has three |
| `g-completion-07` | train | The currency of Japan is the |
| `g-completion-08` | validation | Bread is usually baked in an |
| `g-completion-09` | evaluation | The tallest animal is the |
| `g-completion-10` | train | A year has twelve |
| `g-completion-11` | validation | The chemical symbol for gold is |
| `g-completion-12` | evaluation | Wool comes from a |

Syntax:

| id | partition | text |
| --- | --- | --- |
| `g-syntax-01` | train | When the lights went out, |
| `g-syntax-02` | validation | Unless the door is locked, |
| `g-syntax-03` | evaluation | While the soup was cooling, |
| `g-syntax-04` | train | As soon as the train stopped, |
| `g-syntax-05` | validation | Even though the map was torn, |
| `g-syntax-06` | evaluation | Whenever the bell rings, |
| `g-syntax-07` | train | Since the library was closed, |
| `g-syntax-08` | validation | Provided that the ticket is valid, |
| `g-syntax-09` | evaluation | Once the paint had dried, |
| `g-syntax-10` | train | Whether or not it snows, |
| `g-syntax-11` | validation | Until the music ended, |
| `g-syntax-12` | evaluation | Whereas the first attempt failed, |

Instruction:

| id | partition | text |
| --- | --- | --- |
| `g-instruction-01` | train | Reply with one word. A metal used in coins: |
| `g-instruction-02` | validation | Reply with one word. The meal eaten at noon: |
| `g-instruction-03` | evaluation | Name a tool used for writing: |
| `g-instruction-04` | train | Name a month of the year: |
| `g-instruction-05` | validation | Reply with one word. Opposite of early: |
| `g-instruction-06` | evaluation | Name a kind of tree: |
| `g-instruction-07` | train | Reply with one word. A frozen dessert: |
| `g-instruction-08` | validation | Name a musical instrument: |
| `g-instruction-09` | evaluation | Reply with one word. The number of sides on a square: |
| `g-instruction-10` | train | Name a continent: |
| `g-instruction-11` | validation | Reply with one word. A bird that cannot fly: |
| `g-instruction-12` | evaluation | Name a piece of furniture: |

## D. Partitioning

Within each family, prompt index `k` starting at 1 is assigned by `(k - 1) mod 3`: `0` train, `1` validation, `2` evaluation. Each partition has 12 prompts, four from each family.

Train builds the initial belief. Validation is the only update evidence. Evaluation is held out.

Twelve prompts per cell is double the M23 split and is still small. The protocol keeps three partitions because each cell needs at least two train observations for `belief_from_observations`, a separate update split, and a separate evaluation split. It does not add prompts if an interval includes zero.

The old replication ids are not the evaluation set. The old eighteen prompts are not in any partition.

## E. Prediction timing

There is one belief per cell. Train has no prediction. Its alpha-`+1` effects are the initial observations.

Validation predictions are produced by `predict(1.0)` on the train belief and written to disk first. Predictions for validation are written before validation outcomes exist.

After every validation outcome for every cell has been applied, evaluation predictions are written for the updated belief and for every control. Predictions for evaluation are written before evaluation outcomes exist.

A prediction record contains `predicted_effect`, `predicted_direction`, and `uncertainty`. It is the dict returned by `predict`. The file is not edited after the corresponding forward begins. `update_belief` returns a new object and is not allowed to change the object that was predicted from.

## F. Intervention timing

Order is fixed: all train forwards, then the validation prediction file, then validation forwards, then updates, then the evaluation prediction file, then evaluation forwards. No evaluation forward starts while the evaluation prediction file is absent.

## G. Belief representation

`SelfModelBelief` as already implemented. Fields used: `mechanism_id`, `intervention_type`, `predicted_effect`, `uncertainty`, `validity_scope`, `observations`, `inflation`, `anchor_alpha`, `version`.

`predict` accepts the belief and alpha only. It does not accept prompt text, family, `pre_dot`, baseline margin, or an outcome.

```
predicted_effect = alpha * stored_mean
predicted_direction = sign of that value
uncertainty = abs(alpha) * stored_uncertainty
```

The stored mean is the effect at alpha `+1`. Here alpha is `+1`, so the scale factor is 1.

## H. Update rule

The existing `update_belief` function. It is not redesigned.

After the validation prediction file is frozen, each cell's validation effects are applied in `prompt_id` order to that cell's belief only.

```
mean' = mean(previous observations, new observation)
sd' = sample_sd(previous observations, new observation)
predicted_effect = mean'
uncertainty = sd' * inflation
```

`predicted_effect` becomes the cumulative mean. It is not replaced by the new observation. The implementation raises if that replacement happens.

Contradiction, already defined by `CRITICAL_Z = 1.96` in the existing function:

- sign mismatch: the stored mean and the observation have opposite strict signs
- outside interval: `abs(observation - mean) > 1.96 * uncertainty`
- confident: `abs(mean) >= 1.96 * uncertainty`
- critical: confident and (sign mismatch or outside interval)

A sign mismatch or an outside-interval case sets `validity_scope.status` to `CONTRADICTED`. Inflation doubles, capped at 2, only on a critical case. Reasons remain `CONFIDENT_CONTRADICTION`, `EVIDENCE_OUTSIDE_OR_SIGN`, and `PRECISION_WEIGHTED_MEAN`.

## I. Primary metrics

For each evaluation row, absolute error is `abs(observed_effect - predicted_effect)`.

The primary paired difference is no-update absolute error minus updated absolute error. Positive means the update is closer.

One `paired_mean_ci` over all 72 evaluation rows (12 prompts times 6 cells): 5000 draws, seed `23001`, percentiles 2.5 and 97.5, existing `ci_class`.

- interval entirely above zero: `UPDATE_IMPROVES`
- entirely below zero: `UPDATE_HURTS`
- otherwise: `INCONCLUSIVE`

The same contrast is reported per cell and is not used to drop a cell or to replace the pooled label.

Sign agreement is descriptive. A zero sign is reported and is not counted as agreement. Coverage `abs(error) <= 1.96 * uncertainty` is descriptive and is not a gate. The `1.96` is the inherited `CRITICAL_Z`, not a new cutoff.

## J. Control conditions

All four are frozen before evaluation and are predicted before evaluation outcomes exist.

1. no-update: the train belief for that cell, with validation evidence ignored. This is the persistence baseline.
2. constant-effect: one scalar, the mean of every train effect across all six cells and all twelve train prompts. The same number is predicted for every evaluation row. Cell identity is ignored.
3. outcome-only: the same pooled train mean as the constant-effect baseline. It is leakage-free because it uses no validation outcome, no evaluation outcome, and no mechanism id. It is reported separately so a match to the constant baseline is visible rather than treated as a second success.
4. shuffled-evidence: within each layer, D1's belief is updated with D2's validation effects, and D2's belief is updated with D1's validation effects, in `prompt_id` order, evidence ids prefixed with `shuffle-`. The shuffled belief then predicts that cell's own evaluation prompts.

If the shuffled control's paired interval against no-update is entirely above zero, a primary `UPDATE_IMPROVES` label is reported as `BASELINE_MATCH` and is not a cell-specific update claim.

## K. Leakage rules

- Evaluation prompt ids and texts are disjoint from train and validation.
- No old M22.1 prompt is in the new catalog.
- A cell's primary belief never reads another cell's outcomes. The shuffled control is the only cross-cell path, and it is labeled.
- `predict` has no outcome argument.
- Family, `pre_dot`, and baseline margin are logged only as measurements. They are not passed to `predict` and are not used to choose a cell.
- Evaluation outcomes are not used to add a cell, a prompt, a feature, an alpha, or a cutoff.
- Pairs marked `CONTAMINATED` or `PARTIALLY_CONTAMINATED` in the register are not treated as fresh.

## L. Stopping rule

One execution. When the evaluation outcome file exists, the analysis is the primary interval and the four controls. No cell, prompt, alpha, feature, or cutoff is added after that file exists. An interval that includes zero is the result. It is not a reason to change the family.

## M. Interpretation boundaries

`UPDATE_IMPROVES` means the updated cumulative mean had lower held-out absolute error than persistence on this family, and the shuffled control did not also beat persistence. It does not mean a verified mechanism, a discovered circuit, or a self-model.

`INCONCLUSIVE` includes the case where no validation observation contradicts the train belief. That case is reported. It is not repaired by picking a different cell.

`UPDATE_HURTS` means the update increased held-out absolute error. It does not mean context is irrelevant in general.

This document's readiness is a statement about the protocol. It is not evidence about Qwen.
