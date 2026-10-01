# Post-M23 single-measurement preregistration

This protocol is a design. It does not run Qwen, does not read a new outcome, and does not create an M24 result.

Scientific question: does the single pre-outcome measurement reduce held-out absolute prediction error relative to the cell's scalar mean, under the existing paired-interval rule?

tested cell: M22.1-D1-L15

tested measurement: pre_dot

tested alpha: 1.0

Exactly one intervention cell is tested. Exactly one pre-outcome measurement is tested. No feature selection is performed. No threshold is chosen after outcomes exist.

## 1. What this test is

The baseline prediction is the scalar mean of the train effects for the tested cell. The candidate prediction is that same mean plus a train-only linear deviation in the one measurement. The primary comparison is held-out absolute error.

The measurement is a pre-outcome number. This protocol does not assign it a causal role. A later experiment would still have to separate a superficial association from a causal account if the candidate wins.

## 2. Cell selection

The cell is fixed by the following rule, and by no other rule.

1. The M22.1 candidate layers were declared from the model depth as `{0, 8, 15, 23}`.
2. Layer `23` is the consumed primary site. M23-F records D1 at that layer as `CURRENT_INTERVENTION_INSUFFICIENT`. It is excluded.
3. D1, seed `22101`, is the primary direction. D2, seed `22103`, is the orthogonal control. The control direction is not the tested cell.
4. Among the remaining primary-direction layers `{0, 8, 15}`, the tested layer is the neighbor of the consumed site on that grid. The neighbor of `23` is `15`.

The rule does not use an effect range, a sign count, a correlation, an MAE, or an M23-G secondary interval. Those quantities are not inputs.

Hook: `blocks.15.hook_resid_post`. Position: last token only. Direction: unit Gaussian, seed `22101`, the existing D1 vector. Recorded sha256 `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`. Alpha is `+1` only. Outcome is the intervened yes/no logit margin minus the baseline margin. Token ids must be `9834` and `902` or the run aborts. Model pin, if a later execution occurs: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, float32, `prepend_bos` true.

## 3. The one measurement

Name: `pre_dot`.

Source: the last-token residual at `blocks.15.hook_resid_post` on the unintervened forward for that prompt.

Computation: `pre_dot = dot(h_last, d1)`, using the same D1 vector as the intervention. This is the scalar already recorded by `make_resid_hook` before `apply_last_token_additive`. The protocol uses that definition. It does not take the dot after the add.

Type: scalar. No vector is passed to the predictor. No further reduction is applied.

When: on a forward that does not add the direction, and before any intervention effect for that prompt is written. The measurement forward does not read the intervention effect.

Allowed inputs: the last-token residual at layer 15, and the predeclared D1 vector.

Forbidden inputs: the residual after the add, any other layer, any other direction, logits, the baseline margin, the surface-family label, the prompt string as a regression column, validation effects, evaluation effects, and any coefficient fit outside the train split.

Prompt text: the string is not a column in the fit. The residual depends on the prompt because the frozen model is run on that string.

Intervention identity: the dot uses the tested cell's D1 vector, which is fixed before outcomes. It does not read a menu of other interventions, and it does not read the realized effect.

Post-intervention activity: the measurement forward does not add the direction. A dot taken on a residual that already contains the alpha step is not this measurement.

## 4. Predictors

Let the train rows be `(x_i, y_i)`, with `x` the measurement and `y` the alpha-`+1` effect.

- `m` = mean of the train effects.
- `x_bar` = mean of the train measurements.
- `v` = sum of `(x_i - x_bar)^2`.
- If `v = 0`, the slope `b = 0`.
- If `v > 0`, `b` = sum of `(x_i - x_bar) * (y_i - m)` divided by `v`.

Baseline prediction, for every prompt: `m`.

Candidate prediction: `m + b * (x - x_bar)`.

At `x = x_bar` the candidate equals the baseline. Train outcomes are used to fit `m` and `b`. Validation outcomes are not. Evaluation outcomes are not. The surface family is not a column. No second slope is fit. The degeneracy rule above is the only branch, and it is fixed before outcomes.

`SelfModelBelief` is not used and is not edited. This formula is not an update rule.

## 5. Catalog and partition

Thirty-six new prompts. Identifiers use the prefix `s-`. None of the eighteen M22.1 texts or ids, and none of the thirty-six `g-*` texts or ids, are reused. Surface family is a balance label. It is not a predictor input.

Within each family, index `k` starting at 1 is assigned by `(k - 1) mod 3`: `0` train, `1` validation, `2` evaluation. Each partition has 12 prompts, four from each family.

| prompt_id | family | partition | text |
| --- | --- | --- | --- |
| `s-completion-01` | completion | train | The largest moon of Earth is |
| `s-completion-02` | completion | validation | A spider has eight |
| `s-completion-03` | completion | evaluation | The composer of the Ninth Symphony is |
| `s-completion-04` | completion | train | Steam is the gas form of |
| `s-completion-05` | completion | validation | The Atlantic is an |
| `s-completion-06` | completion | evaluation | A square has four |
| `s-completion-07` | completion | train | The currency of the United Kingdom is the |
| `s-completion-08` | completion | validation | Pottery is usually fired in a |
| `s-completion-09` | completion | evaluation | The fastest land animal is the |
| `s-completion-10` | completion | train | A day has twenty-four |
| `s-completion-11` | completion | validation | The chemical symbol for iron is |
| `s-completion-12` | completion | evaluation | Silk comes from a |
| `s-syntax-01` | syntax | train | After the kettle boiled, |
| `s-syntax-02` | syntax | validation | If the window stays open, |
| `s-syntax-03` | syntax | evaluation | Before the guests arrived, |
| `s-syntax-04` | syntax | train | Although the road was icy, |
| `s-syntax-05` | syntax | validation | Because the ink had dried, |
| `s-syntax-06` | syntax | evaluation | Unless the key is turned, |
| `s-syntax-07` | syntax | train | While the choir was singing, |
| `s-syntax-08` | syntax | validation | As soon as the curtain rose, |
| `s-syntax-09` | syntax | evaluation | Even though the cup was cracked, |
| `s-syntax-10` | syntax | train | Whenever the clock strikes, |
| `s-syntax-11` | syntax | validation | Since the market was empty, |
| `s-syntax-12` | syntax | evaluation | Until the rain stopped, |
| `s-instruction-01` | instruction | train | Reply with one word. A fabric made from flax: |
| `s-instruction-02` | instruction | validation | Reply with one word. The meal eaten in the evening: |
| `s-instruction-03` | instruction | evaluation | Name a tool used for cutting paper: |
| `s-instruction-04` | instruction | train | Name a day that starts the weekend: |
| `s-instruction-05` | instruction | validation | Reply with one word. Opposite of heavy: |
| `s-instruction-06` | instruction | evaluation | Name a kind of flower: |
| `s-instruction-07` | instruction | train | Reply with one word. A frozen drink: |
| `s-instruction-08` | instruction | validation | Name a string instrument: |
| `s-instruction-09` | instruction | evaluation | Reply with one word. The number of sides on a hexagon: |
| `s-instruction-10` | instruction | train | Name an ocean: |
| `s-instruction-11` | instruction | validation | Reply with one word. A mammal that lays eggs: |
| `s-instruction-12` | instruction | evaluation | Name a piece of clothing: |

Twelve per partition copies the M23-G partition size. It is not a sample size chosen from an observed interval.

## 6. Order

1. Train measurement forwards, then train intervention outcomes. Fit `m`, `x_bar`, and `b` from those twelve pairs. Write the coefficients.
2. Validation measurement forwards and evaluation measurement forwards. These forwards record `pre_dot` and do not write effects.
3. Write the validation prediction file and the evaluation prediction file from the train coefficients and those measurements. Each row contains `prompt_id`, `pre_dot`, the baseline prediction, and the candidate prediction. Neither file contains an observed effect.
4. Predictions for validation are written before validation outcomes exist. Predictions for evaluation are written before evaluation outcomes exist. Both prediction files are closed before either held-out outcome file is written.
5. Validation intervention outcomes, then evaluation intervention outcomes.
6. Score the evaluation split for the primary decision. Score the validation split only as a secondary interval after the prediction files exist.

The candidate number for a held-out prompt is a function of `m`, `b`, `x_bar`, and that prompt's `pre_dot`. The effect is not an argument.

## 7. Primary metric and decision

On the twelve evaluation prompts, for each prompt,

`difference = error(baseline) - error(candidate)`

where each error is the absolute deviation of the observed effect from that prediction. Positive means the candidate is closer.

The interval is `paired_mean_ci` from `src/cognitive_self_model/m23/stats.py`: 5000 draws, seed `23001`, percentile indexes `floor(0.025 * (draws - 1))` and `ceil(0.975 * (draws - 1))`. The class is `CI_POSITIVE` if the low end is above 0, `CI_NEGATIVE` if the high end is below 0, and `CI_INCLUDES_ZERO` otherwise.

| class | decision |
| --- | --- |
| `CI_POSITIVE` | `PREDICTIVE_INFORMATION_SUPPORTED` |
| `CI_NEGATIVE` | `MEASUREMENT_HURTS` |
| `CI_INCLUDES_ZERO` | `INCONCLUSIVE` |

The validation interval uses the same function and is not a decision. It cannot replace the evaluation class, change `b`, or admit another measurement.

## 8. Leakage audit for a later execution

A later run is consistent with this protocol only if all of the following hold.

- The coefficient file is a function of the twelve train pairs.
- The evaluation prediction file has no `observed_effect` field and is closed before the evaluation outcome file exists.
- The same is true of the validation prediction file.
- Train, validation, and evaluation prompt ids are disjoint.
- The design matrix has the centered measurement as its only varying column.
- Alpha stays `+1`. The hook stays `blocks.15.hook_resid_post`. The direction stays D1.
- No `g-*` id and no M22.1 id is scored.
- The secondary validation interval is not written back into the coefficients.

## 9. Stopping rule

One execution. When the evaluation outcome file exists, the analysis is the one primary interval. No cell, prompt, alpha, measurement, slope form, or cutoff is added after that file exists. An interval that includes zero is the result. It is not a reason to try a different measurement on these prompts.

## 10. Interpretation

`PREDICTIVE_INFORMATION_SUPPORTED` means this centered linear use of `pre_dot` had lower held-out absolute error than the train mean on these twelve prompts, and the paired interval lay above zero. It establishes predictive information for this cell, this alpha, this catalog, and this split. It does not establish a causal path, a circuit, a scope, a self-model, or a claim about any other measurement.

`MEASUREMENT_HURTS` means the same linear use increased held-out absolute error, with the interval below zero.

`INCONCLUSIVE` means the interval includes zero. The test did not show that this measurement reduces held-out absolute error. It does not prove the measurement is independent of the effect. It does not authorize a second measurement, a second cell, or a refit on this catalog.

## 11. Synthetic harness

`reports/POST_M23_SINGLE_MEASUREMENT_SYNTHETIC_CONTROL.md` runs the same two prediction functions on planted numbers. That check is part of protocol readiness. It is not an observation of the model.
