# Project feasibility gate

Status: `GATE_SPECIFIED`

This file is a design. It does not run Qwen, does not create an M24 result, and does not authorize a forward. A later execution is allowed only after a separate preregistration freezes a new prompt catalog under the rules in section 9. No threshold is estimated from outcomes.

The project should enter this gate. Self-model construction should not continue on the present evidence, and the project should not be killed before this gate is run. No genuine self-model object exists. M23-G is a belief tracker. The `pre_dot` test on `M22.1-D1-L15` is `INCONCLUSIVE`. Those facts leave feasibility open.

## 0. Current evidence the gate inherits

M19 has no scored result. The notebook of that name is a pointer to later M21 prototype cells. M20 has no standalone result. M20.6.3.2 forecast a synthetic stream and finished `FAIL`. Its observability flag is a layer-13 correlation gate, not a representation of an intervention. M21.2.4.3.3 authorizes a benchmark-conditional invalidity score. M21.2.4.3.4 is `CONDITIONAL_SUPPORT_LIMITED`. Neither is an activation-intervention representation.

M22.1 left `M22.1-D1-L23` as `CANDIDATE`. M22.1.3 recorded `FIRST_ORDER_SUPPORTED` and `READOUT_RECONSTRUCTION_EXACT` at that final site: the margin change is the first-order response of the final normalization and unembedding to the final residual perturbation. Prompt-level remainder there is small. M23-E measured an alpha ratio near `2.017` with a prompt range near `0.009`. M23-F marked that cell `CURRENT_INTERVENTION_INSUFFICIENT` for a falsification test.

M23 through M23-G tested a scalar mean per named cell. The scalar can be written down, contradicted, and updated. M23-G did not show that the update reduces held-out error. Twenty-six of 72 validation updates changed sign, so one mean does not determine the next prompt. The post-M23 gap audit called representation the unresolved bottleneck and did not identify which measurement carries the variation. The single-measurement execution then tested `pre_dot`, the state projection `h · d1`, on `M22.1-D1-L15`. Evaluation paired mean `0.002598781549420394`, interval `[-0.0019827815526613146, 0.006615873616569495]`, class `CI_INCLUDES_ZERO`, verdict `INCONCLUSIVE`. That observable stays a recorded inconclusive test. It is not the observable of this gate.

## 1. Exact scientific hypothesis

primary hypothesis: The first-order directional derivative of the yes/no margin along D1, evaluated at the pre-intervention residual, identifies held-out alpha-+1 response variation beyond the cell's scalar mean.

One sentence is the hypothesis. It is a claim about identification of response variation. It is not a claim that a self-model already exists.

## 2. What mechanism representation means operationally

primary representation hypothesis: The mechanism representation is the scalar g = dot(gradient of the yes/no margin with respect to the last-token residual, d1), and the issued prediction is g itself.

Operationally, a mechanism representation in this gate is an explicit scalar, computed before the scored outcome, whose definition is the directional derivative implied by the intervention `h <- h + alpha * d1`. The issued prediction equals that scalar. No coefficient is fit to turn a sensor into the effect.

Four categories stay separate.

| category | operational object | what a success would mean |
| --- | --- | --- |
| A. Scalar effect prediction | One number per cell, the train mean of alpha-+1 effects. The same prediction for every prompt in the cell. | The cell has a typical effect. |
| B. Response-conditioned prediction | A prediction that can differ across prompts because a pre-outcome measurement differs. The `pre_dot` slope was an instance. Its evaluation result was `INCONCLUSIVE`. | Some pre-outcome number tracks the effect. |
| C. Mechanism representation | Category B restricted to the derivative the intervention definition entails. Here the measurement is `g` and the prediction is `g`, with no fitted map. | The causal first-order account identifies the held-out effect. |
| D. Genuine self-model | Category C plus a scope that changes where the prediction applies, an evidence update, and a decision that consumes the prediction, with those pieces beating a non-mechanistic baseline under a frozen rule. | The system uses a self-model. |

This gate can speak to C. A pass does not reach D.

## 3. Minimum representation required

The object under test contains only:

- the cell id, hence the hook, D1, and alpha `+1`
- `g` for that prompt at that hook
- the prediction `g`
- the cell's train-effect mean, used solely as the baseline

No regime, no prompt embedding, no `pre_dot` term, no second derivative, and no learned map are part of the representation.

## 4. Why this representation is motivated

The intervention used throughout M22.1 and M23 is an additive residual step. For a differentiable margin, the first-order change under that step is alpha times `g`. M22.1.3 already found that this first-order account reconstructs the final-layer effect. At that final layer the derivative hardly varies across prompts, which is why a scalar mean was enough and why that cell cannot host a variation test.

M23-G showed that the same additive family at earlier preflight layers does vary in sign across prompts. The gap audit required a within-cell distinction and did not name it. The state projection `pre_dot` was then tested because it was the hook scalar recorded before the add. It did not clear the evaluation interval. The quantity the additive definition itself names is the derivative of the margin, not the projection of the residual onto `d1`.

No other observable is motivated by a recorded mechanistic finding. Selecting the next activation summary after `pre_dot` would be a search. This gate does not do that.

## 5. One primary experimental design

Score alpha `+1` effects of the predeclared family on a new catalog. For each cell, the baseline is that cell's train mean. The candidate is `g`. Predictions on every non-train split are written before that split's outcomes exist. The slope is not estimated. Validation cannot change the prediction rule. Evaluation and replication are the two decision splits. The orthogonal derivative is a specificity control, not a second candidate.

No feature fishing is performed.

## 6. Intervention family

intervention family: M22.1-D1-L0, M22.1-D1-L8, M22.1-D1-L15

| cell | hook | direction | alpha | why it is in the family |
| --- | --- | --- | --- | --- |
| `M22.1-D1-L0` | `blocks.0.hook_resid_post` | D1, seed `22101` | `+1` | Primary direction at a non-consumed preflight layer. |
| `M22.1-D1-L8` | `blocks.8.hook_resid_post` | D1, seed `22101` | `+1` | Same rule. |
| `M22.1-D1-L15` | `blocks.15.hook_resid_post` | D1, seed `22101` | `+1` | Same rule. |

The preflight grid was `{0, 8, 15, 23}`. Layer `23` is excluded because M22.1.3 and M23-E describe a shared, nearly prompt-invariant response there, and M23-F marked `M22.1-D1-L23` `CURRENT_INTERVENTION_INSUFFICIENT`. D2 is the orthogonal control direction, seed `22103`. D2 cells are not members of the family. Their effects are not the outcome. No cell is dropped or added after outcomes. Ranges, sign counts, and secondary intervals are not selection inputs.

Outcome: intervened yes/no logit margin minus the unintervened margin. Token ids must be `9834` and `902` or the run aborts. Model pin, if executed later: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

## 7. Context and state requirements

The state is the pre-intervention last-token residual at the cell's own hook. `g` is computed from that residual by backpropagation of the margin. Surface family may balance the catalog. It is not a state variable and not a model input. No temporal history and no cross-layer graph are state in this gate.

## 8. Pre-intervention observable

predeclared observable: g

Computation, frozen:

1. Run the unintervened model on the prompt.
2. At `blocks.{layer}.hook_resid_post`, retain the last-token residual `h`.
3. Let the margin be the last-position logit of token `9834` minus the logit of token `902`.
4. Backpropagate that margin to `h`.
5. `g = dot(gradient, d1)`.
6. This pass does not apply alpha `+1` and does not read the alpha-+1 effect.

`g` is a scalar. There is no epsilon grid and no finite-difference search. The closed-form final-layer Jacobian from M22.1.3 is not substituted at layers 0, 8, or 15, because those residuals are not the input to the final unembedding.

The alpha-+1 outcome is a later forward. Predictions for validation are written before validation outcomes exist. Predictions for evaluation are written before evaluation outcomes exist. Predictions for replication are written before replication outcomes exist.

## 9. Train, validation, evaluation, and replication

A new catalog, ids prefixed `f-`, must be frozen in a later preregistration before any forward. Texts and ids must be disjoint from the M22.1 catalog, the `g-*` catalog, and the `s-*` catalog. This file does not contain those texts. Inventing them after an outcome exists is a protocol break.

Within each surface family, index `k` starting at 1 is assigned by `(k - 1) mod 4`:

| remainder | partition |
| --- | --- |
| 0 | train |
| 1 | validation |
| 2 | evaluation |
| 3 | replication |

Each partition has 12 prompts, four from each surface family. Twelve is the M23-G partition size. It is not a size chosen from the `pre_dot` interval. The four partitions are fixed before outcomes. Evaluation is not the replication set.

Train supplies only the three scalar means, one per cell. Validation is scored and cannot change `g`, the means, or the decision. Evaluation is the primary split. Replication repeats the same frozen rule on prompts that were not in train, validation, or evaluation.

## 10. Controls

Scalar baseline, category A: for each cell, `m` is the mean of that cell's twelve train effects. Every prompt in the cell receives `m` on every split. Means are not pooled across layers.

Specificity control: from the same gradient, `g_orth = dot(gradient, d2)`. The control prediction is `g_orth`. It is scored against the same `m` on the D1 effects. It is not a second representation hypothesis. If the control wins, the D1 derivative has not been shown to be the specific carrier.

The candidate has no fitted slope. A train-only regression of the effect on `g` is forbidden.

## 11. Leakage prevention

- `g` and `g_orth` are functions of the unintervened residual, D1, and D2.
- The alpha-+1 effect is not an input to either number.
- Train effects are used only to compute the three means.
- Non-train prediction files contain `g`, `g_orth`, `m`, and the two predictions, and they do not contain an observed effect.
- Those files are closed before the matching outcome file exists.
- No `g-*` or `s-*` or M22.1 id is scored.
- No cell outside the family is scored.
- Validation outcomes are not written back into `m` or into the definition of `g`.

## 12. Primary metric

For each cell and prompt, `difference = error(baseline) - error(candidate)`, with errors the absolute deviations of the alpha-+1 effect from `m` and from `g`.

Prompts are the resampling units. For each prompt, average its three cell differences. Apply `paired_mean_ci` from `src/cognitive_self_model/m23/stats.py` to those prompt scores: 5000 draws, seed `23001`, percentile indexes `floor(0.025 * (draws - 1))` and `ceil(0.975 * (draws - 1))`. The same aggregation and the same function are used for the control, with `g_orth` in place of `g`.

The sign co-criterion uses `clopper_pearson` from that module. Inside one cell and one split, count strict sign agreements between the prediction and the effect. A split passes the sign co-criterion only if every family cell has a lower bound above `0.5`. The bound is the M23 Q1 sign rule. It is not refit.

## 13. Exact PASS criterion

`FEASIBILITY_CONTINUE` requires all of the following:

- evaluation class of the candidate is `CI_POSITIVE`
- replication class of the candidate is `CI_POSITIVE`
- evaluation class of the control is not `CI_POSITIVE`
- replication class of the control is not `CI_POSITIVE`
- evaluation passes the sign co-criterion
- replication passes the sign co-criterion

## 14. Exact FAIL/KILL criterion

`FEASIBILITY_KILL` requires both of the following:

- evaluation class of the candidate is `CI_NEGATIVE`
- replication class of the candidate is not `CI_POSITIVE`

A reliable loss to the scalar mean on the primary split, without a replication win, kills this representation hypothesis.

## 15. Exact INCONCLUSIVE criterion

Every other combination of the recorded classes and sign flags is `INCONCLUSIVE`. In particular:

- one split `CI_POSITIVE` and the other not is `INCONCLUSIVE`
- a candidate win that the orthogonal control also achieves is `INCONCLUSIVE`
- a candidate win without the sign co-criterion is `INCONCLUSIVE`
- evaluation `CI_NEGATIVE` together with replication `CI_POSITIVE` is `INCONCLUSIVE`
- both decision splits `CI_INCLUDES_ZERO` is `INCONCLUSIVE`

`scripts/validate_project_feasibility_gate.py` implements this map and checks these cases.

## 16. Replication requirement

Replication is a fourth partition, frozen with the catalog, absent from the fit, and scored with the train means and the same `g` rule. A `FEASIBILITY_CONTINUE` result is impossible without a replication class of `CI_POSITIVE` and a replication sign pass. An evaluation win alone is the pattern already seen when a validation interval cleared zero and the evaluation interval did not. That pattern is not a pass here.

## 17. What would count as evidence of feasibility

`FEASIBILITY_CONTINUE` would be evidence that this pre-intervention representation carries held-out information about intervention-response variation beyond the scalar cell mean, on two disjoint partitions, and that the orthogonal direction's derivative does not do so under the same rule. That is evidence that a mechanism-level response representation is empirically available for this family. It is evidence for category C in this benchmark.

## 18. What would count as evidence against feasibility

`FEASIBILITY_KILL` would be evidence that `g`, as defined here, increases held-out absolute error relative to the scalar mean on the evaluation partition, and that the replication partition does not show the opposite win. The conclusion would be about this observable and this intervention family. The tested first-order representation would have failed to identify response variation under clean held-out conditions.

If this feasibility gate fails, the conclusion is that the project's recorded first-order account does not identify prompt-level intervention response for D1 at layers 0, 8, and 15, beyond a scalar mean, on the frozen catalog and rule. Self-models in general would not have been shown impossible. A different intervention family, a different model, or a different predeclared causal hypothesis could still be feasible. This catalog could not be reused to shop for that hypothesis.

## 19. What a PASS would still not be

A `FEASIBILITY_CONTINUE` result would still not be a genuine self-model. It would not show a scope that changes applicability, an update driven by evidence, a decision that consumes the prediction, transfer to another direction, transfer to another layer, transfer to another model, or a causal graph. It would not reinterpret M22.1 as `VALIDATED_FOR_M22`. It would not convert the `pre_dot` result into a success.

## 20. Architecture work justified only after PASS

Only after `FEASIBILITY_CONTINUE` is it justified to build an explicit object that stores the cell, the rule "prediction = g", the train mean as a baseline, and a residual scale from train pairs, and then to preregister a separate test in which a decision reads that object. That later test would be the first setting in which category D could be asked.

Before a pass, the following stay unjustified: a new update rule, a discovery search, a governance stack, a probe bank, a nonlinear readout, a second observable, and any change to `SelfModelBelief` that adds inputs.

## If the gate passes, what becomes justified

Continuing toward construction of a mechanism-level self-model object whose response map is this derivative, followed by a separate consumption test. The self-model would not already have been achieved.

## If the gate is inconclusive

Do not call `g` useless. Do not start a second observable. Do not drop a layer. Do not refit a slope on these outcomes. Feasibility remains undecided, and construction remains unjustified. A valid inconclusive execution closes this hypothesis on this catalog.

## Stopping

One execution of this gate. The decision function in section 13 through section 15 is the whole decision. No backup observable is waiting.
