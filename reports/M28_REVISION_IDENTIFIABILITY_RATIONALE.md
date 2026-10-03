# M28 scientific rationale

Gate status: `NO_SUITABLE_PRE_EVIDENCE`

M28 asks whether a measurement available before the evidence-generating intervention can identify the direction or magnitude of the future residual `observed_effect - g` strongly enough to justify a mechanism revision. The frozen protocol is `reports/M28_REVISION_IDENTIFIABILITY_PROTOCOL.md`. This file records why that protocol stops before catalog generation and before any model run.

## Question the stored record can answer

The consumed mechanism is `prediction = g`, with `g = dot(margin_gradient, d1)` from `PreInterventionState`. The residual is the later intervention outcome minus that prediction. A useful pre-evidence variable would change a residual prediction before the holdout outcome exists, and that change would have to beat a constant residual, a cell-mean residual, and a shuffle of the same update pairing.

The stored M26 and M27 artifacts were inspected for that variable. Column names and prediction-file keys were read. No new association was fit to the M27 holdout. The published M26 classification and the published M27 gain result were used as they stand.

## What was sealed before the outcome

M26 `update_predictions.json` and M27 `update_predictions.json` close before the matching outcome files. Their varying fields are `g`, `prediction_before` or `prediction` (the same number as `g`), `family`, and the cell label (`intervention_id`, `cell`, `mechanism_id`, `layer`, `hook`). `baseline_prediction` and `initial_gain` repeat a per-cell or global constant. `alpha`, `direction_id`, `response_rule`, and `response_form` are constants.

`baseline_output` is the clean yes/no margin. In both runners it is written in the outcome file together with `observed_effect`. `pre_dot` is computed inside the intervention hook and is absent from the M26 and M27 episode files. The margin-gradient vector is the in-memory argument of `MechanismResponseModel.predict`. The episode files retain `g` and discard the vector. Prompt length, gradient norm, and an orthogonal derivative are absent from the sealed schema.

## Why each sealed field fails independence

`g` is the mechanism output. M27 already asked whether update evidence moves the coefficient in front of `g`. Every cell interval contained `1`, the consumed gain stayed `1`, and the holdout comparison with the initial mechanism was identically zero. A regression of `observed_effect - g` on `g` is that same gain question. Repeating it would rerun M27.

`family` was sealed before outcomes. The M26 gap audit scored surface-family residual means on the update partition and on the holdout partition. The orders did not agree, and the audit classified the stored pattern `STRUCTURED_BUT_UNSTABLE`. Naming `family` now as the primary candidate uses a holdout association that has already been published.

Cell identity is the cell-mean baseline required by the comparison, and it is the pattern the M26 audit found on the update partition and lost on the holdout. It is a baseline, and it is already scored.

`baseline_output` is causally the clean forward pass. The M26 gap audit associated it with the residual inside each cell on both the update partition and the holdout partition. The audit's recommendation states that those residuals do not identify a recorded variable whose association returns on the untouched holdout, and that a later design would need its own predeclared measurement. Selecting `baseline_output` after that publication is a choice made with the holdout in view. Its stored copy also sits in the outcome file.

Outcome fields, residuals, revised gains, and M26 or M27 holdout rows are excluded from the candidate set by the question itself. They are consequences of the evidence or of a fit to that evidence.

## Gate

The eligible primary-candidate set is empty. A new catalog would have no pre-registered measurement to carry, so generating prompts would invent an experiment around a feature that the stored record does not supply. The execution status is `NO_SUITABLE_PRE_EVIDENCE`.

This closes the claim that the measurements already stored before the intervention outcome identify when `g` is wrong. It leaves the M26 and M27 verdicts unchanged. A later protocol can name one new measurement only by sealing that measurement before any outcome and by using a catalog disjoint from the catalogs already scored. This protocol does not name that measurement, does not define a revision operator, and does not authorize a model run.
