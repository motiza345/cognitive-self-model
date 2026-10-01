# Post-M23 single-measurement design

Protocol status: `READY_FOR_EXECUTION`

That status means the cell, the measurement, the catalog, the partition, the metric, and the synthetic harness are specified. It is not evidence about Qwen. No Qwen forward was run for this design. This file does not create an M24 result.

## Cell

The tested cell is `M22.1-D1-L15`.

The selection rule is in the preregistration. It uses the preflight grid `{0, 8, 15, 23}`, excludes the consumed layer `23`, keeps the primary direction D1, and takes the grid neighbor of layer `23`. That neighbor is layer `15`.

M23-G secondary intervals, M23-F ranges, and sign counts are not terms in that rule.

## Measurement

The tested measurement is `pre_dot`.

It is the last-token residual at `blocks.15.hook_resid_post`, dotted with D1, on a forward that has not added the direction. The predictor receives that scalar and no other varying input. The train scalar mean remains the baseline, and the candidate adds one train-only slope in the centered measurement.

## Partition

The new `s-*` catalog has 12 train, 12 validation, and 12 evaluation prompts, with four prompts from each surface family in each partition. Family balances the catalog. The fit does not receive it.

Train fits the mean and the slope. Validation and evaluation predictions are written from that fit before their outcomes exist. The primary decision is the evaluation interval only.

## Decision

`paired_mean_ci` on `error(baseline) - error(candidate)`, seed `23001`, 5000 draws. The labels are `PREDICTIVE_INFORMATION_SUPPORTED`, `MEASUREMENT_HURTS`, and `INCONCLUSIVE`.

## What execution must not do

Execution is not part of this task. When it happens, it must not open a second cell, score a second measurement, score alpha `+2`, fit on `g-*` or M22.1 prompts, or edit `SelfModelBelief`.

## Unresolved risks

- Twelve evaluation prompts can leave a small slope inside an interval that includes zero.
- The slope is linear and centered. A non-linear association can yield `INCONCLUSIVE` or `MEASUREMENT_HURTS` while the raw measurement still varies with the effect.
- If every train measurement is equal, the slope is defined to be zero and the candidate matches the baseline.
- The dot uses D1. An association can be alignment with that fixed vector. This protocol has no second contrast that separates alignment from a causal path.
- M23-G already stored `pre_dot` for this cell on the `g-*` prompts. Those rows are contaminated. They are not the training set.
- Repeats are not in the protocol, so a noise floor is not estimated.
- The historical layer-23 correlations for this same measurement are a different cell. They are not used to set `b`, and they do not predict the layer-15 interval.
