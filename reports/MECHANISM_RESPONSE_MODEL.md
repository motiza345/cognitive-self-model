# Mechanism response model

This file records the object justified by category C. It does not change the feasibility gate, does not add a cell, and does not define a consumption test.

## Why category C is justified

Commit `7aa64543fa45b0e09bb7e247641ce8c926f8a66d` records verdict `FEASIBILITY_CONTINUE` for one frozen rule: the prediction is `g`, the directional derivative of the yes/no margin along D1 at the pre-intervention residual. No slope was fit.

On the family `M22.1-D1-L0`, `M22.1-D1-L8`, and `M22.1-D1-L15`, that prediction beat the per-cell train mean on both decision splits. Evaluation paired mean `0.039940172547681466`, interval `[0.0267728113103658, 0.0560522271854872]`, class `CI_POSITIVE`. Replication paired mean `0.04276723200800242`, interval `[0.02623499902310195, 0.061806415224930765]`, class `CI_POSITIVE`. The orthogonal D2 derivative was `CI_NEGATIVE` on both splits. The Clopper-Pearson sign bound passed in every cell on both splits. `leakage_ok = true`.

Recorded train means, stored as baselines: layer 0 `0.02711375554402669`, layer 8 `0.02379457155863444`, layer 15 `-0.0065801143646240234`.

## What the object represents

`MechanismResponseModel` is one cell of that family. It stores the intervention id, the layer and hook, D1 provenance (id, seed `22101`, sha256, and the unit vector), alpha `+1`, the rule `prediction = g`, the training scalar baseline mean, and the experiment reference `PROJECT_FEASIBILITY_GATE` at the commit above.

`predict(pre_intervention_state)` returns `g = dot(margin_gradient, d1)`. The state is the gradient of the yes/no margin with respect to the last-token residual, evaluated before the add. The margin is last-position logit `9834` minus logit `902`. The baseline mean is stored and is not an input to `predict`.

The three cells are built by `frozen_mechanism_response_models()`. Their means and family membership are read from `reports/PROJECT_FEASIBILITY_GATE_RESULTS.json`. D1 is the existing `primary_direction(896, 22101)` vector, checked with `vector_sha256`.

## What this object does not establish

The object is a mechanism response rule for this family. A validated self-model is a later claim and is not made here. The object has no scope test, no evidence update, no decision that consumes `g`, no transfer claim, and no reinterpretation of M22.1. The `pre_dot` result remains `INCONCLUSIVE`. No residual scale is estimated. No consumption experiment is specified in this file.
