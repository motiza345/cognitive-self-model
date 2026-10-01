# M23-D2 report

Historical M23 remains `INCONCLUSIVE`. The earlier diagnostic labels are unchanged. This comparison does not revise them.

## Result

`INCONCLUSIVE`

Mean paired difference (`abs_error_A - abs_error_B`): `0`

Interval, 5000 paired bootstrap draws, seed `23001`, percentiles 2.5 and 97.5:

`[-0.0008342530992296003, 0.0008342530992296003]`

Class: `CI_INCLUDES_ZERO`

Interval width: `0.0016685061984592006`

| quantity | value |
| --- | ---: |
| n | 6 |
| mean absolute error, constant | 0.003130753835042318 |
| mean absolute error, regime mean | 0.003130753835042318 |
| effect sample variance | 1.2905464397287384e-05 |
| regime-prediction sample variance | 1.2538040361202245e-06 |
| constant-prediction variance | 0 |

The six paired differences are `+0.0007721583048502616`, `-0.0007721583048502616`, `-0.0006722609202067069`, `+0.0006722609202067069`, `-0.001444419225056965`, `+0.001444419225056965`. Within each regime they cancel. Both the constant and the regime mean sit between the two held-out effects of that regime. For two points, every prediction between them has the same total absolute error, so this sample cannot show a gain from moving the prediction inside the pair.

## Per regime

Train n is 2 and evaluation n is 2 in every regime.

| regime | prediction_B | held-out effects | held-out mean |
| --- | ---: | --- | ---: |
| completion | 0.028961896896362305 | 0.030832290649414062, 0.025826454162597656 | 0.02832937240600586 |
| instruction | 0.02886199951171875 | 0.026548385620117188, 0.03369140625 | 0.030119895935058594 |
| syntax | 0.026745319366455078 | 0.03156852722167969, 0.024932861328125 | 0.028250694274902344 |

## Leakage

`leakage_ok = true`

Predictions were written before the evaluation outcome file existed. Predictor A is one scalar. Predictor B is constant within each regime and differs only by the pre-specified regime. Replication prompt ids are absent. Prediction rows have no observed effect. Evaluation ids match the frozen manifest. No activation, `pre_dot`, baseline margin, or prompt embedding was used.

The training regime means had already been printed in the M23 diagnostic report. That limits the blindness of the training estimates. It does not put evaluation outcomes into the predictor: those per-prompt discovery effects were not saved before this run.

## Interpretation

No evidence was obtained in this experiment that regime information improves held-out prediction of the tested D1 intervention effect.

This is not a finding that context is irrelevant in general. It is the result of one constant-versus-regime comparison on six held-out prompts.

## Remaining uncertainty

The evaluation has two prompts per regime. The effect sample variance is about ten times the variance of the regime predictions, and the interval is wider than the spread among those predictions. The unresolved measurement-noise question from the diagnostic is untouched, because this task did not repeat any prompt. A three-way split was not used: the untouched pool could not fill train, calibration, and evaluation without emptying a regime.

## Another diagnostic

This result does not justify another regime-versus-constant diagnostic on the same catalog. The replication prompts were already inspected, and the discovery prompts are now the evaluation set. The present design is also structurally unable to credit a regime mean that stays inside each held-out pair. A larger comparison would be a new experiment, and it is not part of this task.
