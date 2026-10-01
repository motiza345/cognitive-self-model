# Post-M23 single-measurement execution

Protocol status remains `READY_FOR_EXECUTION`. This file does not change that status.

Verdict: `INCONCLUSIVE`

The verdict uses the evaluation interval only. A supported result means `pre_dot` carries predictive information about the response of `M22.1-D1-L15` on this catalog. It does not establish a mechanism representation, causal understanding, a self-model, or generalization across mechanisms, layers, or interventions.

## Train fit

- n: `12`
- effect mean `m`: `0.009110331535339355`
- pre_dot mean: `0.5010738844672838`
- slope `b`: `0.017036139867335046`
- pre_dot sample sd: `0.700183329331844`
- pre_dot min: `-0.25899428129196167`
- pre_dot max: `2.4336299896240234`
- pre_dot sum of squared deviations: `5.392823641416481`

The slope was fit on these twelve train pairs and was not refit.

## Evaluation

- baseline MAE: `0.060514469941457115`
- candidate MAE: `0.05791568839203672`
- paired mean: `0.002598781549420394`
- interval: `[-0.0019827815526613146, 0.006615873616569495]`
- class: `CI_INCLUDES_ZERO`
- draws: `5000`, seed: `23001`

## Validation

Validation is not the decision. The train fit was kept.

- baseline MAE: `0.03475358088811239`
- candidate MAE: `0.031393316918930235`
- paired mean: `0.003360263969182155`
- interval: `[0.00047484457375104714, 0.006428762181473895]`
- class: `CI_POSITIVE`

## Per family

These four-prompt intervals are secondary. They do not change the verdict.

| family | n | baseline MAE | candidate MAE | paired mean | class |
| --- | ---: | ---: | ---: | ---: | --- |
| completion | 4 | 0.060830891132354736 | 0.06431934165891724 | -0.003488450526562502 | `CI_INCLUDES_ZERO` |
| instruction | 4 | 0.08645355701446533 | 0.0845682517755497 | 0.001885305238915621 | `CI_INCLUDES_ZERO` |
| syntax | 4 | 0.03425896167755127 | 0.024859471741643206 | 0.009399489935908063 | `CI_POSITIVE` |

## Leakage

`leakage_ok = true`

Validation and evaluation predictions were written after the train fit and before either held-out outcome file. Those prediction files contain `pre_dot` and the two predictions. They do not contain an observed effect. No `g-*` prompt was scored.

## Execution notes

Maximum absolute difference between the unintervened-forward `pre_dot` and the pre-add dot inside the later intervention hook: `0.0`.
The predictor used the unintervened-forward value. The hook dot was an audit quantity and was not a second measurement.

The interval includes zero. This test does not show that pre_dot reduces held-out absolute error relative to the train mean. The measurement is not declared useless. The uncertainty is the reported interval.
