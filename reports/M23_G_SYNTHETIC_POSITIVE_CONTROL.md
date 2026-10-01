# M23-G synthetic positive control

This is a harness check. It does not count as evidence for a Qwen self-model. It does not load the model and it does not use an inspected outcome.

## Specification

The interface is the existing `belief_from_observations`, `predict`, and `update_belief`. Nothing in `SelfModelBelief` is changed.

Initial observations: `1.0`, `1.0`. Scope status `IN_SCOPE`.

The pre-intervention prediction at alpha `+1` is `predicted_effect = 1`, `predicted_direction = +1`, `uncertainty = 0`. That prediction is kept on the original object.

Validation evidence: `-1.0`. The stored mean and the evidence have opposite strict signs, and the mean is at least `CRITICAL_Z` times the uncertainty, so the existing rule records `CONFIDENT_CONTRADICTION` and sets status `CONTRADICTED`.

The updated prediction is the mean of `1`, `1`, and `-1`, which is `1/3`. It is not `-1`. The original belief still predicts `1`.

Held-out synthetic targets: `-1`, `-1`.

- no-update absolute error: `2`
- updated absolute error: `4/3`
- improvement: `2/3`, which is greater than zero

Shuffled evidence on a fresh copy of the initial belief: `+1.0`. That evidence agrees with the train mean. Its prediction on a target of `-1` stays farther from the target than the contradicted update.

A failure of any of these checks means the protocol is not ready. Passing them means the harness can see a contradiction, refuse to copy the observation, and record a held-out improvement when the later cases match the contradicting evidence.

## Harness result

`HARNESS_PASS`

`scripts/validate_m23_g_protocol.py` ran the specification on the existing update function. Initial prediction `1`. Updated prediction `0.3333333333333333`, which is the cumulative mean and not the observed `-1`. Held-out improvement of no-update absolute error over updated absolute error: `0.6666666666666667`. The pre-intervention belief object was unchanged. The agreeing shuffled evidence stayed farther from the negative holdout than the contradicted update.
