# M24 consumption execution

The protocol in `reports/M24_CONSUMPTION_PROTOCOL.md` is unchanged.

Verdict: `CONSUMPTION_SUPPORTED`

On the frozen evaluation prompts, the consumed derivative reduced absolute error relative to the stored cell mean, the sign decision did not lose to that mean, and the shuffled predictions did not clear the same interval. This is a consumption result for the frozen object on this catalog. It is not a self-model.

## Evaluation

- baseline MAE: `0.06271952169912832`
- mechanism MAE: `0.005885519992307424`
- paired mean: `0.0568340017068209`
- interval: `[0.03877295034976836, 0.07716522058950821]`
- class: `CI_POSITIVE`
- shuffled class: `CI_INCLUDES_ZERO`
- shuffled paired mean: `-0.019247650520564406`
- sign decision does not lose: `True`
- leakage_ok: `True`

## Validation

Validation is not an input to the verdict.

- class: `CI_POSITIVE`
- paired mean: `0.03384582080049758`
- interval: `[0.0255714591770716, 0.042297326068132045]`

## Per-cell evaluation

| cell | paired mean | class | baseline MAE | mechanism MAE |
| --- | ---: | --- | ---: | ---: |
| `M22.1-D1-L0` | 0.06077664711320482 | `CI_POSITIVE` | 0.0730466710196601 | 0.01227002390645529 |
| `M22.1-D1-L8` | 0.05423288841456803 | `CI_POSITIVE` | 0.05714160866207547 | 0.0029087202475074326 |
| `M22.1-D1-L15` | 0.055492469592689865 | `CI_POSITIVE` | 0.057970285415649414 | 0.0024778158229595486 |
