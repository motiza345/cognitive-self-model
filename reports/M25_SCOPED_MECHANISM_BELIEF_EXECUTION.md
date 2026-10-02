# M25 scoped mechanism belief execution

The protocol is unchanged. This file records the one run of `python3 scripts/run_m25_scoped_mechanism_belief.py`. No threshold was changed, and the response model was not refit.

`qwen_loaded = false`

Each condition starts from a fresh belief. The initial status of every cell is `SUPPORTED`, version `1`, inflation `1`.

The contradictory condition uses the predeclared synthetic observation. That transition tests the bookkeeping rule. It is not a discovered contradiction in Qwen.

## Status by condition

| cell | consistent | contradictory synthetic | unsupported catalog |
| --- | --- | --- | --- |
| `M22.1-D1-L0` | `CONTRADICTED` | `CONTRADICTED` | `UNCERTAIN` |
| `M22.1-D1-L8` | `CONTRADICTED` | `CONTRADICTED` | `UNCERTAIN` |
| `M22.1-D1-L15` | `CONTRADICTED` | `CONTRADICTED` | `UNCERTAIN` |

## Uncertainty

Residual scale is the validation sample standard deviation. It is the same number after every condition. Inflation changes only where the inherited critical rule fires.

| cell | residual sd | consistent inflation | consistent effective | synthetic inflation |
| --- | ---: | ---: | ---: | ---: |
| `M22.1-D1-L0` | 0.02029281160603693 | 1.0 | 0.02029281160603693 | 2.0 |
| `M22.1-D1-L8` | 0.009377970182012517 | 1.0 | 0.009377970182012517 | 2.0 |
| `M22.1-D1-L15` | 0.004044481362351381 | 2.0 | 0.008088962724702761 | 2.0 |

The unsupported-catalog case leaves inflation at `1` on every cell.

## Consistent evidence

Twelve held-out M24 evaluation rows per cell, in `prompt_id` order. Version goes from `1` to `13`. Evidence history length is `12`. Contradiction history length is `1`. The response rule remains `prediction = g`.

| cell | transition that changes status | prompt | reason | later rows |
| --- | --- | --- | --- | --- |
| `M22.1-D1-L0` | step 10, `SUPPORTED` to `CONTRADICTED` | `c-syntax-06` | `EVIDENCE_OUTSIDE_OR_SIGN` | 2 `STICKY_CONTRADICTION` |
| `M22.1-D1-L8` | step 10, `SUPPORTED` to `CONTRADICTED` | `c-syntax-06` | `EVIDENCE_OUTSIDE_OR_SIGN` | 2 `STICKY_CONTRADICTION` |
| `M22.1-D1-L15` | step 2, `SUPPORTED` to `CONTRADICTED` | `c-completion-06` | `CONFIDENT_CONTRADICTION` | 10 `STICKY_CONTRADICTION` |

On `M22.1-D1-L0` and `M22.1-D1-L8` the firing row is a strict sign mismatch, not an outside-interval residual, and it is not critical, so inflation stays `1`. On `M22.1-D1-L15` the firing row is outside the interval and critical, so inflation becomes `2`.

The first nine rows on layer 0 and layer 8, and the first row on layer 15, are `CONSISTENT_IN_SCOPE`.

## Contradictory evidence

One synthetic row per fresh belief. Predicted value `0.25`. Observed value is `contradictory_observed`. Version goes from `1` to `2`. Evidence history length is `1`. Reason `CONFIDENT_CONTRADICTION`. Inflation becomes `2`. Scope stays the supported M24 catalog.

## Scope

Every consistent row and every synthetic contradictory row is in mechanism scope and in the supported catalog. The unsupported-catalog row is in mechanism scope, carries a catalog sha of 64 zeros, and leaves the supported catalog sha unchanged. Its reason is `UNSUPPORTED_CATALOG`. No calibration prompt is reused.

## Leakage

The runner did not load Qwen. Evaluation predictions are the stored `g` values, checked equal to `candidate_prediction` before they are applied. `residual_sd` is not recomputed. `training_baseline_mean` and the D1 sha256 are unchanged on every final snapshot.
