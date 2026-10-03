# M25 scoped mechanism belief

Status: `PROTOCOL_FROZEN`

This file freezes the belief and the falsification rule. It does not run Qwen, does not execute the audit, and does not change `MechanismResponseModel`, `SelfModelBelief`, the feasibility gate, or the M24 artifacts.

Scientific question: can the frozen response `g` be kept as an evidence-bearing belief with an explicit scope and an uncertainty taken from recorded residuals, such that new evidence can leave the belief supported, mark it uncertain, or mark it contradicted, without changing the response model?

The object is `MechanismBelief`. It is not a self-model.

## Object

`MechanismBelief` holds:

| field | content |
| --- | --- |
| `mechanism_id` | One of `M22.1-D1-L0`, `M22.1-D1-L8`, `M22.1-D1-L15`. |
| `response_model` | The existing `MechanismResponseModel` for that cell. |
| `validity_scope` | Mechanism claim, supported catalog, and the unverified family marker. |
| `uncertainty` | Validation residual scale, inherited `CRITICAL_Z`, and inflation. |
| `evidence_history` | Append-only applications. |
| `contradiction_history` | Append-only applications that themselves contradicted the belief. |
| `version` | Starts at `1`. Each `apply_evidence` returns a new object at version + 1. |
| `status` | `SUPPORTED`, `UNCERTAIN`, or `CONTRADICTED`. |

`predict(pre_intervention_state)` calls `response_model.predict` and returns that scalar as `predicted_effect`, plus the current effective uncertainty, status, version, and mechanism id. It has no outcome argument.

`apply_evidence` returns a new belief. The previous object is left as it was. The response model on the new belief is the same object.

## Scope

Mechanism scope is the conjunction of fields already fixed by the runners and by M24. An evidence claim that differs on any of them is out of scope. Out-of-scope evidence is stored and does not change status, uncertainty, or the response model.

| field | value | standing |
| --- | --- | --- |
| `model_id` | `Qwen/Qwen2.5-0.5B` | Pinned by the M23 and M24 runners. |
| `model_revision` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | Same pin. |
| `intervention_id`, `layer`, `hook` | The one cell on the belief | M24 family. One belief per cell. |
| `direction_id`, `direction_sha256` | D1, sha256 `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` | The direction `g` uses. |
| `alpha` | `+1` | The audited intervention magnitude. |
| `response_rule` | `prediction = g` | The frozen rule. |
| `positive_token_id`, `negative_token_id` | `9834`, `902` | The margin inside `g`. |
| `supported_catalog_sha256` | `ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78` | The only catalog named by `CONSUMPTION_SUPPORTED`. |
| `surface_family` | `UNVERIFIED` | M24 used family as a balance stratum. It is not a scope filter. |

A different catalog does not by itself make evidence out of mechanism scope. It also does not extend `supported_catalog_sha256`. No evidence append adds a catalog.

Layer 23, D2, and any alpha other than `+1` are outside mechanism scope. M21 `q_invalid` is a different estimand and is not an uncertainty source for this object. No new catalog is created.

## Uncertainty

The scale is the M23 sample standard deviation of `observed_effect - g` on the twelve M24 validation episodes of that cell. The divisor is `n - 1`, the existing `_sample_sd`. Validation is the calibration partition because M24 did not use it for the consumption verdict and did not use it to fit `g`.

`critical_z` is the existing `CRITICAL_Z = 1.96`. Inflation starts at `1`. Effective uncertainty is `residual_sd * inflation`. On a critical contradiction, inflation becomes `min(2, inflation * 2)`, the existing cap. The residual standard deviation is not recomputed. The bootstrap used in M23 and M24 is not turned into a second uncertainty inside the belief.

Calibration prompt ids are stored. Submitting one of them again is `CALIBRATION_REUSE`: the row is kept in the evidence history and does not change status or inflation.

## Evidence

Each record stores `evidence_id`, `source_experiment`, `artifact`, `mechanism_id`, the mechanism claim, `prompt_id`, `predicted_value`, `observed_value`, `residual = observed - predicted`, both signs, `held_out`, `protocol_version`, `provenance`, and `timestamp` when one exists. M24 rows have no per-episode timestamp, so that field is empty for them. History tuples are extended by constructing a new belief. Old records are not rewritten.

## Status transitions

Flags use the current effective uncertainty, before any inflation change. They reuse the M23 comparisons:

- sign mismatch: both signs are strict and they differ. A zero sign is not a mismatch.
- outside: `abs(observed - predicted) > 1.96 * effective_uncertainty`. Equality is not outside.
- confident: `abs(predicted) >= 1.96 * effective_uncertainty`.
- contradicted: in mechanism scope, not a calibration reuse, and either sign mismatch or outside.
- critical: contradicted and confident.

The next status is:

| condition | status after | reason |
| --- | --- | --- |
| Out of mechanism scope | unchanged | `OUT_OF_SCOPE` |
| Calibration reuse | unchanged | `CALIBRATION_REUSE` |
| Contradicted and critical | `CONTRADICTED` | `CONFIDENT_CONTRADICTION` |
| Contradicted and not critical | `CONTRADICTED` | `EVIDENCE_OUTSIDE_OR_SIGN` |
| Already `CONTRADICTED`, and this evidence is not a contradiction | `CONTRADICTED` | `STICKY_CONTRADICTION` |
| Not contradicted, catalog sha is not the supported catalog | `UNCERTAIN` | `UNSUPPORTED_CATALOG` |
| Not contradicted, catalog is supported, current status is `UNCERTAIN` | `UNCERTAIN` | `CONSISTENT_IN_SCOPE` |
| Not contradicted, catalog is supported, current status is `SUPPORTED` | `SUPPORTED` | `CONSISTENT_IN_SCOPE` |

`CONTRADICTED` does not return to `SUPPORTED` or `UNCERTAIN`. `UNCERTAIN` does not return to `SUPPORTED`. A contradiction is also appended to `contradiction_history`. The other reasons are not.

The predeclared contradictory observation is `contradictory_observed`. Its step is `(1.96 + 1) * effective_uncertainty`. Positive predictions step down by that amount, negative predictions step up, and a zero prediction steps to the positive magnitude. If the uncertainty is zero, the step is `1`. That observation is in the supported catalog when the claim says so.

## Future audit

Execution is not part of this freeze. The two conditions are independent. Each starts from a fresh belief built by `belief_from_m24`.

1. Consistent condition. Apply the M24 evaluation rows for that cell in `prompt_id` order. Predicted value is the stored `g`. Observed value is the stored effect. The resulting status is reported. It is not required, in advance, to remain `SUPPORTED`, because that requirement would be a new threshold on residuals this protocol has not used.
2. Contradictory condition. On a fresh belief, apply one `contradictory_observed` pair whose predicted value is `0.25` and whose catalog is the supported M24 catalog. The status must be `CONTRADICTED`. The response model must be the original object.

A third bookkeeping case, also on a fresh belief, applies predicted `0.25` and observed `0.25` with a catalog sha of 64 zeros. The status must be `UNCERTAIN`, and the supported catalog sha must still be the M24 hash.

Command, from the repository root, when execution is separately requested:

`python3 scripts/run_m25_scoped_mechanism_belief.py`

The runner writes `reports/M25_SCOPED_MECHANISM_BELIEF_RESULTS.json` and refuses to start if that file exists. It does not load Qwen.

## What this protocol does not do

It does not fit a slope, embed a prompt, read `pre_dot`, or add a regime feature. It does not edit `SelfModelBelief` or the response model. It does not select a new intervention. It does not treat the resulting belief as a self-model, an introspection trace, or a general agent state. The initial `SUPPORTED` label means M24 consumption passed for this family and this catalog, and the response on later prompts is still `g`.
