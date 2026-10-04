# Architecture reality check

This audit reads the current implementation and the stored reports. It does not run Qwen, does not rescore M24, M25, or M26, and does not change any protocol or source file.

## 1. Executive conclusion

The repository contains a frozen intervention-response rule, `prediction = g`, and a wrapper that records evidence against that rule. Evidence can change the wrapper's status and its inflation factor. Evidence does not change `g`, the stored direction, the hook, the catalog the wrapper calls supported, or the causal form of the prediction.

`MechanismResponseModel` is not a self-model. `MechanismBelief` is not a self-model. Together they are a mechanism-response system with a belief wrapper. The M26 residual correction sits outside both objects, adds a cell constant to a copy of `g`, and the stored verdict for that correction is `INCONCLUSIVE`.

Architecture decision: `CURRENT_ARCHITECTURE_IS_A_MECHANISM_RESPONSE_SYSTEM_WITH_BELIEF_WRAPPER`.

## 2. Current data model

An earlier scalar tracker, `SelfModelBelief` in `src/cognitive_self_model/m23/belief.py`, remains in the tree. `update_belief` replaces `predicted_effect` with the mean of accumulated scalar observations. `reports/PROJECT_FEASIBILITY_GATE.md` describes that M23-G object as a belief tracker and states that no genuine self-model exists. `tests/test_mechanism_belief.py` (`test_module_does_not_grow_a_second_response_rule`) requires that `mechanism_belief.py` not mention `SelfModelBelief`. The mechanism path audited below does not call it.

### MechanismResponseModel

Defined in `src/cognitive_self_model/mechanism_response.py`. The dataclass is frozen. `frozen_mechanism_response_models` builds three cells from `_CELLS` and from `reports/PROJECT_FEASIBILITY_GATE_RESULTS.json`. `reports/MECHANISM_RESPONSE_MODEL.md` lists the same stored contents.

| field | role | class |
| --- | --- | --- |
| `intervention_id` | Cell name, one of `M22.1-D1-L0`, `M22.1-D1-L8`, `M22.1-D1-L15` | FIXED/CONSTANT |
| `layer` | `0`, `8`, or `15` | FIXED/CONSTANT |
| `hook` | `blocks.{layer}.hook_resid_post` | FIXED/CONSTANT |
| `direction_id` | `D1` | FIXED/CONSTANT |
| `direction_seed` | `22101` | USER/EXPERIMENT SUPPLIED, then fixed |
| `direction_sha256` | Recorded D1 digest | FIXED/CONSTANT |
| `direction` | Unit vector from `primary_direction(896, 22101)`, rejected if the digest differs | USER/EXPERIMENT SUPPLIED, then fixed |
| `alpha` | `1.0` | FIXED/CONSTANT |
| `response_rule` | The string `prediction = g` | FIXED/CONSTANT |
| `training_baseline_mean` | Cell train-effect mean read from the feasibility-gate artifact. Stored. Not an argument of `predict` | DERIVED |
| `provenance.experiment` | `PROJECT_FEASIBILITY_GATE` | FIXED/CONSTANT |
| `provenance.commit` | `7aa64543fa45b0e09bb7e247641ce8c926f8a66d` | FIXED/CONSTANT |
| `provenance.results_artifact` | `reports/PROJECT_FEASIBILITY_GATE_RESULTS.json` | FIXED/CONSTANT |
| `provenance.verdict` | `FEASIBILITY_CONTINUE` | FIXED/CONSTANT |

`PreInterventionState` is the argument of `predict`, not a field of the model. Its only field is `margin_gradient`, the last-token gradient of logit `9834` minus logit `902`. That input is OBSERVED at prediction time. `directional_derivative` returns the float64 dot product of that gradient with `direction`. No coefficient is applied.

`__post_init__` rejects a layer, hook, mean, direction, alpha, rule, or provenance that differs from those constants. `tests/test_mechanism_response_model.py` (`test_object_has_no_fit_or_update_method`) asserts that the class has no `fit`, `update`, `update_belief`, `search`, or `calibrate` method.

### MechanismBelief

Defined in `src/cognitive_self_model/mechanism_belief.py`. `belief_from_m24` builds one belief per frozen response model after checking that `reports/M24_CONSUMPTION_RESULTS.json` has verdict `CONSUMPTION_SUPPORTED`.

| field | role | class |
| --- | --- | --- |
| `mechanism_id` | The cell id | FIXED/CONSTANT |
| `response_model` | The `MechanismResponseModel` object | FIXED/CONSTANT |
| `validity_scope.claim.model_id` | `Qwen/Qwen2.5-0.5B` | FIXED/CONSTANT |
| `validity_scope.claim.model_revision` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | FIXED/CONSTANT |
| `validity_scope.claim.intervention_id`, `layer`, `hook` | The cell | FIXED/CONSTANT |
| `validity_scope.claim.direction_id`, `direction_sha256` | D1 | FIXED/CONSTANT |
| `validity_scope.claim.alpha` | `+1` | FIXED/CONSTANT |
| `validity_scope.claim.response_rule` | `prediction = g` | FIXED/CONSTANT |
| `validity_scope.claim.positive_token_id`, `negative_token_id` | `9834`, `902` | FIXED/CONSTANT |
| `validity_scope.claim.catalog_sha256` | Catalog named on the evidence claim | USER/EXPERIMENT SUPPLIED on each evidence record |
| `validity_scope.supported_catalog_sha256` | M24 catalog `ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78` | FIXED/CONSTANT |
| `validity_scope.surface_family` | The string `UNVERIFIED` | FIXED/CONSTANT |
| `uncertainty.calibration_partition` | `validation` | FIXED/CONSTANT |
| `uncertainty.calibration_source` | `reports/M24_CONSUMPTION_EPISODES.csv` | FIXED/CONSTANT |
| `uncertainty.calibration_prompt_ids` | The twelve M24 validation prompt ids for that cell | DERIVED from the episode file |
| `uncertainty.n` | `12` | DERIVED |
| `uncertainty.residual_sd` | Sample standard deviation of `observed_effect - g` on those twelve rows | DERIVED, then fixed |
| `uncertainty.critical_z` | `CRITICAL_Z` from `m23/belief.py`, the protocol states `1.96` | FIXED/CONSTANT |
| `uncertainty.inflation` | Starts at `1`. A critical contradiction sets `min(2, inflation * 2)` | FIXED/CONSTANT at construction; later bookkeeping |
| `uncertainty.effective` | Property `residual_sd * inflation` | DERIVED |
| `evidence_history` | Append-only tuple of `EvidenceApplication` | OBSERVED values plus DERIVED flags |
| `contradiction_history` | Applications that themselves contradicted the belief | DERIVED selection of the history |
| `version` | Starts at `1`. Each `apply_evidence` adds `1` | DERIVED |
| `status` | `SUPPORTED`, `UNCERTAIN`, or `CONTRADICTED` | DERIVED by the fixed transition table |

`predict` returns `response_model.predict(state)` as `predicted_effect`, plus `uncertainty.effective`, `status`, `version`, and `mechanism_id`. The signature has no outcome parameter (`test_prediction_has_no_outcome_and_matches_the_response_model`).

### Residual-update logic

`src/cognitive_self_model/residual_update.py` defines functions and constants. It does not define a stored object, and it does not import `MechanismBelief` or `MechanismResponseModel`.

| name | role | class |
| --- | --- | --- |
| `PROTOCOL_VERSION` | `M26.INDEPENDENT_EVIDENCE_UPDATE.1` | FIXED/CONSTANT |
| `FAMILY` | The same three cell ids | FIXED/CONSTANT |
| `PRIOR_MEAN` | `0` | FIXED/CONSTANT |
| `PRIOR_COUNT` | `1` | FIXED/CONSTANT |
| `SHUFFLE_SEED` | `26001` | FIXED/CONSTANT |
| `FROZEN_SIGMA` | The three M25 `residual_sd` values at inflation `1` | DERIVED in M25, then fixed in this module |
| `posterior_delta` output | `n * residual_mean / (n + 1)` | DERIVED |
| `posterior_sd` output | `sigma / sqrt(n0 + n)` | DERIVED |
| `prediction_before` | Returns `g` | DERIVED from the response rule |
| `prediction_after` | Returns `g + delta` | DERIVED |
| `update_verdict` inputs | CI class labels and `leakage_ok` | USER/EXPERIMENT SUPPLIED results of a frozen comparison |

The M26 runner calls `frozen_mechanism_response_models`, writes `g` before outcomes, then calls `cell_deltas`, `global_delta`, and `prediction_after`. It compares `asdict` of the response models before and after the run. It does not call `MechanismBelief` or `apply_evidence`.

### Evidence records

`EvidenceRecord` fields:

| field | class |
| --- | --- |
| `evidence_id`, `source_experiment`, `artifact`, `provenance`, `timestamp` | USER/EXPERIMENT SUPPLIED |
| `mechanism_id`, `claim` | USER/EXPERIMENT SUPPLIED; `mechanism_id` is copied from `claim.intervention_id` |
| `prompt_id` | USER/EXPERIMENT SUPPLIED |
| `predicted_value` | DERIVED from `g` when the source is an M24 row |
| `observed_value` | OBSERVED when the source is an episode; the synthetic contradictory value is produced by `contradictory_observed` |
| `residual` | DERIVED as `observed_value - predicted_value` |
| `sign_predicted`, `sign_observed` | DERIVED by `_sign` |
| `held_out` | USER/EXPERIMENT SUPPLIED |
| `protocol_version` | FIXED/CONSTANT `M25.SCOPED_MECHANISM_BELIEF.1` |

`EvidenceApplication` adds flags computed inside `apply_evidence`: `in_mechanism_scope`, `calibration_reuse`, `sign_mismatch`, `outside_interval`, `confident`, `critical`, `status_before`, `status_after`, and `reason`. Those flags are DERIVED. The enclosed `EvidenceRecord` is unchanged.

## 3. Learned vs fixed quantities

Nothing in this path estimates a coefficient of `g`. The feasibility-gate execution states that the train means are the only fitted numbers and that the candidate prediction is `g` with no slope (`reports/PROJECT_FEASIBILITY_GATE_EXECUTION.md`). Those means become `training_baseline_mean` and are a baseline, not the issued prediction.

Lifecycle of the mechanism path:

| transition | what changes | code | learned from evidence? | alters the mechanism representation? | alters scope? | alters uncertainty? | alters causal structure? | kind |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Build the initial mechanism | A frozen cell object is constructed | `frozen_mechanism_response_models` | No. The mean is copied from the gate artifact. The vector is the seeded D1 vector | The representation is instantiated as `g`. It is not fit to outcomes | Scope is not an attribute of this object | No uncertainty is stored | The form is the dot product named by the additive intervention | Fixed construction |
| Predict | The issued scalar changes with the prompt's margin gradient. Stored fields stay put | `MechanismResponseModel.predict` | No | No. `test_predict_does_not_change_stored_parameters` | No | No | No | Evaluation of a fixed rule |
| Intervene and record the effect | Episode files gain `observed_effect` | M24 and M26 runners, outside the two classes | The effect is observed. It is not written into the response model | No | No | No | No | Observation |
| Build the initial belief | Status `SUPPORTED`, version `1`, inflation `1`, `residual_sd` from M24 validation | `belief_from_m24` | The scale is a sample standard deviation of recorded validation residuals. `g` is not refit | No | Scope is set to the M24 catalog pin and `surface_family = UNVERIFIED` | The scale is set once | No | Derivation of a scale, then a fixed pin |
| Apply evidence | A new belief gets status, version, histories, and sometimes inflation | `apply_evidence` | The status and inflation follow a fixed rule. `residual_sd` is copied through | No. The new belief carries the same `response_model` object (`mechanism_belief.py`, the return of `apply_evidence`; `test_consistent_evidence_keeps_support_and_does_not_change_the_response`) | No. `validity_scope` is the same object. A foreign catalog hash sets status `UNCERTAIN` and leaves `supported_catalog_sha256` unchanged (`reports/M25_SCOPED_MECHANISM_BELIEF_PROTOCOL.md`, Scope) | Inflation can double, up to `2`, on a critical contradiction. `residual_sd` stays | No | Bookkeeping update |
| Next belief prediction | The returned effect is still `g`. The dict also carries the current effective uncertainty and status | `MechanismBelief.predict` | No | No | No | The reported uncertainty reflects inflation. It is not subtracted from or multiplied into `g` | No | Evaluation of the same rule |
| M26 residual correction | A cell constant `δ` is computed and `prediction_after = g + δ` is scored | `posterior_delta`, `prediction_after`, used by `scripts/run_m26_independent_evidence_update.py` | `δ` is a shrinkage of the update-partition mean residual. That is a statistical parameter update of an additive constant | No. The protocol says `g + δ` is not a new value of `g` (`reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md`). The runner does not write `δ` onto the response model | No | `τ` is computed for a descriptive interval. It is not written into `UncertaintyRecord` | No. The correction is one number per cell, not a new dependence | Statistical parameter update, outside the belief, and the stored verdict is `INCONCLUSIVE` |

`apply_evidence` is a bookkeeping update. `posterior_delta` is a statistical parameter update of an external constant. Neither is a mechanism-representation update. After either one, the stored response rule is still `prediction = g`.

## 4. Mechanism representation audit

The gate's operational definition of a mechanism representation is the scalar `g`, and the issued prediction is `g` itself (`reports/PROJECT_FEASIBILITY_GATE.md`, section 2). Category D in that table is a later claim: category C plus a scope that changes where the prediction applies, an evidence update, and a decision that consumes the prediction. The gate says a pass of the gate does not reach D.

| item | status | location |
| --- | --- | --- |
| Causal graph | ABSENT | No graph type in `mechanism_response.py`, `mechanism_belief.py`, or `residual_update.py`. The stored relation is one dot product |
| Upstream variables | PARTIAL | `PreInterventionState.margin_gradient` is the only upstream input to `predict`. There is no stored parent set |
| Downstream variables | PARTIAL | The predicted quantity is the alpha-`+1` change in the yes/no margin, tokens `9834` and `902` (`mechanism_response.py`, `PreInterventionState`; `mechanism_belief.py`, `POSITIVE_TOKEN_ID` and `NEGATIVE_TOKEN_ID`). One scalar |
| Causal edges | ABSENT | No edge list. The intervention definition `h <- h + alpha * d1` is the experiment that motivates `g` (`reports/PROJECT_FEASIBILITY_GATE.md`, section 2), and it is not stored as an editable edge |
| Interaction terms | ABSENT | `directional_derivative` is a dot product. The gate's minimum representation excludes a second derivative and a learned map (section 3) |
| Context or regime dependence | ABSENT | `surface_family` is `UNVERIFIED` and the M25 protocol says it is not a scope filter. The same protocol says no regime feature is added |
| Temporal state | PARTIAL | `version` and the append-only histories record the order of evidence applications. They do not represent dynamics of the mechanism |
| Intervention alternatives | ABSENT | Alpha other than `+1`, layer 23, and a response rule other than `prediction = g` are rejected (`mechanism_response.py`, `__post_init__`). M24 records `g_orth` and excludes it from `consumption_verdict` |
| Competing mechanism hypotheses | ABSENT | One rule is stored. `test_module_does_not_grow_a_second_response_rule` forbids a second rule in the belief module |
| Mechanism identity inferred from evidence | ABSENT | `intervention_id` is checked against the claim. Out-of-scope evidence keeps the current status and the current model (`_next_status`) |
| Mechanism discovery logic | ABSENT | No search method. `test_object_has_no_fit_or_update_method` bans `search` |
| Mechanism revision logic | ABSENT | `CONTRADICTED` is sticky and does not replace `g` (`reports/M25_SCOPED_MECHANISM_BELIEF_PROTOCOL.md`, status table). M26's `δ` is not written back, and its verdict is `INCONCLUSIVE` |

## 5. Self-Model requirement audit

Operational definition used here: a mechanism-level self-model represents the mechanism, the conditions where it applies, the intervention that probes it, the downstream response, the supporting evidence, the uncertainty, and the evidence that would falsify or revise the mechanism.

| requirement | classification | repository evidence |
| --- | --- | --- |
| What internal mechanism is being modeled | PARTIALLY SATISFIED | The cell, hook, D1, alpha, and the rule `prediction = g` are stored (`MechanismResponseModel`; `reports/MECHANISM_RESPONSE_MODEL.md`). That is the gate's category C scalar. It does not represent a mechanism beyond that derivative |
| Under what conditions it applies | PARTIALLY SATISFIED | `ValidityScope` pins model id, revision, cell, direction, alpha, token ids, and one catalog digest. `surface_family` is `UNVERIFIED` and is not a filter. Evidence does not extend `supported_catalog_sha256` |
| What intervention probes it | SATISFIED | The probe is alpha `+1` along D1 at the named hook. The object represents that one probe and rejects others |
| What downstream response it predicts | SATISFIED | `predict` returns `g`, the directional derivative of the yes/no margin. M24 consumed that scalar as the prediction |
| What evidence supports the claim | PARTIALLY SATISFIED | Initial status `SUPPORTED` means the M24 consumption verdict (`belief_from_m24`; M25 protocol, final section). `evidence_history` then stores predicted value, observed value, and residual. The M25 execution moved every cell to `CONTRADICTED` on the M24 evaluation rows. The history records that transition. It does not replace the claim |
| What uncertainty applies | PARTIALLY SATISFIED | `UncertaintyRecord` stores the validation residual standard deviation, `critical_z`, and inflation. `predict` reports `residual_sd * inflation`. The standard deviation is not recomputed. M26's `τ` is not stored on the belief |
| What evidence would falsify or revise the mechanism | PARTIALLY SATISFIED | The status table can mark `CONTRADICTED`, and a critical contradiction can double inflation up to `2`. The same code path leaves `response_model` and `validity_scope` in place. Revision of the mechanism is not implemented. The M26 correction, which could have changed the numeric prediction, was not established |

No row is graded with a numeric score.

## 6. M24/M25/M26 evidence ledger

### M24

Demonstrated, from `reports/M24_CONSUMPTION_EXECUTION.md`: on the frozen evaluation prompts, consuming `g` before the intervention reduced absolute error relative to the stored cell mean. Baseline MAE `0.06271952169912832`, mechanism MAE `0.005885519992307424`, paired mean `0.0568340017068209`, interval `[0.03877295034976836, 0.07716522058950821]`, class `CI_POSITIVE`. The sign decision did not lose. The shuffled class was `CI_INCLUDES_ZERO`. `leakage_ok` was true. Verdict `CONSUMPTION_SUPPORTED`. The execution text calls this a consumption result for the frozen object on this catalog.

Not demonstrated: the same execution text says it is not a self-model. The protocol excludes the train partition from forwarding, excludes validation and `g_orth` from the verdict, and fits no slope (`reports/M24_CONSUMPTION_PROTOCOL.md`). The response object is unchanged by the audit.

### M25

Demonstrated, from `reports/M25_SCOPED_MECHANISM_BELIEF_EXECUTION.md`: with `qwen_loaded = false`, each fresh belief starts `SUPPORTED` at version `1` and inflation `1`. Applying the stored M24 evaluation rows moves all three cells to `CONTRADICTED`. L0 and L8 change at `c-syntax-06` for a strict sign mismatch, reason `EVIDENCE_OUTSIDE_OR_SIGN`, inflation stays `1`. L15 changes at `c-completion-06`, reason `CONFIDENT_CONTRADICTION`, inflation becomes `2`. `residual_sd` is the same number after every condition: L0 `0.02029281160603693`, L8 `0.009377970182012517`, L15 `0.004044481362351381`. The predeclared synthetic observation also ends `CONTRADICTED` with inflation `2`. The execution states that this synthetic transition tests the bookkeeping rule and is not a discovered contradiction in Qwen. A catalog hash of 64 zeros yields `UNCERTAIN` and leaves the supported catalog hash unchanged. The response rule remains `prediction = g`.

Not demonstrated: no new Qwen measurement, no refit of `g`, no change of `residual_sd`, no return from `CONTRADICTED` to `SUPPORTED`, and no verification of surface family. The protocol says the object is not a self-model (`reports/M25_SCOPED_MECHANISM_BELIEF_PROTOCOL.md`).

### M26

Demonstrated, from `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_EXECUTION.md`: the frozen correction was estimated on the update partition and scored on the untouched holdout. `leakage_ok` is true. Verdict `INCONCLUSIVE`. The holdout class of MAE(`g`) minus MAE(`g + δ_cell`) is `CI_INCLUDES_ZERO`, mean `0.00026372789995481554`, interval `[-0.0020563612402649725, 0.0024766029017848816]`. The class of MAE(`g + δ_global`) minus MAE(`g + δ_cell`) is `CI_INCLUDES_ZERO`, mean `0.0012612554535592411`, interval `[-0.0010591863981880966, 0.003358709975344125]`. The shuffled comparison is `CI_NEGATIVE`, mean `-0.002229452439774266`, interval `[-0.0026054393157923243, -0.0017591169559331434]`. The execution says the correction is unestablished.

Not demonstrated: `UPDATE_SUPPORTED` and `UPDATE_NOT_SUPPORTED` both failed their frozen conditions. The protocol states that a supported result would not retune `g`, would not clear the M25 contradiction record, and would not establish a self-model (`reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md`). The recorded result is the weaker inconclusive case. No second prior was authorized.

### M26 gap audit

Demonstrated, from `reports/M26_GAP_AUDIT.md`: a descriptive classification `STRUCTURED_BUT_UNSTABLE`. On the update partition, cell means account for `0.3538055039598171` of residual sum of squares, with L0 mean `-0.014976908773135936`. On the holdout, cell means account for `3.956851343862743e-05`, and the three cell means lie near `0.002`. The audit states that this cell pattern does not return on the holdout, and that no tested recorded variable has a nontrivial association in both partitions. It leaves the three explanations of the inconclusive interval unresolved: a small correction, twelve prompts per partition, and structure that fails to replicate. The realized holdout paired mean cited there is `0.00026372789995481554`.

Not demonstrated: the audit says it is not a replacement of the M26 verdict, defines no significance cutoff, and does not identify a residual feature that reproduces on the holdout. It does not specify a successor experiment.

## 7. Minimum missing abstraction

The smallest missing abstraction is an evidence-to-mechanism revision.

`apply_evidence` already revises bookkeeping: status, version, history, and, on one rule, inflation. `prediction_after` can add a residual constant to a number computed from `g`, and that constant is not stored on the mechanism or the belief. The next call to `predict` still returns the original dot product, including after status has become `CONTRADICTED`.

A mechanism-level self-model, under the definition in section 5 and under category D of the feasibility gate, requires evidence to change the mechanism claim that the next prediction uses, or to change where that claim applies. The repository has the evidence record and the immutable claim. It does not have an operator from the first to the second.

## 8. Architecture decision

`CURRENT_ARCHITECTURE_IS_A_MECHANISM_RESPONSE_SYSTEM_WITH_BELIEF_WRAPPER`

`CURRENT_ARCHITECTURE_IS_ALREADY_A_SELF_MODEL` is unsupported. Both modules and the M24, M25, and M26 protocols say the objects are not self-models, and the prediction after evidence is still `g`.

`CURRENT_ARCHITECTURE_IS_A_PARTIAL_SELF_MODEL` is also the wrong description of this code. Scope, evidence, and an uncertainty number are present, which is why section 5 marks several requirements partial. Those fields wrap one immutable response rule. They do not constitute a partial mechanism model whose claim evidence can revise. The gate kept that further step in category D and said the feasibility pass does not reach it. M25 then implemented the wrapper and recorded that it is not a self-model. M26 tested an external additive correction and did not establish it.

What the repository contains is the category C response rule, the M24 result that this rule was consumable on one catalog, the M25 wrapper that can call the same rule contradicted, and the M26 result that a cell residual constant was not established on a later holdout.

## 9. Implications for the next experiment

The next experiment has to establish one capability: evidence that was not taken from the scored prompts revises the stored mechanism claim, or revises where that claim applies, and the revised claim predicts a later held-out response under a rule fixed before those outcomes.

A further score of unchanged `g`, a further status label on unchanged `g`, or a search through the stored M26 residuals does not establish that capability. This audit does not specify that experiment.
