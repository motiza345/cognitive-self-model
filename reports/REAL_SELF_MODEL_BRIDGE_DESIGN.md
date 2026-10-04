# Real self-model bridge

Status: design. This file does not implement a Qwen loop, does not load a model, and does not rescore M22.1–M29.

The synthetic acceptance test in `reports/OPERATIONAL_SELF_MODEL_RESULTS.json` returned `SELF_MODEL_OPERATIONAL`. The initial decision was `INTERVENE`, the updated decision was `ABSTAIN`, and the no-update clone stayed `INTERVENE`. The fields that changed were `claim`, `version`, `evidence_history`, `version_history`, and `next_order`.

That object is the spine. The bridge binds it to one existing Qwen intervention. It does not replace the claim.

## Audit

`SelfModel` in `src/cognitive_self_model/operational_self_model.py` owns the decision-relevant claim. `DecisionPolicy.decide` returns `INTERVENE` when `claim == USE_LINEAR` and `ABSTAIN` when `claim == WITHHOLD`. `next_claim` moves `USE_LINEAR` to `WITHHOLD` only when the sealed prediction and the later outcome have opposite nonzero signs. `update` returns a new object and keeps the prediction tuple.

`MechanismResponseModel.predict` returns `g`, the directional derivative of the yes/no margin along frozen D1. It has no update. The margin is logit `9834` minus logit `902` at the last position. The intervention is alpha `+1` through `make_resid_hook` in `src/cognitive_self_model/m23/m22_reuse.py`. Model identity used by the existing runners is `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, float32.

`MechanismBelief` can mark that same rule `SUPPORTED`, `UNCERTAIN`, or `CONTRADICTED`. `apply_evidence` returns the same response model. Those three words are status labels. They are not a claim a decision reads. M25 recorded every cell as `CONTRADICTED` while `prediction = g` stayed in place. Carrying those labels into the bridge would repeat the wrapper the architecture reset rejected.

`SelfModelBelief` replaces a scalar mean and does not choose an intervention. M26's delta, M27's gain, and M29's `rho = kappa_d1 / 2` are historical measurements. M29 showed that `rho` predicts holdout mechanism-response error. Nothing in that result chooses `INTERVENE` or `ABSTAIN`. `rho` may be stored beside a sealed prediction. It is not the self-model, and it is not an input to `decide` or `next_claim`.

No current runner gates `make_resid_hook` on a self-model claim. Every Qwen execution applies alpha `+1` because the protocol said to. The missing join is that gate.

## A. Reuse unchanged

| component | use |
| --- | --- |
| `SelfModel`, `initial_self_model`, `seal_prediction`, `update`, `clone`, `next_claim`, `sign` | The claim, the history, and the transition already tested |
| `DecisionPolicy.decide` | The only action function |
| `PredictionRecord`, `EvidenceRecord` | The operational records. Their fields stay as they are |
| `MechanismResponseModel.predict`, `directional_derivative`, `PreInterventionState` | The number sealed as `prediction` under `USE_LINEAR` |
| `primary_direction`, `vector_sha256` | Frozen D1, digest `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` |
| `logit_margin`, `apply_last_token_additive`, `make_resid_hook` | Clean margin and the alpha-`+1` intervention |
| The load arguments in `scripts/run_m23_direct_qwen_self_model_audit.py` | Model id, revision, dtype, token ids `(9834, 902)`. The script itself is not edited |

A later runner may call these. It does not modify the files that define them.

## B. Historical and diagnostic only

These stay on disk. The bridge does not read them as the claim, the decision, or the success criterion.

- `MechanismBelief` status, inflation, and `residual_sd`
- `SelfModelBelief`
- M26 `delta` and `prediction_after`
- M27 gain
- M29's identifiability verdict, as a reason to change the claim
- `paired_mean_ci` and every MAE comparison
- `surface_family`
- Layer 23, D2, and alpha other than `+1`

`kappa` and `rho` may be copied onto a sealed pre-intervention measurement record. `DecisionPolicy` and `next_claim` do not read that record.

## C. Minimum new abstraction

One frozen binding around the existing `SelfModel`. Name it `BoundSelfModel`. It is not a subclass of `MechanismBelief` or `MechanismResponseModel`.

| field | role |
| --- | --- |
| `self_model` | The operational `SelfModel`. This owns `claim`, `version`, and both histories |
| `binding` | Immutable identity of the intervention `USE_LINEAR` refers to |

`binding` fields, all fixed before the first forward:

| field | value |
| --- | --- |
| `mechanism_id` | `M22.1-D1-L0` |
| `layer` | `0` |
| `hook` | `blocks.0.hook_resid_post` |
| `direction_id` | `D1` |
| `direction_sha256` | the digest above |
| `alpha` | `1.0` |
| `model_id` | `Qwen/Qwen2.5-0.5B` |
| `model_revision` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `positive_token_id` | `9834` |
| `negative_token_id` | `902` |
| `response_rule` | `prediction = g` |

`M22.1-D1-L0` is the first cell in the existing family tuple. The choice is positional so that a later sign cannot be used to pick a cell. It is not a claim that layer 0 is more reliable than layer 8 or 15.

The binding does not change when the claim becomes `WITHHOLD`. `WITHHOLD` means the bound intervention is not applied. It does not install a new direction, a new layer, or a new response formula.

No new module is added by this design. The operational types already exist, and a skeleton would either duplicate them or start the implementation.

## D. Evidence fields

The operational `EvidenceRecord` gains no fields. `SelfModel.update` still receives that record. A bridge envelope, stored beside it and not passed into `next_claim`, carries the Qwen identity:

| field | source |
| --- | --- |
| `evidence_id` | already on `EvidenceRecord` |
| `prediction_record_id` | already on `EvidenceRecord` |
| `observed_outcome` | `intervened_margin - baseline_margin` |
| `evidence_sign` | `sign(observed_outcome)` |
| `provenance` | runner id and artifact path |
| `order` | the operational audit clock |
| `mechanism_id` | the binding |
| `prompt_id` | the evidence prompt |
| `prompt_sha256` | hash of that prompt's text |
| `model_id`, `model_revision`, `hook`, `direction_sha256`, `alpha` | the binding, copied so a mismatched run is rejected |
| `baseline_margin` | clean forward, sealed with the prediction's context |
| `intervened_margin` | forward after the hook |
| `kappa` | optional pre-intervention measurement |
| `rho` | optional, and only as `kappa / 2` from the existing identity |

`update` reads `observed_outcome` through the operational record. It does not read `kappa`, `rho`, `baseline_margin`, or `intervened_margin`. An envelope whose binding fields differ from `BoundSelfModel.binding` is rejected before `update`. An envelope whose `prompt_id` is the held-out prompt is rejected.

The sealed `PredictionRecord` likewise stays operational. Its `prediction` is `g`. A bridge seal file written before the hook adds the same identity fields, `prompt_sha256`, `g`, and the optional `kappa`/`rho` pair. That file has no outcome field. `sealed_before_outcome` remains `True`.

## E. What the decision reads

```text
DecisionPolicy.decide(bound.self_model, DecisionContext(episode_id)).claim path
```

The function reads `bound.self_model.claim` and nothing else. `episode_id` is audit context. `g`, `rho`, status, inflation, version, and the binding's layer are not inputs.

The intervention gate is the same value:

```text
INTERVENE -> make_resid_hook(alpha=1, direction=D1) at binding.hook
ABSTAIN   -> the hook is not called
```

## F. Smallest real Qwen episode

Two prompts, one cell, one evidence intervention. The prompt strings are not chosen in this file. A later execution protocol writes both texts, their ids, and their SHA-256 digests before the model is loaded. Both texts are disjoint from every prompt already used in M22.1 through M29. The assignment of which text is evidence and which is held out is part of that protocol, not a choice made after a margin is seen.

```text
BoundSelfModel
  self_model.claim = USE_LINEAR
  self_model.version = 1
  binding = M22.1-D1-L0
        |
        v
DecisionPolicy -> INTERVENE
        |
        v
clean forward on the evidence prompt
  baseline margin
  g = MechanismResponseModel.predict(PreInterventionState)
  optional kappa, rho = kappa / 2
        |
        v
seal PredictionRecord
  prediction = g
  claim = USE_LINEAR
  version = 1
  file closed before the hook
        |
        v
make_resid_hook(alpha=+1) on that prompt
        |
        v
observed_outcome = intervened_margin - baseline_margin
        |
        v
EvidenceRecord + bridge envelope
        |
        v
SelfModel.update -> next_claim
        |
        v
held-out prompt, no outcome required
  decide(updated) and decide(clone)
```

The no-update clone is `bound.self_model.clone()` taken after the seal and before `update`. The evidence object is in scope. `update` is not called on the clone.

If the signs are opposite and nonzero:

| model | claim | decision on the held-out prompt | hook on the held-out prompt |
| --- | --- | --- | --- |
| initial | `USE_LINEAR` | `INTERVENE` | not this prompt |
| updated | `WITHHOLD` | `ABSTAIN` | not called |
| clone | `USE_LINEAR` | `INTERVENE` | not called for the clone; the decision is the recorded action |

The held-out prompt is not forwarded with the hook in the success path. The evidence prompt's hook has already run, and only because the pre-update decision was `INTERVENE`.

If the signs agree, or either sign is zero, `next_claim` leaves `USE_LINEAR`. Both the updated model and the clone decide `INTERVENE`. That episode does not show a decision change. It is an uninformative observation. The runner stops. It does not load another prompt to look for a sign change.

## G. Genuine failure

Failure is a broken causal chain. It is independent of whether `g` is close to the observed effect.

`SELF_MODEL_NOT_OPERATIONAL` if any of these hold:

- `decide` returns an action that does not match `self_model.claim`
- the evidence prompt's hook runs without a prior `INTERVENE`, or does not run after `INTERVENE`
- the held-out hook runs when the decision is `ABSTAIN`
- the outcome, the intervened margin, or the residual is readable in the sealed prediction file
- the signs are opposite and nonzero, and the updated decision equals the clone's decision
- the signs agree, or either is zero, and the claim changes
- `update` changes a sealed prediction, an earlier evidence record, or an earlier version snapshot
- the claim changes because of `rho`, a status label, a gain, or a residual constant
- the held-out prompt id is the evidence prompt id

`INCONCLUSIVE` if the seal, the gate, and the prompt split are intact, and the evidence signs are not an opposite nonzero pair. A correct update then leaves the later decision equal to the clone's decision.

`SELF_MODEL_OPERATIONAL` if the signs are an opposite nonzero pair, the updated decision is `ABSTAIN`, the clone's decision is `INTERVENE`, the hook gate matches those actions, and the immutability checks hold.

No absolute-error cutoff, interval, or sample size enters these labels.

## H. Freeze before the first Qwen execution

This design is not that execution. Before the model loads, a separate protocol file must already contain all of the following, and the runner must record its SHA-256 and refuse to continue if the file changes:

1. Protocol version and the statement that `next_claim` and `DecisionPolicy.decide` are the functions in `operational_self_model.py`, unmodified.
2. The binding table in section C, including `mechanism_id = M22.1-D1-L0`.
3. The two prompt texts, ids, SHA-256 digests, and which id is evidence versus held out.
4. A disjointness check against prior catalogs, evaluated before the first forward.
5. Artifact order: clean-forward measurements, sealed prediction file, only then the hooked forward, only then the outcome file, only then `update`.
6. The banned keys in the sealed prediction file: `observed_outcome`, `observed_effect`, `intervened_margin`, `baseline_margin` as an outcome, `residual`, `kappa` used as a claim, `rho` used as a claim.
7. The verdict rule in section G, with no accuracy threshold added later.
8. The stop rule: a sign agreement ends the episode as `INCONCLUSIVE` and does not authorize another prompt.
9. The clone procedure: same sealed model, same evidence object, `update` not called.
10. Hashes of `operational_self_model.py`, `mechanism_response.py`, and `m23/m22_reuse.py` at the start of the run. A mismatch stops the run.

`kappa` and `rho`, if sealed, are sealed in the prediction file before the hook, using `rho = kappa / 2` with no fitted scale. Their absence does not change the verdict. Their presence does not change the claim.

## What must not be added

- A status or inflation field as the claim
- A coefficient on `g`, a gain, a residual mean, or a regression
- A decision that reads `rho`, `kappa`, uncertainty, or version
- A second action, a second cell, or a search over layers
- A new prompt after the outcome is known
- An edit to `operational_self_model.py`, `mechanism_response.py`, `mechanism_belief.py`, `residual_update.py`, or any M22.1–M29 protocol, runner, or artifact
- A statistical milestone whose endpoint is prediction error

## Readiness

`READY_FOR_IMPLEMENTATION`

The implementation that this authorizes is the binding, the envelope, the hook gate, and tests that drive `next_claim` with a supplied margin. It does not include loading Qwen. The first Qwen execution waits until the section H protocol exists and is hashed.
