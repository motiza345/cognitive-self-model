# Real self-model Qwen execution

Status: frozen before the model is loaded.

Version: `REAL_SELF_MODEL_QWEN.1`

This protocol is one execution of the bound self-model on Qwen. It is not a statistical milestone. It does not fit a slope, an intercept, or a new mechanism. It does not add a third prompt, a second cell, or a second alpha. `rho` is not a verdict input.

The execution command hashes this file before `from_pretrained`. If the hash differs from the hash recorded in the runner at freeze time, the command stops and does not load the model.

## 1. Mechanism binding

The only permitted cell is `M22.1-D1-L0`.

| field | value |
| --- | --- |
| `mechanism_id` | `M22.1-D1-L0` |
| `model_id` | `Qwen/Qwen2.5-0.5B` |
| `model_revision` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| `layer` | `0` |
| `hook` | `blocks.0.hook_resid_post` |
| `direction_id` | `D1` |
| `direction_sha256` | `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` |
| `alpha` | `1.0` |
| `positive_token_id` | `9834` |
| `negative_token_id` | `902` |
| `positive_text` | ` yes` |
| `negative_text` | ` no` |
| `response_rule` | `prediction = g` |

`g` is `MechanismResponseModel.predict(pre_intervention_state)` for that cell. The margin is logit `9834` minus logit `902` at the last position. No coefficient is applied.

## 2. Two prompts

Exactly two prompts. Both strings are new. Neither string is taken from the M22.1, M23, M23-G, feasibility-gate, M24, M26, M27, or M29 catalogs.

Prompt A is the only prompt whose outcome may update the self-model.

Prompt B is the only prompt used for the later decision. Its outcome is not computed.

| role | id | sha256 | text |
| --- | --- | --- | --- |
| evidence | `qwen-bridge-evidence` | `b5e73ca8b7122cc2133850b9fb1499f7ea710e30d289bd2241de1acaa02f93c2` | `The harbor clock stopped at noon` |
| held-out decision | `qwen-bridge-held-out` | `37f1441e4cda56b9bb2f6c02b80487c6771582cc63e20ccd1a634f95bace318b` | `A kiln fires clay into` |

The hash is SHA-256 of the UTF-8 text with no added newline.

## 3. Initial self-model

`initial_bound_self_model(held_out_prompt_id="qwen-bridge-held-out")`.

The operational claim is `USE_LINEAR`. The version is `1`. The binding is the cell in section 1.

The no-update clone is taken after the Prompt A prediction is sealed and before Prompt A evidence is applied. `apply_evidence` is not called on the clone.

## 4. Seal order for Prompt A

The order is fixed:

1. Clean baseline forward. Record the baseline margin.
2. Clean pre-intervention forward. Record the last-token margin gradient at `blocks.0.hook_resid_post`.
3. `g = MechanismResponseModel.predict` on that gradient.
4. Seal the operational `PredictionRecord` and the bridge seal. Flush that file.
5. Only after that file exists, run the intervention hook.
6. Record the intervened margin.
7. `observed_outcome = intervened_margin - baseline_margin`.

The sealed prediction file must not contain `observed_outcome`, `observed_effect`, `intervened_margin`, `intervened_output`, or `residual`. `baseline_margin` in that file is the clean margin from step 1. `kappa` and `rho`, when present, are from step 2.

Each written file records UTC time, SHA-256, and `mtime_ns` in `reports/real_self_model_qwen_raw/order_log.jsonl`. The prediction file's `mtime_ns` must be strictly less than the intervention file's `mtime_ns`.

## 5. Intervention

Prompt A starts at `USE_LINEAR`, so `decide` returns `INTERVENE`.

The hook is the object returned by `BoundSelfModel.open_hook` for that decision. That method calls `make_resid_hook` with alpha `+1` and the frozen D1 vector. No other alpha and no other hook name are permitted.

`observed_outcome` is the raw difference of the two margins. It is not normalized, regressed, or calibrated.

## 6. Evidence

Exactly one operational `EvidenceRecord` is built from Prompt A, through `BoundSelfModel.apply_evidence`. `evidence_sign` is `sign(observed_outcome)` from `operational_self_model.sign`. The claim transition is `SelfModel.update`, which calls the existing `next_claim`. This protocol does not replace that function.

| current claim | signs | next claim |
| --- | --- | --- |
| `USE_LINEAR` | opposite and both nonzero | `WITHHOLD` |
| `USE_LINEAR` | same nonzero sign | `USE_LINEAR` |
| `USE_LINEAR` | either sign zero | `USE_LINEAR` |
| `WITHHOLD` | any | `WITHHOLD` |

A same nonzero sign, or a zero sign, is an uninformative agreement. The run stops. It does not load another prompt and it does not change cells.

## 7. Curvature metadata

The clean pre-intervention forward may also compute `kappa = d · ∇(∇m · d)` at layer 0, by the same double backward the M29 runner uses, and `rho = kappa / 2`.

`DecisionPolicy`, `next_claim`, the choice to call the hook, and the verdict do not read `kappa` or `rho`. If the double backward fails or is non-finite, both fields are omitted. The forward is not repeated in order to obtain them.

## 8. Held-out decision

After Prompt A evidence has been applied, `decide` is called on Prompt B for the updated model and for the clone.

If Prompt A moved the claim to `WITHHOLD`, the updated decision is `ABSTAIN` and the clone's decision is `INTERVENE`.

The endpoint is `decision(updated) != decision(clone)`. No absolute-error cutoff, interval, or confidence cutoff is used.

## 9. Hook gate on Prompt B

The Prompt B decision is read before any Prompt B intervention.

If that decision is `ABSTAIN`, `open_hook` is called only to show that it returns no hook. The returned value is not applied to the model. A call count on `make_resid_hook` during this step must stay zero.

If that decision is `INTERVENE`, the hook function is not constructed and the model is not run on Prompt B. Prompt B has no outcome.

Prompt B is not tokenized.

## 10. Verdict

`SELF_MODEL_OPERATIONAL` only when the Prompt A signs are opposite and both nonzero, and all of the following hold:

- the prediction file was sealed before the intervention file
- the Prompt A hook ran only after `decide` returned `INTERVENE` on the initial `USE_LINEAR` claim
- the evidence outcome is `intervened_margin - baseline_margin`
- `update` left the claim at `WITHHOLD`
- the updated Prompt B decision is `ABSTAIN`
- the clone's Prompt B decision is `INTERVENE`
- the Prompt B hook was not called
- the sealed prediction, the pre-update version snapshot, and the clone's empty evidence history are unchanged
- the run used the two protocol prompts and the one protocol cell, and it did not substitute a prompt after the outcome

`SELF_MODEL_NOT_OPERATIONAL` when the Prompt A signs are opposite and both nonzero, and any condition in the list above fails.

`INCONCLUSIVE` when the Prompt A signs are not an opposite nonzero pair. The run records the Prompt B decisions and stops. It does not sample another prompt.

## 11. Audit

Before `from_pretrained`, the runner checks:

- this file's SHA-256 equals the freeze-time hash
- both prompt hashes equal the hashes in section 2
- both texts are absent from the earlier catalogs listed in section 2
- `reports/REAL_SELF_MODEL_QWEN_RESULTS.json` and `reports/REAL_SELF_MODEL_QWEN_EXECUTION.md` do not already exist
- `reports/real_self_model_qwen_raw/` does not already contain a prediction or an intervention file

The results record `protocol_sha256`, `prompt_A_sha256`, `prompt_B_sha256`, `model_revision`, `direction_sha256`, the order-log timestamps, `git` HEAD, and the working-tree status. They also record SHA-256 values, taken before the model load and again after the verdict, for `operational_self_model.py`, `mechanism_response.py`, `mechanism_belief.py`, `residual_update.py`, `m29_pre_evidence.py`, `m23/m22_reuse.py`, `bound_self_model.py`, and this protocol. A change fails the audit.

The machine-readable copy of the frozen fields is the JSON block below. It is part of this file.

<!-- FROZEN_SPEC_BEGIN -->
```json
{
  "alpha": 1.0,
  "direction_id": "D1",
  "direction_sha256": "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411",
  "hook": "blocks.0.hook_resid_post",
  "layer": 0,
  "mechanism_id": "M22.1-D1-L0",
  "model_id": "Qwen/Qwen2.5-0.5B",
  "model_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
  "negative_text": " no",
  "negative_token_id": 902,
  "positive_text": " yes",
  "positive_token_id": 9834,
  "prompt_a": {
    "id": "qwen-bridge-evidence",
    "role": "evidence",
    "sha256": "b5e73ca8b7122cc2133850b9fb1499f7ea710e30d289bd2241de1acaa02f93c2",
    "text": "The harbor clock stopped at noon"
  },
  "prompt_b": {
    "id": "qwen-bridge-held-out",
    "role": "held_out_decision",
    "sha256": "37f1441e4cda56b9bb2f6c02b80487c6771582cc63e20ccd1a634f95bace318b",
    "text": "A kiln fires clay into"
  },
  "protocol_version": "REAL_SELF_MODEL_QWEN.1",
  "response_rule": "prediction = g"
}
```
<!-- FROZEN_SPEC_END -->
