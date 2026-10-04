# Real self-model bridge implementation

This is the binding named in `reports/REAL_SELF_MODEL_BRIDGE_DESIGN.md`. It does not load Qwen and it does not change the operational claim transition.

## Files

Created:

- `src/cognitive_self_model/bound_self_model.py`
- `tests/test_bound_self_model.py`
- `reports/REAL_SELF_MODEL_BRIDGE_IMPLEMENTATION.md`

No existing source file was edited. `operational_self_model.py`, `mechanism_response.py`, `mechanism_belief.py`, `residual_update.py`, `m23/m22_reuse.py`, and the M22.1–M29 modules are unchanged by this step. No second module was required.

## API

`MechanismBinding` is frozen. The only legal value is the cell `M22.1-D1-L0`: layer `0`, hook `blocks.0.hook_resid_post`, direction `D1`, alpha `+1`, model `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, tokens `9834` and `902`, response rule `prediction = g`.

`BoundSelfModel` holds:

| field | role |
| --- | --- |
| `self_model` | The operational `SelfModel` |
| `binding` | The frozen cell |
| `held_out_prompt_id` | Prompt id that cannot seal or update |
| `seals` | Append-only pre-intervention seals |
| `envelopes` | Append-only provenance envelopes |

`initial_bound_self_model(held_out_prompt_id)` starts at claim `USE_LINEAR`, version `1`.

`decide(episode_id)` calls `DecisionPolicy.decide(self_model, DecisionContext(episode_id))`.

`seal_prediction(...)` calls `MechanismResponseModel.predict` on the L0 model and then `SelfModel.seal_prediction`. It is rejected when the claim is `WITHHOLD`.

`open_hook(episode_id, direction, recorder)` returns `make_resid_hook(alpha=1, direction, recorder)` when the decision is `INTERVENE`, and `None` when it is `ABSTAIN`.

`apply_evidence(envelope)` checks the envelope against the seal, builds one operational `EvidenceRecord` with `make_evidence`, and calls `SelfModel.update`.

`clone()` returns a new `BoundSelfModel` whose operational model is `self_model.clone()` and whose binding is the same frozen binding.

`BridgeSeal` stores the pre-intervention identity, `g`, the clean `baseline_margin`, and optional `kappa` / `rho`. It has no outcome field.

`BridgeEvidenceEnvelope` stores the same identity, `prompt_sha256`, `baseline_margin`, `intervened_margin`, `observed_outcome`, provenance, and the same optional `kappa` / `rho`. `observed_outcome` must equal `intervened_margin - baseline_margin`. When both curvature fields are present, `rho` must equal `kappa / 2`.

## State flow

```text
initial_bound_self_model
  claim USE_LINEAR, version 1, binding M22.1-D1-L0
        |
        v
decide -> INTERVENE
        |
        v
seal_prediction
  g = L0 MechanismResponseModel.predict(pre_intervention_state)
  operational PredictionRecord sealed, sealed_before_outcome True
  BridgeSeal appended
  envelopes still empty
        |
        v
open_hook -> make_resid_hook
        |
        v
observed_outcome = intervened_margin - baseline_margin
        |
        v
apply_evidence
  make_evidence(...) -> SelfModel.update -> next_claim
        |
        v
USE_LINEAR stays USE_LINEAR when the signs agree or either sign is zero
USE_LINEAR becomes WITHHOLD when the signs are opposite and nonzero
WITHHOLD stays WITHHOLD
```

The binding is the same object identity's value before and after `update`. Version and evidence history advance only inside the operational model.

## Hook gate

`open_hook` reads the decision first. `ABSTAIN` returns `None` on that branch, and `make_resid_hook` is not called. `INTERVENE` calls the existing `make_resid_hook` with the binding's alpha. A `WITHHOLD` model therefore cannot open the hook. A linear seal is also rejected unless `decide` returns `INTERVENE`.

## Evidence path

`apply_evidence` rejects an envelope whose cell, model, hook, direction, or alpha differs from the binding, whose prompt id is the held-out id, or whose prompt hash, baseline margin, `kappa`, or `rho` differs from the seal. The operational record it then builds has only `evidence_id`, `prediction_record_id`, `observed_outcome`, `evidence_sign`, `provenance`, and `order`. `SelfModel.update` consumes that record and calls the existing `next_claim`. This module does not define `next_claim`.

## No-update clone

`clone()` after the seal and before `apply_evidence` has claim `USE_LINEAR`, the same binding, an empty evidence history, and an empty envelope tuple. `apply_evidence` on the original returns a new object. The clone's claim stays `USE_LINEAR`, so its later decision stays `INTERVENE`.

## Tests

Command:

```text
python3 -m pytest tests/test_bound_self_model.py -q
```

Result: `13 passed`.

The tests cover the fixed L0 binding, both decisions, a decision source that does not read `rho`, the seal-before-outcome rule, the margin difference, both sign transitions, `WITHHOLD` remaining `WITHHOLD`, clone independence, prediction immutability, the hook gate, and curvature stored only on the envelope.

## Confirmations

Qwen was not loaded. The bridge module does not call `from_pretrained` and does not import `transformer_lens`. The unit run asserted that `transformer_lens` was not imported.

M22.1–M29 behavior was not modified. This step adds the bridge module and its tests. It does not edit those sources, protocols, or artifacts.
