# Operational self-model

Status: frozen before execution.

Version: `OPERATIONAL_SELF_MODEL.1`

This protocol defines the smallest synthetic loop in which evidence changes the claim a fixed decision reads. It is not a statistical milestone. It does not load Qwen. It does not read M22.1–M29 artifacts. It does not use curvature, a gain, a residual regression, or uncertainty inflation. Predictive accuracy is not an endpoint.

## Objects

The implementation lives in `src/cognitive_self_model/operational_self_model.py`. It does not import `MechanismBelief`, `MechanismResponseModel`, `SelfModelBelief`, or the M26–M29 modules.

### SelfModel

| field | initial value | role |
| --- | --- | --- |
| `claim` | `USE_LINEAR` | The only input of the decision |
| `version` | `1` | Increments only when `update` accepts evidence |
| `evidence_history` | empty tuple | Append-only evidence records |
| `predictions` | empty tuple | Append-only sealed predictions |
| `version_history` | one record, version `1`, claim `USE_LINEAR` | Append-only claim snapshots |
| `next_order` | `1` | Audit clock |

`claim` is one of `USE_LINEAR` or `WITHHOLD`. There is no status field and no inflation field.

`seal_prediction` and `update` return a new `SelfModel`. They do not write into the receiver.

`clone` returns a new object with the same fields. It does not apply evidence.

### DecisionPolicy

```text
decide(self_model, context) =
    INTERVENE    if self_model.claim == USE_LINEAR
    ABSTAIN      if self_model.claim == WITHHOLD
```

`context` carries `episode_id` for the audit trail. The action does not depend on `episode_id`, on the outcome, or on any measurement.

### PredictionRecord

| field | rule |
| --- | --- |
| `record_id` | Non-empty, unique on the model |
| `episode_id` | Non-empty episode the seal belongs to |
| `model_version` | `SelfModel.version` at seal time |
| `claim` | `SelfModel.claim` at seal time |
| `prediction` | Under `USE_LINEAR`, the supplied finite measurement. Under `WITHHOLD`, `0` |
| `order` | The model's `next_order` at seal time |
| `sealed_before_outcome` | `True`. A record with any other value is rejected |

The record has no outcome field. `WITHHOLD` rejects a supplied measurement, so a withhold seal cannot store a linear effect.

### EvidenceRecord

| field | rule |
| --- | --- |
| `evidence_id` | Non-empty, unique on the model that accepts it |
| `prediction_record_id` | Id of a prediction already sealed on that model |
| `observed_outcome` | Finite float, read only after the seal |
| `evidence_sign` | `sign(observed_outcome)` |
| `provenance` | Non-empty string |
| `order` | The model's `next_order` when the evidence is built, strictly greater than the prediction's `order` |

`sign` is `1` when the value is positive, `-1` when it is negative, and `0` when it is zero. Non-finite values are rejected.

## State transition

`update` reads the sealed prediction's sign from the stored `PredictionRecord` and the evidence sign from `sign(observed_outcome)`. It does not read a caller-supplied next claim.

| current claim | predicted sign | observed sign | next claim |
| --- | --- | --- | --- |
| `USE_LINEAR` | nonzero | nonzero and different | `WITHHOLD` |
| `USE_LINEAR` | nonzero | same nonzero sign | `USE_LINEAR` |
| `USE_LINEAR` | zero, or observed sign zero | anything | `USE_LINEAR` |
| `WITHHOLD` | any | any | `WITHHOLD` |

On every accepted update:

- `version` becomes `version + 1`
- the evidence record is appended
- a new version snapshot is appended
- `next_order` becomes `next_order + 1`
- `predictions` is the same tuple the model already held

The prediction sealed under a different claim or a different version than the receiving model is rejected. Evidence whose `evidence_sign` disagrees with `sign(observed_outcome)` is rejected. Evidence whose `order` is not the model's current `next_order` is rejected.

`seal_prediction` appends one prediction and advances `next_order`. It leaves `claim`, `version`, `evidence_history`, and `version_history` as they were.

## Episodes

The fixture constants are part of the protocol.

| name | id | measurement | outcome | role |
| --- | --- | --- | --- | --- |
| A | `episode-a` | `+1` | `-1` | Opposite nonzero signs |
| B | `episode-b` | none | none | Held-out decision after A |
| C | clone of the post-seal, pre-update model of A | the evidence object of A is in scope | `update` is not called | No-update control |
| D | `episode-d` | `+1` | `+2` | Same nonzero sign, fresh model |
| E | the objects of A | none | none | Immutability after A |

Episode B's id is distinct from episode A's id. Episode B's outcome is not an input.

### Required observations

Episode A, before the outcome: claim `USE_LINEAR`, version `1`, decision `INTERVENE`. The prediction is sealed with `sealed_before_outcome = True` and `prediction = +1`. After the outcome `-1` is applied: claim `WITHHOLD`, version greater than `1`, evidence history longer than before.

Episode B, from the updated model: decision `ABSTAIN`.

Episode C, from the clone: decision `INTERVENE`.

Episode D, from a fresh `USE_LINEAR` model: after the agreeing outcome, claim `USE_LINEAR`.

Episode E:

- the sealed prediction's fields are unchanged by `update`
- the evidence record's fields are unchanged by `update`
- the pre-update model still has claim `USE_LINEAR` and version `1`
- the version snapshot taken at version `1` is still the first snapshot of the updated model
- `update` returns a different object

## Primary comparison

The primary result is a decision difference, not an error.

```text
decide(updated model of A, episode B) != decide(clone that did not update, episode B)
```

The clone and the pre-update model are equal on every field. The updated model differs in `claim`, `version`, `evidence_history`, `version_history`, and `next_order`. It does not differ in `predictions`.

Required decisions:

| model | decision |
| --- | --- |
| initial model of A | `INTERVENE` |
| updated model, episode B | `ABSTAIN` |
| no-update clone, episode B | `INTERVENE` |

## Verdict

No error threshold is used.

`SELF_MODEL_OPERATIONAL` when all of the following hold:

- the initial claim is `USE_LINEAR` at version `1`
- the prediction is sealed before the outcome, and its order is strictly earlier than the evidence order
- episode A is an opposite nonzero pair
- episode B is a different episode id from episode A
- the no-update clone equals the pre-update model
- episode D is a same-sign nonzero pair
- the three required decisions hold and the updated decision differs from the clone's decision
- the updated claim is `WITHHOLD`, its version is greater than `1`, and its evidence history grew
- episode D leaves the claim at `USE_LINEAR`
- the prediction, the prior evidence record, and the prior version snapshot are unchanged

`INCONCLUSIVE` when the fixture fails one of the well-formedness conditions above: the signs are not an opposite nonzero pair, the seal is not before the evidence, episode B is not distinct, the clone is not the pre-update state, or episode D is not a same-sign nonzero pair. The loop then cannot attribute a decision change to the update.

`SELF_MODEL_NOT_OPERATIONAL` when the fixture is well formed and any required decision, claim change, or immutability condition fails.

## Execution

Unit tests, and only those, run first:

```text
python -m pytest tests/test_operational_self_model.py
```

After this protocol and the implementation are in place, one acceptance execution writes the artifacts:

```text
python scripts/run_operational_self_model.py
```

That command writes `reports/OPERATIONAL_SELF_MODEL_RESULTS.json` and `reports/OPERATIONAL_SELF_MODEL_EXECUTION.md`. It records the SHA-256 of this file. It does not edit this file. If either artifact already exists, the command stops without rewriting it.
