# Architecture reset

This is an integration audit. It does not implement a self-model, does not define a milestone experiment, does not define a revision operator, and does not authorize a Qwen run. M22.1 through M29 stay as they are.

The previous audit, `reports/ARCHITECTURE_REALITY_CHECK.md`, concluded:

`CURRENT_ARCHITECTURE_IS_A_MECHANISM_RESPONSE_SYSTEM_WITH_BELIEF_WRAPPER`

That conclusion still holds. M27, M28, and M29 add a precise limit to it. M27 applied a frozen gain to `g` and the stored verdict is `M27_INCONCLUSIVE`: every real cell interval contained `1`, so the consumed gain stayed `1`. M28 found no sealed pre-outcome column that could host an independent revision test (`NO_SUITABLE_PRE_EVIDENCE`). M29 showed that a pre-intervention curvature, `rho = kappa_d1 / 2`, predicts mechanism-response error on an untouched holdout. The stored verdict is `M29_IDENTIFIABILITY_SUPPORTED`, primary mean `0.006551810962803386`, interval `[0.003115443572082937, 0.010331481498993796]`. The M29 protocol states that this result justifies a later revision protocol and that this result does not itself write that operator. No such protocol is written here.

`rho` is an unconsumed reliability measurement. It is not an input to `MechanismResponseModel.predict`, `MechanismBelief.predict`, or any action choice. After M29 the repository is a mechanism-response system with a belief wrapper, plus a reliability signal that no decision reads.

## A. What currently deserves the name SelfModel?

**None.**

| object | what it does | why it is not a SelfModel |
| --- | --- | --- |
| `MechanismResponseModel` | Frozen rule `prediction = g`. `predict` returns the directional derivative. The class has no `fit` or `update`. | A fixed response rule. Evidence cannot reach it. |
| `MechanismBelief` | Wraps that rule. `apply_evidence` returns a new belief with a new status, version, history, and sometimes inflation, and the same `response_model` object. `predict` still returns `g`. The M25 protocol says this object is not a self-model. | Bookkeeping around an immutable claim. |
| `SelfModelBelief` | Replaces `predicted_effect` with the mean of scalar observations. `predict(alpha)` scales that mean. It does not read `PreInterventionState` and does not choose an intervention. The M23 report is `INCONCLUSIVE` and places decision improvement and closed-loop self-improvement outside the result. The feasibility gate calls it a belief tracker. | A scalar tracker. The name in the source is not an architectural verdict. |
| M26 `prediction_after` | Returns `g + delta` from an external constant. Verdict `INCONCLUSIVE`. The constant is not written onto the response model. | An unscored correction outside both objects. |
| M27 gain | `prediction = gain * g`, initial gain `1`. Real cells stayed `UNREVISED`. The operator lives in the design test and the runner, not in `MechanismResponseModel`. | An unestablished coefficient on the same rule. |
| M29 `predicted_residual` | Returns `kappa / 2` at alpha `+1`. The holdout comparison supports that quantity as a predictor of `observed_effect - g`. | A measurement. Nothing consumes it. |

Adding another field to `MechanismBelief` would extend the wrapper. It would leave the claim that prediction consumes unchanged.

## B. What information must a SelfModel contain?

These are roles. A role that nothing reads is storage, not part of the model.

**State.** The identity of the claim that the next decision will read, and the pre-intervention measurement that claim is about to use. Today the measurement exists (`PreInterventionState.margin_gradient`, and, in M29, `kappa` computed before alpha `+1`). The claim identity does not: status and version change while the claim stays `prediction = g`.

**Mechanism.** The rule that maps that measurement to a predicted effect of a named intervention. The current rule is the first directional derivative along D1 at alpha `+1`. A self-model's mechanism is whichever rule the decision function is defined to read.

**Prediction.** A forecast sealed before the intervention. M24, M26, M27, and M29 already seal `g` (and M29 also seals `rho`) before outcomes. Sealing is necessary and already present. It becomes part of a self-model only when a later decision reads the claim that produced the seal.

**Scope.** The conditions under which the claim may drive a decision: model identity, hook, direction, alpha, and the episode set the claim is allowed to cover. `ValidityScope` already pins model, revision, cell, direction, alpha, token ids, and one catalog digest. `surface_family` is the constant `UNVERIFIED` and is not a filter. Evidence does not change the supported catalog.

**Uncertainty.** A statement of whether the current claim is actionable. `UncertaintyRecord.effective` is reported beside `g` and is not an input to the issued effect. Inflation can double up to `2` and still leaves `g` in place. M29's `rho` is a pre-intervention statement about the remainder `observed_effect - g`. It is the strongest uncertainty measurement in the tree, and it is unconsumed.

**Evidence.** A pair of a sealed prediction and a later observation, with the signs and the residual. `EvidenceRecord` already has this shape. M23's update history stores the observation against a scalar mean.

**Update history.** An append-only record of claim identity before the evidence, the evidence itself, and claim identity after. `MechanismBelief` records status before and after. It does not record a change of response rule, because that change does not occur. `SelfModelBelief.update_history` records a change of the scalar mean and does not record an action.

**Decision relevance.** A pure function from the current claim to an action, fixed before the evidence is observed. The repository has statistical verdicts and M24's sign-of-effect comparison. It has no function that chooses intervene versus abstain, or one intervention versus another, from the contents of a self-model. Experiments apply alpha `+1` at a pre-specified cell.

## C. What operation is missing?

The missing operation is:

**Evidence changes the claim that a fixed decision function reads, and that change changes a later intervention choice.**

`apply_evidence` changes status, version, history, and sometimes inflation. The next `predict` returns the same `g`. M26 can add a constant to a copy of `g` and does not store that constant where a later decision would read it. M27 can replace the gain and, on the real update rows, did not. M29 can name the remainder and does not write it into the consumed rule.

A revision of a coefficient that no decision reads would repeat the same gap one level up. The operation is the consumption of the updated claim by the next action, together with an ablation in which that update is removed.

## D. Smallest end-to-end task

The smallest task that demonstrates the operation is a two-action loop on a synthetic episode stream.

Actions, fixed before any evidence:

- `INTERVENE`. Apply the named intervention and observe its effect.
- `ABSTAIN`. Do not apply it. The observation is then the absence of an intervention, not a second mechanism.

Claims, and the decision that reads them. The decision is a pure function of claim identity:

- Claim `USE_LINEAR` maps to `INTERVENE`.
- Claim `WITHHOLD` maps to `ABSTAIN`.

Initial self-model, before any action: claim `USE_LINEAR`. Its sealed prediction on an episode is the pre-intervention measurement `g` for that episode.

Update, logical and fixed before outcomes. Use the sign function already in `src/cognitive_self_model/m23/score.py`: positive, negative, or zero. Zero is that function's boundary. It is not a fitted cutoff.

- The prediction record is closed before the observation exists.
- If `sign(observed)` and `sign(predicted)` are both nonzero and differ, the next self-model carries claim `WITHHOLD`.
- If the signs agree, or either sign is zero, the next self-model carries the same claim it carried before.

The task is complete when a later episode, disjoint from the evidence episode, is decided by the updated self-model, and a clone that skipped the update decides that same later episode differently.

This task does not ask whether `g` is accurate, whether a mechanism was identified, whether `rho` predicts a residual, or whether a belief object exists. Those results are already on record and are not this test.

## E. Minimum synthetic environment

No language model.

An episode is a record available before action:

- `episode_id`
- a pre-intervention measurement `g`, finite and nonzero on every episode the test treats as informative
- the effect that `INTERVENE` will produce, hidden until the prediction for that episode is sealed
- the effect that `ABSTAIN` produces, which is defined as no intervention

Two disjoint sets are declared before the run:

- an evidence episode
- a later episode, absent from the evidence set

The informative evidence episode is part of the fixture. Its hidden effect has the opposite sign from its `g`. A second fixture, used only for the attribution check, has an effect with the same sign as its `g`. The later episode's hidden effect is never an input to the update.

The environment can be realized as a quadratic margin whose first derivative is `g` and whose second directional derivative is `kappa`, with `rho = kappa / 2` available as a measurement. That identity is already frozen in M29. The acceptance test does not score it. The environment's only duty is to keep the intervention effect hidden until the prediction is sealed, and to keep the later episode out of the evidence set.

## F. Components to reuse without modification

These stay byte-for-byte as they are. A later implementation may call them. This audit does not.

| component | reuse |
| --- | --- |
| `MechanismResponseModel.predict` and `directional_derivative` | The initial claim's measurement of `g`, when a physical probe is eventually authorized. |
| `PreInterventionState` | The pre-intervention argument of that measurement. |
| `predicted_residual` in `m29_pre_evidence.py` | The identity `rho = kappa / 2` at alpha `+1`. Unfitted. |
| `primary_direction`, `make_resid_hook` | The D1 vector and the residual hook, only if a later physical probe is separately authorized. |
| `paired_mean_ci` | An optional diagnostic. It is not the acceptance criterion below. |
| `_sign` | The sign used by the logical update. |
| M22.1 through M29 protocols, runners, catalogs, and artifacts | Historical evidence. The reset does not rescore them. |

## G. Components to wrap or integrate

Wrapping means a new object reads them. It does not mean editing them.

| component | integration |
| --- | --- |
| `MechanismBelief` construction pattern | Immutable object, append-only evidence, claim recorded before and after. The new object owns the claim the decision reads. `apply_evidence` stays a status transition on `prediction = g`. |
| `EvidenceRecord` shape | Predicted value, observed value, residual, both signs, and a held-out flag. Copy the shape into the new evidence packet. |
| M29 `kappa` and `rho` | A reliability measurement a future decision may be specified to read. Until a decision reads it, it stays an M29 result. |
| The intervention hook | An action sink. It runs only when the decision returns `INTERVENE`. |

`SelfModelBelief.update_belief` shows the pattern "return a new object whose prediction changed." That pattern is the one to copy. The class itself is not the spine: its prediction is a mean of scalar observations, and nothing chooses an intervention from it.

## H. Components not to carry forward

Preserved on disk. Excluded from the integrated loop.

| component | reason |
| --- | --- |
| Status and inflation as the update | They change the wrapper and leave the consumed claim on `g`. |
| `surface_family = UNVERIFIED` | Stored, and not a scope filter. |
| M26's external `delta` | Unestablished, and not written onto the object a decision would read. |
| M27's gain operator | Unestablished on the real cells. A coefficient on `g` is not the missing operation. |
| `SelfModelBelief` as the architecture | Scalar tracker. M23 closed-loop improvement is outside that result. |
| A further field on `MechanismBelief` | Extends the wrapper. |
| " `rho` predicts the residual" as the definition of success | M29 already recorded that. It does not close the loop. |
| Layer 23, a second response rule inside `mechanism_belief.py`, and any refit of D1 | Rejected by the frozen response object and its tests. |

## I. Minimum architecture

```text
SelfModel
  claim identity          USE_LINEAR | WITHHOLD
  measurement binding     g, sealed before action
  scope                   which episodes may update or be decided
  evidence history        sealed prediction paired with a later observation
  update history          claim before, evidence, claim after
        |
        v
prediction
  issued from the current claim
  closed before the intervention
        |
        v
decision
  pure function of claim identity
  USE_LINEAR -> INTERVENE
  WITHHOLD   -> ABSTAIN
        |
        v
intervention
  runs only if the decision is INTERVENE
  ABSTAIN applies nothing
        |
        v
observation
  read only after that episode's prediction is sealed
        |
        v
evidence
  predicted, observed, signs, residual
  episode id must be in the evidence scope
        |
        v
self-model update
  opposite nonzero signs -> claim WITHHOLD
  otherwise              -> claim unchanged
        |
        v
next decision
  the updated claim, on an episode outside the evidence set
```

The control arm is the same object and the same evidence, with the update step removed. Its claim stays `USE_LINEAR`. Its next decision stays `INTERVENE`.

`MechanismResponseModel` remains the source of `g` when the measurement is the real directional derivative. It is not the self-model. `MechanismBelief` remains the record that evidence can mark `prediction = g` contradicted. It is not the self-model.

## J. Single acceptance test

Name: closed-loop claim consumption.

The test fixture builds two self-models from the same initial claim `USE_LINEAR`, and two episode sets declared before any observation is read.

**Arm U.** Evidence is applied with the update rule in section D.

**Arm C.** The same evidence is recorded, and the update rule is not applied.

**Arm A.** A fresh self-model receives an evidence episode whose observed sign agrees with the sealed prediction. This arm exists so a change in Arm U can be attributed to the contradictory evidence.

Procedure:

1. Before any action, both Arm U and Arm C exist, and both carry `USE_LINEAR`.
2. The decision on that claim returns `INTERVENE`.
3. That decision is what causes the intervention. The hook, or its synthetic stand-in, does not run on `ABSTAIN`.
4. The evidence episode's prediction, including `g` and the claim identity, is sealed. The effect is read after that seal.
5. Arm U's update consumes that pair. The history records `USE_LINEAR` before and `WITHHOLD` after.
6. On the later episode, Arm U's decision returns `ABSTAIN`. Arm U's sealed prediction for that episode is the prediction of the `WITHHOLD` claim, which is that the intervention will not be applied.
7. Arm A, given sign-agreeing evidence, still carries `USE_LINEAR` and still decides `INTERVENE` on its own later episode. The change in Arm U is therefore attributable to the contradictory evidence.
8. Arm C, with the update removed, still carries `USE_LINEAR` and decides `INTERVENE` on the same later episode Arm U abstains on.
9. The later episode id is absent from the evidence episode ids. Its effect is not read by the update.

A pass is the conjunction of these nine conditions. Accuracy of `g`, identification of a direction, a residual correlation, and the existence of a belief object are not clauses of the test.

## K. Result labels

No error threshold, interval, or sample-size cutoff is part of these labels. The sign boundary is the one already defined by `_sign`.

**`SELF_MODEL_OPERATIONAL`**

All nine conditions in section J hold, including the ablation contrast and the held-out episode.

**`SELF_MODEL_NOT_OPERATIONAL`**

Any one of the following:

- the decision does not read the claim
- the intervention runs when the decision is `ABSTAIN`, or does not run when the decision is `INTERVENE`
- the evidence episode's effect is readable before its prediction is sealed
- after a nonzero sign disagreement, Arm U and Arm C choose the same action on the later episode
- the later episode was a member of the evidence set

**`INCONCLUSIVE`**

The seal, the decision, and the episode split are intact, and the evidence packet does not meet the update's precondition: the signs agree, or either sign is zero. A correct update then leaves the claim and the later decision unchanged. That is an uninformative packet, not a failed loop.

An empty later set, or an evidence set that overlaps it, is `SELF_MODEL_NOT_OPERATIONAL` when the split was required and broken, and `INCONCLUSIVE` when the split was never successfully declared.

## L. Shortest path from this repository

1. Keep every M22.1–M29 file and artifact unchanged. This report is the specification of the acceptance test. It is not that test, and it does not authorize writing it.
2. When an implementation is separately authorized, add one new object that owns claim identity, one decision function of that identity, one synthetic episode fixture, and one test of the nine conditions. Do not edit `mechanism_response.py`, `mechanism_belief.py`, `residual_update.py`, `m29_pre_evidence.py`, or `m23/belief.py`.
3. Run that test on the synthetic fixture only. Qwen, D1, and the three residual hooks stay off this path until the synthetic result is `SELF_MODEL_OPERATIONAL`.
4. Only after that result may a later specification ask whether the decision should also read `rho`. That specification would be a new document. M29's support for `rho` does not create it, and this audit does not write it.

The path is one closed loop on a synthetic two-action world. It is not another coefficient, another status label, or another holdout correlation.
