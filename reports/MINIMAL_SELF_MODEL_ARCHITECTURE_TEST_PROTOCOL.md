# Minimal Self-Model Architecture Test

Version: `MINIMAL_SELF_MODEL_ARCHITECTURE_TEST.1`

This protocol is sealed before the single execution. It does not use Qwen, M22–M29, or the self-model evaluation suite.

## Research question

Does a self-relevant claim registry, with explicit scope and epistemic status, change later decisions in a way that a two-state flag does not, when the only added structure is the registry itself?

## Hypothesis

A SelfModel is a registry of self-relevant claims, not a predictor and not a decision policy.

Richer self-relevant representation leads to scope-aware evidence interpretation, then to appropriate revision, then to a better decision.

Prediction is a consumer of claims. Decision is outside the SelfModel.

## Environment

The benchmark generator has three hidden states and one intervention.

| State | `ACTION_X` ground-truth effect |
| --- | --- |
| `STATE_A` | `+1.0` |
| `STATE_B` | `+0.25` |
| `STATE_C` | `-1.0` |

`ABSTAIN` has effect `0`.

Outcomes are deterministic. No random draw is used.

`measured_state` is an observable measurement. Every agent receives the same `measured_state` on every episode. The ground-truth effect is not an agent input. The scorer alone uses it.

Random seed `32001` is reserved and is not consumed.

## Evidence classifier

The classifier reads only the evidence outcome.

| Outcome | Class | Strength | Resulting status | Uncertainty |
| --- | --- | --- | --- | --- |
| `> 0.5` | `CONFIRMING` | `strong` | `SUPPORTED` | `0.15` |
| `> 0` and `<= 0.5` | `CONFIRMING` | `weak` | `SUPPORTED` | `0.35` |
| `== 0` | `AMBIGUOUS` | `none` | `UNCERTAIN` | `0.70` |
| `< 0` | `CONTRADICTORY` | `strong` | `CONTRADICTED` | `0.90` |

`CONTRADICTED` is absorbing for the claim that reached it. Later evidence does not restore that claim. Evidence is still appended to the model evidence history.

`UNCERTAIN` is not absorbing. Later confirming evidence can set `SUPPORTED`.

Ambiguous evidence must not write `CONTRADICTED`.

## Shared proposition

Mechanism proposition, for every scoped claim:

`ACTION_X produces a positive self-effect`

## Agents

### B0

Fixed policy, chosen here: always `ACTION_X`.

B0 receives the same measurements and stores none of them.

### B1

Legacy two-state flag.

`claim ∈ {USE_LINEAR, WITHHOLD}`

Initial claim: `USE_LINEAR`.

| Evidence class | Update |
| --- | --- |
| `CONFIRMING` | remain `USE_LINEAR` |
| `AMBIGUOUS` | `WITHHOLD` |
| `CONTRADICTORY` | `WITHHOLD` |

`WITHHOLD` is absorbing. B1 has no scope, no uncertainty, and no recovery.

Decision, external to the flag:

| Claim | Decision |
| --- | --- |
| `USE_LINEAR` | `ACTION_X` |
| `WITHHOLD` | `ABSTAIN` |

### B2

Claim registry.

Fields on the model: `claims`, `version`, `evidence_history`, `revision_history`.

Fields on a claim: `claim_type`, `proposition`, `scope`, `status`, `evidence`, `uncertainty`, `revision_history`, `version`.

`claim_type` is `MECHANISM` in this test.

Scope is the measured state. A claim is applicable only when its scope equals the current `measured_state`. No claim for that state means there is nothing to apply.

Epistemic status is one of `SUPPORTED`, `UNCERTAIN`, `CONTRADICTED`. Status is not an action.

External decision policy:

| Applicable status | Decision |
| --- | --- |
| `SUPPORTED` | `ACTION_X` |
| `UNCERTAIN` | `ABSTAIN` |
| `CONTRADICTED` | `ABSTAIN` |
| no applicable claim | `ABSTAIN` |

The decision function may read only the applicable claim. It must not branch on the state name.

### B2_NO_SCOPE

Same code and same policy as B2.

The only difference: every claim is stored and read under scope `GLOBAL`. Measured state does not select a claim.

## Episode order

Updates are evidence. Probes are decisions. A probe outcome is not written into any model.

| Order | Id | Role | Measured state | Evidence outcome |
| --- | --- | --- | --- | --- |
| 1 | `U1` | update | `STATE_A` | `+1.0` |
| 2 | `P1` | probe | `STATE_A` | none |
| 3 | `P2` | probe | `STATE_C` | none |
| 4 | `U2` | update | `STATE_A` | `0.0` |
| 5 | `P3` | probe | `STATE_A` | none |
| 6 | `U3` | update | `STATE_A` | `-1.0` |
| 7 | `P4` | probe | `STATE_A` | none |
| 8 | `U4` | update | `STATE_B` | `+0.25` |
| 9 | `P5` | probe | `STATE_B` | none |
| 10 | `P5b` | probe | `STATE_B` | none |
| 11 | `P6` | probe | `STATE_A` | none |
| 12 | `P7` | probe | `STATE_C` | none |

`U1` is confirming evidence for `STATE_A`.

`U2` is ambiguous evidence for `STATE_A`.

`U3` is contradictory evidence for `STATE_A`.

`U4` is weak confirming evidence for `STATE_B`.

`P2` is the move to `STATE_C` while the `STATE_A` claim is still the only mechanism claim.

`P7` is the later move to `STATE_C` with no `STATE_C` evidence.

`P5b` repeats `P5` with no intervening evidence.

## Correct action

On a probe, `ACTION_X` is correct when the ground-truth effect is positive. `ABSTAIN` is correct when the ground-truth effect is negative.

This uses the environment table, not the evidence packet. `U2` and `U3` are evidence, and they are not changes to the ground-truth table.

## Primary endpoint

Correct decision rate on held-out state transitions.

A probe is primary when its measured state differs from the measured state of the most recent update.

Under the episode order above, that rule selects `P2`, `P6`, and `P7`.

The rate is the number of primary probes whose decision equals the correct action, divided by the number of primary probes.

Comparisons, in order:

1. `B2` versus `B2_NO_SCOPE`
2. `B2` versus `B1`

`B0` is reported and is not required for the verdict.

Strict inequality is required. An equal rate is not a win.

## Secondary endpoints

### Scope

Every `ACTION_X` decision by B2 must cite a claim whose scope equals the measured state.

B2 must not cite the `STATE_A` claim on `STATE_C`.

`U4` must not change the `STATE_A` claim's status, uncertainty, or version.

### Revision

After `U1`: B2 scope `STATE_A` is `SUPPORTED`, uncertainty `0.15`, and `P1` is `ACTION_X`.

After `U2`: the same claim remains, status is `UNCERTAIN`, uncertainty is `0.70`, status is not `CONTRADICTED`, and `P3` is `ABSTAIN`.

After `U3`: status is `CONTRADICTED`, uncertainty is `0.90`, and `P4` is `ABSTAIN`. `P4` differs from `P1`.

`UNCERTAIN` and `CONTRADICTED` share the action `ABSTAIN`. The status values must still differ.

After `U4`: `STATE_A` is unchanged and still `CONTRADICTED`. A separate `STATE_B` claim is `SUPPORTED` with uncertainty `0.35`.

### Decision causality

`P1` and `P3` are the same state. The applicable status changes from `SUPPORTED` to `UNCERTAIN`, and the decision changes from `ACTION_X` to `ABSTAIN`.

`P3` and `P4` are the same state. The status changes from `UNCERTAIN` to `CONTRADICTED`. Those statuses are not decision-distinct, so the decision stays `ABSTAIN`.

`P5` and `P5b` have no intervening evidence. The decision stays the same.

### Historical immutability

Evidence records and sealed probe records are append-only. A later update must not change the stored bytes of an earlier record.

### Representation

After `U4`, B2 simultaneously has:

- `STATE_A` current status `CONTRADICTED`, with `SUPPORTED` retained in that claim's revision history
- `STATE_B` as a separate current claim
- no `STATE_C` claim

### Bypass probe

After the episode list, on a fresh registry, not on the episode model:

- an empty registry abstains in every state
- a single `SUPPORTED` claim scoped to `STATE_C` selects `ACTION_X` in `STATE_C` and `ABSTAIN` in `STATE_A` and `STATE_B`

This probe does not enter the primary rate.

## Falsification

The architecture fails if any of these is true.

1. B2 cannot hold different current claims for different states.
2. B2's probe decisions are identical to B1's.
3. B2 and `B2_NO_SCOPE` make the same decisions on every primary probe.
4. The bypass probe fails.
5. B2's packet contains a field that B1 and `B2_NO_SCOPE` do not receive. Ground truth and correct action are scorer fields.
6. The run never records `SUPPORTED`, `UNCERTAIN`, and `CONTRADICTED` as distinct statuses.
7. A sealed evidence record or probe record changes after it is written.

## Leakage

- Probe rows do not enter `evidence_history`.
- Agent packets contain `evidence_id`, `measured_state`, `action`, and `outcome` on updates. They do not contain the ground-truth effect or the correct action.
- The same update packet is passed to B1, B2, and `B2_NO_SCOPE`.
- Primary membership follows the state-transition rule above. It is not changed after the run.
- One execution. If the results file already exists, the runner stops.

## Verdict

`ARCHITECTURE_SUPPORTED` only if all of these hold:

- primary rate of B2 is strictly greater than `B2_NO_SCOPE`
- primary rate of B2 is strictly greater than B1
- revision checks pass
- scope checks pass
- leakage checks pass
- the bypass probe passes
- none of the falsification conditions hold

`ARCHITECTURE_NOT_SUPPORTED` if any falsification condition holds, or B2's primary rate is not strictly greater than `B2_NO_SCOPE`, or B2's decisions are not causally dependent on claim status.

Otherwise `INCONCLUSIVE`.
