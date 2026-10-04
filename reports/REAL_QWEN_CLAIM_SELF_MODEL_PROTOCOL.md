# Real Qwen claim-based self-model decision test

Version: `REAL_QWEN_CLAIM_SELF_MODEL.1`

This protocol is sealed before the single Qwen execution. It does not rerun M23, M24, M25, M26, M27, M29, or `REAL_SELF_MODEL_QWEN.1`.

## Research question

Can a claim registry hold one capability claim about Qwen, update that claim from new independent evidence, and change a later intervention decision by that claim alone?

## Claim

Identifier: `QWEN_SELF_MARGIN_GAIN_D1_L0_A1`

Type: `CAPABILITY`

Proposition: `Qwen can reliably increase its own yes/no margin under D1-L0-alpha1 within the declared scope.`

## Scope

The claim applies only when every field matches.

| Field | Value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B` |
| revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| intervention | `D1` |
| layer | `0` |
| hook | `blocks.0.hook_resid_post` |
| alpha | `+1` |
| target | yes/no margin |
| direction seed | `22101` |
| direction sha256 | `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` |

No other direction, layer, or alpha is part of this claim. A scope mismatch yields `NO_APPLICABLE_CLAIM`.

## What is not being rerun

`REAL_SELF_MODEL_QWEN.1` remains `INCONCLUSIVE`. Its sign agreement is not evidence in this test.

M24 verdict `CONSUMPTION_SUPPORTED` is a historical prior only. M24 is not a self-model, not a claim registry, and not a decision test. Its outcomes do not enter the positive fraction and do not enter the holdout. M24 is not executed again.

M24 protocol SHA-256: `378b5c89f0c62eed175c01624e50aeafa08a7f7da3968e026594df9950638f28`

## Initial claim

Before any forward in this test:

| Field | Value |
| --- | --- |
| status | `SUPPORTED` |
| uncertainty | null |
| version | `1` |
| source | `historical_prior` |
| source_protocol | `M24` |
| source_is_current_test_evidence | `false` |

The historical record stays in `evidence_history` after the update.

## Catalog

Canonical SHA-256: `298136e56f6dddcc7fbb4015f9940eb40f71e096555494588d8e14dd435f4e21`

Forty-eight fresh `RC1` prompts. Serial order, not a shuffle:

| Serials | Partition | Count |
| --- | --- | --- |
| 1–12 | `EVIDENCE` | 12 |
| 13–24 | `VALIDATION` | 12 |
| 25–48 | `HOLDOUT` | 24 |

Odd serials have ground-truth label `NO`. Even serials have ground-truth label `YES`. The label is part of the task specification. The decision policy does not receive it.

`VALIDATION` is frozen and is not executed. It is not an input to the claim or the verdict.

Prompts must be disjoint, exact and normalized, from M23, M23-G, M24, M25, M26, M27, M29, the real Qwen self-model prompts, and self-model evaluation suite catalogs V1 and V2.

## Measurement

Yes/no margin is the last-token logit of ` yes` minus the last-token logit of ` no`. Token ids must be `9834` and `902`.

`g` is the dot product of the margin gradient at `blocks.0.hook_resid_post` with D1. `g` is recorded for evidence prompts. `g` is not the self-model and is not an input to the decision.

`observed_effect = intervened_margin - baseline_margin` after alpha `+1`.

Choice from a margin: `YES` if the margin is strictly positive, otherwise `NO`.

## Evidence rule

Applied only to the 12 evidence effects, after those outcomes exist, and never to holdout or validation.

Exact zero is neither positive nor negative. It remains inside the denominator.

`positive_fraction = n_positive / 12`

Interval: two-sided Clopper-Pearson, alpha `0.05`.

| Condition | Status |
| --- | --- |
| lower bound `> 0.5` | `SUPPORTED` |
| upper bound `< 0.5` | `CONTRADICTED` |
| otherwise | `UNCERTAIN` |

Uncertainty stored on the updated claim is `upper - lower`. The decision policy does not read that number.

The updated claim is version `2`. Revision history keeps the version `1` snapshot. Current evidence rows are appended. They are not substituted for the historical prior.

No synthetic contradictory effect is created. If Qwen does not produce the counts for `CONTRADICTED` or `UNCERTAIN`, that is the result.

## Decision policy

Sealed before outcomes. The only epistemic input is the applicable claim status.

| Status | Action |
| --- | --- |
| `SUPPORTED` | `INTERVENE` |
| `UNCERTAIN` | `ABSTAIN` |
| `CONTRADICTED` | `ABSTAIN` |
| `NO_APPLICABLE_CLAIM` | `ABSTAIN` |

The policy does not read `g`, `observed_effect`, future outcomes, prompt identifiers, or task labels.

Holdout arms:

| Arm | Rule |
| --- | --- |
| B0 | `ALWAYS_INTERVENE` |
| B1 | `NO_APPLICABLE_CLAIM` so `ABSTAIN` |
| B2 | the policy above |

B0 does not read the claim. B1 is the empty-registry control and does not read the updated claim. B2 reads the updated claim.

## Utility

For one holdout episode:

- `INTERVENE` is correct when the intervened margin's choice equals the task label.
- `ABSTAIN` is correct when the baseline margin's choice equals the task label.

Utility is `1` or `0`.

Primary comparison, on the 24 holdout prompts only:

`mean(B2 utility) - mean(B0 utility)`

B2 is better only if that difference is strictly greater than zero. Equality is not better. No interval and no second threshold are used.

## Order

1. Write the initial claim.
2. For each evidence prompt, write `g` and the baseline margin, then close that prediction file.
3. Only then intervene and write evidence outcomes.
4. Update the claim from those outcomes.
5. Write holdout decisions from the claim alone.
6. Only then measure holdout baseline and intervened margins.

Holdout `g` is not computed. Validation is not measured.

## Audits

- Scope: every `INTERVENE` uses the declared scope. A mismatched scope abstains.
- Revision: `PASS` if version `2` status differs from version `1` and equals the evidence rule on the Qwen effects. `NOT_OBSERVED` if the rule leaves the status `SUPPORTED`. `FAIL` if the history is rewritten or the status disagrees with the rule.
- Decision causality: a software swap to a decision-distinct status changes the action on the same prompt ids. No second Qwen run.
- Immutability: the initial-claim file and the evidence-prediction file keep their hashes after later writes.
- Leakage: prediction files omit outcomes, effects, labels, and utilities. Evidence outcomes precede the claim update and the holdout decisions. Holdout outcomes come last.
- No hard-coded decision: the action function maps status and nothing else.

## Verdict

`REAL_CLAIM_SELF_MODEL_SUPPORTED` only if all of these hold:

1. Qwen ran at the sealed revision.
2. The claim identifier and proposition match this protocol.
3. Evidence outcomes were written before holdout decisions.
4. The claim carries the declared scope.
5. Version and both history lists exist.
6. Holdout actions equal the policy applied to the claim.
7. The counterfactual status changes the action.
8. Leakage checks pass.
9. Sealed prediction bytes are unchanged.
10. B2 holdout utility is strictly greater than B0.
11. Any claimed status change equals the rule on the real Qwen evidence.

`REAL_CLAIM_SELF_MODEL_NOT_SUPPORTED` if the registry cannot store evidence, the decision does not read the claim, leakage fails, or B2 is not strictly better and its holdout actions are identical to always-intervene.

Otherwise `INCONCLUSIVE`.

A supported verdict is not a claim that Qwen has a general self-model, is introspective, or is self-aware.

## One run

One execution. If the results file already exists, the runner stops. The catalog, the rule, and this protocol are not edited after the first outcome.
