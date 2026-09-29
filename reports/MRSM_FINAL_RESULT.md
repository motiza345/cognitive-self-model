# MRSM final result

The preregistered scientific run did not execute. P construction failed the frozen causal predicate, so DEC-011 stops the program before `p_freeze` and before any locked-holdout read.

## 1. Preregistration identity

`configs/mrsm_prereg.yaml`

- id: `mrsm-prereg-2026-09-29`
- status: `PREREGISTERED`
- frozen_on: `2026-09-29T16:55:00Z`
- This freezes thresholds, H3 parameters, splits, baselines, and the statistical procedure.
- It does not bind a successful checkpoint. `model.arm_p_checkpoint_sha256` remains null because `p_freeze` did not occur.

## 2. P result

Engineering outcome: `FAIL`

Budget outcome: `P_CONSTRUCTION_INCOMPLETE`

Checkpoint sha256: `ea40cda33a054c1c7fd9b169686c2eff6a9ada1aab9f0d75324fbdd74a8986e6`

That hash and the development audit match the already recorded `MRSM_P_v1_smoke` attempt. The rerun used the frozen P_SPEC hyperparameters (`n_steps: 1600`, seed 1729) and did not read a holdout.

Development audit, training mask removed:

| check | value | predicate |
| --- | --- | --- |
| intact accuracy | 0.318359375 | `>= 0.95` failed |
| hop1-only accuracy | 0.30859375 | not evaluated as a pass; intact already failed |
| hop2-only accuracy | 0.24609375 | same |
| both planted heads ablated | 0.23046875 | same |
| specificity gap | 0.07161458333333331 | same |
| sequential signature | false | construction DoD failed |

Train accuracy on the construction split was 0.492. The optimizer ran all 1600 steps. The instrument does not implement the planted mechanism at the required accuracy.

No holdout metric exists. None is inferred.

## 3. Q result

Not attempted. DEC-011 says MRSM does not start after `P_CONSTRUCTION_INCOMPLETE`. Q was not downloaded and not scored.

## 4. H1

`NOT_RUN`

## 5. H2

`NOT_RUN`

## 6. H3

`NOT_RUN`

## 7. H4

`NOT_RUN`

## 8. Baseline comparison

`NOT_RUN`

## 9. Leakage audit

`NOT_RUN`. The scientific runner refuses to start unless construction `budget_outcome` is `DONE`.

## 10. Failures

P construction failed an engineering predicate in `control_plane/P_SPEC.md`. This is not an H1–H4 scientific failure and it is not evidence against the MRSM hypothesis. The static-weight detector remains `NOT_SEPARATELY_SPECIFIED` and was neither passed nor failed.

Continuing would require a new DEC. The construction clock was not restarted. `n_steps`, the architecture, and the planted heads were not changed after seeing the audit.

## 11. Scope

The failure scope is the frozen P_SPEC v1 construction: two-layer HookedTransformer, training mask on L0H0 and L1H1, 1600 AdamW steps, development audit. It does not speak to Qwen.

## 12. Non-claims

- No claim that a self-model discovered L0H0 → L1H1.
- No claim that it failed to discover that edge on holdout.
- No claim that Q discovered a mechanism.
- No claim that representation invariance passed or failed.
- `Q.H1` was not recorded as `PASS`.

## 13. DEC-010 terminal decision

Not applied. There is no eight-key gate record. `p_freeze` is absent. `mrsm.clock_start_commit` is null. Full runs used: 0. Q GPU full runs used: 0.

DEC-011 consequence of this construction outcome: MRSM does not start.
