# P2.2 operational definition resolution

## 1. Executive result

Q.H1 now has an observable diagnostic vector. H3 now has three named representation families and a paired metric vector. Neither has a numerical pass bar. `DEC-010-A1` is still `PROPOSED / NOT APPLIED`. The scientific contract stays `DRAFT / NOT FROZEN`.

MRSM remains NOT READY FOR SCIENTIFIC EXECUTION.

## 2. Repository preflight

Inspection is in `reports/P2_2_repository_preflight.md`.

At inspection the branch was `cursor/p2-1-decision-resolution-4e19` at `05099658b4de8e0a59b0b0716fed130c1df2ec37`, with a clean worktree. This phase is `cursor/p2-2-operational-definitions-4e19`.

P2.1's five decisions are present and were not reopened: DEC-010 binds the terminal tree; P.H1 is truth-validating; Q.H1 is diagnostic/transfer only; the planted mechanism is L0H0 → L1H1; run accounting follows DEC-011.

The implementation specification still disagrees with those decisions. It was not edited. The disagreement is historical and remains non-authoritative.

## 3. Q.H1 definition

Q.H1 asks whether a candidate self-model on Q can issue intervention-linked predictions that are then checked against executed interventions. Each holdout claim records `claim_id`, a representation or mechanism hypothesis, the intervention, predicted behavioral delta, predicted direction, uncertainty, applicable regime, evidence references, and falsification status.

The result is a vector, not a hidden scalar:

- prediction quality (sign correctness and MAE, recorded separately)
- counterfactual consistency
- falsification behavior
- abstention quality, where uncertainty or applicability is recorded
- scope consistency
- `status: DIAGNOSTIC_ONLY`

The machine-readable copy is `configs/mrsm_operational_contract_v0.1.yaml` under `q_h1.evaluates`. `thresholds_status` is `DEFERRED`.

## 4. Q.H1 epistemic boundary

```text
P.H1 = ground-truth-validating discovery gate
Q.H1 = diagnostic/transfer gate
```

```text
Q.H1 PASS ≠ verified mechanism discovery on Q
Q.H1 FAIL ≠ proof that the method cannot discover mechanisms on real models
```

A Q failure is evidence against the current transfer under the tested scope. Q has no mechanism ground truth. DEC-011 records that as `UNKNOWN`. The operational file records `ground_truth_available: false` and `verifies_mechanism_identity: false`.

## 5. Q.H1 leakage boundary

Prohibited on Q: hidden mechanistic labels, manually selected correct heads, post-hoc mechanism labels, unpublished circuit annotations, outcome-derived mechanism identity labels, and test intervention outcomes used to build the pre-intervention prediction.

Train and discovery information may be used to fit the hypothesis. Locked holdout information may be the pre-intervention input. Post-intervention outcomes may be used only after the prediction for that `claim_id` exists. The order must be visible in the run artifact. This phase creates no such artifact.

## 6. H3 definition

H3 tests whether predicted causal consequence, mechanism identity at the declared abstraction level, scope, and falsification behavior stay equivalent when the representation changes by an allowed invertible map and the causal system does not. Raw vectors are not required to stay identical. Raw parameters are not required to stay identical.

M18.7 and A4 are not the justification. The old phase-rotation experiment does not validate H3. The specification's `[X, Y] -> [Y, -X]` formula is not adopted.

## 7. H3 transformation family

Named in the draft, without parameters:

| Id | Family | Map |
| --- | --- | --- |
| T1 | `invertible_affine` | `x' = A x + b`, `A` invertible |
| T2 | `coordinate_permutation` | `x' = P x`, `P` a permutation matrix |
| T3 | `orthogonal_basis_change` | `x' = Q x`, `Q^T Q = I` |

Each map must be invertible, known, semantics-preserving, chosen without test outcomes, free of its own tuned threshold, intervention-identical, and scored on locked holdout cases. A map that changes the causal system is rejected. `A`, `b`, `P`, and `Q` are not chosen. This naming does not amend DEC-011 and does not freeze a PASS combination.

## 8. H3 metrics

Paired comparison of the original representation and the transformed representation on the same locked holdout cases. The record contains prediction equivalence, sign agreement, normalized prediction error difference, mechanism-id consistency at the declared abstraction level, scope consistency, uncertainty consistency, and falsification consistency. No single H3 score. No numeric cutoff.

## 9. P/Q boundary

P ground truth is the frozen L0H0/L1H1 mechanism. Q ground truth is absent. Discovery and mechanism identity can be validated on P only. Q prediction is an empirical behavioral check. Representation invariance is testable on both, and on Q it is not an identity verdict. Transfer to Q is diagnostic. The table is in the contract section `P/Q Epistemic Boundary`.

## 10. DEC-010-A1 proposal status

`PROPOSED / NOT APPLIED`. Text: `reports/P2_2_DEC010_A1_final_proposal.md`.

The eight keys and the terminal structure stay as they are in DEC-010. The proposal adds the diagnostic reading of Q.H1. GO eligibility is unchanged. DEC-010 was not edited.

## 11. Deferred decisions

- Numerical H1–H4 thresholds
- Approval and application of `DEC-010-A1`
- Preregistration freeze
- Qwen execution, still waiting on the pinned offline snapshot
- `p_freeze`
- Locked holdout membership
- Concrete T1–T3 parameters and the H3 PASS combination

## 12. Files changed

- `reports/P2_2_repository_preflight.md`
- `reports/P2_2_DEC010_A1_final_proposal.md`
- `reports/P2_2_operational_definition_resolution.md`
- `reports/MRSM_Scientific_Contract_v0.1.md`
- `configs/mrsm_operational_contract_v0.1.yaml`
- `scripts/validate_mrsm_operational_contract.py`
- `tests/test_mrsm_operational_contract.py`
- `control_plane/FILE_MAP.yaml`

Not changed: DEC-010, DEC-011, `P_SPEC.md`, `MRSM_BUDGET.yaml`, `configs/m22_1_r_replay.yaml`, the replay harness, and the execution record.

## 13. Tests executed

`PYTHONPATH=src python3 -m pytest tests/test_mrsm_operational_contract.py tests/test_control_plane.py tests/test_decision_table.py -q`

45 passed. The new file covers a valid contract, rejection of ground-truth verification, rejection of role `DISCOVERY_VERIFICATION`, deferred thresholds, the three H3 families, a missing family, P true / Q false, and rejection of inserted numeric thresholds.

No Qwen load and no MRSM run.

## 14. Validator result

`python3 scripts/validate_mrsm_operational_contract.py` printed `operational contract OK` and exited 0.

`PYTHONPATH=src python3 -m cognitive_self_model.control_plane.validate` exited 0. The only warning is the pre-existing line `FILE_MAP missing older tracked path reports/mrsm_p_v1/checkpoint_mask_audit.json`.

## 15. Git commit

Parent: `05099658b4de8e0a59b0b0716fed130c1df2ec37`.

Message: `P2.2: operational definition resolution`.

Branch: `cursor/p2-2-operational-definitions-4e19`.

This report is part of that single commit.

## 16. Scientific claims that are now allowed

None as results. The draft allows a future run, after freeze, to *describe* Q.H1 as a diagnostic vector and H3 as paired equivalence under T1–T3. It does not allow those descriptions to be reported as completed findings.

## 17. Scientific claims that remain prohibited

- Q.H1 verified a Qwen mechanism.
- Q.H1 failure proves mechanism discovery is impossible.
- H3 was validated by M18.7 or A4.
- Representation invariance alone establishes a self-model.
- P success is Q success.
- Any H1–H4 numerical threshold is frozen.
- A GO, STOP, or REDEFINE outcome for MRSM.
- The missing Qwen snapshot is a scientific failure.

## 18. Explicit statement

MRSM remains NOT READY FOR SCIENTIFIC EXECUTION.
