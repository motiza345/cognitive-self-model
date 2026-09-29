# P2.1 decision resolution

Date: 2026-09-29.

Working tree parent: `d9e4cc363dcf9bf4b07698b5fcb9bbebcf69fb58` on `cursor/p2-0-contract-reconciliation-4e19`.

This phase resolves the five conflicts recorded in `reports/P2_0_contract_conflicts.md` and `reports/MRSM_Scientific_Contract_v0.1.md`. It does not edit `DEC-010`, `DEC-011`, or `P_SPEC.md`. It does not apply any amendment. It does not freeze a preregistration.

## 1. Purpose

Name the frozen rule that governs each conflict, state where an amendment is required before that rule can be read safely, and leave every unapproved change unapplied.

The implementation specification remains a proposal. Where it disagrees with a frozen DEC, the DEC governs. Correcting the specification is a later documentation edit. It is not done in this phase.

## 2. Authority hierarchy

| Rank | Source | What it governs here |
| --- | --- | --- |
| 1 | `control_plane/DEC-010_decision_table.md`, status FROZEN, `decision_table_version: 1` | Terminal outcomes and how the eight gate keys combine |
| 1 | `control_plane/DEC-011_mrsm_scope.md`, status FROZEN, `decision_version: 1` | Arms, ground-truth status, construction clock, MRSM clock, full-run definition, Q GPU cap, M18.7 prohibition. It does not modify DEC-010 |
| 1 | `control_plane/P_SPEC.md`, status FROZEN | Planted task and the operational mechanism |
| 2 | `control_plane/MRSM_BUDGET.yaml` | Live counters. `governed_by: DEC-011`. Not a new decision |
| 3 | `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` | Proposal. Non-authoritative where it conflicts with rank 1 |
| 3 | `configs/m22_1_r_replay.yaml` | M22.1-R replay vocabulary only. Not an H1–H4 or terminal-tree source |

`STATE.yaml` does not list DEC-010 or DEC-011. That ledger gap does not demote the frozen files. This phase does not edit `STATE.yaml`.

## 3. Decision 1 — terminal tree

Status: `RESOLVED`

DEC-010 is the binding terminal decision authority. The implementation specification does not define an alternative terminal tree.

Reconciliation note:

> The implementation specification is subordinate to DEC-010 for terminal decision semantics. Any conflicting GO/REDEFINE/STOP wording in the implementation specification is non-authoritative and must be corrected before preregistration freeze.

DEC-010 was not edited.

The binding outcomes remain exactly the DEC-010 set:

- `STOP`
- `REDEFINE_CAUSAL_AUDIT`
- `REDEFINE_SCALE`
- `INCOMPLETE_BUDGET`
- `GO`
- `CONTRACT_ERROR`

GO still requires `P.H1`, `P.H2`, `P.H3`, `P.H4`, `Q.H1`, `Q.H2`, `Q.H3`, and `Q.H4` all to be `PASS`. The FIRST_MATCH rules in DEC-010, including `P.H2` FAIL cases that return `STOP` and `P.H4` FAIL cases that return `REDEFINE_CAUSAL_AUDIT`, stay in force.

Specification sections that must be corrected before preregistration freeze, because they state another tree:

| Section | Conflicting sentence | Required later correction |
| --- | --- | --- |
| §1.2 | "Exactly three terminal outcomes are allowed" and "No fourth outcome may be invented." | Replace with the DEC-010 outcome set. `INCOMPLETE_BUDGET` and `CONTRACT_ERROR` are already in the frozen table |
| §24 STOP | STOP if H1-P fails, and no other STOP leaf | State the DEC-010 STOP rules, including the stated `P.H2` FAIL cases |
| §24 first REDEFINE | H1-P and H2-P pass and H3-P fails, outcome name `REDEFINE` | Use `REDEFINE_CAUSAL_AUDIT` |
| §24 second REDEFINE | "If P passes but Q fails", outcome name `REDEFINE` | Use `REDEFINE_SCALE`, and include `Q.H1` among the Q keys |
| §24 GO | GO without `Q.H1` PASS | GO only when all eight keys are `PASS` |
| §24 | No leaf for P.H4 failure | State DEC-010 rule 3: `P.H1`–`P.H3` PASS and `P.H4` FAIL is `REDEFINE_CAUSAL_AUDIT` |
| §0 | Uses `INCOMPLETE_BUDGET` for an unassessed gate | The name matches DEC-010. The surrounding claim in §1.2 that a fourth outcome is forbidden must be removed so this bullet is not read against §1.2 |

§0's use of the name `INCOMPLETE_BUDGET` is not itself a conflict with DEC-010. §1.2's ban on a fourth outcome is the conflict.

No DEC amendment is required for Decision 1. The frozen table already governs. The specification text is what must change later.

## 4. Decision 2 — Q.H1

Status: `RESOLVED_WITH_AMENDMENT_REQUIRED`

```text
P = ground truth known
Q = ground truth unknown
```

```text
P.H1 → truth-validating discovery test
Q.H1 → diagnostic / transfer evidence
```

P.H1 compares a discovered object to the frozen P ground truth in `P_SPEC.md`.

Q.H1 may record whether the same discovery procedure produces internally coherent or externally testable mechanism hypotheses on Q. `Q.H1 PASS` is not verified mechanism discovery. `Q.H1 FAIL` is not proof that the method cannot discover mechanisms, unless an independently grounded criterion supports that stronger sentence. No such criterion is frozen. DEC-011 records Q `ground_truth: UNKNOWN`. Specification §2, §9, §24, and rule 8 already forbid presenting Q.H1 as verified mechanism identity. Those specification sentences are accepted as the epistemic role. They are not accepted as a license to drop `Q.H1` from DEC-010.

DEC-010 lists `Q.H1` as one of the eight keys. GO requires `Q.H1` PASS. A missing key is `CONTRACT_ERROR`. An evaluated `Q.H1` FAIL, once every Q key is evaluated and every P key is PASS, is `REDEFINE_SCALE`.

That combinatorics can stay. The epistemic meaning cannot stay implicit. DEC-010 gives `P.H1` PASS and `Q.H1` PASS the same formal place in the GO rule and contains no sentence that `Q.H1` PASS is not mechanism verification. Treating the diagnostic reading as already inside DEC-010 would rewrite a frozen rule without an amendment.

The amendment is `DEC-010-A1` in `reports/P2_1_amendment_proposals.md`. Status: `PROPOSED / NOT APPLIED`.

Effect if it is later approved, and not before:

- The eight-gate structure remains intact.
- GO eligibility is unchanged: all eight keys must be `PASS`.
- `Q.H1` PASS means the preregistered diagnostic or transfer criterion passed. It does not mean the true Q mechanism was identified.
- `Q.H1` FAIL means that diagnostic or transfer criterion failed. The DEC-010 outcome is still `REDEFINE_SCALE` when the rest of rule 6 matches. That outcome does not assert that mechanism discovery is impossible in principle.
- The numerical PASS bar remains deferred by DEC-011. This phase does not set it.

Until `DEC-010-A1` is approved and applied, a run must not record `Q.H1` as `PASS`. A PASS written against the unamended table would be readable as the same kind of success as `P.H1` PASS.

## 5. Decision 3 — planted mechanism

Status: `RESOLVED`

The scientific object is the frozen operational mechanism in `P_SPEC.md`:

- `planted_A` = L0H0 = `[0, 0]`
- `planted_B` = L1H1 = `[1, 1]`
- episodic two-hop key/value retrieval

No mechanism replacement is authorized in P2.1.

`M_COPY`, `M_INHIBIT`, and `M_BIND` in specification §6.2 are provisional semantic labels and examples. They are not the planted construction. They do not rename L0H0 or L1H1. The sentence "At minimum create 3 mechanisms" in that section is non-authoritative against the frozen spec.

`P_SPEC.md` was not edited. A different planted mechanism requires a new DEC. DEC-011 `post_freeze_changes` already says a substrate or mechanism change is not an engineering fix.

Specification §6.2 must be corrected before preregistration freeze so a reader cannot implement the three examples instead of `P_SPEC.md`. That correction is specification conformance. It is not a DEC amendment.

## 6. Decision 4 — H3 transformation family

Status: `DECISION_REQUIRED`

DEC-011 is binding on the provenance point: M18.7 is `PROVENANCE_UNRESOLVED`, it is not an evidentiary basis for any gate, and the H3 definition must not depend on M18.7. Specification §11's sentence "This directly targets the failure observed in M18.7/A4" is non-authoritative. The historical A4 result is not ground truth and is not a reason to choose a map.

The scientific requirement, independent of M18.7, is:

> H3 must test whether the learned self-model is invariant under a transformation of representation that preserves the underlying mechanism and task semantics.

A transformation family has to satisfy all of the following before it can be frozen:

1. It is a representation-level transformation.
2. The mechanism is preserved.
3. Behavioral task semantics are preserved.
4. The transformation is known before evaluation.
5. It is not selected because of M18.7 or A4 results.
6. It is applied consistently to training, discovery, and holdout under preregistered rules.
7. Holdout outcomes are not available during transformation selection.

No frozen DEC authorizes a numerical map. None is frozen here.

### Candidate record

Parameters are not assigned. A candidate that can meet the seven tests on paper is not an authorization to use it.

| Id | Family | Against the seven tests | Eligible to freeze now |
| --- | --- | --- | --- |
| H3-C1 | The specification's phase rotation `[X, Y] -> [Y, -X]`, adopted because §11 ties it to M18.7/A4 | Fails test 5. The written reason is the prior result. The formula is also a parameter, and no DEC freezes it | No |
| H3-C2 | A preregistered invertible linear change of coordinates on one declared activation subspace, with the task-output path defined so the planted edges and the task labels are unchanged | Can meet tests 1–4 and 6–7 only after the matrix, the subspace, and the output-path rule are written before any holdout read. Can meet test 5 only if the written reason is the invariance requirement and not A4 | No. The matrix is not authorized |
| H3-C3 | A preregistered permutation of coordinates inside one declared representation vector, inverted or confined so the task readout is unchanged | Same pattern as H3-C2. The permutation is a representation-level map. It does not depend on A4 unless someone selects it for that reason | No. The permutation is not authorized |
| H3-C4 | A preregistered multiplication by -1 on one declared subspace, with the same output-path constraint | A minimal invertible representation map. Independent of A4 if the reason is the invariance requirement | No. The subspace is not authorized |

No family is recommended for adoption. H3-C1 is ineligible under test 5 as it is justified in the specification. H3-C2, H3-C3, and H3-C4 are eligible only as later candidates. Choosing among them, or choosing a matrix inside H3-C2, is a new scientific decision. DEC-011 already deferred the H3 components and their PASS combination. This phase does not close that deferral.

`DECISION_REQUIRED`: which family, with which declared subspace and which output-path rule, will be frozen in a later DEC or in the preregistration that DEC-011 names. Not in P2.1.

Proposed H3 numeric bars in specification §11 stay `PROPOSED — NOT FROZEN`.

## 7. Decision 5 — run accounting

Status: `RESOLVED`

DEC-011 is the binding budget authority. `MRSM_BUDGET.yaml` is the live ledger and already follows it.

Binding rules:

- 8 total full runs.
- Q GPU sub-limit of 2. That sub-limit is not an extra allowance and is not a cap on every Q run.
- A full run is any execution that reads a locked holdout.
- Construction clock: 5 × 24h from the commit that set `P_SPEC.md` to FROZEN. It is not an MRSM full run.
- MRSM clock: 28 × 24h from `p_freeze` only.
- `p_freeze` is the commit that records `p_construction.outcome: DONE`. That outcome is null. The MRSM clock has not started.

This phase does not start either clock. It does not reinterpret a full run as "a complete preregistered benchmark execution" where that sentence conflicts with the locked-holdout definition.

Neither clock was started. `MRSM_BUDGET.yaml` was not edited. `p_construction.outcome` remains null. `full_runs.used` remains 0. `q_gpu_full_runs.used` remains 0. The construction deadline already recorded in the ledger is `2026-10-03T14:17:59+00:00`. This phase does not declare `P_CONSTRUCTION_INCOMPLETE`.

### Specification sentences that conflict

| Location | Sentence | Why it is non-authoritative |
| --- | --- | --- |
| §0 | "4 calendar weeks" | DEC-011's MRSM duration is 28 × 24h from the `p_freeze` committer timestamp, and that clock has not started |
| §0 | "Qwen arm Q: maximum 2 full runs" | DEC-011 caps GPU Q full runs at 2, inside the 8. A CPU Q full run increments `full_runs` only |
| §0 | "GPU for Q is permitted only if explicitly recorded in preregistration" | DEC-011 requires an independent environment record and the Q GPU cap. The specification sentence is not that rule |
| §0 | Budget list has no 5-day construction clock and no `p_freeze` start | Those rules are in DEC-011 and are already on the ledger |
| §22 | "Interpret full run as a complete preregistered benchmark execution for one arm/seed/configuration." | Conflicts with "any execution that reads a locked holdout" |
| §22 | "Q maximum: 2 full runs" and "P uses the remaining permitted runs" | Treats 2 as a cap on all Q runs and as a deduction from the 8. DEC-011's 2 is the GPU sub-limit |
| §22 | Counters `budget/q_runs_used` and `budget/q_runs_remaining` | The frozen counter is `q_gpu_full_runs`, a sub-limit, not a separate Q pool |

§0's "maximum 8 full MRSM runs" matches the DEC-011 count. The conflict is the definition of a run and the meaning of the Q cap, not the integer 8.

These sentences must be corrected before preregistration freeze. No DEC-011 amendment is required.

## 8. Required amendments

| Id | Target | What it would do | Applied in P2.1 |
| --- | --- | --- | --- |
| `DEC-010-A1` | DEC-010 | Add the P.H1 / Q.H1 epistemic constraint. Leave the eight-key table and GO eligibility unchanged | No. `PROPOSED / NOT APPLIED` |
| Specification conformance | `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` §0, §1.2, §2, §6.2, §9, §11, §22, §24 | Make terminal wording, mechanism wording, H3 motivation, and budget wording match DEC-010, DEC-011, and `P_SPEC.md` | No. The file was not edited |

Specification conformance is required before freeze. It is not a DEC amendment, and it is not an authorization to edit the specification inside an implementation pass that also changes science.

## 9. Remaining unresolved items

- `DEC-010-A1` is not approved and not applied.
- The H3 family is `DECISION_REQUIRED`. No map, subspace, or output-path rule is frozen.
- H1–H4 numerical thresholds remain `PROPOSED — NOT FROZEN`. DEC-011 still defers them. No bar was chosen.
- The diagnostic criterion that would make `Q.H1` PASS or FAIL, as a metric, is not defined. The amendment forbids a mechanism-identity reading. It does not supply a number.
- Holdout membership for the MRSM run is not frozen. `p_construction.outcome` is null, so `p_freeze` has not occurred and the MRSM clock has not started.
- The specification file still contains the conflicting sentences. They are non-authoritative. They are still on disk.
- M22.1-R remains `REPLAY_BLOCKED` because the pinned offline snapshot is absent. That blocks a Q full run. It is not a terminal MRSM result.

## 10. P2.1 exit criteria

| Criterion | Result |
| --- | --- |
| Each of the five conflicts has one of the four statuses | Yes. See sections 3–7 |
| DEC-010, DEC-011, and `P_SPEC.md` bytes unchanged | Yes |
| No amendment applied | Yes |
| No H1–H4 threshold chosen or edited | Yes |
| No scientific code and no scientific run | Yes |
| MRSM clock not started | Yes |
| Preregistration not frozen | Yes |

P2.1 stops here. It does not open P2.2.
