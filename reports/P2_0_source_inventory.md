# P2.0 source inventory

Date of this inventory: 2026-09-29.

Working tree at the start of this phase: `cursor/m22-1-r-harness-4e19` at `7f55a49a54a17bb710ae5372ad9b1ec14922f6ea`. This phase branches from that commit. It does not merge `main` and it does not merge the audit branch.

A document is listed as a scientific decision only when its own status says it decides scope, a terminal outcome, or a frozen construction. An implementation proposal can contain scientific sentences and still not be a frozen decision.

The P0, P1, and reuse-map reports are not files in this working tree. They were read from commit `430656020c781d2e10441a0ba0b8661e439b15b4` on `cursor/mrsm-p0-p1-audit-4e19`.

## Documents

| Path | Commit or recorded version | Status in the source | Role |
| --- | --- | --- | --- |
| `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` | Version 1.0, dated 2026-09-28. On `main` as `5dd0e076c60e80eb64eabdae0ddb3429d2b46388`. Copied onto this tree as `d96b3eeed1a3a4a8f94b1ac372e35b9a8489cd60` | Not a DEC. No `status: FROZEN` field | Implementation proposal. Contains proposed gates, example mechanisms, a decision tree, and a budget wording |
| `control_plane/DEC-010_decision_table.md` | `status: FROZEN`, `decision_table_version: 1` | Frozen scientific decision | Terminal outcome table. Also implemented by `src/cognitive_self_model/control_plane/decision_table.py` |
| `control_plane/DEC-011_mrsm_scope.md` | `status: FROZEN`, `decision_version: 1`, `accepted_on: 2026-09-28`, `decision_id: DEC-011` | Frozen scientific decision | Arms, P construction, budget semantics, M18.7 prohibition. States that it contains no H1–H4 metric or threshold. `modified_by_this_decision: false` for DEC-010 |
| `control_plane/P_SPEC.md` | `status: FROZEN`. Clock-start commit recorded in the budget file as `ad3cbc0f185a397d34eddabb8032fd3dd2144b1c` | Frozen construction specification | Two-hop planted task, architecture, L0H0/L1H1 mechanism, holdout reservation, causal predicate marked `not_a_gate_threshold: true` |
| `control_plane/MRSM_BUDGET.yaml` | `status: NOT_STARTED`, `governed_by: DEC-011` | Ledger, not a new decision | Construction clock started. MRSM clock not started. Counters at zero |
| `control_plane/P_CONSTRUCTION_ATTEMPTS.yaml` | `governed_by: DEC-011` | Record of pre-freeze smokes | Class `UNCOUNTED_P_CONSTRUCTION`. Not a gate result |
| `configs/m22_1_r_replay.yaml` | File sha256 `d9e1cf5dc3b88535cda0516b9b24ae3510082b7fed452f2e8adb86c574e8039a`. Header is a frozen M22.1-R preregistration | Frozen for replay terminology and tolerance keys. Tolerance `label: PROPOSED` | Replay tiers and engineering tolerances. Not an MRSM gate document |
| `control_plane/CLAIMS.yaml` | Ledger. Entries are M21.2.4.3.3 through M22.1.3 observations | Findings and observations. No MRSM claim | Must not be reread as MRSM evidence |
| `control_plane/STATE.yaml` | `as_of` commit `0e6805e5153b162d4cabf7bae429455a7921e0a9` | Ledger | Decisions listed through DEC-009. DEC-010 and DEC-011 are not entries in this list. M22.1-R status `PREREGISTERED`. M22.2 `NOT_AUTHORIZED` |
| `control_plane/RECOVERY.yaml` | Top-level execution record from the harness phase | Recovery ledger | M22.1-R harness attempt is `REPLAY_BLOCKED`. Milestone `UNKNOWN` strings were not rewritten |
| `reports/MRSM_P0_repository_audit.md` | `430656020c781d2e10441a0ba0b8661e439b15b4` on the audit branch. Absent from this working tree | Audit. Not a decision | Records the same conflicts and does not choose a winner |
| `reports/MRSM_P1_reproducibility_audit.md` | Same audit commit. Absent from this working tree | Audit. Not a decision | Did not assign `REPLAY_FAIL`. Noted that the name is not a yaml tier |
| `reports/MRSM_REUSE_MAP.md` | Same audit commit. Absent from this working tree | Audit. Not a decision | Reuse verdicts. Does not choose the mechanism, the decision table, or thresholds |
| `reports/M22_1_R_harness_completion.md` | Present. Execution-record commit `74efd6bbe960834f4936878c03ecf821c6e69c4b`. Report tip parent `7d0775fe0c0822aba8a91c8a44c9df8c3bda8b66` | Engineering record | Replay tier `REPLAY_BLOCKED`. No scientific claim |

`STATE.yaml` was not updated to register DEC-010 or DEC-011. That gap is recorded here. This phase does not edit `STATE.yaml`.

## Rule index

Each row is a rule this phase had to place. Status words are the classification used in `reports/P2_0_contract_conflicts.md`.

| Rule | Source | Classification |
| --- | --- | --- |
| S1: the Self-Model is consumed by the decision | Spec §1.1 | Proposed. Not in DEC-010 or DEC-011 |
| Exactly three terminal outcomes: GO, REDEFINE, STOP | Spec §1.2 | Proposed. Conflicts with DEC-010 |
| Terminal table with STOP, REDEFINE_CAUSAL_AUDIT, REDEFINE_SCALE, INCOMPLETE_BUDGET, GO, CONTRACT_ERROR | DEC-010 | Frozen |
| DEC-011 does not modify DEC-010 | DEC-011 `depends_on` | Frozen |
| P ground truth known; Q ground truth unknown | DEC-011 `arms`; spec §2 | Frozen for the arm distinction. Spec adds the Q.H1 diagnostic sentence |
| Q.H1 is diagnostic and is not verified mechanism identity | Spec §2, §9, §24, rule 8 | Proposed scientific role |
| Q.H1 is a required gate key; GO requires Q.H1 PASS | DEC-010 | Frozen governance key. Pass criterion deferred by DEC-011 |
| No H1–H4 metric or threshold in DEC-011 | DEC-011 `scope_rule` and `deferred_to_mrsm_preregistration` | Frozen deferral |
| H1 F1 0.80, edge precision 0.80, FDR 0.20 | Spec §9 "Proposed frozen threshold" | Proposed. Not frozen |
| H2 20% MAE reduction, CI excludes 0, sign accuracy 75% | Spec §10 | Proposed. Not frozen |
| H3 identity agreement 90%, sign agreement 90%, MAE degradation 10% | Spec §11 | Proposed. Not frozen |
| H4 utility improvement 10% and CI excludes 0 | Spec §12 | Proposed. Not frozen |
| P causal predicate (0.95 intact accuracy and the hop gaps) | `P_SPEC.md` `causal_validation` | Frozen as a construction check. `not_a_gate_threshold: true` |
| Example mechanisms M_COPY, M_INHIBIT, M_BIND | Spec §6.2, labeled "Example" | Illustrative proposal |
| Planted mechanism L0H0 and L1H1 on the two-hop task | `P_SPEC.md` `mechanism` | Frozen. A later change requires a new DEC |
| Phase rotation `[X, Y] -> [Y, -X]`, motivated by M18.7/A4 | Spec §11 | Proposed candidate. Its requirement cites unresolved provenance |
| M18.7 is not an evidentiary basis; H3 must not depend on M18.7 | DEC-011 `provenance.M18.7` | Frozen prohibition |
| 4 calendar weeks, 8 full runs, 2 full Q runs; full run = one complete benchmark execution | Spec §0 and §22 | Proposed wording |
| 28×24h from `p_freeze`; 8 full runs; full run = any locked-holdout read; `q_gpu_full_runs` 2 as a sub-limit; 5-day construction clock | DEC-011 and `MRSM_BUDGET.yaml` | Frozen budget semantics. Construction outcome still null |
| Replay tiers EXACT, NUMERIC_EQUIVALENT, BEHAVIORAL_EQUIVALENT, DIVERGENT, INVALID, BLOCKED | `configs/m22_1_r_replay.yaml` | Frozen vocabulary for M22.1-R |
| DEC-007 success tiers are only EXACT and NUMERIC_EQUIVALENT | Replay yaml `dec007_success_tiers`; STATE DEC-007 | Frozen for M22.1 status. Replay tolerances remain `label: PROPOSED` |
| `REPLAY_FAIL` removes M22.1 from decision weight and does not stop MRSM | Spec §19 | Proposed label. Not a harness tier |
| Current replay attempt | `artifacts/m22_1_r/execution_record.json` | `REPLAY_BLOCKED`. Snapshot absent. Not a scientific failure |
| M21 `q_invalid` is not a universal invalidity probability and is not M22.1 evidence | `CLAIMS.yaml` F-M21.2.4.3.3-QINVALID; `reports/M22_1/README.md` | Frozen limitation on those claims. Not an MRSM metric |
| M21 linear environment is not arm P | DEC-011 `arms.P.forbidden` | Frozen |
| M22.2 is not authorized | `STATE.yaml` gates | Frozen ledger fact. Spec §0 says MRSM replaces M22.2 as the next scientific gate; that replacement is in the proposal, not in a DEC |
