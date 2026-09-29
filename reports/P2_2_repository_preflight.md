# P2.2 repository preflight

Inspection time: 2026-09-29, before the P2.2 edits.

| Item | Observed |
| --- | --- |
| Branch at inspection | `cursor/p2-1-decision-resolution-4e19` |
| HEAD at inspection | `05099658b4de8e0a59b0b0716fed130c1df2ec37` |
| `git status --porcelain` | empty |
| Phase branch | `cursor/p2-2-operational-definitions-4e19`, created from that HEAD |

## Files inspected

| Path | What it is |
| --- | --- |
| `control_plane/DEC-010_decision_table.md` | Frozen terminal table, version 1. Eight keys. GO requires all eight `PASS`. Not modified |
| `control_plane/DEC-011_mrsm_scope.md` | Frozen scope. Q `ground_truth: UNKNOWN`. Full run = locked-holdout read. H3 must not depend on M18.7. Thresholds deferred. Not modified |
| `control_plane/P_SPEC.md` | Frozen planted mechanism L0H0 and L1H1. Causal predicate marked `not_a_gate_threshold: true`. Not modified |
| `control_plane/MRSM_BUDGET.yaml` | `status: NOT_STARTED`. Construction outcome null. `full_runs.used` 0 of 8. `q_gpu_full_runs.used` 0 of 2. Not modified |
| `reports/MRSM_Scientific_Contract_v0.1.md` | Draft. Not a preregistration |
| `reports/P2_0_source_inventory.md`, `reports/P2_0_contract_conflicts.md` | Conflict inventory. Not a decision |
| `reports/P2_1_decision_resolution.md`, `reports/P2_1_amendment_proposals.md`, `reports/P2_1_freeze_readiness.md` | P2.1 resolution. Freeze status `NOT_READY`. `DEC-010-A1` proposed and not applied |
| `configs/m22_1_r_replay.yaml` | Replay tiers and engineering tolerances. Not an H1–H4 source. Not modified |
| `src/cognitive_self_model/control_plane/validate.py` | Ledger validator. Uses PyYAML |
| `control_plane/FILE_MAP.yaml` | Path inventory. P2.0 and P2.1 reports are mapped and not immutable |
| Tests | `tests/test_decision_table.py` covers DEC-010. `tests/test_control_plane.py` covers the ledger. `tests/test_m22_1_r_harness.py` covers replay. No `src/mrsm/` and no `configs/mrsm_prereg.yaml` |

The M22.1-R execution record remains `REPLAY_BLOCKED` because the pinned snapshot is absent. That is not a scientific failure. This phase does not change the replay result.

## P2.1 decisions, as written

1. DEC-010 is the binding terminal tree. Present in `reports/P2_1_decision_resolution.md` section 3, status `RESOLVED`.
2. P.H1 is the truth-validating discovery test. Present in section 4.
3. Q.H1 is diagnostic/transfer only, and `Q.H1 PASS` is not verified mechanism discovery. Present in section 4, status `RESOLVED_WITH_AMENDMENT_REQUIRED`.
4. The planted mechanism is frozen as L0H0 → L1H1. Present in section 5, status `RESOLVED`. No replacement was authorized.
5. Run accounting follows DEC-011. Present in section 7, status `RESOLVED`.

No conflict with those five statements was found in the frozen files. This phase does not repair DEC-010, DEC-011, `P_SPEC.md`, or the budget ledger.

## Classification

| Topic | Class |
| --- | --- |
| Terminal tree is DEC-010 | Already resolved |
| P.H1 is truth-validating | Already resolved |
| Q.H1 is not mechanism verification | Already resolved as a role. The observable vector was still unresolved |
| L0H0 → L1H1 planted mechanism | Already resolved |
| DEC-011 run accounting | Already resolved |
| `DEC-010-A1` text that binds the Q.H1 reading into the frozen table | Still unresolved as an application. This phase writes a final proposal and does not apply it |
| Q.H1 observable quantities | Proposed resolution in this phase, inside the draft contract and `configs/mrsm_operational_contract_v0.1.yaml` |
| H3 family names T1, T2, T3 | Proposed resolution in this phase, as names only. P2.1 left the choice `DECISION_REQUIRED`. DEC-011's deferral of the PASS combination is not closed |
| Concrete `A`, `b`, permutation, orthogonal matrix, subspace | Intentionally deferred |
| Numerical H1–H4 thresholds | Intentionally deferred |
| Preregistration freeze, `p_freeze`, locked holdout membership, Qwen execution | Intentionally deferred |
| Replay snapshot absence | Already recorded as `REPLAY_BLOCKED`. Not reclassified |

Historical documents still contain the conflicts P2.0 and P2.1 recorded. The implementation specification still states another terminal tree, the three example mechanism names, an M18.7 reason for phase rotation, and a different full-run sentence. Those files are not edited in this phase.
