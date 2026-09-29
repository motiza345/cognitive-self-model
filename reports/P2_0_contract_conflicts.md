# P2.0 contract conflicts

This matrix records conflicts. It does not select a threshold, a mechanism, or a terminal winner.

Classifications used below: frozen, proposed, deferred, illustrative, not a gate, unresolved.

## A. Gate thresholds

DEC-011 defers metrics, units, splits, nulls, baselines, and thresholds for P.H1–H4 and Q.H1–H4. `P_SPEC.md` marks its causal predicate `not_a_gate_threshold: true`. The specification's section heading is "Proposed frozen threshold". That heading does not make the number frozen. No later DEC adopts those numbers.

| Item | Value as written | Source | Classification |
| --- | --- | --- | --- |
| P.H1 mechanism F1 | >= 0.80 | Spec §9 | Proposed. Not frozen |
| P.H1 causal-edge precision | >= 0.80 | Spec §9 | Proposed. Not frozen |
| P.H1 false discovery rate | <= 0.20 | Spec §9 | Proposed. Not frozen |
| H2 MAE reduction versus strongest non-self baseline | at least 20% | Spec §10 | Proposed. Not frozen |
| H2 paired bootstrap | 95% CI for the MAE difference excludes zero | Spec §10 | Proposed procedure detail. The numerical bar is not frozen. Bootstrap count is not stated |
| H2 sign accuracy | >= 75% | Spec §10 | Proposed. Not frozen |
| H3-P identity agreement | >= 90% | Spec §11 | Proposed. Not frozen |
| H3-P causal-effect sign agreement | >= 90% | Spec §11 | Proposed. Not frozen |
| H3-P counterfactual MAE degradation | <= 10% | Spec §11 | Proposed. Not frozen |
| H3-Q "no material performance collapse" | no number, except the P-side 10% figure is not copied onto Q by DEC-011 | Spec §11 | Unresolved. Deferred with the H3 PASS combination |
| H4 utility improvement versus consumption ablation | >= 10% | Spec §12 | Proposed. Not frozen |
| H4 paired bootstrap | 95% CI excludes zero | Spec §12 | Proposed. Not frozen |
| P construction causal predicate | intact accuracy >= 0.95 and the hop and specificity gaps in `P_SPEC.md` | `P_SPEC.md` `causal_validation` | Frozen as a construction definition-of-done check. Not a gate |
| H1–H4 pass bars | none stated | DEC-011 | Deferred |
| H3 component set and how those components combine into PASS | none stated | DEC-011 `deferred_to_mrsm_preregistration` | Deferred |
| M22.1-R score/delta/mean/null tolerances | 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-6, 1.0e-6, label PROPOSED | `configs/m22_1_r_replay.yaml` | Replay engineering tolerances. Not H1–H4 gates. Not changed |

No threshold in this table was edited.

## B. GO / REDEFINE / STOP

DEC-011 does not modify DEC-010. The specification's sections 1.2 and 24 are a different tree. The specification also uses `INCOMPLETE_BUDGET` in sections 0 and 22 while section 1.2 says no fourth outcome may be invented. That is an internal conflict in the specification, separate from DEC-010.

The only executable table in the repository is DEC-010 (`decision_table.py`). Existence of that code does not delete the specification's tree. Both cannot be the sole terminal rule. Resolution is `DECISION_REQUIRED`.

```text
Condition | Required gates | Decision | Source | Conflict? | Resolution needed?
P.H1 is FAIL | P.H1 | STOP | DEC-010 rule 1; spec §24 "If H1-P fails" | No on this row | No for this row alone
P.H1 is PASS and P.H2 is FAIL | P.H1, P.H2 | STOP | DEC-010 rule 1 | Yes. Spec §24 does not state this STOP | DECISION_REQUIRED
P.H1 is NOT_EVALUATED and P.H2 is FAIL | P.H1, P.H2 | STOP | DEC-010 rule 1 | Yes. Spec §24 does not state this STOP | DECISION_REQUIRED
P.H1 and P.H2 PASS, P.H3 FAIL | P.H1, P.H2, P.H3 | REDEFINE_CAUSAL_AUDIT | DEC-010 rule 2 | Name conflict. Spec §24 calls the same scientific situation REDEFINE | DECISION_REQUIRED for the outcome name and for what is retired
P.H1–H3 PASS, P.H4 FAIL | P.H1–H4 | REDEFINE_CAUSAL_AUDIT | DEC-010 rule 3 | Yes. Spec §24 has no H4-fail leaf. GO requires P.H4, so H4 fail is not GO, and the spec does not name the other outcome | DECISION_REQUIRED
P.H1 and P.H2 PASS, P.H3 NOT_EVALUATED, P.H4 FAIL | P.H1–H4 | REDEFINE_CAUSAL_AUDIT | DEC-010 rule 4 | Yes. Absent from spec §24 | DECISION_REQUIRED
P.H1–H4 PASS and any Q.H1–H4 is NOT_EVALUATED | all eight keys | INCOMPLETE_BUDGET | DEC-010 rule 5 | Partial. Spec §0 and §22 use INCOMPLETE_BUDGET for an unassessed gate. Spec §1.2 says only three terminal outcomes. Spec §24 GO does not require Q.H1, so an unevaluated Q.H1 can still be GO in the spec and cannot be GO in DEC-010 | DECISION_REQUIRED
P.H1–H4 PASS, every Q gate evaluated, any Q gate FAIL | all eight keys, including Q.H1 | REDEFINE_SCALE | DEC-010 rule 6 | Name and scope conflict. Spec §24 says REDEFINE if P passes and Q fails, and it does not count Q.H1 as a fail that blocks GO | DECISION_REQUIRED
P.H1–H4 PASS and Q.H1–H4 PASS | all eight keys | GO | DEC-010 rule 7 | Yes. Spec §24 GO does not require Q.H1 PASS | DECISION_REQUIRED
Any of P.H1–H4 is NOT_EVALUATED, and no earlier rule matched | P.H1–H4 | INCOMPLETE_BUDGET | DEC-010 rule 8 | Partial, same fourth-outcome conflict as above | DECISION_REQUIRED
Any other record, or a missing or illegal gate status | eight keys, statuses PASS, FAIL, NOT_EVALUATED | CONTRACT_ERROR | DEC-010 else clause | Yes. Spec §1.2 does not name CONTRACT_ERROR | DECISION_REQUIRED
H1-P passes, H2-P passes, H3-P fails | P H1, P H2, P H3 | REDEFINE | Spec §24 | Same situation as DEC-010 rule 2 under another name | DECISION_REQUIRED
P passes and Q fails | Spec text does not say whether Q.H1 counts | REDEFINE | Spec §24 | See DEC-010 rule 6 | DECISION_REQUIRED
P H1–H4 and Q H2, Q H3, Q H4 pass; leakage audits pass; Q.H1 diagnostic | P H1–H4, Q H2–H4. Q.H1 not a GO requirement | GO | Spec §24 | Contradicts DEC-010 rule 7 | DECISION_REQUIRED
Budget expires with a gate unassessed | unspecified which gates | INCOMPLETE_BUDGET | Spec §0, §22 | Conflicts with spec §1.2 "exactly three" outcomes. Aligns in name with DEC-010 | DECISION_REQUIRED
```

What this phase does not do: it does not add a ninth outcome, it does not edit DEC-010, and it does not treat the specification tree as adopted.

## C. Q-arm H1

| Question | What the sources say | Classification |
| --- | --- | --- |
| Does Q have mechanism ground truth? | DEC-011 `ground_truth: UNKNOWN`. Spec §2: no trusted mechanism ground truth | Frozen: no |
| Can Q.H1 validate discovery against a planted label? | Spec §2, §9, and §24 say no, unless an independently verified external target exists. None is recorded | Diagnostic. Not a truth-validating discovery gate |
| Is Q.H1 still a governance key? | DEC-010 requires the key. Missing key is CONTRACT_ERROR. GO requires PASS. A FAIL among evaluated Q gates is REDEFINE_SCALE | Frozen key. The meaning of PASS is deferred |
| What may be reported? | Spec §9: candidate count, reproducibility across seeds, evidence completeness, internal causal consistency. "NOT called verified mechanism identity" | Proposed report contents. Not a frozen metric |

Q.H1's scientific role in this contract is diagnostic. DEC-010's requirement that the key exist and that GO include Q.H1 PASS is preserved as governance and is not rewritten. Those two statements are not the same claim. Defining a PASS rule that would make Q.H1 certify mechanism identity would contradict the specification and DEC-011. Defining a PASS rule at all is `DECISION_REQUIRED` because DEC-011 deferred it.

Qwen self-consistency, a discovery algorithm's own output, and a proxy label from the same pipeline are not ground truth. Spec §2 and rule 8.

## D. Mechanism definitions

| Object | Source | What it is | Classification |
| --- | --- | --- | --- |
| M_COPY, M_INHIBIT, M_BIND | Spec §6.2 | Three example causal roles. "The exact implementation can be adjusted during P2 design" | Illustrative |
| L0H0 (`planted_A` `[0, 0]`) and L1H1 (`planted_B` `[1, 1]`) on episodic two-hop retrieval | `P_SPEC.md`, status FROZEN | One planted two-hop construction. Training mask removed before audit. Identity hidden from discovery | Frozen construction |
| "At minimum 3 mechanisms" and a primary H1 holdout mechanism family absent from training | Spec §6.2 and §7 | A design minimum in the proposal | Proposed. Conflicts with the frozen two-head construction |
| Post-freeze change of substrate or mechanism | DEC-011 `post_freeze_changes` | Requires a new DEC | Frozen process rule |

These are not the same definition. The examples are not an implementation of L0H0/L1H1, and L0H0/L1H1 is not a renaming of M_COPY, M_INHIBIT, and M_BIND.

This phase does not replace either text. Using the three example names as the planted ground truth would change a frozen mechanism and needs a new DEC. Keeping P_SPEC means the specification's three-mechanism minimum and its "mechanism family absent from training" holdout are not yet satisfiable inside the frozen construction. That gap is `DECISION_REQUIRED`. It is not resolved by calling the two heads three mechanisms.

## E. H3 representation invariance

| Question | Determination | Source |
| --- | --- | --- |
| What property is H3 testing? | Whether the mechanism object (P) or the functional predictions (Q) stay the same when the representation coordinates change and the causal behavior is preserved | Spec §11 scientific question and the P/Q endpoints. The property sentence does not require M18.7 |
| Is phase rotation scientifically required? | The specification says "Required transformation" and ties it to "the failure observed in M18.7/A4". DEC-011 says M18.7 provenance is unresolved and H3 must not depend on M18.7. A requirement that exists because of A4 is not an allowed basis | Spec §11; DEC-011 `provenance.M18.7` |
| Is phase rotation a candidate? | Yes. The formula `[X, Y] -> [Y, -X]` is written in the specification. It is not frozen. Adopting it as mandatory, after removing the M18.7 justification, is `DECISION_REQUIRED` | Spec §11 |
| Can H3 be defined without M18.7? | Yes. The coordinate-independence question and the P/Q endpoints can be stated without the historical result. The PASS combination of H3 components remains deferred | Spec §11; DEC-011 deferral |
| Is the A4 result ground truth? | No | DEC-011: not an evidentiary basis for any gate |

The additional invertible transformation required by spec §11 is unnamed. It is proposed and not frozen.

## F. Budget

Values below are copied. None were edited. `p_construction.outcome` remains null. The MRSM clock has not started. The construction deadline recorded in the ledger is `2026-10-03T14:17:59+00:00`. This phase does not declare `P_CONSTRUCTION_INCOMPLETE`.

| Item | Specification §0 and §22 | DEC-011 | `MRSM_BUDGET.yaml` now | Conflict |
| --- | --- | --- | --- | --- |
| MRSM clock start | Not identified as `p_freeze` | `p_freeze` commit | `mrsm.clock_start_commit: null` | Spec is silent. Ledger follows DEC-011 |
| MRSM duration | 4 calendar weeks | `28 x 24h` from the `p_freeze` committer timestamp | Deadline null until that commit | Wording versus an exact duration. Not interchangeable until a DEC says so |
| Full-run cap | 8 | 8 | `limit: 8`, `used: 0` | Count agrees |
| Full-run definition | A complete preregistered benchmark execution for one arm, seed, and configuration | Any execution that reads a locked holdout | Governed by DEC-011 | Yes. A holdout read that is not a full benchmark counts only under DEC-011. A complete run that does not read a locked holdout counts only under the specification |
| Q cap | Maximum 2 full Qwen runs | `q_gpu_full_runs_limit: 2`, a sub-limit of the 8, not an extra allowance. A CPU Q full run increments `full_runs` only | `q_gpu_full_runs` limit 2, used 0 | Yes. The specification caps all Q runs at 2. DEC-011 caps GPU Q runs at 2 |
| GPU | Permitted for Q only if the preregistration records it. CPU is default | CPU default. GPU only with an independent environment record and inside the Q GPU cap | No GPU run recorded | Extra conditions, not the same sentence |
| P construction clock | Not described | 5 x 24h from the P_SPEC freeze commit. Not a full run. Outcome `DONE` or `P_CONSTRUCTION_INCOMPLETE`. Incomplete means MRSM does not start | Clock start `ad3cbc0f185a397d34eddabb8032fd3dd2144b1c`. Deadline `2026-10-03T14:17:59+00:00`. `outcome: null` | Spec does not describe this clock. Ledger is the DEC-011 clock, still open |
| M22.1-R counted | Not stated | false | false | No conflict in the frozen ledger |
| P construction counted in the 8 | Not stated | false | false | No conflict in the frozen ledger |
| Pre-freeze smokes | Not described | Holdout access during construction is forbidden | `P_CONSTRUCTION_ATTEMPTS.yaml` class `UNCOUNTED_P_CONSTRUCTION`. Counters stayed 0. The record says those smokes generated and hashed holdout tokens | Recorded exception to the holdout rule. Not reclassified here |
| Amendment cap | One per phase, engineering only | Not part of the DEC-011 budget block | Not a counter | Proposed in the specification. Not a frozen budget field |

Normalized reading, without dropping either source: the counters that exist today are the DEC-011 counters in `MRSM_BUDGET.yaml`. The specification's "2 full Qwen runs" and its full-run sentence are not the same rules. Replacing one definition with the other is `DECISION_REQUIRED`. This phase does not replace either.

## G. Replay terminology

| Name | Where it is active | Role |
| --- | --- | --- |
| `REPLAY_EXACT`, `REPLAY_NUMERIC_EQUIVALENT` | Replay yaml tiers. DEC-007 success tiers | Active. Only these two are success tiers for M22.1 status |
| `REPLAY_BEHAVIORAL_EQUIVALENT` | Replay yaml | Active. Not a DEC-007 success tier |
| `REPLAY_DIVERGENT`, `REPLAY_INVALID`, `REPLAY_BLOCKED` | Replay yaml. Emitted by the harness | Active |
| `REPLAY_FAIL` | Spec §19 only | Not a harness tier. P1 recorded that assigning it would change M22.1 decision weight and did not assign it |

`REPLAY_FAIL` has a proposed policy sentence: remove M22.1 evidence from decision weight, diagnose once, and do not stop MRSM. That sentence is not implemented. It is not an obsolete synonym that this phase can map onto `REPLAY_BLOCKED`, `REPLAY_INVALID`, `REPLAY_DIVERGENT`, or `REPLAY_BEHAVIORAL_EQUIVALENT`. Whether any current tier should carry that policy is unresolved.

The harness attempt on this tree is `REPLAY_BLOCKED` because the offline snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987` is absent. Spec §19 says proceed to P2 after `REPLAY_EXACT` or `REPLAY_NUMERIC_EQUIVALENT`. That proceed condition is not met. P2.0 exists because it was separately authorized as contract reconciliation. It is not a replay success.

The replay yaml and the harness were not modified.
