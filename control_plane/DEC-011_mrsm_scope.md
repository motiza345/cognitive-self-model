status: FROZEN
decision_version: 1
accepted_on: 2026-09-28
decision_id: DEC-011
title: MRSM scope, arms, P construction, budget semantics
depends_on:
  DEC-010: {status: FROZEN, decision_table_version: 1, modified_by_this_decision: false}
scope_rule: >
  Defines experiment boundaries only. Contains no metric, unit, split, null,
  baseline, or threshold for any gate P.H1-H4 or Q.H1-H4.

evidence:
  audit_head: 122131a681ff28a8fa483f6e0764b99f2b35b2c3
  a0_result: "No planting procedure in any local ref (C2 ABSENT). M21 environment is a linear DGP."
  history_checks:
    - {cmd: 'git log --all --oneline -S "planted_small"', result: EMPTY}
    - {cmd: 'git log --all --oneline -S "ground_truth" -- control_plane', result: EMPTY}
    - {cmd: 'git grep -niE "M18\.7|phase.?rotation" <all local and remote-tracking refs>', result: EMPTY}
  finding: "DEC-010 arm definitions and budget semantics were not represented in the repository; DEC-011 is their first repository record."

arms:
  P:
    id: planted_small_transformer
    ground_truth: KNOWN
    substrate: small transformer, TransformerLens HookedTransformer
    device: CPU
    construction: HYBRID   # algorithmic task training + predefined structural mechanism
    ground_truth_definition: "mechanism definition + independent causal validation, both frozen before any discovery run"
    forbidden:
      - discovery outputs used to define, select, or validate P ground truth
      - M21 benchmark environment used as P   # linear DGP, not a network
  Q:
    id: Qwen/Qwen2.5-0.5B
    revision: 060db6499f32faf8b98477b0a26969ef7d8b9987
    revision_source: configs/m22_1_r_replay.yaml
    ground_truth: UNKNOWN
    device: CPU default; GPU only with an independent environment record and within q_gpu_full_runs limit

shared_pipeline_rule: >
  Discovery, validation, and self-use code paths are identical across arms.
  Arm-specific code is limited to a model adapter exposing the same
  intervention and observation interface. Discovery code must not branch on arm identity.

p_construction:
  spec_document: control_plane/P_SPEC.md   # task, architecture, mechanism, DoD criteria, holdout seed range
  clock_start: commit that freezes P_SPEC.md
  timebox_calendar_days: 5
  clock_restart: FORBIDDEN
  counted_in_mrsm_full_runs: false
  holdout_access: FORBIDDEN   # construction uses seeds disjoint from the P_SPEC holdout range
  definition_of_done:
    - model artifact produced and sha256 recorded
    - mechanism definition recorded
    - ground truth recorded, derived without any discovery output
    - intervention interface callable through shared adapter
    - observation interface callable through shared adapter
    - deterministic under recorded seeds
    - independent causal validation passes P_SPEC criteria
    - trivial static-weight detector fails to identify the planted mechanism per P_SPEC criteria
    - train/dev/holdout protocol recorded
  outcome_if_dod_not_met_at_timebox: P_CONSTRUCTION_INCOMPLETE
  on_p_construction_incomplete: "MRSM does not start. Any continuation requires a new bounded DEC. No clock reset."
  post_freeze_changes: "Changes to substrate or mechanism are not engineering fixes and require a new DEC."

mrsm_budget:
  clock_start: P freeze (DoD met)   # preregistration drafting counts inside this window
  calendar_days: 28
  full_runs_limit: 8
  q_gpu_full_runs_limit: 2
  full_run_definition: "any execution that reads a locked holdout"
  m22_1_r_counted: false
  p_construction_counted: false
  state_file: control_plane/MRSM_BUDGET.yaml

provenance:
  M18.7:
    status: PROVENANCE_UNRESOLVED
    use: "Not an evidentiary basis for any gate. H3 is defined independently. May be added to provenance only if a recoverable source is found."

deferred_to_mrsm_preregistration:
  - metrics, units, splits, nulls, baselines, thresholds for P.H1-H4 and Q.H1-H4
  - H3 components (context transfer, identity invariance) and their PASS combination
  - H4 audit path: Self-Model output -> decision -> intervention -> outcome, against same-pipeline ablated and marginal-preserving shuffled controls
  non_binding_candidates: {h1: 0.8, h2_rho: 0.7, h2_null: P95, h3_ratio: 0.8}
