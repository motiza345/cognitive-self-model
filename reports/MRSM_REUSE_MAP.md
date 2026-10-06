# MRSM reuse map

Date: 2026-09-28

Specification: `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` at `5dd0e076c60e80eb64eabdae0ddb3429d2b46388`.

Code inventory: `cursor/m22-1-r-behavioral-replay-265a` at `344899e85cde6f8f8a392492c22e5e0b74cf21bb`.

This map says what an implementation may call and what does not exist yet. It does not choose the planted mechanism, the decision table, or the thresholds. Those conflicts are listed in `reports/MRSM_P0_repository_audit.md` section 7.

Verdict labels:

- **Reuse** — frozen behavior can be called without changing it.
- **Adapt** — the module is the closest existing implementation and does not already provide the spec contract.
- **Do not use as MRSM** — using it as the self-model, as arm P, or as verified mechanism identity would contradict the specification or DEC-011.
- **New** — no equivalent module exists.

## 1. Specification section 17 modules

| Spec path | Closest existing code | Verdict |
| --- | --- | --- |
| `src/mrsm/schemas.py` | `m22_1/prompts.py` `PromptRecord`; intervention code is functions, not an `Intervention` or `Prediction` record | **New.** No schema has `mechanism_id`, `scope_id`, `evidence_refs`, `source_component`, `target_component`, `magnitude`, `duration`, `control`, and `hypothesis_id` together |
| `src/mrsm/planted_model.py` | `mrsm_p_v1.py` and frozen `control_plane/P_SPEC.md` | **Adapt, do not copy forward as a second model.** v1 is a root script. Checkpoint bytes are absent. `P_SPEC.md` is the frozen construction document on the scientific branch. Specification section 6.2 examples (M_COPY, M_INHIBIT, M_BIND) are not implemented |
| `src/mrsm/qwen_adapter.py` | `m22_1/loader.py` `load_qwen` and `forward_record` | **Adapt.** Same model id and recorded revision. No shared arm-agnostic adapter (`P_SPEC.md` records `shared_adapter_in_v1_script: ABSENT`). No offline pin. No weight hash. Not safe to treat as verified Q mechanism identity |
| `src/mrsm/interventions.py` | `m22_1/intervention.py` (additive residual); head-zero hooks inside `mrsm_p_v1.py` | **Adapt patterns only.** Spec intervention types are ablation, activation replacement, controlled noise, and patch, each with a control. The M22.1 hook is one additive residual family. The spec `Intervention` record does not exist |
| `src/mrsm/discovery.py` | `m22_1/protocol.py` layer selection; `mrsm_p_v1.py` `discover` | **New for MRSM H1.** M22.1 discovery selects a layer for one seeded direction. v1 discovery is a head search on the planted model, and `P_SPEC.md` forbids running discovery from construction. Neither emits the spec recovery rule (identity, causal direction, effect sign, evidence trace) |
| `src/mrsm/evidence.py` | `control_plane/CLAIMS.yaml`; `benchmark/evidence.py` | **New.** The ledger stores milestone findings. The benchmark encoder stores a 16-D vector. Specification section 5 asks for evidence references on each prediction |
| `src/mrsm/self_model.py` | none | **New.** No `SelfModelCore` with `fit`, `discover`, `predict`, `estimate_uncertainty`, `applicable`, `select_action`, `update`, `explain` |
| `src/mrsm/baselines.py` | `m22_1_2/models.py` mean and low-rank predictors; `benchmark/estimator.py` | **New for B0–B5.** M22.1.2 predictors score frozen response matrices. The benchmark estimator is an invalidity classifier. Specification section 3 says a readout-only classifier is not the self-model. B0–B5 as defined in section 13 are not implemented |
| `src/mrsm/policy.py` | `control_plane/decision_table.py` | **New.** DEC-010 chooses a program outcome from gate labels. It does not select an intervention from candidate predictions. H4 consumption-ablation is not implemented |
| `src/mrsm/metrics.py` | `m22_1/metrics.py`, `m22_1_2/metrics.py` | **Adapt the paired-summary functions. New endpoint definitions.** No mechanism F1, causal-edge precision, false-discovery rate, identity agreement, or consumption-utility metric |
| `src/mrsm/statistics.py` | `bootstrap_mean_interval` in `m22_1/metrics.py` and `m22_1_2/metrics.py`; episode bootstrap in `legacy_import/m212433_calibration_audit.py` | **Adapt the paired bootstrap pattern. New MRSM wrapper.** Existing functions are bound to their milestone configs. Do not retune those configs |
| `src/mrsm/leakage.py` | `m22_1/leakage.py`; `tests/test_benchmark_leakage_controls.py` | **Adapt the test style. New contract.** M22.1 leakage checks imports. M21 leakage checks the 16-D encoder. Specification section 14 forbids planted labels, ground-truth edges, post-intervention outcomes, future evidence, holdout labels, transformation identity, and hidden benchmark metadata at inference time |
| `src/mrsm/runner.py` | `m22_1/preflight.py`; `mrsm_p_v1.py` `run` | **New.** Neither enforces the section 16 run record or the section 22 budget counters. `MRSM_BUDGET.yaml` stores counters and is not consulted by a runner |
| `src/mrsm/audit.py` | `control_plane/validate.py` | **Reuse the ledger validator for the control plane. New for MRSM run audit.** The validator does not check leakage, holdout membership, or H1–H4 |

## 2. Specification scripts, configs, artifacts, tests

| Spec path | Closest existing code | Verdict |
| --- | --- | --- |
| `configs/mrsm_prereg.yaml` | `configs/m22_1_r_replay.yaml`; `control_plane/P_SPEC.md`; `control_plane/DEC-011_mrsm_scope.md` | **New file, not written in this audit.** The replay yaml is the M22.1-R contract only. DEC-011 defers H1–H4 numbers. Do not copy proposed thresholds out of the specification until P2 is authorized |
| `configs/planted.yaml` | `P_SPEC.md` architecture block; `mrsm_p_v1.py` `Config` | **New file.** Numbers for a small CPU transformer already appear in `P_SPEC.md` (2 layers, 4 heads, `d_model` 64, `d_mlp` 128). The mechanism identity conflict in P0 section 7 is unresolved |
| `configs/qwen.yaml` | `configs/m22_1_preflight.json`; replay yaml `recorded_revision` | **New file.** Model id and revision string already exist. Offline pin and weight hash do not |
| `artifacts/planted_ground_truth.json` | mechanism text in `P_SPEC.md`; `reports/mrsm_p_v1/checkpoint_mask_audit.json` | **New.** Required fields in specification section 6.3 (model hash, seed, mechanism ids, causal edges, intervention targets, effect ranges, representation transformations, holdout assignments) are not in one immutable artifact |
| `scripts/build_planted.py` | `mrsm_p_v1.py` | **Adapt v1 only after P2 authorization.** v1 is not a package entry and its checkpoint is not in git |
| `scripts/prereg_check.py` | `control_plane/validate.py` | **New** for the MRSM preregistration contract |
| `scripts/run_p.py`, `scripts/run_q.py` | `scripts/run_m22_1_preflight.py` | **New.** The preflight script is M22.1, and the replay yaml forbids using it as the replay entrypoint |
| `scripts/analyze.py`, `scripts/final_decision.py` | `decision_table.py` `terminal_decision` | **Adapt DEC-010 only if a later decision selects it.** Specification section 24 is not the same function. See P0 section 7. Do not edit the frozen table in an engineering pass |
| `tests/test_schemas.py` and the other section 17 tests | M22 and M21 tests under `tests/` | **New files.** Existing tests can stay as regression checks for the modules they already lock |
| `tests/test_determinism.py` | `test_benchmark_pipeline_reproducibility.py`, `test_m22_1_contract.py`, `test_m22_1_1_panel.py`, `test_m22_1_intervention.py` | **Reuse those tests for their own modules.** An MRSM determinism test does not exist |

## 3. Infrastructure the spec tells P0 and P1 to find

| Spec item (section 18–19) | Where it is | Reuse |
| --- | --- | --- |
| M22.1-R harness | Preregistration and manifest only. Runner absent | **Do not invent a scientific rerun inside P0.** The next engineering piece the specification names is the harness (section 29, step 2). Comparator tolerances stay at the yaml values. Key names stay as DEC-009 wrote them |
| Evidence graph | No graph code. `CLAIMS.yaml` is a ledger with `supersedes` links | **Reuse as provenance.** It is not the self-model evidence object |
| Qwen adapter | `m22_1/loader.py` | **Adapt**, with offline revision pin still unimplemented |
| Benchmark utilities | `src/cognitive_self_model/benchmark/` | **Reuse for M21 regression tests.** DEC-011 forbids this environment as arm P |
| Statistical utilities | bootstrap helpers named in section 1 | **Adapt.** Do not change their milestone thresholds |
| Control plane | `control_plane/` plus `validate.py` | **Reuse.** `STATE.yaml` is stale relative to current `main` (`5dd0e07`) and still names M22.1.4 as next. The specification says MRSM replaces M22.2 and that no new milestone is created without a DEC. This audit does not edit `next_milestone` |
| Run manifest fields (section 16) | partial git fields in `runtime_info.py` and `src/common/reproducibility.py` | **Adapt `runtime_info.py`.** Python, PyTorch, CUDA, model hash, dataset hash, holdout hash, config hash, baseline version, preregistration version, runtime, and hardware are not collected in one record |
| Budget counters | `control_plane/MRSM_BUDGET.yaml` | **Reuse the file as the ledger.** No runner increments it. Semantics differ from specification section 22; see P0 section 7 |

## 4. Explicit non-reuse

| Existing artifact | Why it is not the MRSM self-model |
| --- | --- |
| `benchmark/estimator.py` and `benchmark/evidence.py` | Invalidity score from a 16-D vector. Specification section 3 excludes a readout-only classifier and a hidden predictor with no inspectable mechanism object |
| `benchmark/environment.py` | Linear data-generating process. DEC-011 forbids it as arm P |
| `m22_1` preflight certificate | One additive residual preflight. `reports/M22_1/README.md` says no self-model was trained. Status in the ledger is `CANDIDATE` |
| `m22_1_2` response predictors | Predict frozen deltas. They are not consumed by a decision policy |
| `m22_1_3` readout null | Geometry check of a stored margin. Not mechanism recovery |
| `notebooks/cognitive_self_model_latest.ipynb` | M20.6.3.2 provenance. Recovery grade `POINTER_ONLY`. Not an adapter |
| `cursor/initial-self-model-gate-9523` | Unmerged numpy gate. Not in the control plane and not on the scientific branch |
| `scripts/mrsm_p_v0.py` | Superseded as a construction attempt by the v1 record. Weights and holdout manifest are not in git |
| M22.2 | Not authorized. The specification replaces it with MRSM. No implementation to reuse |

## 5. Dependency order supported by the specification

The specification's own sequence (section 29) after this audit is harness completion, then planted ground truth, schemas, interventions, baselines, `SelfModelCore`, leakage tests, and only then P2 preregistration. P3 is forbidden until P2 is authorized.

What blocks step 2 today, from `reports/MRSM_P1_reproducibility_audit.md`:

- no replay runner and no comparator
- `loader.py` is not an offline pin
- Qwen snapshot is not on this machine and weight SHA256 is not recorded
- M22.1-R environment fields are `UNKNOWN`
- M22.1 recovery is `PARTIAL` with `git_dirty: true` and `ADDITIONS_ONLY_UNVERIFIED`

The specification and the frozen scientific tree are on different branches. Implementing the section 17 package on current `main` would not see the frozen modules. Implementing it only on `344899e` would not see the specification file. This audit does not merge them.
