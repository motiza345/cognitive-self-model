# MRSM P0 — Repository Audit

Date: 2026-09-28

Specification read: `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` (1137 lines, commit `5dd0e076c60e80eb64eabdae0ddb3429d2b46388`).

The requested path `docs/MRSM_Implementation_Spec_v1.0.md` is not in the repository. The file above is the only MRSM v1.0 specification on `origin/main`. This audit treats that file as the current MRSM specification.

This audit does not implement MRSM, does not write `docs/MRSM_CONTRACT.md` or `configs/mrsm_prereg.yaml`, does not change thresholds, and does not start P2, P3, or any scientific run.

## 1. How the repository was inspected

`origin/main` at the start of this audit contained only the specification plus an empty package skeleton. The scientific history lives on other remote branches and is not merged into `main`.

| Ref | Tip | Tracked files | Role |
| --- | --- | --- | --- |
| `main` | `5dd0e076c60e80eb64eabdae0ddb3429d2b46388` | 12 | Specification plus empty `src/`, `tests/`, `configs/`, `docs/`, `scripts/`, `artifacts/`, `legacy/`, `notebooks/` |
| `cursor/m22-1-r-behavioral-replay-265a` | `344899e85cde6f8f8a392492c22e5e0b74cf21bb` | 165 | Latest scientific tree: M21 benchmark, M22.1 through M22.1.3b, control plane, M22.1-R preregistration, P construction records |
| `control-plane-bootstrap` | `0f8b8e2` | 153 | Ancestor of the replay branch, through M22.1.3b |
| `m22-1-3-readout-null` | `0e6805e5153b162d4cabf7bae429455a7921e0a9` | 137 | M22.1.3 analysis head. Ancestor of the replay branch |
| `cursor/m22-1-2-response-space-audit-c559` | `63699b7` | 123 | M22.1.2. Ancestor of the replay branch |
| `cursor/m22-1-1-intervention-space-audit-c559` | `72c6875` | 109 | M22.1.1. Ancestor of the replay branch |
| `cursor/m21-2-4-3-1-integration-c559` | `db8d03fad94e2f20f1a9f593a7f6b9eaa5f81d47` | 97 | M22.1 preflight code. Named by the replay preregistration as code-under-test |
| `chore/repository-scientific-structure` | `56c77fa` | 42 | M21 legacy-import skeleton. Ancestor of the replay branch |
| `cursor/setup-dev-environment-c559` | `cc2ecd1` | 14 | Environment install only. Not an ancestor of the replay branch |
| `cursor/initial-self-model-gate-9523` | `41b6147` | 15 | Unmerged IG-0 numpy gate. Not an ancestor of the replay branch |

`main` is one commit ahead of the replay branch (`5dd0e07`, the specification upload). The replay branch is 49 commits ahead of `main`. Merge-base of the two is `c08f0e64a5dd543002b30dfabfa5183d1b4e39d0`.

The code inspection below is of `344899e` unless a path is explicitly attributed to `main`.

## 2. What `main` contains

```text
.gitignore
README.md
artifacts/.gitkeep
configs/.gitkeep
docs/.gitkeep
docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md
legacy/.gitkeep
notebooks/.gitkeep
notebooks/cognitive_self_model_latest.ipynb
scripts/.gitkeep
src/cognitive_self_model/__init__.py
tests/.gitkeep
```

`src/cognitive_self_model/__init__.py` is a package marker. `legacy/` is empty. There is no `mrsm/` tree, no control plane, no tests, and no M22 or M21 Python on `main`.

`main`'s `.gitignore` ignores `artifacts/*` except `artifacts/.gitkeep`, and ignores `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, and `*.bin`.

## 3. Existing code that can be reused

Reuse verdicts are in `reports/MRSM_REUSE_MAP.md`. This section names the modules and what they actually do.

### 3.1 Control plane

Present only on the replay branch.

| Path | What it is |
| --- | --- |
| `control_plane/STATE.yaml` | Milestone status ledger. `as_of_commit` is `0e6805e` (2026-09-26). `next_milestone` is M22.1.4, status `NOT_STARTED` |
| `control_plane/CLAIMS.yaml` | Evidence-backed claim ledger from M21.2.4.3.3 onward. Entries point at artifact paths and sha256 values. Not a graph library |
| `control_plane/FILE_MAP.yaml` | Path inventory with roles and, for immutable artifacts, sha256. `as_of_commit` is `0e6805e`. 147 mapped paths; 165 tracked files |
| `control_plane/RECOVERY.yaml` | Per-milestone recovery grades, environment records, and an empty `execution_records` list |
| `control_plane/DEC-010_decision_table.md` | Frozen terminal decision table, version 1 |
| `control_plane/DEC-011_mrsm_scope.md` | Frozen arm, budget, and P-construction scope. States that it contains no H1–H4 metric or threshold |
| `control_plane/P_SPEC.md` | Frozen planted-arm construction spec (two-hop retrieval, TransformerLens, CPU) |
| `control_plane/MRSM_BUDGET.yaml` | Counters. `mrsm` clock not started. `full_runs.used` 0 of 8. `q_gpu_full_runs.used` 0 of 2. `p_construction.outcome` is null |
| `control_plane/P_CONSTRUCTION_ATTEMPTS.yaml` | Record of two pre-freeze smoke attempts |
| `src/cognitive_self_model/control_plane/validate.py` | Ledger validator |
| `src/cognitive_self_model/control_plane/decision_table.py` | Executable DEC-010 table |

### 3.2 M22.1 intervention and Qwen load path

`src/cognitive_self_model/m22_1/`

- `loader.py`: `load_qwen`, `resolve_hf_revision`, `forward_record`. Live `huggingface_hub.model_info` call. No `local_files_only`. On revision-load failure it retries `from_pretrained` without a revision.
- `intervention.py`: additive last-token residual hook, `h <- h + alpha * direction`. Alpha 0 returns the same tensor.
- `direction.py`: seeded unit Gaussian and an orthogonal control. Outcome-independent.
- `protocol.py`: bounded layer discovery, then frozen validation and replication. Synthetic-instrument tests exist.
- `metrics.py`: paired delta summaries, bootstrap mean interval, sign classification, Wilcoxon. Does not select an intervention.
- `outcome.py`: frozen logit-margin.
- `prompts.py`: frozen prompt list and split hashes.
- `leakage.py`: import audit so the preflight package does not import the epistemic benchmark.
- `preflight.py` / `run.py`: one-command Qwen preflight writer.
- `runtime_info.py`: git commit, branch, dirty flag, credential-stripped origin. Does not record Python, PyTorch, CUDA, or hardware.
- `config.py`, `status.py`, `artifacts.py`: frozen config load and certificate writing.

Entrypoint: `scripts/run_m22_1_preflight.py`. Config: `configs/m22_1_preflight.json`. Recorded model id: `Qwen/Qwen2.5-0.5B`. Recorded revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.

This is a residual-stream preflight. It is not `SelfModelCore`.

### 3.3 M22.1.1, M22.1.2, M22.1.3, M22.1.3b

| Package | Behavior |
| --- | --- |
| `m22_1_1` | Deterministic residual-direction panel, then a descriptive classification of frozen measurements |
| `m22_1_2` | Held-out response-matrix predictors (global mean, prompt mean, direction mean, additive, imputed SVD, ridge ALS) on frozen M22.1.1 deltas. Does not run the model |
| `m22_1_3` | Readout-geometry null. `predict.py` builds closed-form and Jacobian predictions. `compare.py` is the only reader of the measurement table |
| `m22_1_3b` | Stored-field first-order structure. No model forward |

Configs and reports for these milestones are on the replay branch. M22.1.3 caches `artifacts/m22_1_3/*.npy` and `predictions.json`.

### 3.4 M21.2.4.x benchmark utilities

`src/cognitive_self_model/benchmark/`

- `environment.py`: frozen linear data-generating process and oracle labels.
- `evidence.py`: 16-dimensional endogenous feature encoder. Not an evidence graph and not a mechanism object.
- `estimator.py`: L2 logistic regression from that vector to `q_invalid`.
- `calibration.py`: weighted PAVA isotonic calibrator.
- `pipeline.py`: episode-disjoint splits, estimator fit, eight-key NPZ bundle, reproducibility manifest.
- `bundle.py`, `tracks.py`, `reference_model.py`, `data.py`, `identifiability_dataset.py`.

`src/legacy_import/m212433_calibration_audit.py` (2045 lines) is the calibration-stress audit (ECE, episode bootstrap, transfer, verdict text). `m21_2_4_3_1_pipeline.py` and `m21_2_4_3_3_audit.py` are thin entrypoints onto the maintained modules. `m21_2_4_3_4_identifiability_benchmark.py` is a small CSV benchmark runner.

DEC-011 forbids using the M21 environment as arm P. It is a linear data-generating process, not a network.

### 3.5 Shared reproducibility helpers

`src/common/reproducibility.py`: git commit, branch name, clean/dirty. Returns `None` for the branch in detached HEAD. No Python, library, CUDA, model, dataset, or config hashes.

`src/common/io_contracts.py`: file sha256 and the eight-key NPZ contract.

Empty packages: `src/audits/`, `src/calibration/`, `src/estimator/`.

### 3.6 M22.1-R preregistration

`configs/m22_1_r_replay.yaml` is a frozen preregistration. Its own header says "No run." `artifacts/m22_1_r/manifest.sha256` lists five reference JSON digests. `control_plane/RECOVERY.yaml` milestone `M22.1-R` says the harness is not written and `output_artifacts` is empty.

There is no replay runner, no comparator module, and no script that assigns `REPLAY_EXACT` or `REPLAY_NUMERIC_EQUIVALENT`.

### 3.7 Planted-transformer scripts already in the tree

| Path | What it is |
| --- | --- |
| `scripts/mrsm_p_v0.py` | Standalone CPU smoke script. Not a package |
| `mrsm_p_v1.py` | Later standalone two-hop script. Modes include smoke and a checkpoint audit. Root-level module, not under `src/` |
| `control_plane/P_SPEC.md` | Frozen construction definition: 2 layers, 4 heads, `d_model` 64, `d_mlp` 128, `n_ctx` 20, `d_vocab` 128, seed 1729, planted heads L0H0 and L1H1 |
| `reports/mrsm_p_v1/checkpoint_mask_audit.json` | Audit of checkpoint `reports/mrsm_p_v1/p_model_state.pt` |

The checkpoint file and both smoke holdout manifests named in `P_CONSTRUCTION_ATTEMPTS.yaml` are not in git. `*.pt` is gitignored on both `main` and the replay branch.

`P_SPEC.md` records `shared_adapter_in_v1_script: ABSENT` and `static_weight_detector_procedure: NOT_SEPARATELY_SPECIFIED`. `p_construction.outcome` is still null.

### 3.8 Notebooks

`notebooks/source/` on the replay branch vendors Colab notebooks for provenance. `notebooks/source/README.md` says they must not be imported or executed as the pipeline. It also says the governance layer from the M19 notebook is not ported, and that the M20.6.3.2 Qwen benchmark lives in `notebooks/cognitive_self_model_latest.ipynb`.

No `src/` module is named M20. The M20.6.3.2 recovery record on the replay branch has `recovery_grade_local: POINTER_ONLY` and environment fields `UNKNOWN`.

### 3.9 Unmerged IG-0 gate

`cursor/initial-self-model-gate-9523` contains `src/cognitive_self_model/initial_gate.py`, `configs/initial_gate.json`, `scripts/run_initial_gate.py`, and `tests/test_initial_gate.py`. It is a numpy mechanism-library gate (add, sub, mul, copy) with a linear predictor beside it. It is not referenced by the control plane, not an ancestor of `344899e`, and not on `main`. It is not a frozen MRSM module.

## 4. Duplicated or obsolete implementations

| Item | Observation |
| --- | --- |
| Isotonic calibrator | A smaller PAVA class lives in `benchmark/calibration.py`. The long audit lives in `legacy_import/m212433_calibration_audit.py`. The notebooks describe the package as the maintained benchmark and the legacy module as the audit |
| M21 pipeline entry | Maintained code is `benchmark/pipeline.py`. `legacy_import/m21_2_4_3_1_pipeline.py` and `scripts/run_m21_2_4_3_1_pipeline.py` delegate to it. The notebooks are provenance copies |
| Planted scripts | `scripts/mrsm_p_v0.py` and `mrsm_p_v1.py` are two standalone implementations of a planted transformer. v1 is the later script. Neither is the spec's `src/mrsm/planted_model.py` |
| Bootstrap | Separate implementations in `m22_1/metrics.py`, `m22_1_2/metrics.py`, and `legacy_import/m212433_calibration_audit.py` |
| Git identity | `src/common/reproducibility.py` and `m22_1/runtime_info.py` both read git state. Only `runtime_info.py` records dirty state and strips credentials from the origin URL |
| sha256 helpers | Reimplemented in several modules (`bundle.py`, `io_contracts.py`, `mrsm_p_v1.py`, `m22_1_3`, `m22_1_3b`, the control-plane validator) |
| Evidence | `benchmark/evidence.py` is a feature encoder. `control_plane/CLAIMS.yaml` is a claim ledger. Neither is the prediction `evidence_refs` object in specification section 5 |
| Empty packages | `src/audits`, `src/calibration`, `src/estimator` have no implementation |
| `legacy/` | Empty on `main` and on the replay branch. Historical code is in `notebooks/source/` and `src/legacy_import/` |
| IG-0 | Unmerged side branch. Not wired to M22 or to the specification |
| M22.2 | Control plane status `NOT_AUTHORIZED`. Specification section 0 says MRSM replaces M22.2 as the next scientific gate. No M22.2 implementation was found |

No existing module was deleted or rewritten in this audit.

## 5. Spec section 17 tree

None of the following exist on `main` or on `344899e`:

```text
mrsm/
src/mrsm/{schemas,planted_model,qwen_adapter,interventions,discovery,evidence,self_model,baselines,policy,metrics,statistics,leakage,runner,audit}.py
configs/{mrsm_prereg,planted,qwen}.yaml
tests/{test_schemas,test_planted_ground_truth,test_interventions,test_baselines,test_leakage,test_metrics,test_determinism}.py
artifacts/planted_ground_truth.json
scripts/{build_planted,prereg_check,run_p,run_q,analyze,final_decision}.py
```

`runs/` does not exist. `reports/` exists on the replay branch for earlier milestones, and this audit adds the three P0/P1 reports on the audit branch.

Specification section 17 says not to duplicate a frozen equivalent. The closest frozen modules are listed in the reuse map. They do not implement `SelfModelCore` or the H1–H4 endpoints.

## 6. Current commits named by this audit

| Name | Commit |
| --- | --- |
| Specification on `main` | `5dd0e076c60e80eb64eabdae0ddb3429d2b46388` |
| Scientific head used for the code inventory | `344899e85cde6f8f8a392492c22e5e0b74cf21bb` on `cursor/m22-1-r-behavioral-replay-265a` |
| M22.1 code-under-test named by `configs/m22_1_r_replay.yaml` | `db8d03fad94e2f20f1a9f593a7f6b9eaa5f81d47` |
| M22.1 run-recorded commit | `e4713f0afb99b2a0cc1bacb35fce366a9b2ac47b` |
| P_SPEC freeze commit recorded in `MRSM_BUDGET.yaml` | `ad3cbc0f185a397d34eddabb8032fd3dd2144b1c` |
| Control-plane `STATE.yaml` `as_of_commit` | `0e6805e5153b162d4cabf7bae429455a7921e0a9` |

`STATE.yaml` still records `main` tip as `c08f0e64a5dd543002b30dfabfa5183d1b4e39d0`. That was true when the ledger was written. Current `main` is `5dd0e07`. The same ledger says M22.1.3 was a local branch only. `origin/m22-1-3-readout-null` now exists.

## 7. Ambiguities found while auditing, not resolved

These are conflicts between documents. This audit does not choose a winner.

1. Specification section 18 says P0 creates `docs/MRSM_CONTRACT.md` and `configs/mrsm_prereg.yaml`. The task for this session limits P0 to a repository audit and forbids preregistration. Those two files were not created.

2. Specification section 24 uses terminal outcomes GO, REDEFINE, and STOP, plus `INCOMPLETE_BUDGET` when a gate is unassessed. GO there requires P H1–H4 and Q H2, Q H3, and Q H4. Q H1 is diagnostic. Frozen DEC-010 uses STOP, REDEFINE_CAUSAL_AUDIT, REDEFINE_SCALE, INCOMPLETE_BUDGET, GO, and CONTRACT_ERROR. DEC-010 GO requires all eight gates, including Q.H1, to be PASS. DEC-010 STOP also matches P.H2 FAIL in cases the specification's STOP paragraph does not state. DEC-011 says it does not modify DEC-010.

3. Specification sections 9–12 give proposed numerical thresholds and call them proposed frozen thresholds. Section 20 says P2 freezes the numerical thresholds. DEC-011 says metrics and thresholds for P.H1–H4 and Q.H1–H4 are deferred. `P_SPEC.md` has a causal-validation predicate and marks `not_a_gate_threshold: true`. No threshold file was edited.

4. Specification section 6.2 gives example planted mechanisms M_COPY, M_INHIBIT, and M_BIND, and says the exact implementation can be adjusted during P2. `P_SPEC.md` is already status FROZEN with a two-hop key/value task and planted heads `[0, 0]` and `[1, 1]`.

5. Specification section 11 motivates H3 with M18.7/A4 and requires the phase rotation `[X, Y] -> [Y, -X]` where applicable. DEC-011 records M18.7 as `PROVENANCE_UNRESOLVED` and says the H3 definition must not depend on M18.7. A search recorded in DEC-011 found no `phase rotation` text in the then-local refs. The specification is the first in-repo statement of that rotation.

6. Specification section 0 budgets 4 calendar weeks, 8 full MRSM runs, and at most 2 full Qwen runs. DEC-011 starts the 28-day MRSM clock at `p_freeze`, which has not happened, and defines a full run as any execution that reads a locked holdout. The Q limit in `MRSM_BUDGET.yaml` is `q_gpu_full_runs`, a sub-limit, not an extra allowance. A separate 5-day P-construction clock is recorded with deadline `2026-10-03T14:17:59+00:00`. The specification does not describe that 5-day clock.

7. There is no module named an evidence graph. The claim ledger and the 16-D encoder are different objects. The specification's prediction field `evidence_refs` has no implementation.

## 8. What this P0 audit did not do

- Did not create the section 17 tree.
- Did not merge the scientific branch into `main`.
- Did not rewrite or delete duplicate modules.
- Did not change DEC-010, DEC-011, `P_SPEC.md`, replay tolerances, or any threshold.
- Did not train a planted model, open a holdout, or load Qwen weights.
