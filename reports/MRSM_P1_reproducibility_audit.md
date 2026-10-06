# MRSM P1 — Reproducibility / Infrastructure Audit

Date: 2026-09-28

Specification section 19 (P1 — M22.1-R Replay) requires:

- run the reproducibility harness
- enforce offline/local model loading
- verify the pinned revision
- record SHA256 weight hashes
- freeze comparator tolerances
- eliminate `UNKNOWN` environment fields

This audit ran the engineering checks that already exist. It did not write a harness, did not change tolerances, did not edit recovery records, did not load Qwen weights, and did not assign a replay tier. Assigning `REPLAY_FAIL` would change the decision weight of M22.1. That is outside this audit. Specification section 19 also says a replay failure does not stop MRSM; this document does not apply that consequence.

No MRSM experiment was run. Previous scientific labels were not reinterpreted. M22.1 remains the ledger status `CANDIDATE`. M22.1-R remains `PREREGISTERED`.

## 1. Where the replay infrastructure is

All of it is on `cursor/m22-1-r-behavioral-replay-265a` at `344899e85cde6f8f8a392492c22e5e0b74cf21bb`. None of it is on `main`.

| Piece | Path | State |
| --- | --- | --- |
| Frozen replay preregistration | `configs/m22_1_r_replay.yaml` | Present. Header: frozen, "No run." `revision_pin.status: PENDING_HARNESS_ENFORCEMENT`. Tolerances `label: PROPOSED` |
| Reference digest list | `artifacts/m22_1_r/manifest.sha256` | Present. Five `reports/M22_1/*.json` lines |
| Reference JSON | `reports/M22_1/{certificate,frozen_candidate,intervention_not_validated,intervention_preflight_results,manifest}.json` | Present |
| Recovery entry | `control_plane/RECOVERY.yaml` milestone `M22.1-R` | `output_artifacts: []`. Gaps: "Preregistration only; no replay run; harness not written." `execution_records: []` |
| Schema checks | `src/cognitive_self_model/control_plane/validate.py` (`_check_m22_1_r_replay_config`) and `tests/test_control_plane.py` | Checks required keys, DEC-007 success tiers, and rejection of renamed tolerance keys. Does not compare replay outputs |
| Code named as code-under-test | commit `db8d03fad94e2f20f1a9f593a7f6b9eaa5f81d47` on `cursor/m21-2-4-3-1-integration-c559` | The yaml says this commit has no static revision pin. `loader.py` still calls `model_info(model_id)` |
| Comparator | — | Absent |
| Replay entrypoint | — | Absent |

The yaml's `entrypoint_call` says a future harness must call `run_preflight` with an absolute output directory outside the worktree, and must not call `scripts/run_m22_1_preflight.py` or `run.main()`. That harness source does not exist.

Recorded revision, not enforced in code: `060db6499f32faf8b98477b0a26969ef7d8b9987` for `Qwen/Qwen2.5-0.5B`.

DEC-007 success tiers accepted by the yaml are `REPLAY_EXACT` and `REPLAY_NUMERIC_EQUIVALENT`. The yaml also defines `REPLAY_BEHAVIORAL_EQUIVALENT`, `REPLAY_DIVERGENT`, `REPLAY_BLOCKED`, and `REPLAY_INVALID`. Specification section 19 names `REPLAY_EXACT`, `REPLAY_NUMERIC_EQUIVALENT`, and `REPLAY_FAIL`. `REPLAY_FAIL` is not a tier name in the yaml.

## 2. Checks that were run

Worktree: detached `344899e`, clean. Commands were not scientific runs. Qwen `from_pretrained` was not called. `mrsm_p_v0.py` and `mrsm_p_v1.py` were not executed. Holdout files were not generated.

Audit machine, recorded here and not written into `RECOVERY.yaml`:

| Field | Value |
| --- | --- |
| Python | 3.12.3 |
| NumPy | 2.4.4 |
| PyTorch | 2.14.0+cpu |
| CUDA available | false |
| transformers | 5.17.0 |
| transformer_lens | 3.9.0 |
| SciPy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pandas | 2.3.3 |
| pytest | 8.4.2 |
| Platform | Linux 6.12.94+, x86_64, 4 CPUs |

PyTorch 2.14.0+cpu, transformers 5.17.0, transformer_lens 3.9.0, and NumPy 2.4.4 match the versions stored on the M22.1 recovery record (NumPy) and the M22.1.1 recovery record (Python 3.12.3 and the same library versions). M22.1's own `environment.python` is still `UNKNOWN` in the ledger. This audit did not fill that field.

SciPy, scikit-learn, pandas, and pytest were installed so the existing suite could run. They are not part of a replay execution record.

### 2.1 Control-plane validator

```text
PYTHONPATH=src python3 -m cognitive_self_model.control_plane.validate
```

Exit code 0. Stderr:

```text
warning: FILE_MAP missing older tracked path reports/mrsm_p_v1/checkpoint_mask_audit.json
control plane OK
```

The validator only requires `reports/`, `configs/`, and `artifacts/` paths to be mapped. These tracked paths are unmapped and are outside that check, so they do not fail the validator:

- `mrsm_p_v1.py`
- `scripts/mrsm_p_v0.py`
- `control_plane/DEC-010_decision_table.md`
- `control_plane/DEC-011_mrsm_scope.md`
- `control_plane/P_SPEC.md`
- `control_plane/MRSM_BUDGET.yaml`
- `control_plane/P_CONSTRUCTION_ATTEMPTS.yaml`
- `control_plane/{STATE,CLAIMS,FILE_MAP,RECOVERY,README}.yaml` and `.md`
- `src/cognitive_self_model/control_plane/*.py`
- `tests/test_control_plane.py`
- `tests/test_decision_table.py`

`FILE_MAP.yaml` `as_of_commit` remains `0e6805e`.

### 2.2 Byte hashes of frozen reference artifacts

Recomputed SHA256 on `344899e`. Every listed digest matched.

| Artifact | Result |
| --- | --- |
| Five paths in `artifacts/m22_1_r/manifest.sha256` | all match, including `reports/M22_1/intervention_preflight_results.json` = `90fd22defdfff28f5dccd2d8e79a93d61ea3a5c2391e94e1a963fba964ed5303` |
| `artifacts/m22_1_r/manifest.sha256` itself | `89d07e833ebf0f3822146f7b2b787280604053bbb2c66d1d4c6fea34af0b8ec4`, equal to the recovery record |
| `configs/m22_1_r_replay.yaml` file bytes | `d9e1cf5dc3b88535cda0516b9b24ae3510082b7fed452f2e8adb86c574e8039a`, equal to the recovery `config_sha256` |
| `configs/m22_1_preflight.json` file bytes | `4c52fe878cd3582a3df873b43eab00b545623f98b25053a9c1c00801333716ca`, equal to the recovery file-bytes hash |
| 53 `FILE_MAP.yaml` entries that store `sha256` | 53 match, 0 mismatch, 0 missing files |
| `artifacts/m22_1_3/manifest.sha256` | all six paths match, including both `.npy` caches |
| `artifacts/m22_1_3b/manifest.sha256` | config, results, and report match |

Matching bytes show the stored JSON and caches are intact. They do not show that a new forward pass reproduces them.

### 2.3 Existing tests

`PYTHONPATH` included `src`. The suite on `344899e` was executed. No test in that suite calls `load_qwen`.

After PyTorch and TransformerLens were installed, the result was 114 passed, 0 failed, 0 skipped.

Before PyTorch was installed, `tests/test_m22_1_contract.py::test_loader_revision_helper_does_not_invent_a_sha` failed at import with `No module named 'torch'`. After PyTorch 2.14.0+cpu it passed. `tests/test_m22_1_intervention.py::test_tiny_hooked_transformer_registers_only_the_named_layer` skips unless `transformer_lens` imports. After transformer_lens 3.9.0 it passed, with a deprecation warning that `HookedTransformer` is deprecated and slated for removal in TransformerLens 4.0. `requirements.txt` on the replay branch pins `transformer_lens>=3.0,<4.0`.

Tests that check determinism and that passed:

- `tests/test_benchmark_pipeline_reproducibility.py`: two `run_pipeline` calls write identical `npz_bundle_sha256`.
- `tests/test_m22_1_contract.py`: directions are deterministic, unit, and orthogonal; prompt manifest hash is stable; the revision helper does not invent a sha when `model_info` fails.
- `tests/test_m22_1_1_panel.py`: the direction panel reproduces the frozen M22.1 direction hashes.
- `tests/test_m22_1_intervention.py`: alpha 0 does not change the residual; a repeated nonzero application is bitwise equal; a tiny local `HookedTransformer` fires only the named hook.
- `tests/test_m22_1_protocol.py`: synthetic instrument only. No Qwen weights.
- `tests/test_decision_table.py`: exhaustive DEC-010 table, including the 50-record incomplete-budget case.
- `tests/test_control_plane.py`: real ledger passes; tampered hashes fail; replay config rejects a renamed tolerance key and rejects `REPLAY_BEHAVIORAL_EQUIVALENT` as a DEC-007 success tier.

`scripts/install.sh` on the replay branch installs this stack and then runs `pytest tests/`. It was not used as the runner. The same pins were installed directly so the audit could name versions. The install script would also run pytest; it would not by itself create the missing replay harness.

### 2.4 Local model cache and offline load

`/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B` is absent.

`load_qwen` does not pass `local_files_only`. `resolve_hf_revision` calls `huggingface_hub.model_info(model_id)` with no revision argument. If the pinned load raises, lines 94–101 of `loader.py` drop the revision and call `from_pretrained` again.

This audit did not call the Hub and did not download weights. Offline load of the recorded revision cannot be verified on this machine because the snapshot is not present.

No SHA256 of Qwen weight files is stored in `RECOVERY.yaml`, `configs/m22_1_r_replay.yaml`, or `reports/M22_1/manifest.json`. The replay yaml says a weight-hash mismatch is not a `REPLAY_BLOCKED` condition because no reference weight hash exists.

### 2.5 Planted checkpoint bytes

`P_CONSTRUCTION_ATTEMPTS.yaml` records:

| Attempt | Recorded `model_sha256` | Artifact named | In git |
| --- | --- | --- | --- |
| MRSM_P_v0 smoke | `e13cc72d30d046f3adf63c41b4a0e4ae572215c710611aadc6c04ddd2a1234fa` | `reports/mrsm_p_v0/holdout_manifest.json` | holdout manifest absent; weights absent |
| MRSM_P_v1 smoke | `ea40cda33a054c1c7fd9b169686c2eff6a9ada1aab9f0d75324fbdd74a8986e6` | `reports/mrsm_p_v1/p_model_state.pt` and `reports/mrsm_p_v1/holdout_manifest.json` | both absent |

`reports/mrsm_p_v1/checkpoint_mask_audit.json` repeats checkpoint sha256 `ea40cda33a054c1c7fd9b169686c2eff6a9ada1aab9f0d75324fbdd74a8986e6` and sets `matches_recorded_unmasked_smoke` true. The `.pt` file is not in the tree, so that digest was not recomputed. Both branch gitignores ignore `*.pt`. The replay-branch gitignore also ignores `artifacts/*` and `*.npy`, while some already-tracked artifact files remain in history.

The v1 smoke record says the holdout was generated and content-hashed, with model evaluation null. The later checkpoint-audit JSON says `holdout.generated: false` and `holdout.accessed: false` for that audit mode. Those are two recorded invocations. The smoke holdout files are not available to re-hash.

`P_SPEC.md` lists holdout seeds 311 and 2029 as `ALREADY_GENERATED_PRE_FREEZE`. The files that would let a third party regenerate or confirm those token hashes are not in the repository.

## 3. Specification section 19 checklist

| Requirement | Result of this audit |
| --- | --- |
| Run the reproducibility harness | Not run. No harness exists. The validator and unit tests do not execute `run_preflight` and do not compare a new output to `reports/M22_1` |
| Enforce offline/local model loading | Not enforced in `loader.py`. Local Qwen snapshot absent. No code change was made |
| Verify pinned revision | Not verified by a load. The revision string is written in the yaml and in recovery. `revision_pin.status` remains `PENDING_HARNESS_ENFORCEMENT`. Hub was not contacted |
| Record SHA256 weight hashes | Qwen weight hashes are not recorded. Planted `model_sha256` values are recorded and the corresponding files are absent, so they were not rechecked |
| Freeze comparator tolerances | Not done. The yaml label is `PROPOSED`. DEC-009 freezes the key names `score_atol`, `score_rtol`, `delta_atol`, `mean_atol`, `null_atol`. Values were not edited. The note in the yaml says replay outputs must not be used to tune these values |
| Eliminate `UNKNOWN` environment fields | Not done. See section 4. No execution record was added |

Specification section 16 forbids execution with unresolved `UNKNOWN` environment fields. The replay was not executed.

## 4. `UNKNOWN` environment fields still in the ledger

`execution_records` is an empty list, which satisfies DEC-009's ban on sentinels inside execution records by having no records. Milestone entries still contain `UNKNOWN`. DEC-009 allows those sentinels in milestone preregistration entries. Specification section 19 says to eliminate them before the replay. They are still there.

| Milestone | Environment fields that are `UNKNOWN` | Other recovery facts relevant to replay |
| --- | --- | --- |
| M22.1 | `python` | `git_dirty: true`. `recovery_grade_local: PARTIAL`. `step0_class: ADDITIONS_ONLY_UNVERIFIED`. Run commit `e4713f0`. Code commit `db8d03f`. Torch, transformers, transformer_lens, and numpy are filled |
| M22.1-R | `python`, `torch`, `transformers`, `transformer_lens`, `numpy` | `commit: UNKNOWN`. `entrypoint: UNKNOWN`. No output artifacts |
| M22.1.1 | `numpy` | `recovery_grade_local: EXACT`. Chain grade `PARTIAL` because it depends on M22.1 |
| M22.1.2 | `python`, `torch`, `transformers`, `transformer_lens`, `numpy` | Ledger status in `STATE.yaml` is `INSUFFICIENT_EVIDENCE` |
| M18, M19, M20.6.3.2, and several early M21.2.3 / M21.2.4 entries | commit, entrypoint, config, and environment sentinels | Grades `ABSENT` or `POINTER_ONLY` |

M22.1.3 and M22.1.3b were not re-executed. Their stored manifests match the files on `344899e` (section 2.2). That is a byte check of cached artifacts, not a new readout forward pass.

## 5. What can be reproduced from the tree

Deterministic reproduction that this audit actually observed:

- Reference M22.1 JSON bytes match the replay manifest.
- M22.1.3 and M22.1.3b manifest bytes match.
- FILE_MAP content hashes match the files they name.
- The M21 pipeline test produces the same bundle hash on two runs in one process.
- Seeded residual directions and the M22.1.1 panel hash match the frozen tests.
- Additive residual intervention application is bitwise repeatable under PyTorch 2.14.0+cpu.
- The DEC-010 table and the control-plane ledger validate.

## 6. What cannot be reproduced from the tree

- A new M22.1 behavioral replay. There is no harness, no local Qwen snapshot, no weight hash, and the M22.1 Python version and dirty-tree code hashes are unresolved.
- The planted v0 and v1 checkpoints and their holdout manifests.
- `artifacts/planted_ground_truth.json`, which the specification requires and which does not exist.
- An MRSM run record with the section 16 fields. No runner writes git commit, dirty flag, Python, PyTorch, CUDA, model hash, dataset hash, holdout hash, seed, config hash, baseline version, preregistration version, runtime, and hardware together. `runtime_info.py` covers only git identity.

## 7. Replay tier

No tier is assigned.

`REPLAY_EXACT` and `REPLAY_NUMERIC_EQUIVALENT` require a completed comparison against a new run. That run was not started.

`REPLAY_BLOCKED`, as defined in the yaml, includes a missing local revision, a dirty worktree, or a changed reference hash. The reference hashes did not change. The local revision is missing. The worktree used for the checks was clean. The yaml also says the blocked check is for the harness subprocess, which does not exist. This audit therefore records the blocker and does not write a tier into the ledger.

## 8. Files this P1 audit changed

None of the scientific, config, control-plane, or threshold files. The only repository files added for P0 and P1 are the three reports named in the task. See `reports/MRSM_P0_repository_audit.md` for the repository map and `reports/MRSM_REUSE_MAP.md` for module reuse.
