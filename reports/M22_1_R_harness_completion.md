# M22.1-R harness completion

Engineering record for the M22.1-R replay harness. This document does not change M22.1 status, M22.1-R preregistration status, DEC-010, DEC-011, `P_SPEC.md`, or any tolerance value.

## 1. Starting commit/tree

The phase began on the scientific tree, not on `main`.

| Role | Commit | Branch |
| --- | --- | --- |
| M22.1 / M22.1-R infrastructure | `344899e85cde6f8f8a392492c22e5e0b74cf21bb` | `cursor/m22-1-r-behavioral-replay-265a` |
| MRSM specification on `main` | `5dd0e076c60e80eb64eabdae0ddb3429d2b46388` | `main` |

Working branch: `cursor/m22-1-r-harness-4e19`, created from `344899e85cde6f8f8a392492c22e5e0b74cf21bb`.

The specification file `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` was added from `5dd0e076c60e80eb64eabdae0ddb3429d2b46388` as commit `d96b3eeed1a3a4a8f94b1ac372e35b9a8489cd60`. That commit copies the specification onto the scientific tree. It is not a merge of `main`. Merging `main` into this branch, or merging this branch into `main`, would mix the scientific history with the specification-only history. The pull request base is therefore `cursor/m22-1-r-behavioral-replay-265a`.

`STATE.yaml` still records its historical `main` tip and `as_of` commit. Those fields were left unchanged.

Control-plane validation before the harness code, and again at `74efd6bbe960834f4936878c03ecf821c6e69c4b` before this report was added, exited 0. The only message was the pre-existing warning `FILE_MAP missing older tracked path reports/mrsm_p_v1/checkpoint_mask_audit.json`.

## 2. Final commit/tree

Harness implementation: `adb53f484f7a1a8e62709d132ff5beffedd123d7`.

Git-status fix, which is the commit named by the execution record: `74efd6bbe960834f4936878c03ecf821c6e69c4b`.

At that commit the worktree was clean. The execution record stores `git_commit` `74efd6bbe960834f4936878c03ecf821c6e69c4b` and `git_dirty` false.

The commit that adds the execution record, the `RECOVERY.yaml` splice, the `FILE_MAP.yaml` entries, and the first version of this report is `7d0775fe0c0822aba8a91c8a44c9df8c3bda8b66`. Its parent is `74efd6bbe960834f4936878c03ecf821c6e69c4b`. The tip commit on `cursor/m22-1-r-harness-4e19` only writes that hash into this paragraph. It does not change the harness or the execution record.

## 3. Files changed

Relative to `344899e85cde6f8f8a392492c22e5e0b74cf21bb`:

- `.gitignore` — track `artifacts/m22_1_r/execution_record.json`
- `docs/MRSM_Implementation_Spec_and_Cursor_Instructions_v1.0.md` — specification copied from `main`
- `scripts/run_m22_1_r_harness.py` — harness entrypoint
- `src/cognitive_self_model/m22_1/preflight.py` — optional `loader` argument; the default remains `load_qwen`
- `src/cognitive_self_model/m22_1_r/` — replay package
- `tests/test_m22_1_r_harness.py` — engineering tests
- `artifacts/m22_1_r/execution_record.json` — this attempt
- `control_plane/RECOVERY.yaml` — top-level `execution_records` splice only
- `control_plane/FILE_MAP.yaml` — entries for the new harness files
- `reports/M22_1_R_harness_completion.md` — this report

`configs/m22_1_r_replay.yaml` was not modified. Its file sha256 remains `d9e1cf5dc3b88535cda0516b9b24ae3510082b7fed452f2e8adb86c574e8039a`.

## 4. Existing components reused

- `configs/m22_1_r_replay.yaml` tolerance keys, tier names, and `dec007_success_tiers`
- `artifacts/m22_1_r/manifest.sha256` and `reports/M22_1/*.json`
- `src/cognitive_self_model/common/io_contracts.py` `file_sha256`, via `m22_1_r.hashing`
- `src/cognitive_self_model/m22_1/runtime_info.py` `git_identity` for commit and branch
- DEC-009 execution-record requirements in `control_plane/validate.py` (`entrypoint`, `invocation`, boolean `git_dirty`, `dirty_reason` when dirty, no sentinels)
- The historical `m22_1.loader.load_qwen` path, left in place for non-replay callers

## 5. New components added

- `m22_1_r.strict_loader.load_pinned_revision` — one `from_pretrained` call with the pinned revision and `local_files_only=True`
- `m22_1_r.weights` — local Hugging Face snapshot discovery and SHA256
- `m22_1_r.comparator.compare_bundles` — identity, score, delta, mean, and null comparisons
- `m22_1_r.harness.run_harness` — prerequisite check, tier assignment, execution record
- `m22_1_r.record.build_record` — sentinel rejection
- `scripts/run_m22_1_r_harness.py`

## 6. Revision pin status

Required revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.

The harness requires that exact revision. A different revision is rejected before `from_pretrained`. A pinned-load exception becomes `REPLAY_BLOCKED` and is not followed by an unpinned retry.

`configs/m22_1_r_replay.yaml` still says `revision_pin.status: PENDING_HARNESS_ENFORCEMENT`. That file was not edited. Enforcement is in the replay loader, not in a rewritten preregistration.

The live `huggingface_hub.model_info` recheck described under `revision_resolution.precondition_recheck` was not executed. This phase forbids network-fetched weights. The execution record warning states that. The local-cache clause of `REPLAY_BLOCKED` was evaluated on its own.

## 7. Weight hash status

Local snapshot path checked:

`/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987`

That directory is absent. `weight_status` is `local_snapshot_absent`. `weight_files` is empty.

No reference weight SHA256 is recorded in the repository. The execution record sets `expected_weight_hashes` to `ABSENT_NO_REFERENCE_RECORDED`. No expected hash was invented.

When a caller supplies an expected map and the local files disagree, the harness assigns `REPLAY_DIVERGENT` with `failure_class` `weight_identity` and does not call `from_pretrained`. The preregistration says a weight-hash mismatch is not `REPLAY_BLOCKED`, because no reference hash exists. A supplied mismatch is treated as an identity disagreement under the existing `REPLAY_DIVERGENT` tier.

No model-weight binary was committed.

## 8. Comparator status

`compare_bundles` reads the tolerance numbers from `configs/m22_1_r_replay.yaml`.

- Field equality of the configured numeric fields, with identity fixed, is `REPLAY_EXACT`.
- Identity fixed and every numeric field inside its family tolerance is `REPLAY_NUMERIC_EQUIVALENT`.
- Identity fixed and at least one numeric field outside tolerance is `REPLAY_BEHAVIORAL_EQUIVALENT`.
- Identity or structural mismatch is `REPLAY_DIVERGENT`.
- A non-finite value, or a missing or duplicate composite cell key, is `REPLAY_INVALID`.

`REPLAY_BLOCKED` is assigned by the harness when a prerequisite is missing. The comparator does not emit it.

DEC-007 success tiers in the yaml are only `REPLAY_EXACT` and `REPLAY_NUMERIC_EQUIVALENT`. `REPLAY_BEHAVIORAL_EQUIVALENT` is recorded and is not a success tier. The harness does not emit `REPLAY_FAIL`.

A self-comparison of the recorded `reports/M22_1` JSON is `REPLAY_EXACT` in the engineering test. That comparison does not load Qwen.

## 9. Execution-record status

Path: `artifacts/m22_1_r/execution_record.json`.

SHA256: `d8c8e4cef3c1381a8a7b8e83d575b30f9b8be513ec0284e43edcc8a7cc5e2344`.

The same object is the sole item in the top-level `execution_records` list in `control_plane/RECOVERY.yaml`. Milestone text, including historical `UNKNOWN` values, was not rewritten.

The record includes the DEC-009 strings `entrypoint` and `invocation`, boolean `git_dirty` false, git commit and branch, Python, OS, NumPy, PyTorch, Transformers, TransformerLens, CUDA availability, CUDA version `N/A` because `torch.cuda.is_available()` is false, GPU identity `N/A` for the same reason, model id, exact revision, local snapshot path, weight file list, the absent-reference weight sentinel, reference artifact hashes, holdout identity (the M22.1 prompt manifest; there is no separate holdout file), config path and sha256, tolerance label `PROPOSED` and the unchanged tolerance values, timestamp, command, replay tier, warnings, and the block reason.

No required field is `UNKNOWN`. `N/A` is used only for CUDA version and GPU identity, with a reason string.

`forward_executed` is false. `scientific_failure` is false. `scientific_claim_updated` is false.

The shared git helper treats empty `git status --porcelain` stdout as a failed read. A first harness attempt therefore recorded a false dirty block on a clean tree. That record was discarded. The replay path now reads porcelain itself. The shared helper was not changed. The retained record was produced after that fix, on a clean worktree.

The timestamp is the only nondeterministic field. No forward seed was consumed.

## 10. Replay tier/status

`REPLAY_BLOCKED`.

`failure_class`: `prerequisite`.

`dec007_success`: false.

`scientific_failure`: false.

Single block reason: the local snapshot for revision `060db6499f32faf8b98477b0a26969ef7d8b9987` is absent. This is a missing prerequisite. It is not a tolerance failure and not a scientific failure.

## 11. Tests executed and results

Command:

`PYTHONPATH=src python3 -m pytest tests/test_m22_1_r_harness.py tests/test_m22_1_contract.py tests/test_control_plane.py tests/test_m22_1_protocol.py -q`

Result: 56 passed.

`tests/test_m22_1_r_harness.py` covers:

- A — a different revision is rejected before `from_pretrained`
- B — a pinned-load failure performs one call and does not retry unpinned
- C — a supplied weight-hash mismatch is `REPLAY_DIVERGENT` and does not load
- D — a reference copy is `REPLAY_EXACT`; a 1e-8 score perturbation is `REPLAY_NUMERIC_EQUIVALENT`
- E — a tolerance violation is `REPLAY_BEHAVIORAL_EQUIVALENT`; an identity change is `REPLAY_DIVERGENT`; a non-finite value is `REPLAY_INVALID`
- F — a missing snapshot is `REPLAY_BLOCKED` with `scientific_failure` false
- G — required execution-record fields are present and contain no `UNKNOWN` sentinel
- H — `configs/m22_1_r_replay.yaml` bytes and tolerance numbers are unchanged
- clean porcelain is `git_dirty` false; a non-empty porcelain is dirty
- the strict loader source does not call `model_info` or `resolve_hf_revision`
- the RECOVERY splice preserves milestone text

No test loads Qwen or runs an MRSM experiment.

## 12. Control-plane validator result

Command: `PYTHONPATH=src python3 -m cognitive_self_model.control_plane.validate`.

Before this report was tracked, and after the execution record was spliced into `RECOVERY.yaml` while still untracked: exit 0.

Pre-existing warning, unchanged by this phase: `FILE_MAP missing older tracked path reports/mrsm_p_v1/checkpoint_mask_audit.json`.

After `FILE_MAP.yaml` listed this report and `artifacts/m22_1_r/execution_record.json`, the same command exited 0. The only message was that same pre-existing warning. This phase did not add a validator error.

## 13. Unresolved ambiguity

- `revision_resolution.precondition_recheck` asks for a live `model_info` comparison before the harness. This phase forbids network-fetched weights, so that recheck was not run. The local-cache absence is itself a `REPLAY_BLOCKED` condition in the same yaml paragraph. The two instructions were not collapsed into one action.
- `revision_pin.status` remains `PENDING_HARNESS_ENFORCEMENT` inside the frozen yaml. The harness enforces the pin in code. Changing the yaml status would edit the preregistration.
- The specification text uses `REPLAY_FAIL`. The replay yaml does not. The harness uses the yaml names.
- Specification outcome names, H1–H4 thresholds, mechanism labels, and budget wording still disagree with DEC-010, DEC-011, and `P_SPEC.md`. Those disagreements were not resolved.
- No reference weight hash exists. A future hash has to be recorded by a later authorized step. This phase does not choose one.

## 14. Blocked prerequisite

The Hugging Face snapshot for `Qwen/Qwen2.5-0.5B` at revision `060db6499f32faf8b98477b0a26969ef7d8b9987` is not on this machine. Until that exact offline snapshot is present, a forward replay cannot start. A different revision, `main`, or an unpinned `from_pretrained` is not a substitute.

## 15. No MRSM scientific implementation was started

No `src/mrsm/` package, SelfModelCore, MRSM baseline, H1/H2/H3/H4 implementation, planted ground truth, MRSM runner, or MRSM leakage test was created or modified.

## 16. No scientific threshold was changed

`configs/m22_1_r_replay.yaml` is byte-identical to the preregistered file. Tolerance values remain `score_atol` 1.0e-5, `score_rtol` 1.0e-5, `delta_atol` 1.0e-5, `mean_atol` 1.0e-6, `null_atol` 1.0e-6, label `PROPOSED`. DEC-010, DEC-011, and `P_SPEC.md` were not edited.

## 17. No scientific run was performed

The official command was:

`PYTHONPATH=src python3 scripts/run_m22_1_r_harness.py --update-recovery`

`--execute-forward` was not passed. `forward_executed` is false. The M22.1 preflight scoring pass was not run. Qwen weights were not downloaded.

## 18. Recommendation for the next phase

The harness can audit a replay once the exact offline snapshot exists. The evidence from this attempt supports only `REPLAY_BLOCKED` for that missing snapshot.

Do not start P2 preregistration, planted ground truth, SelfModelCore, MRSM baselines, H1/H2/H3/H4, or a Qwen scientific evaluation from this result.
