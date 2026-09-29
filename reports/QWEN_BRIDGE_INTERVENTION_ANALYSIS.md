# Qwen bridge intervention analysis

Label: `QF-BRIDGE-DIAGNOSTIC`

This file does not replace `reports/MRSM_FINAL_RESULT.md` or `reports/Q_FAILURE_ANALYSIS.md`.

## 1. Executive Summary

Selected bridge experiment: `M22.1`.
Bridge result: `BRIDGE_SIGNAL_PRESENT_BUT_CONTEXT_DEPENDENT`.
Primary diagnosis: `INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED`.

Do not modify the Self-Model or rerun MRSM. The next single step is a new preregistered decision to report each frozen regime separately, keeping every regime, before any architectural change.

## 2. Frozen MRSM Context

P H1–H4 remain PASS. Q remains H1 FAIL, H2 FAIL, H3 PASS, H4 FAIL.
Leakage remains PASS on both arms. DEC-010 remains `REDEFINE_SCALE`.
`p_freeze` remains `2cccafd047044332828eaf602b69f4267852cba2`.
Qwen revision remains `060db6499f32faf8b98477b0a26969ef7d8b9987`.
No MRSM gate was rescored.

## 3. Historical Qwen Experiment Inventory

M21 configs in this repository do not name Qwen. They are not in this inventory.

### M22.1

- commit: `e4713f0afb99b2a0cc1bacb35fce366a9b2ac47b`
- model_name: `Qwen/Qwen2.5-0.5B`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- dtype: `float32`
- device: `cpu`
- tokenizer: `UNKNOWN`
- prompt_task: `frozen M22.1 prompts; logit margin of ' yes' minus ' no'`
- input_distribution: `18 prompts, roles discovery/validation/replication, regimes completion/instruction/syntax`
- layer_head: `candidate layers 0, 8, 15, 23; selected layer 23; not an attention head`
- intervention_type: `additive last-token residual, blocks.{layer}.hook_resid_post`
- intervention_strength: `1.0`
- target_metric: `actual_delta = intervened logit margin minus unhooked margin`
- baseline: `unhooked forward`
- effect_size: `0.029387950897216797`
- control: `alpha 0 null and orthogonal direction at the selected layer`
- reported_result: `CANDIDATE`
- artifact_path: `reports/M22_1/certificate.json`

### M22.1.1

- commit: `UNKNOWN`
- model_name: `Qwen/Qwen2.5-0.5B`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- dtype: `float32`
- device: `cpu`
- tokenizer: `UNKNOWN`
- prompt_task: `same frozen M22.1 prompts; README states logit(9834)-logit(902)`
- input_distribution: `same 18-prompt split`
- layer_head: `blocks.23.hook_resid_post only; directions D1-D8`
- intervention_type: `additive last-token residual`
- intervention_strength: `1.0`
- target_metric: `paired delta`
- baseline: `unhooked margin`
- effect_size: `README discovery D1 mean at +1 is +0.029388`
- control: `D2 is the M22.1 orthogonal control; alpha 0 is on the grid`
- reported_result: `INSUFFICIENT_EVIDENCE`
- artifact_path: `reports/M22_1_1/intervention_space_report.json`

### M22.1.2

- commit: `UNKNOWN`
- model_name: `UNKNOWN`
- model_revision: `UNKNOWN`
- dtype: `UNKNOWN`
- device: `UNKNOWN`
- tokenizer: `UNKNOWN`
- prompt_task: `reuses M22.1.1 measurements; no new model forward`
- input_distribution: `validation and replication matrices from M22.1.1`
- layer_head: `UNKNOWN`
- intervention_type: `none; response-space reanalysis`
- intervention_strength: `UNKNOWN`
- target_metric: `blocked-cell MAE`
- baseline: `B0 through B7 in reports/m22_1_2_response_space_report.md`
- effect_size: `UNKNOWN`
- control: `B7 cell permutation is reported`
- reported_result: `INSUFFICIENT_EVIDENCE`
- artifact_path: `reports/m22_1_2_response_space_results.json`

### M22.1.3

- commit: `UNKNOWN`
- model_name: `Qwen/Qwen2.5-0.5B`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- dtype: `UNKNOWN`
- device: `UNKNOWN`
- tokenizer: `UNKNOWN`
- prompt_task: `readout reconstruction of recorded residual effects`
- input_distribution: `discovery, validation, replication`
- layer_head: `final residual through RMSNorm and unembedding; site inherited from M22.1`
- intervention_type: `reconstruction, not a new intervention`
- intervention_strength: `1.0`
- target_metric: `reconstructed delta versus recorded delta`
- baseline: `recorded M22.1.1 deltas`
- effect_size: `report states validation D1 reconstructed delta +0.028637`
- control: `alpha grid inherited; not a new null experiment`
- reported_result: `READOUT_RECONSTRUCTION_EXACT`
- artifact_path: `reports/m22_1_3_readout_null_report.md`

### M22.1.3b

- commit: `UNKNOWN`
- model_name: `UNKNOWN`
- model_revision: `UNKNOWN`
- dtype: `UNKNOWN`
- device: `UNKNOWN`
- tokenizer: `UNKNOWN`
- prompt_task: `reinterpretation of stored measurements`
- input_distribution: `UNKNOWN`
- layer_head: `UNKNOWN`
- intervention_type: `none`
- intervention_strength: `UNKNOWN`
- target_metric: `energy and cosine summaries`
- baseline: `UNKNOWN`
- effect_size: `UNKNOWN`
- control: `UNKNOWN`
- reported_result: `REINTERPRETATION_DESCRIPTIVELY_SUPPORTED`
- artifact_path: `reports/M22_1_3b/README.md`

### M22.1-R

- commit: `74efd6bbe960834f4936878c03ecf821c6e69c4b`
- model_name: `Qwen/Qwen2.5-0.5B`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- dtype: `UNKNOWN`
- device: `UNKNOWN`
- tokenizer: `UNKNOWN`
- prompt_task: `replay harness; no forward in the recorded attempt`
- input_distribution: `UNKNOWN`
- layer_head: `UNKNOWN`
- intervention_type: `not executed`
- intervention_strength: `UNKNOWN`
- target_metric: `UNKNOWN`
- baseline: `UNKNOWN`
- effect_size: `UNKNOWN`
- control: `UNKNOWN`
- reported_result: `REPLAY_BLOCKED`
- artifact_path: `artifacts/m22_1_r/execution_record.json`

## 4. Model / Revision Identity Audit

- `M22.1`: `SAME_MODEL_DIFFERENT_SETUP`. Same revision, dtype, hook, prompts, alpha 1.0, and outcome texts. Tokenizer name is not stored. Indexing differs because M22.1 selects one layer and MRSM names eight residual slots. The historical metric is intervened-minus-baseline; MRSM stores baseline-minus-intervened.
- `M22.1.1`: `SAME_MODEL_DIFFERENT_SETUP`. Same revision and layer-23 hook, but eight directions at one layer rather than the MRSM eight-slot map.
- `M22.1.2`: `NOT_COMPARABLE`. No new forward. Model revision is not restated in the results JSON.
- `M22.1.3`: `SAME_MODEL_DIFFERENT_SETUP`. Reconstruction of stored effects, not the MRSM eight-slot protocol.
- `M22.1.3b`: `NOT_COMPARABLE`. Stored-number reinterpretation. Revision is not restated in the README front matter.
- `M22.1-R`: `NOT_COMPARABLE`. The recorded attempt did not run a forward. The pin matches, and the snapshot was absent at that time.

No ranking is assigned.

## 5. Selected Bridge Experiment

`M22.1`

Rule, in order: direct causal intervention; negative or null control present; reported effect size; reproducible code; least external dependency, then experiment id.
Eligible ids: `['M22.1', 'M22.1.1']`.
M22.1 is the direct causal intervention with a null, an orthogonal control, a reported effect, and no dependence on a later audit.
Its recorded status is `CANDIDATE`, not `VALIDATED_FOR_M22`. This bridge does not promote that status.

## 6. Historical Setup Reproduction

The historical protocol ran through `execute_protocol` on the local pinned snapshot.
`load_qwen` was not used, so there was no unpinned retry.
Reproduced status: `CANDIDATE`.
Selected layer: `23`.
Match to the recorded certificate: `True`.
- discovery_mean: reproduced `0.029388`, recorded `0.029388`
- validation_mean: reproduced `0.0286371`, recorded `0.0286371`
- replication_mean: reproduced `0.0296938`, recorded `0.0296938`
- validation_control_mean: reproduced `0.105415`, recorded `0.105415`
- replication_control_mean: reproduced `0.107654`, recorded `0.107654`

## 7. MRSM-Compatible Reproduction

The same forwards are reported as MRSM margin drop, which is the negation of `actual_delta`.
No self-model was fit and no H gate was scored.

- discovery drop mean `-0.029388`, sign consistency `1`, SNR `10.698`
- validation drop mean `-0.0286371`, sign consistency `1`, SNR `10.8684`
- replication drop mean `-0.0296938`, sign consistency `1`, SNR `12.5795`

## 8. Intervention Signal Comparison

Historical metric is preserved. The MRSM metric is additional and does not replace it.

| split | historical delta | MRSM drop | null delta | control delta |
| --- | --- | --- | --- | --- |
| discovery | 0.029388 | -0.029388 | 0 | UNKNOWN |
| validation | 0.0286371 | -0.0286371 | 0 | 0.105415 |
| replication | 0.0296938 | -0.0296938 | 0 | 0.107654 |

## 9. Context Stratification

QF-2 regime means are reused. Within-regime standard deviation and median were not stored, so they are `UNKNOWN`.

| slot | completion | instruction | syntax | pooled | cancellation |
| --- | --- | --- | --- | --- | --- |
| L0H0 | 0.0963473 | -0.0752206 | -0.043036 | -0.00730308 | True |
| L0H1 | -0.0637145 | -0.00601387 | -0.0795894 | -0.0497726 | False |
| L0H2 | 0.0429649 | -0.069767 | 0.0139337 | -0.00428947 | True |
| L0H3 | 0.0517416 | 0.04526 | -0.048106 | 0.0162985 | True |
| L1H0 | 0.0727177 | -0.00567102 | -0.0369174 | 0.0100431 | False |
| L1H1 | -0.0316253 | -0.064342 | -0.064806 | -0.0535911 | False |
| L1H2 | -0.0299549 | -0.0298491 | -0.0283537 | -0.0293859 | False |
| L1H3 | -0.10939 | -0.105217 | -0.11155 | -0.108719 | False |

These slots are intervention names. They are not discovered mechanisms.

## 10. Pooled vs Stratified Analysis

Stratification reveals opposing regime signs whose absolute values exceed the pooled mean.
Cancelling slots: `['L0H0', 'L0H2', 'L0H3']`.

## 11. Simpson/Cancellation Audit

A slot is `CONTEXT_CANCELLATION` when two regimes have opposite signs, each absolute regime mean clears the existing 1e-4 floor, and the absolute pooled mean is smaller than both.
No regime was removed.

## 12. Intervention Strength Analysis

Strength values are the recorded M22.1 magnitude grid. The config does not define weak, medium, or strong, so those names are not assigned.

### validation

| alpha | effect | null | SNR | sign | context consistency |
| --- | --- | --- | --- | --- | --- |
| -2 | -0.0559072 | 0 | 10.3669 | -1 | True |
| -1 | -0.0281853 | 0 | 10.5643 | -1 | True |
| 0 | 0 | 0 | null | 0 | True |
| 1 | 0.0286371 | 0 | 10.8684 | 1 | True |
| 2 | 0.0577227 | 0 | 10.9493 | 1 | True |

### replication

| alpha | effect | null | SNR | sign | context consistency |
| --- | --- | --- | --- | --- | --- |
| -2 | -0.0577861 | 0 | 9.76246 | -1 | True |
| -1 | -0.029163 | 0 | 10.5568 | -1 | True |
| 0 | 0 | 0 | null | 0 | True |
| 1 | 0.0296938 | 0 | 12.5795 | 1 | True |
| 2 | 0.0598929 | 0 | 13.7959 | 1 | True |

## 13. Negative Controls

- Intact: unhooked logit margin.
- Null: alpha 0 on the historical grid. Recorded max absolute null delta is 0.
- Orthogonal control at alpha 1, validation `0.105415`, replication `0.107654`.
- Shuffled or permuted intervention: `NOT_AVAILABLE`. The M22.1 protocol does not define one, and none was added.

## 14. Interpretation

Historical reproduction is True. MRSM-metric signal present is True. QF-2 cancellation slots are ['L0H0', 'L0H2', 'L0H3']. The bridge target is the M22.1 primary direction at layer 23, which MRSM names as slot L1H2. That name is a slot, not a discovered mechanism. The orthogonal control remains larger than the primary contrast, which is why the historical status stays CANDIDATE.

## 15. Primary Diagnosis

INTERVENTION_CONTEXT_DEPENDENCE_CONFIRMED

Bridge result: `BRIDGE_SIGNAL_PRESENT_BUT_CONTEXT_DEPENDENT`.

The previous QF label `INTERVENTION_BOTTLENECK` is refined by this result. It is not deleted from `reports/Q_FAILURE_ANALYSIS.md`.

## 16. What This Does NOT Prove

This diagnostic does not discover a mechanism and does not name a discovered mechanism.
It does not change MRSM H1–H4.
Q.H3 is not used as causal evidence.
`CANDIDATE` is not promoted to `VALIDATED_FOR_M22`.
A context cancellation is not a failed mechanism.

## 17. Recommended Next Single Intervention

Do not modify the Self-Model or rerun MRSM. The next single step is a new preregistered decision to report each frozen regime separately, keeping every regime, before any architectural change.

No architectural change was made.

## 18. Reproducibility Manifest

- git_commit: `41110e98ed6e42e2442337a84bd5ce7f20aae085`
- git_branch: `cursor/qf-bridge-intervention-4e19`
- python: `3.12.3`
- torch: `2.14.0+cpu`
- transformer_lens: `3.9.0`
- numpy: `2.4.4`
- model_revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- device: `cpu`
- dtype: `float32`
- weight_sha256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`
- direction_sha256: `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`
- prompt_manifest_sha256: `fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db`
- tokenizer_token_ids: `yes_token_ids_only`
- started_at: `2026-09-29T22:19:03Z`
- forwards: `138`

Scientific status: `QF-BRIDGE-DIAGNOSTIC`.
