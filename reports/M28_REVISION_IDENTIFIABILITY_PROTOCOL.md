# M28 revision identifiability

Status: `NO_SUITABLE_PRE_EVIDENCE`

Protocol version: `M28.REVISION_IDENTIFIABILITY.1`

This file freezes the identifiability gate. It does not run a model, does not generate a catalog, does not collect outcomes, and does not modify M22.1, M23, M24, M25, M26, or M27. The scientific argument is `reports/M28_REVISION_IDENTIFIABILITY_RATIONALE.md`. The synthetic lock is `tests/test_m28_revision_identifiability_design.py`.

`PRIMARY_CANDIDATE: NONE`

`CATALOG_SHA256: NONE`

`UPDATE_N: NONE`

`VALIDATION_N: NONE`

`HOLDOUT_N: NONE`

## 1. Scientific question

Can an independently measured pre-evidence state predict the direction or magnitude of the future residual `observed_effect - g` strongly enough that a mechanism revision would be justified?

The question is about when the existing response `g` is systematically wrong. A model that predicts the raw intervention effect by replacing `g` is outside this protocol.

## 2. Execution gate

The eligible primary-candidate set defined in section 6 is empty. The status of this protocol is therefore `NO_SUITABLE_PRE_EVIDENCE`.

The scoring labels `M28_IDENTIFIABILITY_SUPPORTED`, `M28_IDENTIFIABILITY_NOT_SUPPORTED`, and `M28_INCONCLUSIVE` are defined in section 11 so that a later file cannot reuse this protocol under a softer rule. This file does not issue those labels. Catalog generation, coefficient fitting, shuffling of stored rows, and model execution are blocked.

## 3. Frozen intervention context

These values are the existing apparatus. This protocol does not retune them.

| item | frozen value |
| --- | --- |
| model | `Qwen/Qwen2.5-0.5B` |
| model revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| cells | `M22.1-D1-L0`, `M22.1-D1-L8`, `M22.1-D1-L15` |
| layers | `0`, `8`, `15` |
| hooks | `blocks.0.hook_resid_post`, `blocks.8.hook_resid_post`, `blocks.15.hook_resid_post` |
| direction | D1, seed `22101`, dimension `896` |
| direction sha256 | `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` |
| alpha | `+1` |
| margin tokens | `9834` (`" yes"`) minus `902` (`" no"`), last position |
| response | `prediction = g`, with `g = dot(margin_gradient, d1)` |
| sigma | the published M25 scales in `FROZEN_SIGMA`. This protocol does not re-estimate them |

The intervention effect, when it is observed under a later protocol, remains `intervened_margin - baseline_margin` at alpha `+1` on the cell hook, last token only.

## 4. Target

`residual = observed_effect - g`

`g` is the sealed pre-outcome directional derivative. `observed_effect` is the intervention outcome. The target exists only after the outcome. It is the quantity a candidate would have to predict. It is forbidden as a candidate feature.

## 5. Partitions

A legal execution would use three disjoint partitions.

| partition | role |
| --- | --- |
| `update` | The only rows that may estimate a constant residual, a cell-mean residual, or a mapping from the named candidate |
| `validation` | A sealed sign check of that already-named mapping. Validation cannot add a candidate and cannot replace the primary candidate |
| `holdout` | Scored once, after candidate predictions, the constant, the cell means, and the shuffled mapping are closed |

Sizes are `UPDATE_N: NONE`, `VALIDATION_N: NONE`, and `HOLDOUT_N: NONE`. No prompt text is assigned. Overlap with the M22.1, M23, M23-G, single-measurement, feasibility-gate, M24, M26, or M27 catalogs is undefined because this file contains no prompt text.

## 6. Candidate inventory

A field is eligible only when it is a pre-evidence measurement, sealed in a prediction artifact before the intervention outcome, absent from the M26 holdout analyses and from the M27 gain fit, and distinct from the cell-mean baseline. The inventory below was fixed from artifact schemas and from the published M26 and M27 results. No coefficient was estimated on the M27 holdout to build it.

| name | storage | eligibility |
| --- | --- | --- |
| `g` | sealed prediction file | `MECHANISM_OUTPUT` |
| `prediction_before` | sealed prediction file | `IDENTICAL_TO_MECHANISM_OUTPUT` |
| `prediction` | sealed M27 prediction file | `IDENTICAL_TO_MECHANISM_OUTPUT` |
| `family` | sealed prediction file | `ALREADY_SCORED_ON_M26_HOLDOUT` |
| `baseline_prediction` | sealed prediction file | `CELL_CONSTANT` |
| `layer` | sealed prediction file | `CELL_MEAN_BASELINE` |
| `hook` | sealed prediction file | `CELL_MEAN_BASELINE` |
| `intervention_id` | sealed prediction file | `CELL_MEAN_BASELINE` |
| `cell` | sealed M27 prediction file | `CELL_MEAN_BASELINE` |
| `mechanism_id` | sealed M27 prediction file | `CELL_MEAN_BASELINE` |
| `alpha` | sealed prediction file | `CONSTANT` |
| `direction_id` | sealed prediction file | `CONSTANT` |
| `response_rule` | sealed prediction file | `CONSTANT` |
| `response_form` | sealed M27 prediction file | `CONSTANT` |
| `initial_gain` | sealed M27 prediction file | `CONSTANT` |
| `mechanism_version` | sealed prediction file | `CONSTANT` |
| `outcome_present` | sealed prediction file | `CONSTANT` |
| `partition` | sealed prediction file | `DESIGN_LABEL` |
| `prompt_id` | sealed prediction file | `IDENTIFIER` |
| `timestamp` | sealed prediction file | `CLOSURE_METADATA` |
| `baseline_output` | outcome file | `ALREADY_SCORED_ON_M26_HOLDOUT` |
| `observed_effect` | outcome file | `FORBIDDEN_OUTCOME` |
| `intervened_output` | outcome file | `FORBIDDEN_OUTCOME` |
| `residual` | episode file | `FORBIDDEN_OUTCOME` |
| `residual_initial` | M27 episode file | `FORBIDDEN_OUTCOME` |
| `revised_gain` | post-update artifact | `FORBIDDEN_POST_EVIDENCE` |
| `delta_cell` | post-update artifact | `FORBIDDEN_POST_EVIDENCE` |
| `pre_dot` | absent from the episode files | `ABSENT_FROM_SEALED_SCHEMA` |
| `margin_gradient` | in memory during prediction; absent from the artifacts | `ABSENT_FROM_SEALED_SCHEMA` |
| `prompt_length` | absent from the sealed schema | `ABSENT_FROM_SEALED_SCHEMA` |
| `gradient_norm` | absent from the sealed schema | `ABSENT_FROM_SEALED_SCHEMA` |
| `g_orth` | absent from the sealed schema | `ABSENT_FROM_SEALED_SCHEMA` |

`PRIMARY_CANDIDATE: NONE`

A name that is absent from this table has eligibility `INVENTED_MEASUREMENT`. Adding a row after this freeze is a new protocol.

`g` is ineligible because M27 fit the gain on update rows and every 95 percent interval contained `1`:

| cell | beta | interval | consumed gain |
| --- | ---: | --- | ---: |
| `M22.1-D1-L0` | 0.9968874018382687 | [0.825832602879822, 1.1679422007967155] | 1 |
| `M22.1-D1-L8` | 1.016473791906488 | [0.9306088136122461, 1.1023387702007297] | 1 |
| `M22.1-D1-L15` | 1.0092305686364562 | [0.9802505636398144, 1.038210573633098] | 1 |

`family` and `baseline_output` are ineligible because the M26 gap audit, classification `STRUCTURED_BUT_UNSTABLE`, already reported their residual associations on the M26 holdout. Cell labels are the cell-mean baseline.

## 7. Comparisons and metric

The definitions in this section are frozen and unapplied. The gate in section 2 returns before any of them reads a row.

Episode residual: `residual = observed_effect - g`.

Reference predictions of that residual, estimated on update rows only and frozen before validation outcomes and before holdout outcomes:

1. Constant residual baseline: one arithmetic mean of the update residuals. The same number is used on every cell.
2. Cell-mean baseline: the arithmetic mean of the update residuals inside the row's cell.
3. Shuffled control: sort update rows by `mechanism_id` then `prompt_id`, permute the candidate values with `random.Random(28001)`, and refit the same mapping. Holdout residuals and holdout `g` stay in place. Seed `28001` is reserved. It is unused while the candidate set is empty.

The candidate mapping from the named pre-evidence field to a residual prediction is unspecified. This protocol authorizes no coefficient, no intercept, and no link function. `SHUFFLE_SEED: 28001`

Episode contributions, positive when the candidate is closer to the residual than the reference:

`d_constant = abs(residual - constant) - abs(residual - candidate)`

`d_cell = abs(residual - cell_mean) - abs(residual - candidate)`

`d_shuffle = abs(residual - constant) - abs(residual - shuffled_candidate)`

Within each holdout prompt, average the three cell contributions. The prompt ids are sorted. The interval is the existing `paired_mean_ci` on those prompt scores: 5000 draws, seed `23001`. The class is `CI_POSITIVE` when the lower endpoint is above `0`, `CI_NEGATIVE` when the upper endpoint is below `0`, and `CI_INCLUDES_ZERO` otherwise. Equality at `0` includes zero.

`BOOTSTRAP_DRAWS: 5000`

`BOOTSTRAP_SEED: 23001`

## 8. Leakage rules

`leakage_ok` would require every item below. The gate does not evaluate `leakage_ok`, because no execution file is created.

1. The primary candidate name appears in this protocol before any M28 outcome file exists.
2. Update predictions of the constant, the cell means, and the candidate close before update outcomes are used to form the target.
3. Validation predictions close before validation outcomes.
4. Holdout predictions close before holdout outcomes.
5. Validation and holdout rows are rejected by any fit.
6. M26 outcome files, M27 outcome files, the M26 gap-audit tables, and the M27 holdout are not candidate features and are not fit inputs.
7. No prompt id appears in more than one partition.
8. The response rule remains `prediction = g`. The gain remains the M27 consumed gain. This protocol does not write a new gain.
9. No threshold in this file is edited after a holdout score.

Prediction artifacts that would be legal contain the named pre-evidence field and the issued residual prediction. They contain no `observed_effect`, no residual, and no revised gain.

## 9. What would justify a revision experiment

A revision experiment becomes eligible only after a later protocol issues `M28_IDENTIFIABILITY_SUPPORTED` under section 11. That result would say that one pre-declared pre-evidence measurement, sealed before the outcome, reduced holdout residual absolute error relative to the constant and relative to the cell mean, and that the shuffled update pairing did not account for the reduction.

This file does not define the revision operator that would consume such a measurement. `MechanismResponseModel` stays `prediction = g`. `MechanismBelief` stays a status-and-uncertainty wrapper. The M27 gain operator stays the gain operator.

## 10. What kills the hypothesis

`NO_SUITABLE_PRE_EVIDENCE` kills the claim that the measurements already stored before the intervention outcome identify when `g` is wrong. That is the result of this design. The stored sealed fields are the mechanism output, constants, the cell-mean baseline, or variables whose holdout associations were already published.

`M28_IDENTIFIABILITY_NOT_SUPPORTED` would kill the claim for a named candidate whose constant-baseline interval is strictly negative with `leakage_ok`. This file does not issue that label.

## 11. Decision labels

These labels apply only when section 6 contains an eligible primary candidate and a later protocol has generated its own catalog. They are inactive here.

| label | rule |
| --- | --- |
| `M28_IDENTIFIABILITY_SUPPORTED` | Eligible candidate named in advance, `leakage_ok`, `d_constant` class `CI_POSITIVE`, `d_cell` class `CI_POSITIVE`, and `d_shuffle` class other than `CI_POSITIVE` |
| `M28_IDENTIFIABILITY_NOT_SUPPORTED` | Eligible candidate named in advance, `leakage_ok`, and `d_constant` class `CI_NEGATIVE` |
| `M28_INCONCLUSIVE` | An eligible candidate was scored and neither supported rule nor the not-supported rule holds |
| `NO_SUITABLE_PRE_EVIDENCE` | The eligible set is empty. This protocol issues this status and stops |

## 12. File list

Created by this freeze:

| path | role |
| --- | --- |
| `reports/M28_REVISION_IDENTIFIABILITY_RATIONALE.md` | Scientific argument |
| `reports/M28_REVISION_IDENTIFIABILITY_PROTOCOL.md` | This freeze |
| `src/cognitive_self_model/m28_identifiability.py` | Frozen inventory and refusal |
| `tests/test_m28_revision_identifiability_design.py` | Synthetic lock |
| `scripts/run_m28_revision_identifiability_gate.py` | Gate command. It prints the status and does not load the model |

Absent on purpose:

| path | reason |
| --- | --- |
| catalog module | `CATALOG_SHA256: NONE` |
| `reports/m28_raw/` | no execution |
| M28 episodes, results, and execution report | no execution |
| edits under `src/cognitive_self_model/mechanism_response.py` | response rule stays `prediction = g` |
| edits under `src/cognitive_self_model/mechanism_belief.py` | belief object stays unchanged |
| edits under `src/cognitive_self_model/residual_update.py` | M26 operator stays unchanged |
| edits to M22.1, M23, M24, M25, M26, or M27 artifacts | those results stay as recorded |

## 13. Execution command

Design lock:

`python3 -m pytest tests/test_m28_revision_identifiability_design.py -q`

Gate:

`python3 scripts/run_m28_revision_identifiability_gate.py`

The gate command prints `NO_SUITABLE_PRE_EVIDENCE` and exits `0`. Exit `0` means the refusal completed. A Qwen run, a read of an M26 or M27 outcome file for fitting, or a generated catalog is a violation of this protocol.

## 14. Provenance of the inspection

Git commit at inspection: `1e183467d711ec5bc40291349a9ddbe4a5c1175d`

| file | sha256 |
| --- | --- |
| `reports/M26_GAP_AUDIT.md` | `9f28c378a40b95875b2bbaa785ff5456e7cd5c338117955a0321b8c5526fef8b` |
| `reports/M26_GAP_AUDIT_RESULTS.json` | `7be009550352a5fece0a43ecb6896354f4517d5f63455f12e02b36b27188b4c9` |
| `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_EPISODES.csv` | `4c216f50f45edff5cdcc91cdaeec62b566dd287a5e5386779d054c95004e8527` |
| `reports/M27_MECHANISM_REVISION_PROTOCOL.md` | `5254a9253410cc53ba028c95c798e7eb4f50c03b76d77aa7cc9ea2073699a57f` |
| `reports/M27_MECHANISM_REVISION_EXECUTION.md` | `e2dce2ec43e6bba9b0b5b881dd0d82edcdcbcec256271e4e779bcac0a10f5729` |
| `reports/M27_MECHANISM_REVISION_RESULTS.json` | `3cb2c577a53c8055e34d17ceda8d65b718d6545451407e51c4a42e069fe475f8` |
| `reports/M27_MECHANISM_REVISION_EPISODES.csv` | `a3104d06b672aa45724efbb1d848e0fdcb01f96877fc5b612c290addb50e1ad8` |
| `src/cognitive_self_model/mechanism_response.py` | `daa0eafb06e210c6f83d797a7f36592659eb84a997b694646dc6e36ba0443d44` |
| `src/cognitive_self_model/mechanism_belief.py` | `78eb3607cda239466f407f4a7f6bea872c238eef1f9c07cf50217d5e503ee1ab` |
| `src/cognitive_self_model/residual_update.py` | `ad8c38bd97d3eef00d9cb0138d3ef1d01da4cdd643f01169a588cddc02097a11` |

The inspection recorded schemas and these published results. It did not fit a predictor on the M27 holdout.
