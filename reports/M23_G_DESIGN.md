# M23-G design

Protocol status: `READY_FOR_EXECUTION`

That status means the protocol, the partition, and the synthetic harness check are specified. It is not evidence for or against a self-model. No Qwen forward was run for this design.

## Why this family

M23-F found no existing cell that was both structurally different from the consumed layer-23 primary and still clean on the old catalog. The current cell stays out. The replacement is not the layer or direction with the largest previously seen range.

The preflight layers were `0`, `8`, `15`, and `23`. Layer 23 was adopted and then exhausted. The directions already implemented at the other three layers are D1 and D2. The benchmark precommits to all six of those cells. `M22.1-D2-L15` stays in the family even though its discovery screen did not change sign. Dropping it would be a choice based on an observed pattern.

D3–D8 are not added at new layers. Those cells do not exist.

## What has to be true for the loop

Each cell has its own belief, built only from that cell's twelve train effects at alpha `+1`. Validation predictions are written, then the interventions run, then the existing update consumes the evidence. Evaluation predictions, including the four controls, are written before any evaluation intervention.

The existing contradiction rule is the failure opportunity: an observation with the opposite strict sign from the stored mean, or an observation outside the inherited `1.96` uncertainty interval. No new numeric cutoff is added. The protocol does not promise that a new prompt will produce a sign change. If none does, the result stays `INCONCLUSIVE`.

The update stores the cumulative mean. A future prediction cannot be the last observation unless every stored observation equals that observation, which the existing function rejects when the past observations differ.

## Partition

Thirty-six new prompts, twelve in train, twelve in validation, and twelve in evaluation, with all three surface families in each partition. The old replication set is not the evaluation set. The old prompts are absent.

Twelve per partition is the smallest three-way split that puts four prompts from each family in each partition and doubles the M23 count of six. It is still a small sample. The design does not collapse to two partitions, because the update and the held-out prediction have to be different prompts. It also does not grow the catalog after an interval is seen.

## Controls and decision

The primary comparison is updated absolute error against the no-update belief, one paired interval over all 72 evaluation rows, using the existing `paired_mean_ci`. Per-cell intervals are reported and cannot elect a survivor.

The constant-effect baseline and the outcome-only baseline are the same pooled train mean, with cell identity removed. They are listed as two controls so a shared number is not mistaken for two independent successes. The shuffled-evidence control swaps D1 and D2 within a layer. If that swap also beats persistence, the primary label becomes `BASELINE_MATCH`.

## What execution must not do

Execution is not part of this task. When it happens, it must not load a different family, must not score alpha `+2`, must not pass `pre_dot` or the family label into `predict`, and must not edit `SelfModelBelief`.

## Unresolved risks

New prompts may not change sign. The early-layer screen on the old discovery prompts did not bit-match the original preflight table, so those old numbers are not a forecast for this catalog. Twelve evaluation prompts per cell can leave the pooled interval on zero even if a small shift is real. The inherited zero-uncertainty case marks any departure as outside the interval. Surface family is a balance stratum, not a demonstrated cause of the effect.
