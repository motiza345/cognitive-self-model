# M23 leakage audit

This file states the rules. The runner writes `reports/m23_raw/leakage_execution.json` during the scored run. The scientific report quotes that file. A rule is not added after the run.

## Forbidden at prediction time

When a prediction is produced, the following are forbidden inputs:

- that case's intervention outcome
- any later split's outcome
- future residual or post-intervention activation
- the final verdict
- a ground-truth mechanism label
- a post-hoc class such as "circuit" or "readout"
- the M22.1 published validation or replication means

Discovery outcomes may form the initial belief. They are not available as per-prompt features at replication. Validation outcomes may update the belief only after the validation prediction file has been written. Replication outcomes may be read only after the replication prediction file has been written.

## What `predict` is allowed to read

- `predicted_effect` stored on the belief
- `uncertainty` stored on the belief
- the alpha of the intervention being predicted
- the anchor alpha, fixed at `+1`

`predict` has no parameter for activations, baseline margin, text, prompt id, or observed delta.

The pre-intervention residual dot product is logged on the forward record. It is not passed into `predict`. Using it would be a different, prompt-specific readout model and is out of scope for this audit.

## Ordering the runner must enforce

1. Write validation predictions. The validation outcome file must not exist yet.
2. Run validation interventions and write outcomes.
3. Update beliefs.
4. Write replication predictions. The replication outcome file must not exist yet.
5. Run replication interventions and write outcomes.
6. Score.

The prediction records include `sequence_index`, `belief_version`, and `evidence_version`. Evidence version at validation prediction time is the discovery-only version. Evidence version at replication prediction time is the post-update version.

Control arms are predictions from beliefs that were updated, or deliberately not updated, before step 4. Their predictions are frozen in the same replication file, before replication outcomes exist.

## Inherited exclusions

- Planted MRSM ground truth is not loaded. This audit has no planted labels.
- M20.6.3.2 layer 13 is not read.
- M21 `q_invalid` scores are not read.
- Token ids are not searched. The first ids of the frozen strings are used, then checked against `[9834, 902]`.

## Execution check, pre-registered

`leakage_execution.json` must contain:

- `validation_predictions_before_outcomes: true`
- `replication_predictions_before_outcomes: true`
- `predict_rejects_outcome_argument: true`
- `published_m22_means_not_loaded: true`

Any false value sets the leakage question to fail and the overall verdict cannot be `PASS`. The failure class is `LEAKAGE`.
