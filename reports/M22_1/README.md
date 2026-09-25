# M22.1 causal intervention preflight

Status: `CANDIDATE`

This file records one preflight of an additive last-token residual intervention on `Qwen/Qwen2.5-0.5B`. It is not a self-model, not a training run, and not permission to build the M22.2 dataset unless the status is `VALIDATED_FOR_M22`.

## What was tested

- Hook family: `blocks.{layer}.hook_resid_post` at token position `last`.
- Outcome: `logit_margin` with token ids `[9834, 902]`.
- Candidate layers: `[0, 8, 15, 23]`.
- Frozen layer: `23`.
- Magnitude grid: `[-2.0, -1.0, 0.0, 1.0, 2.0]`.
- Validation directionality (not a gate): `+alpha 0.028637, -alpha -0.028185, sign-blind average 0.000226, class CLEAR_DIRECTIONAL`.
- Replication directionality (not a gate): `+alpha 0.029694, -alpha -0.029163, sign-blind average 0.000265, class CLEAR_DIRECTIONAL`.

## What passed or failed

Null equivalence pass: `True` (max abs difference `0.0`).
Hook integrity pass: `True`.
Primary-direction mean paired delta at alpha = +1: validation `0.028637091318766277`, replication `0.029693762461344402`.
Orthogonal-control mean paired delta at alpha = +1: validation `0.10541534423828125`, replication `0.10765441258748372`.
The frozen status rule requires the primary absolute mean to exceed the control absolute mean on both confirmatory splits. This run's status is `CANDIDATE`.
Dose response is descriptive and is not a pass/fail gate.

## Established

Only the contents of `certificate.json` for this commit, model revision, prompt manifest, and config hash. A status of `VALIDATED_FOR_M22` means that specific intervention produced a paired outcome contrast that survived the pre-declared validation, replication, null, and control checks.

## Not established

- No self-model was trained or shown to predict the outcome.
- The checkpoint was not shown to understand itself, to be self-aware, or to self-improve.
- The intervention is not a universal causal mechanism and is not a discovered circuit.
- Earlier synthetic temporal forecasts and the epistemic benchmark's invalidity score are not evidence for this result.
- Directionality and dose shape are reported separately from the status gate.

## Limitations

- GPU is not required. Scoring uses float32.
- Only the pre-declared layer subset was searched. Unsearched layers are unknown.
- One primary direction and one orthogonal control were used. This is not a circuit discovery.
- Discovery selects the layer. Validation and replication are the only confirmatory splits.
- Each split is small. Intervals are descriptive.
- Dose-response shape and the directionality label are not status gates.
- No predictor was trained.
- This run used device cpu.
- This artifact does not authorize an M22.2 dataset.

## Next permitted step

Do not build M22.2. The intervention is technically suggestive but is not frozen; see limitations.
