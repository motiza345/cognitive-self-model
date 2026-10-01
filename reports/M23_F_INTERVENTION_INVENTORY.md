# M23-F intervention inventory

Inventory date: 2026-10-01. Sources are the working tree and existing commits. Nothing here is promoted to `VERIFIED`. No new intervention is defined.

Historical M23 remains `INCONCLUSIVE`. M23-D2 and M23-E are unchanged.

## What counts as an intervention

An intervention, here, is an implemented change to a model activation during a forward, with a recorded target, direction, and alpha. Analyses that only reread those outputs are listed so they are not mistaken for new mechanisms.

## Qwen residual candidates

All of these use `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, hook family `blocks.{layer}.hook_resid_post`, last token, and the yes/no logit margin (token ids `9834` and `902`).

Directions:

| id | construction | seed |
| --- | --- | --- |
| D1 | unit Gaussian | `22101` |
| D2 | seed `22103`, then Gram-Schmidt against D1 | `22103` |
| D3–D8 | seeds `22111`–`22116`, each Gram-Schmidt against every earlier panel direction | `22111`–`22116` |

D1 hash `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`. D2 hash `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e`.

### M22.1-D1-L23

- Target: `blocks.23.hook_resid_post`, direction D1.
- Alpha: the preflight grid was `-2, -1, 0, +1, +2`. M23 used `+1` and, on replication then on all 18 prompts, `+2`.
- Purpose: primary candidate for a behavioral contrast.
- Prior evidence: discovery mean about `0.0294` (std about `0.0030`, sign consistency 1). Validation mean `0.028637`. Replication mean `0.029694`. M23-E, all 18 prompts: alpha `+1` mean `0.028635`, range `0.009222`; alpha `+2` / alpha `+1` mean ratio `2.016879`. Repeats were identical.
- Status: `CANDIDATE`. The preflight rule required the primary absolute mean to exceed the orthogonal control. It did not. Not validated for M22.2. Not a verified mechanism.
- Outcomes inspected: yes, on discovery, validation, and replication.
- Clean future holdout on this catalog: no.
- Limitation: one shared sign and a small prompt remainder. M23-E found no held-out gain for `pre_dot` or baseline margin over a constant.

### M22.1-D2-L23

- Target: layer 23, direction D2.
- Alpha: same grid in M22.1 and M22.1.1. M23 validation used alpha `+1` only, as a shuffled-evidence control.
- Purpose: orthogonal control, not a second primary.
- Prior evidence: validation mean `0.105415` (std about `0.0095`). Replication mean `0.107654`. Sign consistency 1. Dose at alpha `+2` is about twice alpha `+1`.
- Status: control inside a `CANDIDATE` preflight. Not promoted. The larger mean was already known when M23 kept D1 as primary. Promoting D2 now would select on that inspected contrast.
- Outcomes inspected: yes, all three splits.
- Clean future holdout: no.
- Limitation: internally sign-stable. It differs from D1 in magnitude. That difference is already known.

### M22.1-D3-L23 through M22.1-D8-L23

- Target: layer 23 only. The panel was not run at other layers.
- Alpha: `-2, -1, 0, +1, +2` in M22.1.1.
- Purpose: pre-registered specificity panel. Classification was descriptive and was `INSUFFICIENT_EVIDENCE` on every split.
- Prior evidence, alpha `+1` means (discovery / validation / replication), all with sign consistency 1:

| id | discovery | validation | replication |
| --- | ---: | ---: | ---: |
| D3 | 0.037791 | 0.037810 | 0.038308 |
| D4 | -0.028320 | -0.027256 | -0.028091 |
| D5 | 0.009124 | 0.008388 | 0.008498 |
| D6 | 0.036134 | 0.035595 | 0.036017 |
| D7 | 0.016580 | 0.016218 | 0.016431 |
| D8 | -0.014773 | -0.014931 | -0.015099 |

Within each direction the standard deviation is a few thousandths. Across directions the means differ, including sign.

- Status: `CANDIDATE` family. M22.1.1 explicitly says not a mechanism identity, not a circuit, not a self-model. Not upgraded here.
- Outcomes inspected: yes, all three splits.
- Clean future holdout: no.
- Limitation: each direction is another shared-scale response. The distinguishable part is which direction was added, and that result has already been seen.

### M22.1-D1-L0, M22.1-D1-L8, M22.1-D1-L15

- Target: the same D1 vector at layers `0`, `8`, and `15`. These were the other preflight candidate layers. Layer 23 was selected on the discovery mean.
- Alpha in the preflight selection: `+1` only.
- Purpose: layer screen. Not frozen as the intervention.
- Prior evidence, discovery, alpha `+1`, six prompts:

| layer | mean | std | min | max | signs |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.007306 | 0.106685 | -0.117492 | 0.147843 | both |
| 8 | 0.004291 | 0.066792 | -0.062363 | 0.130961 | both |
| 15 | -0.010039 | 0.061312 | -0.073203 | 0.062324 | both |
| 23 | 0.029388 | 0.003009 | 0.025981 | 0.033731 | positive only |

- Status: not selected. Still `CANDIDATE` material, not verified. The selection rule preferred the larger stable discovery mean. Their sign changes were visible before that choice. Choosing one of them now because the range is large would be selection on an inspected outcome.
- Outcomes inspected: discovery in the M22.1 preflight. MRSM Q also applied D1 at these layers on discovery and on validation. Replication was not the Q holdout.
- Clean future holdout: no. Validation was already consumed. Replication was not reserved before those inspections.
- Limitation: n = 6 on the published discovery rows. No alpha grid was published for these layers.

### M22.1-D2-L0, M22.1-D2-L8, M22.1-D2-L15

- Target: D2 at the same three layers. Ids in the MRSM Q catalog: `L0H1`, `L0H3`, `L1H1`. The layer-23 orthogonal alias is `L1H3`.
- Alpha in that catalog: `+1`.
- Purpose: Q-arm interventions, not a new direction family.
- Prior evidence: the Q arm failed its declared tests. Terminal decision on that branch was `REDEFINE_SCALE`. This inventory does not copy Q's per-prompt scores into a ranking.
- Status: not verified.
- Outcomes inspected: discovery and validation, by MRSM Q. Replication was not that holdout.
- Clean future holdout: no, for the same reason as D1 at these layers.
- Limitation: Q already spent the validation split.

### Null alpha

Alpha `0` at layer 23 matched the unhooked margin exactly (max absolute difference `0`). It is a negative control, not a candidate mechanism.

## Not separate Qwen interventions

| record | what it is | screening |
| --- | --- | --- |
| M22.1.2 | ridge reanalysis of the D1–D8 matrices. No new forward. Label `INSUFFICIENT_EVIDENCE`. | Not rerun. |
| M22.1.3 | readout reconstruction of the same residuals. Label `READOUT_RECONSTRUCTION_EXACT`, linearity `FIRST_ORDER_SUPPORTED`. M22.1 stayed `CANDIDATE`. | Not rerun. |
| M21 `q_invalid` | calibration score on a different estimand. | Not an activation intervention. |
| M20.6.3.2 | notebook predictors. Overall status `FAIL`. Layer 13 was not the M22.1 intervention. | Not rerun. |
| MRSM P | planted benchmark, eight head ids, different data. P passed its gates. | Not this Qwen catalog. Not rerun. |
| MRSM self-model | lookup of a stored effect for a head id. Q failed transfer. | Not reused. |

## Suitability before the new screen

D1 at layer 23 is the current intervention. Its published pattern is a shared, nearly linear function of alpha. That pattern does not give a pre-intervention belief a prompt-level contradiction.

Other implemented cells differ from it. Directions differ in mean and sometimes in sign. Layers 0, 8, and 15 change sign across discovery prompts. Those facts are already inspected. None of them leaves both confirmatory splits unmeasured. This inventory does not pick one of them as the next benchmark.
