# M23-G execution report

Protocol status remains `READY_FOR_EXECUTION`. This run does not change that status. The scientific verdict below is about the tested belief-update loop. It is not a claim that a self-model was validated.

Historical M23 remains `INCONCLUSIVE`. M23-F remains `NO_SUITABLE_EXISTING_INTERVENTION`. D1 at layer 23 was not run.

## Verdict

`INCONCLUSIVE`

The primary paired difference, no-update absolute error minus updated absolute error, has mean `0.0026111067445189844`. The interval is `[-0.0006063265932930853, 0.00589502520031399]`, class `CI_INCLUDES_ZERO`. 5000 draws, seed `23001`, 72 evaluation rows.

The point estimate favors the update. The frozen rule does not call that an improvement, because the interval includes zero.

The shuffled-control interval against no-update also includes zero, so the `BASELINE_MATCH` condition is not met.

## Protocol

Preregistration sha256 `9130a88aad20292cdc2bf1d89d7f4bdb2d05eb48900d6671235bdd853d331b6c`.

Cells, alpha `+1` only:

- `M22.1-D1-L0`
- `M22.1-D2-L0`
- `M22.1-D1-L8`
- `M22.1-D2-L8`
- `M22.1-D1-L15`
- `M22.1-D2-L15`

New catalog only. Twelve prompts in each of train, validation, and evaluation. Four prompts from each surface family in each partition. No old replication id was evaluated.

`scripts/validate_m23_g_protocol.py` returned `PROTOCOL_CHECK_PASS` and `HARNESS_PASS` before the forwards.

## Order

| artifact | utc |
| --- | --- |
| validation predictions | 2026-10-01T16:14:54Z |
| validation outcomes | 2026-10-01T16:15:59Z |
| evaluation predictions | 2026-10-01T16:15:59Z |
| evaluation outcomes | 2026-10-01T16:17:04Z |

File modification order matches that sequence. The evaluation prediction file contains no `observed_effect`. The shuffled prediction values are inside that locked file. The shuffled history JSON was flushed after the evaluation outcomes file; it was computed from validation evidence only.

## Beliefs

Each cell started at version 1 from its twelve train effects and ended at version 13 after twelve validation updates. The original belief objects were not mutated. Inflation stayed `1`. No update was `CONFIDENT_CONTRADICTION`.

| cell | train prediction | final prediction | final uncertainty | status |
| --- | ---: | ---: | ---: | --- |
| D1 layer 0 | 0.041385809580485024 | 0.032807489236195884 | 0.12329138613627405 | `CONTRADICTED` |
| D2 layer 0 | 0.05268343289693197 | 0.03445770343144735 | 0.15526520634520005 | `CONTRADICTED` |
| D1 layer 8 | -0.008025685946146647 | 0.016083439191182453 | 0.076174008265705 | `CONTRADICTED` |
| D2 layer 8 | 0.0035112698872884116 | 0.011889994144439697 | 0.07262252241532736 | `CONTRADICTED` |
| D1 layer 15 | -0.026706496874491375 | -0.021458208560943604 | 0.04978595268101367 | `CONTRADICTED` |
| D2 layer 15 | 0.01736768086751302 | 0.031788408756256104 | 0.05938552807088488 | `CONTRADICTED` |

Across 72 validation updates: 26 sign mismatches, 4 outside the inherited uncertainty interval, 29 with reason `EVIDENCE_OUTSIDE_OR_SIGN`, 43 with reason `PRECISION_WEIGHTED_MEAN`, 0 critical. Contradictions occurred. None met the existing confidence condition, so inflation did not change.

## Held-out errors

| method | mean absolute error | strict-sign agreements |
| --- | ---: | ---: |
| no-update | 0.07394385779345476 | 31 / 72 |
| updated | 0.07133275104893579 | 33 / 72 |
| constant-effect | 0.06729273406075842 | 31 / 72 |
| outcome-only | 0.06729273406075842 | 31 / 72 |
| shuffled | 0.07232269203221357 | 31 / 72 |

The constant-effect value and the outcome-only value are the same train-only pooled mean, `0.013369335068596734`. Their errors match, as specified. That pooled mean has a lower point absolute error than the updated beliefs. The frozen primary comparison is updated versus no-update, not versus this constant. It does not create a second verdict.

Shuffled versus no-update: mean `0.001621165761241206`, interval `[-0.0029924291151541237, 0.006402437609654887]`, class `CI_INCLUDES_ZERO`.

No zero signs appeared. Sign counts are descriptive.

## Per cell

These intervals are not the verdict. They do not add or drop a cell.

| cell | updated MAE | no-update MAE | paired mean | interval class | updated sign agreements |
| --- | ---: | ---: | ---: | --- | ---: |
| D1 layer 0 | 0.09841338462299772 | 0.10584024588267009 | 0.007426861259672368 | `CI_POSITIVE` | 3 / 12 |
| D2 layer 0 | 0.08360395166609023 | 0.09575443797641331 | 0.01215048631032308 | `CI_POSITIVE` | 4 / 12 |
| D1 layer 8 | 0.046210289001464844 | 0.048214607768588595 | 0.002004318767123752 | `CI_INCLUDES_ZERO` | 7 / 12 |
| D2 layer 8 | 0.0661361813545227 | 0.06334327326880561 | -0.0027929080857170955 | `CI_INCLUDES_ZERO` | 4 / 12 |
| D1 layer 15 | 0.07154853145281474 | 0.07082986831665039 | -0.0007186631361643485 | `CI_INCLUDES_ZERO` | 7 / 12 |
| D2 layer 15 | 0.06208416819572449 | 0.059680713547600635 | -0.0024034546481238495 | `CI_INCLUDES_ZERO` | 8 / 12 |

## Per family

Twenty-four evaluation rows in each family. All three secondary intervals include zero.

| family | updated MAE | no-update MAE | paired mean | interval class |
| --- | ---: | ---: | ---: | --- |
| completion | 0.07407835953765445 | 0.07343492574161954 | -0.0006434337960349195 | `CI_INCLUDES_ZERO` |
| instruction | 0.06803059412373437 | 0.0727674663066864 | 0.00473687218295203 | `CI_INCLUDES_ZERO` |
| syntax | 0.07188929948541853 | 0.07562918133205838 | 0.0037398818466398425 | `CI_INCLUDES_ZERO` |

## Leakage

`leakage_ok = true`.

Validation predictions precede validation outcomes. Evaluation predictions precede evaluation outcomes. The evaluation prediction file has no outcome field. Train and validation prompts are disjoint from evaluation. Alpha stayed `+1`. All six cells were retained. The constant uses the 72 train effects only. The shuffled control is the within-layer D1/D2 swap. `predict` was called with alpha only.

## Limitations

The pooled interval includes zero, so the positive point estimate is not a shown gain. Twelve prompts per cell remain a small sample. The constant train mean beat the cell beliefs on point absolute error; that fact is outside the primary gate. Two layer-0 secondary intervals lie above zero and are not a license to keep only those cells. Sign agreement for the updated belief is 33 of 72. No contradiction was confident under the inherited rule. This result does not validate a self-model.
