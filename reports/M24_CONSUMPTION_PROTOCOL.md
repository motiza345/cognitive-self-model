# M24 mechanism response consumption protocol

Status: `READY_FOR_EXECUTION`

This file freezes the consumption audit. It does not run Qwen, does not change `MechanismResponseModel`, does not change `SelfModelBelief`, and does not change the feasibility gate. No M24 outcome exists when this file is written.

Scientific question: does the frozen `MechanismResponseModel`, consumed before an intervention, produce useful held-out predictions of alpha-+1 effects on a new prompt catalog?

The object stays a mechanism response rule. This audit does not make it a self-model.

## Frozen mechanism

Family, and only this family:

- `M22.1-D1-L0` at `blocks.0.hook_resid_post`
- `M22.1-D1-L8` at `blocks.8.hook_resid_post`
- `M22.1-D1-L15` at `blocks.15.hook_resid_post`

Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, float32, `prepend_bos` true. Token ids must be `9834` and `902`. Direction D1 is `primary_direction(896, 22101)`, sha256 `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`. Alpha is `+1`. The response rule is `prediction = g`, and `g` is the return value of `MechanismResponseModel.predict`.

The stored training means on that object are the baseline. They are the feasibility-gate train means at commit `7aa64543fa45b0e09bb7e247641ce8c926f8a66d`. They are not recomputed on the `c-*` catalog.

Layer 23 is not in the family. D2 is not a mechanism. `orthogonal_direction(896, 22103, d1)` supplies `g_orth` from the same margin gradient. That column is recorded and is not an input to the verdict.

## Catalog

catalog sha256: `ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78`

Thirty-six prompts. Twelve in each of completion, syntax, and instruction. Partition is `(index - 1) mod 3`, the same three-way rule as M23-G. Each partition has twelve prompts and four from each surface family. The hash is the sha256 of the catalog JSON with sorted keys.

The train partition is frozen so the evaluation ids are a fixed third of this catalog. Execution does not forward train prompts and does not estimate a statistic from them. Validation is forwarded under the same prediction-before-outcome rule and is not an input to the verdict. Evaluation is the decision partition.

| prompt_id | family | partition | text |
| --- | --- | --- | --- |
| `c-completion-01` | completion | train | The capital of Norway is |
| `c-completion-02` | completion | validation | Sand becomes glass when it is |
| `c-completion-03` | completion | evaluation | The author of Don Quixote is |
| `c-completion-04` | completion | train | A violin has four |
| `c-completion-05` | completion | validation | The Amazon is a |
| `c-completion-06` | completion | evaluation | A cube has six |
| `c-completion-07` | completion | train | The currency of Sweden is the |
| `c-completion-08` | completion | validation | Cheese is often aged in a |
| `c-completion-09` | completion | evaluation | The smallest bird is the |
| `c-completion-10` | completion | train | A minute has sixty |
| `c-completion-11` | completion | validation | The chemical symbol for copper is |
| `c-completion-12` | completion | evaluation | Linen is woven from |
| `c-syntax-01` | syntax | train | In case the ferry is delayed, |
| `c-syntax-02` | syntax | validation | Supposing the lock freezes, |
| `c-syntax-03` | syntax | evaluation | Now that the harvest is in, |
| `c-syntax-04` | syntax | train | By the time the oven cooled, |
| `c-syntax-05` | syntax | validation | So that the wound would close, |
| `c-syntax-06` | syntax | evaluation | Wherever the path divides, |
| `c-syntax-07` | syntax | train | No matter how dark the cellar is, |
| `c-syntax-08` | syntax | validation | The moment the anchor dropped, |
| `c-syntax-09` | syntax | evaluation | Given that the sample was sterile, |
| `c-syntax-10` | syntax | train | Long after the echo faded, |
| `c-syntax-11` | syntax | validation | On condition that the seal holds, |
| `c-syntax-12` | syntax | evaluation | Rather than leave the kiln hot, |
| `c-instruction-01` | instruction | train | Reply with one word. A liquid used in thermometers: |
| `c-instruction-02` | instruction | validation | Reply with one word. A drink made from grapes: |
| `c-instruction-03` | instruction | evaluation | Name a tool used for sewing: |
| `c-instruction-04` | instruction | train | Name a constellation: |
| `c-instruction-05` | instruction | validation | Reply with one word. Opposite of shallow: |
| `c-instruction-06` | instruction | evaluation | Name a kind of nut: |
| `c-instruction-07` | instruction | train | Reply with one word. A soup served cold: |
| `c-instruction-08` | instruction | validation | Name a keyboard instrument: |
| `c-instruction-09` | instruction | evaluation | Reply with one word. The number of legs on an insect: |
| `c-instruction-10` | instruction | train | Name a desert: |
| `c-instruction-11` | instruction | validation | Reply with one word. A reptile with a shell: |
| `c-instruction-12` | instruction | evaluation | Name a type of bridge: |

## Prediction order

For each validation prompt and each evaluation prompt, and for each cell:

1. One unintervened forward. The margin is the last-position logit of token `9834` minus the logit of token `902`.
2. The gradient of that margin with respect to the last-token residual is retained at the cell hook.
3. `PreInterventionState(margin_gradient)` is passed to `MechanismResponseModel.predict`.
4. The returned scalar is stored as `g` and as `candidate_prediction`.
5. `baseline_prediction` is `training_baseline_mean` from that same object.
6. After every prediction in the partition exists, those predictions are shuffled within each cell and the shuffled values are stored in the same prediction file.
7. The prediction file is closed. It has no observed effect.
8. Only after both held-out prediction files are closed does any held-out intervention run.

Alpha-+1 is then applied with the existing last-token residual hook. The outcome is the intervened margin minus the unintervened margin. Outcome files are separate.

`g_orth` is `directional_derivative` of the same gradient with D2. It is stored in the prediction file and is not a second call to `MechanismResponseModel`.

## Primary comparison

For an episode, the paired improvement is `abs(effect - baseline) - abs(effect - g)`.

The primary scores are prompt-level. For each evaluation prompt, average the three cell improvements. `paired_mean_ci` then resamples those twelve scores with 5000 draws and seed `23001`. Prompt ids are sorted before the average list is built. The same function is used for the shuffled column and for the recorded `g_orth` column.

Per-cell intervals resample that cell's twelve prompt improvements directly. They are reported. They do not replace the pooled interval.

MAE is the mean absolute error over episodes.

Strict sign agreement uses the existing rule: a zero sign is excluded, and the remaining matches are counted. Clopper-Pearson intervals are reported. They are not a verdict input.

## Sign decision

The decision question is whether the intervention effect is positive. The threshold is `0`, fixed here. A value is positive only when it is strictly greater than `0`. A zero prediction decides that the effect is not positive. A zero effect is not a positive outcome.

Accuracy is the fraction of evaluation episodes decided correctly. The mechanism arm does not lose when its correct count is greater than or equal to the scalar baseline's correct count. Confusion counts are true positive, false positive, true negative, and false negative for the positive class. Clopper-Pearson on the correct count is reported and is not a second threshold.

## Shuffled control

Within each cell, `random.Random(24001)` permutes the mechanism predictions across the prompts of that partition. Cells are taken in family order, and prompts are sorted by id. Each partition starts its own generator at seed `24001`. The shuffled paired interval uses the same prompt-level rule. A shuffled class of `CI_POSITIVE` reproduces the improvement.

## Verdict

`scripts/m24_consumption_protocol.py` defines `consumption_verdict`. It is the only map.

`CONSUMPTION_SUPPORTED` only when all of these hold:

1. The evaluation prompt-level paired-improvement class is `CI_POSITIVE`.
2. That interval is computed on the evaluation partition alone.
3. The mechanism sign decision does not lose to the scalar baseline on evaluation episodes.
4. The evaluation shuffled class is not `CI_POSITIVE`.
5. `leakage_ok` is true.

`CONSUMPTION_NOT_SUPPORTED` only when the evaluation class is `CI_NEGATIVE` and `leakage_ok` is true.

Every other combination is `INCONCLUSIVE`, including a positive interval whose sign decision loses, a positive interval that the shuffle also achieves, a leakage failure, and an interval that includes zero. A negative interval with `leakage_ok` false is `INCONCLUSIVE`.

`g_orth`, validation, and strict sign agreement do not enter this map.

## Leakage

`leakage_ok` is true only when the prediction files are closed before the matching outcome files, those files contain no observed effect, the mechanism objects are unchanged, and every baseline value is the stored `training_baseline_mean`.

No slope is fit. No prompt embedding is built. `pre_dot` is not a feature. `SelfModelBelief` is not read or written.

## Execution

Command, from the repository root, after this protocol is frozen:

`python3 scripts/run_m24_consumption_audit.py`

The runner refuses to start when `reports/m24_raw/` already contains files or when `reports/M24_CONSUMPTION_RESULTS.json` already exists.
