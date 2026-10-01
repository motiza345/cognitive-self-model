# M23 diagnostic report

The historical verdict stays `INCONCLUSIVE`. This note explains why the predict → contradict → update → held-out-improvement loop did not close. It does not rescore M23 and it does not add a pass threshold.

Intervals below, when shown, are the M23 paired percentile bootstrap already fixed in code: 5000 draws, seed `23001`, six replication prompts. A new cutoff was not chosen after seeing these numbers. Because this diagnostic looked at several representations, an interval that excludes zero is reported as a description. It is not promoted to `SUPPORTED` by itself. The instruction for this task is that a small difference is not a winner, and that a missing threshold stays `INCONCLUSIVE`.

## 1. Historical M23 result

Overall: `INCONCLUSIVE`.

Q1 `PASS`, Q2 `INCONCLUSIVE`, Q3 `PASS`, Q4 `INCONCLUSIVE`, Q5 `PASS`, Q6 `INCONCLUSIVE`.

The issued belief was the discovery mean `0.028899987538655598` for every validation prompt. The update moved that scalar to `0.02854486306508382`. On replication the absolute-error difference between those two constants had mean `0` and an interval that included zero. No validation case was a critical contradiction. Details are in `reports/M23_SCIENTIFIC_REPORT.md`.

## 2. Intervention-design analysis

Saved D1 effects, alpha `+1`:

| split | n | mean | sample sd | CV | range | sign |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| validation | 6 | 0.028190 | 0.003117 | 0.111 | 0.007877 | 6 positive |
| replication | 6 | 0.028816 | 0.003423 | 0.119 | 0.008433 | 6 positive |

The M23 falsification half-width, `1.96 *` the discovery sample sd, is `0.007041`. The validation range sits on that scale. Every validation error fell inside the band. That is the historical Q2 result, not a new test.

Regime share of the sum of squares, validation D1: `0.129`. The three validation regime means are `0.02896`, `0.02886`, and `0.02675`. Replication regime share is `0.980`, with means `0.02883`, `0.03260`, and `0.02502`. With two prompts per regime, that replication share is a description of six points, not a stable variance component.

Alpha `+2` on replication has mean `0.058094`, sd `0.006446`, CV `0.111`. The ratio of the alpha-`+2` mean to the alpha-`+1` replication mean is `2.016`.

Between-seed variance of the same intervention is not estimable. M23 kept one primary seed and one orthogonal control. Those are two interventions.

No alpha-`0` or repeat-measurement noise run is in the saved artifacts. "Negligible relative to measurement noise" therefore has no denominator.

`INTERVENTION_DESIGN_DIAGNOSTIC = INCONCLUSIVE`

What the saved distribution does show, without a new label: under the rule M23 already used, this intervention did not produce a falsifying observation. The loop had almost no residual to learn.

## 3. Identifiability analysis

Pre-intervention `pre_dot` on validation D1 ranges from `-3.479` to `0.128` (range `3.607`, sd `1.457`). The D1 effect range on the same rows is `0.007877`. Replication `pre_dot` ranges from `-4.787` to `1.959`. Large differences in the saved internal projection sit next to a tight effect.

Pearson correlation of `pre_dot` with the D1 delta:

- validation, the only split available for fitting: `0.904`
- replication, not used for fitting: `-0.249`

The association changes sign. Baseline margin versus delta is `0.123` on validation and `0.985` on replication. Neither association is stable across the frozen split.

Held-out absolute error against the validation mean (`0.002738`), n = 6:

| predictor fit on validation only | replication MAE | paired reduction vs validation mean | interval class |
| --- | ---: | ---: | --- |
| OLS on `pre_dot` | 0.004740 | -0.002002 | includes zero |
| OLS on `pre_dot` and baseline margin | 0.004478 | -0.001740 | includes zero |
| OLS on baseline margin | 0.002424 | 0.000315 | entirely above zero |
| regime means | 0.001963 | 0.000775 | entirely above zero |

The internal feature recorded at the intervention site, `pre_dot`, does not improve held-out error. Its point estimate is worse, and the interval includes zero. The two positive intervals are small next to the effect itself, come from a menu of specifications, and were not a single contrast frozen before this diagnostic. They are not treated as identification of a context-dependent mechanism.

`IDENTIFIABILITY_FAILURE = INCONCLUSIVE`

The descriptive pattern that remains: the saved pre-intervention state varies by several units while the D1 effect stays inside a band of about `0.008`, and the only strong in-sample internal correlation does not survive the frozen replication split.

## 4. Representation analysis

Predictor A, the constant, is the validation mean. The historical M23 belief is the discovery mean. On replication their MAE values are `0.002738` and `0.002669`. The interval comparing them includes zero.

Predictor B, regime means, and predictor C, the linear state fits, are the table in section 3. B and the baseline-margin line have lower point MAE. `pre_dot` does not. No oracle that sees the replication outcome was fit. That would be leakage. `ORACLE = NOT_RUN`.

The current `predict` function cannot take regime, baseline margin, or `pre_dot` even if a future split shows that one of them is stable. That is a property of the object, measured in section 7. It is not evidence that those features carry held-out information on this intervention.

`REPRESENTATION_BOTTLENECK = INCONCLUSIVE`

Representation expansion is not justified by these six replication prompts. A later claim that a context feature helps needs a contrast that was frozen before looking at that feature's replication errors.

## 5. Belief-update information sensitivity

The real `update_belief` function was applied to a six-copy stand-in of the discovery mean. The correct-evidence result matches the historical updated mean `0.02854486306508382`, so the scalar recursion is the one M23 used.

| evidence | predicted effect after update | change from 0.028900 |
| --- | ---: | ---: |
| none | 0.028900 | 0 |
| correct D1 validation values | 0.028545 | -0.000355 |
| same D1 values, reversed prompt order | 0.028545 | -0.000355 |
| first D1 value repeated six times | 0.030095 | +0.001195 |
| D2 validation values | 0.066169 | +0.037269 |

The reversed order matches the correct update exactly. The update stores a multiset of scalars. It does not store which prompt produced which number.

The four evidence conditions are not the same outcome. Control-direction evidence moves the mean by about `0.037`. Correct D1 evidence moves it by `0.000355`. The update is sensitive to the numbers and insensitive to their assignment.

`BELIEF_UPDATE_INFORMATION_FAILURE = NOT_SUPPORTED`

The name in the task was reserved for the case where correct, shuffled, control, and empty evidence leave the belief approximately unchanged. That is not what happened.

## 6. Main intervention versus control intervention

Discovery means: D1 `0.028900`, D2 `0.106638`. Validation means: D1 `0.028190`, D2 `0.103438`. Both directions are sign-stable. They are different magnitudes.

`predict` does not read the direction and does not read `pre_dot`. It returns `alpha` times whatever scalar was stored. The runner distinguishes D1 from D2 only because the experimenter decides which direction to add and which stored mean to query. That is knowledge of which intervention is about to be applied.

It is not a representation of a causal mechanism that explains the effect. M22.1 left D1 as `CANDIDATE`, and the orthogonal control had the larger recorded effect. Q5 in M23 already showed that pooling these two magnitudes is a worse predictor of D1 than the D1 mean alone. This diagnostic repeats that fact and does not convert it into mechanism discovery.

## 7. Synthetic diagnostic

Run, with fixed inputs, using the existing belief object. Not a new milestone.

- Case A, stable effect: observations are all `1`. The belief stays `1` after another `1`. Held-out constant error is `0`. The object can represent this case. M23's D1 effect is the empirical relative of this case.
- Case B, context-dependent effect: train contexts are `0, 0` and `2, 2`. The belief predicts `1` for both held-out contexts. MAE is `1`. Context means, which `predict` cannot accept, have MAE `0`.
- Case C, two mechanism ids with the same observations: both predictions are `1`. The ids differ. The issued prediction does not.

So the current object can carry a stable mean and can carry two different means when the evidence numbers differ. It cannot carry a context-dependent effect, because `predict` has no context argument. Case B is a capability statement about the code. It does not show that Qwen's D1 effect is context-dependent. Section 3 is the evidence on that question, and it is inconclusive.

## 8. Evidence for and against each explanation

| explanation | label | why |
| --- | --- | --- |
| `INTERVENTION_DESIGN_FAILURE` | `INCONCLUSIVE` | D1 is sign-stable and its spread is about the width of the old falsification band, but no measurement-noise floor was saved, so "negligible versus noise" is not decided |
| `IDENTIFIABILITY_FAILURE` | `INCONCLUSIVE` | `pre_dot` varies far more than the effect, and its correlation flips sign on the frozen test split; that pattern was not given a cutoff that would turn it into `SUPPORTED` |
| `REPRESENTATION_BOTTLENECK` | `INCONCLUSIVE` | regime means and baseline margin have small held-out MAE drops on n = 6 after several looks; `pre_dot` does not help; a small difference is not a winner |
| `BELIEF_UPDATE_INFORMATION_FAILURE` | `NOT_SUPPORTED` | control evidence changes the stored mean by `0.037`; correct evidence changes it by `0.000355`; the update is not blind to the numbers |

These labels can stay open together. None of them rewrites M23.

The update's blindness to prompt identity is a code fact, verified by the reversed-order check. It would matter if prompt-specific effects were identifiable. Section 3 does not establish that they are.

## 9. Remaining uncertainty

- Discovery prompts were run and then discarded as rows. Their `pre_dot` values cannot be put in the fit.
- Six test prompts cannot separate a `0.0003` MAE change from noise in a way that would justify a new representation.
- The replication correlation between baseline margin and delta is `0.985` after a validation correlation of `0.123`. That split disagreement is unresolved. It is not a license to add baseline margin to the belief.
- No repeat of the same prompt isolates intervention noise from prompt variation.
- Alpha `+2` exists only on replication, so a magnitude model cannot be trained on one split and tested on another.

## 10. Recommended next experiment

Do not run it in this task.

Freeze one comparison before any new forward. Predictor A is the current constant belief. Predictor B is the mean within the existing regime label, and regime is the only extra feature. Use a prompt split that is not the M23 replication set, because those regime errors have already been inspected. Judge A against B only with the existing M23 interval rule: the paired interval for the absolute-error reduction must lie entirely above zero, on the new split, or the result stays inconclusive. Do not add a network, a simulator, or a discovery search.

That experiment follows from the evidence because the present D1 effect is too tight for the old update to move held-out error, the saved internal projection does not transfer, and the only context contrast that looked helpful has already been seen on the current replication prompts. A new split is what would tell those facts apart. If B still does not beat A, the data do not support a richer belief for this intervention.
