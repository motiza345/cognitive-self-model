# S2 pre-registration

**Question:** on the S1 holdout task, does the white-box quadratic (`M2`) beat a black-box quadratic fit from `F(+1)` and `F(-1)`?

**Locked before the FDQ run.** S1 rows are inputs. They are not recomputed. No coefficient, clip, scale, or threshold below is chosen from FDQ output.

## Setup

Same prompts, cells, hooks, directions, tokens, targets, and model as S1.

- 24 M30 HOLDOUT prompts, both arms, three cells each, scales `s ∈ {1, 2, 4, 8}`.
- Target on row `i` is the S1 target `T = s * g_i`. `g` is the frozen M30 prediction copied in `reports/s1_raw/ROWS.json`.
- `M1`, `M2`, and `BB_3` misses are the prompt-level misses already stored in that file. No method except FDQ is run again.
- `F(a)` is the margin with `a * d` added at the last token. `f0 = F(0)`. `h(a) = F(a) - f0`.
- CPU, float32 forwards. Algebra in float64.
- Miss is `|h(alpha) - T| / |T|`, from one verification forward at the chosen alpha. That forward is not charged. A non-finite margin or alpha aborts the run. No row is dropped.

## FDQ

On each holdout row, three forwards:

- `b = (F(1) - F(-1)) / 2`
- `c = (F(1) + F(-1)) / 2 - f0`

Solve `c*a^2 + b*a = T` with the M2 root rule, substituting `(b, c)` for `(g, kappa)`:

- If `c == 0` and `b == 0`, abort.
- If `c == 0` and `b != 0`, `alpha = T / b`. When `b = g` and `T = s*g`, this is `s`, matching M2.
- Otherwise `disc = b^2 + 4*c*T`.
- If `disc < 0`, `alpha = -b / (2*c)`.
- If `disc >= 0`, take the root `(-b ± sqrt(disc)) / (2*c)` nearest to `s`. Equal distance: the smaller root.
- Then clip to `[-4*s, 4*s]`. Clip after selection.

`F(0)`, `F(1)`, and `F(-1)` are measured once per row and reused for every scale. The verification is per scale.

## Cost

FDQ costs 3 forward-equivalents because the fit uses `F(0)`, `F(+1)`, and `F(-1)`. The count is 3. It is not estimated from a time ratio.

Seconds are measured. After one untimed warmup of those three calls, each of the 144 rows records `perf_counter` around `F(0)`, around `F(+1)`, and around `F(-1)`. The row cost is the sum. `cost_FDQ` is the median of the 144 sums. The same timed calls supply `b` and `c`. Verification time is not included.

`cost_M2` is the S1 median Hessian-vector time. It is not measured again.

## Decision

A prompt score is the mean of the three cell misses. Prompts are sorted by `prompt_id`. Medians and means use those 24 scores. For 24 values the median is the average of the two central sorted values.

The paired difference is `miss_FDQ - miss_M2` at the prompt level. Its interval is a bootstrap of the median of those 24 differences:

- `random.Random(23001)`, 5000 draws, resample prompts with replacement
- percentile indexes `floor(0.025 * 4999)` and `ceil(0.975 * 4999)`, the same indexes as `paired_mean_ci`
- each condition starts a new generator at seed 23001

The decision set is the four conditions `{new identity, anchor} × {s = 4, s = 8}`.

Predicate W: `median(miss_M2) <= 0.75 * median(miss_FDQ)` and the bootstrap lower bound is `> 0`.

Predicate N: `median(miss_FDQ) <= median(miss_M2)`.

If `median(miss_FDQ) > 0`, W implies `median(miss_M2) < median(miss_FDQ)`, so N is false. If a condition meets both predicates, abort. Do not emit a label.

- **WHITE_BOX_ADVANTAGE.** W holds on at least 3 of the 4 conditions.
- **NO_ADVANTAGE.** N holds on at least 3 of the 4 conditions.
- **MIXED.** Otherwise.

First matching clause. The two verdicts cannot both match when no condition meets both predicates.

Scales `s = 1` and `s = 2` get the same medians, means, trimmed means, and intervals. They are not part of the decision.

Descriptive only, for every arm, scale, and method `M1`, `M2`, `FDQ`, and `BB_3`: the arithmetic mean of the 24 prompt scores, and the 10% trimmed mean. The trimmed mean drops `int(0.1 * 24) = 2` scores from each end of the sorted list and averages the remaining 20.

No choice in this file is revised after the FDQ forwards.
