# S1 pre-registration

**Question:** does frozen `kappa` have decision value when choosing the scale of a last-token residual steer?

**Locked before the run.** No catalog, direction, layer, token, or threshold below is chosen from S1 outputs. M30 protocol `M30.PROTOCOL.2` is not edited.

## Setup

- Catalog: M30 HOLDOUT, 24 prompts. Both arms.
- New identity: D2, layers 4 / 12 / 20, hooks `blocks.{4,12,20}.hook_resid_post`, margin `" true"` minus `" false"`.
- Anchor: D1, layers 0 / 8 / 15, hooks `blocks.{0,8,15}.hook_resid_post`, margin `" yes"` minus `" no"`. Token ids must be 9834 and 902.
- Model `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, CPU, float32, `prepend_bos=True`.
- `F(a)` is the margin with `a * d` added at the last token only (`make_resid_hook`). `f0 = F(0)`. `h(a) = F(a) - f0`.
- `g` and `kappa` are copied from `reports/m30_raw/PRE_OUTCOME_PREDICTIONS.json`. Prediction SHA-256 must be `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`. Catalog SHA-256 must be `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`. Abort on mismatch.
- `kappa` was stored at alpha `+1`, so the quadratic prediction is `h(a) ≈ g*a + kappa*a^2`. The quadratic coefficient is `kappa`, not `kappa/2`.
- Holdout rows of the M30 outcome file are not inputs. They are not used for `gbar`, for alpha, or for the miss.

## Task

Scales `s ∈ {1, 2, 4, 8}`. On each holdout row `i`, target `T = s * g_i`. Choose `alpha` so that `h(alpha) = T`.

`miss = |h(alpha) - T| / |T|`.

The miss is measured by one verification forward at the chosen alpha. That forward is not charged to the method. `f0` is measured once per prompt and arm, shared by every cell and method, and is not charged.

If any holdout `g` is 0 or non-finite, or any chosen alpha or measured margin is non-finite, abort. Do not drop the row and do not emit a decision.

## Methods

Algebra is float64. Forwards are float32.

**M1.** `alpha = s`. This is `T/g` when `g ≠ 0`. No clip.

**M2.** Solve `kappa*a^2 + g*a = T`.

- If `kappa == 0`, `alpha = s`.
- Otherwise `disc = g^2 + 4*kappa*T`.
- If `disc < 0`, use the vertex `alpha = -g / (2*kappa)`.
- If `disc >= 0`, the roots are `(-g ± sqrt(disc)) / (2*kappa)`. Take the real root nearest to `s`. If the distances are equal, take the smaller root.
- Then clip to `[-4*s, 4*s]`. Clip after selection, not before.

**BB_k**, `k ∈ {1, 2, 3, 4, 6, 8}`, forwards only.

`gbar_cell` is the mean `observed_effect` on the 12 M30 UPDATE rows of that arm and cell (alpha `+1` outcomes). No holdout outcome enters the mean. If a cell mean is 0 or non-finite, abort.

`h(0) = 0` is known and is not a probe. Secant steps always use `(0, 0)` and the latest probe:

- Probe 1 is `a_1 = T / gbar_cell`. Evaluate `h_1 = h(a_1)`.
- The estimate from a probe `(a, h)` is `a` when `h == 0`, otherwise `T * a / h`.
- Probe `j+1` is the estimate from probe `j`.
- After `k` evaluated probes, the answer is the estimate from probe `k`. That answer is not evaluated inside the method.

One probe sequence of length `max(8, k_eq)` is run per row and scale. `BB_k` reads only the first `k` probes. There is no clip on black-box alphas.

## Cost

On each of the 144 holdout rows, after one untimed warmup of the same three calls, record wall-clock with `time.perf_counter`:

- one forward: hooked margin at alpha `1`
- M1: margin gradient at the unperturbed state, `autograd.grad` without `create_graph`, then the dot product with `d`
- M2: one call of the frozen `measure_kappa` (create-graph Hessian-vector product)

The numbers produced by those timed calls are discarded. They do not replace frozen `g` or `kappa`.

`cost_forward`, `cost_M1`, and `cost_M2` are the medians of those 144 times. Forward-equivalents are `cost_M1 / cost_forward` and `cost_M2 / cost_forward`. Black-box `k` costs `k` forwards and `k * cost_forward` seconds.

`k_eq = ceil(cost_M2 / cost_forward)`. If `cost_forward <= 0`, abort. If `k_eq` is outside `{1, 2, 3, 4, 6, 8}`, run that `k` as well.

## Analysis

For each arm, scale, and method, the prompt score is the mean of the three cell misses. Means and medians are over the 24 prompt scores, prompts sorted by `prompt_id`. The median of 24 values is the average of the two central sorted values.

Contrasts use `paired_mean_ci` from `src.cognitive_self_model.m23.stats`: 5000 draws, seed 23001, percentiles 2.5 and 97.5. Each contrast starts a new generator at that seed.

- M2 versus M1: prompt difference `miss_M1 - miss_M2`. Relative reduction `1 - mean(miss_M2) / mean(miss_M1)`, defined only when `mean(miss_M1) > 0`.
- M2 versus equal-cost black box, and M2 versus `BB_2`: the same paired difference with `miss_BB` in place of `miss_M1`. Those intervals are reported. Only the median comparison below enters the decision.

M2 beats M1 at one `(arm, s)` when the reduction is defined, the reduction is `>= 0.25`, and the paired interval lower bound is `> 0`.

Let `S` be the four pairs `{new identity, anchor} × {s = 4, s = 8}`.

1. **GO.** On every pair in `S`, M2 beats M1, and `median(miss_M2) <= median(miss of BB_{k_eq})`.
2. **PIVOT.** On every pair in `S`, M2 beats M1, and on every pair in `S` the equal-cost black box has the strictly lower median miss.
3. **STOP.** M2 does not beat M1 on every pair in `S`.
4. **MIXED.** M2 beats M1 on every pair in `S`, and the four median comparisons are neither all `<=` nor all strict black-box wins.

First matching clause. Scales `s = 1` and `s = 2` are reported and are not part of the decision. `BB_2` is not the equal-cost arm unless `k_eq = 2`.

The accuracy-versus-cost table reports, for M1, M2, and each executed `BB_k`, the median prompt-level miss at each arm and scale, the median seconds, and the forward-equivalents.

No coefficient, clip, scale, or `k` is changed after the holdout forwards.
