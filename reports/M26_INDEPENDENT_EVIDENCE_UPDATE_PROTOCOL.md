# M26 independent evidence update

Status: `PROTOCOL_FROZEN`

This file freezes the residual update and the catalogs. It does not run Qwen, does not execute the benchmark, and does not change `MechanismResponseModel`, `MechanismBelief`, the M24 artifacts, or the M25 artifacts.

Scientific question: after `g` is held fixed, does a predeclared residual correction estimated on a new update catalog reduce absolute error on a post-update holdout that was not used to form the correction, by more than one global correction and by more than a shuffle of the same residuals?

The updated prediction is `g + δ`. This is not a self-model, and it is not a new value of `g`.

## Initial belief

The response remains the existing `MechanismResponseModel` for `M22.1-D1-L0`, `M22.1-D1-L8`, and `M22.1-D1-L15`. Its rule stays `prediction = g`. The initial correction is `δ = 0`, so the pre-update prediction is `g`.

The observation scale `σ` is the published M25 calibration residual standard deviation at inflation `1`. Those three numbers were computed from the M24 validation partition only. M24 evaluation rows are not used to set `σ` or `δ`. The M25 status labels are not an input.

| cell | σ |
| --- | ---: |
| `M22.1-D1-L0` | 0.02029281160603693 |
| `M22.1-D1-L8` | 0.009377970182012517 |
| `M22.1-D1-L15` | 0.004044481362351381 |

Surface family is balance only. It does not receive its own `δ`. Layer 23 and D2 are not update targets.

## Update equation

For one cell, the update residuals are `r = observed_effect - g` on the update partition. Let `n` be the count of those residuals and `r_bar` their mean.

The prior is `δ ~ Normal(μ0, τ0²)` with `μ0 = 0` and `τ0 = σ`. That is one pseudo-observation at residual `0`. The prior count is `n0 = 1`. It is not estimated from the update catalog.

The likelihood is `r_i | δ ~ Normal(δ, σ²)`, independent across the update rows of that cell. `σ` stays the published calibration scale. It is not re-estimated.

The posterior mean and posterior standard deviation are

`δ = (n0 μ0 + n r_bar) / (n0 + n) = n r_bar / (n + 1)`

`τ = σ / sqrt(n0 + n)`

The updated prediction is `prediction_after = g + δ`.

A new residual around that prediction has scale `sqrt(τ² + σ²)`. An episode is inside the descriptive interval when `abs(observed_effect - prediction_after) <= 1.96 * sqrt(τ² + σ²)`. The factor `1.96` is the existing `CRITICAL_Z`. Coverage is reported and is not a verdict input. Equality counts as covered.

The global baseline uses the same shrinkage on the pooled update residuals of all three cells:

`δ_global = n_all r_bar_all / (n_all + 1)`

Every cell then uses `g + δ_global`. This is one number. It does not depend on which cell produced a residual.

The shuffled control sorts update rows by `intervention_id` then `prompt_id`, using ordinary string order, and permutes the residual values with `random.Random(26001)`. Under that string order, `M22.1-D1-L15` precedes `M22.1-D1-L8`. The permuted residuals are assigned back to those rows and the cell posterior means are recomputed with the same equation. The holdout `g` is not shuffled.

The stored scalar baseline is `training_baseline_mean` on the response model. Its prediction does not add `δ`. It is reported and is not a verdict input.

## Catalog

catalog sha256: `015c3d41855da5fa630c382fe0e4a9ea63c868e12e4a1243e104e1c4e371ea6f`

Twenty-four `u-*` prompts. Eight in each surface family. Partition is `(index - 1) mod 2`, so each family contributes four update prompts and four holdout prompts. Twelve prompts are the update partition. Twelve are the holdout. The two sets are disjoint, and both are disjoint from the M22.1, M23-G, single-measurement, feasibility-gate, and M24 catalogs. The hash is the sha256 of the catalog JSON with sorted keys.

| prompt_id | family | partition | text |
| --- | --- | --- | --- |
| `u-completion-01` | completion | update | The capital of Portugal is |
| `u-completion-02` | completion | holdout | Brick is fired from |
| `u-completion-03` | completion | update | The author of The Odyssey is |
| `u-completion-04` | completion | holdout | A harp has many |
| `u-completion-05` | completion | update | The Danube is a |
| `u-completion-06` | completion | holdout | A stop sign has eight |
| `u-completion-07` | completion | update | The currency of Mexico is the |
| `u-completion-08` | completion | holdout | Ink is often stored in a |
| `u-syntax-01` | syntax | update | Assuming the valve sticks, |
| `u-syntax-02` | syntax | holdout | By the time the glacier retreated, |
| `u-syntax-03` | syntax | update | So that the seedling would root, |
| `u-syntax-04` | syntax | holdout | Wherever the creek bends, |
| `u-syntax-05` | syntax | update | No matter how loud the mill is, |
| `u-syntax-06` | syntax | holdout | The moment the comet appeared, |
| `u-syntax-07` | syntax | update | Provided the cistern stays sealed, |
| `u-syntax-08` | syntax | holdout | Whereas the third assay failed, |
| `u-instruction-01` | instruction | update | Reply with one word. A liquid used in lamps: |
| `u-instruction-02` | instruction | holdout | Reply with one word. A drink made from barley: |
| `u-instruction-03` | instruction | update | Name a tool used for weaving: |
| `u-instruction-04` | instruction | holdout | Name a moon of Jupiter: |
| `u-instruction-05` | instruction | update | Reply with one word. Opposite of hollow: |
| `u-instruction-06` | instruction | holdout | Name a kind of spice: |
| `u-instruction-07` | instruction | update | Reply with one word. A bread served with soup: |
| `u-instruction-08` | instruction | holdout | Name a wind instrument: |

## Order

1. Freeze this catalog.
2. On the update prompts, compute `g` from `MechanismResponseModel.predict` and write the prediction file before any update outcome exists.
3. Run the alpha-`+1` intervention and write update outcomes in a separate file.
4. Freeze `δ`, `δ_global`, and `δ_shuffled` from those update residuals only.
5. On the holdout prompts, compute `g` and write every prediction, including `g`, `g + δ`, `g + δ_global`, `g + δ_shuffled`, and the stored scalar, before any holdout outcome exists.
6. Run the holdout interventions and write those outcomes in a separate file.

Prediction files contain no observed effect. `update_residuals` rejects a row whose partition is not `update`.

## Primary comparison

On the holdout, the paired improvement against no update is

`abs(effect - g) - abs(effect - (g + δ_cell))`

The comparison against the global correction is

`abs(effect - (g + δ_global)) - abs(effect - (g + δ_cell))`

The shuffled comparison is

`abs(effect - g) - abs(effect - (g + δ_shuffled_cell))`

Each score is averaged across the three cells of a holdout prompt. `paired_mean_ci` resamples the twelve prompt scores with 5000 draws and seed `23001`. Prompt ids are sorted before the list is built. Per-cell intervals are reported and do not replace the pooled interval.

Sign agreement uses the existing strict-sign rule and is reported with a Clopper-Pearson interval. It is not a verdict input. The scalar baseline's MAE is reported. It is not a verdict input.

## Verdict

`update_verdict` in `src/cognitive_self_model/residual_update.py` is the only map.

`UPDATE_SUPPORTED` only when all of these hold:

1. The holdout class of `MAE(g) - MAE(g + δ_cell)` is `CI_POSITIVE`.
2. The holdout class of `MAE(g + δ_global) - MAE(g + δ_cell)` is `CI_POSITIVE`.
3. The holdout class of `MAE(g) - MAE(g + δ_shuffled)` is not `CI_POSITIVE`.
4. `leakage_ok` is true.

`UPDATE_NOT_SUPPORTED` only when the first of those classes is `CI_NEGATIVE` and `leakage_ok` is true.

Every other combination is `INCONCLUSIVE`. That includes an improvement over `g` that does not beat the global correction, an improvement that the shuffle also achieves, an interval that includes zero, and any result with `leakage_ok` false.

## Leakage

`leakage_ok` is true only when the update prediction file is closed before the update outcome file, the three corrections are frozen before the holdout prediction file, the holdout prediction file is closed before the holdout outcome file, no holdout row enters `δ`, the response model object is unchanged, and `σ` is still the three published values.

No slope is fit to `g`. No prompt embedding is built. `pre_dot` is not a feature. `SelfModelBelief` is not read or written.

## Seeds and manifest

Bootstrap draws `5000`, seed `23001`. Shuffle seed `26001`. Prior count `1`. Prior mean `0`. No other random source enters `δ`.

This freeze creates the protocol, the catalog module, the update equations, and the unit tests. It does not create `reports/m26_raw/`, an execution report, or a results file. The benchmark command, to be used only after this protocol is reviewed, is

`python3 scripts/run_m26_independent_evidence_update.py`

## What a result would not mean

`UPDATE_SUPPORTED` would mean that this per-cell residual correction, fit on this update catalog, reduced holdout absolute error relative to raw `g`, and that a single pooled correction and a shuffle of the same residuals did not account for that reduction. It would not retune `g`, would not clear the M25 contradiction record, and would not establish a self-model.

`UPDATE_NOT_SUPPORTED` would mean the correction increased holdout absolute error relative to raw `g` on this catalog. It would not show that a residual update is impossible under every other catalog or model.

`INCONCLUSIVE` leaves the correction unestablished. It does not authorize a second prior, a second shrinkage count, or a new observable.
