# M30 — Replication of the frozen kappa estimator

**Status:** DESIGN_LOCKED  
**Parent result:** M29-D verdict `SUPPORTED` in `reports/m29d_raw/VERDICT.json`  
**Purpose:** Design only. No Qwen prediction, intervention, or outcome is authorized by this document.

## 1. Question

Does the same pre-outcome curvature estimator that was frozen for M29-D predict residual error of `g` on a catalog disjoint from M22.1 through M29-D, under one new intervention identity fixed before any M30 measurement?

This is a replication of the estimator, not a search for a new correction.

## 2. Estimator, unchanged

The estimator is the M29-D definition. It is not refit.

- `F(t)` is the scalar margin with `t * d` added at the last token of the frozen residual hook.
- `g = F'(0) = <grad F, d>`
- `kappa = alpha^2 / 2 * F''(0)`
- `alpha = +1`
- `r_hat = kappa`
- `y_hat = g + kappa`
- Estimator: PyTorch `autograd.grad(..., create_graph=True)` Hessian-vector product at the unperturbed state.
- dtype: `float32`
- device: CPU, matching the M29-D numerical regime
- finite differences: forbidden
- polynomial fits to outcomes: forbidden
- sign or scale of kappa: not adjustable after any outcome

Model: `Qwen/Qwen2.5-0.5B`  
Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`  
`d_model`: 896  
`n_layers`: 24

## 3. New intervention identity

The new identity is fixed by rule, before any M30 prompt is scored. It is not chosen from M29 holdout cell errors, medians, correlations, or signs.

Layers are the M29 stride of 8, shifted by `+4`, and kept inside `0 .. 23`. That set is disjoint from the M29 layers `0`, `8`, and `15`:

| cell | layer | hook |
| --- | ---: | --- |
| `M30-D2-L4` | 4 | `blocks.4.hook_resid_post` |
| `M30-D2-L12` | 12 | `blocks.12.hook_resid_post` |
| `M30-D2-L20` | 20 | `blocks.20.hook_resid_post` |

Direction is the already published orthogonal direction D2:

- construction: `orthogonal_direction(896, 22103, primary_direction(896, 22101))`
- seed: `22103`
- SHA-256: `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e`
- alpha: `+1`
- the add remains last-token only, through `make_resid_hook`

Observable, replacing the M29 yes/no margin:

- positive text: `" true"`
- negative text: `" false"`
- margin: last-position logit of the positive text minus last-position logit of the negative text
- both strings must encode to exactly one token
- if either string is not a single token, execution stops
- no alternate wording may be substituted after that stop

`g` and `kappa` are derivatives of this new margin. They are not the M29 yes/no derivatives copied onto a new hook.

## 4. Catalog

- module: `scripts/m30_catalog.py`
- audit: `reports/M30_CATALOG_AUDIT.md`
- 36 prompts
- UPDATE 12, VALIDATION 12, HOLDOUT 12
- families: completion, syntax, instruction
- assignment within each family: `(index - 1) mod 3` maps to UPDATE, VALIDATION, HOLDOUT
- disjoint from M22.1, M23-G, the post-M23 single-measurement catalog, the feasibility-gate catalog, M24, M26, and M29-D
- overlap checks: exact id, exact text, and whitespace-collapsed case-folded text

The catalog hash in the audit report is the lock. A future run must recompute it and abort on mismatch.

## 5. Order

1. Confirm the catalog hash and the D2 hash.
2. Resolve the observable token ids. Abort if either text is not one token. Record the ids. Do not search for a replacement.
3. On UPDATE, VALIDATION, and HOLDOUT, compute `g` and `kappa` and persist prediction rows.
4. Freeze the prediction hash.
5. Run the magnitude check in section 7 on UPDATE predictions only.
6. Only if that check passes, measure scored outcomes.
7. Score HOLDOUT once with the rule in section 6.

VALIDATION predictions may be stored. They are not an input to acceptance, to the magnitude check, or to any choice of layer, direction, observable, sign, or scale.

## 6. Acceptance

Primary baseline: `g` alone.

On HOLDOUT rows:

- `MAE(g) = mean(|y - g|)`
- `MAE(g + kappa) = mean(|y - (g + kappa)|)`
- `Delta = MAE(g) - MAE(g + kappa)`

Prompt-level interval, fixed now:

- within each holdout prompt, average the three cell paired differences `|y - g| - |y - (g + kappa)|`
- 12 prompt scores
- `paired_mean_ci` from `src.cognitive_self_model.m23.stats`
- 5000 draws
- seed `23001`
- percentiles 2.5 and 97.5

`SUPPORTED` requires all of:

1. `Delta > 0`
2. prompt-level 95% interval lower bound `> 0`
3. both quantities use the same holdout prompts
4. leakage checks pass
5. the section 7 magnitude check already passed before any outcome

`INCONCLUSIVE` if `Delta > 0` and the interval lower bound is not above 0, or if leakage fails.

`NOT_SUPPORTED` if `Delta <= 0`.

No other result may be relabeled `SUPPORTED`.

### Secondary, not a gate

B1 is the mean of `y - g` inside each cell, estimated from UPDATE outcomes only. Its holdout MAE and a prompt-level interval of `MAE(B1) - MAE(g + kappa)` are reported. B1 does not choose the verdict.

There is no permutation gate and no second feature.

## 7. Magnitude check

Computed on M30 UPDATE predictions only. Outcomes are not available to it.

For each cell:

- `median(|kappa|)`
- `median(|g|)`
- ratio `median(|kappa|) / median(|g|)`

The check passes only when every UPDATE `g` and `kappa` is finite, every cell median absolute kappa is strictly positive, and every cell median absolute `g` is strictly positive.

If the check fails, stop. Do not measure outcomes. Do not change alpha, sign, layers, direction, or the observable. The ratio is recorded and is not a threshold for `SUPPORTED`.

## 8. Leakage

Unavailable to `kappa`:

- observed `y`
- residual `y - g`
- holdout outcomes
- M29 holdout cell scores
- any statistic used to retune this identity

Prediction rows are closed before outcome rows. Holdout rows do not enter B1 or the magnitude check.

## 9. What a later run must save

- protocol identity and catalog hash
- model revision
- direction hash
- resolved token ids
- dtype, device, alpha
- raw `g` and `kappa` before outcomes
- magnitude-check values
- raw `y` and residual after outcomes
- row-level MAE values
- prompt-level bootstrap seed, draw count, and interval
- leakage audit
- one verdict

## 10. Stop

M30 is design-locked here. The next edit that measures Qwen is a separate execution step and is not authorized by this file. A failed replication is a result, not a license to change the estimator or to pick a different layer from M29 cell scores.
