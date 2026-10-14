# M30 — Replication of the frozen kappa estimator

**Status:** DESIGN_LOCKED  
**Revision:** M30.PROTOCOL.2  
**Parent protocol SHA-256:** `a2662cfae992380524725f1457a412e4aabda9662fcd597f9245ff5b911b6cf2`  
**Parent commit:** `a2aeef32c1bdfe72a52c86f9c86ae3712a6a054d`  
**Parent result:** M29-D verdict `SUPPORTED` in `reports/m29d_raw/VERDICT.json`  
**Purpose:** Design only. No Qwen prediction, intervention, or outcome is authorized by this document.

## 0. Revision diff against the parent protocol

The parent hash above is the SHA-256 of `docs/M30_PROTOCOL.md` at the parent commit. This section is the diff. Unlisted clauses are unchanged.

Unchanged, and not retuned from any outcome:

- `kappa = alpha^2 / 2 * F''(0)` with `alpha = +1`
- Hessian-vector product, dtype `float32`, CPU
- new-identity layers `4`, `12`, and `20`
- direction D2, seed `22103`, hash `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e`
- observable `" true"` minus `" false"`
- primary baseline `g`
- bootstrap procedure: `paired_mean_ci`, 5000 draws, seed `23001`, percentiles 2.5 and 97.5
- magnitude-check predicate: finite values, strictly positive cell medians of `|kappa|` and `|g|`, UPDATE predictions only, ratio recorded and not a threshold
- B1 does not choose the verdict

Changed:

1. Catalog. The previous catalog hash was `8d88819766cf7b454edafecb1582b6284dacea8a8d16f14f2f872090a17316a9` (36 prompts, HOLDOUT 12). This revision's catalog hash is `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770` (48 prompts, HOLDOUT 24). The previous 12 UPDATE texts and 12 VALIDATION texts stay byte-identical. The previous 12 HOLDOUT texts stay. Twelve new HOLDOUT prompts are the image of rule M30-HX1, written in section 4 before those strings were stored.
2. `SUPPORTED` now also requires relative reduction `1 - MAE(g+kappa)/MAE(g) >= 0.25`. A result with prompt-level 95% CI lower bound `> 0` that fails that reduction is `SUPPORTED_WEAK`, not `SUPPORTED`.
3. Anchor arm. The M29-D identity (D1 seed `22101`, layers `0` / `8` / `15`, yes/no margin, token ids `9834` / `902`, alpha `+1`) is scored on this M30 catalog. Same order, same bootstrap, separate `ANCHOR_*` verdict. The anchor does not change the new-identity verdict. Section 6A records the 2x2 interpretation.
4. Per-cell descriptive report for both arms: `MAE(g)`, `MAE(g+kappa)`, `median(|kappa|)`, `median(|g|)`. Not a gate.
5. Holdout prompt scores in the bootstrap change from 12 to 24 because HOLDOUT now has 24 prompts. The bootstrap procedure does not change.
6. The catalog hash is locked in this protocol, in `scripts/m30_catalog.py`, and in `reports/M30_CATALOG_AUDIT.md`. A mismatch aborts.

Leakage failure stays `INCONCLUSIVE`. `Delta <= 0` stays `NOT_SUPPORTED`. `Delta > 0` with a CI lower bound that is not above 0 stays `INCONCLUSIVE`.

## 1. Question

Does the same pre-outcome curvature estimator that was frozen for M29-D predict residual error of `g` on a catalog disjoint from M22.1 through M29-D, under one new intervention identity fixed before any M30 measurement?

This is a replication of the estimator, not a search for a new correction.

The anchor arm asks the same question under the inherited M29-D identity, on this same catalog. It is a comparison, not a second search.

## 2. Estimator, unchanged

The estimator is the M29-D definition. It is not refit. Both arms use it.

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
- catalog SHA-256: `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`
- 48 prompts
- UPDATE 12, VALIDATION 12, HOLDOUT 24
- families: completion, syntax, instruction
- UPDATE and VALIDATION: 4 prompts from each family
- HOLDOUT: 8 prompts from each family
- assignment of index 1..12 within each family: `(index - 1) mod 3` maps to UPDATE, VALIDATION, HOLDOUT
- indices 13..16 within each family are HOLDOUT under rule M30-HX1
- the 12 UPDATE texts and the 12 VALIDATION texts are byte-identical to the parent revision
- disjoint from M22.1, M23-G, the post-M23 single-measurement catalog, the feasibility-gate catalog, M24, M26, and M29-D
- also disjoint inside M30: the 12 new holdout prompts against UPDATE and against VALIDATION
- overlap checks: exact id, exact text, and whitespace-collapsed case-folded text

The catalog hash in the module, in this protocol, and in the audit report is one lock. A future run must recompute it and abort on mismatch.

### Holdout extension rule M30-HX1

This rule was written before the twelve new strings were generated. The catalog stores the image of the rule. It does not store a hand-edited sentence list. If a disjointness check fails, the rule is rejected and replaced. An emitted string is not rewritten on its own.

For `k = 0, 1, 2, 3`, prompt index is `13 + k`:

| k | article | material | implement | target | prompt index |
| --- | --- | --- | --- | --- | ---: |
| 0 | A | birch | burnisher | fore-edge | 13 |
| 1 | A | copper | bodkin | headband | 14 |
| 2 | An | ivory | mallet | sewing-frame | 15 |
| 3 | A | wool | caliper | lying-press | 16 |

Templates:

- completion: `The {material} {implement} is kept beside the {target}`
- syntax: `Were the {material} {implement} left upon the {target},`
- instruction: `Answer with a single noun. {article} {material} {implement} belongs with the:`

`prompt_id` is `m30-{family}-{index}` with the index zero-padded to two digits. The partition is HOLDOUT.

The previous holdout indices in each family are 3, 6, 9, and 12. Those four texts stay. Each family therefore has 8 holdout prompts.

## 5. Order

The order is the same one locked in the parent revision. It is applied to the new identity and to the anchor arm. No outcome is measured for either arm until both arms have finished the steps before outcomes.

1. Confirm the catalog hash, the D2 hash, and the D1 hash.
2. Resolve the observable token ids. Abort if either text is not one token. Record the ids. Do not search for a replacement.
   - New identity: `" true"` and `" false"`.
   - Anchor: `" yes"` and `" no"`. The ids must be `9834` and `902`.
3. On UPDATE, VALIDATION, and HOLDOUT, for both arms, compute `g` and `kappa` and persist prediction rows.
4. Freeze the prediction hash.
5. Run the magnitude check in section 7 on UPDATE predictions only, separately for each arm.
6. Only if both checks pass, measure scored outcomes.
7. Score HOLDOUT once per arm with the rule in section 6.

VALIDATION predictions may be stored. They are not an input to acceptance, to the magnitude check, or to any choice of layer, direction, observable, sign, or scale.

The anchor score does not change the new-identity verdict.

## 6. Acceptance

Primary baseline: `g` alone.

On HOLDOUT rows of the arm being scored:

- `MAE(g) = mean(|y - g|)`
- `MAE(g + kappa) = mean(|y - (g + kappa)|)`
- `Delta = MAE(g) - MAE(g + kappa)`
- relative reduction `1 - MAE(g+kappa)/MAE(g)`, defined only when `MAE(g) > 0`

The reduction uses the same pooled holdout rows as `Delta`. It is not a mean of per-cell reductions.

Prompt-level interval, fixed now:

- within each holdout prompt, average the three cell paired differences `|y - g| - |y - (g + kappa)|`
- 24 prompt scores
- `paired_mean_ci` from `src.cognitive_self_model.m23.stats`
- 5000 draws
- seed `23001`
- percentiles 2.5 and 97.5

First matching clause:

1. Leakage checks fail: `INCONCLUSIVE`.
2. `Delta <= 0`: `NOT_SUPPORTED`.
3. Prompt-level 95% interval lower bound is not `> 0`: `INCONCLUSIVE`.
4. `MAE(g) == 0`, or the relative reduction is `< 0.25`: `SUPPORTED_WEAK`.
5. Otherwise: `SUPPORTED`.

`SUPPORTED` therefore requires all of:

1. `Delta > 0`
2. prompt-level 95% interval lower bound `> 0`
3. `1 - MAE(g+kappa)/MAE(g) >= 0.25`
4. both quantities use the same holdout prompts
5. leakage checks pass
6. the section 7 magnitude check already passed before any outcome

A prompt-level 95% interval lower bound `> 0` that does not meet the relative-reduction requirement is `SUPPORTED_WEAK`. No other result may be relabeled `SUPPORTED` or `SUPPORTED_WEAK`.

The anchor arm uses this same list on its own holdout rows. Its labels are `ANCHOR_INCONCLUSIVE`, `ANCHOR_NOT_SUPPORTED`, `ANCHOR_SUPPORTED_WEAK`, and `ANCHOR_SUPPORTED`. The anchor label is not an input to the new-identity label.

### Secondary, not a gate

B1 is the mean of `y - g` inside each cell, estimated from UPDATE outcomes only. Its holdout MAE and a prompt-level interval of `MAE(B1) - MAE(g + kappa)` are reported for each arm. B1 does not choose the verdict.

There is no permutation gate and no second feature.

## 6A. Anchor arm and the 2x2 table

The anchor arm copies the M29-D identity onto the M30 catalog. It is not chosen from M30 or M29 cell scores.

| cell | layer | hook |
| --- | ---: | --- |
| `M22.1-D1-L0` | 0 | `blocks.0.hook_resid_post` |
| `M22.1-D1-L8` | 8 | `blocks.8.hook_resid_post` |
| `M22.1-D1-L15` | 15 | `blocks.15.hook_resid_post` |

- direction D1: `primary_direction(896, 22101)`
- seed: `22101`
- SHA-256: `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`
- alpha: `+1`
- positive text: `" yes"`
- negative text: `" no"`
- token ids: `9834` and `902`
- margin: last-position logit of token `9834` minus last-position logit of token `902`
- if the resolved ids are not that pair, or either text is not one token, execution stops
- dtype `float32`, same estimator, same predictions-before-outcomes order, same bootstrap

The anchor does not change the new-identity verdict.

Interpretation uses the primary labels only. `SUPPORTED_WEAK` counts as not `SUPPORTED`. The table does not relabel either verdict.

|  | New identity `SUPPORTED` | New identity not `SUPPORTED` |
| --- | --- | --- |
| Anchor `ANCHOR_SUPPORTED` | Both identities beat `g` on this catalog under the frozen rule. | The inherited identity beats `g` on this catalog. The new identity does not meet `SUPPORTED`. |
| Anchor not `ANCHOR_SUPPORTED` | The new identity meets `SUPPORTED` on this catalog. The inherited identity does not. | Neither identity meets `SUPPORTED` on this catalog. |

## 6B. Per-cell description, not a gate

After HOLDOUT is scored once, report the following for every cell of both arms, using that arm's HOLDOUT rows:

- `MAE(g)`
- `MAE(g + kappa)`
- `median(|kappa|)`
- `median(|g|)`

New-identity cells: `M30-D2-L4`, `M30-D2-L12`, `M30-D2-L20`.  
Anchor cells: `M22.1-D1-L0`, `M22.1-D1-L8`, `M22.1-D1-L15`.

These numbers do not choose a verdict, a layer, a direction, a sign, a scale, or an observable. They are not compared to the `0.25` threshold. The threshold applies only to the pooled holdout reduction in section 6.

## 7. Magnitude check

Computed on M30 UPDATE predictions only, separately for each arm. Outcomes are not available to it.

For each cell:

- `median(|kappa|)`
- `median(|g|)`
- ratio `median(|kappa|) / median(|g|)`

The check passes only when every UPDATE `g` and `kappa` is finite, every cell median absolute kappa is strictly positive, and every cell median absolute `g` is strictly positive.

If either arm fails, stop. Do not measure outcomes. Do not change alpha, sign, layers, direction, or the observable. The ratio is recorded and is not a threshold for `SUPPORTED`. The relative-reduction requirement in section 6 is not part of this check.

## 8. Leakage

Unavailable to `kappa`, for either arm:

- observed `y`
- residual `y - g`
- holdout outcomes
- M29 holdout cell scores
- any statistic used to retune this identity

Prediction rows are closed before outcome rows. Holdout rows do not enter B1 or the magnitude check. The anchor arm does not supply a feature, a sign, or a scale to the new identity.

## 9. What a later run must save

- protocol identity, protocol hash, and catalog hash
- model revision
- D1 hash and D2 hash
- resolved token ids for both observables
- dtype, device, alpha
- raw `g` and `kappa` for both arms, before outcomes
- magnitude-check values for both arms
- raw `y` and residual after outcomes
- row-level MAE values
- relative reduction for both arms
- per-cell descriptive table from section 6B
- prompt-level bootstrap seed, draw count, and interval for both arms
- leakage audit
- the new-identity verdict and the separate anchor verdict

## 10. Stop

M30 is design-locked here. The next edit that measures Qwen is a separate execution step and is not authorized by this file. A failed replication is a result, not a license to change the estimator or to pick a different layer from M29 cell scores.

PROTOCOL_SHA256: 1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3
