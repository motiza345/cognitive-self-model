# M30 — Design lock

M30 replicates the frozen M29-D curvature estimator on a new catalog and one new intervention identity.

The estimator stays `kappa = alpha^2 / 2 * F''(0)` with `alpha = +1`, the same Hessian-vector product, and float32. The new identity is layers 4, 12, and 20, direction D2, and the `" true"` minus `" false"` margin. Those choices are fixed before any M30 measurement and are not taken from M29 cell scores.

Revision M30.PROTOCOL.2 enlarges HOLDOUT from 12 to 24 prompts. The previous UPDATE and VALIDATION texts stay byte-identical. The twelve new holdout prompts are the image of rule M30-HX1. Catalog SHA-256: `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`.

`SUPPORTED` requires `Delta > 0`, a prompt-level 95% CI lower bound `> 0`, and relative reduction `1 - MAE(g+kappa)/MAE(g) >= 0.25`. Otherwise a CI lower bound `> 0` is `SUPPORTED_WEAK`. An anchor arm runs the M29-D identity on this catalog and records a separate `ANCHOR_*` verdict. It does not change the new-identity verdict. Per-cell MAE and median absolute `g` and `kappa` are reported for both arms and are not a gate.

Primary comparison: holdout `MAE(g) - MAE(g + kappa)`, with a prompt-level paired bootstrap of 5000 draws and seed 23001. The cell-mean baseline B1 is secondary. A magnitude check on UPDATE predictions must pass before any outcome.

## Files

- protocol: `docs/M30_PROTOCOL.md`
- catalog: `scripts/m30_catalog.py`
- disjointness audit: `reports/M30_CATALOG_AUDIT.md`

## Status

DESIGN_LOCKED. No Qwen run.
