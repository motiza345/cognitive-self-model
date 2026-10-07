# M30 — Design lock

M30 replicates the frozen M29-D curvature estimator on a new catalog and one new intervention identity.

The estimator stays `kappa = alpha^2 / 2 * F''(0)` with `alpha = +1`, the same Hessian-vector product, and float32. The new identity is layers 4, 12, and 20, direction D2, and the `" true"` minus `" false"` margin. Those choices are fixed before any M30 measurement and are not taken from M29 cell scores.

Primary comparison, also fixed now: holdout `MAE(g) - MAE(g + kappa)`, with a prompt-level paired bootstrap of 5000 draws and seed 23001. The cell-mean baseline B1 is secondary. A magnitude check on UPDATE predictions must pass before any outcome.

## Files

- protocol: `docs/M30_PROTOCOL.md`
- catalog: `scripts/m30_catalog.py`
- disjointness audit: `reports/M30_CATALOG_AUDIT.md`

## Status

DESIGN_LOCKED. No Qwen run.
