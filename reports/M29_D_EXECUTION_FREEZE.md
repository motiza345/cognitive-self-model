# M29-D Execution Freeze

**Status:** EXECUTION_FROZEN — pre-outcome dry-run only  
**Branch:** `research/m29-d-design`

## Frozen inherited mechanism

From the committed M24/M22.1 artifacts:

- Model: `Qwen/Qwen2.5-0.5B`
- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- d_model: 896
- intervention direction: D1
- direction seed: 22101
- direction SHA-256: `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`
- alpha: +1.0
- outcome observable: last-position logit margin
- positive token: 9834 (" yes")
- negative token: 902 (" no")
- frozen cells:
  - M22.1-D1-L0 -> `blocks.0.hook_resid_post`
  - M22.1-D1-L8 -> `blocks.8.hook_resid_post`
  - M22.1-D1-L15 -> `blocks.15.hook_resid_post`

The inherited response rule is exactly:

`g = F'(0) = <grad_x F, d>`

with no additional coefficient.

## Frozen M29 measurement

For each prompt/cell:

`F(t) = margin(model with t*d added to the last-token residual at the frozen hook)`

At the unperturbed state:

`g = F'(0)`

`F''(0) = d^T H d`

`kappa = alpha^2 / 2 * F''(0)`

`r_hat_M29 = kappa`

`y_hat_M29 = g + kappa`

The implementation uses PyTorch automatic differentiation with a Hessian-vector product. It does not run finite differences and does not fit a polynomial to outcomes.

Implementation artifact:

`scripts/m29d_measurement.py`

## Numerical lock

- dtype: float32
- device: CUDA if available, otherwise CPU
- tolerance: 1e-6
- alpha: +1.0
- autodiff: `torch.autograd.grad(..., create_graph=True)`
- finite difference: forbidden
- outcome input to measurement: forbidden
- residual input to measurement: forbidden

## Pre-outcome order

For each partition:

1. Load frozen model/revision.
2. Resolve the frozen outcome token IDs.
3. Compute g and kappa at the unperturbed state.
4. Persist prediction-only rows.
5. Freeze prediction-row hashes.
6. Only after prediction rows are persisted may scored intervention outcomes be measured.
7. Join predictions with outcomes only in the scoring stage.

The validation partition is processed before its outcomes; the holdout partition is not inspected until all protocol choices are frozen.

## Dry-run result

A structural dry-run was completed against the frozen M29-D catalog:

- catalog rows: 36
- UPDATE: 12
- VALIDATION: 12
- HOLDOUT: 12
- cells per scored prompt: 3
- expected validation prediction rows: 36
- expected holdout prediction rows: 36
- expected total scored prediction rows: 72
- outcome rows are not part of the measurement interface
- no UPDATE/VALIDATION/HOLDOUT outcome has been loaded

The dry-run verifies the execution plan and row cardinalities only. It does **not** claim that Qwen autodiff has been numerically executed.

## Critical reproducibility point

The M24 artifact proves the inherited direction and intervention definition through a SHA-256, but the vector itself is regenerated deterministically from the frozen seed 22101 and verified against that hash. No direction is guessed or copied from an outcome.

## Execution gate

All pre-Qwen conditions currently pass:

- design lock: PASS
- synthetic S1-S6: PASS
- leakage: PASS
- determinism: PASS
- catalog disjointness: PASS
- inherited mechanism identity: PASS
- intervention direction provenance: PASS
- M29 measurement implementation frozen: PASS
- pre-outcome execution order: PASS

**The only remaining operation is the actual Qwen pre-outcome prediction run.**

That run requires a Python environment with the repository and Qwen/TransformerLens model available (normally the project's CUDA/Colab or Cursor execution environment). It must produce prediction-only artifacts before any intervention outcome is generated.

No holdout outcome has been consumed.
