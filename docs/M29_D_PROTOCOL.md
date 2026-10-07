# M29-D — Pre-Outcome Residual Predictor Design Lock

**Status:** DESIGN_LOCKED  
**Branch:** `research/m29-d-design`  
**Parent:** `10e5f9a5a67661f65a42654f4e1ba00f7bd9383d`  
**Purpose:** Design-only. No Qwen scored run is permitted by this document.

## 1. Scientific question

Can a genuinely pre-outcome, model-internal measurement predict the residual error of the frozen M22.1/M24 mechanism response?

For a frozen mechanism response prediction (g) and observed post-intervention effect (y):

[
r = y-g
]

M29 asks whether a measurement available **before the scored intervention outcome** can predict (r).

The primary candidate is not selected by searching features on holdout data. It is fixed here from the local Taylor expansion of the frozen response functional.

## 2. Frozen objects inherited from M22–M28

M29 does not refit or redefine:

- mechanism/intervention identity;
- intervention direction;
- intervention amplitude (alpha);
- response functional used to obtain (g);
- observed-effect definition (y);
- train/update/validation/holdout semantics.

If the inherited artifact uses a mathematically equivalent convention in which (g) already includes (alpha), the implementation must record that convention explicitly before execution. No silent rescaling is allowed.

## 3. Primary pre-outcome measurement

Let (F(t)) be the frozen scalar response functional evaluated when moving from the unperturbed model along the frozen intervention direction (d):

[
F(t)=F(x+t d)
]

The inherited mechanism prediction is the first-order term:

[
g = alpha F'(0)
]

M29 defines the pre-outcome curvature measurement:

[
kappa = rac{alpha^2}{2}F''(0)
]

The candidate residual predictor is:

[
hat r_{mathrm{M29}}=kappa
]

Therefore the corrected prediction is:

[
hat y_{mathrm{M29}}=g+kappa
]

### Why this candidate is frozen

If the actual response is locally twice differentiable,

[
F(alpha)-F(0)
=
alpha F'(0)
+
rac{alpha^2}{2}F''(0)
+
O(alpha^3)
]

so the first-order mechanism response (g) has a specific, mechanistically motivated source of residual error: local curvature.

This is a hypothesis, not an assumption that the Qwen residual must be curvature-driven.

## 4. Measurement admissibility

(kappa) is admissible only if:

1. it is computed before the scored intervention outcome is observed;
2. it uses no post-intervention behavioral/output observation;
3. it does not use the observed residual (r);
4. it does not use holdout outcomes for feature selection, calibration, threshold selection, or sign selection;
5. it is deterministic given the frozen model, intervention direction, amplitude, and pre-outcome input;
6. its numerical procedure is frozen before scored evaluation;
7. it is not reconstructed from a previously consumed M22–M26 outcome catalog.

The implementation may use automatic differentiation/Hessian-vector products at the unperturbed point. It must not obtain (kappa) by fitting a polynomial to observed intervention outcomes.

## 5. Prediction target

Primary target:

[
r_i = y_i-g_i
]

Primary comparison:

[
|r_i-hat r_{mathrm{M29},i}|
quad	ext{vs}quad
|r_i-hat r_{mathrm{baseline},i}|
]

The M29 prediction is frozen before observing the corresponding (y_i).

## 6. Baselines

Three baselines are frozen:

### B0 — Global residual mean

Mean residual from UPDATE only.

### B1 — Mechanism-cell residual mean

Mean residual for the exact frozen mechanism/intervention cell from UPDATE only.

This is the strongest simple non-mechanistic baseline and is the primary baseline.

### B2 — Permuted M29 control

The frozen M29 (kappa) values are permuted within the evaluation catalog using a preregistered seed.

This tests whether any gain is attributable to the pairing between the pre-outcome measurement and residual.

No alternative feature may replace (kappa) after seeing validation/holdout results.

## 7. Data partitions

M29 requires a completely new catalog disjoint from every previously consumed M22.1–M26 scored catalog.

Three partitions are frozen before scored execution:

- UPDATE
- VALIDATION
- HOLDOUT

Rules:

- UPDATE may estimate B0/B1 only.
- VALIDATION may test pipeline correctness and numerical stability.
- HOLDOUT is evaluated exactly once after all protocol choices are frozen.
- No holdout result may change (kappa), its sign, scaling, model, threshold, partitioning, or baseline definition.

If a genuinely disjoint catalog cannot be constructed, M29-D is not executable and must stop at design.

## 7A. Frozen M29-D catalog

The new pre-outcome catalog is frozen at:

- catalog module: `scripts/m29d_catalog.py`
- catalog SHA-256: `d6ecab612a2214d947be11c82671612f883c0d13e479b07e8cbaa09b635d4d77`
- total prompts: 36
- UPDATE: 12
- VALIDATION: 12
- HOLDOUT: 12
- families: completion, syntax, instruction
- disjointness audit: `reports/M29_D_CATALOG_AUDIT.md`

The audit checked exact and normalized text/identifier overlap against M22.1, M23-G, POST-M23 single-measurement, feasibility-gate, M24, and M26 catalogs. All overlap counts were zero.

No Qwen outcome, residual, or holdout statistic was loaded during this audit.

## 8. No feature search

The following are explicitly forbidden:

- feature sweeps;
- correlation-based feature selection;
- choosing among gradient norm, activation norm, margin, entropy, curvature, etc. after seeing residuals;
- fitting a flexible residual model on validation and selecting the best one;
- selecting the sign of (kappa) from observed residuals;
- adding a second feature because the primary candidate is weak.

A failure is a scientific result, not a reason to expand the feature set inside M29.

## 9. Primary statistical claim

Primary claim:

> The pre-outcome curvature measurement (kappa) predicts residual error better than the frozen cell-mean baseline B1 on an untouched holdout catalog.

Primary effect:

[
Delta =
MAE(B1)-MAE(M29)
]

Acceptance requires:

1. (Delta>0);
2. paired 95% bootstrap CI lower bound (>0);
3. M29 beats B1 on the same holdout rows;
4. M29 is not explained by the permutation control;
5. leakage audit passes.

A positive point estimate with a CI containing zero is **INCONCLUSIVE**, not supported.

## 10. Secondary analyses

Report, but do not gate the primary claim on:

- correlation between (kappa) and residual;
- residual sign accuracy;
- M29 corrected-effect MAE;
- B0 comparison;
- B2 permutation control;
- magnitude-stratified error;
- numerical stability of (kappa);
- per-cell results.

These analyses cannot be used post hoc to redefine the primary claim.

## 11. Synthetic gate — mandatory before Qwen

Before any Qwen scored run, the implementation must pass synthetic tests where the ground-truth response function is known.

Required synthetic families:

### S1 — Pure linear

[
F(t)=a+bt
]

Expected:

[
r=0,quadkappa=0
]

M29 must not manufacture a correction.

### S2 — Quadratic

[
F(t)=a+bt+ct^2
]

Expected:

[
g=alpha b,quad r=alpha^2c,quadkappa=alpha^2c
]

M29 must recover the residual to numerical tolerance.

### S3 — Cubic

[
F(t)=a+bt+ct^2+dt^3
]

M29 should recover the quadratic component but leave the cubic remainder. This tests that the candidate is a second-order predictor, not an oracle.

### S4 — Sign reversal

Use positive and negative (c).

The pipeline must preserve the sign of the curvature contribution without any outcome-driven sign selection.

### S5 — Zero curvature

Use nonlinear response terms whose second derivative at zero is zero.

M29 must return approximately zero and must not use higher-order residual structure automatically.

### S6 — Numerical scaling

Repeat S2/S3 across small and moderate (alpha).

The implementation must record numerical error and must not silently switch to a different estimator when the measurement becomes difficult.

## 12. Synthetic acceptance criteria

All must pass:

- analytic-vs-measured (kappa) error within preregistered numerical tolerance;
- no post-outcome values enter measurement computation;
- S1 correction remains approximately zero;
- S2 residual recovery passes;
- S3 residual is correctly incomplete;
- S4 sign is preserved;
- S5 zero-curvature behavior passes;
- deterministic repeat gives identical measurement within numerical tolerance;
- leakage audit passes.

If synthetic tests fail, **no Qwen run is allowed**.

## 13. Numerical implementation lock

Preferred estimator:

- automatic differentiation;
- Hessian-vector product in the frozen intervention direction;
- evaluation at the unperturbed model state.

Finite-difference fallback is allowed only if autodiff is impossible and must be separately frozen before use. It may not be selected after inspecting Qwen results.

The implementation must record:

- dtype;
- device;
- numerical tolerance;
- autodiff method;
- intervention direction normalization;
- (alpha);
- scalar response definition;
- whether gradients are taken before/after any frozen model preprocessing.

## 14. Leakage boundary

The following are unavailable to the measurement function:

- observed intervention outcome (y);
- residual (r);
- post-intervention activation/output;
- validation/holdout residual statistics;
- ground-truth labels that are not available to the agent before intervention.

The measurement may use only the frozen model and pre-outcome input/intervention specification.

## 15. Reproducibility

The scored run must save:

- protocol hash;
- catalog hash;
- model revision;
- intervention identity;
- measurement definition;
- numerical configuration;
- partition hashes;
- random seeds;
- raw per-row (kappa);
- raw (g);
- raw observed (y);
- residual;
- B0/B1 predictions;
- permuted B2 predictions;
- per-row paired errors;
- bootstrap seed and draw count;
- leakage audit;
- verdict.

No result file may overwrite a previous catalog.

## 16. Stop conditions

M29-D stops without Qwen if:

1. no genuinely disjoint catalog is available;
2. inherited response functional cannot be reconstructed without consuming old evidence;
3. (kappa) cannot be computed without outcome leakage;
4. synthetic tests fail;
5. numerical implementation is not deterministic enough for the frozen tolerance.

None of these conditions may be repaired by changing the scientific question after seeing results.

## 17. Scientific interpretation

### If M29 is supported

The correct claim is limited to:

> A pre-outcome, mechanism-derived curvature measurement predicts residual error of the frozen mechanism response on the new holdout catalog.

This would establish a first concrete form of **state-dependent predictive self-modeling**, but would not yet establish general self-modeling, active experimentation, self-repair, or self-improvement.

### If M29 is inconclusive

The candidate does not establish predictive residual information. No stronger claim follows.

### If M29 is clearly negative

The result supports rejecting local second-order curvature as the explanatory pre-outcome signal for this mechanism/catalog. It does not prove that no pre-outcome self-state exists.

## 18. M29-D freeze rule

After this protocol is committed:

- no Qwen scored run before synthetic gate;
- no candidate-feature expansion;
- no holdout inspection before final execution;
- no threshold tuning;
- no baseline replacement;
- no post hoc sign reversal;
- no redefinition of residual.

M29-D is a hypothesis test of one specific mechanistic pre-outcome signal, not a search for a feature that happens to work.
