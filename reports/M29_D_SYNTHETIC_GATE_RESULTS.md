# M29-D Synthetic Gate Results

**Status:** PASS  
**Branch:** `research/m29-d-design`  
**Protocol:** `docs/M29_D_PROTOCOL.md`  
**Gate spec:** `docs/M29_D_SYNTHETIC_GATE.md`

## Execution lock

Before execution, the numeric synthetic tolerance was fixed at **1e-10** in float64 because the design documents require a preregistered numerical tolerance but did not specify its numeric value.

Fixed alpha grid for S6:
`[0.1, 0.25, 0.5, 1.0]`

Measurement:
`kappa = alpha^2 / 2 * F''(0)`

Implementation:
PyTorch automatic differentiation, float64, CPU, evaluation at the unperturbed point.

Environment:
- PyTorch 2.10.0+cpu
- Python 3.13.5
- Linux x86_64

## Results

### S1 — Pure linear

`F(t)=a+bt`

Maximum absolute measured kappa across alpha = **0.0**

**PASS**

### S2 — Quadratic

`F(t)=a+bt+ct^2`

Maximum absolute error:
- measured kappa vs analytic `alpha^2*c`: **0.0**
- recovered residual vs measured kappa: **1.1102230246251565e-16**

**PASS**

### S3 — Cubic

`F(t)=a+bt+ct^2+dt^3`

Maximum absolute error:
- measured kappa vs analytic quadratic component: **0.0**
- `residual - kappa` vs analytic cubic remainder `d*alpha^3`: **4.2365199889871086e-17**

The cubic remainder was not absorbed into kappa.

**PASS**

### S4 — Sign reversal

For `c=+0.37`, alpha=0.5:
- measured kappa = **+0.0925**
- analytic kappa = **+0.0925**

For `c=-0.37`, alpha=0.5:
- measured kappa = **-0.0925**
- analytic kappa = **-0.0925**

No outcome-driven sign selection was used.

**PASS**

### S5 — Zero local curvature

A nonlinear cubic response with `F''(0)=0` was used.

Maximum absolute kappa across alpha = **0.0**

The higher-order nonlinear residual was not automatically converted into a curvature correction.

**PASS**

### S6 — Numerical scaling

Fixed alpha grid:
`0.1, 0.25, 0.5, 1.0`

Maximum measured-kappa error against analytic quadratic component:
- alpha=0.1: **0.0**
- alpha=0.25: **0.0**
- alpha=0.5: **0.0**
- alpha=1.0: **0.0**

For the cubic family the same maximum error was **0.0** at every alpha.

**PASS**

## Determinism

Same frozen function and alpha were measured five times.

Maximum repeat-to-repeat difference:
**0.0**

**PASS**

## Leakage audit

The measurement function accepts only the frozen response function, alpha, and pre-outcome specification. A deliberate test attempted to pass an observed outcome into the measurement path; the implementation rejected it.

Leakage test:
**PASS**

No observed outcome or residual was used to calculate kappa in S1-S6.

## Gate verdict

| Gate | Result |
|---|---|
| S1 Linear | PASS |
| S2 Quadratic | PASS |
| S3 Cubic | PASS |
| S4 Sign reversal | PASS |
| S5 Zero local curvature | PASS |
| S6 Numerical scaling | PASS |
| Determinism | PASS |
| Leakage | PASS |

# FINAL VERDICT: M29-D SYNTHETIC GATE PASSED

This clears the synthetic prerequisite only.

It does **not** establish that curvature predicts residuals on Qwen, and it does not justify changing the frozen M29-D candidate.

**Next permitted step:** construct and audit a genuinely disjoint M29-D UPDATE/VALIDATION/HOLDOUT catalog. Only after catalog disjointness and execution artifacts are frozen may the Qwen scored run occur.
