# M29-D Synthetic Gate Specification

This document is executable-design specification only. It contains no Qwen result.

## Required tests

S1 Linear:
F(t)=a+bt
Expected kappa=0.

S2 Quadratic:
F(t)=a+bt+ct^2
Expected g=alpha*b and residual=alpha^2*c.
Expected kappa=alpha^2*c.

S3 Cubic:
F(t)=a+bt+ct^2+dt^3
Expected kappa=alpha^2*c.
The cubic remainder must not be silently absorbed.

S4 Sign:
Run S2 with c>0 and c<0. The measured kappa must preserve sign.

S5 Zero local curvature:
Choose a smooth nonlinear F with F''(0)=0. Measured kappa must be approximately zero.

S6 Scale:
Repeat S2/S3 over a fixed alpha grid selected before execution. The implementation must report numerical error and deterministic repeatability.

## Mandatory leakage test

The synthetic measurement function receives only:
- F and its frozen parameters,
- alpha,
- direction specification,
- pre-outcome state.

It must not receive observed y or residual.

A test must fail if outcome data are passed into the measurement function.

## Gate

All six families plus leakage and determinism tests must pass before any Qwen scored run.
