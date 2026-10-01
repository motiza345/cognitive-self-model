# M23-E leakage audit

Rules fixed before the M23-E forwards. The runner records each check in `reports/M23_E_RESULTS.json`.

## Allowed in the fit

- Repeat 0 of the nine train prompt ids listed in the preregistration.
- At alpha `+1`, or, in a separate fit, at alpha `+2`.
- One predictor uses `pre_dot` only. The other uses baseline margin only. The constant uses the train effects only.

## Not allowed in the fit

- Evaluation prompt ids.
- Repeat 1.
- The other alpha.
- Regime, prompt text, prompt id, or an embedding.
- Intervened output, observed effect as a feature, post-intervention activation, or a future residual.
- Coefficients or correlations copied from M23, the diagnostic, or M23-D2.
- A second specification chosen after evaluation error is known.

## Ordering

The split file is written while the measurement file is absent. Forwards start after that. Train coefficients are written before evaluation absolute errors are computed. Coefficients are not updated afterward.

## Replication

The evaluation ids include `completion-02`, `instruction-02`, and `syntax-02`, which are not replication prompts. A check fails if every evaluation id is a replication id.

## Repeats

Both repeats use the same prompt, direction seed `22101`, model, hook, and alpha. No noise source is added.

## If a check fails

`leakage_ok` is false. Every identifiability contrast is reported as `INCONCLUSIVE` with note `LEAKAGE`. Scaling numbers can still be shown as descriptions of the forwards.
