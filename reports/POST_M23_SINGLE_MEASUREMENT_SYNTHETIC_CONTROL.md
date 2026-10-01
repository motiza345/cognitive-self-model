# Post-M23 single-measurement synthetic positive control

This is a harness check. It is not Qwen evidence. It does not load the model, and it does not use an inspected outcome.

The functions are `fit_candidate`, `predict_baseline`, and `predict_candidate` in `scripts/validate_post_m23_single_measurement.py`. They are the baseline and the candidate from the preregistration.

## Planted train

Measurements: `0, 1, 2, 3`. Effects: `0, 0.2, 0.4, 0.6`. The planted relationship is `effect = 0.2 * pre_dot`.

The fit returns mean `0.3`, measurement mean `1.5`, and slope `0.2`. The candidate at the train measurement mean equals `0.3`, the scalar mean.

## Planted holdout

Measurements: `4, 5, 6, 7`. Effects: `0.8, 1.0, 1.2, 1.4`.

Baseline absolute errors: `0.5, 0.7, 0.9, 1.1`. Candidate absolute errors: `0, 0, 0, 0`. Paired differences, baseline minus candidate: `0.5, 0.7, 0.9, 1.1`.

`paired_mean_ci` on those four differences, 5000 draws, seed `23001`: mean `0.8`, low `0.6`, high `1.0`, class `CI_POSITIVE`. The harness result is `HARNESS_PASS`.

## Degenerate measurement

Train measurements `1, 1, 1, 1` with effects `1, 2, 3, 4` have measurement variance zero. The slope is defined to be `0`. The candidate prediction equals the baseline prediction.

## What the pass means

The same interface that a later execution would use can detect a measurement that linearly determines the effect, and it can return the scalar mean when the measurement does not vary. The planted numbers are not a layer-15 residual and not an intervention effect.
