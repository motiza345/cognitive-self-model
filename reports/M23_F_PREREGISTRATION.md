# M23-F preregistration

Frozen before any M23-F forward.

Historical M23 remains `INCONCLUSIVE`. No historical artifact is rewritten. No predictor is fit. No candidate is ranked by correlation, absolute error, or self-model score.

## Question

Does any existing intervention leave a falsifiable contrast for a later predict → violate → revise → held-out predict test, without spending a split that is already inspected?

The current published pattern is a shared response to alpha plus a small prompt remainder. This file does not assume that pattern forces a new intervention. The screen measures the existing cells and applies the rules below.

## Screening set

The six M22.1 discovery prompts only:

- `completion-01`, `completion-04`
- `instruction-01`, `instruction-04`
- `syntax-01`, `syntax-04`

Validation and replication prompts are not forwarded. They are the confirmatory splits. Measuring them in this audit would spend the only cells that might still have been unmeasured.

## Candidates

Existing cells only.

- D1 and D2 at layers `0`, `8`, `15`, and `23`.
- D3, D4, D5, D6, D7, and D8 at layer `23` only.

No other layer, seed, hook, or target is added. Alphas are `+1` and `+2` only. Two repeats. Same prompt, seed, and alpha on both repeats. No noise is added.

`pre_dot` is recorded and is not correlated with the effect. Baseline margin is recorded and is not used as a feature.

## Structural rules

These rules do not use a numeric pass cutoff.

For one candidate at alpha `+1`, repeat 0:

- Same sign: every screening effect is strictly positive, or every one is strictly negative.
- Sign disagreement: both signs appear.
- Numerical spread: maximum effect is not equal to minimum effect. This is reported. It is not, by itself, enough to call the current intervention falsifiable. M23-E already showed a small nonzero range around one shared response.

A candidate contradicts a shared-sign prediction if it has sign disagreement.

A candidate contradicts "this is the same response as D1 at layer 23" if, on at least one screening prompt, its alpha `+1` effect is not equal to that D1 effect.

Alpha scaling is descriptive: the six ratios `effect(+2) / effect(+1)`, their mean, and their sample standard deviation. A ratio is not a selection score. Exact doubling is not required, and a departure from 2 is not a pass.

Repeatability: the maximum absolute difference between repeat 0 and repeat 1. Equality means the runtime repeated. It does not estimate stochastic noise.

## What is not allowed

- Fitting a line, a correlation, or a belief.
- Sorting candidates by range, variance, absolute mean, or ratio and keeping the top one.
- Treating sign disagreement on the discovery screen as permission to run validation or replication in this task.
- Upgrading any status to `VERIFIED`.

## Inspection map

| candidates | discovery | validation | replication |
| --- | --- | --- | --- |
| D1–D8 at layer 23 | inspected | inspected | inspected |
| D1 and D2 at layers 0, 8, 15 | inspected | inspected by MRSM Q | not the Q holdout |

A clean future experiment on this catalog requires both confirmatory splits to be unmeasured. Leftover replication, after validation was already used, is not counted as a reserved holdout.

## Status map

`current_intervention_status` is `CURRENT_INTERVENTION_INSUFFICIENT` if D1 at layer 23 has one shared sign on this screen. Otherwise that field is `INCONCLUSIVE`.

`audit_status`:

- `MULTIPLE_CANDIDATES_AVAILABLE` if two or more candidates both contradict a shared-sign prediction or contradict equality with D1 at layer 23, and both have a clean future experiment under the rule above.
- `CANDIDATE_INTERVENTION_AVAILABLE` if exactly one such candidate has a clean future experiment.
- `NO_SUITABLE_EXISTING_INTERVENTION` if none do.
- `INCONCLUSIVE` only if the screen cannot compute the signs.

Candidates that meet a structural contrast but are already confirmatory-inspected are retained in the report and are not deleted for having a smaller effect. They do not count as available for a clean experiment.

## Prior result that this screen does not reopen

M23-E: no held-out evidence that `pre_dot` or baseline margin identifies D1 layer-23 variation, deterministic repeats, and a mean alpha ratio near `2.017`. This screen does not fit that comparison again.
