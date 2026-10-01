# M23-D2 leakage audit

Rules fixed before the evaluation forward. The runner records whether they held in `reports/M23_D2_RESULTS.json`. This file is not a score.

## Predictor B may use

- `regime_id` from the frozen M22.1 prompt record
- the six already saved validation D1 effects, grouped by that same regime

## Predictor B may not use

- prompt id as a feature (two prompts in a regime must receive the same prediction)
- prompt text or an embedding
- `pre_dot`
- baseline margin
- activation
- observed effect, intervened margin, or future residual
- replication rows
- discovery outcomes, which do not exist as rows yet

## Predictor A may use

- the mean of the same six validation D1 effects, and nothing else

## Ordering

The evaluation prediction file is written while the evaluation outcome file is absent. Forwards start only after that file exists. The script aborts if an evaluation outcome file is already present.

## Checks the runner must record

- `predictions_before_outcomes`
- `prediction_B_constant_within_regime`
- `prediction_A_constant_across_regimes`
- `replication_prompts_absent`
- `no_outcome_field_in_prediction_rows`
- `evaluation_prompt_ids_match_manifest`

Any false check sets `leakage_ok` false. The comparison is then not interpretable as support for regime information, and the reported result is `INCONCLUSIVE` with failure note `LEAKAGE`.
