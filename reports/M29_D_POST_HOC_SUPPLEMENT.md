# M29-D post-hoc supplement

**Status:** POST_HOC
**Source:** `reports/m29d_raw/VERDICT.json` holdout `paired_rows` only
**Script:** `scripts/m29d_post_hoc_supplement.py`

These analyses were not pre-registered. They cannot change the verdict in `reports/m29d_raw/VERDICT.json`. The preregistered primary comparison remains MAE(B1) minus MAE(M29) on the holdout rows, with the row-level paired bootstrap already stored in that file.

Nothing here refits kappa, changes its sign or scale, or selects a new baseline.

## Holdout description

- Rows: 36
- MAE of g alone, mean(|y - g|): `0.01248311497167581`
- MAE of g + kappa, mean(|y - (g + kappa)|): `0.004513239650502025`
- Delta versus g, MAE(g) - MAE(g + kappa): `0.00796987532117378`
- Prompt-level paired bootstrap: 5000 draws, seed 23001, 12 prompts, three cells averaged within each prompt
- Prompt-level mean of paired differences: `0.007969875321173782`
- Prompt-level 95% interval: `[0.004742243285161546, 0.01148920214715569]`
- Prompt-level interval class: `CI_POSITIVE`
- Pearson correlation of kappa and residual: `0.8639837450934315`
- Sign agreement, sign(kappa) against sign(residual): 34/36

The prompt-level mean equals the row-level Delta because every prompt has the same three cells. The interval is still the prompt-level bootstrap, not the row-level interval stored in the verdict.

## Per cell

| cell | MAE(g) | MAE(g+kappa) | median(|kappa|)/median(|r|) |
| --- | ---: | ---: | ---: |
| `M22.1-D1-L0` | 0.02766812701399128 | 0.01248975100073343 | 0.854869480950888 |
| `M22.1-D1-L8` | 0.005958667413021128 | 0.0008334555313922465 | 1.085331999567653 |
| `M22.1-D1-L15` | 0.003822550488015016 | 0.0002165124193804028 | 0.9781572212719309 |

The ratio is median(|kappa|) divided by median(|residual|), not the median of the per-row ratios.

## What this file does not do

It does not replace `SUPPORTED`. It does not authorize a different primary baseline. A positive description against g was not an acceptance criterion when the holdout was scored.
