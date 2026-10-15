# M30 post hoc: does |kappa| predict the remaining error?

**POST HOC / EXPLORATORY.** Not pre-registered. Cannot change any verdict. A finite-difference probe
(S2) computes the same curvature, so this gives no privileged access.

Holdout rows only. Spearman rho between the feature and `abs_error_g_plus_kappa`; 95% percentile CI,
5000 bootstrap draws over rows, seed 23001.

| arm | feature | rows | rho | CI low | CI high |
| --- | --- | ---: | ---: | ---: | ---: |
| new_identity | abs(kappa) | 72 | 0.764 | 0.660 | 0.833 |
| new_identity | abs(g) | 72 | 0.174 | -0.065 | 0.389 |
| anchor | abs(kappa) | 72 | 0.605 | 0.443 | 0.723 |
| anchor | abs(g) | 72 | 0.271 | 0.054 | 0.465 |

## Risk-coverage

Mean remaining absolute error among the kept fraction of holdout rows with the smallest feature value.

| arm | feature | 25% | 50% | 75% | 100% |
| --- | --- | ---: | ---: | ---: | ---: |
| new_identity | abs(kappa) | 0.000027 | 0.000072 | 0.000154 | 0.000312 |
| new_identity | abs(g) | 0.000220 | 0.000203 | 0.000254 | 0.000312 |
| anchor | abs(kappa) | 0.000303 | 0.002158 | 0.001912 | 0.003949 |
| anchor | abs(g) | 0.001350 | 0.001426 | 0.002036 | 0.003949 |
