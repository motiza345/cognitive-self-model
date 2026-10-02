# M26 independent evidence update execution

The protocol in `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md` is unchanged. The update equation, seeds, sigma values, and verdict map are unchanged.

Verdict: `INCONCLUSIVE`

The recorded holdout classes do not meet the supported rule or the not-supported rule. The correction is unestablished. No second prior, second shrinkage count, or new observable is authorized.

## Primary intervals

Prompt-level means of the three cells, then `paired_mean_ci` with 5000 draws and seed 23001.

- A, MAE(g) - MAE(g + δ_cell): mean `0.00026372789995481554`, interval `[-0.0020563612402649725, 0.0024766029017848816]`, class `CI_INCLUDES_ZERO`
- B, MAE(g + δ_global) - MAE(g + δ_cell): mean `0.0012612554535592411`, interval `[-0.0010591863981880966, 0.003358709975344125]`, class `CI_INCLUDES_ZERO`
- shuffled, MAE(g) - MAE(g + δ_shuffled): mean `-0.002229452439774266`, interval `[-0.0026054393157923243, -0.0017591169559331434]`, class `CI_NEGATIVE`
- leakage_ok: `true`

## Corrections

| quantity | M22.1-D1-L0 | M22.1-D1-L8 | M22.1-D1-L15 |
| --- | ---: | ---: | ---: |
| δ_cell | -0.013824838867510096 | 0.0031455535872802207 | 0.0013218623187378634 |
| δ_shuffled | -0.0007903753228265151 | -0.0008963060172000625 | -0.007670741621465434 |
| τ | 0.0056282132899151465 | 0.0026009809500783456 | 0.0011217373026627966 |
| σ | 0.02029281160603693 | 0.009377970182012517 | 0.004044481362351381 |
| n | 12 | 12 | 12 |

- δ_global: `-0.0032877432026863825`

## Holdout descriptive scores

Sign agreement and coverage are not verdict inputs.

- MAE g: `0.010909644912524517`
- MAE g + δ_cell: `0.010645917012569702`
- MAE g + δ_global: `0.011907172466128942`
- MAE g + δ_shuffled: `0.013139097352298783`
- MAE stored scalar: `0.05077518357170953`
- sign g: `{"correct": 34, "n_nonzero": 36, "n_excluded_zero": 0, "accuracy": 0.9444444444444444, "low": 0.813363293542446, "high": 0.9931996993459773}`
- sign g + δ_cell: `{"correct": 34, "n_nonzero": 36, "n_excluded_zero": 0, "accuracy": 0.9444444444444444, "low": 0.813363293542446, "high": 0.9931996993459773}`
- coverage of g + δ_cell: `{"covered": 34, "n": 36, "rate": 0.9444444444444444, "low": 0.813363293542446, "high": 0.9931996993459773}`

## Per-cell holdout intervals

| cell | A mean | A class | B mean | B class | shuffled mean | shuffled class |
| --- | ---: | --- | ---: | --- | ---: | --- |
| `M22.1-D1-L0` | 1.998526153243604e-05 | `CI_INCLUDES_ZERO` | -0.0010759291393630242 | `CI_INCLUDES_ZERO` | 0.00026345844094217307 | `CI_INCLUDES_ZERO` |
| `M22.1-D1-L8` | 0.0007981596634918041 | `CI_INCLUDES_ZERO` | 0.002578716969531702 | `CI_INCLUDES_ZERO` | -0.00029876867240002084 | `CI_INCLUDES_ZERO` |
| `M22.1-D1-L15` | -2.69612251597937e-05 | `CI_INCLUDES_ZERO` | 0.0022809785305090455 | `CI_POSITIVE` | -0.006653047087864949 | `CI_NEGATIVE` |

## Closure

| artifact | utc | sha256 |
| --- | --- | --- |
| `reports/m26_raw/update_predictions.json` | `2026-10-02T21:17:47Z` | `70e1b4e15d440bb27a053ec4480ce8b2e54d127b6d2813267d0ba326dc96f665` |
| `reports/m26_raw/update_outcomes.json` | `2026-10-02T21:18:54Z` | `08794424093129a01a128a49db10f24510cc3871c9d8e05df6a95780e65a7eda` |
| `reports/m26_raw/corrections.json` | `2026-10-02T21:18:54Z` | `3e1839931c82dd2fc2a682361d97fb3e8ca11c0b344ac7013ecb0330af947122` |
| `reports/m26_raw/holdout_predictions.json` | `2026-10-02T21:19:30Z` | `edce9dd998ab377c3a75871212d853fdcc186edb44f0d78e9aab960066cc71b9` |
| `reports/m26_raw/holdout_outcomes.json` | `2026-10-02T21:20:32Z` | `205e9de98c9d8167797906ce52afa8abcd13f7ef4fac93b6695cfec7763c798d` |

## Provenance

- model: `Qwen/Qwen2.5-0.5B`
- revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- direction sha256: `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`
- catalog sha256: `015c3d41855da5fa630c382fe0e4a9ea63c868e12e4a1243e104e1c4e371ea6f`
- protocol version: `M26.INDEPENDENT_EVIDENCE_UPDATE.1`
- git commit at execution: `55c3ee5688b1c040f6f32f04382b129a5a432304`
- freeze commit: `a81388248d13b5cd602ee6bd786fe871d7909fbc`
- git status at execution: `clean`
