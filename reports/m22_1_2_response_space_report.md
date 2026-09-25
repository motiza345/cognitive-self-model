# M22.1.2 response-space identifiability

Descriptive label: `INSUFFICIENT_EVIDENCE`.

The audit reuses the frozen M22.1.1 measurements. It does not run a new model forward.
M22.1 remains `CANDIDATE`. M22.2 remains not authorized.

## Frozen estimator

- Primary estimator: `ridge_als` rank 1, ridge `0.01`.
- Decision model: `B4_rank1`.
- Secondary SVD is reported and is not used for the label.
- Manifest: `978e715812bc8291077e8e32447f1d20db3da78f675d3b0501c38af49b98ebaa`.

## Blocked-cell MAE

Means are over the five pre-registered mask seeds. Lower is better.

| model | validation | replication |
| --- | ---: | ---: |
| B0_global_mean | 0.028062 | 0.028516 |
| B1_prompt_mean | 0.031685 | 0.032384 |
| B2_direction_mean | 0.003119 | 0.003383 |
| B3_additive | 0.002930 | 0.003087 |
| B4_rank1 | 0.003555 | 0.003811 |
| B5_rank2 | 0.003629 | 0.003849 |
| B6_rank3 | 0.003629 | 0.003849 |
| B7_cell_permutation_rank1 | 0.044227 | 0.045783 |

Median and standard deviation of masked-cell MAE across mask seeds:

| model | validation median | validation std | replication median | replication std |
| --- | ---: | ---: | ---: | ---: |
| B0_global_mean | 0.029121 | 0.002689 | 0.028503 | 0.002012 |
| B2_direction_mean | 0.003286 | 0.000519 | 0.003843 | 0.000890 |
| B4_rank1 | 0.003205 | 0.000860 | 0.003223 | 0.001139 |

## Negative controls and secondary audits

### validation

- Seeds on which rank-1 beats both baselines by the frozen margin: 1 of 5.
- Normalized MAE versus global mean: 0.1267.
- Improvement over direction mean (baseline minus rank-1): -0.000436.
- Improvement over additive model: -0.000625.
- Global-permutation rank-1 MAE: 0.044129.
- Sign-permutation rank-1 MAE: 0.003142.
- Magnitude-only rank-1 MAE: 0.006475.
- Leave-one-prompt rank-1 MAE: 0.003394.
- Leave-one-regime rank-1 MAE: 0.003323.
- Calibration rank-1 / constant / global MAE: 0.007790 / 0.003758 / 0.031090.
- Symmetry mean |Δ(+1)+Δ(-1)|: 0.000451.
- Per-prompt blocked rank-1 MAE: completion-02 0.007506, completion-05 0.001476, instruction-02 0.004160, instruction-05 0.003367, syntax-02 0.001727, syntax-05 0.003094.
- Per-direction blocked rank-1 MAE: D1 0.002122, D2 0.011844, D3 0.001698, D4 0.003238, D5 0.002588, D6 0.002128, D7 0.001289, D8 0.003201.

### replication

- Seeds on which rank-1 beats both baselines by the frozen margin: 2 of 5.
- Normalized MAE versus global mean: 0.1337.
- Improvement over direction mean (baseline minus rank-1): -0.000429.
- Improvement over additive model: -0.000725.
- Global-permutation rank-1 MAE: 0.044884.
- Sign-permutation rank-1 MAE: 0.002873.
- Magnitude-only rank-1 MAE: 0.006015.
- Leave-one-prompt rank-1 MAE: 0.003178.
- Leave-one-regime rank-1 MAE: 0.003581.
- Calibration rank-1 / constant / global MAE: 0.007410 / 0.003387 / 0.031727.
- Symmetry mean |Δ(+1)+Δ(-1)|: 0.000557.
- Per-prompt blocked rank-1 MAE: completion-03 0.005899, completion-06 0.001448, instruction-03 0.004025, instruction-06 0.003567, syntax-03 0.004417, syntax-06 0.003512.
- Per-direction blocked rank-1 MAE: D1 0.001553, D2 0.012090, D3 0.002729, D4 0.004685, D5 0.002222, D6 0.001667, D7 0.001246, D8 0.004171.

## Interpretation limit

Held-out cell prediction tests whether the low-rank description is useful inside this frozen panel.
It does not identify a causal circuit, a direction's semantic meaning, or a SelfModel.
An unseen direction without a calibration cell is outside the claim.
