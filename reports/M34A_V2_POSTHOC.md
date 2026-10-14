# M34a-v2 post hoc operand features

**POST HOC / EXPLORATORY.** Not in `docs/M34A_V2_PREREG.md`. Cannot change A or C.

Schoolbook multiplication of n-digit `a` by one-digit `b`, least-significant digit first. Carry-out at a position is `floor((digit*b + carry_in)/10)`. `n_carries` = number of positions with nonzero carry-out; `sum_carries` = sum of carry-out values; `product_length` = number of decimal digits of `a*b`.

## (a) Within-level AUROC for correctness (all pools pooled, then mean over levels)

Scores: `answer_logprob` (higher better); `-n_carries`; `-sum_carries`; `-product_length`.

| score | mean AUROC |
| --- | ---: |
| `answer_logprob` | 0.934035 |
| `-n_carries` | 0.531487 |
| `-sum_carries` | 0.541580 |
| `-product_length` | 0.484883 |

Per-level `answer_logprob`:

| level | AUROC |
| ---: | ---: |
| 2 | 0.9822029822029822 |
| 3 | 0.9674122352244112 |
| 4 | 0.9328615496280167 |
| 5 | 0.9470616804273919 |
| 6 | 0.8851648351648351 |
| 7 | 0.8895049904224216 |

## (b) Logistic regression (CAL train / TEST eval)

Level dummies + listed features, standardized on CAL, L2 `0.01` on weights (gradient `2*L2*w`), `4000` steps, `lr=0.05`. Metrics: mean within-level AUROC and mean within-level log-loss on TEST.

| spec | AUROC | log-loss |
| --- | ---: | ---: |
| level only | 0.500000 | 0.507505 |
| + operand features | 0.605906 | 0.503067 |
| + answer_logprob | 0.930901 | 0.361658 |
| both | 0.908521 | 0.358077 |

## Comparison to the independent reproduction (do not retune)

| quantity | this run | reproduction | abs diff |
| --- | ---: | ---: | ---: |
| AUROC logprob | 0.934035 | 0.934000 | 0.000035 |
| AUROC -n_carries | 0.531487 | 0.531000 | 0.000487 |
| AUROC -sum_carries | 0.541580 | 0.542000 | 0.000420 |
| AUROC -product_length | 0.484883 | 0.485000 | 0.000117 |
| logit level AUROC | 0.500000 | 0.500000 | 0.000000 |
| logit level logloss | 0.507505 | 0.513000 | 0.005495 |
| logit +operand AUROC | 0.605906 | 0.659000 | 0.053094 |
| logit +operand logloss | 0.503067 | 0.510000 | 0.006933 |
| logit +logprob AUROC | 0.930901 | 0.931000 | 0.000099 |
| logit +logprob logloss | 0.361658 | 0.359000 | 0.002658 |
| logit both AUROC | 0.908521 | 0.907000 | 0.001521 |
| logit both logloss | 0.358077 | 0.357000 | 0.001077 |

## Claimed

- These descriptive associations on the frozen v2 cache only.

## NOT claimed

- Any change to A or C; privileged access; other models or operations.

## Scope

- Same cache as `reports/M34A_V2_REPORT.md`: Qwen2.5-3B-Instruct, `mul_n1`, levels 2–7.
