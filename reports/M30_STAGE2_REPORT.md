# M30 stage 2

Protocol `M30.PROTOCOL.2`, hash `1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3`.
Catalog `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`. Predictions `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`.
Runner commit `70b052137e54ffcfb4cd714299d9b8f50b78c188`.

## Verdicts

- New identity: `SUPPORTED`
- Anchor: `ANCHOR_SUPPORTED`
- 2x2: Both identities beat `g` on this catalog under the frozen rule.

## Primary scores

| arm | Delta | 95% CI low | 95% CI high | relative reduction | MAE(g) | MAE(g+kappa) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| new_identity | 0.0018781779430759747 | 0.001404496922987164 | 0.0024205371190772792 | 0.8576368215994614 | 0.002189945552446362 | 0.00031176760937038733 |
| anchor | 0.007659008026773032 | 0.004227186764587209 | 0.012326632464262528 | 0.6598186836856698 | 0.01160774651604394 | 0.003948738489270909 |

Bootstrap: 24 prompt scores, 5000 draws, seed 23001, percentiles 2.5 and 97.5.

## Per-cell description

| arm | cell | MAE(g) | MAE(g+kappa) | median |kappa| | median |g| |
| --- | --- | ---: | ---: | ---: | ---: |
| new_identity | M30-D2-L4 | 0.004480923021522661 | 0.0007344694264853994 | 0.004028581432066858 | 0.025595253333449364 |
| new_identity | M30-D2-L12 | 0.0018231048015877604 | 0.00019399385837459704 | 0.0014914526254869998 | 0.029646319337189198 |
| new_identity | M30-D2-L20 | 0.00026580883422866464 | 6.839543251165499e-06 | 0.00021513471438083798 | 0.023147032596170902 |
| anchor | M22.1-D1-L0 | 0.025842321726183098 | 0.011080695413208256 | 0.011207777541130781 | 0.06137063726782799 |
| anchor | M22.1-D1-L8 | 0.006886994602003445 | 0.0005672018742188811 | 0.004469689447432756 | 0.02800502348691225 |
| anchor | M22.1-D1-L15 | 0.0020939232199452817 | 0.000198318180385589 | 0.0015930073568597436 | 0.055400462821125984 |

These per-cell numbers are not a gate.

## Old holdout versus new holdout

| arm | cohort | prompts | Delta | MAE(g) | MAE(g+kappa) |
| --- | --- | ---: | ---: | ---: | ---: |
| new_identity | old | 12 | 0.0019593528446017043 | 0.002262604215906726 | 0.000303251371305022 |
| new_identity | new | 12 | 0.0017970030415502454 | 0.002117286888985998 | 0.00032028384743575263 |
| anchor | old | 12 | 0.00887171714728336 | 0.014656273842168352 | 0.005784556694884991 |
| anchor | new | 12 | 0.006446298906262705 | 0.008559219189919531 | 0.0021129202836568262 |

Old holdout is prompt index 3, 6, 9, 12. New holdout is prompt index 13 through 16. Descriptive only.

## Secondary B1

| arm | MAE(B1) | MAE(B1) - MAE(g+kappa) | 95% CI low | 95% CI high |
| --- | ---: | ---: | ---: | ---: |
| new_identity | 0.002279088942385796 | 0.001967321333015409 | 0.0014806173632543372 | 0.0024953142714433914 |
| anchor | 0.01335540308435965 | 0.00940666459508874 | 0.005078287791682804 | 0.014168236704184286 |

B1 is the UPDATE cell mean of `y - g`. It does not choose the verdict.

Leakage audit passed: `true`.
