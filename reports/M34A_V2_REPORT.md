# M34A-v2 report

**Primary verdict (V3, n_h=10):** A=`SUPPORTED_WEAK` · C=`INTERNAL_VALUE` · GATE_A=`True` · GATE_C=`True`

- cache sha256: `e0eb10cd12bbf25b4e6a258ab1391e24e7110a81611c0e9dc87e0914008c5425`
- model: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- device/dtype: `cuda` / `float16`
- torch/transformers: `2.11.0+cu130` / `5.18.0`
- collection git commit: `2c96f40c32975cdcf4fff569ddd2a353aa4a91ff`
- analysis git commit: `2a63d27290f74f639510ac18ee106ca70b1161e2` (clean tree required)
- GATE_A gap U(oracle_level)-U(global) = 0.058333 (need >= 0.05)
- self-global: mean=0.025278 CI=[0.011472, 0.037625]
- self-shuffled: mean=0.046389 CI=[0.029069, 0.062500]
- level_lp_cal-level_cal: mean=0.088750 CI=[0.046656, 0.131250]

## V3 mean utilities (n_h=10)

| arm | mean U |
| --- | ---: |
| `naive` | 0.516667 |
| `global` | 0.700000 |
| `self` | 0.725278 |
| `shuffled` | 0.678889 |
| `oracle_level` | 0.758333 |
| `oracle_instance` | 0.927500 |
| `level_cal` | 0.758333 |
| `level_lp_cal` | 0.847083 |

## Descriptive: V2 mean utilities (n_h=10)

| arm | mean U |
| --- | ---: |
| `naive` | 0.516667 |
| `global` | 0.516667 |
| `self` | 0.501944 |
| `shuffled` | 0.486944 |
| `oracle_level` | 0.475000 |
| `oracle_instance` | 0.758333 |
| `level_cal` | 0.475000 |
| `level_lp_cal` | 0.633333 |

## Descriptive: n_h curve (V3)

| n_h | self | global | shuffled |
| ---: | ---: | ---: | ---: |
| 2 | 0.700000 | 0.669444 | 0.700000 |
| 5 | 0.707778 | 0.675556 | 0.623333 |
| 10 | 0.721111 | 0.693889 | 0.666944 |
| 25 | 0.738056 | 0.700000 | 0.650556 |

## Per-level accuracy (all pools, descriptive)

| level | accuracy |
| ---: | ---: |
| 2 | 0.9450 |
| 3 | 0.7850 |
| 4 | 0.8350 |
| 5 | 0.7100 |
| 6 | 0.6500 |
| 7 | 0.5450 |

## Descriptive: AUROC of answer_logprob vs correctness

- mean of finite per-level AUROCs: 0.9340347121783431

| level | AUROC |
| ---: | ---: |
| 2 | 0.9822029822029822 |
| 3 | 0.9674122352244112 |
| 4 | 0.9328615496280167 |
| 5 | 0.9470616804273919 |
| 6 | 0.8851648351648351 |
| 7 | 0.8895049904224216 |

## Descriptive: risk-coverage (TEST, ranked by answer_logprob)

| coverage | n_kept | accuracy | risk |
| ---: | ---: | ---: | ---: |
| 0.25 | 60 | 1.0000 | 0.0000 |
| 0.5 | 120 | 1.0000 | 0.0000 |
| 0.75 | 180 | 0.9278 | 0.0722 |
| 1.0 | 240 | 0.7583 | 0.2417 |

Per-level at 50% coverage:

| level | n | n_kept | accuracy | risk |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 40 | 20 | 1.0000 | 0.0000 |
| 3 | 40 | 20 | 1.0000 | 0.0000 |
| 4 | 40 | 20 | 1.0000 | 0.0000 |
| 5 | 40 | 20 | 0.9500 | 0.0500 |
| 6 | 40 | 20 | 0.9500 | 0.0500 |
| 7 | 40 | 20 | 0.9000 | 0.1000 |

## Claimed

- Labels A=`SUPPORTED_WEAK` and C=`INTERNAL_VALUE` under `docs/M34A_V2_PREREG.md` on this cache.

## NOT claimed

- Privileged internal access in the mechanistic sense; Qwen residual κ; planning/games; other models or operations.

## Scope

- One small local model (`Qwen/Qwen2.5-3B-Instruct`), multiplication by one digit (`mul_n1`) only, levels 2–7, greedy float16, V3 primary, static warm-start history, n_h=10.

**Interpretation:** weak track-record value plus instance-level logprob value on this task.

## Erratum

The first written report ended with "**Interpretation:** no support at this scale." That sentence was a leftover fallthrough in `scripts/m34a_v2_analyze.py`: only `A=SUPPORTED` (not `SUPPORTED_WEAK`) was mapped, so `A=SUPPORTED_WEAK` and `C=INTERNAL_VALUE` hit the v1 default. Labels and every number above are unchanged. The script now maps every `(A, C)` pair; this file's interpretation line was regenerated from those labels only. Analysis git commit of the one run remains `2a63d27290f74f639510ac18ee106ca70b1161e2`.
