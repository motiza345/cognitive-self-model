# M34A report

**Primary verdict (V3, n_h=10, Amendment 1):** A=`NOT_INFORMATIVE` · C=`INTROSPECTION_NONE` · GATE_OK=`False` · SPREAD_OK=`True` · VAR_OK=`True`

- cache sha256: `290307e5b95a062ee00141f43467c660ef48ddae95d6ba282f1269baaa200f99`
- manifest model: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- device/dtype: `cuda` / `float16`
- torch/transformers: `2.11.0+cu130` / `5.18.0`
- collection git commit: `ece1d19c19aaded8b8afdae2764389d3a2774e72`
- analysis git commit: `ece1d19c19aaded8b8afdae2764389d3a2774e72`
- GATE gap U(oracle_level)-U(global) = 0.000000 (need >= 0.05)
- SPREAD_OK (CAL acc in [0.15,0.85] on >=3 levels): `True`
- VAR_OK (CAL >=15 correct and >=15 wrong on >=3 levels): `True`
- self-global: mean=-0.001389 CI=[-0.003083, 0.000111]
- self-shuffled: mean=0.001944 CI=[-0.000542, 0.004431]
- verbal_cal-self: mean=0.001389 CI=[-0.000111, 0.003083]

## V3 mean utilities (n_h=10)

| arm | mean U |
| --- | ---: |
| `naive` | 0.116667 |
| `none` | 0.700000 |
| `global` | 0.700000 |
| `self` | 0.698611 |
| `shuffled` | 0.696667 |
| `verbal_raw` | 0.700000 |
| `verbal_cal` | 0.700000 |
| `combined` | 0.700000 |
| `oracle_level` | 0.700000 |
| `oracle_instance` | 0.867500 |

## Descriptive: V2 mean utilities (n_h=10)

| arm | mean U |
| --- | ---: |
| `naive` | 0.116667 |
| `none` | 0.116667 |
| `global` | 0.093333 |
| `self` | 0.109167 |
| `shuffled` | 0.088333 |
| `verbal_raw` | 0.116667 |
| `verbal_cal` | 0.093333 |
| `combined` | 0.107500 |
| `oracle_level` | 0.125000 |
| `oracle_instance` | 0.558333 |

## Descriptive: self mean U by n_h (V3)

| n_h | self | global | shuffled |
| ---: | ---: | ---: | ---: |
| 2 | 0.700000 | 0.700000 | 0.700000 |
| 5 | 0.661944 | 0.700000 | 0.650556 |
| 10 | 0.697222 | 0.700000 | 0.696389 |
| 25 | 0.700000 | 0.700000 | 0.700000 |

## Per-level accuracy (all pools pooled descriptive)

| level | accuracy |
| ---: | ---: |
| 5 | 0.7050 |
| 6 | 0.6650 |
| 7 | 0.5100 |
| 8 | 0.4450 |
| 9 | 0.4950 |
| 10 | 0.4700 |

ECE of stated confidence (TEST+HIST+CAL pooled): 0.048333

## Claimed

- Labels A=`NOT_INFORMATIVE` and C=`INTROSPECTION_NONE` under Amendment 1 of the frozen M34a prereg on this cache.

## NOT claimed

- Privileged internal access; Qwen mechanistic κ; agent planning/games (M34b/c); other models.

## Scope

- Qwen2.5-3B-Instruct, Amendment 1 family `mul_n1`, levels 5–10, greedy float16, V3 primary, static warm-start history, n_h=10 primary. Prompt: integer only (stated confidence defaults to 50).

**Interpretation:** no support at this scale.

## Erratum

The "analysis git commit" field cites `ece1d19`, but the script actually used was committed later in `63bbe98` (it adds the Amendment 1 CAL gates; utilities are identical: rerunning both script versions on the cached responses gives 0 differing values among the 64 shared summary keys; cache sha256 `290307e5b95a062ee00141f43467c660ef48ddae95d6ba282f1269baaa200f99` matches the manifest).

No number in this file has been changed.
