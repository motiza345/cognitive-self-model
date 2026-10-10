# M34A-v3 report

**Aggregate C:** `C_NOT_INFORMATIVE` · S1 A=`NOT_SUPPORTED` C=`NOT_INFORMATIVE` · S2 A=`NOT_INFORMATIVE` C=`NOT_INFORMATIVE`

- analysis git commit: `3fed6fc0d7b43025b5518f700345163539918eb0` (clean tree required)

## S1: Qwen2.5-3B-Instruct add_nn levels 3–8

**Verdict:** A=`NOT_SUPPORTED` · C=`NOT_INFORMATIVE` · GATE_A=`True` · GATE_C=`False`

- cache sha256: `60876bff0ec1a0613a21bf56a4bd06297a5c118935790530f01e63e5cab8251e`
- model: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- device/dtype: `cuda` / `float16`
- collection git commit: `3fb287edbc8ec5bb99b9926d2318a836b234bea8`
- GATE_A gap = 0.094444
- self-global: mean=0.023056 CI=[-0.010167, 0.060695]
- self-shuffled: mean=0.040278 CI=[0.020137, 0.062167]
- level_lp_cal-level_cal: mean=0.072500 CI=[0.023333, 0.125000]
- mean within-level AUROC answer_logprob: 0.9362053970325702
- mean within-level AUROC (−n_carries): 0.5750584660051864

### V3 mean utilities (n_h=10)

| arm | mean U |
| --- | ---: |
| `naive` | 0.741667 |
| `global` | 0.722222 |
| `self` | 0.745278 |
| `shuffled` | 0.705000 |
| `oracle_level` | 0.816667 |
| `oracle_instance` | 0.961250 |
| `level_cal` | 0.816667 |
| `level_lp_cal` | 0.889167 |

### Descriptive: n_h curve (V3)

| n_h | self | global | shuffled |
| ---: | ---: | ---: | ---: |
| 2 | 0.700000 | 0.716667 | 0.700000 |
| 5 | 0.762222 | 0.719444 | 0.700833 |
| 10 | 0.739167 | 0.718056 | 0.706389 |
| 25 | 0.791944 | 0.706944 | 0.718889 |

## S2: Phi-3.5-mini-instruct mul_n1

- status: `STOP` · A=`NOT_INFORMATIVE` · C=`NOT_INFORMATIVE`
- detail: `{"best_gap_hat": 0.033333333333333326, "chosen_levels": null, "model_id": "microsoft/Phi-3.5-mini-instruct", "note": "Prereg: if best gap_hat < 0.05, STOP and mark S2 NOT_INFORMATIVE; no pool collection.", "pilot_levels_path": "reports/m34a_v3_s2_pilot/levels.json", "reason": "STOP_GAP_LT_0.05", "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77", "setting": "s2", "status": "STOP_GAP_LT_0.05"}`

## Claimed

- Per-setting A/C under `docs/M34A_V3_PREREG.md` (v2 gates/labels). Aggregate C=`C_NOT_INFORMATIVE`.

## NOT claimed

- Privileged access (logprobs are API-visible); mechanistic κ; planning/games; other models/operations beyond S1/S2.

## Scope

- S1: one Qwen2.5-3B-Instruct, addition (`add_nn`) levels 3–8 only.
- S2: one Phi-3.5-mini-instruct, multiplication by one digit (`mul_n1`) only (or NOT_RUN / STOP).
- Static warm-start history, n_h=10 primary, greedy float16. M34a closed after v3.
