# M22.1.3 readout-geometry null audit

Proposed finding id: `F-M22.1.3-READOUT`.
Primary label: `READOUT_RECONSTRUCTION_EXACT`.
Linearity finding: `FIRST_ORDER_SUPPORTED`.

Status: M22.1 remains `CANDIDATE` (unchanged). M22.2 is not authorized.
Any status change requires a separate decision audit and explicit user approval.

The recorded M22.1/M22.1.1 effects are exactly reconstructible from the final-residual perturbation through the model's final RMSNorm and unembedding.

## Environment and realized load

- Python 3.12.3, Torch 2.14.0+cpu, Transformers 5.17.0, TransformerLens 3.9.0.
- Realized normalization `RMSPre`, `ln_final` class `RMSNormPre`, weight present `False`, eps `1e-06`.
- cfg flags: fold_ln `not_on_cfg`, center_writing_weights `not_on_cfg`, center_unembed `not_on_cfg`, default_prepend_bos `False`.
- Config sha256 `6429c4b1d5151d7b01eb13d5622b2e55d9fb0b3a78bf46993b871c213aac8bd3`. Predictions sha256 `7391dd85d4bf6ccb16b3304795e9ede6c2edc349908ec335f735ef55b45cdf30`.

## Level B reconstruction

- validation: pass 192/192 = 1.0, median abs 3.814697265625e-06, p95 9.012222290039052e-06, max 1.2874603271484375e-05.
- replication: pass 192/192 = 1.0, median abs 3.814697265625e-06, p95 1.1444091796875e-05, max 1.71661376953125e-05.
- discovery: pass 192/192 = 1.0, median abs 3.814697265625e-06, p95 1.049041748046875e-05, max 1.6689300537109375e-05.

## Three-way table at alpha = +1

### validation

| direction | Δ_rec | Δ̂_exact f32 | Δ₁ Jacobian | Δ₀ frozen |
| --- | ---: | ---: | ---: | ---: |
| D1 | +0.028637 | +0.028643 | +0.028415 | +0.029415 |
| D2 | +0.105415 | +0.105414 | +0.105191 | +0.105151 |
| D3 | +0.037810 | +0.037811 | +0.037600 | +0.036967 |
| D4 | -0.027256 | -0.027255 | -0.027478 | -0.027448 |
| D5 | +0.008388 | +0.008388 | +0.008166 | +0.009461 |
| D6 | +0.035595 | +0.035595 | +0.035363 | +0.036526 |
| D7 | +0.016218 | +0.016218 | +0.016003 | +0.015529 |
| D8 | -0.014931 | -0.014931 | -0.015151 | -0.014668 |

### replication

| direction | Δ_rec | Δ̂_exact f32 | Δ₁ Jacobian | Δ₀ frozen |
| --- | ---: | ---: | ---: | ---: |
| D1 | +0.029694 | +0.029690 | +0.029429 | +0.030072 |
| D2 | +0.107654 | +0.107651 | +0.107400 | +0.107500 |
| D3 | +0.038308 | +0.038302 | +0.038055 | +0.037793 |
| D4 | -0.028091 | -0.028091 | -0.028351 | -0.028061 |
| D5 | +0.008498 | +0.008496 | +0.008238 | +0.009672 |
| D6 | +0.036017 | +0.036015 | +0.035750 | +0.037342 |
| D7 | +0.016431 | +0.016431 | +0.016179 | +0.015876 |
| D8 | -0.015099 | -0.015101 | -0.015357 | -0.014995 |

### discovery

| direction | Δ_rec | Δ̂_exact f32 | Δ₁ Jacobian | Δ₀ frozen |
| --- | ---: | ---: | ---: | ---: |
| D1 | +0.029388 | +0.029387 | +0.029112 | +0.030113 |
| D2 | +0.108718 | +0.108723 | +0.108455 | +0.107646 |
| D3 | +0.037791 | +0.037794 | +0.037529 | +0.037844 |
| D4 | -0.028320 | -0.028319 | -0.028592 | -0.028100 |
| D5 | +0.009124 | +0.009126 | +0.008856 | +0.009685 |
| D6 | +0.036134 | +0.036135 | +0.035856 | +0.037393 |
| D7 | +0.016580 | +0.016579 | +0.016313 | +0.015898 |
| D8 | -0.014773 | -0.014773 | -0.015044 | -0.015016 |

## Level A2 local Jacobian versus exact

- validation: n=184, R²=0.9998671749577204, sign=1.0, pass=0.9891304347826086.
- replication: n=182, R²=0.999818111208195, sign=1.0, pass=0.9945054945054945.

## Level A frozen denominator and direction alignment

- validation Δ₀ vs exact: R²=0.9987870661805858, sign=1.0.
- validation b ≈ k c: k_plus=0.3698229773506419, R²=0.999749991709367, |log(k / mean 1/R)|=7.355173982428909e-05, c[D2]/c[D1]=3.574732128767665, b[D2]/b[D1]=3.6810772108409324.

| direction | c = r·d | mean Δ_rec(+1) | mean Δ_rec(−1) |
| --- | ---: | ---: | ---: |
| D1 | +0.079544 | +0.028637 | -0.028185 |
| D2 | +0.284349 | +0.105415 | -0.104954 |
| D3 | +0.099967 | +0.037810 | -0.037384 |
| D4 | -0.074226 | -0.027256 | +0.027701 |
| D5 | +0.025584 | +0.008388 | -0.007943 |
| D6 | +0.098774 | +0.035595 | -0.035125 |
| D7 | +0.041994 | +0.016218 | -0.015788 |
| D8 | -0.039664 | -0.014931 | +0.015368 |

- replication Δ₀ vs exact: R²=0.9987229768137037, sign=1.0.
- replication b ≈ k c: k_plus=0.37748925106671133, R²=0.999767924778546, |log(k / mean 1/R)|=0.0015020234591209584, c[D2]/c[D1]=3.574732128767665, b[D2]/b[D1]=3.625489115016299.

| direction | c = r·d | mean Δ_rec(+1) | mean Δ_rec(−1) |
| --- | ---: | ---: | ---: |
| D1 | +0.079544 | +0.029694 | -0.029163 |
| D2 | +0.284349 | +0.107654 | -0.107128 |
| D3 | +0.099967 | +0.038308 | -0.037798 |
| D4 | -0.074226 | -0.028091 | +0.028608 |
| D5 | +0.025584 | +0.008498 | -0.007977 |
| D6 | +0.098774 | +0.036017 | -0.035477 |
| D7 | +0.041994 | +0.016431 | -0.015921 |
| D8 | -0.039664 | -0.015099 | +0.015611 |

## Non-claims

- No mechanism is explained.
- No claim is made about self-representation.
- D1–D8 have no assigned semantic meaning.
- Nothing is claimed about sites other than L23.
- The D2>D1 ordering is reported only as readout alignment.

## Status

M22.1 remains `CANDIDATE`.
M22.2 is `NOT AUTHORIZED`.
