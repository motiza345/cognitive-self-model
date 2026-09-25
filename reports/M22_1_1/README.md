# M22.1.1 intervention-space audit

Descriptive characterization of residual directions at one frozen site.
This does not change the M22.1 candidate and does not authorize M22.2.

## Protocol

- Model: `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Architecture: `Qwen2.5-0.5B`, d_model `896`, layers `24`.
- Device `cpu`, dtype `float32`.
- Torch `2.14.0+cpu`, TransformerLens `3.9.0`, Transformers `5.17.0`.
- Site: `blocks.23.hook_resid_post`, last token only.
- Outcome: `logit(9834) - logit(902)` for `" yes"` and `" no"`.
- Prompts and discovery/validation/replication roles are the M22.1 frozen set.
- Magnitudes: `-2, -1, 0, +1, +2` for every direction.
- D1 is the M22.1 primary, seed 22101.
- D2 is the M22.1 control, seed 22103, Gram-Schmidt against D1 only.
- D3-D8 use seeds 22111-22116 and are Gram-Schmidt orthogonalized against every earlier panel member.
- The panel manifest was written before the first hooked forward.
- No direction was added, removed, or reseeded after an effect was observed.
- Geometric orthogonality does not imply causal or functional independence.

## Result

- Descriptive label: `INSUFFICIENT_EVIDENCE`.
- M22.1 status remains `CANDIDATE`.
- M22.2 remains not authorized.

The label is the frozen rule, not a new specificity pass.
A split matches at most one of the pre-registered patterns.
Zero matches, or more than one match, is `INSUFFICIENT_EVIDENCE`.
The panel label is that shared validation/replication label.
Discovery is reported and is not used for the label.

The frozen comparisons, evaluated at alpha `+1`, are:

- broad: at least 6 substantial directions and max/min absolute mean below 4.0
- structured: at least 6 substantial directions and max/median absolute mean at least 4.0
- low-dimensional: at most 3 substantial directions and leading uncentered energy at least 0.8
- regime-dependent: at least 4 substantial directions whose regime means change sign and whose regime range exceeds half the absolute mean

- validation: label `INSUFFICIENT_EVIDENCE`, substantial `8`, max/min `12.567216500705845`, max/median `3.772044772044772`, leading energy `0.999382`, regime sign-change directions `[]`.
- replication: label `INSUFFICIENT_EVIDENCE`, substantial `8`, max/min `12.668867606898356`, max/median `3.7260258147035157`, leading energy `0.999234`, regime sign-change directions `[]`.
- discovery: label `INSUFFICIENT_EVIDENCE`, substantial `8`, max/min `11.915666428583874`, max/median `3.7678824235813875`, leading energy `0.999294`, regime sign-change directions `[]`.

Validation and replication both match none of the four patterns, so the panel label is `INSUFFICIENT_EVIDENCE`.
The raw profiles below are the result. The category was not adjusted after seeing them.

## Effect profile

Means are paired deltas on that split. `effect_per_unit_norm` equals the mean at `+1` because every direction has unit norm.

### discovery

| direction | role | mean Δ(+1) | mean Δ(-1) | sign consistency | symmetry |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 | m22_1_primary | +0.029388 | -0.028833 | 1.000 | +0.000555 |
| D2 | m22_1_control | +0.108718 | -0.108171 | 1.000 | +0.000548 |
| D3 | pre_registered_new | +0.037791 | -0.037259 | 1.000 | +0.000532 |
| D4 | pre_registered_new | -0.028320 | +0.028861 | 1.000 | +0.000541 |
| D5 | pre_registered_new | +0.009124 | -0.008586 | 1.000 | +0.000538 |
| D6 | pre_registered_new | +0.036134 | -0.035576 | 1.000 | +0.000559 |
| D7 | pre_registered_new | +0.016580 | -0.016047 | 1.000 | +0.000533 |
| D8 | pre_registered_new | -0.014773 | +0.015311 | 1.000 | +0.000538 |

| direction | Δ(-2) | Δ(-1) | Δ(0) | Δ(+1) | Δ(+2) | completion | instruction | syntax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | -0.057091 | -0.028833 | +0.000000 | +0.029388 | +0.059316 | +0.029959 | +0.029856 | +0.028349 |
| D2 | -0.215735 | -0.108171 | +0.000000 | +0.108718 | +0.217945 | +0.109396 | +0.105218 | +0.111540 |
| D3 | -0.073970 | -0.037259 | +0.000000 | +0.037791 | +0.076101 | +0.039018 | +0.036642 | +0.037713 |
| D4 | +0.058247 | +0.028861 | +0.000000 | -0.028320 | -0.056084 | -0.028444 | -0.027501 | -0.029015 |
| D5 | -0.016624 | -0.008586 | +0.000000 | +0.009124 | +0.018785 | +0.010639 | +0.009647 | +0.007085 |
| D6 | -0.070569 | -0.035576 | +0.000000 | +0.036134 | +0.072813 | +0.037136 | +0.036718 | +0.034549 |
| D7 | -0.031555 | -0.016047 | +0.000000 | +0.016580 | +0.033676 | +0.016921 | +0.015891 | +0.016929 |
| D8 | +0.031153 | +0.015311 | +0.000000 | -0.014773 | -0.029006 | -0.014566 | -0.014202 | -0.015551 |

### validation

| direction | role | mean Δ(+1) | mean Δ(-1) | sign consistency | symmetry |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 | m22_1_primary | +0.028637 | -0.028185 | 1.000 | +0.000452 |
| D2 | m22_1_control | +0.105415 | -0.104954 | 1.000 | +0.000461 |
| D3 | pre_registered_new | +0.037810 | -0.037384 | 1.000 | +0.000426 |
| D4 | pre_registered_new | -0.027256 | +0.027701 | 1.000 | +0.000445 |
| D5 | pre_registered_new | +0.008388 | -0.007943 | 1.000 | +0.000445 |
| D6 | pre_registered_new | +0.035595 | -0.035125 | 1.000 | +0.000471 |
| D7 | pre_registered_new | +0.016218 | -0.015788 | 1.000 | +0.000430 |
| D8 | pre_registered_new | -0.014931 | +0.015368 | 1.000 | +0.000437 |

| direction | Δ(-2) | Δ(-1) | Δ(0) | Δ(+1) | Δ(+2) | completion | instruction | syntax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | -0.055907 | -0.028185 | +0.000000 | +0.028637 | +0.057723 | +0.029934 | +0.028282 | +0.027695 |
| D2 | -0.209393 | -0.104954 | +0.000000 | +0.105415 | +0.211239 | +0.107589 | +0.101017 | +0.107639 |
| D3 | -0.074323 | -0.037384 | +0.000000 | +0.037810 | +0.076032 | +0.039016 | +0.035501 | +0.038912 |
| D4 | +0.055829 | +0.027701 | +0.000000 | -0.027256 | -0.054051 | -0.026716 | -0.026347 | -0.028704 |
| D5 | -0.015442 | -0.007943 | +0.000000 | +0.008388 | +0.017216 | +0.009463 | +0.009154 | +0.006547 |
| D6 | -0.069762 | -0.035125 | +0.000000 | +0.035595 | +0.071648 | +0.037260 | +0.035153 | +0.034373 |
| D7 | -0.031133 | -0.015788 | +0.000000 | +0.016218 | +0.032857 | +0.016555 | +0.014865 | +0.017233 |
| D8 | +0.031163 | +0.015368 | +0.000000 | -0.014931 | -0.029421 | -0.013944 | -0.014072 | -0.016777 |

### replication

| direction | role | mean Δ(+1) | mean Δ(-1) | sign consistency | symmetry |
| --- | --- | ---: | ---: | ---: | ---: |
| D1 | m22_1_primary | +0.029694 | -0.029163 | 1.000 | +0.000531 |
| D2 | m22_1_control | +0.107654 | -0.107128 | 1.000 | +0.000527 |
| D3 | pre_registered_new | +0.038308 | -0.037798 | 1.000 | +0.000510 |
| D4 | pre_registered_new | -0.028091 | +0.028608 | 1.000 | +0.000517 |
| D5 | pre_registered_new | +0.008498 | -0.007977 | 1.000 | +0.000520 |
| D6 | pre_registered_new | +0.036017 | -0.035477 | 1.000 | +0.000540 |
| D7 | pre_registered_new | +0.016431 | -0.015921 | 1.000 | +0.000510 |
| D8 | pre_registered_new | -0.015099 | +0.015611 | 1.000 | +0.000513 |

| direction | Δ(-2) | Δ(-1) | Δ(0) | Δ(+1) | Δ(+2) | completion | instruction | syntax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| D1 | -0.057786 | -0.029163 | +0.000000 | +0.029694 | +0.059893 | +0.030284 | +0.031952 | +0.026846 |
| D2 | -0.213690 | -0.107128 | +0.000000 | +0.107654 | +0.215772 | +0.109727 | +0.113583 | +0.099653 |
| D3 | -0.075078 | -0.037798 | +0.000000 | +0.038308 | +0.077089 | +0.040195 | +0.039893 | +0.034836 |
| D4 | +0.057715 | +0.028608 | +0.000000 | -0.028091 | -0.055655 | -0.027644 | -0.029803 | -0.026827 |
| D5 | -0.015435 | -0.007977 | +0.000000 | +0.008498 | +0.017508 | +0.008629 | +0.010434 | +0.006430 |
| D6 | -0.070408 | -0.035477 | +0.000000 | +0.036017 | +0.072547 | +0.035968 | +0.039517 | +0.032564 |
| D7 | -0.031333 | -0.015921 | +0.000000 | +0.016431 | +0.033358 | +0.016397 | +0.016685 | +0.016213 |
| D8 | +0.031718 | +0.015611 | +0.000000 | -0.015099 | -0.029688 | -0.014492 | -0.015779 | -0.015025 |

## Geometry and functional correlation

Pairwise dots of the unit directions are in `direction_panel_manifest.json`.
Off-diagonal dots are numerical zeros. That is geometric orthogonality.
Prompt-level Pearson correlations of the alpha `+1` deltas are in `intervention_space_report.json`.
Those correlations are not the same object as the dots. With six prompts in a split, they are descriptive.

The uncentered singular-value energy of each prompt-by-direction delta matrix is stored per split.
A large leading energy describes the observed matrix. It is not a causal mechanism, and it was not used to build a new intervention.

## Interpretation limit

On this frozen checkpoint, at this frozen L23 residual site and frozen logit-margin outcome, the pre-registered panel has a measurable causal response.
The control is geometrically orthogonal to the primary and is still sensitive to the same outcome. Other panel directions are sensitive as well.
The frozen category for that pattern is `INSUFFICIENT_EVIDENCE` because the magnitude spread sits between the broad and structured cutoffs, every direction is substantial, and regime signs do not flip.
This does not establish a mechanism identity, a causal circuit, a SelfModel, self-awareness, or general behavioral control.
M22.2 is not authorized.
