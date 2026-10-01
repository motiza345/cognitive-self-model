# Project feasibility gate execution

The gate specification remains `GATE_SPECIFIED`. This file records the run. It does not change the decision map.

Verdict: `FEASIBILITY_CONTINUE`

The frozen rule clears both held-out partitions, the orthogonal control does not, and the sign bound holds. That is evidence that g carries held-out response information beyond the scalar mean for this family. It is not a self-model. Construction of an object whose response map is g becomes the justified next step, followed by a separate consumption test.

## Train means

These are the only fitted numbers. The candidate prediction is `g`, with no slope.

| cell | train mean m |
| --- | ---: |
| `M22.1-D1-L0` | 0.02711375554402669 |
| `M22.1-D1-L8` | 0.02379457155863444 |
| `M22.1-D1-L15` | -0.0065801143646240234 |

## Evaluation

- baseline MAE: `0.047729690869649254`
- candidate MAE: `0.007789518321967787`
- paired mean: `0.039940172547681466`
- interval: `[0.0267728113103658, 0.0560522271854872]`
- class: `CI_POSITIVE`
- control class: `CI_NEGATIVE`
- control paired mean: `-0.022324956073943112`
- sign co-criterion: `True`

## Replication

- baseline MAE: `0.057014765562834566`
- candidate MAE: `0.01424753355483214`
- paired mean: `0.04276723200800242`
- interval: `[0.02623499902310195, 0.061806415224930765]`
- class: `CI_POSITIVE`
- control class: `CI_NEGATIVE`
- control paired mean: `-0.02336364419682434`
- sign co-criterion: `True`

## Validation

Validation is not an input to the decision.

- candidate class: `CI_POSITIVE`
- paired mean: `0.03879704234983634`
- interval: `[0.0263656137459394, 0.05281986432425954]`
- control class: `CI_NEGATIVE`
- sign co-criterion: `True`

## Sign counts

| split | cell | successes | n | low | pass |
| --- | --- | ---: | ---: | ---: | --- |
| evaluation | `M22.1-D1-L0` | 12 | 12 | 0.7353515306029487 | `True` |
| evaluation | `M22.1-D1-L8` | 11 | 12 | 0.6152038348490558 | `True` |
| evaluation | `M22.1-D1-L15` | 12 | 12 | 0.7353515306029487 | `True` |
| replication | `M22.1-D1-L0` | 10 | 12 | 0.5158622513140327 | `True` |
| replication | `M22.1-D1-L8` | 11 | 12 | 0.6152038348490558 | `True` |
| replication | `M22.1-D1-L15` | 11 | 12 | 0.6152038348490558 | `True` |
| validation | `M22.1-D1-L0` | 10 | 12 | 0.5158622513140327 | `True` |
| validation | `M22.1-D1-L8` | 12 | 12 | 0.7353515306029487 | `True` |
| validation | `M22.1-D1-L15` | 12 | 12 | 0.7353515306029487 | `True` |

## Leakage

`leakage_ok = true`

Held-out prediction files were closed before the matching outcome files. They contain `g`, `g_orth`, and `m`. They do not contain an observed effect. No slope was fit.
