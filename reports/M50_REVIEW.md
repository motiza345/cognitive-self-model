# M50 review (after the single official analysis)

Protocol: `docs/M50_PREREG.md`. Official numbers: `reports/M50_REPORT.md` and `reports/m50_raw/analysis_summary.json`, produced once by `scripts/m50_analyze.py` on commit `0ab37351fbb0f56666cfa405c5e4baa29633b5ff` (clean tree, 8 shuffle seeds). Data: `Qwen/Qwen2.5-3B-Instruct`, 12 scope-levels (3 tasks x 4 levels), 1200 learning items and 960 test items, collected at commit `8c0c3bb` (manifest and hashes verified; 0 non-finite values). Nothing in the protocol, the labels or the code was changed after seeing the results. Section 3 is post hoc and exploratory.

## 1. Official labels

| | H-A gate works | H-B calibration matters | H-C scoping matters | H-D probe vs log probability | status |
|---|---|---|---|---|---|
| natural | **A_SUPPORTED** | **B_SUPPORTED** | C_NOT_SUPPORTED | UNRESOLVED | OK |
| implanted | A_SUPPORTED | B_SUPPORTED | C_NOT_SUPPORTED | LOGPROB_BETTER | **POSITIVE_CONTROL_FAILED** |

## 2. Numbers (natural condition; mean utility per test item, answer +1/-1, verify +0.7)

| arm | utility | verify rate |
|---|---:|---:|
| A1 always answer | 0.5271 | 0 |
| A2 always verify | 0.7000 | 1 |
| A3 random verify (matched budget) | 0.5840 | 0.418 |
| A4 raw confidence | 0.7935 | 0.348 |
| A5 global calibrated layer | 0.8072 | 0.497 |
| **A6 scoped layer** | **0.8205** | 0.418 |
| A7 track record | 0.7000 | 1 |
| A8 probe on prompt state | 0.7004 | 0.721 |
| A9 probe on answer state | 0.8236 | 0.366 |
| R1 instance oracle (reference) | 0.9291 | |

- **H-A (primary):** scoped layer minus the best simple policy (always verify): +0.1205, 95% interval [+0.0982, +0.1406]. The layer answers 58% of items with 95.3% accuracy among answered items and closes 53% of the gap between always-verify and the instance oracle. Its learning-phase utility (online, before freezing) is 0.8033, above always-verify.
- **H-B:** scoped minus raw confidence: +0.0270 [+0.0092, +0.0467], Holm significant.
- **H-C:** scoped minus global calibrated: +0.0133 [-0.0028, +0.0291]; not supported (interval includes zero and the point estimate is below the 0.02 floor).
- **H-D:** answer-state probe minus scoped layer: +0.0031 [-0.0206, +0.0267]; unresolved. The prompt-state probe never beats always-verify (0.7004).
- **Track record (A7)** equals always-verify: no scope-level accuracy exceeds 0.85, so a per-scope accuracy alone never justifies answering; instance-level signals are what the gain comes from.
- Implanted condition: H-A +0.0953 [+0.0765, +0.1129], H-B +0.0851 [+0.0559, +0.1161], H-C +0.0028 [-0.0079, +0.0124], H-D -0.0395 [-0.0627, -0.0170] (the probe, trained on the corrupted labels, is worse).

## 3. Why the positive control failed (post hoc diagnostics)

The decision table of the pre-registration says to check the library for a defect when the positive control fails. Result of the check: **no defect found; the control was not capable of passing the pre-registered floor.**

- **Library behaviour.** In the implanted condition the scoped layer's claims for 3 of the 4 `sort_n` scope-levels stay `UNCERTAIN` (within-scope AUROC 0.60 to 0.65, lower bound at or below 0.55), because the corruption makes the log probability less informative; those scopes use the pooled calibration. The scoped layer works as specified.
- **Headroom.** With a corruption probability of 0.3 the `sort_n` test accuracy is 0.584. With a verify value of 0.7, answering pays only above 0.85, so both layers verify about 90% of `sort_n` items (scoped 0.903, global 0.919). Only 31 of 320 items are answered by the scoped layer (accuracy 0.77); the possible gain from scope-specific knowledge on this task is below 0.01 overall.
- **Where the small difference comes from.** Per task, scoped minus global utility is about +0.000 (`mul_n1`), +0.010 (`add_nn`, which the layer verifies less: 0.403 vs 0.519) and -0.002 (`sort_n`). The scoping benefit is natural, small, and sits in `add_nn`, not in the implanted task.
- **Sensitivity to the implanted failure (same cache, seeds 0 and 1, scoped minus global):** corruption 0.3: +0.003 (official, 8 seeds); 0.5: +0.0125 [+0.0017, +0.0230]; 0.8: +0.0132 [+0.0017, +0.0238]. The effect saturates near +0.013, below the pre-registered floor of 0.02. **The floor was unreachable at this scale and payoff structure; the pre-registration did not check the achievable effect of the positive control in the pilot.**

## 4. Reading under the pre-registered decision table

Natural A_SUPPORTED and B_SUPPORTED; C not supported in the natural condition; implanted `POSITIVE_CONTROL_FAILED`. The prescribed action was to check the library for a defect and, if one is found, repair it once under an amendment and re-analyse the same cache. No defect was found, so no amendment and no re-analysis are made. M50 is closed after its single analysis; any follow-up needs a new pre-registration.

Consequence for the question "does per-scope calibration add value over one global calibration": **not established.** The data are consistent with a small positive effect (about +0.01) and cannot rule out zero; the positive control cannot be used to conclude anything about scoping.

## 5. Claimed / not claimed / scope

**Claimed.** For `Qwen/Qwen2.5-3B-Instruct` on these three verifiable tasks at the chosen levels, with payoffs answer +1/-1, verify +0.7: a module outside the model that turns the model's own answer probability into answer-or-verify decisions with online per-scope calibration beats always answering, always verifying and random verification at matched budget (+0.12 over always-verify), and calibration adds value over using the raw probability (+0.027). A linear probe on the answer-token hidden states performs about as well as the log probability.

**Not claimed.** That per-scope calibration beats one global calibration (not established); that the layer generalizes to other models, tasks or payoffs; anything about self-awareness or introspection; that an agent-specific hidden failure is detected by scoped claims (the positive control failed on feasibility); that probes are no better than log probabilities in general (one linear probe, 1200 training items, one model).

**Scope and limits.** One model, three tasks, 12 scope-levels with 80 test items each; the payoff structure makes answering profitable only above 0.85, which bounds every possible gain from scope-level information; the implanted condition is a seeded post hoc corruption of one task; the verifier is exact and its cost fixed; hidden-state probes use a ridge logistic model with cross-validated penalty on the learning pool only; all labels use the pre-registered intervals.

## 6. Observations for the library (not applied)

- When a scope's confidence signal is uninformative the claim stays `UNCERTAIN` and calibration backs off to the pooled claim; a scope-specific base rate (hierarchical intercept) would use the information that is left. The headroom computed above says this would not change this result.
- Per-observation refits make an analysis run take about 30 s per seed per arm at this size; incremental statistics would remove that cost.

## 7. What this means for the project

The library does what it was built for (calibrated, cost-aware gating), at a measurable gain over simple policies, and without the benefit of scoped claims being demonstrated. Under the decision table fixed in the roadmap, the branch "gate supported, scoping not supported" applies: report calibrated gating as the contribution, state that global calibration was enough in these data, and stop library development there unless a new, feasibility-checked test of scoping is pre-registered. A future test of scoping needs a payoff structure or a failure in which the achievable gain exceeds the floor (computed in the pilot), for example a cheaper verifier or an agent-specific failure that leaves a larger share of items above the answering threshold.
