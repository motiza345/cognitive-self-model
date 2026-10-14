# M34a-v3 — Generalization of the v2 result (final M34a step)

**Status:** DESIGN_LOCKED — no v3 CAL/HIST/TEST outcomes exist  
**Branch:** `cursor/m34a-v3` (from `cursor/m34a-v2` @ `4504820`)  
**Earlier files:** every v1/v2 report, raw cache, and pool plan is read-only. Reuse v2 analysis logic via **new** `scripts/m34a_v3_*` files only.

**Question:** Does the v2 instance-level logprob value (label C) and track-record utility (label A) generalize beyond one model × one operation?

---

## 1. Settings

### S1 — same model, new operation (`add_nn`)

- Model: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Family: **`add_nn`** (n-digit + n-digit)
- Prompt: `Compute A + B. Reply with only the integer.` (via the shared template with `op=+`)
- Level selection: **only** pilot-2 `add_nn` accuracies in `reports/m34a_pilot2/levels.json`, using the **v2** `gap_hat` rule (window of 6 contiguous levels; max `gap_hat`; ties → lower levels; STOP if `gap_hat < 0.05`). No v1/v2 pool outcomes.
- **Chosen (computed, no generation):** levels **`[3, 4, 5, 6, 7, 8]`**, `gap_hat = 0.1`, `p̄ = 0.833333…`. See `reports/m34a_v3_s1/levels.json`.
- Pools seed **34004**. CAL 100 / HIST 60 / TEST 40 per level = 1200 generations. Plan committed before collection.

### S2 — new model, same operation (`mul_n1`)

- Model: `microsoft/Phi-3.5-mini-instruct`
- Revision: resolve with `huggingface_hub.model_info().sha` and write it into a revision pin file / manifest **before** any S2 collection
- Compute: Colab T4, `float16`, greedy, chat template, `max_new_tokens=60`
- Prompt: `Compute A x B. Reply with only the integer.`
- Family: **`mul_n1`**
- **New pilot:** 20 problems per level, n = 2..10, seed **34005**, disjoint from all earlier pilots and pools. Selection by the same `gap_hat` rule. If best `gap_hat < 0.05`, STOP and mark S2 `NOT_INFORMATIVE` (no pool collection).
- Pools seed **34006**, same sizes (1200) if levels are chosen.
- If the model cannot be loaded or its tokenizer breaks the answer-token logprob rule (sum of generated-token `log_softmax` probs, **EOS excluded**), record that and mark S2 **`NOT_RUN`** (no substitution).

Shared: no verbal confidence; parse = first integer; float32 rerun on non-finite logits/logprobs as in v2.

## 2. Disjointness

S1 operands disjoint from pilot 1, pilot 2, v1 pools, and v2 pools.  
S2 pilot and pools additionally disjoint from the S1 plan and from each other. Unit-tested.

## 3. Per-setting analysis (identical to v2)

Arms, utilities, gates, labels, bootstrap, history draws, and primary `n_h=10` / V3 are exactly as in `docs/M34A_V2_PREREG.md` §§5–7:

| Action | Utility |
| --- | ---: |
| answer correct | +1 |
| answer wrong | −1 |
| abstain | 0 |
| verify | +0.7 |

Arms: `naive`, `global`, `self`, `shuffled`, `oracle_level`, `oracle_instance`, `level_cal`, `level_lp_cal` (CAL terciles of `answer_logprob`). Paired bootstrap 5000 draws, seed 23001, 240 TEST problems.

**GATE_A / GATE_C / A / C** per setting: same exhaustive table as v2. `NOT_RUN` (S2 only) is treated like gate failure for that setting's contribution to the aggregate (`C` treated as `NOT_INFORMATIVE` for aggregation; A for that setting recorded as `NOT_RUN`).

## 4. Aggregate for C (exhaustive)

Map each setting's C ∈ {`INTERNAL_VALUE`, `INTERNAL_NONE`, `NOT_INFORMATIVE`}. **`NOT_RUN` counts as `NOT_INFORMATIVE`.**

| C(S1) | C(S2) | Aggregate |
| --- | --- | --- |
| INTERNAL_VALUE | INTERNAL_VALUE | `C_GENERAL` |
| INTERNAL_VALUE | INTERNAL_NONE | `C_MIXED` |
| INTERNAL_NONE | INTERNAL_VALUE | `C_MIXED` |
| INTERNAL_VALUE | NOT_INFORMATIVE | `C_PARTIAL` |
| NOT_INFORMATIVE | INTERNAL_VALUE | `C_PARTIAL` |
| INTERNAL_NONE | INTERNAL_NONE | `C_NOT_REPLICATED` |
| INTERNAL_NONE | NOT_INFORMATIVE | `C_NOT_REPLICATED` |
| NOT_INFORMATIVE | INTERNAL_NONE | `C_NOT_REPLICATED` |
| NOT_INFORMATIVE | NOT_INFORMATIVE | `C_NOT_INFORMATIVE` |

Rule summary: both VALUE → `C_GENERAL`; one VALUE and one NONE → `C_MIXED`; one VALUE and one NOT_INFORMATIVE → `C_PARTIAL`; no VALUE and at least one NONE → `C_NOT_REPLICATED`; both NOT_INFORMATIVE → `C_NOT_INFORMATIVE`.

Per-setting **A** labels are reported separately (no aggregate A in this prereg).

## 5. Descriptive only

Within-level AUROC of `answer_logprob` for correctness; within-level AUROC of the number of schoolbook carries (addition: digit-wise carry-outs; multiplication: as in the v2 post hoc) for correctness; risk-coverage; `n_h` curve. Do not change labels.

## 6. Stop / integrity

- Analysis scripts are committed **before** collection. At run time analysis reads `git rev-parse HEAD`, **refuses a dirty working tree**, and writes that hash into the report.
- Cache SHA-256 must match the manifest.
- One collection per setting (resumable crash recovery only). One analysis run per setting. No retuning.
- **M34a ends after v3.** No further variants. No threshold or rule changes after seeing data.

## 7. Interpretation (fixed)

- Aggregate `C_GENERAL`: instance-level logprob value on both settings.
- `C_MIXED` / `C_PARTIAL`: limited generalization; report both setting Cs.
- `C_NOT_REPLICATED` / `C_NOT_INFORMATIVE`: no replicated instance-level claim.
- Per-setting A interpreted as in the v2 report erratum (including `SUPPORTED_WEAK`).

## 8. Artifacts

| Path | Role |
| --- | --- |
| `docs/M34A_V3_PREREG.md` | this file |
| `docs/M34A_V3_RUN.md` | Colab runbook |
| `reports/m34a_v3_s1/` | S1 levels + pool plan |
| `reports/m34a_v3_s2_pilot/` | S2 pilot levels (after pilot) |
| `reports/m34a_v3_s2/` | S2 pool plan (after levels chosen) |
| `reports/m34a_v3_s1_raw/`, `reports/m34a_v3_s2_raw/` | caches |
| `reports/M34A_V3_REPORT.md` | one-run report |
