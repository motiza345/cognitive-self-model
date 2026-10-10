# M34a-v2 — Track-record self-knowledge (fresh data)

**Status:** DESIGN_LOCKED — no v2 CAL/HIST/TEST outcomes exist  
**Branch:** `cursor/m34a-v2` (from `cursor/m34a` @ `63bbe98`)  
**v1:** closed as `NOT_INFORMATIVE`. Every v1 file under `reports/m34a_raw/`, `reports/M34A_REPORT.md` numbers, and `reports/m34a_pilot2/pool_plan.json` is read-only.

**Question (unchanged):** Does self-knowledge from a track record improve a real LLM's answer/abstain/verify decisions beyond global calibration, and does an internal instance-level signal (answer log-probability) add value beyond a level-only CAL rate?

---

## Why v1 failed

1. With verify utility `+0.7`, answering beats verifying only when `p > 0.85`. The Amendment 1 selection rule maximized the count of levels with accuracy in `[0.15, 0.85]` and picked levels **5–10** (pilot accuracies 0.45–0.70). Every informed arm then always verified; `U(oracle_level) − U(global)` was exactly 0.
2. No confidence was elicited. Verbal arms ran with a constant 50, so label C was uninformative. A logprob design requested earlier was not implemented, and the v1 report did not flag that deviation (see the erratum on `reports/M34A_REPORT.md`).

---

## 1. Answerer and compute

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1` (re-verify with `model_info().sha` before collection; record in the manifest)
- Colab T4, `float16` default; if any non-finite logit or logprob appears, rerun that item in `float32` and flag it
- Greedy (`do_sample=False`), `max_new_tokens=60`
- Prompt (identical every call): `Compute A x B. Reply with only the integer.`
- Parse: first integer in the reply (commas stripped); unparsable = wrong
- **No verbal confidence anywhere**

## 2. Family and level selection (no new generation)

Family **`mul_n1` only** (n-digit × 1-digit, n = 2..10).

Selection uses **only** `reports/m34a_pilot2/levels.json` → `choice.family_accuracies.mul_n1`. **No v1 pool outcome is used.**

For each window of 6 contiguous levels:

`gap_hat = mean_L max(2 p_L − 1, 0.7) − max(2 p̄ − 1, 0.7)`

where `p_L` is the pilot-2 accuracy of that level and `p̄` is the window mean of `p_L`. Choose the maximum `gap_hat`; ties → **lower** levels. If `gap_hat < 0.05`, **STOP** (no collection).

**Chosen (computed, no generation):** levels **`[2, 3, 4, 5, 6, 7]`**, `gap_hat = 0.083333…`, `p̄ = 0.791667`. See `reports/m34a_v2/levels.json`.

## 3. Pools

Seed **34003**. Fresh operands, disjoint from pilot 1, pilot 2, and the v1 pool plan (`reports/m34a_pilot2/pool_plan.json`). Unit-tested.

| Pool | Per level | Role |
| --- | ---: | --- |
| CAL | 100 | `oracle_level`, `level_cal`, `level_lp_cal` (and GATE_C) |
| HIST | 60 | history sample for `global` / `self` / `shuffled` |
| TEST | 40 | unit of analysis |

1200 generations. Plan: `reports/m34a_v2/pool_plan.json`, committed **before** collection.

## 4. Collection fields (per generation)

Raw text; generated token ids; per-token logprob (`log_softmax` of each generated token, **EOS excluded**); `answer_logprob` = sum of those logprobs; `mean_logprob`; parsed answer; correct; latency; dtype; non-finite flag. Resumable JSONL cache. Manifest: commit hash, torch/transformers versions, device, dtype, runtime, cache SHA-256.

## 5. Decision problem

| Action | Utility |
| --- | ---: |
| answer correct | +1 |
| answer wrong | −1 |
| abstain | 0 |
| verify | +0.7 |

Primary: **V3** (answer / abstain / verify). V2 (answer / abstain) is descriptive.

Expected utilities: answer `2p−1`, abstain `0`, verify `0.7`. Ties: **answer**, then **verify**.

## 6. Arms (offline; static warm-start history)

History: `n_h ∈ {2, 5, 10, 25}` per level from HIST without replacement, **30** draws, primary **`n_h = 10`**. Unit: TEST problem (utility averaged over draws). Bootstrap: paired, **5000** draws, seed **23001**, 240 TEST problems, 95% percentile CI.

| Arm | `p` |
| --- | --- |
| `naive` | always answer |
| `global` | HIST accuracy pooled over levels (that draw) |
| `self` | `p_L = (correct_L + 1) / (n_h + 2)` from HIST |
| `shuffled` | `self` with a derangement of level labels per draw |
| `oracle_level` | true CAL accuracy per level |
| `oracle_instance` | 1 if the TEST answer is correct, else 0 (reference) |
| `level_cal` | `(correct+1)/(n+2)` per level from **CAL** (not HIST) |
| `level_lp_cal` | `(correct+1)/(n+2)` from CAL in the cell `(level, tercile of answer_logprob)` |

**Terciles:** edges per level from CAL `answer_logprob` at sample quantiles 1/3 and 2/3 (linear interpolation on the sorted list). Assignment: `lp ≤ e1` → low; `e1 < lp ≤ e2` → mid; `lp > e2` → high. **Ties at an edge go to the lower bin.** Empty CAL cell → fall back to `level_cal` for that level.

## 7. Gates and exhaustive labels (V3, `n_h = 10`)

**GATE_A:** `U(oracle_level) − U(global) ≥ 0.05` (point estimate). Else `A = NOT_INFORMATIVE`.

**GATE_C:** at least 3 levels with ≥ 15 correct and ≥ 15 wrong in CAL. Else `C = NOT_INFORMATIVE`.

**A** (if GATE_A):

- `SUPPORTED` if CI lower bound of `(self − global) > 0` and point `≥ 0.05` and CI lower bound of `(self − shuffled) > 0`
- `SUPPORTED_WEAK` if CI lower bound of `(self − global) > 0` otherwise
- else `NOT_SUPPORTED`

**C** (if GATE_C):

- `INTERNAL_VALUE` if CI lower bound of `(level_lp_cal − level_cal) > 0` and point `≥ 0.03`
- else `INTERNAL_NONE`

**Allowed `(GATE_A, GATE_C, A, C)` — no gaps, no other triples emitted:**

| GATE_A | GATE_C | A | C |
| --- | --- | --- | --- |
| false | false | NOT_INFORMATIVE | NOT_INFORMATIVE |
| false | true | NOT_INFORMATIVE | INTERNAL_NONE |
| false | true | NOT_INFORMATIVE | INTERNAL_VALUE |
| true | false | NOT_SUPPORTED | NOT_INFORMATIVE |
| true | false | SUPPORTED_WEAK | NOT_INFORMATIVE |
| true | false | SUPPORTED | NOT_INFORMATIVE |
| true | true | NOT_SUPPORTED | INTERNAL_NONE |
| true | true | NOT_SUPPORTED | INTERNAL_VALUE |
| true | true | SUPPORTED_WEAK | INTERNAL_NONE |
| true | true | SUPPORTED_WEAK | INTERNAL_VALUE |
| true | true | SUPPORTED | INTERNAL_NONE |
| true | true | SUPPORTED | INTERNAL_VALUE |

## 8. Descriptive only

Within-level AUROC of `answer_logprob` vs correctness (per level and mean); risk-coverage; per-level accuracy; V2; `n_h` curve.

## 9. Stop / integrity

- Analysis is committed **before** collection. At run time it reads `git rev-parse HEAD`, **refuses a dirty working tree**, and writes that hash into the report.
- Analysis reads only the cache; refuses if cache SHA-256 ≠ manifest.
- One collection (resumable crash recovery only). One analysis run. No retuning.
- **Whatever the labels, M34a is closed after v2.** No further variants without discussion.

## 10. Interpretation (fixed)

- A `SUPPORTED` and C `INTERNAL_VALUE` → track-record value plus instance-level logprob value on this task.
- A `SUPPORTED` and C `INTERNAL_NONE` → track-record calibration only; no instance-level claim.
- otherwise → no support at this scale (including any `NOT_INFORMATIVE`).

## 11. Artifacts

| Path | Role |
| --- | --- |
| `docs/M34A_V2_PREREG.md` | this file |
| `docs/M34A_V2_RUN.md` | Colab runbook |
| `reports/m34a_v2/levels.json` | selection from pilot-2 mul_n1 |
| `reports/m34a_v2/pool_plan.json` | 1200 operands, seed 34003 |
| `reports/m34a_v2_raw/` | cache + manifest |
| `reports/M34A_V2_REPORT.md` | one-run report |
