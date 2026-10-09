# M34a — Track-record self-knowledge (answer / abstain / verify)

**Status:** DESIGN_LOCKED (answerer/compute amended to local Qwen; no pilot or pool data yet)  
**Branch:** `cursor/m34a`  
**Question:** Does self-knowledge from a track record improve a real LLM's answer/abstain/verify decisions beyond global calibration, and does the LLM's own stated confidence add instance-level value beyond a difficulty-level track record?

This file is the pre-registration. Outcomes must not be collected until this file is committed and pushed, the collection/analysis scripts are committed and pushed, and (after the Colab pilot) `reports/m34a_pilot/levels.json` plus the pool plan JSON are committed and pushed.

---

## 1. Answerer and compute (amended)

- **Primary model:** `Qwen/Qwen2.5-3B-Instruct`
- **Fallback (only if 3B cannot be loaded):** `Qwen/Qwen2.5-1.5B-Instruct`
- **Revision:** resolve with `huggingface_hub.model_info(<repo_id>).sha` **before** collection; record that SHA in the collection manifest. Do not change the revision after the first successful load for a run.
- **Hardware:** Google Colab **T4** GPU.
- **dtype:** `float16` by default. **Guard:** if any non-finite logit or logprob appears for an item, **rerun that item in `float32`**, keep the float32 result, and flag the item in the manifest (`float32_rerun: true` for that problem id).
- **Decoding:** greedy (`do_sample=False`), no tools, no chain-of-thought scaffolding beyond the fixed user prompt.
- **max_new_tokens:** 60.
- **Prompt (identical for every call):**
  `Compute A x B. Reply exactly in this format: Answer: <integer>; Confidence: <0-100 probability your answer is exactly correct>`
- **Parse rule:** answer = first integer after `Answer:` (commas stripped); anything unparsable = wrong. Confidence unparsable = 50. Store raw responses, request params, and token counts.
- **Manifest must record:** model id, revision SHA, device name, dtype policy, torch version, transformers version, git commit used for the run, total runtime seconds, float32-rerun ids, input/output token totals, cost estimate N/A (local GPU).

## 2. Task

- Multiplication of two **n-digit** integers (no leading zeros), `n ∈ {2,3,4,5,6,7,8}` (7 candidate levels).
- Random seed for problem generation: **34001**.
- STEP 0 pilot (discarded afterwards for analysis): 20 problems per candidate level (140 calls). Choose **6 contiguous** levels maximizing `(max accuracy − min accuracy)`; ties → **lower** levels. If no window has range `≥ 0.5`, **STOP** and report (no main experiment). Commit `reports/m34a_pilot/levels.json` before any pool generation/collection.
- **Chosen levels:** *TBD — filled from the Colab pilot `levels.json` before pool plan commit.* Placeholder field: `chosen_levels: null` until that commit.

## 3. Pools (after levels chosen)

Per chosen level, disjoint from each other and from pilot problems:

| Pool | Per level | Role |
| --- | ---: | --- |
| CAL | 100 | oracle_level accuracies only |
| HIST | 60 | history sample source |
| TEST | 40 | unit of analysis |

Total **1200** generations. Pool operands and problem ids live in a committed plan JSON (`reports/m34a_pilot/pool_plan.json`), generated once from seed 34001 after levels are fixed, and **committed before collection**. Collection reads only that plan; it never reads analysis code.

## 4. Decision problem

Utilities on each TEST problem:

| Action | Utility |
| --- | ---: |
| answer correct | +1 |
| answer wrong | −1 |
| abstain | 0 |
| verify | +0.7 (always treated as correct) |

- **V3** (answer / abstain / verify): **primary**.
- **V2** (answer / abstain): descriptive only.

Agent picks the option with highest expected utility given estimate `p` of being correct:

- answer: `2p − 1`
- abstain: `0`
- verify: `0.7`

Ties: **answer**, then **verify**.

## 5. Arms (offline from the same cached answers)

Sequential history is **not** used. Static warm-start history only.

| Arm | `p` |
| --- | --- |
| `naive` | always answer (no `p` gate; always choose answer) |
| `none` | `p = 0.5` |
| `global` | accuracy over the history sample (pooled over levels) |
| `self` | `p_L = (correct_L + 1) / (n_h + 2)` per level from history |
| `shuffled` | `self` with level labels permuted (a derangement) per history draw |
| `verbal_raw` | stated confidence / 100 on that TEST problem |
| `verbal_cal` | stated confidence binned `[0–59, 60–79, 80–89, 90–94, 95–100]`; bin accuracy Beta(1,1) from history pairs; empty bin → global `p` |
| `combined` | mean of `self` and `verbal_cal` |
| `oracle_level` | true accuracy per level from the CAL pool |
| `oracle_instance` | knows if the answer is correct (reference only) |

History sample: `n_h` per level drawn without replacement from HIST, **30 draws**, `n_h ∈ {2, 5, 10, 25}`; **primary `n_h = 10`**.

## 6. Statistics

- Unit of analysis: TEST problem (utility averaged over the 30 history draws).
- Paired bootstrap over the **240** TEST problems, **5000** draws, seed **23001**, 95% percentile CI.

## 7. Exhaustive labels (V3, `n_h = 10`)

**GATE:** `U(oracle_level) − U(global) ≥ 0.05` (point estimate). Else `A = NOT_INFORMATIVE` (and do not claim support for A).

**A (track-record self-knowledge):**

- `SUPPORTED` if CI lower bound of `(self − global) > 0` **and** point estimate `≥ 0.05` **and** CI lower bound of `(self − shuffled) > 0`
- `SUPPORTED_WEAK` if CI lower bound of `(self − global) > 0` but not `SUPPORTED`
- otherwise `NOT_SUPPORTED`

**B (instance-level introspection):**

- `INTROSPECTION_VALUE` if CI lower bound of `(verbal_cal − self) > 0` **and** point estimate `≥ 0.03`
- otherwise `INTROSPECTION_NONE`

Every `(GATE, A, B)` combination maps to exactly one recorded triple; no gaps.

## 8. Descriptive only

Utilities of all arms and all `n_h`; V2 results; selective accuracy; risk-coverage; ECE of stated confidence; per-level accuracy; `verbal_raw` vs `naive`.

## 9. Interpretation (fixed)

- A `SUPPORTED` and B `INTROSPECTION_VALUE` → real LLM shows state-dependent self-knowledge value plus instance-level calibrated verbal signal.
- A `SUPPORTED` and B `INTROSPECTION_NONE` → the value is plain calibration from a track record; no project-specific introspection claim.
- otherwise → no support at this scale.

## 10. Stop / integrity rules

- One pilot, one pool-plan generation, one collection pass (resumable cache only for crash recovery; no re-prompting of completed ids).
- Analysis reads **only** the cached responses; refuses to run if the cache SHA-256 differs from the manifest.
- Do not change thresholds, arms, or labels after seeing results.
- Do not use any TEST or CAL outcome before the analysis script is committed.
- Budget: local Colab GPU; abort the design only if the model cannot be loaded even at fallback.

## 11. Artifacts

| Path | Role |
| --- | --- |
| `docs/M34A_PREREG.md` | this file |
| `docs/M34A_RUN.md` | Colab runbook |
| `reports/m34a_pilot/levels.json` | pilot choice (after Colab) |
| `reports/m34a_pilot/pool_plan.json` | committed operands before collection |
| `reports/m34a_raw/` | JSONL cache + manifest |
| `reports/M34A_REPORT.md` | one-run report |
| `reports/CLAIM_LADDER.md` | append-only update after report |
