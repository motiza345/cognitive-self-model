> **Archived / partly superseded.** Stages M41 to M43 of this plan were replaced by `docs/LANDSCAPE_AND_PIVOT.md` and `docs/ROADMAP_COGNITIVE_LAYER.md` after a prior-art review. Kept for the record.

# M40 program: comprehensive Colab test plan for the remaining self-model hypotheses

Status: plan, not a pre-registration. Each stage below becomes its own pre-registration file (`docs/<ID>_PREREG.md`) written and pushed before any outcome exists, following `AGENTS.md`. This plan fixes the design logic so the stages can be executed one after another, with go/no-go gates, on Colab (free-tier T4-class GPU or CPU runtime) and without any paid API.

## 0. Why this program, and what it must not repeat

**What is still untested** (see `reports/TECHNICAL_REPORT_v1.1.md`, claim ladder): real LLM agents in games and planning; whether the benefit appears when the self-model is a module around an LLM versus text the LLM must read; cross-weights specificity and stability after modification; a learned external observer; generalization of the instance-level own-output-probability result.

**Failure modes of earlier tests, and the safeguard written into every stage below**

| earlier failure | safeguard |
|---|---|
| Benchmark could not reward a self-model (M21.5) | A **ceiling gate**: the oracle self-knowledge arm must beat the baseline by a stated margin in the pilot, or the stage stops as `NOT_INFORMATIVE` / `UTILIZATION_FAILURE` |
| Utility geometry made one action dominant (M34a v1) | Conditions are chosen so **both sides of each decision boundary occur**, using a pilot with 40 items per level |
| Gate on within-level variance failed (M34a v3 S1) | The pilot rule requires at least 3 levels with accuracy in [0.25, 0.75] |
| Placebo carried information (S-SIM) | The placebo is a **true null** (a zero-information estimate with matched noise); cross-agent placebos are secondary |
| Safe option never taught the agent (S-SIM E3) | Every learning arm has forced exploration or feedback on all actions |
| Decision rule not exhaustive (S1) | Every label table in this plan is exhaustive; the pre-registration must keep that property |
| Weak baseline (M29-D B1) | Baselines always include "no correction / no history" and the strongest cheap alternative (opponent model, probe, track record) |
| Substitutes (opponent model, probe) matched the self-model | Substitution arms are mandatory and a `SUBSTITUTE` label exists |

## 1. Hypotheses

| ID | Hypothesis | Stage |
|---|---|---|
| H1 | A pre-outcome self-measurement keeps predicting the response of other weights (cross-weights replication) but not by borrowing another model's measurement (specificity) | M31 |
| H2 | After the model is modified, a re-measured self-model tracks the change; a stale one does not | M32 |
| H3 | A learned external observer that sees only behaviour does not match the white-box measurement (privileged access) | M33 |
| H4 | The instance-level own-output-probability value generalizes across models and tasks; per-level track-record knowledge adds little | M41 |
| H5 | An LLM can use a table of its own accuracy in context (utilization), and a module can use it better | M41 (parts b, c) |
| H6 | In a repeated game against an adaptive opponent, an LLM agent with a self-model is less predictable; no effect against a static opponent | M42 |
| H7 | When the opponent's predictions are visible, an opponent model substitutes for the self-model; when hidden, it does not | M42 |
| H8 | In a head-to-head match, the agent with the self-model wins; identical agents tie | M42 |
| H9 | With an execution bias in planning, an LLM planner plus a self-model module beats the planner alone, with no effect when there is no bias | M43 |

## 2. Shared infrastructure (build once; Stage 0)

- **Workflow per stage:** pilot, gate, freeze (pre-registration pushed), collect once, analyse offline from the cache once, report. Never change a rule after seeing outcome data.
- **Colab runbook** for each stage (`docs/<ID>_RUN.md`): select GPU runtime, `git clone` and `git checkout <commit>`, `pip install -r requirements-m34a.txt`, run the pilot, download the gate file, give it to the coding agent to commit and push, run collection with `--resume`, hash the cache, zip, download. Cache every generation as one JSON line; mount Google Drive optionally so a disconnect loses nothing.
- **Batching:** episodes (seeds) are stepped in lock-step so one forward pass handles many seeds. This is the main lever for fitting the free tier.
- **Models:** `Qwen/Qwen2.5-3B-Instruct` (about 6 GB in fp16) as the main agent; `Qwen/Qwen2.5-1.5B-Instruct` and `microsoft/Phi-3.5-mini-instruct` (about 7.6 GB) as replications. A 7B model in fp16 does not fit a 16 GB T4 comfortably; do not substitute quantized weights without a protocol amendment.
- **Manifest** per run: model id and revision (resolved before collection), dtype, device name, library versions, git commit, seeds, cache SHA-256, runtime. Analysis refuses to run on a dirty tree or a hash mismatch.
- **Units and statistics:** unit is the episode (games, budgeting) or the test problem (M41); paired percentile bootstrap, 5000 draws, seed 23001, 95% interval; effect-size floors stated per label; Holm correction over secondary contrasts only.
- **Exploration of the agent's own outcomes** is forced in all learning arms (epsilon 0.1 on the choice distribution, or feedback on every action).

## 3. Stage 1: mechanistic completion (CPU runtime is enough; float32)

Specs are in `docs/NEXT_STEPS.md` (M31, M32, M33). Summary of decisions fixed here:

- **M31** model B `Qwen/Qwen2.5-0.5B-Instruct`, M30 catalog and both identities unchanged, labels `SUPPORTED` / `SUPPORTED_WEAK` / `INCONCLUSIVE` for own predictions and `SPECIFIC` / `NOT_SPECIFIC` / `MIXED` for scoring the frozen M30 predictions on model B outcomes.
- **M32** two pre-fixed modifications on Qwen2.5-0.5B (zero the MLP output at layer 10; zero the attention output at layer 6). Informative only if the median absolute observed effect changes by at least 25%. Arms: fresh (re-measured), stale (frozen M30 predictions), `g` only. Labels `TRACKS`, `DOES_NOT_TRACK`, `NOT_INFORMATIVE`.
- **M33** learned external observer (ridge, k-NN) trained on at least 500 fresh template prompts with measured outcomes, behaviour-only features, scored on the M30 holdout against `g + kappa` and `g`. Labels `PRIVILEGED` (MAE(g + kappa) at most 0.75 of the observer's, interval above zero), `NO_ADVANTAGE`, `MIXED`. Report the training-size curve.
- Compute: M30 took about two minutes locally; M33 needs roughly 3000 measurement rows, so budget one to two hours on a Colab CPU runtime (estimate, not measured).

## 4. Stage 2 (M41): generalization of the competence self-knowledge result (GPU)

**Purpose.** Replace the single (model, task) result of M34a v2 by a pre-registered replication grid, fix the gate geometry, and add the part the earlier tests lacked: whether the LLM itself can use self-knowledge in context.

**M41a: grid.** Models {Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, Phi-3.5-mini-instruct} times tasks {`mul_n1`: n-digit by one digit, n = 2..10; `add_nn`: n = 3..15; `sort_n`: sort n distinct two-digit integers, n = 3..12; exact-match scoring}. 9 conditions, fresh seeds and pools (do not reuse earlier caches).
- **Pilot per condition:** 40 problems per candidate level, greedy, fp16.
- **Level selection (fixed now):** over windows of 6 contiguous levels, maximize `gap_hat = mean over levels of max(2p - 1, 0.7) - max(2*pbar - 1, 0.7)` (p = pilot accuracy, pbar = window mean), subject to at least 3 levels with pilot accuracy in [0.25, 0.75]; ties go to lower levels. If no window qualifies the condition is `NOT_INFORMATIVE` and not collected.
- **Pools:** CAL 100, HIST 60, TEST 40 per level (about 1200 generations per condition).
- **Decision problem and arms:** as in M34a v2: answer (+1 / -1), abstain (0), verify (+0.7). Arms naive, global, self (track record from a history sample, n_h in {2, 5, 10, 25, 50}, primary 10), shuffled, level_cal, level_lp_cal (within-level terciles of `answer_logprob`, calibrated on CAL), oracle_level, oracle_instance. Gates and labels `A` and `C` exactly as in `docs/M34A_V2_PREREG.md`.
- **Aggregation (exhaustive).** Let I = number of conditions where the relevant gate passes, V = number with the positive label (for C: `INTERNAL_VALUE`; for A: `SUPPORTED` or `SUPPORTED_WEAK`). If I < 4: `*_NOT_INFORMATIVE`. Else if V/I >= 0.8: `*_GENERAL`; else if V/I >= 0.5: `*_PARTIAL`; else `*_NOT_REPLICATED`.
- **Descriptive:** dose-response of A in n_h; within-level AUROC of `answer_logprob` per condition; risk-coverage.

**M41b: verbal confidence.** For `Qwen2.5-3B-Instruct` on `mul_n1`, a second collection in which the model also states a confidence from 0 to 100. Arms `level_lp_cal`, `level_verbal_cal` (within-level terciles of stated confidence), `level_lp_verbal_cal`. Labels: `VERBAL_ADDS` if the combined arm beats `level_lp_cal` (interval above zero, at least 0.02); `VERBAL_WORSE` if `level_verbal_cal` is below `level_lp_cal` (interval below zero); else `VERBAL_EQUIVALENT`.

**M41c: in-context utilization.** For `Qwen2.5-3B-Instruct` on the `mul_n1` setting chosen in M41a, the model itself chooses answer, abstain or verify after reading a table. Arms: no table; own per-level accuracy table (from CAL); shuffled table (level labels permuted by a derangement); global accuracy only; program decision with the same table (reference). Utility on TEST (240 problems), bootstrap over problems. Labels: `UTILIZED` if table minus no-table and table minus shuffled are both above zero (intervals) and the table-arm closes at least 30% of the gap to the program decision; `PARTIAL_UTILIZATION` if only the first holds; `UTILIZATION_FAILURE` otherwise. An LLM that cannot use a table in context is an informative result for the "text the LLM must read" design.

**Compute (estimate from M34a, 0.56 s per generation, not re-measured):** 9 conditions about 15 minutes each, plus M41b and M41c about 1 hour: roughly 3 to 4 GPU hours.

## 5. Stage 3 (M42): repeated game, with and without a self-model (GPU) — tests H6, H7, H8

**Game.** Three-place hide-and-seek, 200 rounds per episode (rounds 51 to 200 scored), 40 episodes per cell stepped in lock-step. The LLM agent hides; a program seeker guesses; the agent's score is 1 minus the seeker's hit rate (chance 1/3). The seeker's guess is shown after each round.

**Agent.** `Qwen2.5-3B-Instruct`; the prompt asks for one of A, B or C; the distribution `p_LLM` is read from the logits of the three option tokens; the action is sampled from the arm's policy `q` with a seeded generator. LLMs are known to be poor and predictable randomizers, so a natural property exists without injection; an optional factor adds an injected habit `h` in {0, 0.3} (the executed move repeats the previous executed move with probability `h`).

**Opponents.** static (always guesses A) and adaptive (order-1 frequency table conditioned on the agent's previous executed move, decay 0.98).

**Arms, module-mediated (the architecture the project proposes: an LLM wrapped by a cognitive layer):**
1. `base`: `q = p_LLM`.
2. `self_module`: `q(a)` proportional to `p_LLM(a) * (1/3) / c(a | previous move)`, where `c` is the Laplace-smoothed, decayed empirical frequency of the agent's own executed moves in that context (pushes the executed distribution toward uniform).
3. `self_null`: the same correction with a zero-information estimate (uniform `c` plus noise matched to the learning noise of arm 2): a true null.
4. `self_cross`: `c` borrowed from another episode (secondary placebo).
5. `opp_module`: `q(a)` proportional to `p_LLM(a) * exp(-g(a) / 0.15)`, where `g` is the empirical distribution of the seeker's recent guesses in the same context (visible predictions).
6. `both`: arms 2 and 5 combined.
7. `oracle_module`: `c` computed from the true executed-move distribution given `p_LLM` and `h` (reference, not a competitor).
Epsilon-greedy forced exploration is applied to the learned estimates (epsilon 0.1).

**Arms, in-prompt (text the LLM must read):** `none`; `raw_history` (own last 10 moves and results, as a control for "any history"); `self_summary` (own move-transition frequencies and overall predictability); `self_shuffled_summary` (table from a different episode); `opp_summary`; `both_summary`; `oracle_summary` (true habit and bias stated in words).

**Hidden-prediction variant:** the seeker's guesses are not shown; `opp_module` must learn from hit or miss only.

**Pilot and gates.** 10 episodes per arm. **Utilization gate U:** `oracle_module` (and separately `oracle_summary`) must reduce the adaptive-opponent hit rate by at least 0.05 against `base`; otherwise that agent type is labelled `UTIL_FAIL` (an informative finding) and the other arms for that type are descriptive only.

**Labels for each agent type (exhaustive).**
- If U fails: `UTIL_FAIL`.
- Else `SELF_SUPPORTED` requires all of: (a) `base - self_*` at least 0.03 with interval above zero against the adaptive opponent; (b) `self_null - self_*` interval above zero; (c) against the static opponent |`self_* - base`| at most 0.02. `SELF_WEAK`: (a) holds, but (b) or (c) fails. `SELF_NOT_SUPPORTED`: (a) fails.
- Substitution (visible-prediction condition, only if `SELF_SUPPORTED` or `SELF_WEAK`): `COMPLEMENT` if `opp_module - both` is at least 0.02 with interval above zero; `SUBSTITUTE` if that interval includes zero and `base - opp_module` is at least 0.03; `OTHER` otherwise. In the hidden-prediction condition report `self_* - opp_module` descriptively.

**Matches (H8).** Two agents of the arms named, each round both hide and seek with the same adaptive seeker, score is seeker hits of A minus seeker hits of B, 40 matches per pairing: `self_module` vs `base`, `base` vs `base`, `self_module` vs `self_module`, `self_null` vs `base`, `opp_module` vs `base`. `MATCH_SUPPORTED` if `self_module` vs `base` is above zero with interval above zero and at least 0.05, and both identical pairings are within 0.02 of zero and `self_null` vs `base` is at most 0.02; `MATCH_PARTIAL` if only the first holds; `MATCH_NOT_SUPPORTED` otherwise.

**Compute estimate:** a forward pass of a short prompt is well under 0.1 s; with 40 episodes batched this is minutes per arm. Budget about 2 GPU hours for all arms and both opponents (estimate, not measured).

## 6. Stage 4 (M43): planning with an execution bias (GPU) — tests H9

**Task.** Four spending categories, 24 periods, weights `w = [0.4, 0.3, 0.2, 0.1]`, per-period utility `sum w_k sqrt(x_k) - 4 * max(0, sum x - 1)` (as in S-SIM). The LLM planner receives the category importances and produces a plan as four numbers in a fixed format (invalid output counts as a failure and falls back to the equal split, with the failure rate reported). The executed spending is `max(0, plan + b + noise)`; `b` is a bias vector on one random category of size `beta * U(0.1, 0.3)`, `beta` in {0, 1}, noise standard deviation 0.03. The planner sees its plan and the executed amounts of past periods only in the history arms.

**Arms.** `llm_only`; `self_module` (running mean of executed minus planned subtracted from the next plan); `self_null` (a zero-information estimate with matched noise, true null); `self_cross` (estimate from another seed); `aggregate_only` and `allocation_only` (decomposition of the estimate); `raw_history_prompt` and `self_summary_prompt` (text the LLM must read); `oracle_module`; `program_planner` (the analytic optimum, reference).

**Pilot and gate:** 10 seeds; at `beta = 1`, `oracle_module` must exceed `llm_only` by at least 0.05 in mean utility, else `NOT_INFORMATIVE`.

**Labels (exhaustive).** `BUD_SUPPORTED` if: at `beta = 1`, `self_module - llm_only` has interval above zero and closure to the oracle at least 0.5; at `beta = 0`, |`self_module - llm_only`| at most 0.005; and `self_null - llm_only` at most 0.002. `BUD_WEAK` if the first holds but the others do not. `BUD_NOT_SUPPORTED` otherwise. Separately label the in-prompt agent `UTILIZED` / `PARTIAL_UTILIZATION` / `UTILIZATION_FAILURE` as in M41c.

**Compute estimate:** 30 seeds batched; 24 periods; about 8 arms; one to two GPU hours (estimate).

## 7. Stage 5 (optional, M36): GPT-2 small IOI analogue

Repeat the curvature measurement and its finite-difference control on GPT-2 small IOI prompts with a head-scaling intervention, same acceptance and substitution logic as M30 and S2. Only if time remains; it tests another architecture family, not an agent claim.

## 8. Order, dependencies and stop rules

Recommended order, by value for the hypotheses about agents and by cost: **M42 first**, then **M43**, then **M41**, then **Stage 1 (M31, M32, M33)**, then M36 if wanted. Stages are independent; shared code is built once.

- Stop a stage at its pilot if its gate fails (label as defined). A stage never reruns with new thresholds.
- After M42 and M43, decide whether the "text-the-LLM-must-read" design is viable (`UTILIZED`) or whether only the module design works.
- No stage may start collection before its pre-registration and scripts are pushed (or, if the network is unavailable, committed locally and pushed before the analysis, with the commit ids recorded).

## 9. Synthesis rule (fixed now)

The program is read as follows, with every claim carrying its own label:

| claim | evidence |
|---|---|
| C-privileged: the self-measurement has access others lack | M33 `PRIVILEGED` |
| C-general: own-output probability value generalizes | M41 C aggregate `C_GENERAL` or `C_PARTIAL` |
| C-games: a self-model helps LLM agents in games | M42 `SELF_SUPPORTED` (module form) and `MATCH_SUPPORTED` |
| C-planning: a self-model helps LLM planning under bias | M43 `BUD_SUPPORTED` |
| C-substitution: when other signals are visible the self-model is redundant | M42 `SUBSTITUTE` |

Project verdict: **continue** (as a research programme worth further investment) only if at least two of {C-games, C-planning, C-general} are supported with true-null placebos and the substitution controls reported. Otherwise the verdict is: the explicit self-model module offers little beyond existing signals in the tested regimes, and the programme is written up as v1.1 of the technical report. A failure of the utilization gate is itself reported as a finding about LLM agents.

## 10. Task cards for the coding agent (Cursor or Codex)

Give one card at a time. Each card: write `docs/<ID>_PREREG.md` from this plan (exhaustive rule, arms, gates, seeds, sample sizes, stop rule) and push it; write scripts and tests with a mock model so that tests pass offline; write `docs/<ID>_RUN.md` with the Colab commands; stop and wait for the downloaded cache; then run the offline analysis once and write `reports/<ID>_REPORT.md` with Claimed / NOT claimed / Scope.

Suggested files: `scripts/m42_game_lib.py` (agents, seekers, arms, lock-step batching, model-agnostic so it can be tested with a mock policy), `scripts/m42_collect.py`, `scripts/m42_analyze.py`, `tests/test_m42.py`; the same pattern for `m43` and `m41`.

## 11. Total budget (estimates, not measured)

Stage 1 about 2 to 3 CPU hours; M41 about 3 to 4 GPU hours; M42 about 2 GPU hours; M43 about 1 to 2 GPU hours. About 8 to 11 hours of Colab in several sessions, plus coding-agent time for the cards. Free-tier availability varies; resumable collection and batching are there to make short sessions sufficient.
