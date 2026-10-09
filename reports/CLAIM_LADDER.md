# Claim ladder (consolidation)

Consolidation only. No new experiments, audits, or protocols.
Research tip at tagging: `4078ea8cce1608f9ca498bd1070044d684c308e9` (`cursor/m30-holdout-lock-9d56`, S2 recorded).

Shared mechanistic scope (unless a section says otherwise): one model `Qwen/Qwen2.5-0.5B` revision `060db6499f32faf8b98477b0a26969ef7d8b9987`; last-token residual steering; alpha `+1`; outcome = last-position logit margin. Directions and layers differ by milestone and are listed per section.

---

## M21.5 (v1 benchmark defects)

**Verdict:** `M21.5_V1_CLAIM_NOT_ACCEPTED`

**Claimed.** One frozen synthetic utility run of agents B0–B3 under the v1 environment. Primary contrast B3 − BestSimple failed the preregistered acceptance criteria (DeltaU −0.009, CI below zero). Integrity audits (leakage, compute, reproducibility, immutability) passed. No language model was called.

**NOT claimed.** That a self-model is useless in general; consciousness; weight editing; that B3’s causal table is identified as the failure mode. The report itself notes the observed advantage may be explained by memory/reflection rather than causal self-modeling.

**Scope limits.** Synthetic tabular environment (`M21.5-env-v1.0`, impl `m21.5-impl-2`); no Qwen; seeds `11…223`; static/OOD/ablation controls only as recorded.

**v1 instrument defects (why the null says little about self-models).** Detector almost never fires (`b3_abs_error_threshold = 0.71` vs expected window MAE ~0.24–0.58); HIGH/MEDIUM share optimal actions so state-aware ceiling is tiny (~0.019 utility); no state recurrence; DSR = 0 for all agents; `no-update` and `no-intervention-prediction` ablations collapse to the same score.

**Links.** Branch `origin/cursor/m21-5-utility-benchmark-5fe5`; run commit `d058635376d2649d87c3971a2411d3a69482d447`; code freeze `900a01a76892ce33b26509e005c7d36b89bf910f`; report `M21.5/M21.5_v1_report.md` / `M21.5/results/M21.5_v1_report.md`. Not an ancestor of the M24–S2 tip.

---

## M21.6 (gate history; local-only risk)

**Role.** Redesign of the utility benchmark after M21.5 defects: validity / power / ceiling / detector-calibration gates **before** any evaluation-seed B3 run; Latin-square action matrix; 240-episode schedule with state recurrence; separate `evaluation/verdict.py`.

**Claimed (as history).** That a gate-first protocol was specified and that a local closeout path existed (`M21.6/results/verdict.json`, diagnostics under `M21.6/results/diagnostics/`). Prior agent record cites local commit `4793bbb` with push denied (403). M21.7 was design+feasibility only (copy of M21.6 on branch `m21-7`), with no evaluation-seed agent runs authorized in that tasking.

**NOT claimed.** Any remote-authenticated scientific verdict for M21.6/M21.7 in this repository tip. Do not treat chat summaries as hash-locked outcomes.

**Scope limits.** Still a synthetic agent benchmark (no Qwen mechanistic identity). Distinct from the M22–S2 last-token margin line.

**Remote / archive presence (this consolidation).**

| Artifact | On `origin` (checked 2026-10-08) | In this workspace |
| --- | --- | --- |
| `M21.6/` tree / verdict | **Absent** (no branch/tag matching m21-6 / M21.6) | **Absent** |
| `M21.7/` tree | **Absent** | **Absent** |
| `archive/*.bundle` for pre-M21.7 | **Absent** | **Absent** |

A prior cloud session mentioned `/workspace/cognitive-self-model-pre-m217.bundle` (~27 refs); that bundle is not in this checkout and was not pushed here. **M21.6/M21.7 are not present on the remote from this machine.** Recovery requires the local/cloud commit or that bundle from elsewhere.

**Links.** Instructions locally at `Downloads/M21.6_cursor_instructions.md` (not in-repo). No protocol/report/commit id on `origin` for the closed run.

---

## M24

**Verdict:** `CONSUMPTION_SUPPORTED`

**Claimed.** On a frozen held-out catalog, consuming the frozen mechanism response object `g` (pre-intervention) beat the stored training cell mean on absolute error; sign decision did not lose to that mean; shuffled predictions did not clear the same interval. Evaluation paired mean ≈ 0.0568, CI `[0.0388, 0.0772]`, class `CI_POSITIVE`; mechanism MAE ≈ 0.0059 vs baseline MAE ≈ 0.0627.

**NOT claimed.** A self-model; belief updating; residual/κ correction; transfer beyond this catalog/family; Layer 23 or D2 as mechanisms.

**Scope limits.** Model/revision as above; family only `M22.1-D1-L0/L8/L15` (`blocks.{0,8,15}.hook_resid_post`); direction D1 seed `22101`, sha256 `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411`; alpha `+1`; last-token margin tokens `9834`/`902` (" yes"/" no"); catalog sha256 `ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78`.

**Links.** Protocol `reports/M24_CONSUMPTION_PROTOCOL.md` (freeze commit `f27c37d`); execution `reports/M24_CONSUMPTION_EXECUTION.md` / `reports/M24_CONSUMPTION_RESULTS.json` (record commit `3af9edf`); raw `reports/m24_raw/`.

---

## M26

**Verdict:** `INCONCLUSIVE` (not “insufficient”)

**Claimed.** Independent residual-correction update was executed under the frozen protocol. Holdout primary intervals for cell and global corrections include zero (`CI_INCLUDES_ZERO`); shuffled control is `CI_NEGATIVE`. Leakage audit passed. Interpretation: correction is **unestablished**; no second prior/shrinkage/observable authorized.

**NOT claimed.** That evidence updating failed as a concept; that `g` itself failed (M24 stands); support or not-support of the δ_cell rule.

**Scope limits.** Same Qwen revision, D1 family/layers/alpha/margin as M24; catalog sha256 `015c3d41855da5fa630c382fe0e4a9ea63c868e12e4a1243e104e1c4e371ea6f`; protocol version `M26.INDEPENDENT_EVIDENCE_UPDATE.1`.

**Links.** Protocol `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_PROTOCOL.md` (freeze `a813882`); execution `reports/M26_INDEPENDENT_EVIDENCE_UPDATE_EXECUTION.md` / `RESULTS.json` (execution commit `55c3ee5`, record `1e18346`); raw `reports/m26_raw/`.

---

## M29-D

**Verdict:** `SUPPORTED`

**Claimed.** A genuinely pre-outcome internal measurement `kappa = α²/2 · F''(0)` (Hessian–vector product at the unperturbed state), combined as `ŷ = g + kappa`, beats baselines on the frozen holdout under the preregistered rule. Holdout bootstrap mean Δ ≈ 0.0157, CI `[0.0098, 0.0223]`, class `CI_POSITIVE`; MAE(M29) ≈ 0.0045 vs MAE(B0) ≈ 0.0213 / MAE(B1) ≈ 0.0202. Predictions were hashed before outcomes.

**Frozen hashes.** Prediction sha256 `9ff6a11e6ad28128d9446db66e66f67057a4ae7ba6a4cfbf886297af6aeda33a`; outcome sha256 `686689d5c9b6039af3027c087c21988ccfabbebb4965d3814d22a08aa8e811e9`; catalog sha256 `d6ecab612a2214d947be11c82671612f883c0d13e479b07e8cbaa09b635d4d77`.

**NOT claimed.** Privileged access vs cheap probing; decision utility in agent tasks; competition with other methods; transfer to new identities (that is M30); that M26’s δ update is rescued.

**Scope limits.** Same Qwen revision; D1 layers 0/8/15; alpha `+1`; last-token margin yes/no; float32; finite differences forbidden for the M29 measurement.

**Links.** Protocol `docs/M29_D_PROTOCOL.md`; freeze `reports/M29_D_EXECUTION_FREEZE.md`; verdict `reports/m29d_raw/VERDICT.json` (record commit `7f62465`); manifests `reports/m29d_raw/PRE_OUTCOME_MANIFEST.json`, `OUTCOME_MANIFEST.json`.

---

## M30

**Verdicts:** new identity `SUPPORTED`; anchor `ANCHOR_SUPPORTED`

**Claimed.** On an enlarged catalog (HOLDOUT n=24), the same κ estimator under a **new** frozen identity (D2, layers 4/12/20, margin `" true"`−`" false"`) beats `g` with relative reduction **0.858** (exact 0.8576368215994614) and absolute MAE(g) **≈ 0.0022** (0.002189945552446362). The M29-D identity as anchor on the same catalog is separately `ANCHOR_SUPPORTED` with relative reduction **0.660** (exact 0.6598186836856698). 2×2: both identities beat `g` under the frozen rule. Leakage audit passed.

**NOT claimed.** That the new identity is universally better; decision value (S1); white-box privilege over FD probing (S2); agent-task utility; competition.

**Scope limits.** Qwen2.5-0.5B revision as above; new identity direction D2 seed `22103` sha256 `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e`, layers 4/12/20; anchor = M29-D D1 L0/L8/L15 yes/no; alpha `+1`; last-token residual steering; catalog sha256 `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`; predictions sha256 `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`; protocol `M30.PROTOCOL.2` sha256 `1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3`.

**Links.** `docs/M30_PROTOCOL.md`; `reports/M30_STAGE2_REPORT.md` / `reports/m30_raw/VERDICT.json` (outcomes commit `50147c8`; runner `70b0521`).

---

## S1

**Decision:** `UNDEFINED_BY_RULE` (erratum; earlier `STOP` withdrawn)

**Claimed.** Steering-scale decision experiment was run once on M30 HOLDOUT under the preregistration. Numbers are frozen. A literal reading of the three decision clauses yields neither `GO`, `PIVOT`, nor `STOP`: GO fails (new-identity s=4 CI lower bound negative); PIVOT fails (M2 median miss is smaller than BB_3 on the decision pairs); STOP fails (M2 beats M1 on three of four decision conditions).

**NOT claimed.** That κ has no decision value; that STOP/GO was scientifically decided; agent-task utility beyond this margin-steering toy.

**Scope limits.** Same Qwen revision and M30 arms/cells; scales s∈{1,2,4,8}; equal-cost black box `BB_3` (3 forward-equivalents); prediction/catalog hashes must match M30.

**Links.** `docs/S1_PREREG.md`; `reports/S1_REPORT.md` (erratum commit `f31ad65`; run record `32aedac`; runner `cb1b582`); raw `reports/s1_raw/`.

---

## S2

**Verdict:** `MIXED` — white-box advantage **0/4** decision conditions; FDQ (3 forwards) matches M2 on median miss and wall time

**Claimed.** Finite-difference quadratic (FDQ) from `F(+1)` and `F(−1)` plus shared `f0` was compared to white-box M2 on the S1 holdout task. Decision classes: `WHITE` 0, `NO_ADVANTAGE` 1, `NEITHER` 3. FDQ median seconds ≈ 0.325 vs M2 ≈ 0.307 (M2 time reused from S1); FDQ forward-equivalents = 3. Median misses of FDQ and M2 are close across arms/scales (see report tables).

**NOT claimed.** That autodiff κ is useless for prediction (M29-D/M30 stand); that black-box probing is universally sufficient outside this task; competition or agent utility.

**Scope limits.** Identical to S1 prompts/cells/targets; only FDQ newly measured; scales 1–2 descriptive; decision scales 4 and 8.

**Links.** `docs/S2_PREREG.md`; `reports/S2_REPORT.md` (commit `4078ea8`; runner `2e5fd7a`); raw `reports/s2_raw/`.

---

## S-SIM (simulation study; `simulations/ssim/`)

**Verdict:** pre-registered predictions in `SIM_PREREG.md`: 9 met, 4 not met (P3, P7 second part, P11, P14). Not a hash-locked Qwen result.

**Claimed.** Simulated learning agents with an injected property of the agent itself (execution bias, habit, error profile). A self-model that learns the property from the agent's own history closes about 87% to 99% of the gap to an agent that knows it (budgeting, hide-and-seek against an adaptive predictor, verify-or-abstain). No gain against a static opponent; no change in unaided accuracy; gains when allocation options exist. In a head-to-head match the self-model agent wins by 0.198 hits per round, identical agents tie. Sophistication can hurt (quasi-hyperbolic benchmark reproduced). Opponent model and self-model are substitutes when the opponent's predictions are visible.

**NOT claimed.** Anything about LLM agents; that the placebo checks (P3, P7, P11) are interpretable (the placebo carried partial information); that the self-model beats global calibration in E3 V3 (the self-model never learned there: P14, an exploration trap).

**Scope limits.** Injected properties, observed own execution, tabular agents, 30 seeds, one run.

**Links.** `simulations/ssim/SIM_PREREG.md`, `SIM_RESULTS.md`, `sim_all.py`, `sim_results.json`.

## Exploratory (post hoc): reliability signal

On M30 outcome rows, abs(kappa) predicts the remaining error after correction (Spearman 0.764 [0.66, 0.83] new identity; 0.605 [0.44, 0.72] anchor, holdout), far above abs(g). Keeping the 25% lowest abs(kappa) rows cuts the mean remaining error about 10x. Not pre-registered; a finite-difference probe computes the same quantity, so it gives no privileged access.

---

## M34a (track-record self-knowledge; Amendment 1)

**Verdict:** A=`NOT_INFORMATIVE` · C=`INTROSPECTION_NONE` · GATE_OK=`false` · SPREAD_OK=`true` · VAR_OK=`true`

**Claimed.** One frozen collection of 1200 greedy Qwen2.5-3B-Instruct answers on `mul_n1` (n-digit × 1-digit, levels 5–10). Cache sha256 `290307e5b95a062ee00141f43467c660ef48ddae95d6ba282f1269baaa200f99`. V3 utilities of `oracle_level` and `global` both 0.700 (always verify); GATE gap 0.000 (need ≥ 0.05). Self did not beat global (mean −0.0014, CI includes 0). Interpretation: **no support at this scale**.

**NOT claimed.** Privileged access; mechanistic κ; planning/games (M34b/c); that verify-or-abstain has no value in other tasks; that a confidence prompt was tested (Amendment 1 asks for the integer only; stated confidence defaults to 50).

**Scope limits.** One Instruct Qwen 3B, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, Colab T4 float16, greedy, family `mul_n1` only, levels 5–10, static warm-start history, n_h=10 primary, 240 TEST problems.

**Links.** `docs/M34A_PREREG.md` (Amendment 1); `reports/m34a_pilot2/`; `reports/M34A_REPORT.md`; `reports/m34a_raw/`; collection commit `ece1d19`.

---

## Claim ladder

1. **Pre-outcome prediction of own response:** **supported in scope** (M24 consumption of `g`; M29-D `SUPPORTED`; M30 new identity `SUPPORTED` + `ANCHOR_SUPPORTED`). Scope = one small Qwen, listed directions/layers, last-token margin, alpha steering as above. Exploratory: abs(kappa) is a reliability signal.
2. **Privileged access over cheap probing:** **NOT supported** (S2: 0/4 white-box advantage; FDQ with 3 forwards matches M2 on median miss and time). Learned external observer: untested (planned M33).
3. **Decision utility in agent tasks:** **supported in simulation only** (S-SIM, injected properties). **M34a (real LLM, Amendment 1):** GATE failed (`NOT_INFORMATIVE`); no support at this scale. S1 is margin-steering only and ended `UNDEFINED_BY_RULE`.
4. **Competition:** **simulation only** (S-SIM match); with a visible opponent model the self-model is a substitute, not a complement. LLM competition untested.
5. **Cross-weights specificity and stability after modification:** untested (planned M31, M32).
