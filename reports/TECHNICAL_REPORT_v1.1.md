# Pre-registered tests of self-models in small language models: one narrow positive result, no privileged access, and lessons for test design

**Version 1.1 (revised 2026-10-10 after a prior-art review; supersedes v1.0).** Author: Morteza [surname to be added], independent researcher. Repository: `github.com/motiza345/cognitive-self-model`. All numbers below are copied from frozen reports and manifests in the repository; section 8 says how to recompute them.

## Abstract

We asked whether a "self-model", a component that lets a system predict properties of its own behaviour, gives measurable value to a small language model or to a decision-making agent. We ran a sequence of pre-registered tests with frozen predictions, hashed artifacts, exhaustive decision rules and prompt-level paired bootstrap intervals. Four findings survive. (1) A pre-outcome second-order curvature measurement predicts the residual of the first-order response of `Qwen2.5-0.5B` to an activation intervention on a frozen holdout, and replicates under a new direction, new layers, a new observable and a disjoint catalog (relative error reduction 0.66 to 0.86, but the first-order error was already small). This is a small pre-registered replication of a Hessian-vector-product correction that was published for attribution patching (Zhang and Wang, 2026); it is not a new method. (2) This advantage is not privileged: a finite-difference probe with three forward passes matches it. (3) On a real model (`Qwen2.5-3B-Instruct`), the model's own answer probability has strong instance-level value for answer, abstain or verify decisions (+0.089 utility, within-level AUROC 0.93), while a per-condition track record adds little (+0.025). A pre-registered generalization attempt was not informative. (4) In simulation, a self-model that learns an injected property of the agent closes 87% to 99% of the gap to an agent that knows it, except where an opponent model or a cheap signal substitutes for it. Several tests were uninformative because of design errors; we report them and the lessons. We do not claim that a general self-model architecture improves agents.

## 1. Scope and definitions

"Self-model" is used for any component by which a system predicts properties of its own behaviour or performance. This report separates four objects that were originally treated as one:

1. **Mechanistic self-response.** The response of the model to a small intervention on its own residual stream. First-order prediction `g = F'(0)`; second-order term `kappa = (alpha^2 / 2) F''(0)` with `alpha = +1`, computed by automatic differentiation (Hessian-vector product). Observed effect `y = F(1) - F(0)`; residual `r = y - g`. `F` is a last-position logit margin. Exact hooks, directions and tokens are in `docs/M29_D_PROTOCOL.md` and `docs/M30_PROTOCOL.md`.
2. **Track-record self-knowledge.** Statistics of the agent's own past accuracy per condition (here, per difficulty level).
3. **Own-output probability.** The probability the model assigned to its own answer (`answer_logprob`, the sum of generated-token log probabilities, end-of-sequence excluded). This is visible to anyone with API access to log probabilities, so it is not privileged.
4. **Execution-bias model.** An agent's learned model of the gap between what it plans and what it executes (simulation only).

Research questions: **Q1** predictive validity of a self-measurement; **Q2** privileged access over cheap external alternatives; **Q3** decision utility; **Q4** competition between agents with and without a self-model.

## 2. Related work (brief; verify references before submission)

Binder et al. (2024) show models predict their own behaviour better than other models do, with limited generalization. Betley et al. (2025) show fine-tuned models describe learned behaviours without examples. Kadavath et al. (2022) show token probabilities carry calibrated information about correctness. Lindsey et al. (2025) find unreliable, context-dependent introspection with concept injection. Work on opponent modelling (LOLA, PR2, structured opponent modelling) models others rather than the self, with PR2 and LOLA containing implicit self-influence terms. O'Donoghue and Rabin (1999) show that awareness of one's own bias can help or hurt depending on payoff timing. We did not find a study that ablates a self-model, as opposed to an opponent model, in a strategic or budgeting task; this is a statement about our searches, not about the literature.

**Prior art found in a later review (October 2026), which changes how the results should be read.** (a) Zhang and Wang (2026) show that the first-order attribution-patching estimate has a dominant second-order error term computable with one Hessian-vector product, give a reliability score (the ratio of the quadratic term to the first-order estimate), and evaluate the correction on five model families from 124M to 9B parameters, including Qwen2.5-1.5B; they also report a regime where a single correction step overshoots and a multi-step variant is needed. Our curvature measurement and our post hoc finding that its magnitude predicts the remaining error are the same ideas in a steering-style setting. (b) Frontier models do not turn calibrated confidence into abstention policy and an explicit instruction to use confidence does not help, whereas enforcing the optimal policy outside the model improves utility (Wang et al., 2026, RiskEval). (c) Models' self-knowledge does not transfer across domains, compose, update from feedback or translate into action, and the suggested remedy is external scaffolding (Mirror benchmark, 2026). (d) Probes on hidden states give better calibrated confidence than verbalized scores and than black-box baselines, with weaker generalization on mathematical reasoning (2026). (e) LLM choices in rock-paper-scissors are strongly biased and instructions to randomize can make this worse; work on strategic randomization isolates the decision to randomize from the generation of randomness. (f) Intrinsic self-correction without external feedback does not reliably help (Huang et al., 2023). Consequences for this report: the mechanistic result is a replication, the instance-level own-output-probability result is an expected phenomenon, and a self-model is more plausibly an external module than text the model reads.

## 3. Method

**Pre-registration.** Each confirmatory test has a protocol file written and committed before outcomes exist, containing: question, model and revision, estimator, baselines (always including "no correction", `g` alone), metric, an exhaustive decision rule (every outcome maps to one label), effect-size floor, seed, bootstrap and unit of analysis, and stop rule. Predictions are computed and hashed before outcomes; holdout is scored once. Statistics: paired percentile bootstrap, 5000 draws, seed 23001, 95% interval; mechanistic tests use prompt-level means (the three cells of a prompt are averaged first).

**Mechanistic tests** run on CPU in float32 with `Qwen/Qwen2.5-0.5B` (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`), TransformerLens hooks, and fixed random directions. **Real-model decision tests (M34a)** run on a Colab GPU (the GPU type was not recorded; manifests record `cuda`) in float16 with greedy decoding.

**Provenance caveats.** Git commit timestamps can be set locally. Several commits were made in sandboxes without network access and pushed later. The simulation pre-registration (`simulations/ssim/SIM_PREREG.md`) was written before the single run but was committed together with the results, so it has no independent timestamp. `docs/PREREG_INDEX.md` lists each protocol with its first commit time next to the first commit time of its outcome artifact.

## 4. Results

### 4.1 Mechanistic line (Qwen2.5-0.5B)

History (labels as recorded): belief-update audits M23, M23-D2, M23-E, M23-G `INCONCLUSIVE`; M23-F `NO_SUITABLE_EXISTING_INTERVENTION`; project feasibility gate `FEASIBILITY_CONTINUE`; post-M23 single measurement `INCONCLUSIVE`; **M24 `CONSUMPTION_SUPPORTED`**; M25 synthetic belief bookkeeping (no model outcomes); **M26 `INCONCLUSIVE`**; M28 `NO_SUITABLE_PRE_EVIDENCE`.

**M24.** On frozen evaluation prompts the first-order derivative `g` had MAE 0.00589 against 0.06272 for the stored cell mean (paired mean 0.0568, 95% interval [0.0388, 0.0772]); the shuffled control did not clear zero.

**M26.** A residual correction estimated from independent evidence did not beat `g`: mean 0.00026, interval [-0.00206, 0.00248], `INCONCLUSIVE`; the shuffled correction was worse (interval entirely negative).

**M29-D** (frozen prediction file `9ff6a11e6ad28128d9446db66e66f67057a4ae7ba6a4cfbf886297af6aeda33a`; 12 holdout prompts, three cells each). The pre-outcome `kappa` reduced MAE from 0.02021 (cell-mean baseline B1) to 0.00451: delta 0.0157, interval [0.0098, 0.0223], verdict `SUPPORTED`; the permutation control included zero; leakage checks passed. Post hoc, against `g` alone (MAE 0.01248): delta 0.0080, prompt-level interval [0.0047, 0.0115]; correlation of `kappa` with the residual 0.864; sign agreement 34 of 36. B1 was a weak bar (worse than `g` alone), so from M30 on `g` alone is the primary baseline.

**M30** (design lock before measurement; catalog hash `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`, prediction hash `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`; 24 holdout prompts):

| arm | verdict | MAE(g) | MAE(g+kappa) | delta [95% interval] | relative reduction |
|---|---|---:|---:|---|---:|
| new identity (direction D2, layers 4/12/20, true/false margin) | `SUPPORTED` | 0.00219 | 0.00031 | 0.00188 [0.00140, 0.00242] | 0.858 |
| anchor (D1, layers 0/8/15, yes/no margin, M29-D identities on the new catalog) | `ANCHOR_SUPPORTED` | 0.01161 | 0.00395 | 0.00766 [0.00423, 0.01233] | 0.660 |

The absolute first-order error of the new identity was already about 0.002, so the practical gain at `alpha = 1` is small.

**Steering test S1.** To choose a step size that reaches a target margin shift, second-order solving beat first-order in 7 of 8 conditions (interval above zero). At scale 1 a black box with three or more probes matched or beat it; at scales 4 and 8 it did not within eight probes. The decision rule was not exhaustive (some outcomes mapped to no label), so the verdict is `UNDEFINED_BY_RULE`; the earlier "STOP" label was withdrawn by erratum.

**Control S2 (finite-difference quadratic, FDQ).** FDQ uses three forward passes and no gradients. Result `MIXED`: of the four decision conditions, 0 met the white-box-advantage criterion, 1 met the no-advantage criterion, and 3 met neither. Median wall time FDQ 0.325 s against 0.307 s. Median miss at scale 8: FDQ 0.390 against 0.392 (new identity), 0.520 against 0.502 (anchor). **There is no demonstrated advantage of the white-box method.**

**Exploratory (post hoc).** `|kappa|` predicts the error that remains after correction: Spearman 0.764 [0.660, 0.833] (new identity) and 0.605 [0.443, 0.723] (anchor), far above `|g|`. Keeping the 25% rows with the smallest `|kappa|` cuts the remaining error about tenfold. Not pre-registered; the same curvature is available from a probe.

### 4.2 Simulation S-SIM

Simulated learning agents with an injected property of the agent itself (execution bias, habit, error profile), 30 seeds, one run (`simulations/ssim/`). 9 of 13 pre-specified predictions were met.

- **Met.** A self-model that learns the property closes 87% to 99% of the gap to an agent that knows it (budgeting: +0.459 utility, closure 0.96; hide-and-seek against an adaptive predictor: hit rate 0.534 to 0.340, closure 0.99; verify or abstain: closure 0.87 to 0.91). No gain against a static opponent (0.007). Unaided accuracy is unchanged when no option exists, and utility rises only through allocation. In a head-to-head match the self-model agent wins by 0.198 hits per round, and identical agents tie. Known sophistication results are reproduced (awareness helps for immediate costs, hurts for immediate rewards).
- **Not met (4).** Three placebo checks were badly specified: the "wrong" self-model still carried partial information (in budgeting about 87% of the gain comes from the aggregate correction). In one variant the self-model never learned because the safe option dominated under the prior (exploration trap).
- **Unexpected.** When the opponent's predictions are visible, an opponent model alone gives the same protection (hit rate 0.318 against 0.340 for the self-model); the two are substitutes (exploratory interaction -0.19).

### 4.3 Real-model decision tests (M34a, Qwen2.5-3B-Instruct)

Task: multiplication by a single digit or addition of n-digit numbers at graded difficulty. Actions: answer (+1 correct, -1 wrong), abstain (0), verify (+0.7). Arms: always answer, global accuracy, per-level track record (**A**), shuffled track record, calibrated own-output probability within level (**C**), oracles. History sample of 10 per level, 30 draws, 240 test problems.

| version | design | result |
|---|---|---|
| v1 | multiplication by one digit, levels 5 to 10 | A `NOT_INFORMATIVE`: gate gap exactly 0. With a verify value of 0.7, answering beats verifying only above 0.85 accuracy, and the selected levels had accuracy 0.45 to 0.70. C untested: confidence was not elicited (constant 50) and log probabilities were not stored. |
| v2 | same task, levels 2 to 7, fresh pools, log probabilities stored | A `SUPPORTED_WEAK`: self minus global +0.025 [0.011, 0.038], below the 0.05 floor for `SUPPORTED`. C `INTERNAL_VALUE`: calibrated log probability within level minus level only +0.089 [0.047, 0.131]. Within-level AUROC of `answer_logprob` 0.934. Utilities: global 0.700, self 0.725, oracle by level 0.758, level plus log probability 0.847, instance oracle 0.928. |
| v3 | generalization: S1 same model on addition, levels 3 to 8; S2 Phi-3.5-mini on multiplication | S1: A `NOT_SUPPORTED` (+0.023 [-0.010, 0.061]); C `NOT_INFORMATIVE` because the pre-registered gate failed (only 2 levels with at least 15 correct and 15 wrong in the calibration pool). S2: pilot stop (best gap 0.033 below 0.05). Aggregate `C_NOT_INFORMATIVE`. Descriptive S1: level plus log probability minus level only +0.073 [0.023, 0.125], AUROC 0.936. |

Post hoc (v2, exploratory): operand-based difficulty features (carries) reach within-level AUROC about 0.53; a logistic model trained on the calibration pool reaches AUROC 0.606 with operand features, 0.931 with log probability, 0.909 with both. The signal is not explained by simple operand difficulty. This is a known phenomenon (token probabilities carry information about correctness) available to anyone who can read log probabilities.

## 5. Negative results and lessons

- **Synthetic bandit benchmark M21.5.** The instrument could not reward a self-model: its change detector fired in 2 of 40 runs (the expected error under the largest shift, about 0.58, stayed below the 0.71 threshold); the ceiling for state-aware gain was about 0.019 utility; hidden states never recurred; the decision-separation metric was 0 for all agents; two ablations collapsed to the same policy. Verdict `CLAIM_NOT_ACCEPTED`.
- **Redesign M21.6.** Validity gates for ceiling, detector, metric and power passed on tuning seeds, but the identifiability gate failed after two pre-registered redesigns, so evaluation seeds were never run. An ideal observer with the true model reached normalized mutual information of only 0.31 (maximum 0.42 over the grid), so episode-level identification was information-limited. Raw artifacts were not recovered; `reports/M21_6_RECOVERED_RECORD.md` is transcribed from tool reports and is not hash-locked.
- **Lessons.** Write exhaustive decision rules before running (S1 failed this). Include "no correction" as a baseline (B1 was worse than `g` alone). Check that a metric is non-degenerate for a reference agent. Use an ideal observer to calibrate identifiability thresholds. Choose test conditions that straddle the decision boundaries of the utility (M34a v1 failed this). A placebo must be a true null. Learning from one's own outcomes needs exploration. Keep tuning and evaluation data disjoint and score once.

## 6. Claim ladder

| rung | claim | status |
|---|---|---|
| 1 | A pre-outcome measurement predicts a model's own first-order response residual | **Supported in scope** (M24, M29-D, M30); one small model, specific directions and layers, small absolute gain |
| 2 | That prediction needs privileged internal access | **Not supported** (S2); learned external observer untested |
| 3a | Own-output probability has instance-level decision value (real model) | **Supported** in M34a v2 (`INTERNAL_VALUE`); not privileged; generalization not established (v3 gates) |
| 3b | Per-condition track record improves decisions | **Weak** (v2 `SUPPORTED_WEAK`; v3 S1 not supported) |
| 3c | Self-model value in planning and games with real LLM agents | **Untested** |
| 4 | Self-model agents win competitions | **Simulation only**; substitutable by an opponent model when predictions are visible |
| 5 | Cross-weights specificity; stability after the model is modified | **Untested** |

## 7. Limitations

One small model family for the mechanistic line (a 0.5B model) and one 3B model for decision tests; narrow tasks (single-digit multiplication, addition); simulated agents with injected properties and observed own execution; 12 to 24 prompts per holdout and clustered cells, so intervals are wide in places and several prompts in M30 share templates; effects at `alpha = 1` are small in absolute terms; the pre-registrations were written by one person with AI assistance and have only git-based timestamps; the M21.6 and M21.7 artifacts are missing; library versions differed between M29-D (transformer_lens 3.9.0, torch 2.14.0) and M30 (transformer_lens 2.16.1, torch 2.14.1).

## 8. Reproducibility

Repository tag `v1.0` (see `REPRODUCE.md`). Without any model download: `python -m scripts.verify_frozen_verdicts` recomputes the M29-D and M30 headline numbers from stored rows; `python simulations/ssim/sim_all.py` regenerates `sim_results.json` exactly; `python -m scripts.m34a_analyze`, `m34a_v2_analyze` and `m34a_v3_analyze` recompute the M34a results from cached responses; `python -m scripts.posthoc_reliability_m30` regenerates the exploratory table. Re-running model measurements needs the pinned environments in `requirements-mechanistic.txt` and `requirements-m34a.txt`.

## 9. Disclosures

Code, protocols and reports were produced with assistance from AI tools (Cursor, Codex-style agents and Claude), under the author's direction; the author is responsible for all claims. No external funding. Model licences should be checked before redistributing model outputs (some Qwen2.5 sizes use a licence different from Apache-2.0, to the best of our knowledge).

## References (check details before submission)

- Binder et al. (2024). Looking Inward: Language Models Can Learn About Themselves by Introspection. arXiv:2410.13787 (ICLR 2025).
- Betley et al. (2025). Tell Me About Yourself: LLMs Are Aware of Their Learned Behaviors. arXiv:2501.11120 (ICLR 2025).
- Kadavath et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Lindsey et al. (2025). Emergent Introspective Awareness in Large Language Models. arXiv:2512.12411.
- O'Donoghue and Rabin (1999). Doing It Now or Later. American Economic Review 89(1), 103-124.
- Foerster et al. (2018). Learning with Opponent-Learning Awareness. arXiv:1709.04326.
- Zhang and Wang (2026). When Attribution Patching Lies: Diagnosis and a Second-Order Correction. arXiv:2606.09899.
- Kramar et al. (2024). AtP*: An efficient and scalable method for localizing LLM behaviour to components. arXiv:2403.00745.
- Wang et al. (2026). Are LLM Decisions Faithful to Verbal Confidence? (RiskEval). arXiv:2601.07767.
- Huang et al. (2023). Large Language Models Cannot Self-Correct Reasoning Yet. arXiv:2310.01798.
- Vidler et al. (2025). Playing games with Large language models: Randomness and strategy. arXiv:2503.02582.
- Mirror: A Hierarchical Benchmark for Metacognitive Calibration in Large Language Models (2026). arXiv:2604.19809.
- Cencerrado et al. (2026). No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes. arXiv:2509.10625.
- Wen et al. (2019). Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning. arXiv:1901.09207.

## Appendix A. Pre-specified predictions not met

S-SIM: P3 (wrong self-model no better than naive in budgeting), P7 second part (wrong self-model no better than base in hide-and-seek), P11 (wrong-versus-base match near zero), P14 (self-model beats global calibration in verify-or-abstain). M34a v3: gate C failed in S1; S2 stopped at the pilot. M34a v1: gate A failed.

## Appendix B. Errata and deviations

1. Steering test S1: decision rule not exhaustive; label `UNDEFINED_BY_RULE`; "STOP" withdrawn.
2. M29-D: the primary baseline B1 was weaker than "no correction"; the comparison against `g` alone is post hoc (supplement), and primary from M30 on.
3. M34a v1: the report cited an analysis commit older than the script actually used (the script gained the pre-registered Amendment 1 gates; outputs identical for all shared fields, 0 differing values among 64 keys); the planned log-probability arm was not implemented; verbal-confidence arms ran with a constant value.
4. M34a v2: a leftover branch in the interpretation text printed "no support at this scale" although labels were `SUPPORTED_WEAK` and `INTERNAL_VALUE`; fixed in a separate commit with an erratum; numbers and labels unchanged.
5. S-SIM: pre-registration committed together with results; four predictions not met; placebo not a true null.
6. M21.6 and M21.7: only a transcribed record exists.
7. M34a pilot 2: the levels file was reconstructed from printed output after a notebook reset.
8. Software versions differ between M29-D and M30 (section 7).
9. Phi-3.5 pilot (S2): the model's remote code needed a compatibility patch for `DynamicCache.seen_tokens` with the installed `transformers` (see `scripts/m34a_v3_generate.py`).

## Appendix C. Key hashes

M29-D predictions `9ff6a11e6ad28128d9446db66e66f67057a4ae7ba6a4cfbf886297af6aeda33a`; M29-D outcomes `686689d5c9b6039af3027c087c21988ccfabbebb4965d3814d22a08aa8e811e9`; M30 catalog `4ceb314e304424fbdee6aeec0d9957e6aa044128908c52ab3fee9a3ec0711770`; M30 predictions `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`; M30 protocol revision `1161099eddf95b5221ed4b2ff6541d996eb441814778078f4bc3a1a0cd5545f3`; M34a v2 cache `e0eb10cd12bbf25b4e6a258ab1391e24e7110a81611c0e9dc87e0914008c5425`; M34a v3 S1 cache `60876bff0ec1a0613a21bf56a4bd06297a5c118935790530f01e63e5cab8251e`; M34a v1 cache `290307e5b95a062ee00141f43467c660ef48ddae95d6ba282f1269baaa200f99`; Qwen2.5-0.5B revision `060db6499f32faf8b98477b0a26969ef7d8b9987`; Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
