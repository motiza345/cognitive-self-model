# Landscape and pivot: what we have, what others did, where it is useful (October 2026)

Based on about 15 targeted searches and the full text of two papers (Zhang and Wang 2026; Wang et al. 2026). Statements of the form "we found no study that..." refer to these searches only. Many sources are 2026 preprints; peer review status was not checked.

## 1. What we have

| asset | what it does | evidence in this repo | scope |
|---|---|---|---|
| **A1. Second-order curvature (kappa) via Hessian-vector product** | Predicts the first-order residual of a model's own response to an activation intervention; its magnitude flags when the first-order estimate is unreliable | M29-D, M30 (relative error reduction 0.66 to 0.86), post hoc reliability (Spearman 0.76 and 0.61); no advantage over a 3-forward-pass probe (S2) | Qwen2.5-0.5B, last-token logit margin, alpha = 1 |
| **A2. Own-output-probability gate** | Instance-level estimate of "my answer is right", calibrated within difficulty level, drives answer / abstain / verify by expected utility | M34a v2: +0.089 utility over level-only, within-level AUROC 0.93; v3 S1 descriptive +0.073 (confirmatory gate not met) | Qwen2.5-3B-Instruct, one-digit multiplication and addition |
| **A3. Track-record self-knowledge** | Per-condition accuracy from the agent's own history | Weak: +0.025 (v2), not supported in v3 S1 | same |
| **A4. Simulation framework** | Agents with an injected self-property; self-module debiasing; hide-and-seek; budgeting with execution bias | S-SIM, 9 of 13 predictions met; self-module closes 87% to 99% of the gap to an agent that knows the property | simulation only |
| **A5. Method** | Pre-registration, frozen hashes, exhaustive decision rules, gates, offline verification scripts, errata | repo, `REPRODUCE.md`, `PREREG_INDEX.md` | transferable |

## 2. State of the art relevant to each asset

### A1 (curvature and steering)
- **Prior art for the correction itself.** Zhang and Wang (arXiv 2606.09899, June 2026): the first-order attribution-patching error is dominated by a second-order term computable with one Hessian-vector product; they give a reliability score (quadratic term over first-order estimate), a multi-step variant for large perturbations (single-step can overshoot, catastrophically in one Pythia-410M setting), and evaluate five model families (124M to 9B) including Qwen2.5-1.5B, with error reductions of 72 to 90% on attention heads. **Our M29-D and M30 are a pre-registered small replication of this idea in a steering-style setting; not novel.** AtP* (Kramar et al. 2024) documents first-order failure modes (attention saturation, cancellation).
- **Steering literature.** Steering responses are nonlinear and unreliable in places: curvature in judged behaviour responses is real but small (Forecasting Side Effects of Activation Steering, 2026); nonlinear steering beats linear when activation-manifold curvature is high (Curveball steering, 2026); geometric predictors of steering unreliability (2026); trait expression follows an inverted-U in the coefficient (What Can We Actually Steer?, 2025); adaptive per-input strength and PID feedback steering (IDS, CLAS, PID Steering at ICLR 2026).
- **Implication.** A practical, unexplored-by-us use is a curvature-based reliability flag and coefficient predictor for steering and patching. Competition is real; the contribution would be an application and a benchmark, not a method.

### A2 (confidence-gated decisions) and A3 (track record)
- **Positive, established.** Token probabilities carry calibrated information about correctness (Kadavath et al. 2022). Selective prediction and escalation with confidence is a developed area: cost-sensitive escalation (Act or Escalate, 2026), surrogate log-probabilities to audit black-box agents with accept-the-most-confident selective prediction (Proxy Confidence, 2026), confidence-guided control in tool-using agents (REFLEX, 2026), Bayesian self-escalation (2026), LLM cascades (FrugalGPT, RouteLLM).
- **Probes beat verbalized confidence.** A hidden-state probe's calibration error is 5 to 38 times lower than the verbalized score (Calibration, Not Answer Selection, 2026); question-only probes predict correctness before generation and beat black-box baselines and verbal confidence, with weaker generalization on mathematical reasoning (2026); verbal confidence is computed internally and cached, explaining variance beyond log probabilities (Kumaran et al. 2026).
- **Negative, established.** Frontier models do not adapt abstention to penalties; verbal confidence stays calibrated but unused; an instruction to use confidence does not help; enforcing the optimal policy outside the model improves utility (RiskEval, Wang et al. 2026). Self-knowledge does not transfer across domains, compose, update from feedback or translate into action; the proposed fix is external scaffolding (Mirror benchmark, 2026). LLMs are overconfident about their capabilities, though some learn from in-context failures (Barkan et al. 2025). Self-awareness correlates with agent success and degrades for internal-capability tasks (KAPRO, 2026).
- **Self-correction negative.** Intrinsic self-correction without external feedback does not reliably improve reasoning (Huang et al. 2023); self-verification can reduce accuracy, and sound external verifiers help (Stechly et al.).
- **Implication.** Our A2 result is the expected phenomenon. The consistent lesson is architectural: confidence must be converted into action by a module outside the model, using calibrated signals, not by asking the model to read its own confidence.

### A4 (games, budgeting)
- LLMs are poor randomizers (about 79% rock in a base prompt; instructions to randomize can worsen the bias; Vidler et al. 2025), keep canonical strategies when payoffs change (Georgousis et al. 2026), and are exploitable by adaptive opponents (Profit is the Red Team, 2026; Are LLM Agents Exploitable Negotiators?, ICLR 2026). Work on strategic randomization isolates the decision to randomize from the generation of random bits, using system-provided randomness. Opponent modelling is the dominant direction (SOM, Strat-Reasoner, MARSHAL, LOLA, PR2).
- **Implication.** The straightforward fix (offload randomization to a tool) makes a learned self-model unnecessary in simple zero-sum games; the value of a self-model there would be knowing when to offload, and that decision is largely given by the game structure. We found no direct ablation of a self-model versus an opponent model with LLM agents; our simulation suggests they are substitutes when the opponent's predictions are visible.

### Introspection (A2/A3 background)
Binder et al. (2024): models predict themselves better than other models do, with limited generalization. Betley et al. (2025): fine-tuned behaviours can be described without examples. Lindsey et al. (2025): introspection is unreliable and context-dependent.

## 3. What not to test (settled or not worth our budget)

| planned test | reason to drop |
|---|---|
| In-prompt self-knowledge utilization (M41c, in-prompt arms of M42 and M43) | Known negative (RiskEval, Mirror); fine to cite instead of re-measuring |
| Verbal confidence versus log probability (M41b) | Heavily studied; probes are the stronger competitor |
| LLM self-correction | Known negative |
| Base LLM randomization bias in games | Documented; offload to a tool |
| First-order versus second-order attribution on more models (M31, M36) | Covered by the June 2026 paper |
| LLM game versions of E1 and E2 as an end in themselves (M42, M43) | Low novelty; the module result is already shown in simulation and in RiskEval |

## 4. Where our assets are useful

1. **Reliability gating of agent actions** (A2, A3, A5): a thin module that converts a calibrated per-instance confidence into answer / verify / abstain / escalate. Supported by our v2 result and consistent with the literature; the gain is cost and reliability, not capability.
2. **Interpretability and steering tooling** (A1): a second-order corrected response predictor and a per-input reliability flag for patching and steering coefficient choice. Prior art exists for patching; steering coefficient prediction is the open application.
3. **Evaluation methodology** (A5): pre-registered, gated tests that avoid the failure modes we hit (no ceiling, utility geometry, non-null placebo, non-exhaustive rules, weak baselines). This may be the most reusable asset.
4. **Not supported:** a general self-model that makes an LLM or agent better at tasks. The evidence points to modest, task-specific, scaffold-dependent gains.

## 5. Can this be used to write a better agent or LLM?

Narrow yes, broad no. A better **agent** in the sense of cheaper and more reliable: a controller module that (i) reads a calibrated confidence (log probability, a probe if hidden states are available, a per-condition track record), (ii) applies a cost-aware threshold computed outside the model, and (iii) offloads randomness and verification to tools. This is consistent with every positive result found, including the literature on cascades and selective prediction. A better **LLM**: nothing here changes capability. The honest competitive position is engineering and evaluation, not a new capability; novelty is limited by Act or Escalate, Proxy Confidence, REFLEX, RiskEval, the probe literature and the June 2026 HVP paper.

## 6. Revised program (replaces `M40_COMPREHENSIVE_PLAN.md` stages M41 to M43)

**Decision test D1: a gate on a realistic tool-use task (one Colab session).**
- Task: arithmetic and word problems with a calculator or code tool as the verifier (cost per call), graded difficulty, pools as in M34a (CAL, HIST, TEST), Qwen2.5-3B-Instruct, optionally Qwen2.5-7B only if memory allows without changing numerical regime.
- Gate arms (decisions made by a module outside the model): always answer, always verify, random verify at matched call budget, track-record per level, answer log probability calibrated on CAL, hidden-state probe trained on CAL (new; needs hidden states stored at generation time), combination of probe and log probability, oracle.
- Metrics: accuracy versus verifier-call budget (area under the cost-accuracy curve), utility with the answer/abstain/verify payoffs, selective accuracy.
- Pre-registered ceiling gate: oracle must beat always-verify by a stated margin in the cost-adjusted utility, else `NOT_INFORMATIVE`.
- Labels (exhaustive): `PROBE_BEST`, `LOGPROB_BEST` (log probability not worse than probe within a stated margin), `TRACK_RECORD_ENOUGH`, `NO_SIGNAL_BEATS_RANDOM`. Report that this replicates known qualitative results; the contribution is the cost-utility accounting with a verify option and a pre-registered protocol on small open models.

**Optional D2: curvature-based steering coefficient prediction.** Use kappa to predict the effect of a steering step and to flag unreliable steps; compare with a three-forward-pass probe and a multi-step HVP variant; behavioural outcomes, not only logit margins. Only if the interpretability application is of interest.

**Optional D3 (writing).** Update the technical report with the related work above (done in v1.1) and position the contribution as: a pre-registered replication, a methodology of gates and exhaustive rules, a list of failure modes, and the scaffold-over-prompt result.

## 7. Limits of this review
About 15 searches and two full papers; not a systematic review. Sources are mostly 2026 preprints. Absence of a found study does not mean absence of one. Quantitative claims quoted from papers should be checked against the papers before reuse.

## Sources (as found)
- Zhang and Wang. When Attribution Patching Lies: Diagnosis and a Second-Order Correction. arXiv:2606.09899
- Kramar et al. AtP*. arXiv:2403.00745
- Forecasting Side Effects of Activation Steering. arXiv:2608.11227
- Curveball Steering. arXiv:2603.09313
- Understanding Unreliability of Steering Vectors. arXiv:2602.17881
- What Can We Actually Steer? arXiv:2511.18284
- Activation Steering with a Feedback Controller (PID Steering), ICLR 2026
- Wang et al. Are LLM Decisions Faithful to Verbal Confidence? arXiv:2601.07767
- Act or Escalate? arXiv:2604.08588
- Proxy Confidence. arXiv:2610.03894
- REFLEX with Jev. arXiv:2609.26532
- Knowing When to Ask for Help: Bayesian Self-Escalation. arXiv:2608.24087
- Calibration, Not Answer Selection. arXiv:2609.33290
- No Answer Needed: question-only probes. arXiv:2509.10625
- How do LLMs Compute Verbal Confidence? arXiv:2603.17839
- Barkan et al. Do Large Language Models Know What They Are Capable Of? arXiv:2512.24661
- From Knowing to Acting (KAPRO). arXiv:2606.20661
- Mirror: A Hierarchical Benchmark for Metacognitive Calibration. arXiv:2604.19809
- Huang et al. Large Language Models Cannot Self-Correct Reasoning Yet. arXiv:2310.01798
- Vidler et al. Playing games with Large language models: Randomness and strategy. arXiv:2503.02582
- Do LLMs Know When to Flip a Coin? arXiv:2506.18928
- Discovering Differences in Strategic Behavior Between Humans and LLMs. arXiv:2602.10324
- Binder et al. arXiv:2410.13787; Betley et al. arXiv:2501.11120; Lindsey et al. arXiv:2512.12411
