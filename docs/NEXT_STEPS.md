# NEXT_STEPS: remaining tests, order, specs

Base branch: `cursor/m30-holdout-lock-9d56` (tag `v0.1-mechanistic-line`, tip `6193bfa`) plus the handoff files in this bundle. Merge it to `main` first if possible.

## Where we are

- Supported (in scope): a pre-outcome curvature measurement `kappa` predicts the residual of the first-order response `g` for Qwen2.5-0.5B (M29-D, M30 new identity and anchor, relative reductions 0.66 to 0.86; MAE(g) is small).
- Not supported: privileged access over a cheap probe (S2: finite-difference quadratic with 3 forward passes matches the white-box method).
- Exploratory positive: abs(kappa) predicts the remaining error (rho 0.76 and 0.61), a usable reliability signal.
- Simulation (`simulations/ssim`): a self-model that learns an injected property of the agent closes 87% to 99% of the gap to an agent that knows it; no gain against a static opponent; opponent model and self-model are substitutes when the opponent's predictions are visible; three placebo checks were badly specified (not interpretable).
- Untested: anything with real LLM agents; cross-weights specificity; stability after the model is modified; a learned external observer.
- Not recoverable: M21.6 and M21.7 artifacts (see `reports/M21_6_RECOVERED_RECORD.md`).

## Order (cheapest and most informative first)

| ID | Test | Compute | Why |
|---|---|---|---|
| M35 | Simulation v2: repair the placebo, hidden opponent predictions, exploration | CPU, hours | closes the badly specified controls |
| M31 | Cross-weights replication and specificity (Qwen2.5-0.5B vs -Instruct) | CPU | "another model" control |
| M32 | Stability after the model is modified | CPU | closest to the self-model idea: does the self-measurement track a changed self |
| M33 | Learned external observer (Binder-style privileged access) | CPU | the missing privileged-access criterion |
| M34 | Real LLM agents: track-record self-knowledge (answer / abstain / verify), then planning and game variants | API calls, small budget | the decision-utility question |
| M36 | GPT-2 small IOI analogue (optional) | CPU | a second architecture family |

Decision checkpoints: after M31 to M33 decide whether the mechanistic claim stays narrow (likely) or gains privileged access. After M34 decide whether the project continues at all; the project-specific claim needs a result beyond plain calibration from a track record.

## Specs

### M35 Simulation v2 (extends `simulations/ssim/sim_all.py`, new files only)
- Per-seed heterogeneity: habit strength `h ~ U(0.1, 0.3)` in E2; bias size already random in E1.
- Placebo arms: `cross` (estimate taken from another seed's agent), E1 `aggregate_only` (keep only the sum of the estimate) and `allocation_only` (estimate minus its mean) to decompose the gain; E2 `cross_h`.
- E2 hidden-prediction variant: the agent does not see the opponent's predictions, only whether it was hit; compare `opp` (bandit feedback), `self`, `both`.
- E3 variants: epsilon-greedy exploration (epsilon 0.1 on answering) and a variant where verifying reveals whether the unaided answer would have been correct.
- Prereg predictions to write: self beats cross/aggregate/allocation placebo arms by a stated margin; in the hidden variant `self` >= `opp`; with feedback or exploration self > global in E3 V3.

### M31 Cross-weights replication and specificity
- Model B: `Qwen/Qwen2.5-0.5B-Instruct` (same shapes as Qwen2.5-0.5B: d_model 896, 24 layers), revision pinned. Use the M30 catalog and both M30 identities unchanged (new identity D2 layers 4/12/20 true/false; anchor D1 layers 0/8/15 yes/no).
- Predictions: compute `g`, `kappa` on model B. Outcomes: real intervention on model B.
- Primary: own-prediction acceptance exactly as M30 (g alone baseline, relative reduction >= 0.25, prompt-level CI > 0).
- Specificity: score the FROZEN M30 predictions of model A (hash `f363d93869f27d65cf6ad907a6442600a02f999fa374c7a7bcb1bda6d6e6abb5`) against model B's outcomes. Label `SPECIFIC` if own-prediction MAE is lower than cross-prediction MAE by >= 25% with CI > 0; `NOT_SPECIFIC` if cross-prediction is within 25%; else `MIXED`. Write the exhaustive rule in the prereg.

### M32 Stability after modification
- Modifications fixed in the prereg before any run: (a) zero the MLP output at layer 10, (b) zero the attention output at layer 6, on Qwen2.5-0.5B, using M30 catalog, new identity and anchor.
- Informative only if the modification changes the median absolute observed effect by >= 25% relative to the unmodified model (else `NOT_INFORMATIVE`, no verdict).
- Arms: `fresh` (g, kappa re-measured on the modified model), `stale` (frozen M30 predictions from the unmodified model), `g_fresh_only` (no kappa).
- Primary: fresh beats stale (MAE difference, prompt-level CI > 0) and fresh passes the M30 acceptance rule against `g_fresh_only`. Labels: `TRACKS`, `DOES_NOT_TRACK`, `NOT_INFORMATIVE` (exhaustive).

### M33 Learned external observer
- Build a labelled training set of at least 500 additional template-generated prompts (disjoint from every earlier catalog; document the generation rule first), measure real intervention outcomes on them (training outcomes only; never the M30 holdout).
- Observer sees only behaviour: prompt text features, baseline last-position logit statistics (margin, entropy, top-k logits), prompt length and family. It cannot run interventions and has no activations. Models: ridge and k-NN; choose by cross-validation on the training set only.
- Compare on the M30 holdout (new identity and anchor): MAE of the best observer vs MAE of `g + kappa` vs `g` alone.
- Labels: `PRIVILEGED` if MAE(g + kappa) <= 0.75 x MAE(best observer) with prompt-level CI > 0; `NO_ADVANTAGE` if the observer is within 25% or better; else `MIXED`. Report the training-set size curve (observer MAE at 50, 100, 250, 500 examples).

### M34 Real LLM agents (needs an API key and a budget cap written in the prereg)
- M34a track-record self-knowledge: arithmetic or verifiable questions with graded difficulty (digit counts), an LLM as the answerer. Arms: no history, global accuracy summary, per-difficulty track record, shuffled track record, oracle (true per-difficulty accuracy from a held-out calibration run). Actions: answer / abstain / verify (cost 0.3). Metrics: net utility, selective accuracy, risk-coverage. Hypotheses: per-difficulty > global > none; shuffled does not beat global.
- M34b planning with an injected execution bias (a wrapper perturbs the numbers the LLM executes; the LLM sees plan and executed history), arms with and without a history summary, shuffled summary, oracle.
- M34c hide-and-seek: an LLM hides under an injected habit wrapper against an adaptive predictor program; arms with and without a summary of its own history.
- Always include the control "opponent or environment model without self-history". Budget cap and models fixed in the prereg.

### M36 GPT-2 IOI analogue (optional, last)
- Repeat the `kappa` and finite-difference comparison on GPT-2 small IOI prompts with a head-scaling intervention. Same acceptance and control logic as M30 and S2.

## Reporting
Each test ends with `reports/<ID>_REPORT.md`, `reports/<id>_raw/`, a verdict, and an update of `reports/CLAIM_LADDER.md` (append only). Keep the "Claimed / NOT claimed / Scope" structure.
