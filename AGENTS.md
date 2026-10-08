# AGENTS.md: rules for coding agents working in this repo

Read this file and `docs/NEXT_STEPS.md` before doing anything. The repository holds a pre-registered research line on self-models (Qwen mechanistic measurements M22 to S2, an earlier synthetic benchmark M21, and a simulation study S-SIM). The owner wants answers quickly: do NOT add audits, extra protocols, or cleanliness work beyond what a task asks for.

## Hard rules

1. **Frozen artifacts are read-only.** Never edit or regenerate: anything under `reports/*_raw/`, `M21.5/`, existing `docs/*PROTOCOL*.md`, `docs/S1_PREREG.md`, `docs/S2_PREREG.md`, frozen catalogs, prediction/outcome files, `reports/m29d_raw/VERDICT.json`, `reports/m30_raw/VERDICT.json`. New work goes in new files. Existing runners and tests may only be extended additively, never changed in behaviour.
2. **Pre-register before you measure.** For every new test: write `docs/<ID>_PREREG.md` first, commit and push it, and only then write or run any code that touches outcomes. A pre-registration contains: the question; exact data/model/revision; estimator or method; baselines (always including "no correction / g alone" and the strongest cheap black-box alternative); the metric; an **exhaustive** decision rule (every possible result maps to exactly one label, no gaps); an effect-size floor; the seed, bootstrap and unit of analysis; the stop rule.
3. **One run.** One execution per test, from a clean tree, after tests pass. No tuning, re-running or threshold change after seeing results. If a mistake is found, append an erratum to the report; do not rewrite history or numbers.
4. **Statistics defaults.** Prompt-level (or seed-level) paired bootstrap, 5000 draws, seed 23001, 95% percentile CI. Average within the unit before bootstrapping. Report effect size and relative reduction, not only the CI.
5. **Numerical regime for Qwen measurements:** CPU, float32, TransformerLens hooks as in `scripts/m29d_measurement.py` (autodiff Hessian-vector product, finite differences forbidden for the white-box method; finite differences are allowed only as the designated black-box control). Model revisions are pinned by commit hash.
6. **Claims.** Every report has "Claimed" and "NOT claimed" sections. Never write "self-model" for what is only a white-box approximation of a model's response. Simulation results are never described as evidence about language models.
7. **No leakage:** predictions are computed and hashed before outcomes exist; holdout is scored once.

## Workflow per test (keep it lean)

1. Branch `codex/<ID>` from the branch named in `docs/NEXT_STEPS.md`.
2. Commit + push the prereg. 3. Implement runner + one smoke test with a mock model; `python -m pytest tests -q` must pass. Commit + push.
4. Run once. Commit raw rows, `VERDICT.json`, `reports/<ID>_REPORT.md` (verdict, table of numbers, commit ids, hashes). Push. Open a PR.
5. Stop and report the verdict. Do not start the next test without being asked.

## Repo map

`src/cognitive_self_model/` library code; `scripts/` runners (`run_m30_*`, `run_s1_steering.py`, `run_s2_fdq.py`, `m29d_measurement.py`); `tests/`; `docs/` protocols and preregs; `reports/` reports and raw rows; `simulations/ssim/` simulation study. Summary of everything so far: `reports/CLAIM_LADDER.md` and `reports/FINAL_REPORT.md`.

## Environment

Python 3.10+, `torch`, `transformer_lens`, `numpy`, `pytest`. Models come from HuggingFace; if the sandbox has no network for downloads, say so and stop rather than substituting a model.
