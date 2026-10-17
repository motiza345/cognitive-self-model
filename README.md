# cognitive-self-model

Pre-registered tests of whether a "self-model" gives measurable value to a small language model or a decision-making agent. **Status (2026-10-10):** the mechanistic self-measurement line is closed (technical report v1.1). The current track is a small metacognitive library, `layer/`, and one decisive pre-registered evaluation of it; see `docs/ROADMAP_COGNITIVE_LAYER.md`. The main document is [`reports/TECHNICAL_REPORT_v1.1.md`](reports/TECHNICAL_REPORT_v1.1.md); prior art and positioning are in [`docs/LANDSCAPE_AND_PIVOT.md`](docs/LANDSCAPE_AND_PIVOT.md).

## What was found (short version)

| question | answer |
|---|---|
| Can a pre-outcome measurement predict a model's own response to an activation intervention? | **Yes, in a narrow scope** (Qwen2.5-0.5B, specific directions and layers; relative error reduction 0.66 to 0.86; the first-order error was already small). |
| Does that need privileged internal access? | **No evidence for it**: a finite-difference probe with three forward passes matches it. |
| Does a real LLM's own answer probability help answer, abstain or verify decisions? | **Yes** for Qwen2.5-3B-Instruct on one task (+0.089 utility); known phenomenon, readable from API log probabilities; a pre-registered generalization attempt was not informative. |
| Does a per-condition track record help? | **Weakly** (+0.025 in v2; not supported in the S1 generalization). |
| Do self-models help agents in budgeting and games? | **In simulation only**, with injected properties; substitutable by an opponent model when its predictions are visible. |
| Do LLM agents with a self-model win competitions or plan better? | **Untested.** |

Negative results and design lessons (broken benchmark M21.5, failed identifiability gate M21.6, badly specified controls, gate geometry) are part of the report.

## Repository map

- `reports/` frozen reports and raw rows (`*_raw/`), `TECHNICAL_REPORT_v1.1.md`, `CLAIM_LADDER.md`, `M21_6_RECOVERED_RECORD.md`.
- `layer/` the cognitive layer library (calibrated own-output probability turned into cost-aware answer / verify / abstain decisions, scoped claims, sealed predictions, hash-chained ledger), with tests and `scripts/replay_m34a.py`.
- `docs/` protocols and pre-registrations, `PREREG_INDEX.md` (each protocol with its first commit time next to its outcome artifact), `NEXT_STEPS.md` (what a continuation would do).
- `scripts/` runners, scorers and analyses; `src/` library code; `tests/`.
- `simulations/ssim/` simulation study (pre-registration, code, results).
- `AGENTS.md` rules for coding agents working in the repo.

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Without downloading any model: `python -m scripts.verify_frozen_verdicts`, `python simulations/ssim/sim_all.py`, and the library replay in `REPRODUCE.md`.

## Caveats

One small model family, narrow tasks, simulated agents with injected properties, wide intervals in places, git-based timestamps for the pre-registrations, and two missing early artifacts (M21.6, M21.7). Code and reports were produced with AI assistance under the author's direction. See the report, sections 7 and 9.

## License and citation

License: MIT (see `LICENSE`) for the code and documents in this repository. The licences of the models whose outputs are included are separate (Qwen2.5 sizes differ; Phi-3.5-mini); check them before redistributing model outputs. Citation: see `CITATION.cff`.
