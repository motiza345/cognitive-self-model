# Roadmap: from cognitive layer v0.2.1 to a published, finished result

Time box: about 3 weeks of wall-clock time, 3 to 4 GPU hours (estimate). Owner of each step is marked. Stop rules are binding: when a stop rule fires, skip to Phase E and publish what exists.

## Rules for the whole track

1. **Freeze scope.** No new hypotheses and no new experiments until Phase E is done. Everything else goes to `docs/BACKLOG.md` (M31 to M33, M36, M41 to M43, steering-coefficient work).
2. **One decisive evaluation (called M50).** One pre-registration, one pilot, one collection, one analysis.
3. **Gate before collecting.** The ceiling gate and the scope-heterogeneity gate are computed on pilot data. If either fails, the label is fixed (below) and collection does not start.
4. **One amendment at most.** If the pilot shows a design flaw, one amendment is allowed, pushed before collection. A second flaw ends the evaluation (publish the negative or inconclusive result).
5. **Everything is committed and pushed before it is run** (pre-registration, scripts, tests, commit hash); results are analysed once from the cache.
6. **Claims stay scoped.** "Calibrated own-output probability turned into a cost-aware decision" is the claim; nothing about general self-awareness.

## Phase A: integrate what exists (1 to 2 days)

| step | owner | how | done when |
|---|---|---|---|
| A1 | you | Unzip `cognitive_layer_v0.2.1.zip`. In GitHub, branch `main` (or a new branch `layer/v0.2.1`), Add file, Upload files: upload the folder contents so the repo has `cognitive_layer/` (package), `tests/` entries (`test_v02.py`, `test_platt_extreme_confidences.py`, `fixtures/`), `scripts/replay_m34a.py`, `CHANGELOG.md`. To avoid clashes with the existing `tests/` and `scripts/`, put the library in a subfolder `layer/` (all its files, keeping its own `tests/` and `scripts/` inside it) | files visible on GitHub |
| A2 | you | Open a PR from `cursor/m34a-v3` to `main`, merge. Then merge or open a PR for the library branch. Tag `v1.1` | `main` shows all work |
| A3 | you (or me if you paste errors) | In a clean venv: `pip install numpy scipy pytest`, then in `layer/`: `PYTHONPATH=. python -m pytest -q` (the long test takes about 3.5 minutes) | all 16 tests pass |
| A4 | you | Add `docs/BACKLOG.md` (copy the freeze list above) | file exists |

Optional Cursor prompt for A1 to A4 (only if you prefer Cursor): "Create a branch `layer/v0.2.1`. Add the contents of the attached `cognitive_layer_v021` folder to the repo under `layer/` without modifying any existing file. Run the tests there with `PYTHONPATH=. python -m pytest -q` and report the result. Add `docs/BACKLOG.md` listing: M31, M32, M33, M36, M41-M43, steering-coefficient prediction. Commit and push. Do nothing else."

## Phase B: pre-register the decisive evaluation M50 (2 to 3 days; I write it, you commit)

**Question.** Does a cost-aware module that turns calibrated own-output probability into answer / verify / abstain decisions beat simple policies, and does per-scope calibration add value over one global calibration?

**Setting.** `Qwen/Qwen2.5-3B-Instruct` (fp16, greedy, Colab GPU). Workload: a mixed stream of three verifiable task types as scopes, `mul_n1`, `add_nn`, `sort_n` (exact-match verifier = the tool), each with graded difficulty; scope = (task, level). Payoffs as in M34a: answer +1 or -1, abstain 0, verify +0.7.

**Two conditions.** (1) **Natural**: no manipulation. (2) **Implanted**: a wrapper silently corrupts the output of one scope with probability 0.3 (invisible to log probabilities, visible through verification): the positive control for scoping.

**Arms (decided by the module, outside the model).** always answer; always verify; random verify at the call budget of the scoped layer; raw confidence (uncalibrated); global calibrated claim; **scoped layer**; hidden-state probe (trained on the calibration pool; stored at generation time); per-level track record; instance oracle (reference).

**Protocol.** Pilot (40 items per scope-level) then gates, then freeze, then collection: calibration stream CAL (100 per scope-level), TEST (40 per scope-level), replay protocol as in `replay_m34a.py`. Seeds, 5000-draw paired bootstrap over TEST items, seed 23001. Learning-phase utility reported (cost of learning).

**Gates (computed on the pilot, fixed in the pre-registration).**
- G-ceiling: instance oracle minus always-verify at least 0.15 utility, else `NOT_INFORMATIVE`.
- G-scope: in the pilot, scope-specific calibration (cross-validated) must beat one global calibration by at least 0.02 in the implanted condition; in the natural condition this is reported, not required.
- G-variance: at least 3 scope-levels with at least 15 correct and 15 wrong answers expected in CAL (pilot accuracy between 0.25 and 0.75).

**Hypotheses and exhaustive labels (to be written in full in the pre-registration).**
- **M50-A (gate works):** scoped layer minus the best of {always answer, always verify, random verify at matched budget} at least 0.03 with interval above zero.
- **M50-B (calibration matters):** scoped minus raw confidence at least 0.02, interval above zero.
- **M50-C (scoping matters):** scoped minus global calibrated at least 0.02, interval above zero; evaluated separately in the implanted and the natural condition.
- **M50-D (probe versus log probability):** `PROBE_BETTER`, `EQUIVALENT` (within 0.02), or `LOGPROB_BETTER`.
- A fourth label covers every combination so no outcome is unlabeled; a failed gate gives `NOT_INFORMATIVE` for the affected hypotheses.

**Deliverables.** `docs/M50_PREREG.md`; `scripts/m50_generate.py` (resumable cache, stores token log probabilities and a hidden-state vector per generation, verifier tool, implanted wrapper); `scripts/m50_pilot.py`; `scripts/m50_analyze.py` (runs the layer replay and all arms offline, refuses a dirty tree); tests with a mock model; `docs/M50_RUN.md` with Colab cells. Written and tested offline by me; you upload.

## Phase C: run it (1 to 2 Colab sessions)

1. After the pre-registration and scripts are on GitHub, note the commit hash.
2. Colab: Runtime, Change runtime type, GPU. Cells: `!git clone https://github.com/motiza345/cognitive-self-model`, `%cd cognitive-self-model`, `!git checkout <commit>`, `!pip install -r requirements-m34a.txt scipy`, run the pilot script. Download `levels.json` and the gate output, send them to me (or commit them yourself) **before** collection.
3. If a gate fails, stop (label fixed, go to Phase E). Otherwise run the collection script with `--resume` (mount Google Drive so a disconnect loses nothing), compute the cache hash, zip `reports/m50_raw`, download.
4. Do not look at TEST accuracy or run any analysis other than the committed analysis script.

## Phase D: analyse and report (1 to 2 days)

1. Commit the cache and manifest; run `scripts/m50_analyze.py` once on a clean tree.
2. Send me the output; I check integrity (hash, counts, disjointness), reproduce the numbers, and write `reports/M50_REPORT.md` with Claimed / NOT claimed / Scope.
3. Append to `reports/CLAIM_LADDER.md`.

## Phase E: publish (2 to 3 days)

1. Technical report v1.2: add M50 and the library (positioned as: an engineering and evaluation contribution; mechanistic line is a replication; known results cited).
2. Repo: `README` rewrite, `LICENSE` (your choice), `CITATION.cff` with your surname, `requirements*.txt`, `REPRODUCE.md` extended with the library replay.
3. Release `v1.2`, optional Zenodo DOI, optional arXiv.

## Decision rules at the end

| outcome | action |
|---|---|
| M50-A and M50-C supported (implanted) and M50-B supported | Library is useful; write it up as a metacognitive gating layer with an honest scope statement; decide on 0.3 (incremental statistics, automatic scopes, adapters) |
| M50-A supported, M50-C not supported | Report that global calibration is enough in these settings; the library reduces to calibrated gating (still useful); stop development there |
| M50-A not supported or gates fail | Report as an inconclusive or negative result; publish the library and the replay anyway; stop |

## Checkpoints

End of week 1: Phase A done and M50 pre-registration pushed. End of week 2: pilot done and gates decided, collection finished. End of week 3: report and release. If a checkpoint is missed by more than 3 days, apply the stop rule and publish.

## What I will do versus what you will do

- **Me:** write the pre-registration, scripts, tests (offline with mocks), the Colab runbook, check integrity and reproduce numbers, write the reports.
- **You:** upload files to GitHub, run the Colab cells, download caches, make the licence and authorship decisions.
- **Cursor or Codex:** optional; not required for any step.
