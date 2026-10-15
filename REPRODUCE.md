# REPRODUCE

Two levels: (A) recompute every reported number from the stored raw data, with no model download; (B) re-run the model measurements. Level A was checked by the author's assistant on the commits named below (see "Verified"); level B was run on the original machines and has not been re-run independently.

## A. Recompute from stored data (no GPU, no model download)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pytest

# Frozen mechanistic verdicts (M29-D and M30): recomputes the numbers and compares with VERDICT.json
python -m scripts.verify_frozen_verdicts                  # exit code 0 = all match

# Simulation study: regenerates simulations/ssim/sim_results.json (deterministic)
python simulations/ssim/sim_all.py

# Real-model decision tests: recompute from the cached responses
python -m scripts.m34a_analyze                            # v1 (reads reports/m34a_raw)
python -m scripts.m34a_v2_analyze                         # v2 (reads reports/m34a_v2_raw)
python -m scripts.m34a_v3_analyze --setting s1            # v3, S1

# Cognitive layer library: tests and acceptance replay on the cached M34a data
(cd layer && PYTHONPATH=. python -m pytest -q)             # two tests are long (about 4 minutes)
(cd layer && PYTHONPATH=. python scripts/replay_m34a.py ../reports/m34a_v2_raw/responses.jsonl 1 scoped raw global_only base_rate_only)

# Exploratory tables
python -m scripts.posthoc_reliability_m30                 # reports/M30_POSTHOC_RELIABILITY.md
python -m scripts.m34a_v2_posthoc_operand_features        # reports/M34A_V2_POSTHOC.md
```

Notes. The analysis scripts of M34a refuse to run on a dirty working tree and write the git commit into the output; some of them overwrite their own summary file, so run them on a clean checkout and restore with `git checkout -- .` afterwards. The summary will differ from the committed one only in the recorded commit hash.

Tests that need no torch: `python -m pytest tests/test_m29d_score.py tests/test_m34a.py tests/test_m34a_v2.py tests/test_m34a_v3.py`. The other test files import torch through `src/` and need the mechanistic environment.

## B. Re-run model measurements

Environments (pins are the versions recorded in the manifests; `numpy` is not pinned):

- Mechanistic line (M29-D, M30, S1, S2): `requirements-mechanistic.txt`, Python 3.12, CPU, float32. Library versions differed between M29-D and M30 (see the report, section 7).
- M34a decision tests: `requirements-m34a.txt`, a GPU runtime (Colab was used), float16, greedy decoding. Follow `docs/M34A_V2_RUN.md` and `docs/M34A_V3_RUN.md`. Generation is not bit-for-bit reproducible across hardware, so the cached responses are the data of record; analysis from the cache is deterministic.

Do not overwrite the frozen artifacts (`reports/*_raw/`). To repeat a test, create a new protocol (see `AGENTS.md`).

## Verified

- `scripts.verify_frozen_verdicts`: all M29-D and M30 numbers match (checked on branch `cursor/m34a-v3`).
- `simulations/ssim/sim_all.py`: output identical to the committed `sim_results.json`.
- M34a analyses: v1 and v2 reproduce the committed summaries (v1 with the old and new script versions identical on all shared fields); v3 S1 identical apart from the recorded commit hash.
- Torch-free tests: 30 passed (`test_m29d_score` 5, `test_m34a` 10, `test_m34a_v2` 7, `test_m34a_v3` 8). The full suite with torch was not run by the author's assistant.
