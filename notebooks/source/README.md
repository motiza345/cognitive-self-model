# Source notebooks (provenance archive)

These are the original Colab research notebooks (from the `starlight` repo and a
shared Google Drive) that the scientific milestones were developed in. They are
vendored here **for provenance only** — the maintained, tested implementations
live in `src/`. Code-cell outputs were stripped for tidiness; the code is
otherwise unmodified.

## Contents and lineage

| File | Milestones inside | Role | Where it now lives in the repo |
| --- | --- | --- | --- |
| `M19_governance_and_genesis.ipynb` | M21.2.3.1–M21.2.3.6, M21.2.4.0–.2 | Genesis of the benchmark + the epistemic **governance** subsystem (quarantine / abstention / decision engine) | Governance layer is **not yet ported** (Phase D). Benchmark genesis superseded by `src/cognitive_self_model/benchmark/`. |
| `M21_frozen_benchmark_and_calibration.ipynb` | M21.2.4.1 → M21.2.4.3.2 | Frozen environment/encoder/estimator + isotonic calibration (authoritative M21.2.4.3.1 in its final cell) | `src/cognitive_self_model/benchmark/` |
| `M22_integration_and_bundle_export.ipynb` | M21.2.4.3 / .3.1 / .3.3 | The 8-key bundle generator (`m21_2_4_3_1_pipeline.py`) + the calibration audit + repo population | `src/cognitive_self_model/benchmark/bundle.py`, `pipeline.py`; `src/legacy_import/m212433_calibration_audit.py` |
| `M21_2_4_3_4_identifiability_working.ipynb` | M21.2.4.3.3 / .3.4 | Working notebook for the identifiability milestone (originally used a **synthetic** fallback dataset) | Real oracle dataset generator now in `src/cognitive_self_model/benchmark/identifiability_dataset.py` |

## Important notes

- The maintained code is the source of truth. These notebooks are a historical
  record and must not be imported or executed as part of the pipeline.
- `self_model_invalid` in the maintained code is the environment's
  `is_contextually_invalid` flag (the two names are used interchangeably across
  these notebooks).
- The M20.6.3.2 benchmark (real `Qwen` activations) lives separately in
  `notebooks/cognitive_self_model_latest.ipynb`; connecting it to the evidence
  interface is Phase B of the roadmap.
