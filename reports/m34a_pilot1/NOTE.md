# M34a pilot 1 — recorded and superseded

Pilot 1 ran on Colab (Qwen/Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, seed 34001, multiplication `n × n` for `n = 2..8`, 20 problems per level).

Per-level accuracy: **0.85, 0.10, 0, 0, 0, 0, 0** for `n = 2..8`.

The pre-registered selection rule chose levels **2..7** (accuracy range 0.85). That outcome is a **step**, not a gradient: for `n ≥ 4` there is no within-level variance (accuracy identically 0). Track-record and instance-level arms would not be informative on such a difficulty ladder.

**No CAL / HIST / TEST pool was collected under this design.** Artifact `reports/m34a_pilot/pool_plan.json` (levels 2..7, mul n×n) remains on disk for provenance and is **not** used for collection.

This directory freezes pilot 1. Amendment 1 in `docs/M34A_PREREG.md` supersedes the candidate task and selection rule before pilot 2.
