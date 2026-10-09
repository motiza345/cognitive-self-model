# M34a pilot 2 — levels recovered from Colab stdout

Pilot 2 ran on Colab at commit `681a7dd` (`python -m scripts.m34a_pilot --pilot 2`). Status `LEVELS_CHOSEN`. After a runtime reset the on-disk `levels.json` was `MISSING`.

This directory reconstructs the printed choice JSON (580 calls). Fields not printed (`input_tokens`, `output_tokens`, `runtime_seconds`, `float32_rerun_ids`) are null/empty. Model/revision are the Amendment 1 pin.

**Chosen:** family `mul_n1`, levels `[5, 6, 7, 8, 9, 10]`, best_score `6`, window accuracies 0.80, 0.65, 0.50, 0.25, 0.45, 0.55.

No CAL/HIST/TEST collection had started.
