# M34a Colab runbook

Prereg: `docs/M34A_PREREG.md`. Do **not** collect pools until `levels.json` and `pool_plan.json` are committed on `cursor/m34a`.

Replace `COMMIT_HASH` below with the scripts commit printed after the agent push (step 2).

## Cell 1 — GPU runtime

Runtime → Change runtime type → **T4 GPU**.

## Cell 2 — clone + checkout

```bash
%cd /content
!git clone https://github.com/motiza345/cognitive-self-model.git
%cd cognitive-self-model
!git fetch origin
!git checkout COMMIT_HASH
```

## Cell 3 — install

```bash
!pip install -q "torch" "transformers>=4.44" "accelerate" "huggingface_hub" "numpy" "pytest"
```

## Cell 4 — pilot (140 calls)

```bash
!python -m scripts.m34a_pilot
!ls -la reports/m34a_pilot/
!python - <<'PY'
import json
print(json.load(open("reports/m34a_pilot/levels.json"))["status"])
print(json.load(open("reports/m34a_pilot/levels.json")).get("chosen_levels"))
PY
```

Download `reports/m34a_pilot/levels.json` (and optionally `pilot_raw.json`) to your machine and send `levels.json` back to the agent to **commit + push before pools**.

If status is `STOP_RANGE_TOO_SMALL`, stop. Do not build pools.

## Cell 5 — pool plan (only after levels.json is committed on the branch)

```bash
!git pull origin cursor/m34a
!python -m scripts.m34a_make_pools
!ls -la reports/m34a_pilot/pool_plan.json
```

Download `pool_plan.json` and have the agent commit + push it **before** collection. Then:

```bash
!git pull origin cursor/m34a
```

## Cell 6 — collect 1200 (resumable)

```bash
!python -m scripts.m34a_collect
!python - <<'PY'
import json
m=json.load(open("reports/m34a_raw/manifest.json"))
print(m["n_cached"], m["cache_sha256"], m["model_id"], m["revision"])
PY
```

If the session drops, re-run the same cell; completed ids are skipped.

## Cell 7 — hash check + zip download

```bash
!python - <<'PY'
import hashlib, json
from pathlib import Path
p=Path("reports/m34a_raw/responses.jsonl")
h=hashlib.sha256(p.read_bytes()).hexdigest()
m=json.load(open("reports/m34a_raw/manifest.json"))
assert h==m["cache_sha256"], (h, m["cache_sha256"])
print("OK", h)
PY
!zip -r /content/m34a_raw.zip reports/m34a_raw
```

Download `/content/m34a_raw.zip`. Offline analysis is run later on CPU from that cache (agent): `python -m scripts.m34a_analyze`.
