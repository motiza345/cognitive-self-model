# M34a Colab runbook (Amendment 1)

Prereg: `docs/M34A_PREREG.md` (Amendment 1). Pilot 1 is frozen under `reports/m34a_pilot1/` and is **not** used for pools.

Replace `COMMIT_HASH` with the scripts commit from the agent.

## Cell 1 — GPU runtime

Runtime → Change runtime type → **T4 GPU**.

## Cell 2 — clone + checkout

```bash
%cd /content
!rm -rf cognitive-self-model
!git clone https://github.com/motiza345/cognitive-self-model.git
%cd cognitive-self-model
!git fetch origin
!git checkout COMMIT_HASH
!pwd
!ls scripts/m34a_pilot.py
```

## Cell 3 — install

```bash
!pip install -q "torch" "transformers>=4.44" "accelerate" "huggingface_hub" "numpy" "pytest"
```

## Cell 4 — pilot 2 (~580 calls; Amendment 1 families)

```bash
!python -m scripts.m34a_pilot --pilot 2
!python - <<'PY'
import json
d=json.load(open("reports/m34a_pilot2/levels.json"))
print(d["status"], d.get("chosen_family"), d.get("chosen_levels"))
print("best_score", d.get("choice",{}).get("best_score"))
PY
```

Download `reports/m34a_pilot2/levels.json` and send it to the agent to **commit + push before pools**.

If status is `STOP_SCORE_LT_4`, stop. Do not build pools.

## Cell 5 — pool plan (only after levels.json is on the branch)

```bash
!git pull origin cursor/m34a
!python -m scripts.m34a_make_pools
!ls -la reports/m34a_pilot2/pool_plan.json
```

Have the agent commit `pool_plan.json` **before** collection, then:

```bash
!git pull origin cursor/m34a
```

## Cell 6 — collect 1200 (resumable)

```bash
!python -m scripts.m34a_collect --pool-plan reports/m34a_pilot2/pool_plan.json
!python - <<'PY'
import json
m=json.load(open("reports/m34a_raw/manifest.json"))
print(m["n_cached"], m["cache_sha256"], m["model_id"], m["revision"])
PY
```

## Cell 7 — hash check + zip

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
