# M34a-v2 Colab runbook

Prereg: `docs/M34A_V2_PREREG.md`. Pool plan is already committed. Do **not** edit v1 files.

Replace `COMMIT_HASH` with the scripts commit from the agent.

## Cell 1 — T4 GPU

Runtime → Change runtime type → **T4 GPU**.

## Cell 2 — clone + checkout + collect

```bash
%cd /content
!rm -rf cognitive-self-model
!git clone https://github.com/motiza345/cognitive-self-model.git
%cd cognitive-self-model
!git checkout COMMIT_HASH
!pip install -q "torch" "transformers>=4.44" "accelerate" "huggingface_hub" "numpy"
!python -m scripts.m34a_v2_collect --pool-plan reports/m34a_v2/pool_plan.json
```

Re-run the last line if the session drops (resumable).

## Cell 3 — zip

```bash
!python - <<'PY'
import hashlib, json
from pathlib import Path
p=Path("reports/m34a_v2_raw/responses.jsonl")
h=hashlib.sha256(p.read_bytes()).hexdigest()
m=json.load(open("reports/m34a_v2_raw/manifest.json"))
print(m["n_cached"], h, m["cache_sha256"], h==m["cache_sha256"])
PY
!zip -r /content/m34a_v2_raw.zip reports/m34a_v2_raw
```

Download `/content/m34a_v2_raw.zip` and send it back. Do not run analysis on Colab.
