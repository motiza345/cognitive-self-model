# M34a-v3 Colab runbook

Prereg: `docs/M34A_V3_PREREG.md`. S1 pool plan is already committed. Do **not** edit v1/v2 files. Do **not** run analysis on Colab.

Replace `COMMIT_HASH` with the scripts commit from the agent (analysis frozen before collection).

## Shared Cell 0 — T4 GPU

Runtime → Change runtime type → **T4 GPU**.

## Phase A — S1 collection (Qwen add_nn)

```bash
%cd /content
!rm -rf cognitive-self-model
!git clone https://github.com/motiza345/cognitive-self-model.git
%cd /content/cognitive-self-model
!git checkout COMMIT_HASH
!pip install -q "torch" "transformers>=4.44" "accelerate" "huggingface_hub" "numpy"
!python -m scripts.m34a_v3_collect --setting s1
```

Re-run only the last line if the session drops (resumable). Expect `n_cached: 1200`.

Zip (no heredoc):

```python
import hashlib, json, os
from pathlib import Path
os.chdir("/content/cognitive-self-model")
p = Path("reports/m34a_v3_s1_raw/responses.jsonl")
h = hashlib.sha256(p.read_bytes()).hexdigest()
m = json.load(open("reports/m34a_v3_s1_raw/manifest.json"))
print(m["n_cached"], h, m["cache_sha256"], h == m["cache_sha256"])
```

```bash
%cd /content/cognitive-self-model
!zip -r /content/m34a_v3_s1_raw.zip reports/m34a_v3_s1_raw
```

Download and send `m34a_v3_s1_raw.zip`.

## Phase B — S2 pin revision + pilot (Phi mul_n1)

Same checkout. Pin revision **before** any S2 generation:

```bash
%cd /content/cognitive-self-model
!python -m scripts.m34a_v3_pin_s2_revision
!python -m scripts.m34a_v3_pilot_s2
```

```python
import os, json
os.chdir("/content/cognitive-self-model")
print(open("reports/m34a_v3_s2/revision.json").read())
print(open("reports/m34a_v3_s2_pilot/levels.json").read()[:2000])
```

```bash
!zip -r /content/m34a_v3_s2_pilot.zip reports/m34a_v3_s2_pilot reports/m34a_v3_s2/revision.json
```

Send the zip / `levels.json`. If status is `STOP_GAP_LT_0.05` or `NOT_RUN`, stop S2 collection.

## Phase C — S2 collection (only if levels chosen)

After the agent commits the S2 pool plan from your pilot levels:

```bash
%cd /content/cognitive-self-model
!git pull
!git checkout COMMIT_HASH_S2_PLAN
!python -m scripts.m34a_v3_collect --setting s2
```

```python
import hashlib, json, os
from pathlib import Path
os.chdir("/content/cognitive-self-model")
p = Path("reports/m34a_v3_s2_raw/responses.jsonl")
h = hashlib.sha256(p.read_bytes()).hexdigest()
m = json.load(open("reports/m34a_v3_s2_raw/manifest.json"))
print(m["n_cached"], h, m["cache_sha256"], h == m["cache_sha256"])
```

```bash
!zip -r /content/m34a_v3_s2_raw.zip reports/m34a_v3_s2_raw
```

Download and send `m34a_v3_s2_raw.zip`.
