# M50 runbook (Colab)

Protocol: `docs/M50_PREREG.md`. Two Colab sessions with a decision step in between. Do not edit any file in the repository on Colab; the manifests record the git commit and every script refuses to run with modified tracked files.

## Before you start

- Colab: Runtime, Change runtime type, GPU (T4 is enough; the 3B model needs about 7 GB).
- You need the commit hash that the pre-registration and scripts were merged at (given in the announcement message). Below it is written `<COMMIT>`.
- Everything is resumable: if Colab disconnects, run the same cells again; finished items are skipped.

## Session 1: smoke test and pilot (about 20 to 30 minutes)

Cell 1, check the GPU:
```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
```
It must print `True` and a GPU name. If not, change the runtime type first.

Cell 2, get the code at the exact commit:
```bash
!git clone https://github.com/motiza345/cognitive-self-model.git
%cd cognitive-self-model
!git checkout <COMMIT>
!git status --short
!pip install -q scipy huggingface_hub
```
`git status --short` must print nothing. Do not reinstall torch or transformers; the manifest records the versions that Colab provides.

Cell 3, optional but recommended: keep the results on Google Drive so a disconnect loses nothing.
```python
from google.colab import drive
drive.mount('/content/drive')
```
At the end of the session you will copy the result folders to Drive (Cell 7). If the session dies, copy back before resuming.

Cell 4, smoke test (about 2 minutes: loads the model, generates 8 problems, compares batch and single generation):
```bash
!python -m scripts.m50_generate --stage pilot --smoke
```
It must end with `SMOKE PASSED`. A warning that the output depends on batch size is allowed; if you see it, use `--batch 4` in the next cell and tell me. If it says FAILED or crashes, stop and send me the full text of the output.

Cell 5, generate the pilot (about 1300 short generations, roughly 10 to 15 minutes):
```bash
!python -m scripts.m50_generate --stage pilot
```
(The pilot plan is already committed. If the cell stops, run it again.)

Cell 6, package the pilot results:
```bash
!zip -r m50_pilot_raw.zip reports/m50_pilot
```
Cell 7, download `m50_pilot_raw.zip` from the Files panel (left side) and send it to me. Optionally copy it to Drive: `!cp m50_pilot_raw.zip /content/drive/MyDrive/`.

## Decision step (not on Colab)

I run `python -m scripts.m50_pilot` on the pilot file, which writes `levels.json` and `gates.json`. If a gate fails, the pre-registration forbids collection and the evaluation ends with the label `NOT_INFORMATIVE`. Otherwise I commit `levels.json`, `gates.json` and the main pool plan, and give you a new commit hash.

## Session 2: main collection (about 30 to 60 minutes)

Repeat Cells 1 to 3 with the NEW `<COMMIT>` (fresh runtime or `!git fetch && !git checkout <NEW_COMMIT>`).

Cell 8, smoke test again (the first batch after loading is the riskiest):
```bash
!python -m scripts.m50_generate --stage main --smoke
```
Cell 9, collection (about 1700 generations, with hidden states; resumable):
```bash
!python -m scripts.m50_generate --stage main
```
Cell 10, package and download:
```bash
!zip -r m50_raw.zip reports/m50_raw
```
Download `m50_raw.zip` and send it to me. Do not open or analyse the files yourself; the analysis is run once, by the committed script, after the cache and its hash are committed.

## If something goes wrong

- Out of memory: add `--batch 8` (or 4) and rerun the same cell.
- The session disconnects: reconnect, rerun Cells 1 and 2 (and mount Drive, copying the saved folders back), then the same generation cell.
- A cell prints an error you do not understand: send me the whole output.
- Never delete `responses.jsonl` to "start fresh" without telling me; the manifest and plan hashes protect the protocol.
