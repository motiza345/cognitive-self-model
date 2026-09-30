# Historical Colab archive

This directory is the historical experimental memory of the Cognitive Self-Model project as it was recorded in two Google Colab exports. It is an import of those notebooks. It is not a rerun, a cleanup, or a new scientific result.

## Source archives

Two ZIP exports were read. Their declared names, and the SHA256 of the ZIP bytes that were actually read, are stored in `manifests/placement.json`.

| Archive label | Declared ZIP name | SHA256 of the ZIP that was read |
| --- | --- | --- |
| `archive_01` | `Colab Notebooks-20260930T183832Z-1-001.zip` | `6ddf910e5666f3a6afe135827522438514ee304475b5f8a0cc8a4881d6b2e662` |
| `archive_02` | `Colab Notebooks-20260930T183712Z-1-001 (1).zip` | `e8c0eb795c28c8f701224e783abe84fee840bfec8226ff0871205e00e7f110f7` |

The ZIP binaries themselves are not committed (`*.zip` is gitignored). Each ZIP member is stored unchanged under `raw/archive_01/` and `raw/archive_02/`.

## Why the notebooks are preserved

`notebooks/` holds one byte-for-byte copy of each ZIP member. Code cells, markdown cells, execution counts, outputs, stdout, stderr, plots stored in the notebook, and notebook metadata are left as exported. Notebooks were not executed, reformatted, or repaired.

Shared filenames such as `Untitled8.ipynb` are kept from both archives. The archived names are `Untitled8__archive01.ipynb` and `Untitled8__archive02.ipynb`. A name that occurs in only one ZIP keeps its original filename, including odd extensions such as `01_repository_setup.ipynb.ipynb`. Nothing was overwritten.

## Why this is not the current canonical record

Files under `experiments/`, current reports, frozen artifacts, and the current Self-Model are a later repository state. This archive does not replace them and is not placed in `experiments/`. A Colab output that says PASS or FAIL is a historical reported status from that notebook. It is not a scientific validation of the current project. Every inventory record sets `scientific_interpretation` to `UNKNOWN`.

## How the inventory is built

`tools/build_colab_history_inventory.py` reads the archived notebooks. It does not run them. For each notebook it records archive source, original filename, SHA256, cell counts, and strings found in code, markdown, and cell outputs:

- milestone ids only from an `M<id>` token or the explicit phrase `MILESTONE <id>` (an optional trailing letter is kept, as in `M5.1.5b`)
- model names, revisions, library version snippets, seeds, datasets, regimes, layer and head assignments
- metric lines and status words (`PASS`, `FAIL`, `SUCCESS`, `INCONCLUSIVE`, `STOP`, `GO`, `REDEFINE`) labeled `historical_reported`

Filename text alone is not a milestone. If no milestone string is present, the milestone is `UNKNOWN`. If no line is labeled Objective, Question, or Goal, the objective is `NOT_DETERMINED`.

Exact duplicates are identical file bytes. Near duplicates are the same code and markdown source with different bytes, or code-line Jaccard at least 0.92. Both copies stay in the archive. The relationship is recorded; nothing is deleted.

## How to find a notebook

1. Open `docs/HISTORICAL_EXPERIMENT_INDEX.md` and use the `HIST-####` record.
2. Open the archived file named there under `archive/colab_history/notebooks/`.
3. Use `docs/HISTORICAL_MILESTONE_MAP.md` to see which archived notebooks contain a given milestone string.
4. Use `manifests/notebook_inventory.json` for the same records in JSON, including `similar_to` for near duplicates.

## What SHA256 is for

`manifests/notebook_hashes.sha256` is the SHA256 of each file in `notebooks/`, which is the archived copy and the same bytes as the ZIP member. It identifies that historical file. It is not a hash of a rerun.

## What this archive is not

Do not treat this archive, by itself, as scientific validation. Reported statuses stay historical. Where the notebooks do not state a result, the inventory leaves it `NOT_DETERMINED` or `UNKNOWN`.
