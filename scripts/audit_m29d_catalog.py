"""Audit the M29-D catalog against every consumed M22-M26 catalog.

This is a pre-outcome catalog audit. It does not load Qwen and does not
inspect or generate intervention outcomes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m29d_catalog import PARTITIONS, FAMILIES, catalog, catalog_sha256

from scripts.m24_consumption_catalog import catalog as m24_catalog
from scripts.m26_update_catalog import catalog as m26_catalog
from scripts.validate_m23_g_protocol import catalog as m23g_catalog
from scripts.validate_post_m23_single_measurement import catalog as single_measurement_catalog
from scripts.feasibility_gate_catalog import catalog as feasibility_catalog
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts


def flatten(rows):
    return {str(row["prompt_id"]): str(row["text"]) for row in rows}


def audit():
    new = catalog()
    if len(new) != 36:
        raise AssertionError("M29-D catalog must contain 36 rows")
    ids = [row["prompt_id"] for row in new]
    texts = [row["text"] for row in new]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate M29-D prompt ids")
    if len(texts) != len(set(texts)):
        raise AssertionError("duplicate M29-D prompt texts")
    if {r["family"] for r in new} != set(FAMILIES):
        raise AssertionError("family coverage is incomplete")
    if {r["partition"] for r in new} != set(PARTITIONS):
        raise AssertionError("partition coverage is incomplete")
    for p in PARTITIONS:
        rows = [r for r in new if r["partition"] == p]
        if len(rows) != 12:
            raise AssertionError(f"{p} must contain 12 prompts")
        if {r["family"] for r in rows} != set(FAMILIES):
            raise AssertionError(f"{p} is missing a family")

    old_catalogs = {
        "M22.1": [{"prompt_id": p.prompt_id, "text": p.text} for p in frozen_prompts()],
        "M23-G": m23g_catalog(),
        "POST-M23-SINGLE": single_measurement_catalog(),
        "FEASIBILITY": feasibility_catalog(),
        "M24": m24_catalog(),
        "M26": m26_catalog(),
    }

    old_ids = set()
    old_texts = set()
    overlaps = {}
    for name, rows in old_catalogs.items():
        pairs = flatten(rows)
        ids_here = set(pairs)
        texts_here = set(pairs.values())
        id_overlap = sorted(set(ids) & ids_here)
        text_overlap = sorted(set(texts) & texts_here)
        overlaps[name] = {"id_overlap": id_overlap, "text_overlap": text_overlap}
        if id_overlap or text_overlap:
            raise AssertionError(f"overlap with {name}: {overlaps[name]}")
        old_ids |= ids_here
        old_texts |= texts_here

    # Stronger local checks: no M29-D text is an exact prefix/suffix duplicate
    # of an old prompt after whitespace normalization.
    norm = lambda s: " ".join(s.split()).casefold()
    normalized_old = {norm(x) for x in old_texts}
    normalized_new = {norm(x) for x in texts}
    if normalized_new & normalized_old:
        raise AssertionError("normalized text overlap detected")

    # Partition isolation is internal to M29-D.
    partition_sets = {p: {r["prompt_id"] for r in new if r["partition"] == p} for p in PARTITIONS}
    for left, right in (("update", "validation"), ("update", "holdout"), ("validation", "holdout")):
        if partition_sets[left] & partition_sets[right]:
            raise AssertionError(f"partition overlap: {left}/{right}")

    return {
        "status": "PASS",
        "catalog_sha256": catalog_sha256(),
        "n_rows": len(new),
        "partitions": {p: len(partition_sets[p]) for p in PARTITIONS},
        "families": {f: sum(r["family"] == f for r in new) for f in FAMILIES},
        "historical_catalogs_checked": list(old_catalogs),
        "historical_exact_id_overlaps": {k: v["id_overlap"] for k, v in overlaps.items()},
        "historical_exact_text_overlaps": {k: v["text_overlap"] for k, v in overlaps.items()},
        "normalized_text_overlap": [],
        "outcome_data_loaded": False,
        "qwen_loaded": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
