"""Audit the M30 catalog against M22.1 through M29-D.

This is a pre-outcome catalog audit. It does not load Qwen and does not
read M29 or M30 intervention outcomes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feasibility_gate_catalog import catalog as feasibility_catalog
from scripts.m24_consumption_catalog import catalog as m24_catalog
from scripts.m26_update_catalog import catalog as m26_catalog
from scripts.m29d_catalog import catalog as m29_catalog
from scripts.m30_catalog import FAMILIES, PARTITIONS, catalog, catalog_sha256
from scripts.validate_m23_g_protocol import catalog as m23g_catalog
from scripts.validate_post_m23_single_measurement import catalog as single_measurement_catalog
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts

REPORT_PATH = ROOT / "reports" / "M30_CATALOG_AUDIT.md"


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _index(row: dict[str, str]) -> int:
    return int(str(row["prompt_id"]).rsplit("-", 1)[1])


def _overlap(left_ids: list[str], left_texts: list[str], right: list[dict[str, str]]) -> dict[str, list[str]]:
    right_ids = [str(row["prompt_id"]) for row in right]
    right_texts = [str(row["text"]) for row in right]
    return {
        "id_overlap": sorted(set(left_ids) & set(right_ids)),
        "text_overlap": sorted(set(left_texts) & set(right_texts)),
        "normalized_text_overlap": sorted({_norm(text) for text in left_texts} & {_norm(text) for text in right_texts}),
    }


def audit() -> dict[str, object]:
    new = catalog()
    if len(new) != 48:
        raise AssertionError("M30 catalog must contain 48 rows")
    ids = [row["prompt_id"] for row in new]
    texts = [row["text"] for row in new]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate M30 prompt ids")
    if len(texts) != len(set(texts)):
        raise AssertionError("duplicate M30 prompt texts")
    if {row["family"] for row in new} != set(FAMILIES):
        raise AssertionError("family coverage is incomplete")
    if {row["partition"] for row in new} != set(PARTITIONS):
        raise AssertionError("partition coverage is incomplete")
    expected_n = {"update": 12, "validation": 12, "holdout": 24}
    expected_per_family = {"update": 4, "validation": 4, "holdout": 8}
    for partition in PARTITIONS:
        rows = [row for row in new if row["partition"] == partition]
        if len(rows) != expected_n[partition]:
            raise AssertionError(f"{partition} must contain {expected_n[partition]} prompts")
        for family in FAMILIES:
            count = sum(row["family"] == family for row in rows)
            if count != expected_per_family[partition]:
                raise AssertionError(f"{partition} must contain {expected_per_family[partition]} {family} prompts")
    extension = [row for row in new if _index(row) >= 13]
    if len(extension) != 12:
        raise AssertionError("M30-HX1 must add 12 holdout prompts")
    for family in FAMILIES:
        indexes = sorted(_index(row) for row in extension if row["family"] == family)
        if indexes != [13, 14, 15, 16]:
            raise AssertionError(f"M30-HX1 indices for {family} must be 13..16")
    if any(row["partition"] != "holdout" for row in extension):
        raise AssertionError("M30-HX1 rows must be holdout prompts")

    old_catalogs = {
        "M22.1": [{"prompt_id": prompt.prompt_id, "text": prompt.text} for prompt in frozen_prompts()],
        "M23-G": m23g_catalog(),
        "POST-M23-SINGLE": single_measurement_catalog(),
        "FEASIBILITY": feasibility_catalog(),
        "M24": m24_catalog(),
        "M26": m26_catalog(),
        "M29-D": m29_catalog(),
    }
    overlaps = {}
    for name, rows in old_catalogs.items():
        overlaps[name] = _overlap(ids, texts, rows)
        item = overlaps[name]
        if item["id_overlap"] or item["text_overlap"] or item["normalized_text_overlap"]:
            raise AssertionError(f"overlap with {name}: {item}")

    internal_overlaps = {}
    extension_ids = [row["prompt_id"] for row in extension]
    extension_texts = [row["text"] for row in extension]
    for partition in ("update", "validation"):
        other = [row for row in new if row["partition"] == partition]
        internal_overlaps[partition] = _overlap(extension_ids, extension_texts, other)
        item = internal_overlaps[partition]
        if item["id_overlap"] or item["text_overlap"] or item["normalized_text_overlap"]:
            raise AssertionError(f"new holdout overlaps {partition}: {item}")

    partition_sets = {partition: {row["prompt_id"] for row in new if row["partition"] == partition} for partition in PARTITIONS}
    for left, right in (("update", "validation"), ("update", "holdout"), ("validation", "holdout")):
        if partition_sets[left] & partition_sets[right]:
            raise AssertionError(f"partition overlap: {left}/{right}")

    return {
        "status": "PASS",
        "catalog_sha256": catalog_sha256(),
        "n_rows": len(new),
        "partitions": {partition: len(partition_sets[partition]) for partition in PARTITIONS},
        "families": {family: sum(row["family"] == family for row in new) for family in FAMILIES},
        "historical_catalogs_checked": list(old_catalogs),
        "overlaps": overlaps,
        "internal_overlaps": internal_overlaps,
        "outcome_data_loaded": False,
        "qwen_loaded": False,
        "m29_holdout_statistics_loaded": False,
    }


def render(result: dict[str, object]) -> str:
    overlaps = result["overlaps"]
    lines = [
        "# M30 catalog audit",
        "",
        "**Status:** PASS",
        "**Catalog:** `scripts/m30_catalog.py`",
        f"**Catalog SHA-256:** `{result['catalog_sha256']}`",
        "",
        "## Structure",
        "",
        "- Total prompts: **48**",
        "- UPDATE: **12**",
        "- VALIDATION: **12**",
        "- HOLDOUT: **24**",
        "- Families: completion / syntax / instruction",
        "- UPDATE and VALIDATION contain 4 prompts from each family.",
        "- HOLDOUT contains 8 prompts from each family.",
        "- Partition intersections: **0**",
        "",
        "## Historical catalogs",
        "",
        "Exact identifier overlap, exact text overlap, and whitespace-collapsed case-folded text overlap were checked against M22.1 through M29-D.",
        "",
        "| Historical source | Exact text overlap | Normalized text overlap | ID overlap |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = {
        "M22.1": "M22.1",
        "M23-G": "M23-G",
        "POST-M23-SINGLE": "POST-M23 single measurement",
        "FEASIBILITY": "Feasibility gate",
        "M24": "M24",
        "M26": "M26",
        "M29-D": "M29-D",
    }
    for name in result["historical_catalogs_checked"]:
        item = overlaps[name]
        lines.append(
            f"| {labels[name]} | {len(item['text_overlap'])} | {len(item['normalized_text_overlap'])} | {len(item['id_overlap'])} |"
        )
    internal = result["internal_overlaps"]
    lines.extend(
        [
            "",
            "## Inside M30",
            "",
            "The twelve M30-HX1 holdout prompts were compared with UPDATE and with VALIDATION on exact id, exact text, and whitespace-collapsed case-folded text.",
            "",
            "| Comparison | Exact text overlap | Normalized text overlap | ID overlap |",
            "| --- | ---: | ---: | ---: |",
            f"| New holdout vs UPDATE | {len(internal['update']['text_overlap'])} | {len(internal['update']['normalized_text_overlap'])} | {len(internal['update']['id_overlap'])} |",
            f"| New holdout vs VALIDATION | {len(internal['validation']['text_overlap'])} | {len(internal['validation']['normalized_text_overlap'])} | {len(internal['validation']['id_overlap'])} |",
            "",
            "## Outcome leakage",
            "",
            "- Qwen loaded: **false**",
            "- M30 outcomes loaded: **false**",
            "- M29 holdout statistics loaded: **false**",
            "- Feature selection from M29 cell scores: **false**",
            "",
            "This is a pre-outcome catalog audit.",
            "",
            "## Frozen partition rule",
            "",
            "For each family, index 1..12 is assigned `(index - 1) mod 3` to UPDATE, VALIDATION, HOLDOUT. Indices 13..16 are HOLDOUT by rule M30-HX1.",
            "",
            "## Scientific status",
            "",
            "**CATALOG_DISJOINTNESS_PASS**",
            "",
            "The catalog is eligible for the M30 design lock. No scored run is authorized by this audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = audit()
    REPORT_PATH.write_text(render(result), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "catalog_sha256", "n_rows", "qwen_loaded")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
