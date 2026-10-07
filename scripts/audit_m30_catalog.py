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


def _flatten(rows) -> dict[str, str]:
    return {str(row["prompt_id"]): str(row["text"]) for row in rows}


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def audit() -> dict[str, object]:
    new = catalog()
    if len(new) != 36:
        raise AssertionError("M30 catalog must contain 36 rows")
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
    for partition in PARTITIONS:
        rows = [row for row in new if row["partition"] == partition]
        if len(rows) != 12:
            raise AssertionError(f"{partition} must contain 12 prompts")
        if {row["family"] for row in rows} != set(FAMILIES):
            raise AssertionError(f"{partition} is missing a family")

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
    old_texts: set[str] = set()
    for name, rows in old_catalogs.items():
        pairs = _flatten(rows)
        id_overlap = sorted(set(ids) & set(pairs))
        text_overlap = sorted(set(texts) & set(pairs.values()))
        normalized_overlap = sorted({_norm(text) for text in texts} & {_norm(text) for text in pairs.values()})
        overlaps[name] = {
            "id_overlap": id_overlap,
            "text_overlap": text_overlap,
            "normalized_text_overlap": normalized_overlap,
        }
        if id_overlap or text_overlap or normalized_overlap:
            raise AssertionError(f"overlap with {name}: {overlaps[name]}")
        old_texts |= set(pairs.values())

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
        "- Total prompts: **36**",
        "- UPDATE: **12**",
        "- VALIDATION: **12**",
        "- HOLDOUT: **12**",
        "- Families: completion / syntax / instruction",
        "- Each partition contains 4 prompts from each family.",
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
    lines.extend(
        [
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
            "For each family, index 1..12 is assigned `(index - 1) mod 3` to UPDATE, VALIDATION, HOLDOUT.",
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
