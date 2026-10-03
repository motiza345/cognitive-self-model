"""Frozen u-* catalog for the M26 evidence update.

The update partition and the post-update holdout are fixed together,
before any M26 outcome. Partition is (index - 1) mod 2.
"""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("update", "holdout")
FAMILIES = ("completion", "syntax", "instruction")
CATALOG_SHA256 = "015c3d41855da5fa630c382fe0e4a9ea63c868e12e4a1243e104e1c4e371ea6f"

_COMPLETION = (
    "The capital of Portugal is",
    "Brick is fired from",
    "The author of The Odyssey is",
    "A harp has many",
    "The Danube is a",
    "A stop sign has eight",
    "The currency of Mexico is the",
    "Ink is often stored in a",
)
_SYNTAX = (
    "Assuming the valve sticks,",
    "By the time the glacier retreated,",
    "So that the seedling would root,",
    "Wherever the creek bends,",
    "No matter how loud the mill is,",
    "The moment the comet appeared,",
    "Provided the cistern stays sealed,",
    "Whereas the third assay failed,",
)
_INSTRUCTION = (
    "Reply with one word. A liquid used in lamps:",
    "Reply with one word. A drink made from barley:",
    "Name a tool used for weaving:",
    "Name a moon of Jupiter:",
    "Reply with one word. Opposite of hollow:",
    "Name a kind of spice:",
    "Reply with one word. A bread served with soup:",
    "Name a wind instrument:",
)


def catalog() -> list[dict[str, str]]:
    rows = []
    for family, texts in (
        ("completion", _COMPLETION),
        ("syntax", _SYNTAX),
        ("instruction", _INSTRUCTION),
    ):
        if len(texts) != 8:
            raise RuntimeError(f"{family} does not have 8 prompts")
        for index, text in enumerate(texts, start=1):
            rows.append(
                {
                    "prompt_id": f"u-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": PARTITIONS[(index - 1) % 2],
                }
            )
    return rows


def catalog_sha256() -> str:
    encoded = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
