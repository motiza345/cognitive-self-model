"""Frozen c-* catalog for the M24 consumption audit.

Texts and the partition rule are fixed before any M24 outcome.
Twelve prompts in each surface family. Partition is (index - 1) mod 3.
"""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("train", "validation", "evaluation")
FAMILIES = ("completion", "syntax", "instruction")
CATALOG_SHA256 = "ca3fcf652a7f5cd395fc1f0dea718742f7522b7f298671bb8005dccce8370d78"

_COMPLETION = (
    "The capital of Norway is",
    "Sand becomes glass when it is",
    "The author of Don Quixote is",
    "A violin has four",
    "The Amazon is a",
    "A cube has six",
    "The currency of Sweden is the",
    "Cheese is often aged in a",
    "The smallest bird is the",
    "A minute has sixty",
    "The chemical symbol for copper is",
    "Linen is woven from",
)
_SYNTAX = (
    "In case the ferry is delayed,",
    "Supposing the lock freezes,",
    "Now that the harvest is in,",
    "By the time the oven cooled,",
    "So that the wound would close,",
    "Wherever the path divides,",
    "No matter how dark the cellar is,",
    "The moment the anchor dropped,",
    "Given that the sample was sterile,",
    "Long after the echo faded,",
    "On condition that the seal holds,",
    "Rather than leave the kiln hot,",
)
_INSTRUCTION = (
    "Reply with one word. A liquid used in thermometers:",
    "Reply with one word. A drink made from grapes:",
    "Name a tool used for sewing:",
    "Name a constellation:",
    "Reply with one word. Opposite of shallow:",
    "Name a kind of nut:",
    "Reply with one word. A soup served cold:",
    "Name a keyboard instrument:",
    "Reply with one word. The number of legs on an insect:",
    "Name a desert:",
    "Reply with one word. A reptile with a shell:",
    "Name a type of bridge:",
)


def catalog() -> list[dict[str, str]]:
    rows = []
    for family, texts in (
        ("completion", _COMPLETION),
        ("syntax", _SYNTAX),
        ("instruction", _INSTRUCTION),
    ):
        if len(texts) != 12:
            raise RuntimeError(f"{family} does not have 12 prompts")
        for index, text in enumerate(texts, start=1):
            rows.append(
                {
                    "prompt_id": f"c-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": PARTITIONS[(index - 1) % 3],
                }
            )
    return rows


def catalog_sha256() -> str:
    encoded = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
