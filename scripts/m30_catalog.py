"""Frozen M30 catalog.

Forty-eight prompts, fixed before any M30 prediction or outcome.
Twelve UPDATE, twelve VALIDATION, twenty-four HOLDOUT.
The UPDATE and VALIDATION texts are the previous revision, byte-identical.
The previous twelve HOLDOUT texts remain. Twelve further HOLDOUT prompts are
the image of rule M30-HX1 in docs/M30_PROTOCOL.md. They are not a hand-picked list.
Texts and ids are disjoint from the M22.1 through M29-D catalogs.
"""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("update", "validation", "holdout")
FAMILIES = ("completion", "syntax", "instruction")
HOLDOUT_EXTENSION_RULE = "M30-HX1"

# Slot tables for M30-HX1. k = 0,1,2,3 is the only index. Do not edit an
# emitted string in place; change this rule in the protocol first.
_EXTENSION_MATERIALS = ("birch", "copper", "ivory", "wool")
_EXTENSION_ARTICLES = ("A", "A", "An", "A")
_EXTENSION_IMPLEMENTS = ("burnisher", "bodkin", "mallet", "caliper")
_EXTENSION_TARGETS = ("fore-edge", "headband", "sewing-frame", "lying-press")
_EXTENSION_TEMPLATES = {
    "completion": "The {material} {implement} is kept beside the {target}",
    "syntax": "Were the {material} {implement} left upon the {target},",
    "instruction": "Answer with a single noun. {article} {material} {implement} belongs with the:",
}

_COMPLETION = (
    "The spine of a bound volume faces",
    "A sextant measures the angle between",
    "A cobbler lasts a shoe on a",
    "Prime-meridian bearings are read from",
    "A double-entry book balances each",
    "A musical rest marks a",
    "Hoarfrost collects on a clear",
    "Silt drops where a current",
    "A watermark becomes visible when the sheet is",
    "The helmsman keeps a ship on its",
    "Shingles are riven along the",
    "A concordance points from a word to its",
)
_SYNTAX = (
    "Should the clasp refuse to shut,",
    "Had the ferrule split along the grain,",
    "Were the colophon left off the final leaf,",
    "Might the inkwell tip during the crossing,",
    "Could the vellum buckle in damp air,",
    "Ought the quire be resewn before shelving,",
    "Must the caption agree with the plate,",
    "Shall the folio be trimmed after drying,",
    "Need the bookmark stay in the gutter,",
    "Dare the apprentice pare the edge thinner,",
    "Would the hinge tarnish in salt air,",
    "Will the signature remain after pressing,",
)
_INSTRUCTION = (
    "Answer with a single noun. The hinged cover of a book is its:",
    "Answer with a single noun. A punch for leather holes is an:",
    "Answer with a single noun. The raised rim of a wheel is the:",
    "Answer with a single noun. A map's symbol key is its:",
    "Answer with a single noun. The line that holds a sail is a:",
    "Answer with a single noun. A shallow stone for grinding pigment is a:",
    "Answer with a single noun. A person who sets type is a:",
    "Answer with a single noun. A notch cut to keep score is a:",
    "Answer with a single noun. The pattern a tailor traces is a:",
    "Answer with a single noun. A small cask for nails is a:",
    "Answer with a single noun. The loop that joins two rope ends is a:",
    "Answer with a single noun. A stand for a canvas is an:",
)


def holdout_extension_text(family: str, k: int) -> str:
    """Emit one M30-HX1 holdout prompt. k is 0, 1, 2, or 3."""
    if family not in _EXTENSION_TEMPLATES:
        raise RuntimeError(f"unknown family {family}")
    if k not in range(4):
        raise RuntimeError("M30-HX1 index k must be 0, 1, 2, or 3")
    return _EXTENSION_TEMPLATES[family].format(
        article=_EXTENSION_ARTICLES[k],
        material=_EXTENSION_MATERIALS[k],
        implement=_EXTENSION_IMPLEMENTS[k],
        target=_EXTENSION_TARGETS[k],
    )


def catalog() -> list[dict[str, str]]:
    rows = []
    for family, texts in (
        ("completion", _COMPLETION),
        ("syntax", _SYNTAX),
        ("instruction", _INSTRUCTION),
    ):
        if len(texts) != 12:
            raise RuntimeError(f"{family} does not have 12 base prompts")
        for index, text in enumerate(texts, start=1):
            partition = PARTITIONS[(index - 1) % 3]
            rows.append(
                {
                    "prompt_id": f"m30-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": partition,
                }
            )
        for k in range(4):
            index = 13 + k
            rows.append(
                {
                    "prompt_id": f"m30-{family}-{index:02d}",
                    "family": family,
                    "text": holdout_extension_text(family, k),
                    "partition": "holdout",
                }
            )
    return rows


def partition_manifest_sha256(partition: str) -> str:
    """SHA-256 of one partition's rows, sorted by prompt_id.

    The payload is JSON with sorted keys and no whitespace. This is the
    hash the design-lock test compares to the previous revision.
    """
    if partition not in PARTITIONS:
        raise RuntimeError(f"unknown partition {partition}")
    subset = sorted(
        (row for row in catalog() if row["partition"] == partition),
        key=lambda row: row["prompt_id"],
    )
    encoded = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog_sha256() -> str:
    encoded = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CATALOG_SHA256 = catalog_sha256()
