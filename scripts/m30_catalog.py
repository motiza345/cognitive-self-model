"""Frozen M30 catalog.

Thirty-six prompts, fixed before any M30 prediction or outcome.
Twelve UPDATE, twelve VALIDATION, twelve HOLDOUT.
Texts and ids are disjoint from the M22.1 through M29-D catalogs.
"""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("update", "validation", "holdout")
FAMILIES = ("completion", "syntax", "instruction")

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
            partition = PARTITIONS[(index - 1) % 3]
            rows.append(
                {
                    "prompt_id": f"m30-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": partition,
                }
            )
    return rows


def catalog_sha256() -> str:
    encoded = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CATALOG_SHA256 = catalog_sha256()
