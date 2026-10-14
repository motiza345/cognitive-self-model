"""Frozen M29-D catalog.

Thirty-six prompts, fixed before any M29-D outcome.
Twelve UPDATE, twelve VALIDATION, twelve HOLDOUT.
Texts and ids are disjoint from all consumed M22.1-M26 catalogs.
"""

from __future__ import annotations

import hashlib
import json

PARTITIONS = ("update", "validation", "holdout")
FAMILIES = ("completion", "syntax", "instruction")

_COMPLETION = (
    "The first element in the periodic table is",
    "A hexagon has",
    "The deepest ocean trench is the",
    "Photosynthesis primarily occurs in",
    "The SI unit of electric current is the",
    "The planet with the shortest year is",
    "A prism has two congruent",
    "The freezing point of water in Celsius is",
    "The instrument that measures atmospheric pressure is the",
    "The largest internal organ of the human body is the",
    "A right angle measures",
    "The process by which plants release water vapor is called",
)
_SYNTAX = (
    "Although the bronze had cooled,",
    "If the rotor begins to wobble,",
    "After the signal crossed the threshold,",
    "While the mixture remained clear,",
    "Because the bearing was misaligned,",
    "Unless the chamber is evacuated,",
    "Once the archive has been sealed,",
    "Even if the pressure falls,",
    "As the pendulum approached equilibrium,",
    "Before the sensor reaches saturation,",
    "Provided the actuator receives power,",
    "Whenever the current reverses direction,",
)
_INSTRUCTION = (
    "Return one word. The unit of frequency is:",
    "Return one word. A polygon with eight sides is a:",
    "Return one word. The force that attracts masses is:",
    "Return one word. A device for measuring temperature is a:",
    "Return one word. The study of earthquakes is:",
    "Return one word. A material that resists electric current is an:",
    "Return one word. The center of an atom is its:",
    "Return one word. The SI unit of energy is the:",
    "Return one word. A triangle with three equal sides is:",
    "Return one word. The path of a planet around a star is an:",
    "Return one word. A substance with pH below seven is:",
    "Return one word. The change from liquid directly to gas is:",
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
            rows.append({
                "prompt_id": f"m29d-{family}-{index:02d}",
                "family": family,
                "text": text,
                "partition": partition,
            })
    return rows

def catalog_sha256() -> str:
    encoded = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

CATALOG_SHA256 = catalog_sha256()
