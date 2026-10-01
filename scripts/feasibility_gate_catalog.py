"""Frozen f-* catalog for the project feasibility gate.

Texts were fixed before any feasibility-gate outcome. The partition rule is
(k - 1) mod 4, as specified in reports/PROJECT_FEASIBILITY_GATE.md.
"""

from __future__ import annotations

PARTITIONS = ("train", "validation", "evaluation", "replication")

_COMPLETION = (
    "The boiling point of ethanol is",
    "A standard chessboard has",
    "The painter of the Mona Lisa is",
    "Mercury is a metal that is",
    "The Nile is a",
    "A pentagon has five",
    "The currency of Canada is the",
    "Glass is usually melted in a",
    "The slowest mammal is the",
    "An hour has sixty",
    "The chemical symbol for silver is",
    "Cotton grows on a",
    "The inventor of the telephone is",
    "A leap year has",
    "The largest bone in the body is the",
    "Honey is made by a",
)
_SYNTAX = (
    "After the bridge closed,",
    "If the ink spills,",
    "Before the curtain fell,",
    "Although the well was dry,",
    "Because the rope had snapped,",
    "Unless the gate is shut,",
    "While the dough was rising,",
    "As soon as the whistle blew,",
    "Even though the lens was cracked,",
    "Whenever the tide turns,",
    "Since the harbor was empty,",
    "Until the frost ended,",
    "Once the letter was sealed,",
    "Provided the axle is greased,",
    "Whether or not the lamp works,",
    "Whereas the second sample passed,",
)
_INSTRUCTION = (
    "Reply with one word. A gas used in balloons:",
    "Reply with one word. The meal eaten at dawn:",
    "Name a tool used for digging:",
    "Name a month with thirty days:",
    "Reply with one word. Opposite of narrow:",
    "Name a kind of grain:",
    "Reply with one word. A baked fruit dessert:",
    "Name a percussion instrument:",
    "Reply with one word. The number of sides on a pentagon:",
    "Name a sea:",
    "Reply with one word. A fish that can shock:",
    "Name a piece of jewelry:",
    "Reply with one word. A stone used in pencils:",
    "Name a unit of length:",
    "Reply with one word. The color of emeralds:",
    "Name a room in a house:",
)


def catalog() -> list[dict[str, str]]:
    rows = []
    for family, texts in (
        ("completion", _COMPLETION),
        ("syntax", _SYNTAX),
        ("instruction", _INSTRUCTION),
    ):
        if len(texts) != 16:
            raise RuntimeError(f"{family} does not have 16 prompts")
        for index, text in enumerate(texts, start=1):
            rows.append(
                {
                    "prompt_id": f"f-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": PARTITIONS[(index - 1) % 4],
                }
            )
    return rows
