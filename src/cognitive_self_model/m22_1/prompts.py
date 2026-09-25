"""Frozen M22.1 preflight prompts.

Texts, regimes, and the split rule are fixed before any intervention is run.
This is not an M22.2 training corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

ROLES = ("discovery", "validation", "replication")

_TEXTS: dict[str, tuple[str, ...]] = {
    "completion": (
        "The capital of France is",
        "Water freezes at a temperature of",
        "The opposite of hot is",
        "Two plus two equals",
        "The sun rises in the",
        "A cat is a type of",
    ),
    "syntax": (
        "If it rains, then the ground is",
        "She opened the door and",
        "Because the road was closed,",
        "After the meeting ended,",
        "Although the box was empty,",
        "Before sunrise, the sky was",
    ),
    "instruction": (
        "Reply with one word. Color of the sky:",
        "Reply with one word. A common pet:",
        "Reply with one word. Result of 1+1:",
        "Name a day of the week:",
        "Name a primary color:",
        "Name a season of the year:",
    ),
}


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    regime_id: str
    role: str
    text: str
    prompt_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_role(index_within_regime: int) -> str:
    if index_within_regime < 0:
        raise ValueError("Prompt index must be non-negative.")
    return ROLES[index_within_regime % 3]


def frozen_prompts() -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for regime_id in sorted(_TEXTS):
        texts = _TEXTS[regime_id]
        for index, text in enumerate(texts):
            prompt_id = f"{regime_id}-{index + 1:02d}"
            records.append(
                PromptRecord(
                    prompt_id=prompt_id,
                    regime_id=regime_id,
                    role=assign_role(index),
                    text=text,
                    prompt_sha256=prompt_sha256(text),
                )
            )
    records.sort(key=lambda record: record.prompt_id)
    return records


def prompts_by_role(prompts: list[PromptRecord] | None = None) -> dict[str, list[PromptRecord]]:
    catalog = prompts if prompts is not None else frozen_prompts()
    grouped = {role: [] for role in ROLES}
    for record in catalog:
        if record.role not in grouped:
            raise ValueError(f"Unknown role: {record.role}")
        grouped[record.role].append(record)
    return grouped


def prompt_manifest_sha256(prompts: list[PromptRecord] | None = None) -> str:
    catalog = prompts if prompts is not None else frozen_prompts()
    payload = [record.to_dict() for record in catalog]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_split_integrity(prompts: list[PromptRecord] | None = None) -> None:
    catalog = prompts if prompts is not None else frozen_prompts()
    hashes = {role: set() for role in ROLES}
    seen_ids: set[str] = set()
    for record in catalog:
        if record.prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {record.prompt_id}")
        seen_ids.add(record.prompt_id)
        if record.prompt_sha256 != prompt_sha256(record.text):
            raise ValueError(f"Hash mismatch for {record.prompt_id}")
        hashes[record.role].add(record.prompt_sha256)
    if hashes["discovery"] & hashes["validation"]:
        raise ValueError("Discovery and validation prompts overlap.")
    if hashes["discovery"] & hashes["replication"]:
        raise ValueError("Discovery and replication prompts overlap.")
    if hashes["validation"] & hashes["replication"]:
        raise ValueError("Validation and replication prompts overlap.")
    if any(len(hashes[role]) == 0 for role in ROLES):
        raise ValueError("Every split must be non-empty.")
