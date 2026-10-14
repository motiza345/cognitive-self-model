"""Append-only, hash-chained event log. Nothing is edited; state is a replay.

Origins: sealed seal/intervention ordering and sha256 checks in
real_self_model_qwen (order[], prediction_seal_sha256_after == before).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .core import canonical_sha256


class LedgerError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                self.events.append(json.loads(line))
            self.verify()

    @property
    def head(self) -> str:
        return self.events[-1]["event_sha256"] if self.events else "GENESIS"

    def append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {"seq": len(self.events) + 1, "kind": kind, "prev": self.head, "payload": payload}
        event["event_sha256"] = canonical_sha256(event)
        self.events.append(event)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        return event

    def verify(self) -> None:
        prev = "GENESIS"
        for i, event in enumerate(self.events, start=1):
            body = {k: v for k, v in event.items() if k != "event_sha256"}
            if event["seq"] != i or event["prev"] != prev or canonical_sha256(body) != event["event_sha256"]:
                raise LedgerError(f"ledger broken at seq {i}")
            prev = event["event_sha256"]

    def of_kind(self, kind: str) -> Iterator[dict[str, Any]]:
        return (e for e in self.events if e["kind"] == kind)
