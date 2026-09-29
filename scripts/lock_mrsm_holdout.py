"""Lock holdout membership. Does not load a checkpoint and does not score."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.mrsm.p_data import holdout_manifest, write_json


def main() -> int:
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = holdout_manifest(424242, 512, created)
    path = _ROOT / "artifacts" / "mrsm" / "holdout_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("hash") != manifest["hash"] or existing.get("sample_ids") != manifest["sample_ids"]:
            raise SystemExit("holdout manifest already exists and does not match this seed")
        print(json.dumps({"status": "UNCHANGED", "hash": existing["hash"], "scored": False}))
        return 0
    write_json(path, manifest)
    print(json.dumps({"status": "LOCKED", "hash": manifest["hash"], "n": manifest["n"], "scored": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
