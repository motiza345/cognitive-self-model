"""One execution of the sealed minimal self-model architecture test.

The protocol hash is checked before the episode list runs. An existing
results file stops the process.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.minimal_self_model import (  # noqa: E402
    EXPECTED_PROTOCOL_SHA256,
    render_execution,
    run_architecture_test,
)

PROTOCOL = ROOT / "reports" / "MINIMAL_SELF_MODEL_ARCHITECTURE_TEST_PROTOCOL.md"
RESULTS = ROOT / "reports" / "MINIMAL_SELF_MODEL_ARCHITECTURE_TEST_RESULTS.json"
EXECUTION = ROOT / "reports" / "MINIMAL_SELF_MODEL_ARCHITECTURE_TEST_EXECUTION.md"
EPISODES = ROOT / "reports" / "MINIMAL_SELF_MODEL_ARCHITECTURE_TEST_EPISODES.csv"

COLUMNS = (
    "episode_id",
    "role",
    "measured_state",
    "evidence_outcome",
    "evidence_class",
    "evidence_strength",
    "primary",
    "ground_truth_effect",
    "correct_action",
    "decision_B0",
    "decision_B1",
    "decision_B2_NO_SCOPE",
    "decision_B2",
    "b2_status",
    "b2_scope",
)


def main() -> None:
    if RESULTS.exists():
        raise SystemExit("results already exist; the architecture test was not rerun")
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if digest != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("protocol hash mismatch; the architecture test was not run")
    result = run_architecture_test(digest)
    RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    EXECUTION.write_text(render_execution(result), encoding="utf-8")
    with EPISODES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result["episodes"]:
            writer.writerow(row)
    print(result["verdict"])
    print(json.dumps(result["primary_correct_count"], sort_keys=True))


if __name__ == "__main__":
    main()
