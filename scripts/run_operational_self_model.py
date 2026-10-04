"""Run the frozen operational self-model acceptance loop once.

This command does not load a language model and does not read earlier
milestone artifacts.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.operational_self_model import run_acceptance

PROTOCOL_PATH = ROOT / "reports" / "OPERATIONAL_SELF_MODEL_PROTOCOL.md"
RESULTS_PATH = ROOT / "reports" / "OPERATIONAL_SELF_MODEL_RESULTS.json"
EXECUTION_PATH = ROOT / "reports" / "OPERATIONAL_SELF_MODEL_EXECUTION.md"


def _render(report: dict[str, object], protocol_sha256: str) -> str:
    lines = [
        "# Operational self-model execution",
        "",
        f"Protocol version: `{report['protocol_version']}`",
        "",
        f"Protocol SHA-256: `{protocol_sha256}`",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "The endpoint is the decision difference between the updated model and the no-update clone.",
        "",
        "| check | value |",
        "| --- | --- |",
        f"| initial claim | `{report['initial_claim']}` |",
        f"| initial version | `{report['initial_version']}` |",
        f"| initial decision | `{report['initial_decision']}` |",
        f"| sealed before outcome | `{report['sealed_before_outcome']}` |",
        f"| prediction | `{report['prediction_value']}` |",
        f"| observed outcome | `{report['observed_outcome']}` |",
        f"| updated claim | `{report['updated_claim']}` |",
        f"| updated version | `{report['updated_version']}` |",
        f"| updated decision | `{report['updated_decision']}` |",
        f"| no-update decision | `{report['no_update_decision']}` |",
        f"| decisions differ | `{report['decisions_differ']}` |",
        f"| agreeing claim | `{report['agreeing_claim']}` |",
        f"| prediction unchanged | `{report['prediction_unchanged']}` |",
        f"| evidence unchanged | `{report['evidence_unchanged']}` |",
        f"| prior version unchanged | `{report['prior_version_unchanged']}` |",
        f"| changed fields | `{', '.join(report['changed_fields'])}` |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if RESULTS_PATH.exists() or EXECUTION_PATH.exists():
        raise SystemExit("operational self-model artifacts already exist")
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    report = run_acceptance()
    payload = dict(report)
    payload["protocol_sha256"] = protocol_sha256
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    EXECUTION_PATH.write_text(_render(report, protocol_sha256) + "\n", encoding="utf-8")
    print(report["verdict"])


if __name__ == "__main__":
    main()
