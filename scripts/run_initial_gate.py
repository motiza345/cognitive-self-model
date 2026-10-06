"""Run the IG-0 mechanism-identity gate and write a JSON result."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognitive_self_model.initial_gate import format_report, run_gate


def main() -> int:
    result = run_gate()
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "initial_gate_result.json"
    stored = {key: value for key, value in result.items() if key != "seeds"}
    stored["seed_gate_failures"] = [
        {"seed": report["seed"], "failed": [name for name, ok in report["gates"].items() if not ok]}
        for report in result["seeds"]
        if not report["passed"]
    ]
    output_path.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    print(format_report(result))
    print(f"\nWrote {output_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
