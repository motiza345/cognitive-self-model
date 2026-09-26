"""Run the M22.1.3b first-order structure check. No model forward."""

from __future__ import annotations

from pathlib import Path

from cognitive_self_model.m22_1_3b.compute import run_audit, write_outputs

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = run_audit(ROOT)
    write_outputs(payload, ROOT)
    print(payload["status"])
    print(payload["label"])
    print(payload["first_failure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
