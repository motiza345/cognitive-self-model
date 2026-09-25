#!/usr/bin/env python3
"""Run the M22.1.2 response-space audit from frozen M22.1.1 measurements."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m22_1_2.protocol import run_audit

if __name__ == "__main__":
    run_audit(ROOT / "configs" / "m22_1_2_response_space_audit.json", ROOT / "reports")
