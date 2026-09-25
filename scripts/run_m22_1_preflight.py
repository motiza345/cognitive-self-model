#!/usr/bin/env python3
"""Run the M22.1 causal intervention preflight from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cognitive_self_model.m22_1.run import main


if __name__ == "__main__":
    raise SystemExit(main())
