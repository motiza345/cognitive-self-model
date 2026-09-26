#!/usr/bin/env python3
"""Run the M22.1.3 readout-geometry null audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m22_1_3.compare import run_compare
from src.cognitive_self_model.m22_1_3.predict import run_predictions


if __name__ == "__main__":
    run_predictions()
    run_compare()
