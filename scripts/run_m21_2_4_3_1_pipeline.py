#!/usr/bin/env python3
"""
Runnable entrypoint for the M21.2.4.3.1 benchmark + bundle generator.

Usage (from repository root):
    python scripts/run_m21_2_4_3_1_pipeline.py
    python scripts/run_m21_2_4_3_1_pipeline.py --config configs/m21_2_4_3_1_config.json --output-dir reports/M21_2_4_3_1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cognitive_self_model.benchmark.pipeline import load_config, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M21.2.4.3.1 pipeline.")
    parser.add_argument("--config", default=None, help="Path to config JSON.")
    parser.add_argument("--output-dir", default=None, help="Output directory for artifacts.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    summary = run_pipeline(config=config, output_dir=args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
