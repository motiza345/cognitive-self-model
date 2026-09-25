#!/usr/bin/env python3
"""
Generate the real M21.2.4.3.4 ``input_dataset.csv`` from the benchmark oracle.

This replaces the synthetic ``binomial``-labeled fallback dataset with real
oracle output (``label_invalid`` = ``self_model_invalid`` from the environment,
``raw_q_invalid`` = the raw estimator probability).

Usage (from repository root):
    python scripts/generate_m21_2_4_3_4_dataset.py
    python scripts/generate_m21_2_4_3_4_dataset.py --output reports/M21_2_4_3_4/input_dataset.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cognitive_self_model.benchmark.identifiability_dataset import (
    generate_identifiability_csv,
)
from src.cognitive_self_model.benchmark.pipeline import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the real M21.2.4.3.4 dataset.")
    parser.add_argument(
        "--output",
        default="reports/M21_2_4_3_4/input_dataset.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--config", default=None, help="Benchmark config JSON (M21.2.4.3.1).")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    path = generate_identifiability_csv(args.output, config=config)
    print(f"[OK] real identifiability dataset written to: {path}")


if __name__ == "__main__":
    main()
