"""
M21.2.4.3.1 pipeline entrypoint.

Historically this module was an empty placeholder; the actual implementation now
lives in the clean package :mod:`src.cognitive_self_model.benchmark`. This module
is a thin, stable entrypoint that runs that pipeline and writes the eight-key
``M21_2_4_3_3_input.npz`` prediction bundle consumed by the M21.2.4.3.3 audit.

Usage:
    python -m src.legacy_import.m21_2_4_3_1_pipeline
    python -m src.legacy_import.m21_2_4_3_1_pipeline --output-dir reports/M21_2_4_3_1
"""

from __future__ import annotations

import argparse
import json

from src.cognitive_self_model.benchmark.pipeline import load_config, run_pipeline


def main(input_csv: str | None = None, config_json: str | None = None) -> dict:
    """Run the M21.2.4.3.1 pipeline.

    ``input_csv`` is accepted for backward-compatible call signatures but is
    unused: this milestone *generates* its data from the benchmark oracle rather
    than consuming an external CSV.
    """
    config = load_config(config_json) if config_json else None
    summary = run_pipeline(config=config)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the M21.2.4.3.1 pipeline.")
    parser.add_argument("--config", default=None, help="Path to config JSON.")
    parser.add_argument("--output-dir", default=None, help="Output directory for artifacts.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else None
    result = run_pipeline(config=config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))
