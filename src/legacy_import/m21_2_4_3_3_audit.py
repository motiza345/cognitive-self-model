"""
M21.2.4.3.3 audit entrypoint.

Historically this module was an empty placeholder. The full calibration-stress,
support, exact-output, and transfer audit is implemented in
:mod:`src.legacy_import.m212433_calibration_audit`. This module is a thin, stable
entrypoint that runs that audit against an eight-key prediction bundle.

Usage:
    python -m src.legacy_import.m21_2_4_3_3_audit --input_npz reports/M21_2_4_3_1/M21_2_4_3_3_input.npz
"""

from __future__ import annotations

import argparse

from src.legacy_import.m212433_calibration_audit import main as _audit_main


def main(input_npz: str) -> None:
    _audit_main(input_npz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the M21.2.4.3.3 calibration audit.")
    parser.add_argument(
        "--input_npz",
        required=True,
        help="Path to the eight-key M21.2.4.3.3 input NPZ bundle.",
    )
    args = parser.parse_args()
    main(args.input_npz)
