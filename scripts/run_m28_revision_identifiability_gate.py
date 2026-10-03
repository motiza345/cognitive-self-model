"""Print the frozen M28 gate status.

This command does not load Qwen and does not read outcome files.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m28_identifiability import execution_gate


def main() -> None:
    print(execution_gate())


if __name__ == "__main__":
    main()
