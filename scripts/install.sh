#!/usr/bin/env bash
#
# Idempotent environment bootstrap for the Cognitive Self-Model repository.
#
# Installs the CPU build of PyTorch from the official wheel index, then the rest
# of the Python dependencies from PyPI. Safe to run repeatedly.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

echo "==> Upgrading pip tooling"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

echo "==> Installing PyTorch (CPU build)"
"$PYTHON" -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0"

echo "==> Installing project requirements"
"$PYTHON" -m pip install -r requirements.txt

echo "==> Verifying key imports"
"$PYTHON" - <<'PY'
import numpy, scipy, sklearn, torch, transformers
from transformer_lens import HookedTransformer
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("environment OK")
PY

echo "==> Install complete"
