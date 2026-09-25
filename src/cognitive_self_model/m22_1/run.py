"""One-command M22.1 preflight.

Usage from the repository root:
    python -m src.cognitive_self_model.m22_1.run
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .preflight import run_preflight
from .runtime_info import git_identity


def main() -> int:
    identity = git_identity()
    print("branch", identity["branch"])
    print("commit", identity["commit"])
    print("dirty", identity["dirty"])
    print("origin", identity["origin"])
    certificate = run_preflight(Path("reports") / "M22_1")
    print("M22.1 status", certificate["status"])
    print("model_revision", certificate.get("model_revision"))
    print("selected_layer", certificate.get("selected_layer"))
    test = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=False)
    if test.returncode != 0:
        print("pytest failed")
        return int(test.returncode)
    print("pytest passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
