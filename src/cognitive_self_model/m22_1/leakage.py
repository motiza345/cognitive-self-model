"""Import audit. Fails if the preflight package pulls in epistemic-benchmark code."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_MODULES = (
    "src.cognitive_self_model.benchmark",
    "src.legacy_import",
    "src.estimator",
    "src.calibration",
    "src.audits",
)


def audit_package_imports(package_dir: Path | None = None) -> dict[str, object]:
    root = package_dir if package_dir is not None else Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [module, *[f"{module}.{alias.name}" for alias in node.names]]
            else:
                continue
            for name in names:
                if any(name == forbidden or name.startswith(forbidden + ".") for forbidden in _FORBIDDEN_MODULES):
                    offenders.append(f"{path.name}: {name}")
    return {"pass": len(offenders) == 0, "offenders": offenders}
