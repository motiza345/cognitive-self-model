"""Score the Qwen scope-update rule on the frozen CCSO measurements.

Does not run Qwen again and does not edit those measurements.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.mrsm.ccso import DIRECTION_SEEDS, LAYERS, REGIMES  # noqa: E402
from src.mrsm.qf_gradient_field import QWEN_REVISION  # noqa: E402
from src.mrsm.qwen_belief_update import (  # noqa: E402
    CONTROL_EXPECTATION,
    PRIMARY_EXPECTATION,
    decide_qwen,
    evaluate_cases,
    preregistration_document,
)

INDEX_PATH = ROOT / "artifacts" / "ccso" / "measurement_index.json"
NPZ_PATH = ROOT / "artifacts" / "ccso" / "measurements.npz"
RULE_PATH = ROOT / "src" / "mrsm" / "belief_update.py"
OUT = ROOT / "artifacts" / "qwen_belief_update"
REPORT = ROOT / "reports" / "QWEN_BELIEF_UPDATE.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Qwen belief scope update",
        "",
        "Diagnostic on the frozen CCSO measurements. Not a new Qwen forward, not an MRSM rescore, and not M22.",
        "",
        "## 1. Question",
        "",
        "Does a belief stored from one regime hold on later prompts of that regime, revise scope when another regime misses, and abstain when the direction was never stored?",
        "",
        f"- Decision: `{result['decision']['decision']}`.",
        f"- {result['decision']['reason']} No mechanism was named. The P head table and the P floor of 1 were not used.",
        "",
        "## 2. What was fixed before scoring",
        "",
        "- Fit uses discovery prompts only. Validation and replication are records.",
        "- Each stored intervention has its own discovery standard deviation of Δ/α. Tolerance is twice that value.",
        "- Train directions are 23101–23106. Novel directions 23107 and 23108 are absent from the belief.",
        "- Primary alphas only. ±0.25 is excluded.",
        "- A novel direction makes the two-change case abstain because the belief has no prediction for it.",
        "",
        "## 3. Combined outputs",
        "",
        "| Case | Output | Expected |",
        "| --- | --- | --- |",
    ]
    for name, expected in PRIMARY_EXPECTATION.items():
        lines.append(f"| {name} | `{result['primary'].get(name)}` | `{expected}` |")
    lines.extend(["", "| Case | Output | Expected |", "| --- | --- | --- |"])
    for name, expected in CONTROL_EXPECTATION.items():
        lines.append(f"| {name} | `{result['controls'].get(name)}` | `{expected}` |")
    lines.extend(["", "## 4. Pair trace", "", "This trace is not a decision input and does not reopen the label.", ""])
    trace = result["trace"]
    for name in ("same_regime", "regime_change", "two_changes", "scope_string_alone", "discovery_self"):
        lines.append(f"### {name}")
        lines.append("")
        for key, label in trace.get(name, {}).items():
            if key == "decision_input":
                continue
            lines.append(f"- {key}: `{label}`")
        lines.append("")
    lines.extend(
        [
            "## 5. What this does not say",
            "",
            "Separation means these frozen slices received the three preregistered updates.",
            "A mismatch means the stored regime effect did not move in the way the three labels require.",
            "Neither result names a mechanism or transfers the eight-head table onto Qwen.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _write(OUT / "preregistration.json", preregistration_document())
    rule_before = _sha256(RULE_PATH)
    npz_before = _sha256(NPZ_PATH)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    split_ok = (
        index["revision"] == QWEN_REVISION
        and index["layers"] == list(LAYERS)
        and index["direction_seeds"] == list(DIRECTION_SEEDS)
        and index["roles"].count("discovery") == 6
        and set(index["regimes"]) == set(REGIMES)
    )
    stored = np.load(NPZ_PATH)
    evaluated = evaluate_cases(
        stored["delta"],
        stored["alphas"],
        list(index["roles"]),
        list(index["regimes"]),
        [int(seed) for seed in index["direction_seeds"]],
    )
    if not evaluated["ok"]:
        decision = {
            "decision": "INCONCLUSIVE",
            "reason_code": "PRECONDITION",
            "reason": "The frozen split did not contain every regime.",
        }
        primary = {}
        controls = {}
        finite_ok = False
        self_ok = False
        trace = evaluated["trace"]
    else:
        primary = evaluated["primary"]
        controls = evaluated["controls"]
        finite_ok = bool(evaluated["finite"])
        self_ok = bool(evaluated["self_ok"])
        trace = evaluated["trace"]
        decision = decide_qwen(primary, controls, split_ok=split_ok, finite_ok=finite_ok, self_ok=self_ok)
    result = {
        "controls": controls,
        "decision": decision,
        "finite_ok": finite_ok if evaluated["ok"] else False,
        "measurement_sha256_after": _sha256(NPZ_PATH),
        "measurement_sha256_before": npz_before,
        "mechanism_identity": "NOT_EVALUATED",
        "m22_started": False,
        "new_qwen_forward": False,
        "p_floor_used": False,
        "p_head_table_used": False,
        "primary": primary,
        "rule_sha256_after": _sha256(RULE_PATH),
        "rule_sha256_before": rule_before,
        "self_ok": self_ok if evaluated["ok"] else False,
        "split_ok": split_ok,
        "trace": trace,
    }
    _write(OUT / "decision.json", result)
    manifest = [f"{_sha256(path)}  {path.name}" for path in sorted(OUT.glob("*.json"))]
    (OUT / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    REPORT.write_text(_markdown(result), encoding="utf-8")
    unchanged = result["measurement_sha256_before"] == result["measurement_sha256_after"]
    rule_same = result["rule_sha256_before"] == result["rule_sha256_after"]
    print(decision["decision"])
    print(decision["reason_code"])
    print(primary)
    print("measurements_unchanged", unchanged, "p_rule_unchanged", rule_same)


if __name__ == "__main__":
    main()
