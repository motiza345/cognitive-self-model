"""Score the sign-level evidence trace on frozen CCSO measurements.

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

from src.mrsm.evidence_belief import decide, evaluate_measurements, preregistration_document  # noqa: E402
from src.mrsm.qf_gradient_field import QWEN_REVISION  # noqa: E402

INDEX_PATH = ROOT / "artifacts" / "ccso" / "measurement_index.json"
NPZ_PATH = ROOT / "artifacts" / "ccso" / "measurements.npz"
RULE_PATH = ROOT / "src" / "mrsm" / "belief_update.py"
QWEN_RULE_PATH = ROOT / "src" / "mrsm" / "qwen_belief_update.py"
OUT = ROOT / "artifacts" / "evidence_belief"
REPORT = ROOT / "reports" / "EVIDENCE_BELIEF.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pct(correct: int, total: int) -> str:
    if total == 0:
        return "n=0"
    return f"{correct}/{total} = {correct / total:.6f}"


def _markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    decision = result["decision"]
    lines = [
        "# Evidence, belief, scope, falsification, persistence",
        "",
        "Diagnostic on the frozen CCSO measurements. Not a new Qwen forward, not a new Self-Model, and not M22.",
        "",
        "## 1. Question",
        "",
        "Can a sign claim stored from two discovery prompts of one regime predict later prompts of that regime, leave itself unchanged when another regime disagrees, contest itself when its own regime disagrees, keep the evidence, and use the revised state on the next prompt?",
        "",
        f"- Decision: `{decision['decision']}`.",
        f"- {decision['reason']}",
        "- Mechanism identity was not evaluated. A historical notebook status was not used as a label.",
        "",
        "## 2. What was fixed before scoring",
        "",
        "- The effect is the mean of Δ/α over the six primary alphas. The sign is the sign of that mean. Zero is not a sign.",
        "- A claim exists only when the two discovery prompts of one regime share a nonzero sign. Scope is that regime.",
        "- Later prompts are applied in frozen index order. Validation sign accuracy must be strictly above 0.5.",
        "- An out-of-scope prompt is recorded and does not change the claim. Scope is not expanded.",
        "- An in-scope opposite sign contests the claim. A later match does not reactivate it.",
        "- Evidence records are append-only. The magnitude tolerance from the earlier Qwen update is not used.",
        "- Train directions are 23101–23106. Novel directions and alpha ±0.25 are excluded.",
        "- M18.7 is not a gate. Missing archive notebooks were not invented.",
        "",
        "## 3. Counts",
        "",
        f"- Scoreable claims: {summary['scoreable']} / {summary['possible_cells']}.",
        f"- Validation predictions: {_pct(summary['prediction_correct'], summary['prediction_n'])}.",
        f"- Confirmations: {summary['confirmation_n']}.",
        f"- In-scope falsifications: {summary['falsification_n']}.",
        f"- Out-of-scope prompts: {summary['scope_n']}.",
        f"- Out-of-scope sign conflicts: {summary['scope_conflict_n']}.",
        f"- Scope violations: {summary['scope_violations']}.",
        f"- Falsification-flag violations: {summary['falsification_violations']}.",
        f"- Update violations: {summary['update_violations']}.",
        f"- Persistence violations: {summary['persistence_violations']}.",
        "",
        "| Regime | Validation sign accuracy | In-scope falsifications |",
        "| --- | --- | --- |",
    ]
    for regime, row in summary["by_regime"].items():
        lines.append(
            f"| {regime} | {_pct(row['prediction_correct'], row['prediction_n'])} | {row['falsification_n']} |"
        )
    lines.extend(
        [
            "",
            "## 4. What this does not say",
            "",
            "A held trace means these frozen sign checks occurred and matched the rule. It does not name a mechanism and it does not transfer the planted Self-Model onto Qwen.",
            "A prediction failure means the discovery sign did not stay above chance on later prompts of the same regime.",
            "An inconclusive result means a required case, such as an in-scope sign conflict, did not occur in this frozen slice.",
            "The earlier magnitude result, `QWEN_UPDATE_DOES_NOT_SEPARATE`, is unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    document = preregistration_document()
    _write(OUT / "preregistration.json", document)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if index["revision"] != QWEN_REVISION:
        raise SystemExit("measurement index revision does not match the frozen Qwen revision")
    npz_before = _sha256(NPZ_PATH)
    rule_before = _sha256(RULE_PATH)
    qwen_rule_before = _sha256(QWEN_RULE_PATH)
    with np.load(NPZ_PATH) as payload:
        delta = np.array(payload["delta"])
        alphas = np.array(payload["alphas"])
    evaluated = evaluate_measurements(
        delta,
        alphas,
        list(index["roles"]),
        list(index["regimes"]),
        [int(seed) for seed in index["direction_seeds"]],
    )
    decision = decide(evaluated["summary"])
    result = {
        "decision": decision,
        "measurement_sha256_after": _sha256(NPZ_PATH),
        "measurement_sha256_before": npz_before,
        "mechanism_identity": "NOT_EVALUATED",
        "m22_started": False,
        "new_qwen_forward": False,
        "new_self_model": False,
        "qwen_rule_sha256_after": _sha256(QWEN_RULE_PATH),
        "qwen_rule_sha256_before": qwen_rule_before,
        "revision": index["revision"],
        "rule_sha256_after": _sha256(RULE_PATH),
        "rule_sha256_before": rule_before,
        "summary": evaluated["summary"],
        "traces": evaluated["traces"],
    }
    _write(OUT / "decision.json", result)
    manifest_paths = [OUT / "preregistration.json", OUT / "decision.json"]
    manifest = [f"{_sha256(path)}  {path.name}" for path in manifest_paths]
    (OUT / "manifest.sha256").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    REPORT.write_text(_markdown(result), encoding="utf-8")
    unchanged = result["measurement_sha256_before"] == result["measurement_sha256_after"]
    rules_same = result["rule_sha256_before"] == result["rule_sha256_after"]
    qwen_same = result["qwen_rule_sha256_before"] == result["qwen_rule_sha256_after"]
    print(decision["decision"])
    print(decision["reason_code"])
    print(
        "scoreable",
        result["summary"]["scoreable"],
        "accuracy",
        result["summary"]["prediction_accuracy"],
        "falsifications",
        result["summary"]["falsification_n"],
        "scope_conflicts",
        result["summary"]["scope_conflict_n"],
    )
    print("measurements_unchanged", unchanged, "p_rule_unchanged", rules_same, "qwen_rule_unchanged", qwen_same)


if __name__ == "__main__":
    main()
