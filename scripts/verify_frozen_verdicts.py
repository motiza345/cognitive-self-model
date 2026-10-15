"""Recompute headline numbers of the frozen M29-D and M30 verdicts from stored rows.

Needs only the Python standard library (no torch, no model download). It reads the
committed raw artifacts, recomputes the numbers with the same scoring helpers
(`scripts.m29d_score`, `src.cognitive_self_model.m23.stats.paired_mean_ci`), and compares
them with the frozen `VERDICT.json` files. It writes nothing.

Usage: python -m scripts.verify_frozen_verdicts
Exit code 0 when everything matches, 1 otherwise.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m29d_score import EXPECTED_PREDICTION_SHA256, score_m29  # noqa: E402
from src.cognitive_self_model.m23.stats import paired_mean_ci  # noqa: E402

TOL = 1e-9
failures: list[str] = []


def check(name: str, got: float, expected: float, tol: float = TOL) -> None:
    ok = abs(got - expected) <= tol
    print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: recomputed {got:.9f} frozen {expected:.9f}")
    if not ok:
        failures.append(name)


def check_text(name: str, got: str, expected: str) -> None:
    ok = got == expected
    print(f"  [{'ok' if ok else 'MISMATCH'}] {name}: recomputed {got} frozen {expected}")
    if not ok:
        failures.append(name)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m29d() -> None:
    print("M29-D (frozen holdout)")
    raw = ROOT / "reports" / "m29d_raw"
    verdict = json.loads((raw / "VERDICT.json").read_text(encoding="utf-8"))
    check_text("prediction sha256", sha256(raw / "PRE_OUTCOME_PREDICTIONS.json"), EXPECTED_PREDICTION_SHA256)
    outcomes = json.loads((raw / "OUTCOMES.json").read_text(encoding="utf-8"))
    update = [r for r in outcomes if r["partition"] == "update"]
    holdout = [r for r in outcomes if r["partition"] == "holdout"]
    scored = score_m29(update, holdout, leakage_ok=True)
    frozen = verdict["holdout"]
    check_text("verdict", str(scored["verdict"]), str(verdict["verdict"]))
    holdout_scored = scored["holdout"]
    for key in ("delta", "mae_b0", "mae_b1", "mae_m29"):
        check(key, float(holdout_scored[key]), float(frozen[key]))
    check("bootstrap low", float(holdout_scored["bootstrap"]["low"]), float(frozen["bootstrap"]["low"]))
    check("bootstrap high", float(holdout_scored["bootstrap"]["high"]), float(frozen["bootstrap"]["high"]))


def m30() -> None:
    print("M30 (frozen holdout, prompt-level paired bootstrap)")
    raw = ROOT / "reports" / "m30_raw"
    verdict = json.loads((raw / "VERDICT.json").read_text(encoding="utf-8"))
    outcomes = json.loads((raw / "OUTCOMES.json").read_text(encoding="utf-8"))
    for arm, frozen in verdict["arms"].items():
        rows = [r for r in outcomes if r["arm"] == arm and r["partition"] == "holdout"]
        mae_g = sum(abs(r["observed_effect"] - r["g"]) for r in rows) / len(rows)
        mae_gk = sum(abs(r["observed_effect"] - (r["g"] + r["kappa"])) for r in rows) / len(rows)
        by_prompt: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_prompt[r["prompt_id"]].append(
                abs(r["observed_effect"] - r["g"]) - abs(r["observed_effect"] - (r["g"] + r["kappa"]))
            )
        prompt_scores = [sum(v) / len(v) for _, v in sorted(by_prompt.items())]
        interval = paired_mean_ci(prompt_scores)
        print(f" arm {arm}: {len(rows)} rows, {len(prompt_scores)} prompts")
        check("MAE(g)", mae_g, float(frozen["mae_g"]))
        check("MAE(g+kappa)", mae_gk, float(frozen["mae_g_plus_kappa"]))
        check("relative reduction", 1 - mae_gk / mae_g, float(frozen["relative_reduction"]))
        check("bootstrap low", float(interval["low"]), float(frozen["bootstrap"]["low"]))
        check("bootstrap high", float(interval["high"]), float(frozen["bootstrap"]["high"]))
    print("  prediction sha256 recorded in verdict:", verdict["prediction_sha256"])
    print("  prediction file sha256:                ", sha256(raw / "PRE_OUTCOME_PREDICTIONS.json"))
    check_text("M30 prediction hash", sha256(raw / "PRE_OUTCOME_PREDICTIONS.json"), verdict["prediction_sha256"])


def main() -> int:
    m29d()
    m30()
    print()
    if failures:
        print("MISMATCHES:", ", ".join(failures))
        return 1
    print("All recomputed numbers match the frozen verdict files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
