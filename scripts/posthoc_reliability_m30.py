"""POST HOC / EXPLORATORY. Not part of any pre-registration; cannot change any verdict.

Question: does the magnitude of the pre-outcome curvature |kappa| predict the error that
remains after the correction g + kappa on the M30 rows? Spearman correlation between a
feature and the remaining absolute error, plus a risk-coverage table.

Needs numpy only. Reads reports/m30_raw/OUTCOMES.json. Writes a markdown file.
Usage: python -m scripts.posthoc_reliability_m30 [--out reports/M30_POSTHOC_RELIABILITY.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ROOT / "reports" / "m30_raw" / "OUTCOMES.json"
SEED = 23001
DRAWS = 5000


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    return ranks


def spearman(a, b) -> float:
    return float(np.corrcoef(_rank(np.asarray(a)), _rank(np.asarray(b)))[0, 1])


def bootstrap_spearman(a, b) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    values = []
    for _ in range(DRAWS):
        i = rng.integers(0, n, n)
        values.append(spearman(a[i], b[i]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "M30_POSTHOC_RELIABILITY.md")
    args = parser.parse_args(argv)
    rows_all = json.loads(OUTCOMES.read_text(encoding="utf-8"))
    lines = [
        "# M30 post hoc: does |kappa| predict the remaining error?",
        "",
        "**POST HOC / EXPLORATORY.** Not pre-registered. Cannot change any verdict. A finite-difference probe",
        "(S2) computes the same curvature, so this gives no privileged access.",
        "",
        f"Holdout rows only. Spearman rho between the feature and `abs_error_g_plus_kappa`; 95% percentile CI,",
        f"{DRAWS} bootstrap draws over rows, seed {SEED}.",
        "",
        "| arm | feature | rows | rho | CI low | CI high |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    risk_lines = [
        "",
        "## Risk-coverage",
        "",
        "Mean remaining absolute error among the kept fraction of holdout rows with the smallest feature value.",
        "",
        "| arm | feature | 25% | 50% | 75% | 100% |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("new_identity", "anchor"):
        rows = [r for r in rows_all if r["arm"] == arm and r["partition"] == "holdout"]
        error = np.array([r["abs_error_g_plus_kappa"] for r in rows])
        features = {
            "abs(kappa)": np.array([abs(r["kappa"]) for r in rows]),
            "abs(g)": np.array([abs(r["g"]) for r in rows]),
        }
        for name, values in features.items():
            rho = spearman(values, error)
            low, high = bootstrap_spearman(values, error)
            lines.append(f"| {arm} | {name} | {len(rows)} | {rho:.3f} | {low:.3f} | {high:.3f} |")
            order = np.argsort(values, kind="stable")
            sorted_error = error[order]
            cuts = [float(sorted_error[: int(c * len(rows))].mean()) for c in (0.25, 0.5, 0.75, 1.0)]
            risk_lines.append(f"| {arm} | {name} | " + " | ".join(f"{c:.6f}" for c in cuts) + " |")
    text = "\n".join(lines + risk_lines) + "\n"
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
