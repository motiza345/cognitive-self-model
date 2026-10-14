"""POST HOC / EXPLORATORY operand features for M34a-v2. Not in the prereg. Cannot change verdicts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m34a_v2_common import mann_whitney_auroc  # noqa: E402

RAW = ROOT / "reports" / "m34a_v2_raw" / "responses.jsonl"
OUT = ROOT / "reports" / "M34A_V2_POSTHOC.md"
L2 = 0.01
STEPS = 4000
LR = 0.05
LEVELS = (2, 3, 4, 5, 6, 7)


def schoolbook_mul_n1(a: int, b: int) -> dict[str, int]:
    """Schoolbook a (n-digit) * b (one digit), LSD first.

    Carry-out of a position is floor((digit*b + carry_in) / 10).
    n_carries = count of positions with nonzero carry-out.
    sum_carries = sum of those carry-out values (including zeros as 0).
    product_length = number of decimal digits of a*b.
    """
    if not (0 <= int(b) <= 9):
        raise ValueError(f"b must be one digit, got {b}")
    n_carries = 0
    sum_carries = 0
    carry = 0
    n = abs(int(a))
    # at least one digit
    while True:
        digit = n % 10
        total = digit * int(b) + carry
        carry = total // 10
        if carry != 0:
            n_carries += 1
        sum_carries += carry
        n //= 10
        if n == 0:
            break
    return {
        "n_carries": n_carries,
        "sum_carries": sum_carries,
        "product_length": len(str(abs(int(a) * int(b)))),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def attach_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        feat = schoolbook_mul_n1(int(r["a"]), int(r["b"]))
        item = dict(r)
        item.update(feat)
        out.append(item)
    return out


def mean_level_auroc(rows: list[dict[str, Any]], score_fn) -> dict[str, Any]:
    per: dict[str, float | None] = {}
    finite: list[float] = []
    by: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[int(r["level"])].append(r)
    for L in sorted(by):
        subset = by[L]
        val = mann_whitney_auroc(
            [float(score_fn(r)) for r in subset],
            [bool(r["correct"]) for r in subset],
        )
        per[str(L)] = val
        if val is not None:
            finite.append(val)
    return {"per_level": per, "mean": (sum(finite) / len(finite)) if finite else None}


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def apply_std(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def level_dummies(levels: list[int], universe: tuple[int, ...] = LEVELS) -> np.ndarray:
    idx = {L: i for i, L in enumerate(universe)}
    X = np.zeros((len(levels), len(universe)), dtype=float)
    for n, L in enumerate(levels):
        X[n, idx[int(L)]] = 1.0
    return X


def design(rows: list[dict[str, Any]], cols: tuple[str, ...]) -> np.ndarray:
    lev = level_dummies([int(r["level"]) for r in rows])
    if not cols:
        return lev
    extra = np.column_stack([[float(r[c]) for r in rows] for c in cols])
    return np.column_stack([lev, extra])


def fit_logit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(STEPS):
        z = X @ w + b
        z = np.clip(z, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        w = w - LR * (X.T @ err / n + 2.0 * L2 * w)
        b = b - LR * float(err.mean())
    return w, b


def predict_p(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    z = np.clip(X @ w + b, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def mean_level_metrics(rows: list[dict[str, Any]], p: np.ndarray) -> dict[str, float]:
    by: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[int(r["level"])].append(i)
    aurocs: list[float] = []
    lls: list[float] = []
    for L in sorted(by):
        idx = by[L]
        ys = [bool(rows[i]["correct"]) for i in idx]
        ps = [float(p[i]) for i in idx]
        au = mann_whitney_auroc(ps, ys)
        if au is not None:
            aurocs.append(au)
        yv = np.array([1.0 if y else 0.0 for y in ys], dtype=float)
        lls.append(logloss(yv, np.array(ps, dtype=float)))
    return {
        "auroc": (sum(aurocs) / len(aurocs)) if aurocs else float("nan"),
        "logloss": (sum(lls) / len(lls)) if lls else float("nan"),
        "n_levels_auroc": len(aurocs),
    }


def eval_spec(
    cal: list[dict[str, Any]],
    test: list[dict[str, Any]],
    cols: tuple[str, ...],
) -> dict[str, float]:
    Xc = design(cal, cols)
    Xt = design(test, cols)
    mu, sd = standardize_fit(Xc)
    Xc = apply_std(Xc, mu, sd)
    Xt = apply_std(Xt, mu, sd)
    y = np.array([1.0 if r["correct"] else 0.0 for r in cal], dtype=float)
    w, b = fit_logit(Xc, y)
    p = predict_p(Xt, w, b)
    return mean_level_metrics(test, p)


def render(doc: dict[str, Any]) -> str:
    au = doc["univariate"]
    lg = doc["logistic"]
    lines = [
        "# M34a-v2 post hoc operand features",
        "",
        "**POST HOC / EXPLORATORY.** Not in `docs/M34A_V2_PREREG.md`. Cannot change A or C.",
        "",
        "Schoolbook multiplication of n-digit `a` by one-digit `b`, least-significant digit first. "
        "Carry-out at a position is `floor((digit*b + carry_in)/10)`. "
        "`n_carries` = number of positions with nonzero carry-out; "
        "`sum_carries` = sum of carry-out values; "
        "`product_length` = number of decimal digits of `a*b`.",
        "",
        "## (a) Within-level AUROC for correctness (all pools pooled, then mean over levels)",
        "",
        "Scores: `answer_logprob` (higher better); `-n_carries`; `-sum_carries`; `-product_length`.",
        "",
        f"| score | mean AUROC |",
        f"| --- | ---: |",
        f"| `answer_logprob` | {au['logprob']['mean']:.6f} |",
        f"| `-n_carries` | {au['n_carries']['mean']:.6f} |",
        f"| `-sum_carries` | {au['sum_carries']['mean']:.6f} |",
        f"| `-product_length` | {au['product_length']['mean']:.6f} |",
        "",
        "Per-level `answer_logprob`:",
        "",
        "| level | AUROC |",
        "| ---: | ---: |",
    ]
    for L, v in au["logprob"]["per_level"].items():
        lines.append(f"| {L} | {v} |")
    lines += [
        "",
        "## (b) Logistic regression (CAL train / TEST eval)",
        "",
        "Level dummies + listed features, standardized on CAL, L2 `0.01` on weights "
        f"(gradient `2*L2*w`), `{STEPS}` steps, `lr={LR}`. "
        "Metrics: mean within-level AUROC and mean within-level log-loss on TEST.",
        "",
        "| spec | AUROC | log-loss |",
        "| --- | ---: | ---: |",
        f"| level only | {lg['level']['auroc']:.6f} | {lg['level']['logloss']:.6f} |",
        f"| + operand features | {lg['operand']['auroc']:.6f} | {lg['operand']['logloss']:.6f} |",
        f"| + answer_logprob | {lg['logprob']['auroc']:.6f} | {lg['logprob']['logloss']:.6f} |",
        f"| both | {lg['both']['auroc']:.6f} | {lg['both']['logloss']:.6f} |",
        "",
        "## Comparison to the independent reproduction (do not retune)",
        "",
        "| quantity | this run | reproduction | abs diff |",
        "| --- | ---: | ---: | ---: |",
    ]
    pairs = [
        ("AUROC logprob", au["logprob"]["mean"], 0.934),
        ("AUROC -n_carries", au["n_carries"]["mean"], 0.531),
        ("AUROC -sum_carries", au["sum_carries"]["mean"], 0.542),
        ("AUROC -product_length", au["product_length"]["mean"], 0.485),
        ("logit level AUROC", lg["level"]["auroc"], 0.500),
        ("logit level logloss", lg["level"]["logloss"], 0.513),
        ("logit +operand AUROC", lg["operand"]["auroc"], 0.659),
        ("logit +operand logloss", lg["operand"]["logloss"], 0.510),
        ("logit +logprob AUROC", lg["logprob"]["auroc"], 0.931),
        ("logit +logprob logloss", lg["logprob"]["logloss"], 0.359),
        ("logit both AUROC", lg["both"]["auroc"], 0.907),
        ("logit both logloss", lg["both"]["logloss"], 0.357),
    ]
    for name, got, exp in pairs:
        lines.append(f"| {name} | {got:.6f} | {exp:.6f} | {abs(got - exp):.6f} |")
    lines += [
        "",
        "## Claimed",
        "",
        "- These descriptive associations on the frozen v2 cache only.",
        "",
        "## NOT claimed",
        "",
        "- Any change to A or C; privileged access; other models or operations.",
        "",
        "## Scope",
        "",
        "- Same cache as `reports/M34A_V2_REPORT.md`: Qwen2.5-3B-Instruct, `mul_n1`, levels 2–7.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    rows = attach_features(load_rows(args.raw))
    uni = {
        "logprob": mean_level_auroc(rows, lambda r: float(r["answer_logprob"])),
        "n_carries": mean_level_auroc(rows, lambda r: -float(r["n_carries"])),
        "sum_carries": mean_level_auroc(rows, lambda r: -float(r["sum_carries"])),
        "product_length": mean_level_auroc(rows, lambda r: -float(r["product_length"])),
    }
    cal = [r for r in rows if r["pool"] == "cal"]
    test = [r for r in rows if r["pool"] == "test"]
    logistic = {
        "level": eval_spec(cal, test, ()),
        "operand": eval_spec(cal, test, ("n_carries", "sum_carries", "product_length")),
        "logprob": eval_spec(cal, test, ("answer_logprob",)),
        "both": eval_spec(
            cal, test, ("n_carries", "sum_carries", "product_length", "answer_logprob")
        ),
    }
    doc = {"univariate": uni, "logistic": logistic, "n": len(rows)}
    text = render(doc)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"univariate_means": {k: v["mean"] for k, v in uni.items()}, "logistic": logistic}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
