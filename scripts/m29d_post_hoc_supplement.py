"""Post-hoc description of the committed M29-D holdout rows.

Reads reports/m29d_raw/VERDICT.json and writes
reports/M29_D_POST_HOC_SUPPLEMENT.md. It does not read or write the
prediction artifact and it does not rewrite the verdict.

These summaries were not pre-registered. They cannot change the verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from scripts.m24_consumption_protocol import prompt_level_differences
from src.cognitive_self_model.m23.score import _agreement
from src.cognitive_self_model.m23.stats import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_mean_ci

ROOT = Path(__file__).resolve().parents[1]
VERDICT_PATH = ROOT / "reports" / "m29d_raw" / "VERDICT.json"
OUTPUT_PATH = ROOT / "reports" / "M29_D_POST_HOC_SUPPLEMENT.md"
CELL_ORDER = ("M22.1-D1-L0", "M22.1-D1-L8", "M22.1-D1-L15")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires two equal columns")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    scale_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    scale_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if scale_x == 0.0 or scale_y == 0.0:
        raise ValueError("correlation is undefined for a constant column")
    return numerator / (scale_x * scale_y)


def _fmt(value: float) -> str:
    return format(value, ".16g")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 36:
        raise ValueError(f"expected 36 holdout rows, got {len(rows)}")
    if any(row.get("partition") != "holdout" for row in rows):
        raise ValueError("supplement accepts holdout paired rows only")
    if {row["intervention_id"] for row in rows} != set(CELL_ORDER):
        raise ValueError("holdout cells are not the frozen M29 family")

    annotated = []
    for row in rows:
        residual = float(row["residual"])
        kappa = float(row["kappa"])
        copied = dict(row)
        copied["paired_vs_g"] = abs(residual) - abs(residual - kappa)
        annotated.append(copied)

    mae_g = sum(abs(float(row["residual"])) for row in annotated) / len(annotated)
    mae_gk = sum(abs(float(row["residual"]) - float(row["kappa"])) for row in annotated) / len(annotated)
    delta_rows = mae_g - mae_gk
    prompt_scores = prompt_level_differences(annotated, "paired_vs_g", expected_prompts=12)
    interval = paired_mean_ci(prompt_scores)
    if abs(float(interval["mean"]) - delta_rows) > 1e-12:
        raise RuntimeError("prompt-level mean and row-level Delta diverged")

    agreements = [_agreement(float(row["kappa"]), float(row["residual"])) for row in annotated]
    if any(item is None for item in agreements):
        raise ValueError("a holdout row has a zero kappa or residual; sign count is not 36")
    sign_matches = sum(1 for item in agreements if item)

    cells = []
    for cell in CELL_ORDER:
        group = [row for row in annotated if row["intervention_id"] == cell]
        if len(group) != 12:
            raise ValueError(f"{cell} does not have 12 holdout rows")
        cell_mae_g = sum(abs(float(row["residual"])) for row in group) / len(group)
        cell_mae_gk = sum(abs(float(row["residual"]) - float(row["kappa"])) for row in group) / len(group)
        med_kappa = float(median(abs(float(row["kappa"])) for row in group))
        med_residual = float(median(abs(float(row["residual"])) for row in group))
        if med_residual == 0.0:
            raise ValueError(f"{cell} has a zero median absolute residual")
        cells.append(
            {
                "intervention_id": cell,
                "mae_g": cell_mae_g,
                "mae_g_plus_kappa": cell_mae_gk,
                "delta_vs_g": cell_mae_g - cell_mae_gk,
                "median_abs_kappa": med_kappa,
                "median_abs_residual": med_residual,
                "median_ratio": med_kappa / med_residual,
            }
        )

    return {
        "rows": len(annotated),
        "mae_g": mae_g,
        "mae_g_plus_kappa": mae_gk,
        "delta_vs_g": delta_rows,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "mean": float(interval["mean"]),
            "low": float(interval["low"]),
            "high": float(interval["high"]),
            "class": str(interval["class"]),
            "unit": "prompt",
            "prompts": 12,
        },
        "pearson_kappa_residual": _pearson(
            [float(row["kappa"]) for row in annotated],
            [float(row["residual"]) for row in annotated],
        ),
        "sign_matches": sign_matches,
        "sign_rows": len(agreements),
        "cells": cells,
    }


def render(summary: dict[str, Any]) -> str:
    boot = summary["bootstrap"]
    cell_lines = [
        "| cell | MAE(g) | MAE(g+kappa) | median(|kappa|)/median(|r|) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cell in summary["cells"]:
        cell_lines.append(
            "| `{intervention_id}` | {mae_g} | {mae_gk} | {ratio} |".format(
                intervention_id=cell["intervention_id"],
                mae_g=_fmt(cell["mae_g"]),
                mae_gk=_fmt(cell["mae_g_plus_kappa"]),
                ratio=_fmt(cell["median_ratio"]),
            )
        )
    return "\n".join(
        [
            "# M29-D post-hoc supplement",
            "",
            "**Status:** POST_HOC",
            "**Source:** `reports/m29d_raw/VERDICT.json` holdout `paired_rows` only",
            "**Script:** `scripts/m29d_post_hoc_supplement.py`",
            "",
            "These analyses were not pre-registered. They cannot change the verdict in `reports/m29d_raw/VERDICT.json`. The preregistered primary comparison remains MAE(B1) minus MAE(M29) on the holdout rows, with the row-level paired bootstrap already stored in that file.",
            "",
            "Nothing here refits kappa, changes its sign or scale, or selects a new baseline.",
            "",
            "## Holdout description",
            "",
            f"- Rows: {summary['rows']}",
            f"- MAE of g alone, mean(|y - g|): `{_fmt(summary['mae_g'])}`",
            f"- MAE of g + kappa, mean(|y - (g + kappa)|): `{_fmt(summary['mae_g_plus_kappa'])}`",
            f"- Delta versus g, MAE(g) - MAE(g + kappa): `{_fmt(summary['delta_vs_g'])}`",
            f"- Prompt-level paired bootstrap: {boot['draws']} draws, seed {boot['seed']}, {boot['prompts']} prompts, three cells averaged within each prompt",
            f"- Prompt-level mean of paired differences: `{_fmt(boot['mean'])}`",
            f"- Prompt-level 95% interval: `[{_fmt(boot['low'])}, {_fmt(boot['high'])}]`",
            f"- Prompt-level interval class: `{boot['class']}`",
            f"- Pearson correlation of kappa and residual: `{_fmt(summary['pearson_kappa_residual'])}`",
            f"- Sign agreement, sign(kappa) against sign(residual): {summary['sign_matches']}/{summary['sign_rows']}",
            "",
            "The prompt-level mean equals the row-level Delta because every prompt has the same three cells. The interval is still the prompt-level bootstrap, not the row-level interval stored in the verdict.",
            "",
            "## Per cell",
            "",
            *cell_lines,
            "",
            "The ratio is median(|kappa|) divided by median(|residual|), not the median of the per-row ratios.",
            "",
            "## What this file does not do",
            "",
            "It does not replace `SUPPORTED`. It does not authorize a different primary baseline. A positive description against g was not an acceptance criterion when the holdout was scored.",
            "",
        ]
    )


def main() -> None:
    payload = json.loads(VERDICT_PATH.read_text(encoding="utf-8"))
    summary = summarize(payload["holdout"]["paired_rows"])
    OUTPUT_PATH.write_text(render(summary), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
