"""Render the M23 scientific report from an already computed verdict."""

from __future__ import annotations

from typing import Any


LIMITED_PASS = (
    "A limited operational self-model loop was demonstrated on the tested "
    "Qwen mechanism/intervention setting."
)
GENERAL_FORBIDDEN = "We built a general Self-Model."


def render_report(payload: dict[str, Any]) -> str:
    verdict = str(payload["verdict"])
    questions = payload["questions"]
    lines = [
        "# M23 scientific report",
        "",
        f"Overall verdict: `{verdict}`",
        "",
        "This audit tested one frozen candidate intervention, `M22.1-D1-L23`.",
        "M22.1 had already left that intervention as `CANDIDATE`.",
        "M23 did not search for a new mechanism and did not change a threshold after scoring.",
        "",
        "## Question results",
        "",
    ]
    labels = {
        "Q1": "Prediction before intervention",
        "Q2": "Falsifiability",
        "Q3": "Explicit revision",
        "Q4": "Held-out improvement after update",
        "Q5": "Specificity against an outcome-only baseline",
        "Q6": "Transfer to a held-out magnitude",
    }
    for key, label in labels.items():
        lines.append(f"- {key} {label}: `{questions[key]}`")
    lines.extend(["", "## Metrics", ""])
    metrics = payload.get("metrics", {})
    q1 = metrics.get("Q1", {})
    if q1:
        lines.append(
            f"- Q1 MAE belief `{q1.get('mae_belief')}`, MAE predict-zero `{q1.get('mae_zero')}`."
        )
        interval = q1.get("mae_interval") or {}
        lines.append(
            f"- Q1 paired interval for (zero error − belief error): "
            f"mean `{interval.get('mean')}`, low `{interval.get('low')}`, "
            f"high `{interval.get('high')}`, class `{interval.get('class')}`."
        )
        lines.append(
            f"- Q1 sign agreements `{q1.get('sign_successes')}` / `{q1.get('sign_n')}`, "
            f"Clopper-Pearson low `{q1.get('sign_low')}`, high `{q1.get('sign_high')}`."
        )
        lines.append(
            f"- Critical epistemic cases: `{q1.get('n_critical')}`. "
            f"Descriptive 95% coverage: `{q1.get('uncertainty_coverage')}`."
        )
    for name in ("eval_context", "eval_magnitude"):
        block = metrics.get(name) or {}
        if not block:
            continue
        lines.append(f"- {name} MAE updated `{block.get('mae_updated')}`, "
                     f"no-update `{block.get('mae_no_update')}`, "
                     f"shuffled `{block.get('mae_shuffled')}`, "
                     f"outcome-only `{block.get('mae_outcome_only')}`.")
        for key in ("update_vs_no_update", "shuffled_vs_no_update", "specific_vs_outcome_only"):
            item = block.get(key) or {}
            lines.append(
                f"- {name} {key}: mean `{item.get('mean')}`, "
                f"low `{item.get('low')}`, high `{item.get('high')}`, class `{item.get('class')}`."
            )
    lines.extend(["", "## Failure class", ""])
    classes = payload.get("failure_class") or []
    if classes:
        for item in classes:
            lines.append(f"- `{item}`")
    else:
        lines.append("None assigned by the pre-registered map.")
    lines.extend(
        [
            "",
            "## What this does not establish",
            "",
            "The following are outside this result: a general self-model, a verified Qwen circuit,",
            "representation invariance, decision improvement, and closed-loop self-improvement.",
            "",
            "## Allowed conclusion",
            "",
        ]
    )
    if verdict == "PASS":
        lines.append(LIMITED_PASS)
    else:
        lines.append(
            "The limited operational loop was not demonstrated on this setting. "
            f"The pre-registered verdict is `{verdict}`."
        )
    lines.extend(["", "## Reproducibility", ""])
    repro = payload.get("reproducibility", {})
    for key in sorted(repro):
        lines.append(f"- {key}: `{repro[key]}`")
    lines.append("")
    text = "\n".join(lines)
    if GENERAL_FORBIDDEN in text:
        raise RuntimeError("report used a forbidden general claim")
    return text
