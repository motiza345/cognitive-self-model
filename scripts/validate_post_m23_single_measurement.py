"""Check the frozen single-measurement protocol. Does not load Qwen.

The synthetic positive control uses the baseline/candidate formulas in this file.
It is a harness check. It is not a Qwen measurement.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_m23_g_protocol import catalog as g_catalog
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts
from src.cognitive_self_model.m23.stats import paired_mean_ci

TESTED_CELL = "M22.1-D1-L15"
TESTED_LAYER = 15
TESTED_DIRECTION = "D1"
TESTED_ALPHA = "1.0"
TESTED_MEASUREMENT = "pre_dot"
PARTITIONS = ("train", "validation", "evaluation")

_COMPLETION = (
    "The largest moon of Earth is",
    "A spider has eight",
    "The composer of the Ninth Symphony is",
    "Steam is the gas form of",
    "The Atlantic is an",
    "A square has four",
    "The currency of the United Kingdom is the",
    "Pottery is usually fired in a",
    "The fastest land animal is the",
    "A day has twenty-four",
    "The chemical symbol for iron is",
    "Silk comes from a",
)
_SYNTAX = (
    "After the kettle boiled,",
    "If the window stays open,",
    "Before the guests arrived,",
    "Although the road was icy,",
    "Because the ink had dried,",
    "Unless the key is turned,",
    "While the choir was singing,",
    "As soon as the curtain rose,",
    "Even though the cup was cracked,",
    "Whenever the clock strikes,",
    "Since the market was empty,",
    "Until the rain stopped,",
)
_INSTRUCTION = (
    "Reply with one word. A fabric made from flax:",
    "Reply with one word. The meal eaten in the evening:",
    "Name a tool used for cutting paper:",
    "Name a day that starts the weekend:",
    "Reply with one word. Opposite of heavy:",
    "Name a kind of flower:",
    "Reply with one word. A frozen drink:",
    "Name a string instrument:",
    "Reply with one word. The number of sides on a hexagon:",
    "Name an ocean:",
    "Reply with one word. A mammal that lays eggs:",
    "Name a piece of clothing:",
)


def catalog() -> list[dict[str, str]]:
    rows = []
    for family, texts in (
        ("completion", _COMPLETION),
        ("syntax", _SYNTAX),
        ("instruction", _INSTRUCTION),
    ):
        for index, text in enumerate(texts, start=1):
            rows.append(
                {
                    "prompt_id": f"s-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": PARTITIONS[(index - 1) % 3],
                }
            )
    return rows


def fit_candidate(pre_dots: list[float], effects: list[float]) -> dict[str, float]:
    """Train-only fit. The slope is zero when the measurement does not vary."""
    if len(pre_dots) != len(effects) or len(pre_dots) < 2:
        raise ValueError("fit requires at least two paired rows")
    count = float(len(pre_dots))
    mean = sum(effects) / count
    center = sum(pre_dots) / count
    variance = sum((value - center) ** 2 for value in pre_dots)
    if variance == 0.0:
        slope = 0.0
    else:
        slope = sum((x - center) * (y - mean) for x, y in zip(pre_dots, effects)) / variance
    return {"mean": float(mean), "pre_dot_mean": float(center), "slope": float(slope)}


def predict_baseline(fit: dict[str, float], pre_dot: float) -> float:
    del pre_dot
    return float(fit["mean"])


def predict_candidate(fit: dict[str, float], pre_dot: float) -> float:
    return float(fit["mean"]) + float(fit["slope"]) * (float(pre_dot) - float(fit["pre_dot_mean"]))


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _fail(message: str) -> None:
    raise SystemExit(message)


def check_documents() -> None:
    prereg = _read("reports/POST_M23_SINGLE_MEASUREMENT_PREREGISTRATION.md")
    design = _read("reports/POST_M23_SINGLE_MEASUREMENT_DESIGN.md")
    register = _read("reports/POST_M23_SINGLE_MEASUREMENT_CONTAMINATION.md")
    synthetic = _read("reports/POST_M23_SINGLE_MEASUREMENT_SYNTHETIC_CONTROL.md")
    combined = "\n".join((prereg, design, register, synthetic))
    if "mechanism representation" in combined.lower():
        _fail("documents call the measurement a mechanism representation")
    for banned in ("threshold tuned", "winning cell", "set after evaluation", "feature ranking"):
        if banned in combined.lower():
            _fail(f"documents contain banned phrase: {banned}")
    required = (
        "tested cell: M22.1-D1-L15",
        "tested measurement: pre_dot",
        "tested alpha: 1.0",
        "Exactly one intervention cell is tested.",
        "Exactly one pre-outcome measurement is tested.",
        "No feature selection is performed.",
        "No threshold is chosen after outcomes exist.",
        "Predictions for validation are written before validation outcomes exist.",
        "Predictions for evaluation are written before evaluation outcomes exist.",
        "The measurement forward does not read the intervention effect.",
        "error(baseline) - error(candidate)",
        "paired_mean_ci",
        "PREDICTIVE_INFORMATION_SUPPORTED",
        "MEASUREMENT_HURTS",
        "INCONCLUSIVE",
        "blocks.15.hook_resid_post",
    )
    for phrase in required:
        if phrase not in prereg:
            _fail(f"preregistration missing {phrase}")
    if prereg.count("tested measurement:") != 1:
        _fail("preregistration must declare exactly one tested measurement")
    if prereg.count("tested cell:") != 1:
        _fail("preregistration must declare exactly one tested cell")
    if prereg.count("tested alpha:") != 1:
        _fail("preregistration must declare exactly one tested alpha")
    for row in catalog():
        if row["prompt_id"] not in prereg or row["text"] not in prereg:
            _fail(f"preregistration missing prompt {row['prompt_id']}")
    if "READY_FOR_EXECUTION" not in design:
        _fail("design is missing the protocol status")
    if "not qwen evidence" not in synthetic.lower():
        _fail("synthetic control must say it is not Qwen evidence")
    for label in ("CLEAN", "PARTIALLY_CONTAMINATED", "CONTAMINATED"):
        if label not in register:
            _fail(f"contamination register missing {label}")


def check_catalog() -> None:
    rows = catalog()
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        _fail("duplicate prompt ids")
    if len(rows) != 36:
        _fail("catalog is not the frozen 36 prompts")
    old_ids = {record.prompt_id for record in frozen_prompts()}
    old_texts = {record.text for record in frozen_prompts()}
    g_ids = {row["prompt_id"] for row in g_catalog()}
    g_texts = {row["text"] for row in g_catalog()}
    new_ids = set(ids)
    new_texts = {row["text"] for row in rows}
    if new_ids & old_ids or new_ids & g_ids:
        _fail("new catalog reuses a previous prompt id")
    if new_texts & old_texts or new_texts & g_texts:
        _fail("new catalog reuses a previous prompt text")
    by_partition: dict[str, list[dict[str, str]]] = {name: [] for name in PARTITIONS}
    for row in rows:
        by_partition[row["partition"]].append(row)
    for left, right in (("train", "validation"), ("train", "evaluation"), ("validation", "evaluation")):
        left_ids = {row["prompt_id"] for row in by_partition[left]}
        right_ids = {row["prompt_id"] for row in by_partition[right]}
        if left_ids & right_ids:
            _fail(f"{left} and {right} share a prompt")
    for partition, members in by_partition.items():
        if len(members) != 12:
            _fail(f"{partition} does not have 12 prompts")
        families = {row["family"] for row in members}
        if families != {"completion", "syntax", "instruction"}:
            _fail(f"{partition} is missing a surface family")
        if len(members) != len({row["family"] + row["prompt_id"] for row in members}):
            _fail(f"{partition} has a duplicated row")


def _old_status(role: str) -> str:
    if role in {"discovery", "validation"}:
        return "CONTAMINATED"
    if role == "replication":
        return "PARTIALLY_CONTAMINATED"
    raise RuntimeError(f"unknown historical role {role}")


def expected_matrix() -> list[dict[str, str]]:
    rows = []
    for prompt in catalog():
        rows.append(
            {
                "catalog": "POST_M23_S",
                "intervention_id": TESTED_CELL,
                "layer": str(TESTED_LAYER),
                "direction_id": TESTED_DIRECTION,
                "alpha": TESTED_ALPHA,
                "measurement": TESTED_MEASUREMENT,
                "prompt_id": prompt["prompt_id"],
                "partition": prompt["partition"],
                "contamination": "CLEAN",
            }
        )
    for prompt in g_catalog():
        rows.append(
            {
                "catalog": "M23-G",
                "intervention_id": TESTED_CELL,
                "layer": str(TESTED_LAYER),
                "direction_id": TESTED_DIRECTION,
                "alpha": TESTED_ALPHA,
                "measurement": TESTED_MEASUREMENT,
                "prompt_id": prompt["prompt_id"],
                "partition": prompt["partition"],
                "contamination": "CONTAMINATED",
            }
        )
    for prompt in frozen_prompts():
        rows.append(
            {
                "catalog": "M22.1",
                "intervention_id": TESTED_CELL,
                "layer": str(TESTED_LAYER),
                "direction_id": TESTED_DIRECTION,
                "alpha": TESTED_ALPHA,
                "measurement": TESTED_MEASUREMENT,
                "prompt_id": prompt.prompt_id,
                "partition": prompt.role,
                "contamination": _old_status(prompt.role),
            }
        )
    return rows


def check_matrix() -> None:
    path = ROOT / "reports" / "POST_M23_SINGLE_MEASUREMENT_MATRIX.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        found = list(csv.DictReader(handle))
    expected = expected_matrix()
    if found != expected:
        _fail("matrix does not match the frozen catalog and contamination map")
    cells = {row["intervention_id"] for row in found}
    measurements = {row["measurement"] for row in found}
    alphas = {row["alpha"] for row in found}
    if cells != {TESTED_CELL}:
        _fail("matrix does not contain exactly one intervention cell")
    if measurements != {TESTED_MEASUREMENT}:
        _fail("matrix does not contain exactly one measurement")
    if alphas != {TESTED_ALPHA}:
        _fail("alpha is not frozen at 1.0")
    clean = [row for row in found if row["contamination"] == "CLEAN"]
    if {row["catalog"] for row in clean} != {"POST_M23_S"}:
        _fail("a previously used catalog is marked CLEAN")
    if {row["prompt_id"] for row in clean} != {row["prompt_id"] for row in catalog()}:
        _fail("CLEAN rows are not exactly the new catalog")


def check_no_previous_outcomes() -> None:
    blobs = []
    for relative in (
        "reports/m23_raw",
        "reports/m23_d2_raw",
        "reports/m23_e_raw",
        "reports/m23_f_raw",
        "reports/m23_g_raw",
        "reports/M23_E_EPISODES.csv",
        "reports/M23_D2_EPISODES.csv",
        "reports/M23_F_SCREENING_EPISODES.csv",
        "reports/M23_EPISODE_RESULTS.csv",
        "reports/M23_G_EPISODES.csv",
        "reports/M23_DIAGNOSTIC_EPISODES.csv",
    ):
        path = ROOT / relative
        if path.is_file():
            blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    blobs.append(child.read_text(encoding="utf-8", errors="ignore"))
    haystack = "\n".join(blobs)
    for row in catalog():
        if row["prompt_id"] in haystack or row["text"] in haystack:
            _fail(f"new prompt already appears in an outcome file: {row['prompt_id']}")


def synthetic_positive_control() -> dict[str, float | str | bool]:
    """Harness check only. These numbers are not Qwen measurements."""
    fit = fit_candidate([0.0, 1.0, 2.0, 3.0], [0.0, 0.2, 0.4, 0.6])
    if abs(fit["mean"] - 0.3) > 1e-12 or abs(fit["pre_dot_mean"] - 1.5) > 1e-12:
        _fail("synthetic train summary is not the declared mean")
    if abs(fit["slope"] - 0.2) > 1e-12:
        _fail("synthetic slope is not the planted slope")
    if abs(predict_candidate(fit, fit["pre_dot_mean"]) - fit["mean"]) > 1e-12:
        _fail("candidate at the train measurement mean is not the scalar mean")
    held_x = [4.0, 5.0, 6.0, 7.0]
    held_y = [0.8, 1.0, 1.2, 1.4]
    differences = []
    for pre_dot, effect in zip(held_x, held_y):
        baseline = predict_baseline(fit, pre_dot)
        candidate = predict_candidate(fit, pre_dot)
        if abs(candidate - effect) > 1e-12:
            _fail("synthetic candidate missed the planted held-out effect")
        differences.append(abs(effect - baseline) - abs(effect - candidate))
    interval = paired_mean_ci(differences)
    if interval["class"] != "CI_POSITIVE":
        _fail("synthetic predictive measurement was not detected")
    if abs(float(interval["mean"]) - 0.8) > 1e-12:
        _fail("synthetic paired mean is not 0.8")
    flat = fit_candidate([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0])
    if flat["slope"] != 0.0:
        _fail("zero measurement variance did not force a zero slope")
    if predict_candidate(flat, 1.0) != predict_baseline(flat, 1.0):
        _fail("degenerate candidate diverged from the scalar mean")
    return {
        "mean": fit["mean"],
        "pre_dot_mean": fit["pre_dot_mean"],
        "slope": fit["slope"],
        "paired_mean": float(interval["mean"]),
        "low": float(interval["low"]),
        "high": float(interval["high"]),
        "class": str(interval["class"]),
        "result": "HARNESS_PASS",
    }


def main() -> None:
    check_documents()
    check_catalog()
    check_matrix()
    check_no_previous_outcomes()
    result = synthetic_positive_control()
    print("PROTOCOL_CHECK_PASS")
    print(result["result"])
    print(
        "synthetic_paired_mean",
        result["paired_mean"],
        "low",
        result["low"],
        "high",
        result["high"],
        "class",
        result["class"],
    )


if __name__ == "__main__":
    main()
