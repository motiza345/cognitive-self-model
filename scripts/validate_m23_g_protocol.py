"""Check the frozen M23-G protocol. Does not load Qwen or read new outcomes.

The synthetic positive control uses SelfModelBelief as it exists.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.belief import belief_from_observations, update_belief
from src.cognitive_self_model.m23.m22_reuse import frozen_prompts

CELLS = (
    {"intervention_id": "M22.1-D1-L0", "layer": 0, "direction_id": "D1", "alpha": 1.0},
    {"intervention_id": "M22.1-D2-L0", "layer": 0, "direction_id": "D2", "alpha": 1.0},
    {"intervention_id": "M22.1-D1-L8", "layer": 8, "direction_id": "D1", "alpha": 1.0},
    {"intervention_id": "M22.1-D2-L8", "layer": 8, "direction_id": "D2", "alpha": 1.0},
    {"intervention_id": "M22.1-D1-L15", "layer": 15, "direction_id": "D1", "alpha": 1.0},
    {"intervention_id": "M22.1-D2-L15", "layer": 15, "direction_id": "D2", "alpha": 1.0},
)

_COMPLETION = (
    "The largest planet in the solar system is",
    "A bicycle has two",
    "The author of Hamlet is",
    "Ice is the solid form of",
    "The Pacific is an",
    "A triangle has three",
    "The currency of Japan is the",
    "Bread is usually baked in an",
    "The tallest animal is the",
    "A year has twelve",
    "The chemical symbol for gold is",
    "Wool comes from a",
)
_SYNTAX = (
    "When the lights went out,",
    "Unless the door is locked,",
    "While the soup was cooling,",
    "As soon as the train stopped,",
    "Even though the map was torn,",
    "Whenever the bell rings,",
    "Since the library was closed,",
    "Provided that the ticket is valid,",
    "Once the paint had dried,",
    "Whether or not it snows,",
    "Until the music ended,",
    "Whereas the first attempt failed,",
)
_INSTRUCTION = (
    "Reply with one word. A metal used in coins:",
    "Reply with one word. The meal eaten at noon:",
    "Name a tool used for writing:",
    "Name a month of the year:",
    "Reply with one word. Opposite of early:",
    "Name a kind of tree:",
    "Reply with one word. A frozen dessert:",
    "Name a musical instrument:",
    "Reply with one word. The number of sides on a square:",
    "Name a continent:",
    "Reply with one word. A bird that cannot fly:",
    "Name a piece of furniture:",
)
_PARTITIONS = ("train", "validation", "evaluation")


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
                    "prompt_id": f"g-{family}-{index:02d}",
                    "family": family,
                    "text": text,
                    "partition": _PARTITIONS[(index - 1) % 3],
                }
            )
    return rows


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _fail(message: str) -> None:
    raise SystemExit(message)


def check_documents() -> None:
    prereg = _read("reports/M23_G_PREREGISTRATION.md")
    design = _read("reports/M23_G_DESIGN.md")
    register = _read("reports/M23_G_CONTAMINATION_REGISTER.md")
    synthetic = _read("reports/M23_G_SYNTHETIC_POSITIVE_CONTROL.md")
    for name, text in (
        ("preregistration", prereg),
        ("design", design),
        ("register", register),
        ("synthetic", synthetic),
    ):
        for banned in ("post-hoc threshold", "threshold tuned", "winning cell", "set after evaluation"):
            if banned in text.lower():
                _fail(f"{name} contains banned phrase: {banned}")
    required = (
        "Predictions for validation are written before validation outcomes exist.",
        "Predictions for evaluation are written before evaluation outcomes exist.",
        "CRITICAL_Z",
        "no-update",
        "constant-effect",
        "outcome-only",
        "shuffled-evidence",
        "paired_mean_ci",
        "predicted_effect",
        "predicted_direction",
        "uncertainty",
    )
    for phrase in required:
        if phrase not in prereg:
            _fail(f"preregistration missing {phrase}")
    for cell in CELLS:
        if cell["intervention_id"] not in prereg:
            _fail(f"preregistration missing {cell['intervention_id']}")
    for row in catalog():
        if row["prompt_id"] not in prereg or row["text"] not in prereg:
            _fail(f"preregistration missing prompt {row['prompt_id']}")
    if "READY_FOR_EXECUTION" not in design:
        _fail("design is missing the protocol status")
    if "does not count as evidence" not in synthetic.lower() and "does not count as evidence" not in synthetic:
        _fail("synthetic control must say it is not Qwen evidence")


def check_catalog() -> None:
    rows = catalog()
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        _fail("duplicate prompt ids")
    old_ids = {record.prompt_id for record in frozen_prompts()}
    old_texts = {record.text for record in frozen_prompts()}
    if set(ids) & old_ids:
        _fail("new catalog reuses an old prompt id")
    if {row["text"] for row in rows} & old_texts:
        _fail("new catalog reuses an old prompt text")
    by_partition: dict[str, list[str]] = {name: [] for name in _PARTITIONS}
    for row in rows:
        by_partition[row["partition"]].append(row["prompt_id"])
    if set(by_partition["evaluation"]) & set(by_partition["train"]):
        _fail("evaluation prompt appears in train")
    if set(by_partition["evaluation"]) & set(by_partition["validation"]):
        _fail("evaluation prompt appears in validation")
    if set(by_partition["train"]) & set(by_partition["validation"]):
        _fail("train prompt appears in validation")
    old_replication = {record.prompt_id for record in frozen_prompts() if record.role == "replication"}
    if set(by_partition["evaluation"]) & old_replication:
        _fail("old replication ids are in the new evaluation set")
    for partition, members in by_partition.items():
        if len(members) != 12:
            _fail(f"{partition} does not have 12 prompts")
        families = {prompt_id.split("-")[1] for prompt_id in members}
        if families != {"completion", "syntax", "instruction"}:
            _fail(f"{partition} is missing a context family")


def _historical_status(role: str) -> str:
    if role in {"discovery", "validation"}:
        return "CONTAMINATED"
    if role == "replication":
        return "PARTIALLY_CONTAMINATED"
    raise RuntimeError(f"unknown historical role {role}")


def expected_matrix() -> list[dict[str, str]]:
    rows = []
    for cell in CELLS:
        for prompt in catalog():
            rows.append(
                {
                    "catalog": "M23-G",
                    "intervention_id": cell["intervention_id"],
                    "layer": str(cell["layer"]),
                    "direction_id": cell["direction_id"],
                    "alpha": "1.0",
                    "prompt_id": prompt["prompt_id"],
                    "partition": prompt["partition"],
                    "contamination": "CLEAN",
                }
            )
        for prompt in frozen_prompts():
            rows.append(
                {
                    "catalog": "M22.1",
                    "intervention_id": cell["intervention_id"],
                    "layer": str(cell["layer"]),
                    "direction_id": cell["direction_id"],
                    "alpha": "1.0",
                    "prompt_id": prompt.prompt_id,
                    "partition": prompt.role,
                    "contamination": _historical_status(prompt.role),
                }
            )
    return rows


def check_matrix() -> None:
    path = ROOT / "reports" / "M23_G_CANDIDATE_MATRIX.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        found = list(csv.DictReader(handle))
    expected = expected_matrix()
    if found != expected:
        _fail("candidate matrix does not match the frozen catalog and contamination map")
    for row in found:
        if row["contamination"] == "CLEAN" and row["catalog"] != "M23-G":
            _fail(f"contaminated pair marked CLEAN: {row['intervention_id']} {row['prompt_id']}")
        if row["alpha"] != "1.0":
            _fail("alpha is not frozen at 1.0")
        if row["intervention_id"] not in {cell["intervention_id"] for cell in CELLS}:
            _fail("matrix contains an intervention outside the frozen family")
    register = _read("reports/M23_G_CONTAMINATION_REGISTER.md")
    for label in ("CLEAN", "PARTIALLY_CONTAMINATED", "CONTAMINATED"):
        if label not in register:
            _fail(f"contamination register missing {label}")


def check_no_new_outcomes() -> None:
    """New prompt ids must not already appear in inspected outcome files."""
    blobs = []
    for relative in (
        "reports/m23_raw",
        "reports/m23_d2_raw",
        "reports/m23_e_raw",
        "reports/m23_f_raw",
        "reports/M23_E_EPISODES.csv",
        "reports/M23_D2_EPISODES.csv",
        "reports/M23_F_SCREENING_EPISODES.csv",
        "reports/M23_EPISODE_RESULTS.csv",
    ):
        path = ROOT / relative
        if path.is_file():
            blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    blobs.append(child.read_text(encoding="utf-8", errors="ignore"))
    haystack = "\n".join(blobs)
    for row in catalog():
        if row["prompt_id"] in haystack or row["text"] in haystack:
            _fail(f"new prompt already appears in an outcome file: {row['prompt_id']}")


def synthetic_positive_control() -> dict[str, float | str | bool]:
    """Harness check only. The numbers are not Qwen measurements."""
    scope = {"status": "IN_SCOPE", "hook": "synthetic", "mechanism_id": "SYNTHETIC-SIGN-FLIP"}
    initial = belief_from_observations(
        "SYNTHETIC-SIGN-FLIP",
        "synthetic_additive",
        [1.0, 1.0],
        validity_scope=scope,
    )
    before = initial.predict(1.0)
    if before["predicted_effect"] != 1.0 or before["predicted_direction"] != 1:
        _fail("synthetic initial prediction is not +1")
    updated = update_belief(initial, -1.0, "synthetic-validation-01")
    if initial.predicted_effect != 1.0 or initial.version != 1:
        _fail("update mutated the pre-intervention belief")
    if updated.predicted_effect == -1.0:
        _fail("update copied the observed outcome")
    if abs(float(updated.predicted_effect) - (1.0 / 3.0)) > 1e-12:
        _fail("updated mean is not the cumulative mean")
    if updated.validity_scope["status"] != "CONTRADICTED":
        _fail("sign contradiction was not recorded")
    if updated.update_history[-1]["update_reason"] != "CONFIDENT_CONTRADICTION":
        _fail("critical sign contradiction was not classified")
    after = updated.predict(1.0)
    heldout = (-1.0, -1.0)
    no_update_errors = [abs(value - float(before["predicted_effect"])) for value in heldout]
    updated_errors = [abs(value - float(after["predicted_effect"])) for value in heldout]
    improvement = sum(no_update_errors) / 2.0 - sum(updated_errors) / 2.0
    if improvement <= 0.0:
        _fail("synthetic update did not reduce held-out absolute error")
    shuffled = update_belief(initial, 1.0, "shuffle-synthetic-01")
    shuffled_error = abs(-1.0 - float(shuffled.predict(1.0)["predicted_effect"]))
    if shuffled_error <= updated_errors[0]:
        _fail("shuffled agreeing evidence beat the contradicted update on the negative holdout")
    return {
        "initial_prediction": float(before["predicted_effect"]),
        "contradiction": True,
        "updated_prediction": float(after["predicted_effect"]),
        "copied_outcome": False,
        "heldout_improvement": float(improvement),
        "shuffled_heldout_abs_error": float(shuffled_error),
        "result": "HARNESS_PASS",
    }


def main() -> None:
    check_documents()
    check_catalog()
    check_matrix()
    check_no_new_outcomes()
    result = synthetic_positive_control()
    print("PROTOCOL_CHECK_PASS")
    print(result["result"])
    print(
        "synthetic_improvement",
        result["heldout_improvement"],
        "updated_prediction",
        result["updated_prediction"],
    )


if __name__ == "__main__":
    main()
