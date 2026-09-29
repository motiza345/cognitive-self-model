"""Compare a replay bundle to the recorded M22.1 reference.

Tiers and numeric formulas come from ``configs/m22_1_r_replay.yaml``.
Tolerance numbers are read from that document and are not redefined here.

``REPLAY_BLOCKED`` is not produced by this comparator. The harness assigns it
when a prerequisite is missing. A tolerance miss with unchanged identity is
``REPLAY_BEHAVIORAL_EQUIVALENT``, which is not a DEC-007 success tier.
``REPLAY_INVALID`` is reserved for a non-finite value or a broken cell key.
"""

from __future__ import annotations

import math
from typing import Any

from .config import ReplayConfig

RESULTS = "intervention_preflight_results.json"
REFERENCE_FILES = (
    RESULTS,
    "frozen_candidate.json",
    "certificate.json",
    "manifest.json",
    "intervention_not_validated.json",
)


def _dig(payload: dict[str, Any], dotted: str) -> tuple[bool, Any]:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _resolve(bundle: dict[str, Any], field: str) -> tuple[bool, Any]:
    if field.startswith("certificate."):
        return _dig(bundle["certificate.json"], field[len("certificate.") :])
    if field.startswith("frozen_candidate."):
        return _dig(bundle["frozen_candidate.json"], field[len("frozen_candidate.") :])
    if field == "discovery_means" or field.startswith("discovery_means"):
        return _dig(bundle[RESULTS], field)
    return _dig(bundle[RESULTS], field)


def _cell_blocks(config: ReplayConfig) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for item in config.document["blocks"]:
        filename, dotted = str(item).split(":", 1)
        blocks.append((filename, dotted))
    return blocks


def _rows(bundle: dict[str, Any], filename: str, dotted: str) -> tuple[bool, list[dict[str, Any]]]:
    present, value = _dig(bundle[filename], dotted)
    if not present or not isinstance(value, list):
        return False, []
    if not all(isinstance(row, dict) for row in value):
        return False, []
    return True, value


def _cell_key(block: str, row: dict[str, Any]) -> tuple[Any, ...] | None:
    required = ("prompt_id", "direction_id", "alpha", "layer")
    if any(key not in row for key in required):
        return None
    return (block, row["prompt_id"], row["direction_id"], float(row["alpha"]), int(row["layer"]))


def _index_cells(bundle: dict[str, Any], config: ReplayConfig) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[str]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    problems: list[str] = []
    for filename, dotted in _cell_blocks(config):
        block = f"{filename}:{dotted}"
        present, rows = _rows(bundle, filename, dotted)
        if not present:
            problems.append(f"missing cell block {block}")
            continue
        for row in rows:
            key = _cell_key(block, row)
            if key is None:
                problems.append(f"cell in {block} is missing a composite-key field")
                continue
            if key in index:
                problems.append(f"duplicate composite cell key in {block}")
                continue
            index[key] = row
    return index, problems


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _equal(left: Any, right: Any) -> bool:
    return left == right


def _score_within(new: float, ref: float, tolerances: dict[str, float]) -> bool:
    limit = tolerances["score_atol"] + tolerances["score_rtol"] * abs(ref)
    return abs(new - ref) <= limit


def _delta_within(new: float, ref: float, tolerances: dict[str, float]) -> bool:
    return abs(new - ref) <= tolerances["delta_atol"]


def _mean_within(new: float, ref: float, tolerances: dict[str, float]) -> bool:
    return abs(new - ref) <= tolerances["mean_atol"]


def _null_within(new: float, ref: float, tolerances: dict[str, float]) -> bool:
    return abs(new) <= tolerances["null_atol"] and abs(new - ref) <= tolerances["delta_atol"]


def _mean_entries(value: Any) -> tuple[bool, dict[str, Any]]:
    if isinstance(value, dict):
        return True, value
    return False, {}


def _dec007_success(config: ReplayConfig, tier: str) -> bool:
    names = (config.document.get("dec007_success_tiers") or {}).get("tiers") or []
    return tier in names


def compare_bundles(reference: dict[str, Any], replay: dict[str, Any], config: ReplayConfig) -> dict[str, Any]:
    """Return the yaml tier for two in-memory artifact bundles.

    Both bundles map the reference filenames to parsed JSON objects.
    """
    tolerances = config.tolerances
    contract = config.document["comparison_contract"]
    invalid: list[str] = []
    identity_mismatches: list[dict[str, Any]] = []
    numeric_outside: list[dict[str, Any]] = []
    numeric_unequal: list[str] = []

    for filename in (RESULTS, "frozen_candidate.json", "certificate.json", "manifest.json"):
        if filename not in reference or filename not in replay:
            invalid.append(f"missing artifact {filename}")

    ref_cells, ref_problems = _index_cells(reference, config)
    new_cells, new_problems = _index_cells(replay, config)
    invalid.extend(ref_problems)
    invalid.extend(new_problems)
    if not invalid and set(ref_cells) != set(new_cells):
        invalid.append("composite cell keys differ between reference and replay")

    cell_identity = [
        item["key"]
        for item in contract["identity"]
        if item.get("path") == "cell"
    ]
    scalar_identity = [
        item for item in contract["identity"] if item.get("path") not in {"cell"}
    ]

    if not invalid:
        for item in scalar_identity:
            present_ref, ref_value = _dig(reference[item["path"]], item["key"])
            present_new, new_value = _dig(replay[item["path"]], item["key"])
            if not present_ref or not present_new:
                invalid.append(f"missing identity field {item['path']} {item['key']}")
                continue
            if not _equal(ref_value, new_value):
                identity_mismatches.append(
                    {"path": item["path"], "key": item["key"], "reference": ref_value, "replay": new_value}
                )
        for key, ref_row in ref_cells.items():
            new_row = new_cells[key]
            for field in cell_identity:
                if field not in ref_row or field not in new_row:
                    invalid.append(f"missing cell identity field {field}")
                    continue
                if not _equal(ref_row[field], new_row[field]):
                    identity_mismatches.append(
                        {
                            "path": "cell",
                            "key": field,
                            "cell": list(key),
                            "reference": ref_row[field],
                            "replay": new_row[field],
                        }
                    )

    def note_numeric(label: str, ref_value: Any, new_value: Any, within) -> None:
        if not _finite(ref_value) or not _finite(new_value):
            invalid.append(f"non-finite numeric field {label}")
            return
        if not _equal(ref_value, new_value):
            numeric_unequal.append(label)
        if not within(float(new_value), float(ref_value), tolerances):
            numeric_outside.append(
                {"field": label, "reference": ref_value, "replay": new_value}
            )

    def compare_scalar(field: str, within) -> None:
        present_ref, ref_value = _resolve(reference, field)
        present_new, new_value = _resolve(replay, field)
        if not present_ref or not present_new:
            invalid.append(f"missing numeric field {field}")
            return
        if isinstance(ref_value, dict) or isinstance(new_value, dict):
            ref_ok, ref_map = _mean_entries(ref_value)
            new_ok, new_map = _mean_entries(new_value)
            if not ref_ok or not new_ok:
                invalid.append(f"numeric map is not an object: {field}")
                return
            if set(ref_map) != set(new_map):
                identity_mismatches.append({"path": field, "key": "keys", "reference": sorted(ref_map), "replay": sorted(new_map)})
                return
            for map_key in sorted(ref_map):
                note_numeric(f"{field}.{map_key}", ref_map[map_key], new_map[map_key], within)
            return
        note_numeric(field, ref_value, new_value, within)

    if not invalid:
        for field in contract["scores"]["fields"]:
            for key, ref_row in ref_cells.items():
                new_row = new_cells[key]
                note_numeric(f"cell.{key}.{field}", ref_row.get(field), new_row.get(field), _score_within)
        for field in contract["deltas"]["fields"]:
            for key, ref_row in ref_cells.items():
                new_row = new_cells[key]
                note_numeric(f"cell.{key}.{field}", ref_row.get(field), new_row.get(field), _delta_within)
        for field in contract["means"]["fields"]:
            compare_scalar(field, _mean_within)
        for field in contract["null_near_zero"]["fields"]:
            text = str(field)
            if "alpha" in text and "actual_delta" in text:
                for key, ref_row in ref_cells.items():
                    if float(key[3]) != 0.0:
                        continue
                    new_row = new_cells[key]
                    note_numeric(
                        f"cell.{key}.actual_delta alpha==0",
                        ref_row.get("actual_delta"),
                        new_row.get("actual_delta"),
                        _null_within,
                    )
                continue
            compare_scalar(text, _null_within)

    if invalid:
        tier = "REPLAY_INVALID"
    elif identity_mismatches:
        tier = "REPLAY_DIVERGENT"
    elif not numeric_unequal and not numeric_outside:
        tier = "REPLAY_EXACT"
    elif not numeric_outside:
        tier = "REPLAY_NUMERIC_EQUIVALENT"
    else:
        tier = "REPLAY_BEHAVIORAL_EQUIVALENT"

    return {
        "replay_tier": tier,
        "dec007_success": _dec007_success(config, tier),
        "scientific_failure": False,
        "failure_class": {
            "REPLAY_INVALID": "incomplete_or_nonfinite",
            "REPLAY_DIVERGENT": "identity",
            "REPLAY_EXACT": "none",
            "REPLAY_NUMERIC_EQUIVALENT": "none",
            "REPLAY_BEHAVIORAL_EQUIVALENT": "tolerance",
        }[tier],
        "invalid_reasons": invalid,
        "identity_mismatch_count": len(identity_mismatches),
        "numeric_outside_tolerance_count": len(numeric_outside),
        "numeric_unequal_count": len(numeric_unequal),
        "identity_mismatches": identity_mismatches[:20],
        "numeric_outside_tolerance": numeric_outside[:20],
    }
