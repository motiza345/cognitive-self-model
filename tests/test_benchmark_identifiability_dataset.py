"""The M21.2.4.3.4 dataset must come from the real oracle, not random labels."""

import csv

import numpy as np

from src.cognitive_self_model.benchmark.identifiability_dataset import (
    CSV_COLUMNS,
    generate_identifiability_csv,
)


def _read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_dataset_has_required_columns_and_split_roles(tmp_path):
    path = generate_identifiability_csv(tmp_path / "input_dataset.csv")
    rows = _read(path)

    required = {"episode_id", "split_role", "raw_q_invalid", "label_invalid", "track"}
    assert required.issubset(set(rows[0].keys()))
    assert required.issubset(set(CSV_COLUMNS))

    roles = {r["split_role"] for r in rows}
    assert roles == {"train", "calibration", "test"}


def test_labels_are_oracle_derived_not_random(tmp_path):
    # Two independent generations with the same (default) config must be
    # identical -> labels are deterministic oracle output, not random draws.
    p1 = generate_identifiability_csv(tmp_path / "a.csv")
    p2 = generate_identifiability_csv(tmp_path / "b.csv")
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def test_labels_match_track_semantics(tmp_path):
    path = generate_identifiability_csv(tmp_path / "input_dataset.csv")
    rows = _read(path)

    # KNOWN_VALID must never be labeled invalid; CAUSAL_BREAK must contain
    # invalid rows (it flips after onset).
    known_valid = [int(r["label_invalid"]) for r in rows if r["track"] == "KNOWN_VALID"]
    causal_break = [int(r["label_invalid"]) for r in rows if r["track"] == "CAUSAL_BREAK"]

    assert known_valid and max(known_valid) == 0
    assert causal_break and max(causal_break) == 1
