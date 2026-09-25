"""
Real M21.2.4.3.4 identifiability dataset generator.

Produces ``input_dataset.csv`` with the columns the M21.2.4.3.4 identifiability
benchmark requires (``episode_id, split_role, raw_q_invalid, label_invalid,
track``) from the **real benchmark oracle**, replacing the synthetic
``binomial``-labeled fallback that was previously used in Colab.

Column provenance (all from the frozen-reference benchmark):
    - ``raw_q_invalid``  = raw estimator probability ``q_invalid`` (uncalibrated)
    - ``label_invalid``  = ``self_model_invalid`` (oracle contextual invalidity)
    - ``track``          = benchmark track name
    - ``episode_id``     = stable per-episode id
    - ``split_role``     = train / calibration / test (episode-disjoint)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .pipeline import build_scored_dataset

CSV_COLUMNS = [
    "episode_id",
    "split_role",
    "raw_q_invalid",
    "label_invalid",
    "self_model_invalid",
    "track",
]


def _rows_for_split(split: Dict[str, np.ndarray], split_role: str) -> List[Dict[str, Any]]:
    rows = []
    n = len(split["y"])
    for i in range(n):
        label = int(split["y"][i])
        rows.append(
            {
                "episode_id": str(split["episode_id"][i]),
                "split_role": split_role,
                "raw_q_invalid": float(split["raw_probability"][i]),
                "label_invalid": label,
                "self_model_invalid": label,
                "track": str(split["track"][i]),
            }
        )
    return rows


def generate_identifiability_csv(
    output_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the real M21.2.4.3.4 ``input_dataset.csv`` and return its path."""
    scored = build_scored_dataset(config)

    rows: List[Dict[str, Any]] = []
    rows += _rows_for_split(scored["train"], "train")
    rows += _rows_for_split(scored["calibration"], "calibration")
    rows += _rows_for_split(scored["test"], "test")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path
