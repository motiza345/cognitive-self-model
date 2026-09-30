"""Extract and index historical Colab notebooks. Does not execute them.

Milestone ids are read from code, markdown, and cell outputs. Notebook metadata
ids and Colab authorship tags are not treated as milestones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "colab_history"
NOTEBOOKS = ARCHIVE / "notebooks"
RAW = ARCHIVE / "raw"
MANIFESTS = ARCHIVE / "manifests"
PLACEMENT = MANIFESTS / "placement.json"
INVENTORY = MANIFESTS / "notebook_inventory.json"
HASHES = MANIFESTS / "notebook_hashes.sha256"
INDEX = ROOT / "docs" / "HISTORICAL_EXPERIMENT_INDEX.md"
MILESTONE_MAP = ROOT / "docs" / "HISTORICAL_MILESTONE_MAP.md"

DECLARED_ZIPS = {
    "archive_01": "Colab Notebooks-20260930T183832Z-1-001.zip",
    "archive_02": "Colab Notebooks-20260930T183712Z-1-001 (1).zip",
}

MILESTONE_RE = re.compile(r"(?<![A-Za-z0-9_])M\d+(?:\.\d+)*[A-Za-z]?\b")
MILESTONE_WORD_RE = re.compile(
    r"\bMILESTONE\s+(\d+(?:\.\d+)*[A-Za-z]?)\b",
    re.IGNORECASE,
)
MODEL_RE = re.compile(
    r"Qwen/Qwen[A-Za-z0-9._\-]+|Qwen2\.5[A-Za-z0-9._\-]*|\bgpt2\b|\bgpt-2\b|GPT-2",
    re.IGNORECASE,
)
REVISION_RE = re.compile(
    r"revision[\"'\s:=]{1,12}([0-9a-f]{40})",
    re.IGNORECASE,
)
SEED_RE = re.compile(r"\bseed\s*[=:]\s*(\d+)\b", re.IGNORECASE)
LAYER_RE = re.compile(
    r"(?:n_layers|layers)\s*=\s*(?:\[[^\]]{0,80}\]|\d+)",
    re.IGNORECASE,
)
HEAD_RE = re.compile(
    r"(?:n_heads|heads)\s*=\s*(?:\[[^\]]{0,80}\]|\d+)",
    re.IGNORECASE,
)
OBJECTIVE_RE = re.compile(
    r"^(?:objective|question|goal)\s*[:：]\s*(\S.{0,240})$",
    re.IGNORECASE | re.MULTILINE,
)
STATUS_RE = re.compile(
    r"\b(PASS|FAIL|SUCCESS|INCONCLUSIVE|STOP|GO|REDEFINE)\b"
)
METRIC_NAMES = (
    "MAE",
    "MSE",
    "accuracy",
    "correlation",
    "sign accuracy",
    "z-score",
    "coverage",
)
METHOD_NAMES = (
    "ablation",
    "patching",
    "intervention",
    "counterfactual",
    "blind discovery",
    "self-model",
    "self model",
    "causal graph",
    "epistemic graph",
    "adaptive probing",
)
REGIME_NAMES = ("completion", "instruction", "syntax")
LIBRARY_RE = re.compile(
    r"\b(torch|transformers|transformer_lens|transformer-lens|numpy)\b[^\n]{0,60}\d+\.\d+(?:\.\d+)?"
)
JACCARD_MIN = 0.92
MIN_CODE_LINES = 5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def cell_corpus(notebook: dict[str, Any]) -> dict[str, str]:
    code: list[str] = []
    markdown: list[str] = []
    outputs: list[str] = []
    output_cells = 0
    for cell in notebook.get("cells", []):
        kind = cell.get("cell_type")
        source = _as_text(cell.get("source"))
        if kind == "code":
            code.append(source)
        elif kind == "markdown":
            markdown.append(source)
        cell_outputs = cell.get("outputs") or []
        if cell_outputs:
            output_cells += 1
        for output in cell_outputs:
            if output.get("output_type") == "stream":
                outputs.append(_as_text(output.get("text")))
            outputs.append(_as_text(output.get("ename")))
            outputs.append(_as_text(output.get("evalue")))
            for line in output.get("traceback") or []:
                outputs.append(_as_text(line))
            data = output.get("data") or {}
            if isinstance(data, dict):
                for key in ("text/plain", "text/markdown"):
                    if key in data:
                        outputs.append(_as_text(data[key]))
    return {
        "code": "\n".join(code),
        "markdown": "\n".join(markdown),
        "outputs": "\n".join(outputs),
        "output_cell_count": str(output_cells),
        "code_cell_count": str(sum(1 for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")),
        "markdown_cell_count": str(
            sum(1 for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown")
        ),
        "cell_count": str(len(notebook.get("cells", []))),
    }


def unique(items: list[str], limit: int = 40) -> list[str]:
    seen: list[str] = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= limit:
            break
    return seen


def milestones_in(text: str) -> list[str]:
    found = list(MILESTONE_RE.findall(text))
    found.extend(f"M{item}" for item in MILESTONE_WORD_RE.findall(text))
    return unique(found, limit=80)


def status_lines(text: str) -> list[str]:
    rows = []
    for line in text.splitlines():
        if STATUS_RE.search(line):
            compact = " ".join(line.split())
            if compact:
                rows.append(compact[:240])
    return unique(rows, limit=20)


def metric_lines(text: str) -> list[dict[str, str]]:
    found = []
    lower = text.lower()
    present = [name for name in METRIC_NAMES if name.lower() in lower]
    for name in present:
        for line in text.splitlines():
            if name.lower() in line.lower() and re.search(r"\d", line):
                found.append(
                    {
                        "metric": name,
                        "line": " ".join(line.split())[:240],
                        "label": "historical_reported",
                    }
                )
                break
    return found[:20]


def methods_in(text: str) -> list[str]:
    lower = text.lower()
    return [name for name in METHOD_NAMES if name in lower]


def regimes_in(text: str) -> list[str]:
    found = []
    for chunk in text.split("\n\n"):
        if "regime" not in chunk.lower():
            continue
        for name in REGIME_NAMES:
            if name in chunk.lower() and name not in found:
                found.append(name)
    return found


def objective_of(text: str) -> str:
    match = OBJECTIVE_RE.search(text)
    if not match:
        return "NOT_DETERMINED"
    return match.group(1).strip()


def code_line_set(code: str) -> set[str]:
    lines = set()
    for line in code.splitlines():
        compact = " ".join(line.split())
        if len(compact) >= 15 and not compact.startswith("#"):
            lines.add(compact)
    return lines


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def milestone_sort_key(value: str) -> tuple:
    if value == "UNKNOWN":
        return ((10**6, ""),)
    parts = []
    for part in value[1:].split("."):
        match = re.fullmatch(r"(\d+)([A-Za-z]?)", part)
        if not match:
            return ((10**6, value),)
        parts.append((int(match.group(1)), match.group(2).lower()))
    return tuple(parts)


def load_notebook(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "cells" not in payload:
        return None
    return payload


def extract_archives(zip_args: list[str]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[tuple[str, bytes]]] = {}
    sources = []
    for item in zip_args:
        label, raw_path = item.split("=", 1)
        path = Path(raw_path)
        data = path.read_bytes()
        sources.append(
            {
                "archive_source": label,
                "declared_name": DECLARED_ZIPS.get(label, path.name),
                "read_from": str(path),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
        with zipfile.ZipFile(path) as handle:
            members = []
            for info in handle.infolist():
                if info.is_dir():
                    continue
                payload = handle.read(info)
                target = RAW / label / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                members.append((Path(info.filename).name, payload))
            grouped[label] = members
    names: dict[str, int] = {}
    for members in grouped.values():
        for name, _payload in members:
            names[name] = names.get(name, 0) + 1
    placement = []
    for label, members in grouped.items():
        for name, payload in members:
            if names[name] > 1:
                suffix = label.replace("archive_", "")
                archived = f"{Path(name).stem}__archive{suffix}{Path(name).suffix}"
                if not Path(name).suffix:
                    archived = f"{name}__archive{suffix}"
            else:
                archived = name
            destination = NOTEBOOKS / archived
            if destination.exists():
                raise SystemExit(f"refusing to overwrite {destination}")
            destination.write_bytes(payload)
            placement.append(
                {
                    "archived_name": archived,
                    "archive_source": label,
                    "original_filename": name,
                    "raw_path": str((RAW / label / "Colab Notebooks" / name).relative_to(ARCHIVE)),
                    "source_zip_sha256": next(row["sha256"] for row in sources if row["archive_source"] == label),
                    "declared_zip_name": next(
                        row["declared_name"] for row in sources if row["archive_source"] == label
                    ),
                }
            )
    PLACEMENT.write_text(
        json.dumps({"sources": sources, "notebooks": placement}, indent=2) + "\n",
        encoding="utf-8",
    )


def build_records() -> list[dict[str, Any]]:
    placement = json.loads(PLACEMENT.read_text(encoding="utf-8"))
    rows = []
    for spec in sorted(
        placement["notebooks"],
        key=lambda item: (item["archive_source"], item["original_filename"], item["archived_name"]),
    ):
        path = NOTEBOOKS / spec["archived_name"]
        digest = sha256_file(path)
        notebook = load_notebook(path)
        record: dict[str, Any] = {
            "archive_experiment_id": "",
            "archive_source": spec["archive_source"],
            "original_filename": spec["original_filename"],
            "archived_name": spec["archived_name"],
            "declared_zip_name": spec["declared_zip_name"],
            "sha256": digest,
            "scientific_interpretation": "UNKNOWN",
            "objective": "NOT_DETERMINED",
            "milestone_ids": [],
            "experiment_ids": [],
            "model_names": [],
            "model_revisions": [],
            "library_versions": [],
            "seeds": [],
            "datasets": [],
            "regimes": [],
            "layers": [],
            "heads": [],
            "metrics": [],
            "decisions": [],
            "status_keywords": [],
            "methods": [],
            "duplicate_relations": [],
            "parse_status": "OK",
        }
        if notebook is None:
            record["parse_status"] = "NOT_A_NOTEBOOK"
            record["cell_count"] = 0
            record["code_cell_count"] = 0
            record["markdown_cell_count"] = 0
            record["output_cell_count"] = 0
            rows.append(record)
            continue
        corpus = cell_corpus(notebook)
        combined = "\n".join((corpus["code"], corpus["markdown"], corpus["outputs"]))
        content_milestones = milestones_in(combined)
        record.update(
            {
                "cell_count": int(corpus["cell_count"]),
                "code_cell_count": int(corpus["code_cell_count"]),
                "markdown_cell_count": int(corpus["markdown_cell_count"]),
                "output_cell_count": int(corpus["output_cell_count"]),
                "milestone_ids": content_milestones,
                "objective": objective_of(corpus["markdown"] + "\n" + corpus["code"]),
                "model_names": unique(MODEL_RE.findall(combined)),
                "model_revisions": unique(REVISION_RE.findall(combined)),
                "library_versions": unique(
                    [match.group(0)[:120] for match in LIBRARY_RE.finditer(combined)]
                ),
                "seeds": unique(SEED_RE.findall(corpus["code"])),
                "datasets": unique(
                    re.findall(r"\bdataset(?:_name|_id)?\s*=\s*[\"']([^\"']+)[\"']", corpus["code"], re.IGNORECASE)
                ),
                "regimes": regimes_in(combined),
                "layers": unique(LAYER_RE.findall(corpus["code"])),
                "heads": unique(HEAD_RE.findall(corpus["code"])),
                "metrics": metric_lines(corpus["outputs"] + "\n" + corpus["markdown"]),
                "status_keywords": unique(STATUS_RE.findall(corpus["outputs"] + "\n" + corpus["markdown"])),
                "decisions": [
                    {"status": status, "label": "historical_reported", "line": line}
                    for line in status_lines(corpus["outputs"] + "\n" + corpus["markdown"])
                    for status in STATUS_RE.findall(line)[:1]
                ],
                "methods": methods_in(combined),
                "_code": corpus["code"],
                "_source_norm": sha256_bytes(" ".join((corpus["code"] + "\n" + corpus["markdown"]).split()).encode()),
            }
        )
        rows.append(record)
    for index, record in enumerate(rows, start=1):
        record["archive_experiment_id"] = f"HIST-{index:04d}"
    attach_duplicates(rows)
    for record in rows:
        record.pop("_code", None)
        record.pop("_source_norm", None)
    return rows


def attach_duplicates(rows: list[dict[str, Any]]) -> None:
    by_hash: dict[str, list[str]] = {}
    by_source: dict[str, list[str]] = {}
    code_sets: dict[str, set[str]] = {}
    for record in rows:
        by_hash.setdefault(record["sha256"], []).append(record["archive_experiment_id"])
        if "_source_norm" in record:
            by_source.setdefault(record["_source_norm"], []).append(record["archive_experiment_id"])
            code_sets[record["archive_experiment_id"]] = code_line_set(record.get("_code", ""))
    ids = [record["archive_experiment_id"] for record in rows]
    relations: dict[str, list[dict[str, str]]] = {item: [] for item in ids}
    for group in by_hash.values():
        if len(group) < 2:
            continue
        for left in group:
            for right in group:
                if left != right:
                    relations[left].append({"kind": "EXACT_DUPLICATE", "with": right})
    for group in by_source.values():
        if len(group) < 2:
            continue
        exact_pairs = {
            (row["with"], item)
            for item, rels in relations.items()
            for row in rels
            if row["kind"] == "EXACT_DUPLICATE"
        }
        for left in group:
            for right in group:
                if left < right and (right, left) not in exact_pairs and (left, right) not in exact_pairs:
                    relations[left].append({"kind": "NEAR_DUPLICATE", "with": right, "reason": "same_cell_source"})
                    relations[right].append({"kind": "NEAR_DUPLICATE", "with": left, "reason": "same_cell_source"})
    id_list = list(ids)
    for index, left in enumerate(id_list):
        for right in id_list[index + 1 :]:
            if any(row["with"] == right for row in relations[left]):
                continue
            left_lines = code_sets.get(left, set())
            right_lines = code_sets.get(right, set())
            if len(left_lines) < MIN_CODE_LINES or len(right_lines) < MIN_CODE_LINES:
                continue
            score = jaccard(left_lines, right_lines)
            if score >= JACCARD_MIN:
                relations[left].append(
                    {"kind": "NEAR_DUPLICATE", "with": right, "reason": f"code_jaccard={score:.3f}"}
                )
                relations[right].append(
                    {"kind": "NEAR_DUPLICATE", "with": left, "reason": f"code_jaccard={score:.3f}"}
                )
    by_id = {record["archive_experiment_id"]: record for record in rows}
    for record in rows:
        record["duplicate_relations"] = relations[record["archive_experiment_id"]]
        record["duplicate_class"] = classify_duplicate(record["duplicate_relations"])
        record["similar_to"] = [
            row["with"] for row in record["duplicate_relations"] if row["kind"] == "NEAR_DUPLICATE"
        ]
        if record["archive_experiment_id"] not in by_id:
            continue


def classify_duplicate(relations: list[dict[str, str]]) -> str:
    kinds = {row["kind"] for row in relations}
    if "EXACT_DUPLICATE" in kinds and "NEAR_DUPLICATE" in kinds:
        return "EXACT_DUPLICATE"
    if "EXACT_DUPLICATE" in kinds:
        return "EXACT_DUPLICATE"
    if "NEAR_DUPLICATE" in kinds:
        return "NEAR_DUPLICATE"
    return "DISTINCT"


def write_outputs(records: list[dict[str, Any]]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY.write_text(json.dumps({"notebooks": records}, indent=2) + "\n", encoding="utf-8")
    lines = []
    for record in records:
        lines.append(f"{record['sha256']}  notebooks/{record['archived_name']}")
    HASHES.write_text("\n".join(lines) + "\n", encoding="utf-8")
    INDEX.write_text(render_index(records), encoding="utf-8")
    MILESTONE_MAP.write_text(render_map(records), encoding="utf-8")


def reported_result(record: dict[str, Any]) -> str:
    if record["status_keywords"]:
        return "HISTORICAL_REPORTED: " + ", ".join(record["status_keywords"])
    return "NOT_DETERMINED"


def render_index(records: list[dict[str, Any]]) -> str:
    blocks = [
        "# Historical experiment index",
        "",
        "Archive index of imported Colab notebooks. Reported statuses are historical.",
        "They are not scientific validations of the current repository.",
        "Where an objective or result is not explicit in the notebook, the field stays `NOT_DETERMINED`.",
        "`scientific_interpretation` stays `UNKNOWN`.",
        "",
    ]
    for record in records:
        milestones = ", ".join(f"`{item}`" for item in record["milestone_ids"]) or "`UNKNOWN`"
        models = ", ".join(f"`{item}`" for item in record["model_names"]) or "`NOT_DETERMINED`"
        config_bits = []
        if record["seeds"]:
            config_bits.append("seeds " + ", ".join(record["seeds"][:12]))
        if record["layers"]:
            config_bits.append("layers " + "; ".join(record["layers"][:6]))
        if record["heads"]:
            config_bits.append("heads " + "; ".join(record["heads"][:6]))
        if record["regimes"]:
            config_bits.append("regimes " + ", ".join(record["regimes"]))
        configuration = "; ".join(config_bits) if config_bits else "NOT_DETERMINED"
        metric_text = "NOT_DETERMINED"
        if record["metrics"]:
            metric_text = "; ".join(
                f"{item['metric']}: {item['line']}" for item in record["metrics"][:6]
            )
        notes = [f"parse `{record['parse_status']}`", f"duplicate `{record['duplicate_class']}`"]
        if record["similar_to"]:
            notes.append("similar_to " + ", ".join(record["similar_to"]))
        exact = [row["with"] for row in record["duplicate_relations"] if row["kind"] == "EXACT_DUPLICATE"]
        if exact:
            notes.append("exact_duplicate_of " + ", ".join(exact))
        if record["methods"]:
            notes.append("methods " + ", ".join(record["methods"]))
        blocks.extend(
            [
                f"## {record['archive_experiment_id']}",
                "",
                f"- Notebook: `{record['archived_name']}`",
                f"- Original filename: `{record['original_filename']}`",
                f"- Archive source: `{record['archive_source']}` (`{record['declared_zip_name']}`)",
                f"- Detected milestone: {milestones}",
                f"- Objective: {record['objective']}",
                f"- Model: {models}",
                f"- Important configuration: {configuration}",
                f"- Reported metrics: {metric_text}",
                f"- Reported result: {reported_result(record)}",
                f"- Historical decision: {reported_result(record)}",
                f"- SHA256: `{record['sha256']}`",
                f"- Notes: {'. '.join(notes)}",
                "",
            ]
        )
    return "\n".join(blocks)


def render_map(records: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        keys = record["milestone_ids"] or ["UNKNOWN"]
        for key in keys:
            grouped.setdefault(key, []).append(record)
    lines = [
        "# Historical milestone map",
        "",
        "Milestones below are `M<id>` tokens, including an optional trailing letter such as `M5.1.5b`, or the explicit phrase `MILESTONE <id>`, found in notebook code, markdown, or cell outputs.",
        "A notebook with no such string is `UNKNOWN`. Filename text alone is not a milestone.",
        "Outcomes are `HISTORICAL_REPORTED` when a status word is present, otherwise `NOT_DETERMINED`.",
        "",
        "Coverage is only what these two archives contain. Absence from this list does not mean a milestone does not exist elsewhere in the project.",
        "",
    ]
    for milestone in sorted(grouped, key=milestone_sort_key):
        members = grouped[milestone]
        notebooks = ", ".join(f"`{item['archived_name']}`" for item in members)
        experiments = ", ".join(item["archive_experiment_id"] for item in members)
        models = sorted({model for item in members for model in item["model_names"]})
        model_text = ", ".join(f"`{item}`" for item in models) if models else "`NOT_DETERMINED`"
        outcomes = []
        for item in members:
            outcomes.append(f"{item['archive_experiment_id']} {reported_result(item)}")
        lines.extend(
            [
                f"## {milestone}",
                "",
                f"- Notebooks: {notebooks}",
                f"- Experiments: {experiments}",
                f"- Models: {model_text}",
                "- Reported outcomes:",
            ]
        )
        for outcome in outcomes:
            lines.append(f"  - {outcome}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="append", default=[], help="archive_01=/path/to.zip")
    args = parser.parse_args()
    if args.zip:
        extract_archives(args.zip)
    if not PLACEMENT.exists():
        raise SystemExit("placement.json is missing; pass --zip to extract first")
    records = build_records()
    write_outputs(records)
    counts = {"EXACT_DUPLICATE": 0, "NEAR_DUPLICATE": 0, "DISTINCT": 0}
    for record in records:
        counts[record["duplicate_class"]] = counts.get(record["duplicate_class"], 0) + 1
    print(f"notebooks={len(records)}")
    print(
        "exact_files="
        + str(counts["EXACT_DUPLICATE"])
        + " near_files="
        + str(counts["NEAR_DUPLICATE"])
        + " distinct_files="
        + str(counts["DISTINCT"])
    )


if __name__ == "__main__":
    main()
