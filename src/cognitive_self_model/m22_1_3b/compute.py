"""M22.1.3b Step B: stored-field first-order structure (float64, no forwards)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DIRECTION_IDS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
EVAL_SPLITS = ("validation", "replication")
ALL_SPLITS = ("discovery", "validation", "replication")
EPS = 1.0e-6
N = 896
ALPHA = 1.0
SOURCED_ENERGY = {
    "validation": 0.999382,
    "replication": 0.999234,
    "discovery": 0.999294,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def leading_energy(matrix: np.ndarray) -> float:
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    total = float(np.sum(array ** 2))
    if total <= 1e-18:
        return float("nan")
    _, singular, _ = np.linalg.svd(array, full_matrices=False)
    return float((singular[0] ** 2) / float(np.sum(singular ** 2)))


def leading_singular_vectors(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(matrix, dtype=np.float64)
    left, _, right = np.linalg.svd(array, full_matrices=False)
    return left[:, 0].astype(np.float64), right[0].astype(np.float64)


def abs_cosine(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).ravel()
    right = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return float("nan")
    return float(abs(float(np.dot(left, right))) / denom)


def inv_rms_from_residual(x: np.ndarray, eps: float = EPS) -> float:
    residual = np.asarray(x, dtype=np.float64).ravel()
    scale = float(np.sqrt(float(np.mean(residual ** 2)) + float(eps)))
    return 1.0 / scale


def least_squares_scale(target: np.ndarray, predictor: np.ndarray) -> tuple[float, float]:
    y = np.asarray(target, dtype=np.float64).ravel()
    x = np.asarray(predictor, dtype=np.float64).ravel()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return float("nan"), float("nan")
    coef = float(np.dot(x, y) / denom)
    residual = y - coef * x
    y_norm = float(np.linalg.norm(y))
    rel = float("nan") if y_norm <= 0.0 else float(np.linalg.norm(residual) / y_norm)
    return coef, rel


def degenerate(t2: np.ndarray, t1: np.ndarray, ratio: float = 1.0e-12) -> bool:
    t2_norm = float(np.linalg.norm(np.asarray(t2, dtype=np.float64)))
    t1_norm = float(np.linalg.norm(np.asarray(t1, dtype=np.float64)))
    return t2_norm <= float(ratio) * t1_norm


def _cell_map(cells: list[dict]) -> dict[tuple[str, str, float], dict]:
    out = {}
    for row in cells:
        key = (row["prompt_id"], row["direction_id"], float(row["alpha"]))
        out[key] = row
    return out


def _matrix_from_cells(
    prompt_ids: list[str],
    cells: dict[tuple[str, str, float], dict],
    field: str,
) -> np.ndarray:
    matrix = np.zeros((len(prompt_ids), len(DIRECTION_IDS)), dtype=np.float64)
    for i, prompt_id in enumerate(prompt_ids):
        for j, direction_id in enumerate(DIRECTION_IDS):
            matrix[i, j] = float(cells[(prompt_id, direction_id, ALPHA)][field])
    return matrix


def _t1_matrix(
    prompt_ids: list[str],
    cells: dict[tuple[str, str, float], dict],
) -> np.ndarray:
    matrix = np.zeros((len(prompt_ids), len(DIRECTION_IDS)), dtype=np.float64)
    for i, prompt_id in enumerate(prompt_ids):
        for j, direction_id in enumerate(DIRECTION_IDS):
            row = cells[(prompt_id, direction_id, ALPHA)]
            matrix[i, j] = float(row["alpha"]) * float(row["inv_rms"]) * float(row["r_dot_d"])
    return matrix


def run_audit(repo_root: Path | None = None) -> dict:
    root = Path(repo_root) if repo_root is not None else ROOT
    config_path = root / "configs" / "m22_1_3b_first_order_structure.yaml"
    predictions = json.loads((root / "artifacts" / "m22_1_3" / "predictions.json").read_text(encoding="utf-8"))
    recorded = json.loads((root / "reports" / "m22_1_2_response_space_results.json").read_text(encoding="utf-8"))
    residuals = np.load(root / "artifacts" / "m22_1_3" / "clean_resid_l23_last.npy").astype(np.float64)
    directions = np.load(root / "artifacts" / "m22_1_3" / "directions_float64.npy").astype(np.float64)
    prompt_ids_global = list(predictions["prompt_ids"])
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids_global)}
    cells = _cell_map(predictions["cells"])
    clean_margins = predictions["clean_margins"]

    gates: dict[str, dict] = {}
    first_failure = None

    def fail(name: str, payload: dict) -> dict:
        nonlocal first_failure
        payload["pass"] = False
        gates[name] = payload
        if first_failure is None:
            first_failure = name
        return payload

    def succeed(name: str, payload: dict) -> dict:
        payload["pass"] = True
        gates[name] = payload
        return payload

    # G0
    g0_rows = []
    g0_ok = True
    for prompt_id, index in prompt_index.items():
        stored = float(cells[(prompt_id, "D1", ALPHA)]["inv_rms"])
        recomputed = inv_rms_from_residual(residuals[index], EPS)
        rel = abs(recomputed - stored) / max(abs(stored), 1e-18)
        g0_rows.append({"prompt_id": prompt_id, "rel": rel})
        if rel > 1.0e-6:
            g0_ok = False
    if g0_ok:
        succeed("G0", {"max_rel": max(row["rel"] for row in g0_rows), "n": len(g0_rows)})
    else:
        fail("G0", {"max_rel": max(row["rel"] for row in g0_rows), "n": len(g0_rows)})

    split_payload: dict[str, dict] = {}
    if first_failure is None:
        for split in ALL_SPLITS:
            stored = recorded["matrices"][split]
            prompt_ids = list(stored["prompt_ids"])
            recorded_delta = np.asarray(stored["alpha_plus1"], dtype=np.float64)
            delta1 = _matrix_from_cells(prompt_ids, cells, "delta_jacobian")
            t1 = _t1_matrix(prompt_ids, cells)
            t2 = t1 - delta1
            energy_delta = leading_energy(recorded_delta)
            energy_delta1 = leading_energy(delta1)
            energy_t1 = leading_energy(t1)
            prompt_rows = []
            n_degenerate = 0
            for i, prompt_id in enumerate(prompt_ids):
                x = residuals[prompt_index[prompt_id]]
                x_dot_d = directions @ x
                t1_row = t1[i]
                t2_row = t2[i]
                is_deg = degenerate(t2_row, t1_row)
                inv = float(cells[(prompt_id, "D1", ALPHA)]["inv_rms"])
                row = {
                    "prompt_id": prompt_id,
                    "degenerate": is_deg,
                    "c_p": None,
                    "g3_rel": None,
                    "r_dot_x_implied": None,
                    "s_stored": float(clean_margins[prompt_id]["from_clean_logits"]),
                    "s_implied": None,
                    "g4_abs_diff": None,
                }
                if is_deg:
                    n_degenerate += 1
                    prompt_rows.append(row)
                    continue
                coef, rel = least_squares_scale(t2_row, x_dot_d)
                r_dot_x = coef * N / (ALPHA * inv ** 3)
                s_implied = inv * r_dot_x
                s_stored = row["s_stored"]
                g4_diff = abs(s_stored - s_implied)
                row.update(
                    {
                        "c_p": coef,
                        "g3_rel": rel,
                        "r_dot_x_implied": r_dot_x,
                        "s_implied": s_implied,
                        "g4_abs_diff": g4_diff,
                        "g4_tol": 1.0e-3 * max(1.0, abs(s_stored)),
                    }
                )
                prompt_rows.append(row)
            left_d, right_d = leading_singular_vectors(recorded_delta)
            left_t, right_t = leading_singular_vectors(t1)
            split_payload[split] = {
                "prompt_ids": prompt_ids,
                "energy_recorded_Delta": energy_delta,
                "energy_Delta1": energy_delta1,
                "energy_T1": energy_t1,
                "projection_share": float(np.sum(t2 ** 2) / np.sum(delta1 ** 2)),
                "abs_cos_left": abs_cosine(left_d, left_t),
                "abs_cos_right": abs_cosine(right_d, right_t),
                "n_degenerate": n_degenerate,
                "prompts": prompt_rows,
                "energy_abs_diff_vs_sourced": abs(energy_delta - SOURCED_ENERGY[split]),
                "energy_abs_diff_Delta1_vs_Delta": abs(energy_delta1 - energy_delta),
            }

        g1_fail = next(
            (
                {
                    "split": split,
                    "energy": split_payload[split]["energy_recorded_Delta"],
                    "sourced": SOURCED_ENERGY[split],
                    "abs_diff": split_payload[split]["energy_abs_diff_vs_sourced"],
                }
                for split in ALL_SPLITS
                if split_payload[split]["energy_abs_diff_vs_sourced"] > 1.0e-6
            ),
            None,
        )
        if g1_fail:
            fail("G1", g1_fail)
        else:
            succeed("G1", {"sourced": SOURCED_ENERGY, "abs_tol": 1.0e-6})

        if first_failure is None:
            g2_fail = next(
                (
                    {"split": split, "energy_T1": split_payload[split]["energy_T1"]}
                    for split in ALL_SPLITS
                    if abs(split_payload[split]["energy_T1"] - 1.0) > 1.0e-9
                ),
                None,
            )
            if g2_fail:
                fail("G2", g2_fail)
            else:
                succeed("G2", {"abs_tol": 1.0e-9})

        if first_failure is None:
            g3_fail = next(
                (
                    {"split": split, "prompt_id": row["prompt_id"], "g3_rel": row["g3_rel"]}
                    for split in EVAL_SPLITS
                    for row in split_payload[split]["prompts"]
                    if not row["degenerate"] and row["g3_rel"] > 1.0e-6
                ),
                None,
            )
            if g3_fail:
                fail("G3", g3_fail)
            else:
                succeed("G3", {"rel_tol": 1.0e-6, "required_splits": list(EVAL_SPLITS)})

        if first_failure is None:
            g4_fail = next(
                (
                    {
                        "split": split,
                        "prompt_id": row["prompt_id"],
                        "abs_diff": row["g4_abs_diff"],
                        "tol": row["g4_tol"],
                    }
                    for split in EVAL_SPLITS
                    for row in split_payload[split]["prompts"]
                    if not row["degenerate"] and row["g4_abs_diff"] > row["g4_tol"]
                ),
                None,
            )
            if g4_fail:
                fail("G4", g4_fail)
            else:
                succeed("G4", {"b_U": "ABSENT_OR_ZERO", "required_splits": list(EVAL_SPLITS)})

    label = None
    status: str
    if first_failure is not None:
        status = f"GATE_FAILED_{first_failure}"
    else:
        def split_ok_supported(name: str) -> bool:
            block = split_payload[name]
            if block["n_degenerate"] > 1:
                return False
            return (
                abs(block["energy_Delta1"] - block["energy_recorded_Delta"]) <= 0.002
                and block["abs_cos_left"] >= 0.99
                and block["abs_cos_right"] >= 0.99
                and block["projection_share"] <= 0.01
            )

        def split_ok_projection(name: str) -> bool:
            block = split_payload[name]
            if block["n_degenerate"] > 1:
                return False
            return (
                abs(block["energy_Delta1"] - block["energy_recorded_Delta"]) <= 0.002
                and block["abs_cos_left"] >= 0.99
                and block["abs_cos_right"] >= 0.99
                and block["projection_share"] > 0.01
            )

        if any(split_payload[name]["n_degenerate"] > 1 for name in EVAL_SPLITS):
            label = "INSUFFICIENT_EVIDENCE"
        elif all(split_ok_supported(name) for name in EVAL_SPLITS):
            label = "REINTERPRETATION_DESCRIPTIVELY_SUPPORTED"
        elif all(split_ok_projection(name) for name in EVAL_SPLITS):
            label = "PROJECTION_TERM_NON_NEGLIGIBLE"
        else:
            label = "INSUFFICIENT_EVIDENCE"
        status = f"COMPLETED_{label}"

    return {
        "milestone": "M22.1.3b",
        "config_path": "configs/m22_1_3b_first_order_structure.yaml",
        "config_sha256": sha256_file(config_path),
        "status": status,
        "label": label,
        "first_failure": first_failure,
        "gates": gates,
        "splits": split_payload,
        "m22_1_status_preserved": "CANDIDATE",
        "m22_2_authorized": False,
        "finding_status_rule": "F-M22.1.2-REINTERPRETATION changes only via a separate DECISION",
        "compute_dtype": "float64",
        "center_writing_weights": "DISABLED",
        "b_U": "ABSENT_OR_ZERO",
        "amendment": "A2",
    }


def write_outputs(payload: dict, repo_root: Path | None = None) -> dict[str, str]:
    root = Path(repo_root) if repo_root is not None else ROOT
    artifact_dir = root / "artifacts" / "m22_1_3b"
    report_dir = root / "reports" / "M22_1_3b"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    results_path = artifact_dir / "results.json"
    report_path = report_dir / "README.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(_render_report(payload), encoding="utf-8")
    listed = [
        ("configs/m22_1_3b_first_order_structure.yaml", sha256_file(root / "configs" / "m22_1_3b_first_order_structure.yaml")),
        ("artifacts/m22_1_3b/results.json", sha256_file(results_path)),
        ("reports/M22_1_3b/README.md", sha256_file(report_path)),
    ]
    manifest = artifact_dir / "manifest.sha256"
    manifest.write_text("".join(f"{digest}  {path}\n" for path, digest in listed), encoding="utf-8")
    return {
        "results": sha256_file(results_path),
        "report": sha256_file(report_path),
        "manifest": sha256_file(manifest),
        "config": listed[0][1],
    }


def _render_report(payload: dict) -> str:
    lines = [
        "# M22.1.3b first-order structure",
        "",
        f"Status: `{payload['status']}`",
        f"Label: `{payload['label']}`",
        f"First failure: `{payload['first_failure']}`",
        "",
        "M22.1 remains `CANDIDATE`. M22.2 remains `NOT AUTHORIZED`.",
        "Status of `F-M22.1.2-REINTERPRETATION` is unchanged (separate DECISION required).",
        "",
        "## Gates",
        "",
    ]
    for name in ("G0", "G1", "G2", "G3", "G4"):
        block = payload["gates"].get(name)
        if block is None:
            lines.append(f"- {name}: not evaluated")
            continue
        lines.append(f"- {name}: `{'PASS' if block.get('pass') else 'FAIL'}` { {k: v for k, v in block.items() if k != 'pass'} }")
    lines.extend(["", "## Metrics", ""])
    lines.append("| split | energy Δ | energy Δ₁ | energy T1 | proj share | |cos|_L | |cos|_R | n_deg |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, block in payload["splits"].items():
        lines.append(
            "| {split} | {energy_recorded_Delta:.9f} | {energy_Delta1:.9f} | {energy_T1:.9f} | "
            "{projection_share:.6e} | {abs_cos_left:.9f} | {abs_cos_right:.9f} | {n_degenerate} |".format(
                split=split, **block
            )
        )
    lines.extend(["", "## Per-prompt (evaluation splits)", ""])
    for split in EVAL_SPLITS:
        if split not in payload["splits"]:
            continue
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| prompt | deg | c_p | G3 rel | implied r·x | s_stored | s_implied | G4 |diff| |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in payload["splits"][split]["prompts"]:
            lines.append(
                "| {prompt_id} | {degenerate} | {c_p} | {g3_rel} | {r_dot_x_implied} | "
                "{s_stored} | {s_implied} | {g4_abs_diff} |".format(**row)
            )
        lines.append("")
    lines.extend(
        [
            "Discovery is descriptive only.",
            "",
            "NO_SELF_HASH_EMBEDDING: hashes live in `artifacts/m22_1_3b/manifest.sha256`.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
