"""M23-F structural screen of existing interventions.

No predictor is fit. Validation and replication prompts are not forwarded.
Candidates are not ranked.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.m23.m22_reuse import (
    frozen_prompts,
    logit_margin,
    make_resid_hook,
    orthogonal_direction,
    primary_direction,
    prompt_manifest_sha256,
    vector_sha256,
)

FROZEN_HASHES = {
    "reports/M23_F_PREREGISTRATION.md": "bf3410e4ede664bd1aa450d86c2f56bce7b492465db72cf93648d9978aeaa1a4",
    "reports/M23_F_INTERVENTION_INVENTORY.md": "93a2a63905ede01718583bd7543c6fa9302387484d83335c749faeef507458d5",
}
PROMPT_MANIFEST = "fad6049b26ee8444ef39360b27b4cb328ed0691279b54a9b54e8c6227dda94db"
SCREEN_IDS = (
    "completion-01",
    "completion-04",
    "instruction-01",
    "instruction-04",
    "syntax-01",
    "syntax-04",
)
D1_SHA = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
D2_SHA = "8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e"
NEW_SEEDS = (22111, 22112, 22113, 22114, 22115, 22116)
LAYERS = (0, 8, 15, 23)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _sample_variance(values: list[float]) -> float:
    center = _mean(values)
    return float(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _sample_sd(values: list[float]) -> float:
    return float(math.sqrt(_sample_variance(values)))


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _check_frozen() -> None:
    for relative, expected in FROZEN_HASHES.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"frozen hash mismatch for {relative}: {actual}")
    if prompt_manifest_sha256() != PROMPT_MANIFEST:
        raise SystemExit("prompt catalog hash changed")
    found = tuple(record.prompt_id for record in frozen_prompts() if record.role == "discovery")
    if found != SCREEN_IDS:
        raise SystemExit(f"discovery ids changed: {found}")


def _project(raw: np.ndarray, accepted: list[np.ndarray]) -> tuple[np.ndarray, float]:
    projected = np.array(raw, dtype=np.float64, copy=True)
    for previous in accepted:
        projected = projected - float(np.dot(projected, previous)) * previous
    return projected, float(np.linalg.norm(projected))


def _new_direction(dimension: int, seed: int, accepted: list[np.ndarray]) -> tuple[np.ndarray, int]:
    for attempt in range(3):
        used_seed = int(seed) + attempt * 10000
        raw = primary_direction(dimension, used_seed)
        projected, residual_norm = _project(raw, accepted)
        if residual_norm >= 1e-8:
            return projected / residual_norm, used_seed
    raise RuntimeError(f"direction seed {seed} collapsed")


def _directions(dimension: int) -> list[dict[str, Any]]:
    d1 = primary_direction(dimension, 22101)
    if vector_sha256(d1) != D1_SHA:
        raise SystemExit("D1 hash mismatch")
    d2 = orthogonal_direction(dimension, 22103, d1)
    if vector_sha256(d2) != D2_SHA:
        raise SystemExit("D2 hash mismatch")
    records = [
        {"direction_id": "D1", "seed": 22101, "seed_used": 22101, "vector": d1, "layers": LAYERS},
        {"direction_id": "D2", "seed": 22103, "seed_used": 22103, "vector": d2, "layers": LAYERS},
    ]
    accepted = [d1, d2]
    for index, seed in enumerate(NEW_SEEDS, start=3):
        vector, used_seed = _new_direction(dimension, seed, accepted)
        accepted.append(vector)
        records.append(
            {
                "direction_id": f"D{index}",
                "seed": int(seed),
                "seed_used": int(used_seed),
                "vector": vector,
                "layers": (23,),
            }
        )
    return records


def _runner():
    path = ROOT / "scripts" / "run_m23_direct_qwen_self_model_audit.py"
    spec = importlib.util.spec_from_file_location("m23_historical_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load historical runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _forward(model, runner, text: str, positive_id: int, negative_id: int, hook: dict[str, Any] | None):
    import torch

    tokens = model.to_tokens(text, prepend_bos=True)
    recorder: dict[str, Any] = {}
    with torch.no_grad():
        if hook is None:
            logits = runner._as_logits(model(tokens))
            return logit_margin(logits, positive_id, negative_id), recorder
        direction = torch.tensor(hook["vector"], dtype=torch.float32)
        hook_fn = make_resid_hook(float(hook["alpha"]), direction, recorder)
        hook_name = f"blocks.{int(hook['layer'])}.hook_resid_post"
        logits = runner._as_logits(model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)]))
    if int(recorder.get("fired", 0)) != 1:
        raise RuntimeError(f"hook fired {recorder.get('fired')} times")
    if not recorder.get("other_unchanged", False):
        raise RuntimeError("intervention modified a non-final position")
    if not recorder.get("last_modified", False):
        raise RuntimeError("intervention did not modify the final position")
    return logit_margin(logits, positive_id, negative_id), recorder


def _measure(runner, directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model, _device = runner.load_model()
    if int(model.cfg.d_model) != 896:
        raise SystemExit(f"direction dimension does not match the model: {model.cfg.d_model}")
    tokens = runner.resolve_outcome_tokens(model.tokenizer, runner.POSITIVE_TEXT, runner.NEGATIVE_TEXT)
    if (tokens["positive_token_id"], tokens["negative_token_id"]) != runner.EXPECTED_TOKEN_IDS:
        raise SystemExit(f"token ids changed: {tokens['token_ids']}")
    prompts = {record.prompt_id: record for record in frozen_prompts()}
    rows: list[dict[str, Any]] = []
    for repeat_index in (0, 1):
        for prompt_id in SCREEN_IDS:
            if prompts[prompt_id].role != "discovery":
                raise SystemExit(f"{prompt_id} is not a discovery prompt")
            text = prompts[prompt_id].text
            baseline, _empty = _forward(
                model, runner, text, tokens["positive_token_id"], tokens["negative_token_id"], None
            )
            for direction in directions:
                for layer in direction["layers"]:
                    for alpha in (1.0, 2.0):
                        intervened, recorder = _forward(
                            model,
                            runner,
                            text,
                            tokens["positive_token_id"],
                            tokens["negative_token_id"],
                            {"alpha": alpha, "layer": layer, "vector": direction["vector"]},
                        )
                        rows.append(
                            {
                                "episode_id": f"{direction['direction_id']}-L{layer}-{prompt_id}-a{int(alpha)}-r{repeat_index}",
                                "intervention_id": f"M22.1-{direction['direction_id']}-L{layer}",
                                "prompt_id": prompt_id,
                                "seed": direction["seed"],
                                "seed_used": direction["seed_used"],
                                "direction_id": direction["direction_id"],
                                "layer": layer,
                                "alpha": alpha,
                                "repeat_index": repeat_index,
                                "pre_dot": float(recorder["pre_dot"]),
                                "baseline_output": float(baseline),
                                "intervened_output": float(intervened),
                                "observed_effect": float(intervened - baseline),
                            }
                        )
            print(f"repeat {repeat_index} {prompt_id}", flush=True)
    return rows


def _cells(rows: list[dict[str, Any]]) -> list[tuple[str, int]]:
    found = {(row["direction_id"], int(row["layer"])) for row in rows}
    return sorted(found, key=lambda item: (item[0], item[1]))


def _subset(rows: list[dict[str, Any]], direction_id: str, layer: int, alpha: float, repeat_index: int) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["direction_id"] == direction_id
        and int(row["layer"]) == layer
        and float(row["alpha"]) == alpha
        and int(row["repeat_index"]) == repeat_index
    ]
    return sorted(selected, key=lambda row: row["prompt_id"])


def _inspected(direction_id: str, layer: int) -> dict[str, bool]:
    panel = direction_id in {f"D{index}" for index in range(1, 9)} and layer == 23
    early = direction_id in {"D1", "D2"} and layer in {0, 8, 15}
    return {
        "discovery": True,
        "validation": bool(panel or early),
        "replication": bool(panel),
    }


def _summarize_candidate(rows: list[dict[str, Any]], direction_id: str, layer: int, reference: dict[str, float]) -> dict[str, Any]:
    primary = _subset(rows, direction_id, layer, 1.0, 0)
    doubled = _subset(rows, direction_id, layer, 2.0, 0)
    effects = [float(row["observed_effect"]) for row in primary]
    signs = sorted({_sign(value) for value in effects})
    ratios = []
    undefined = 0
    by_prompt = {row["prompt_id"]: float(row["observed_effect"]) for row in doubled}
    for row in primary:
        first = float(row["observed_effect"])
        second = by_prompt[row["prompt_id"]]
        if first == 0.0:
            undefined += 1
            ratios.append({"prompt_id": row["prompt_id"], "ratio": None})
        else:
            ratios.append({"prompt_id": row["prompt_id"], "ratio": second / first})
    defined = [float(item["ratio"]) for item in ratios if item["ratio"] is not None]
    max_repeat = 0.0
    for alpha in (1.0, 2.0):
        left = _subset(rows, direction_id, layer, alpha, 0)
        right = _subset(rows, direction_id, layer, alpha, 1)
        for first, second in zip(left, right):
            for field in ("pre_dot", "baseline_output", "intervened_output", "observed_effect"):
                max_repeat = max(max_repeat, abs(float(first[field]) - float(second[field])))
    contrasts = [
        {
            "prompt_id": row["prompt_id"],
            "effect": float(row["observed_effect"]),
            "d1_l23_effect": reference[row["prompt_id"]],
            "equal_to_d1_l23": float(row["observed_effect"]) == reference[row["prompt_id"]],
        }
        for row in primary
    ]
    inspection = _inspected(direction_id, layer)
    clean = not inspection["validation"] and not inspection["replication"]
    sign_disagreement = signs == [-1, 1]
    differs_from_current = any(not item["equal_to_d1_l23"] for item in contrasts)
    return {
        "intervention_id": f"M22.1-{direction_id}-L{layer}",
        "direction_id": direction_id,
        "layer": layer,
        "n": len(effects),
        "alpha_plus_1": {
            "mean": _mean(effects),
            "sample_variance": _sample_variance(effects),
            "sample_sd": _sample_sd(effects),
            "min": float(min(effects)),
            "max": float(max(effects)),
            "range": float(max(effects) - min(effects)),
            "signs": signs,
            "same_sign": len(signs) == 1 and 0 not in signs,
            "sign_disagreement": sign_disagreement,
            "effects": [
                {"prompt_id": row["prompt_id"], "observed_effect": float(row["observed_effect"]), "pre_dot": float(row["pre_dot"])}
                for row in primary
            ],
        },
        "alpha_scaling": {
            "undefined_ratios": undefined,
            "mean_ratio": None if not defined else _mean(defined),
            "sample_sd_ratio": None if len(defined) < 2 else _sample_sd(defined),
            "ratios": ratios,
        },
        "repeat_max_abs_difference": max_repeat,
        "repeatable_identical": max_repeat == 0.0,
        "differs_from_d1_l23": differs_from_current if not (direction_id == "D1" and layer == 23) else False,
        "structural_contrast": bool(sign_disagreement or (differs_from_current and not (direction_id == "D1" and layer == 23))),
        "inspection": inspection,
        "clean_future_experiment": clean,
        "contrasts_with_d1_l23": contrasts,
    }


def _reference(rows: list[dict[str, Any]]) -> dict[str, float]:
    primary = _subset(rows, "D1", 23, 1.0, 0)
    if [row["prompt_id"] for row in primary] != list(SCREEN_IDS):
        raise RuntimeError("D1 layer 23 screen is incomplete")
    return {row["prompt_id"]: float(row["observed_effect"]) for row in primary}


def main() -> None:
    _check_frozen()
    raw_dir = ROOT / "reports" / "m23_f_raw"
    if (raw_dir / "measurements.json").exists():
        raise SystemExit("M23-F measurements already exist; refusing to mix runs")
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "screen_set.json").write_text(
        json.dumps({"prompt_ids": list(SCREEN_IDS), "alphas": [1.0, 2.0], "repeats": 2}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    runner = _runner()
    # Dimension is fixed by the pinned architecture. Directions are built before the forward
    # so a hash failure stops the run without a scored outcome.
    directions = _directions(896)
    rows = _measure(runner, directions)
    if any(row["prompt_id"] not in SCREEN_IDS for row in rows):
        raise SystemExit("a non-discovery prompt was measured")
    (raw_dir / "measurements.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    reference = _reference(rows)
    candidates = [
        _summarize_candidate(rows, direction_id, layer, reference) for direction_id, layer in _cells(rows)
    ]
    clean = [row["intervention_id"] for row in candidates if row["structural_contrast"] and row["clean_future_experiment"]]
    if len(clean) >= 2:
        audit_status = "MULTIPLE_CANDIDATES_AVAILABLE"
    elif len(clean) == 1:
        audit_status = "CANDIDATE_INTERVENTION_AVAILABLE"
    else:
        audit_status = "NO_SUITABLE_EXISTING_INTERVENTION"
    current = next(row for row in candidates if row["intervention_id"] == "M22.1-D1-L23")
    if current["alpha_plus_1"]["same_sign"]:
        current_status = "CURRENT_INTERVENTION_INSUFFICIENT"
    else:
        current_status = "INCONCLUSIVE"
    payload = {
        "historical_m23_verdict": "INCONCLUSIVE",
        "audit_status": audit_status,
        "current_intervention_status": current_status,
        "clean_candidates": clean,
        "structural_contrasts_not_clean": [
            row["intervention_id"] for row in candidates if row["structural_contrast"] and not row["clean_future_experiment"]
        ],
        "candidates": candidates,
        "predictor_fit": False,
        "ranking_used": False,
        "validation_or_replication_measured": False,
    }
    (ROOT / "reports" / "M23_F_RESULTS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    fieldnames = [
        "episode_id",
        "intervention_id",
        "prompt_id",
        "seed",
        "seed_used",
        "direction_id",
        "layer",
        "alpha",
        "repeat_index",
        "pre_dot",
        "baseline_output",
        "intervened_output",
        "observed_effect",
    ]
    with (ROOT / "reports" / "M23_F_SCREENING_EPISODES.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    print(audit_status, current_status, flush=True)
    for row in candidates:
        block = row["alpha_plus_1"]
        print(
            row["intervention_id"],
            "signs",
            block["signs"],
            "range",
            f"{block['range']:.6f}",
            "ratio",
            row["alpha_scaling"]["mean_ratio"],
            "repeat0",
            row["repeatable_identical"],
            flush=True,
        )


if __name__ == "__main__":
    main()
