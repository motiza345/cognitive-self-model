"""Build readout predictions from frozen prompts and directions.

This module does not open or name the M22.1.1 measurement table.
Cell keys come only from the frozen prompt list and the direction panel.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from ..m22_1.config import load_config as load_m22_1_config
from ..m22_1.direction import vector_sha256
from ..m22_1.loader import load_qwen
from ..m22_1.outcome import logit_margin
from ..m22_1.prompts import frozen_prompts, prompt_manifest_sha256
from ..m22_1_1.panel import build_direction_panel, load_audit_config

ROOT = Path(__file__).resolve().parents[3]
HOOK_NAME = "blocks.23.hook_resid_post"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_audit_yaml(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else ROOT / "configs" / "m22_1_3_readout_null.yaml"
    with config_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rms_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + float(eps))


def closed_form_margin(x: torch.Tensor, r: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    scale = rms_scale(x, eps)
    return (x / scale * r).sum(dim=-1) + bias


def jacobian_first_order(x: torch.Tensor, direction: torch.Tensor, r: torch.Tensor, eps: float) -> torch.Tensor:
    n = float(x.shape[-1])
    scale = rms_scale(x, eps).squeeze(-1)
    r_dot_d = (r * direction).sum(dim=-1)
    r_dot_x = (r * x).sum(dim=-1)
    x_dot_d = (x * direction).sum(dim=-1)
    return r_dot_d / scale - r_dot_x * x_dot_d / (n * scale.pow(3))


def frozen_denominator(x: torch.Tensor, direction: torch.Tensor, r: torch.Tensor, eps: float) -> torch.Tensor:
    scale = rms_scale(x, eps).squeeze(-1)
    return (r * direction).sum(dim=-1) / scale


def module_margin(residual: torch.Tensor, ln_final, unembed, positive_id: int, negative_id: int) -> torch.Tensor:
    if residual.ndim == 1:
        residual = residual.view(1, 1, -1)
    elif residual.ndim == 2:
        residual = residual.unsqueeze(1)
    normalized = ln_final(residual)
    logits = unembed(normalized)
    return logits[..., int(positive_id)] - logits[..., int(negative_id)]


def r_squared_no_intercept(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    truth = np.asarray(actual, dtype=np.float64).ravel()
    estimate = np.asarray(predicted, dtype=np.float64).ravel()
    denom = float(np.sum(truth ** 2))
    if denom <= 1e-18:
        return None
    return float(1.0 - np.sum((truth - estimate) ** 2) / denom)


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> dict:
    left = np.asarray(x, dtype=np.float64).ravel()
    right = np.asarray(y, dtype=np.float64).ravel()
    denom = float(np.dot(left, left))
    slope = 0.0 if denom == 0.0 else float(np.dot(left, right) / denom)
    fitted = slope * left
    return {"k": slope, "r2": r_squared_no_intercept(right, fitted)}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def load_prompt_records(config: dict) -> list[dict]:
    """Load prompts from the frozen M22.1 catalog, not from measurement rows."""
    prompts = frozen_prompts()
    if not prompts:
        return []
    expected_hash = str(config["prompt_manifest_sha256"])
    if prompt_manifest_sha256(prompts) != expected_hash:
        raise RuntimeError("frozen prompt hash does not match the M22.1.3 config")
    artifact = ROOT / config["paths"]["prompt_artifact"]
    if artifact.is_file():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("prompt_manifest_sha256") != expected_hash:
            raise RuntimeError("M22.1 prompt artifact hash does not match the frozen catalog")
        by_id = {row["prompt_id"]: row for row in payload["prompts"]}
        for prompt in prompts:
            recorded = by_id[prompt.prompt_id]
            if recorded["text"] != prompt.text or recorded["role"] != prompt.role:
                raise RuntimeError(f"prompt artifact diverged from frozen catalog: {prompt.prompt_id}")
    return [
        {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "split": prompt.role,
            "regime": prompt.regime_id,
            "prompt_sha256": prompt.prompt_sha256,
        }
        for prompt in prompts
    ]


def build_cell_keys(prompts: list[dict], direction_ids: list[str], alphas: list[float]) -> list[dict]:
    cells = []
    for prompt in prompts:
        for direction_id in direction_ids:
            for alpha in alphas:
                cells.append(
                    {
                        "prompt_id": prompt["prompt_id"],
                        "split": prompt["split"],
                        "regime": prompt["regime"],
                        "direction_id": direction_id,
                        "alpha": float(alpha),
                    }
                )
    return cells


def _cfg_flag(cfg, name: str):
    if hasattr(cfg, name):
        return getattr(cfg, name)
    extra = getattr(cfg, "__dict__", {})
    if name in extra:
        return extra[name]
    return "not_on_cfg"


def inspect_loaded_model(bundle) -> dict:
    model = bundle.model
    cfg = model.cfg
    ln_final = getattr(model, "ln_final", None)
    weight = None
    if ln_final is not None and hasattr(ln_final, "w"):
        weight = ln_final.w.detach().to(dtype=torch.float64).cpu().numpy()
    has_weight = weight is not None
    eps = None
    if ln_final is not None and hasattr(ln_final, "eps"):
        eps = float(ln_final.eps)
    elif hasattr(cfg, "eps"):
        eps = float(cfg.eps)
    return {
        "normalization_type": getattr(cfg, "normalization_type", "UNKNOWN"),
        "ln_final_class": type(ln_final).__name__ if ln_final is not None else None,
        "ln_final_has_weight": has_weight,
        "ln_final_weight_is_ones": bool(has_weight and np.allclose(weight, 1.0)),
        "eps": eps,
        "fold_ln": _cfg_flag(cfg, "fold_ln"),
        "center_writing_weights": _cfg_flag(cfg, "center_writing_weights"),
        "center_unembed": _cfg_flag(cfg, "center_unembed"),
        "default_prepend_bos": _cfg_flag(cfg, "default_prepend_bos"),
        "d_model": int(cfg.d_model),
        "n_layers": int(cfg.n_layers),
        "dtype": str(getattr(cfg, "dtype", bundle.dtype)),
        "device": str(bundle.device),
        "model_revision": bundle.model_revision,
        "revision_pinned": bundle.revision_pinned,
        "load_used_revision": bundle.load_used_revision,
    }


def extract_readout(model, realized: dict, positive_id: int, negative_id: int) -> dict:
    w_u = model.W_U.detach()
    if w_u.ndim != 2:
        raise RuntimeError("unembedding matrix is not two-dimensional")
    if w_u.shape[0] == realized["d_model"]:
        pos = w_u[:, int(positive_id)]
        neg = w_u[:, int(negative_id)]
    elif w_u.shape[1] == realized["d_model"]:
        pos = w_u[int(positive_id)]
        neg = w_u[int(negative_id)]
    else:
        raise RuntimeError("cannot align W_U with d_model")
    delta = pos - neg
    if realized["ln_final_has_weight"] and not realized["ln_final_weight_is_ones"]:
        weight = model.ln_final.w.detach().to(dtype=delta.dtype, device=delta.device)
        readout = weight * delta
        construction = "w_ln_odot_unembed_difference"
    else:
        readout = delta
        construction = "unembed_difference_gain_folded_or_absent"
    if hasattr(model, "b_U") and model.b_U is not None:
        bias = model.b_U.detach()[int(positive_id)] - model.b_U.detach()[int(negative_id)]
    else:
        bias = torch.zeros((), dtype=delta.dtype, device=delta.device)
    return {
        "r": readout.detach(),
        "b": bias.detach(),
        "construction": construction,
        "eps": float(realized["eps"]),
    }


def _unembed_module(model):
    if hasattr(model, "unembed"):
        return model.unembed
    raise RuntimeError("loaded model has no unembed module")


def cache_last_residual(
    model,
    text: str,
    prepend_bos: bool,
    hook_name: str,
    positive_id: int,
    negative_id: int,
) -> tuple[torch.Tensor, float]:
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    captured: dict[str, torch.Tensor] = {}

    def record_only(residual: torch.Tensor, hook) -> torch.Tensor:
        captured["x"] = residual[:, -1, :].detach().clone()
        return residual

    with torch.no_grad():
        output = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, record_only)])
    if "x" not in captured:
        raise RuntimeError(f"clean cache hook did not fire on {hook_name}")
    if captured["x"].shape[0] != 1:
        raise RuntimeError("M22.1 scores one prompt at a time")
    logits = output if torch.is_tensor(output) else output[0]
    return captured["x"][0], logit_margin(logits, int(positive_id), int(negative_id))


def replay_margin_float64(x: torch.Tensor, r: torch.Tensor, bias: torch.Tensor, eps: float) -> float:
    return float(closed_form_margin(x.to(torch.float64), r.to(torch.float64), bias.to(torch.float64), eps).item())


def check_closed_form(xs: list[torch.Tensor], ln_final, unembed, readout: dict, positive_id: int, negative_id: int, rel_tol: float) -> dict:
    rows = []
    for index, residual in enumerate(xs):
        with torch.no_grad():
            module_value = float(module_margin(residual, ln_final, unembed, positive_id, negative_id).reshape(-1)[0].item())
        closed = float(
            closed_form_margin(
                residual.to(dtype=readout["r"].dtype),
                readout["r"].to(device=residual.device),
                readout["b"].to(device=residual.device),
                readout["eps"],
            ).reshape(-1)[0].item()
        )
        denom = max(abs(module_value), 1e-8)
        rel = abs(closed - module_value) / denom
        rows.append({"index": index, "module": module_value, "closed_form": closed, "rel_error": rel})
        if rel > float(rel_tol):
            raise RuntimeError(f"closed-form margin diverged from modules: rel={rel}")
    return {"pass": True, "rows": rows}


def check_jacobian(xs: list[torch.Tensor], directions: np.ndarray, readout: dict, rel_tol: float, seed: int) -> dict:
    generator = np.random.default_rng(int(seed))
    rows = []
    r = readout["r"].detach().to(dtype=torch.float64).cpu()
    eps = float(readout["eps"])
    for index in range(5):
        residual = xs[index % len(xs)].detach().to(dtype=torch.float64).cpu().clone()
        direction = torch.tensor(directions[int(generator.integers(0, len(directions)))], dtype=torch.float64)
        residual.requires_grad_(True)
        value = closed_form_margin(residual, r, readout["b"].detach().to(dtype=torch.float64).cpu(), eps)
        gradient = torch.autograd.grad(value, residual)[0]
        automatic = float(torch.dot(gradient, direction).item())
        closed = float(jacobian_first_order(residual.detach(), direction, r, eps).item())
        denom = max(abs(automatic), 1e-12)
        rel = abs(closed - automatic) / denom
        rows.append({"index": index, "autograd": automatic, "closed": closed, "rel_error": rel})
        if rel > float(rel_tol):
            raise RuntimeError(f"Jacobian closed form diverged from autograd: rel={rel}")
    return {"pass": True, "rows": rows}


def run_predictions(config_path: str | Path | None = None, output_dir: str | Path | None = None) -> dict:
    config_path = Path(config_path) if config_path is not None else ROOT / "configs" / "m22_1_3_readout_null.yaml"
    config = load_audit_yaml(config_path)
    config_hash = sha256_file(config_path)
    output = Path(output_dir) if output_dir is not None else ROOT
    artifact_dir = output / "artifacts" / "m22_1_3"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_log = {
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "predictions_started": False,
        "environment": {
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "transformer_lens": _package_version("transformer-lens"),
        },
    }
    prompts = load_prompt_records(config)
    if not prompts:
        payload = {
            "gate": config["labels"]["prompt_missing"],
            "run_log": run_log,
            "cells": [],
        }
        _write_predictions(output / config["paths"]["predictions"], payload)
        return payload
    direction_config = load_audit_config(ROOT / config["paths"]["m22_1_1_config"])
    panel = build_direction_panel(int(config["expected_d_model"]), direction_config)
    hashes = {row["direction_id"]: row["final_sha256"] for row in panel["directions"]}
    expected = config["direction_hashes"]
    if hashes != expected:
        payload = {
            "gate": config["labels"]["directions_fail"],
            "run_log": run_log,
            "observed_hashes": hashes,
            "cells": [],
        }
        _write_predictions(output / config["paths"]["predictions"], payload)
        return payload
    vectors = np.stack([row["vector"] for row in panel["directions"]]).astype(np.float64)
    for index, row in enumerate(panel["directions"]):
        if vector_sha256(vectors[index]) != row["final_sha256"]:
            raise RuntimeError(f"{row['direction_id']} hash diverged after stacking")
    directions_path = output / config["paths"]["directions_npy"]
    np.save(directions_path, vectors)
    cells = build_cell_keys(prompts, list(config["direction_ids"]), [float(value) for value in config["alphas"]])
    m22_1 = load_m22_1_config(ROOT / config["paths"]["m22_1_config"])
    bundle = load_qwen(m22_1)
    realized = inspect_loaded_model(bundle)
    model = bundle.model
    model.eval()
    readout = extract_readout(model, realized, int(config["positive_id"]), int(config["negative_id"]))
    unembed = _unembed_module(model)
    residuals = []
    clean_margins = {}
    for prompt in prompts:
        residual, clean_from_logits = cache_last_residual(
            model,
            prompt["text"],
            bool(config["prepend_bos"]),
            HOOK_NAME,
            int(config["positive_id"]),
            int(config["negative_id"]),
        )
        residuals.append(residual.detach().cpu())
        with torch.no_grad():
            clean_from_modules = float(
                module_margin(residual, model.ln_final, unembed, int(config["positive_id"]), int(config["negative_id"]))
                .reshape(-1)[0]
                .item()
            )
        clean_margins[prompt["prompt_id"]] = {
            "from_clean_logits": float(clean_from_logits),
            "from_modules": clean_from_modules,
            "from_closed_form": replay_margin_float64(residual, readout["r"], readout["b"], readout["eps"]),
            "split": prompt["split"],
        }
    residual_matrix = torch.stack(residuals, dim=0)
    resid_path = output / config["paths"]["clean_resid_npy"]
    np.save(resid_path, residual_matrix.numpy())
    closed_form_check = check_closed_form(
        residuals[: int(config["n_closed_form_checks"])],
        model.ln_final,
        unembed,
        readout,
        int(config["positive_id"]),
        int(config["negative_id"]),
        float(config["closed_form_rel_tol"]),
    )
    jacobian_check = check_jacobian(
        residuals,
        vectors,
        readout,
        float(config["jacobian_rel_tol"]),
        int(config["check_seed"]),
    )
    prompt_index = {prompt["prompt_id"]: index for index, prompt in enumerate(prompts)}
    direction_index = {direction_id: index for index, direction_id in enumerate(config["direction_ids"])}
    r_f32 = readout["r"].detach().to(dtype=torch.float32)
    b_f32 = readout["b"].detach().to(dtype=torch.float32)
    r_f64 = readout["r"].detach().to(dtype=torch.float64)
    b_f64 = readout["b"].detach().to(dtype=torch.float64)
    eps = float(readout["eps"])
    records = []
    alpha_zero_ok = True
    for cell in cells:
        residual = residual_matrix[prompt_index[cell["prompt_id"]]]
        direction64 = torch.tensor(vectors[direction_index[cell["direction_id"]]], dtype=torch.float64)
        direction32 = direction64.to(dtype=torch.float32)
        alpha = float(cell["alpha"])
        with torch.no_grad():
            clean = float(module_margin(residual, model.ln_final, unembed, int(config["positive_id"]), int(config["negative_id"])).reshape(-1)[0].item())
            perturbed = residual + float(alpha) * direction32
            exact32 = float(module_margin(perturbed, model.ln_final, unembed, int(config["positive_id"]), int(config["negative_id"])).reshape(-1)[0].item()) - clean
        exact64 = replay_margin_float64(residual.to(torch.float64) + float(alpha) * direction32.to(torch.float64), r_f64, b_f64, eps) - replay_margin_float64(
            residual, r_f64, b_f64, eps
        )
        first = float(alpha) * float(jacobian_first_order(residual.to(torch.float64), direction32.to(torch.float64), r_f64, eps).item())
        frozen = float(alpha) * float(frozen_denominator(residual.to(torch.float64), direction32.to(torch.float64), r_f64, eps).item())
        if alpha == 0.0 and exact32 != 0.0:
            alpha_zero_ok = False
        records.append(
            {
                **cell,
                "delta_exact_f32": exact32,
                "delta_exact_f64": exact64,
                "delta_jacobian": first,
                "delta_frozen": frozen,
                "inv_rms": float(1.0 / float(rms_scale(residual.to(torch.float64), eps).item())),
                "r_dot_d": float(torch.dot(r_f64.cpu(), direction32.to(torch.float64)).item()),
            }
        )
    gate = None if alpha_zero_ok else config["labels"]["sanity_fail"]
    run_log["predictions_started"] = True
    run_log["realized"] = realized
    run_log["readout_construction"] = readout["construction"]
    run_log["readout_norm"] = float(torch.linalg.norm(r_f64).item())
    run_log["readout_bias"] = float(b_f64.item()) if b_f64.ndim == 0 else float(b_f64.reshape(-1)[0].item())
    run_log["closed_form_check"] = closed_form_check
    run_log["jacobian_check"] = jacobian_check
    run_log["directions_sha256"] = sha256_file(directions_path)
    run_log["clean_resid_sha256"] = sha256_file(resid_path)
    payload = {
        "gate": gate,
        "run_log": run_log,
        "prompt_ids": [prompt["prompt_id"] for prompt in prompts],
        "prompt_splits": {prompt["prompt_id"]: prompt["split"] for prompt in prompts},
        "direction_ids": list(config["direction_ids"]),
        "direction_hashes": hashes,
        "clean_margins": clean_margins,
        "cells": records,
    }
    predictions_path = output / config["paths"]["predictions"]
    digest = _write_predictions(predictions_path, payload)
    payload["run_log"]["predictions_sha256"] = digest
    predictions_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_predictions(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)
    return sha256_bytes(encoded)
