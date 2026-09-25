"""Qwen loader and paired forward adapter.

The loader records the checkpoint revision when Hugging Face exposes it and
passes that revision into the load. It does not invent a revision.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .intervention import make_resid_hook
from .outcome import logit_margin


@dataclass
class LoadedModel:
    model: Any
    model_id: str
    model_revision: str | None
    revision_pinned: bool
    revision_source: str | None
    revision_error: str | None
    load_used_revision: bool
    device: str
    dtype: str
    versions: dict[str, str]
    n_layers: int
    d_model: int
    config_sha256: str


def _distribution_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def resolve_hf_revision(model_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        sha = getattr(info, "sha", None)
        if not sha:
            return {
                "model_revision": None,
                "revision_pinned": False,
                "revision_source": None,
                "revision_error": "model_info returned no sha",
            }
        return {
            "model_revision": str(sha),
            "revision_pinned": True,
            "revision_source": "huggingface_hub.model_info",
            "revision_error": None,
        }
    except Exception as exc:  # network, auth, or hub API failure
        return {
            "model_revision": None,
            "revision_pinned": False,
            "revision_source": None,
            "revision_error": f"{type(exc).__name__}: {exc}",
        }


def load_qwen(config: dict[str, Any]) -> LoadedModel:
    from transformer_lens import HookedTransformer

    model_id = str(config["model_id"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if str(config["dtype"]) != "float32":
        raise ValueError("Refusing to load with an unfrozen dtype.")
    revision_info = resolve_hf_revision(model_id)
    load_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    load_used_revision = False
    signature = inspect.signature(HookedTransformer.from_pretrained)
    accepts_revision = "revision" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if revision_info["revision_pinned"] and accepts_revision:
        load_kwargs["revision"] = revision_info["model_revision"]
        load_used_revision = True
    try:
        model = HookedTransformer.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        if "revision" not in load_kwargs:
            raise
        load_kwargs.pop("revision", None)
        load_used_revision = False
        revision_info["revision_pinned"] = False
        revision_info["revision_error"] = f"{type(exc).__name__}: {exc}"
        model = HookedTransformer.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if getattr(model, "tokenizer", None) is not None:
        try:
            model.tokenizer.padding_side = "left"
            if model.tokenizer.pad_token_id is None and model.tokenizer.eos_token is not None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
        except Exception:
            pass
    versions = {
        "torch": _distribution_version("torch"),
        "transformer_lens": _distribution_version("transformer-lens"),
        "transformers": _distribution_version("transformers"),
        "numpy": _distribution_version("numpy"),
    }
    from .config import config_sha256

    return LoadedModel(
        model=model,
        model_id=model_id,
        model_revision=revision_info["model_revision"],
        revision_pinned=bool(revision_info["revision_pinned"] and load_used_revision),
        revision_source=revision_info["revision_source"],
        revision_error=revision_info["revision_error"],
        load_used_revision=load_used_revision,
        device=device,
        dtype="float32",
        versions=versions,
        n_layers=int(model.cfg.n_layers),
        d_model=int(model.cfg.d_model),
        config_sha256=config_sha256(config),
    )


def _as_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Model forward did not return logits.")


def forward_record(
    bundle: LoadedModel,
    *,
    text: str,
    prepend_bos: bool,
    hook: bool,
    layer: int | None,
    alpha: float,
    direction: torch.Tensor | None,
    positive_id: int,
    negative_id: int,
    expected_hook_name: str | None,
) -> dict[str, Any]:
    tokens = bundle.model.to_tokens(text, prepend_bos=prepend_bos)
    recorder: dict[str, Any] = {}
    with torch.no_grad():
        if not hook:
            logits = _as_logits(bundle.model(tokens))
            return {
                "s": logit_margin(logits, positive_id, negative_id),
                "hook_fired": False,
                "last_modified": False,
                "other_unchanged": True,
                "hook_name": None,
            }
        if direction is None or layer is None or expected_hook_name is None:
            raise ValueError("Hooked forward requires a layer, direction, and hook name.")
        hook_fn = make_resid_hook(float(alpha), direction, recorder)
        logits = _as_logits(
            bundle.model.run_with_hooks(tokens, fwd_hooks=[(expected_hook_name, hook_fn)])
        )
    if int(recorder.get("fired", 0)) != 1:
        fired = False
    else:
        fired = True
    return {
        "s": logit_margin(logits, positive_id, negative_id),
        "hook_fired": fired,
        "last_modified": bool(recorder.get("last_modified", False)),
        "other_unchanged": bool(recorder.get("other_unchanged", False)),
        "hook_name": recorder.get("hook_name"),
    }


def direction_tensor(vector: np.ndarray, device: str) -> torch.Tensor:
    array = np.asarray(vector, dtype=np.float32)
    return torch.tensor(array, device=device, dtype=torch.float32)
