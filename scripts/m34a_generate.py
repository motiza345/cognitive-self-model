"""Device-agnostic Qwen generation for M34a pilot/collection (no analysis imports)."""

from __future__ import annotations

import math
import time
from typing import Any

from scripts.m34a_common import (
    FALLBACK_MODEL,
    MAX_NEW_TOKENS,
    PRIMARY_MODEL,
    parse_response,
)


class MockQwenClient:
    """Deterministic mock: correct for digit levels below fail_from; confidence scales with n."""

    def __init__(self, fail_from_digits: int = 6) -> None:
        self.fail_from_digits = fail_from_digits
        self.model_id = "mock-qwen"
        self.revision = "mock"
        self.device_name = "cpu"
        self.dtype_name = "float32"

    def generate(self, prompt: str) -> dict[str, Any]:
        import re

        m = re.search(r"Compute (\d+) x (\d+)\.", prompt)
        if not m:
            raise ValueError(f"bad mock prompt: {prompt!r}")
        a, b = int(m.group(1)), int(m.group(2))
        n = max(len(str(a)), len(str(b)))
        product = a * b
        if n >= self.fail_from_digits:
            ans = product + 1
            conf = 35
        else:
            ans = product
            conf = min(95, 55 + 5 * (8 - n))
        text = f"Answer: {ans}; Confidence: {conf}"
        return {
            "text": text,
            "input_tokens": 40 + n,
            "output_tokens": 18,
            "float32_rerun": False,
            "nonfinite_detected": False,
            "dtype_used": "float32",
            "request_params": {
                "model": self.model_id,
                "revision": self.revision,
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": "float32",
            },
            "latency_s": 0.0,
        }


def resolve_revision(repo_id: str) -> str:
    from huggingface_hub import model_info

    return str(model_info(repo_id).sha)


def _pick_torch_dtype(name: str):
    import torch

    return torch.float16 if name == "float16" else torch.float32


class TransformersQwenClient:
    """Load Qwen2.5 Instruct; greedy decode; float16 with float32 rerun on non-finite."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        prefer_dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        candidates = [model_id] if model_id else [PRIMARY_MODEL, FALLBACK_MODEL]
        last_err: Exception | None = None
        self.model = None
        self.tokenizer = None
        self.model_id = ""
        self.revision = ""
        for repo in candidates:
            if not repo:
                continue
            try:
                rev = resolve_revision(repo)
                tok = AutoTokenizer.from_pretrained(repo, revision=rev, trust_remote_code=True)
                if device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = _pick_torch_dtype(prefer_dtype if device.startswith("cuda") else "float32")
                model = AutoModelForCausalLM.from_pretrained(
                    repo,
                    revision=rev,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                model.to(device)
                model.eval()
                self.model = model
                self.tokenizer = tok
                self.model_id = repo
                self.revision = rev
                self.device_name = str(device)
                self.dtype_name = "float16" if dtype == torch.float16 else "float32"
                self.prefer_dtype = self.dtype_name
                break
            except Exception as exc:  # noqa: BLE001 — try fallback
                last_err = exc
                self.model = None
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(f"could not load Qwen models {candidates}: {last_err}")

    def _forward_once(self, prompt: str, dtype_name: str) -> dict[str, Any]:
        torch = self.torch
        dtype = _pick_torch_dtype(dtype_name)
        # Move weights if dtype changes for a rerun.
        if next(self.model.parameters()).dtype != dtype:
            self.model.to(dtype=dtype)

        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device_name) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency = time.perf_counter() - t0
        sequences = out.sequences
        gen_ids = sequences[0, input_len:]
        text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        nonfinite = False
        if out.scores:
            for step_scores in out.scores:
                if not torch.isfinite(step_scores).all():
                    nonfinite = True
                    break
                # log_softmax can introduce -inf for masked positions; treat NaN/+inf as bad.
                logp = torch.log_softmax(step_scores.float(), dim=-1)
                if torch.isnan(logp).any() or torch.isposinf(logp).any():
                    nonfinite = True
                    break

        return {
            "text": text_out,
            "input_tokens": input_len,
            "output_tokens": int(gen_ids.numel()),
            "nonfinite_detected": nonfinite,
            "dtype_used": dtype_name,
            "latency_s": latency,
            "request_params": {
                "model": self.model_id,
                "revision": self.revision,
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "dtype": dtype_name,
                "device": self.device_name,
            },
        }

    def generate(self, prompt: str) -> dict[str, Any]:
        first = self._forward_once(prompt, self.prefer_dtype)
        if not first["nonfinite_detected"]:
            first["float32_rerun"] = False
            return first
        second = self._forward_once(prompt, "float32")
        second["float32_rerun"] = True
        second["nonfinite_detected"] = True
        return second


def run_problem(client: Any, problem: dict[str, Any]) -> dict[str, Any]:
    out = client.generate(problem["prompt"])
    parsed_ans, conf = parse_response(out["text"])
    correct = parsed_ans is not None and parsed_ans == int(problem["product"])
    return {
        "problem_id": problem["problem_id"],
        "pool": problem["pool"],
        "level": int(problem["level"]),
        "a": int(problem["a"]),
        "b": int(problem["b"]),
        "product": int(problem["product"]),
        "prompt": problem["prompt"],
        "raw_response": out["text"],
        "parsed_answer": parsed_ans,
        "parsed_confidence": float(conf),
        "correct": bool(correct),
        "input_tokens": int(out["input_tokens"]),
        "output_tokens": int(out["output_tokens"]),
        "float32_rerun": bool(out.get("float32_rerun", False)),
        "nonfinite_detected": bool(out.get("nonfinite_detected", False)),
        "dtype_used": out.get("dtype_used"),
        "request_params": out.get("request_params", {}),
        "latency_s": float(out.get("latency_s", 0.0)),
    }
