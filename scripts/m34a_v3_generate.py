"""M34a-v3 generation: parameterized chat models with per-token logprobs (no analysis)."""

from __future__ import annotations

import math
import re
import time
from typing import Any

from scripts.m34a_common import MAX_NEW_TOKENS, parse_response
from scripts.m34a_v3_common import PHI_MODEL_ID, QWEN_MODEL_ID, QWEN_REVISION


class MockV3Client:
    """Deterministic mock; correctness falls with digit length; supports + and x."""

    def __init__(self, fail_from_digits: int = 7, model_id: str = "mock-v3") -> None:
        self.fail_from_digits = fail_from_digits
        self.model_id = model_id
        self.revision = "mock"
        self.device_name = "cpu"
        self.dtype_name = "float32"

    def generate(self, prompt: str) -> dict[str, Any]:
        m = re.search(r"Compute (\d+) ([+x]) (\d+)\.", prompt)
        if not m:
            raise ValueError(f"bad mock prompt: {prompt!r}")
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        n = max(len(str(a)), len(str(b)))
        target = a + b if op == "+" else a * b
        if n >= self.fail_from_digits:
            ans = target + 1
            lp = -2.0 - 0.1 * n
        else:
            ans = target
            lp = -0.2 - 0.05 * n
        text = str(ans)
        n_tok = max(1, len(text))
        token_lps = [lp / n_tok] * n_tok
        return {
            "text": text,
            "generated_token_ids": list(range(n_tok)),
            "token_logprobs": token_lps,
            "answer_logprob": float(sum(token_lps)),
            "mean_logprob": float(sum(token_lps) / n_tok),
            "input_tokens": 40 + n,
            "output_tokens": n_tok,
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


class TransformersChatClient:
    """Greedy HF chat model with EOS-excluded token logprobs (same rule as v2)."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        prefer_dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        rev = revision or resolve_revision(model_id)
        tok = AutoTokenizer.from_pretrained(model_id, revision=rev, trust_remote_code=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _pick_torch_dtype(prefer_dtype if str(device).startswith("cuda") else "float32")
        # Phi-3.5 + recent transformers: DynamicCache.seen_tokens crash unless eager + no KV cache.
        load_kwargs: dict[str, Any] = {
            "revision": rev,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, attn_implementation="eager", **load_kwargs
            )
            attn_impl = "eager"
        except (TypeError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            attn_impl = "default"
        model.to(device)
        model.eval()
        self.model = model
        self.tokenizer = tok
        self.model_id = model_id
        self.revision = rev
        self.device_name = str(device)
        self.dtype_name = "float16" if dtype == torch.float16 else "float32"
        self.prefer_dtype = self.dtype_name
        self.eos_id = tok.eos_token_id
        self.attn_implementation = attn_impl
        # use_cache=False avoids DynamicCache.seen_tokens on Phi; scores still returned.
        self.generate_use_cache = False

    def _forward_once(self, prompt: str, dtype_name: str) -> dict[str, Any]:
        torch = self.torch
        dtype = _pick_torch_dtype(dtype_name)
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
                use_cache=self.generate_use_cache,
            )
        latency = time.perf_counter() - t0
        gen_ids = out.sequences[0, input_len:]
        ids = [int(x) for x in gen_ids.tolist()]
        scores = list(out.scores) if out.scores else []
        token_lps: list[float] = []
        nonfinite = False
        for i, tid in enumerate(ids):
            if self.eos_id is not None and tid == self.eos_id:
                continue
            if i >= len(scores):
                nonfinite = True
                break
            step = scores[i]
            if not torch.isfinite(step).all():
                nonfinite = True
                break
            logp = torch.log_softmax(step.float(), dim=-1)
            if torch.isnan(logp).any() or torch.isposinf(logp).any():
                nonfinite = True
                break
            token_lps.append(float(logp[0, tid].item()))
        if token_lps and any(not math.isfinite(x) for x in token_lps):
            nonfinite = True
        text_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        ans_lp = float(sum(token_lps)) if token_lps else float("nan")
        mean_lp = float(sum(token_lps) / len(token_lps)) if token_lps else float("nan")
        return {
            "text": text_out,
            "generated_token_ids": ids,
            "token_logprobs": token_lps,
            "answer_logprob": ans_lp,
            "mean_logprob": mean_lp,
            "input_tokens": input_len,
            "output_tokens": len(ids),
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
                "attn_implementation": self.attn_implementation,
                "use_cache": self.generate_use_cache,
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


def make_client(
    *,
    mock: bool,
    model_id: str,
    revision: str | None,
) -> Any:
    if mock:
        return MockV3Client(model_id=f"mock:{model_id}")
    return TransformersChatClient(model_id=model_id, revision=revision)


def run_problem(client: Any, problem: dict[str, Any]) -> dict[str, Any]:
    out = client.generate(problem["prompt"])
    parsed_ans, _conf = parse_response(out["text"])
    target = int(problem.get("target", problem["product"]))
    correct = parsed_ans is not None and parsed_ans == target
    return {
        "problem_id": problem["problem_id"],
        "pool": problem["pool"],
        "family": problem.get("family"),
        "level": int(problem["level"]),
        "a": int(problem["a"]),
        "b": int(problem["b"]),
        "op": problem.get("op"),
        "target": target,
        "prompt": problem["prompt"],
        "raw_response": out["text"],
        "parsed_answer": parsed_ans,
        "correct": bool(correct),
        "generated_token_ids": list(out.get("generated_token_ids") or []),
        "token_logprobs": [float(x) for x in (out.get("token_logprobs") or [])],
        "answer_logprob": float(out.get("answer_logprob")),
        "mean_logprob": float(out.get("mean_logprob")),
        "input_tokens": int(out["input_tokens"]),
        "output_tokens": int(out["output_tokens"]),
        "float32_rerun": bool(out.get("float32_rerun", False)),
        "nonfinite_detected": bool(out.get("nonfinite_detected", False)),
        "dtype_used": out.get("dtype_used"),
        "request_params": out.get("request_params", {}),
        "latency_s": float(out.get("latency_s", 0.0)),
    }


# Defaults used by S1 / S2 collectors
DEFAULT_S1 = (QWEN_MODEL_ID, QWEN_REVISION)
DEFAULT_S2 = (PHI_MODEL_ID, None)
