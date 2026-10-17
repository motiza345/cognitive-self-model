"""M50 generation: pilot and main collection with a resumable cache, hidden states and a manifest.

Protocol: docs/M50_PREREG.md. Runbook: docs/M50_RUN.md.

  python -m scripts.m50_generate --stage pilot --plan-only     # write the pilot plan
  python -m scripts.m50_generate --stage pilot --smoke         # check the model interface on 8 problems
  python -m scripts.m50_generate --stage pilot                 # generate the pilot (resumable)
  python -m scripts.m50_pilot                                  # offline: levels and gates
  python -m scripts.m50_generate --stage main --plan-only      # write the pool plan (needs passing gates)
  python -m scripts.m50_generate --stage main                  # generate CAL and TEST (resumable)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from scripts.m50_common import (CAL_PER_SCOPE, HIDDEN_LAYER, MAX_NEW_TOKENS, MODEL_ID, MODEL_REVISION, PILOT_PER_LEVEL,
                                ROOT, SEED_PILOT, SEED_POOLS, TASK_LEVELS, earlier_exclusions, git_commit, git_dirty,
                                is_correct, make_pool, parse_answer, payload_key, sha256_file)

PILOT_DIR = ROOT / "reports" / "m50_pilot"
MAIN_DIR = ROOT / "reports" / "m50_raw"


# ----------------------------------------------------------------------------- plans
def _exclusion_keys(problems: list[dict]) -> set:
    keys = set()
    for p in problems:
        keys.add(payload_key(p["task"], p["payload"]))
        if p["task"] != "sort_n":
            keys.add(tuple(int(x) for x in p["payload"]))
    return keys


def build_pilot_plan(exclude: set) -> list[dict]:
    plan = []
    for task, levels in TASK_LEVELS.items():
        for level in levels:
            plan += make_pool(task, level, "pilot", PILOT_PER_LEVEL, SEED_PILOT, exclude)
    return plan


def build_main_plan(levels: dict, test_per_scope: int, exclude: set) -> list[dict]:
    plan = []
    for task, info in levels["chosen"].items():
        for level in info["levels"]:
            cal = make_pool(task, level, "cal", CAL_PER_SCOPE, SEED_POOLS, exclude)
            test = make_pool(task, level, "test", test_per_scope, SEED_POOLS, exclude | _exclusion_keys(cal))
            plan += cal + test
    return plan


def plan_hash(plan: list[dict]) -> str:
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- generators
class MockGenerator:
    """Deterministic stand-in for tests and dry runs. `solve(prompt) -> (text, token_logprobs)`."""

    def __init__(self, solve, dim: int = 8, with_hidden: bool = True):
        self.solve, self.dim, self.with_hidden = solve, dim, with_hidden
        self.batch_sizes: list[int] = []

    def generate(self, prompts: list[str]) -> list[dict]:
        self.batch_sizes.append(len(prompts))
        out = []
        for prompt in prompts:
            text, lps = self.solve(prompt)
            seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            out.append({
                "text": text, "token_ids": list(range(len(lps))), "token_logprobs": list(lps),
                "h_prompt": rng.normal(size=self.dim).astype(np.float16) if self.with_hidden else None,
                "h_answer": rng.normal(size=self.dim).astype(np.float16) if self.with_hidden else None,
            })
        return out


class HFGenerator:
    """Greedy generation with Hugging Face transformers; stores per-token log probabilities (end-of-sequence
    excluded), the hidden state after the last prompt token at `HIDDEN_LAYER`, and the mean last-layer hidden
    state over the generated tokens."""

    def __init__(self, model_id: str = MODEL_ID, revision: str = MODEL_REVISION, device: str = "cuda",
                 dtype: str = "float16", hidden_layer: int = HIDDEN_LAYER):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.device, self.dtype_name, self.hidden_layer = torch, device, dtype, hidden_layer
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:  # `torch_dtype` is what the earlier Colab runs used; newer versions may prefer `dtype`
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=getattr(torch, dtype))
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
        self.model = model.to(device).eval()
        commit = getattr(self.model.config, "_commit_hash", None)
        if commit and commit != revision:
            raise RuntimeError(f"loaded revision {commit} differs from the pinned revision {revision}")
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.eos_token_id}
        self.eos_ids.discard(None)

    def generate(self, prompts: list[str]) -> list[dict]:
        torch = self.torch
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None,
                                      top_k=None, output_scores=True, output_hidden_states=True,
                                      return_dict_in_generate=True, pad_token_id=self.tok.pad_token_id)
            prompt_len = enc["input_ids"].shape[1]
            seqs = out.sequences[:, prompt_len:]
            steps = len(out.scores)
            tok_lp = torch.zeros(seqs.shape[0], steps, dtype=torch.float32)
            for t in range(steps):
                logp = torch.log_softmax(out.scores[t].float(), dim=-1)
                tok_lp[:, t] = logp.gather(-1, seqs[:, t:t + 1]).squeeze(-1).cpu()
            h_prompt = out.hidden_states[0][self.hidden_layer][:, -1, :].float().cpu().numpy()
            last = torch.stack([out.hidden_states[t][-1][:, -1, :] for t in range(steps)], dim=1).float().cpu().numpy()
        results = []
        for i in range(seqs.shape[0]):
            ids = seqs[i].tolist()
            n = steps
            for t, token in enumerate(ids[:steps]):
                if token in self.eos_ids:
                    n = t
                    break
            text = self.tok.decode(ids[:n], skip_special_tokens=True)
            results.append({
                "text": text, "token_ids": ids[:n], "token_logprobs": [float(x) for x in tok_lp[i, :n].tolist()],
                "h_prompt": h_prompt[i].astype(np.float16),
                "h_answer": (last[i, :n].mean(axis=0) if n > 0 else np.zeros(last.shape[-1])).astype(np.float16),
            })
        return results


# ----------------------------------------------------------------------------- collection
def load_done(cache: Path) -> set:
    if not cache.exists():
        return set()
    return {json.loads(line)["problem_id"] for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()}


def run_generation(plan: list[dict], generator, out_dir: Path, batch: int = 16, hidden: bool = True,
                   log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "responses.jsonl"
    hidden_dir = out_dir / "hidden"
    if hidden:
        hidden_dir.mkdir(exist_ok=True)
    done = load_done(cache)
    todo = [p for p in plan if p["problem_id"] not in done]
    log(f"{len(plan)} planned, {len(done)} already in the cache, {len(todo)} to generate")
    chunk_index = len(list(hidden_dir.glob("chunk_*.npz"))) if hidden else 0
    started = time.time()
    for start in range(0, len(todo), batch):
        group = todo[start:start + batch]
        t0 = time.time()
        results = generator.generate([p["prompt"] for p in group])
        per_item = (time.time() - t0) / len(group)
        rows = []
        for p, res in zip(group, results):
            lps = res["token_logprobs"]
            finite = all(math.isfinite(x) for x in lps)
            rows.append({
                "problem_id": p["problem_id"], "task": p["task"], "level": p["level"], "pool": p["pool"], "scope": p["scope"],
                "payload": p["payload"], "prompt": p["prompt"], "text": res["text"],
                "parsed": parse_answer(p["task"], res["text"]), "correct": bool(is_correct(p["task"], p["payload"], res["text"])),
                "token_ids": res["token_ids"], "token_logprobs": lps, "n_tokens": len(lps),
                "answer_logprob": float(sum(lps)) if finite else float("-inf"),
                "mean_logprob": float(sum(lps) / len(lps)) if (finite and lps) else float("-inf"),
                "nonfinite": not finite, "latency_s": per_item,
            })
        if hidden:
            np.savez_compressed(hidden_dir / f"chunk_{chunk_index:05d}.npz", ids=np.array([p["problem_id"] for p in group]),
                                h_prompt=np.stack([r["h_prompt"] for r in results]),
                                h_answer=np.stack([r["h_answer"] for r in results]))
            chunk_index += 1
        with open(cache, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        log(f"  {min(start + batch, len(todo))}/{len(todo)}  ({per_item:.2f} s per item)")
    return {"generated": len(todo), "wall_s": time.time() - started}


def write_manifest(stage: str, plan: list[dict], out_dir: Path, generator_info: dict, batch: int) -> dict:
    cache = out_dir / "responses.jsonl"
    rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunk_hashes = {p.name: sha256_file(p) for p in sorted((out_dir / "hidden").glob("chunk_*.npz"))} if (out_dir / "hidden").exists() else {}
    manifest = {
        "stage": stage, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "batch_size": batch,
        "plan_sha256": plan_hash(plan), "n_planned": len(plan), "n_rows": len(rows),
        "cache_sha256": sha256_file(cache), "hidden_chunks_sha256": chunk_hashes,
        "hidden_chunks_combined_sha256": hashlib.sha256("".join(chunk_hashes.values()).encode()).hexdigest() if chunk_hashes else None,
        "nonfinite_rows": sum(1 for r in rows if r["nonfinite"]), "runtime_s_sum_latency": sum(r["latency_s"] for r in rows),
        "git_commit": git_commit(), "git_dirty": git_dirty(), "python": platform.python_version(), "numpy": np.__version__,
        "seeds": {"pilot": SEED_PILOT, "pools": SEED_POOLS}, "hidden_layer": HIDDEN_LAYER, "max_new_tokens": MAX_NEW_TOKENS,
        **generator_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def generator_info(generator) -> dict:
    info = {"generator": type(generator).__name__}
    if isinstance(generator, HFGenerator):
        import torch
        import transformers
        info.update({"device": generator.device, "dtype": generator.dtype_name, "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    return info


def smoke(generator, plan: list[dict], log=print) -> bool:
    """Structural checks of the model interface; batch sensitivity is reported as a warning."""
    problems = plan[:8]
    batch = generator.generate([p["prompt"] for p in problems])
    ok = True
    for p, r in zip(problems, batch):
        lps = r["token_logprobs"]
        if not r["text"] or not lps or not all(math.isfinite(x) for x in lps):
            ok = False
            log(f"  STRUCTURE PROBLEM for {p['problem_id']}: text={r['text']!r} n_logprobs={len(lps)}")
    log("  first outputs:", [(p["task"], p["level"], r["text"][:30], round(sum(r["token_logprobs"]), 4)) for p, r in zip(problems[:3], batch[:3])])
    if batch[0]["h_prompt"] is not None:
        log("  hidden shapes:", batch[0]["h_prompt"].shape, batch[0]["h_answer"].shape)
    singles = [generator.generate([p["prompt"]])[0] for p in problems[:3]]
    for i, s in enumerate(singles):
        same = s["text"] == batch[i]["text"]
        delta = abs(sum(s["token_logprobs"]) - sum(batch[i]["token_logprobs"]))
        log(f"  batch vs single item {i}: same text={same}, abs diff of summed log probability={delta:.4f}")
        if not same:
            log("  WARNING: output depends on batch size; use a smaller --batch and record it")
    log("SMOKE", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "main"], required=True)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    if git_dirty() and not args.allow_dirty and not args.plan_only:
        print("The working tree is not clean. Commit or stash first (the manifest records the commit).")
        return 2
    exclude = earlier_exclusions()
    if args.stage == "pilot":
        out_dir, plan = PILOT_DIR, build_pilot_plan(exclude)
        plan_path = PILOT_DIR / "pilot_plan.json"
    else:
        out_dir = MAIN_DIR
        levels_path, gates_path = PILOT_DIR / "levels.json", PILOT_DIR / "gates.json"
        if not (levels_path.exists() and gates_path.exists()):
            print("Run the pilot and `python -m scripts.m50_pilot` first.")
            return 2
        levels, gates = json.loads(levels_path.read_text()), json.loads(gates_path.read_text())
        if not gates.get("all_pass"):
            print("A gate failed in the pilot; the pre-registration forbids collection. Label: NOT_INFORMATIVE.")
            return 3
        pilot_plan = json.loads((PILOT_DIR / "pilot_plan.json").read_text())
        plan = build_main_plan(levels, int(gates["test_per_scope_level"]), exclude | _exclusion_keys(pilot_plan))
        plan_path = MAIN_DIR / "pool_plan.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(f"wrote {plan_path} ({len(plan)} problems, sha256 {plan_hash(plan)[:16]}...). Commit and push it before collecting.")
        return 0
    if not plan_path.exists():
        print(f"{plan_path} is missing: run with --plan-only, commit and push it first.")
        return 2
    if json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        print("The committed plan differs from the regenerated plan; aborting.")
        return 2
    generator = HFGenerator(device=args.device)
    if args.smoke:
        return 0 if smoke(generator, plan) else 1
    run_generation(plan, generator, out_dir, batch=args.batch, hidden=(args.stage == "main"))
    manifest = write_manifest(args.stage, plan, out_dir, generator_info(generator), args.batch)
    print("cache sha256:", manifest["cache_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
