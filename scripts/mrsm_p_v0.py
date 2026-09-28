"""MRSM_P_v0: Minimal Real Self-Model planted-transformer benchmark.

This is a standalone first implementation intended to be copied into the project
and adapted only where the existing discovery adapter has a different API.

Modes:
  --mode smoke   : build/train/validate P on development data; does not generate or read holdout
  --mode full    : disabled pending MRSM preregistration and budget enforcement

Design:
  * small HookedTransformer, CPU
  * associative key->value retrieval task
  * two planted attention heads are constrained during training
  * training constraint is removed before discovery
  * discovery receives no planted-head identities
  * ground truth is retained only by the evaluator

Important: this first version deliberately separates P construction from the
scientific gate thresholds. It reports raw measurements; it does not declare
H1/H2/H3 PASS/FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

Head = Tuple[int, int]


@dataclass
class Config:
    seed: int = 11
    model_seed: int = 11
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16
    d_mlp: int = 128
    n_ctx: int = 12
    d_vocab: int = 64
    n_steps: int = 1200
    batch_size: int = 64
    lr: float = 2e-3
    train_examples: int = 4096
    dev_examples: int = 512
    holdout_examples: int = 512
    max_head_scan: int = 8
    pair_top_k: int = 4
    effect_threshold: float = 0.05
    output_dir: str = "reports/mrsm_p_v0"
    planted_a: Head = (0, 0)
    planted_b: Head = (1, 1)


@dataclass
class Dataset:
    tokens: torch.Tensor
    targets: torch.Tensor
    key_ids: torch.Tensor
    value_ids: torch.Tensor


@dataclass
class DiscoveryResult:
    head_scores: Dict[str, float]
    ranked_heads: List[str]
    pair_scores: Dict[str, float]
    ranked_pairs: List[str]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# ---------------------------------------------------------------------------
# Synthetic retrieval task
# ---------------------------------------------------------------------------

def make_dataset(cfg: Config, n: int, seed: int) -> Dataset:
    """Synthetic two-hop retrieval dataset.

    Token IDs are arranged as pairs (KEY_i, VALUE_i). The query token Q_i
    identifies the key family, while the value token is keyed by a latent
    nonce carried by KEY_i. The intended two-hop path is:

        query -> key/nonce (hop 1) -> value (hop 2)

    The training mask is responsible for constraining which heads may perform
    these two hops. This dataset is deliberately small and synthetic.
    """
    g = torch.Generator().manual_seed(seed)
    n_keys = 8
    # key_i, value_i, query_i occupy disjoint token ranges.
    key = torch.randint(8, 8 + n_keys, (n,), generator=g)
    idx = key - 8
    value = 24 + idx
    query = 40 + idx
    distractor = 8 + torch.randint(0, n_keys, (n,), generator=g)
    x = torch.zeros((n, cfg.n_ctx), dtype=torch.long)
    x[:, 0] = 1      # BOS
    x[:, 1] = key    # key carrying the latent nonce
    x[:, 2] = value  # value keyed by that nonce
    x[:, 3] = distractor + 8
    x[:, 4] = query  # query identifies the key family, not the value token
    if cfg.n_ctx > 5:
        x[:, 5:] = 3
    return Dataset(x, value.clone(), key, value)


def batch_iter(ds: Dataset, batch_size: int, seed: int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(ds.tokens.shape[0], generator=g)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        yield ds.tokens[idx], ds.targets[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: Config) -> HookedTransformer:
    torch.manual_seed(cfg.model_seed)
    mcfg = HookedTransformerConfig(
        n_layers=cfg.n_layers,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_head=cfg.d_head,
        d_mlp=cfg.d_mlp,
        n_ctx=cfg.n_ctx,
        d_vocab=cfg.d_vocab,
        attn_only=False,
        normalization_type="LN",
        device="cpu",
        seed=cfg.model_seed,
    )
    model = HookedTransformer(mcfg)
    model.train()
    return model


# ---------------------------------------------------------------------------
# Training-time mechanism constraint
# ---------------------------------------------------------------------------

def make_training_mask(model: HookedTransformer, cfg: Config) -> Dict[str, torch.Tensor]:
    """Construct a two-hop attention constraint for P construction.

    Hop A (L0.H0): query position 4 may read only the key position 1.
    Hop B (L1.H1): query position 4 may read only the value position 2.

    The mask is a construction constraint, not ground truth supplied to
    discovery. It is removed before any discovery call.
    """
    masks: Dict[str, torch.Tensor] = {}
    for layer in range(cfg.n_layers):
        name = f"blocks.{layer}.attn.hook_pattern"
        mask = torch.ones((cfg.n_heads, cfg.n_ctx, cfg.n_ctx), dtype=torch.float32)
        for h in range(cfg.n_heads):
            if (layer, h) == cfg.planted_a:
                mask[h, 4, :] = 0.0
                mask[h, 4, 1] = 1.0
            elif (layer, h) == cfg.planted_b:
                mask[h, 4, :] = 0.0
                mask[h, 4, 2] = 1.0
        masks[name] = mask
    return masks


def masked_pattern_hook(mask: torch.Tensor):
    def hook(pattern: torch.Tensor, hook) -> torch.Tensor:
        m = mask.to(pattern.device, pattern.dtype).unsqueeze(0)
        z = pattern * m
        denom = z.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return z / denom
    return hook


def train_planted(model: HookedTransformer, cfg: Config, train: Dataset) -> Dict[str, float]:
    masks = make_training_mask(model, cfg)
    hooks = [(name, masked_pattern_hook(mask)) for name, mask in masks.items()]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    start = time.time()
    losses: List[float] = []
    model.train()
    for step in range(cfg.n_steps):
        # Deterministic cycling through shuffled minibatches.
        seed = cfg.seed + step // max(1, math.ceil(len(train.tokens) / cfg.batch_size))
        batches = list(batch_iter(train, cfg.batch_size, seed))
        xb, yb = batches[step % len(batches)]
        logits = model.run_with_hooks(xb, fwd_hooks=hooks)
        loss = torch.nn.functional.cross_entropy(logits[:, 4, :], yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    return {
        "steps": cfg.n_steps,
        "final_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "seconds": time.time() - start,
    }


# ---------------------------------------------------------------------------
# Evaluation and interventions
# ---------------------------------------------------------------------------

def accuracy(model: HookedTransformer, ds: Dataset) -> float:
    with torch.no_grad():
        logits = model(ds.tokens)[:, 4, :]
        pred = logits.argmax(dim=-1)
    return float((pred == ds.targets).float().mean())


def target_margin(model: HookedTransformer, ds: Dataset) -> float:
    with torch.no_grad():
        logits = model(ds.tokens)[:, 4, :]
        target = logits.gather(1, ds.targets[:, None]).squeeze(1)
        wrong = logits.clone()
        wrong.scatter_(1, ds.targets[:, None], -torch.inf)
        other = wrong.max(dim=-1).values
    return float((target - other).mean())


def head_zero_hook(layer: int, head: int):
    def hook(z: torch.Tensor, hook) -> torch.Tensor:
        z = z.clone()
        z[:, :, head, :] = 0.0
        return z
    return hook


def ablate_head(model: HookedTransformer, ds: Dataset, head: Head) -> float:
    layer, h = head
    hook_name = f"blocks.{layer}.attn.hook_z"
    with torch.no_grad():
        logits = model.run_with_hooks(ds.tokens, fwd_hooks=[(hook_name, head_zero_hook(layer, h))])
        pred = logits[:, 4, :].argmax(dim=-1)
    return float((pred == ds.targets).float().mean())


def head_effect(model: HookedTransformer, ds: Dataset, head: Head) -> float:
    base = target_margin(model, ds)
    layer, h = head
    hook_name = f"blocks.{layer}.attn.hook_z"
    with torch.no_grad():
        logits = model.run_with_hooks(ds.tokens, fwd_hooks=[(hook_name, head_zero_hook(layer, h))])
        target = logits[:, 4, :].gather(1, ds.targets[:, None]).squeeze(1)
        wrong = logits[:, 4, :].clone()
        wrong.scatter_(1, ds.targets[:, None], -torch.inf)
        margin = (target - wrong.max(dim=-1).values).mean().item()
    return float(base - margin)


def pair_zero_hook(layer1: int, h1: int, layer2: int, h2: int):
    def hook(z: torch.Tensor, hook) -> torch.Tensor:
        z = z.clone()
        if layer1 == layer2:
            z[:, :, h1, :] = 0.0
            z[:, :, h2, :] = 0.0
        return z
    return hook


# ---------------------------------------------------------------------------
# Blind discovery: adapted from the project's existing GPT-2 head scan
# ---------------------------------------------------------------------------

def discover(model: HookedTransformer, ds: Dataset, cfg: Config) -> DiscoveryResult:
    heads = [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    scores: Dict[str, float] = {}
    for head in heads:
        scores[f"L{head[0]}H{head[1]}"] = head_effect(model, ds, head)
    ranked = sorted(scores, key=lambda k: abs(scores[k]), reverse=True)
    top = ranked[: min(cfg.pair_top_k, len(ranked))]
    pair_scores: Dict[str, float] = {}
    # Pair score is a simple non-additivity diagnostic, not a causal edge claim.
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a = next(h for h in heads if f"L{h[0]}H{h[1]}" == top[i])
            b = next(h for h in heads if f"L{h[0]}H{h[1]}" == top[j])
            ea = scores[top[i]]
            eb = scores[top[j]]
            # Independent ablations as a conservative v0 synergy proxy.
            if a[0] == b[0]:
                name = f"L{a[0]}H{a[1]}+L{b[0]}H{b[1]}"
                # Two-head hook in one layer.
                hook_name = f"blocks.{a[0]}.attn.hook_z"
                def both(z, hook, h1=a[1], h2=b[1]):
                    z = z.clone(); z[:, :, h1, :] = 0.0; z[:, :, h2, :] = 0.0; return z
                base = target_margin(model, ds)
                with torch.no_grad():
                    logits = model.run_with_hooks(ds.tokens, fwd_hooks=[(hook_name, both)])
                    target = logits[:, 4, :].gather(1, ds.targets[:, None]).squeeze(1)
                    wrong = logits[:, 4, :].clone(); wrong.scatter_(1, ds.targets[:, None], -torch.inf)
                    pair_margin = (target - wrong.max(dim=-1).values).mean().item()
                pair_scores[name] = float(base - pair_margin - ea - eb)
            else:
                pair_scores[f"L{a[0]}H{a[1]}+L{b[0]}H{b[1]}"] = 0.0
    ranked_pairs = sorted(pair_scores, key=lambda k: abs(pair_scores[k]), reverse=True)
    return DiscoveryResult(scores, ranked, pair_scores, ranked_pairs)


# ---------------------------------------------------------------------------
# Ground-truth comparison and artifacts
# ---------------------------------------------------------------------------

def evaluate_discovery(discovery: DiscoveryResult, cfg: Config) -> Dict[str, object]:
    planted = {f"L{cfg.planted_a[0]}H{cfg.planted_a[1]}", f"L{cfg.planted_b[0]}H{cfg.planted_b[1]}"}
    top2 = set(discovery.ranked_heads[:2])
    intersection = len(planted & top2)
    precision = intersection / max(1, len(top2))
    recall = intersection / len(planted)
    return {
        "planted_heads": sorted(planted),
        "top2_discovered_heads": discovery.ranked_heads[:2],
        "intersection": intersection,
        "precision_at_2": precision,
        "recall_at_2": recall,
        "jaccard_at_2": intersection / max(1, len(planted | top2)),
    }


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run(cfg: Config, mode: str) -> Dict[str, object]:
    seed_all(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if mode != "smoke":
        raise RuntimeError("only smoke mode may construct; full mode is disabled")

    train = make_dataset(cfg, cfg.train_examples, cfg.seed + 100)
    dev = make_dataset(cfg, cfg.dev_examples, cfg.seed + 200)

    model = build_model(cfg)
    train_stats = train_planted(model, cfg, train)

    dev_acc = accuracy(model, dev)
    dev_margin = target_margin(model, dev)

    ablations = {}
    for head in [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]:
        ablations[f"L{head[0]}H{head[1]}"] = ablate_head(model, dev, head)

    result: Dict[str, object] = {
        "benchmark": "MRSM_P_v0",
        "mode": mode,
        "status": "SMOKE_ONLY" if mode == "smoke" else "FULL_RUN_CANDIDATE",
        "config": asdict(cfg),
        "construction": {
            "training_mask": "planted heads receive restricted query-position attention; non-planted heads unrestricted",
            "mask_removed_before_discovery": True,
            "weights_frozen_before_discovery": True,
            "ground_truth_hidden_from_discovery": True,
        },
        "train": train_stats,
        "dev": {"accuracy": dev_acc, "margin": dev_margin},
        "dev_single_head_ablation_accuracy": ablations,
        "holdout": {
            "generated": False,
            "accessed": False,
            "accuracy": None,
            "margin": None,
        },
    }

    model_path = out / "p_model_state.pt"
    torch.save(model.state_dict(), model_path)
    result["model_sha256"] = sha256_file(model_path)

    write_json(out / "mrsm_p_v0_result.json", result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.mode == "full":
        print(
            "full mode disabled: pending MRSM preregistration and budget enforcement",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cfg = Config()
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.output_dir:
        cfg.output_dir = args.output_dir

    print("MRSM_P_v0", args.mode)
    print("config:", asdict(cfg))
    t0 = time.time()
    result = run(cfg, args.mode)
    print(json.dumps({
        "status": result["status"],
        "dev_accuracy": result["dev"]["accuracy"],
        "holdout_generated": result["holdout"]["generated"],
        "seconds_total": round(time.time() - t0, 2),
        "result": str(Path(cfg.output_dir) / "mrsm_p_v0_result.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
