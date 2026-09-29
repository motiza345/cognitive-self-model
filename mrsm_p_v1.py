"""MRSM_P_v1: two-hop mechanistic retrieval benchmark.

Goal
----
Create a small Transformer whose task genuinely requires a two-stage mechanism:

    QUERY --(hop 1)--> KEY --(hop 2)--> VALUE

The key/value pairing is randomized per episode, so QUERY -> VALUE cannot be
learned as a fixed lexical association. The discovery stage sees no planted
head identities or training masks.

Scientific status
-----------------
This file is an engineering candidate, not a preregistered MRSM run.
Smoke trains and validates the mechanism on development data only.
Full mode is disabled until MRSM preregistration and budget enforcement.
Discovery is not run. Construction and smoke do not generate or read holdout.

The evaluator may use planted identity for the construction audit.
Discovery itself, when later authorized, must not.
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
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

Head = Tuple[int, int]


@dataclass
class Config:
    seed: int = 1729
    model_seed: int = 1729
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    d_head: int = 16
    d_mlp: int = 128
    n_ctx: int = 20
    d_vocab: int = 128
    n_steps: int = 1600
    batch_size: int = 64
    lr: float = 2e-3
    train_examples: int = 8192
    dev_examples: int = 512
    holdout_examples: int = 512
    n_pairs: int = 4
    n_keys: int = 16
    n_values: int = 16
    planted_a: Head = (0, 0)
    planted_b: Head = (1, 1)
    output_dir: str = "reports/mrsm_p_v1"


@dataclass
class Dataset:
    tokens: torch.Tensor
    targets: torch.Tensor
    query_key: torch.Tensor
    pair_keys: torch.Tensor
    pair_values: torch.Tensor
    query_pos: int


@dataclass
class MechanismAudit:
    intact_accuracy: float
    hop1_only_accuracy: float
    hop2_only_accuracy: float
    both_planted_ablated_accuracy: float
    controls_mean_accuracy: float
    planted_specificity_gap: float
    sequential_signature: bool


@dataclass
class DiscoveryResult:
    head_effects: Dict[str, float]
    ranked_heads: List[str]
    pair_effects: Dict[str, float]
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
# Episode generator
# ---------------------------------------------------------------------------

def make_dataset(cfg: Config, n: int, seed: int) -> Dataset:
    """Generate episodes with randomized key->value permutations.

    Layout:
      [BOS, K0,V0, K1,V1, K2,V2, K3,V3, SEP, QUERY, PAD...]

    Query identifies one of the keys. The value paired with that key is
    randomized independently in each episode. Thus there is no fixed
    QUERY->VALUE token association to memorize.
    """
    g = torch.Generator().manual_seed(seed)
    key_base = 8
    value_base = 32
    query_base = 64
    sep_id = 2
    pad_id = 3
    bos_id = 1

    x = torch.full((n, cfg.n_ctx), pad_id, dtype=torch.long)
    x[:, 0] = bos_id
    keys = torch.empty((n, cfg.n_pairs), dtype=torch.long)
    vals = torch.empty((n, cfg.n_pairs), dtype=torch.long)

    for i in range(n):
        k = torch.randperm(cfg.n_keys, generator=g)[: cfg.n_pairs]
        v = torch.randperm(cfg.n_values, generator=g)[: cfg.n_pairs]
        keys[i] = key_base + k
        vals[i] = value_base + v
        for j in range(cfg.n_pairs):
            x[i, 1 + 2 * j] = keys[i, j]
            x[i, 2 + 2 * j] = vals[i, j]

    x[:, 1 + 2 * cfg.n_pairs] = sep_id
    query_index = torch.randint(cfg.n_pairs, (n,), generator=g)
    query_key = keys.gather(1, query_index[:, None]).squeeze(1)
    query_tok = query_base + (query_key - key_base)
    qpos = 2 + 2 * cfg.n_pairs
    x[:, qpos] = query_tok
    target = vals.gather(1, query_index[:, None]).squeeze(1)
    return Dataset(x, target, query_key, keys, vals, qpos)


def batch_iter(ds: Dataset, batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(ds.tokens.shape[0], generator=g)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        yield ds.tokens[idx], ds.targets[idx]


# ---------------------------------------------------------------------------
# Model and construction constraint
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
    return HookedTransformer(mcfg)


def make_training_mask(model: HookedTransformer, cfg: Config, qpos: int):
    """Constrain only the planted heads during P construction.

    L0H0 can read only key positions. L1H1 can read only value positions.
    This is a construction aid; it is removed before every audit/discovery.
    """
    masks = {}
    key_positions = [1 + 2 * j for j in range(cfg.n_pairs)]
    value_positions = [2 + 2 * j for j in range(cfg.n_pairs)]
    for layer in range(cfg.n_layers):
        name = f"blocks.{layer}.attn.hook_pattern"
        mask = torch.ones((cfg.n_heads, cfg.n_ctx, cfg.n_ctx), dtype=torch.float32)
        if layer == cfg.planted_a[0]:
            mask[cfg.planted_a[1], qpos, :] = 0.0
            mask[cfg.planted_a[1], qpos, key_positions] = 1.0
        if layer == cfg.planted_b[0]:
            # L1H1 consumes the key information already written into the
            # query residual and attends only to VALUE positions.
            mask[cfg.planted_b[1], qpos, :] = 0.0
            mask[cfg.planted_b[1], qpos, value_positions] = 1.0
        masks[name] = mask
    return masks


def masked_pattern_hook(mask: torch.Tensor):
    def hook(pattern: torch.Tensor, hook):
        m = mask.to(pattern.device, pattern.dtype).unsqueeze(0)
        z = pattern * m
        return z / z.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return hook


def planting_score_mask(cfg: Config, qpos: int, block_value_to_key: bool) -> Dict[str, torch.Tensor]:
    """Pre-softmax construction mask.

    The planted heads keep the P_SPEC read restriction. Other heads cannot read
    episode content from the query. Phase 2 also stops value positions from
    copying the adjacent key, which is the bypass that made L0H0 unnecessary.
    The mask is a training hook. Audit and discovery run without it.
    """
    key_positions = [1 + 2 * j for j in range(cfg.n_pairs)]
    value_positions = [2 + 2 * j for j in range(cfg.n_pairs)]
    masks = {}
    for layer in range(cfg.n_layers):
        mask = torch.ones((cfg.n_heads, cfg.n_ctx, cfg.n_ctx), dtype=torch.float32)
        for head in range(cfg.n_heads):
            if (layer, head) == cfg.planted_a:
                mask[head, qpos, :] = 0.0
                mask[head, qpos, key_positions] = 1.0
            elif (layer, head) == cfg.planted_b:
                mask[head, qpos, :] = 0.0
                mask[head, qpos, value_positions] = 1.0
            else:
                mask[head, qpos, :] = 0.0
                mask[head, qpos, qpos] = 1.0
            if block_value_to_key:
                for key_pos in key_positions:
                    for value_pos in value_positions:
                        mask[head, value_pos, key_pos] = 0.0
        masks[f"blocks.{layer}.attn.hook_attn_scores"] = mask
    return masks


def _score_penalty_hook(mask: torch.Tensor, bucket: List[torch.Tensor]):
    def hook(scores: torch.Tensor, hook):
        allowed_flag = mask.to(scores.device, scores.dtype).unsqueeze(0)
        allowed = scores.masked_fill(allowed_flag == 0, -1e9)
        illegal = scores.masked_fill(allowed_flag == 1, -1e9)
        gap = illegal.amax(dim=-1) - allowed.amax(dim=-1)
        bucket.append(torch.relu(gap + 2.0).mean())
        return scores.masked_fill(allowed_flag == 0, torch.finfo(scores.dtype).min)
    return hook


def _train_phase(model, cfg: Config, train: Dataset, steps: int, start_step: int, block_value_to_key: bool, lr: float) -> List[float]:
    masks = planting_score_mask(cfg, train.query_pos, block_value_to_key)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    n_batches = max(1, math.ceil(len(train.tokens) / cfg.batch_size))
    cached_epoch = None
    batches = []
    model.train()
    for step in range(steps):
        absolute = start_step + step
        epoch = absolute // n_batches
        if epoch != cached_epoch:
            cached_epoch = epoch
            batches = list(batch_iter(train, cfg.batch_size, cfg.seed + epoch))
        xb, yb = batches[absolute % len(batches)]
        bucket: List[torch.Tensor] = []
        hooks = [(name, _score_penalty_hook(mask, bucket)) for name, mask in masks.items()]
        logits = model.run_with_hooks(xb, fwd_hooks=hooks)
        ce = torch.nn.functional.cross_entropy(logits[:, train.query_pos, :], yb)
        penalty = torch.stack(bucket).mean()
        loss = ce + 5.0 * penalty
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def train_planted(model: HookedTransformer, cfg: Config, train: Dataset) -> Dict[str, float]:
    """Install L0H0 → L1H1, then remove the training hook before audit.

    The frozen 1600-step post-softmax mask does not leave both planted heads
    necessary once the hook is removed. Phase 1 learns the task. Phase 2 blocks
    the value-to-key copy and retrains so the unmasked model needs both heads.
    Learning rate 1e-3 is the rate that converges; 0.002 fits one batch and
    forgets the rule. Step counts were fixed on development data before any
    holdout read.
    """
    phase1_steps = 2000
    phase2_steps = 1000
    plant_lr = 1e-3
    t0 = time.time()
    phase1 = _train_phase(model, cfg, train, phase1_steps, 0, False, plant_lr)
    phase2 = _train_phase(model, cfg, train, phase2_steps, phase1_steps, True, plant_lr)
    model.eval()
    losses = phase1 + phase2
    return {
        "steps": phase1_steps + phase2_steps,
        "phase1_steps": phase1_steps,
        "phase2_steps": phase2_steps,
        "lr": plant_lr,
        "spec_n_steps": cfg.n_steps,
        "spec_lr": cfg.lr,
        "engineering_correction": "pre_softmax_two_phase_planter",
        "final_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "seconds": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Evaluation/interventions
# ---------------------------------------------------------------------------

def logits_at(model, ds: Dataset, hooks=None):
    with torch.no_grad():
        if hooks:
            return model.run_with_hooks(ds.tokens, fwd_hooks=hooks)[:, ds.query_pos, :]
        return model(ds.tokens)[:, ds.query_pos, :]


def accuracy(model: HookedTransformer, ds: Dataset, hooks=None) -> float:
    logits = logits_at(model, ds, hooks)
    return float((logits.argmax(-1) == ds.targets).float().mean())


def margin(model: HookedTransformer, ds: Dataset, hooks=None) -> float:
    logits = logits_at(model, ds, hooks)
    target = logits.gather(1, ds.targets[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, ds.targets[:, None], -torch.inf)
    return float((target - other.max(-1).values).mean())


def zero_head_hook(head: Head):
    layer, h = head
    def hook(z: torch.Tensor, hook):
        z = z.clone()
        z[:, :, h, :] = 0.0
        return z
    return f"blocks.{layer}.attn.hook_z", hook


def multi_zero_hooks(heads: List[Head]):
    by_layer: Dict[int, List[int]] = {}
    for layer, h in heads:
        by_layer.setdefault(layer, []).append(h)
    out = []
    for layer, hs in by_layer.items():
        def hook(z: torch.Tensor, hook, hs=tuple(hs)):
            z = z.clone()
            for h in hs:
                z[:, :, h, :] = 0.0
            return z
        out.append((f"blocks.{layer}.attn.hook_z", hook))
    return out


def run_mechanism_audit(model: HookedTransformer, ds: Dataset, cfg: Config) -> MechanismAudit:
    a, b = cfg.planted_a, cfg.planted_b
    intact = accuracy(model, ds)
    a_only = accuracy(model, ds, multi_zero_hooks([b]))
    b_only = accuracy(model, ds, multi_zero_hooks([a]))
    both = accuracy(model, ds, multi_zero_hooks([a, b]))

    controls = []
    planted = {a, b}
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            if (l, h) not in planted:
                controls.append(accuracy(model, ds, multi_zero_hooks([(l, h)])))
    controls_mean = float(np.mean(controls)) if controls else float("nan")
    specificity = controls_mean - min(a_only, b_only)

    # A valid sequential signature requires both planted components to matter,
    # while single unrelated-head removal does not reproduce that damage.
    sequential = (
        intact >= 0.95
        and a_only < intact - 0.10
        and b_only < intact - 0.10
        and both <= min(a_only, b_only) + 0.05
        and specificity > 0.05
    )
    return MechanismAudit(intact, a_only, b_only, both, controls_mean, specificity, sequential)


# ---------------------------------------------------------------------------
# Blind discovery: head effects + pair non-additivity on DEV only
# ---------------------------------------------------------------------------

def head_effect(model, ds, head: Head) -> float:
    base = margin(model, ds)
    ab = margin(model, ds, [zero_head_hook(head)])
    return base - ab


def pair_effect(model, ds, a: Head, b: Head) -> float:
    ea = head_effect(model, ds, a)
    eb = head_effect(model, ds, b)
    base = margin(model, ds)
    joint = margin(model, ds, multi_zero_hooks([a, b]))
    return (base - joint) - ea - eb


def discover(model, ds, cfg: Config) -> DiscoveryResult:
    heads = [(l, h) for l in range(cfg.n_layers) for h in range(cfg.n_heads)]
    effects = {f"L{l}H{h}": head_effect(model, ds, (l, h)) for l, h in heads}
    ranked = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)
    pairs = {}
    for i, a in enumerate(heads):
        for b in heads[i + 1 :]:
            key = f"L{a[0]}H{a[1]}+L{b[0]}H{b[1]}"
            pairs[key] = pair_effect(model, ds, a, b)
    ranked_pairs = sorted(pairs, key=lambda k: abs(pairs[k]), reverse=True)
    return DiscoveryResult(effects, ranked, pairs, ranked_pairs)


# ---------------------------------------------------------------------------
# Artifacts and run
# ---------------------------------------------------------------------------

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def run(cfg: Config, mode: str):
    if mode != "smoke":
        raise RuntimeError("only smoke mode may construct; full mode is disabled")

    seed_all(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = make_dataset(cfg, cfg.train_examples, cfg.seed + 100)
    dev = make_dataset(cfg, cfg.dev_examples, cfg.seed + 200)

    model = build_model(cfg)
    train_stats = train_planted(model, cfg, train)

    # Construction audit is on DEV. Smoke does not generate or read holdout.
    audit = run_mechanism_audit(model, dev, cfg)
    status = "SMOKE_ONLY" if audit.sequential_signature else "BLOCKED_CONSTRUCTION_AUDIT"
    result = {
        "benchmark": "MRSM_P_v1",
        "mode": mode,
        "status": status,
        "config": asdict(cfg),
        "construction": {
            "task": "episodic two-hop key/value retrieval",
            "query_value_mapping_randomized_per_episode": True,
            "direct_query_value_association": "not stable across episodes",
            "training_mask_removed_before_audit_and_discovery": True,
            "weights_frozen_before_audit_and_discovery": True,
            "planted_identity_hidden_from_discovery": True,
        },
        "train": train_stats,
        "mechanism_audit_dev": asdict(audit),
        "discovery": None,
        "holdout": {"generated": False, "accessed": False, "accuracy": None, "margin": None},
    }

    model_path = out / "p_model_state.pt"
    torch.save(model.state_dict(), model_path)
    result["model_sha256"] = sha256_file(model_path)
    write_json(out / "mrsm_p_v1_result.json", result)
    return result


def evaluate_saved_checkpoint(cfg: Config, checkpoint: Path) -> Dict[str, object]:
    """Dev-only masked/unmasked ablation of a saved checkpoint. No discovery, no holdout."""
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint)
    seed_all(cfg.seed)
    dev = make_dataset(cfg, cfg.dev_examples, cfg.seed + 200)
    model = build_model(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    masks = make_training_mask(model, cfg, dev.query_pos)
    mask_hooks = [(name, masked_pattern_hook(mask)) for name, mask in masks.items()]
    planted_a, planted_b = cfg.planted_a, cfg.planted_b
    ablations = {
        "intact": [],
        "ablate_A": multi_zero_hooks([planted_a]),
        "ablate_B": multi_zero_hooks([planted_b]),
        "ablate_both": multi_zero_hooks([planted_a, planted_b]),
    }
    accuracies: Dict[str, float] = {}
    for name, ablation_hooks in ablations.items():
        accuracies[f"unmasked_{name}"] = accuracy(model, dev, ablation_hooks or None)
        accuracies[f"masked_{name}"] = accuracy(model, dev, [*mask_hooks, *ablation_hooks])

    result: Dict[str, object] = {
        "benchmark": "MRSM_P_v1",
        "mode": "checkpoint-audit",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "dev_seed": cfg.seed + 200,
        "dev_examples": cfg.dev_examples,
        "planted_A": list(planted_a),
        "planted_B": list(planted_b),
        "discovery_run": False,
        "holdout": {"generated": False, "accessed": False},
        "accuracies": accuracies,
    }
    out = Path(cfg.output_dir)
    write_json(out / "checkpoint_mask_audit.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full", "checkpoint-audit"], default="smoke")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--checkpoint", default="reports/mrsm_p_v1/p_model_state.pt")
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

    t0 = time.time()
    if args.mode == "checkpoint-audit":
        result = evaluate_saved_checkpoint(cfg, Path(args.checkpoint))
        print(json.dumps({
            "benchmark": "MRSM_P_v1",
            "status": "CHECKPOINT_AUDIT",
            "checkpoint_sha256": result["checkpoint_sha256"],
            "accuracies": result["accuracies"],
            "holdout_generated": False,
            "discovery_run": False,
            "seconds_total": round(time.time() - t0, 2),
            "result": str(Path(cfg.output_dir) / "checkpoint_mask_audit.json"),
        }, indent=2))
        return

    result = run(cfg, args.mode)
    print(json.dumps({
        "benchmark": "MRSM_P_v1",
        "status": result["status"],
        "dev": result["mechanism_audit_dev"],
        "holdout_generated": result["holdout"]["generated"],
        "seconds_total": round(time.time() - t0, 2),
        "result": str(Path(cfg.output_dir) / "mrsm_p_v1_result.json"),
    }, indent=2))
    if result["status"] == "BLOCKED_CONSTRUCTION_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
