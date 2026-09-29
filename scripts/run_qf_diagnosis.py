"""Run the QF diagnostic. This does not rescore MRSM and does not edit the frozen result."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cognitive_self_model.m22_1.prompts import frozen_prompts, prompts_by_role
from src.mrsm.p_data import sha256_file
from src.mrsm.qf_localize import (
    FROZEN_HOLDOUT_HASH,
    FROZEN_HOLDOUT_N,
    FROZEN_HOLDOUT_SEED,
    FROZEN_P_FREEZE,
    FROZEN_REVISION,
    SCOPE,
    assign_partitions,
    build_payload,
    dump_json,
    effect_floor,
    fit_core,
    intervention_signal_row,
    loo_scores,
    partition_b2_stability,
    partition_row,
    partitions_are_regime_confounded,
    render_report,
    scores_from_saved,
    h2_buckets,
    subset_means,
    tracked_edge_row,
    write_manifest,
)
from src.mrsm.run_q import PINNED_REVISION, SNAPSHOT, _catalog, _effects, _load_model, _margins

FROZEN_FILES = (
    "configs/mrsm_prereg.yaml",
    "reports/MRSM_FINAL_RESULT.md",
    "artifacts/mrsm/holdout_manifest.json",
    "artifacts/mrsm/q_run_001/result.json",
    "artifacts/mrsm/p_run_001/result.json",
    "control_plane/MRSM_BUDGET.yaml",
    "src/mrsm/run_q.py",
    "src/mrsm/self_model.py",
)


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()


def _hashes() -> dict[str, str]:
    return {rel: sha256_file(_ROOT / rel) for rel in FROZEN_FILES}


def _verify(hashes: dict[str, str]) -> dict[str, object]:
    prereg = yaml.safe_load((_ROOT / "configs/mrsm_prereg.yaml").read_text(encoding="utf-8"))
    final = (_ROOT / "reports/MRSM_FINAL_RESULT.md").read_text(encoding="utf-8")
    holdout = json.loads((_ROOT / "artifacts/mrsm/holdout_manifest.json").read_text(encoding="utf-8"))
    q_result = json.loads((_ROOT / "artifacts/mrsm/q_run_001/result.json").read_text(encoding="utf-8"))
    p_result = json.loads((_ROOT / "artifacts/mrsm/p_run_001/result.json").read_text(encoding="utf-8"))
    checks = {
        "prereg_status": prereg.get("status") == "PREREGISTERED",
        "prereg_revision": prereg["model"]["arm_q_revision"] == FROZEN_REVISION,
        "prereg_holdout_seed": prereg["seeds"]["holdout_seed"] == FROZEN_HOLDOUT_SEED,
        "final_result_mentions_redefine_scale": "REDEFINE_SCALE" in final,
        "final_result_mentions_p_freeze": FROZEN_P_FREEZE in final,
        "holdout_hash": holdout.get("hash") == FROZEN_HOLDOUT_HASH,
        "holdout_n": holdout.get("n") == FROZEN_HOLDOUT_N,
        "q_revision": q_result.get("revision") == FROZEN_REVISION,
        "q_gates": q_result.get("gates") == {"H1": "FAIL", "H2": "FAIL", "H3": "PASS", "H4": "FAIL"},
        "q_leakage": q_result.get("leakage") == "PASS",
        "p_gates": p_result.get("gates") == {"H1": "PASS", "H2": "PASS", "H3": "PASS", "H4": "PASS"},
        "snapshot_directory": SNAPSHOT.is_dir(),
        "pinned_constant": PINNED_REVISION == FROZEN_REVISION,
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError(f"frozen verification failed: {failed}")
    return {"checks": checks, "file_sha256": hashes}


def _environment(hashes: dict[str, str], weight_sha256: str) -> dict[str, str]:
    import torch
    from importlib.metadata import version

    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_status": _git(["status", "--short"]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.cuda.is_available()),
        "transformer_lens": version("transformer_lens"),
        "transformers": version("transformers"),
        "model_revision": FROZEN_REVISION,
        "device": "cpu",
        "weight_sha256": weight_sha256,
        "prereg_sha256": hashes["configs/mrsm_prereg.yaml"],
        "final_result_sha256": hashes["reports/MRSM_FINAL_RESULT.md"],
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    before = _hashes()
    verification = _verify(before)
    prereg = yaml.safe_load((_ROOT / "configs/mrsm_prereg.yaml").read_text(encoding="utf-8"))
    min_abs = float(prereg["identifiability"]["min_abs_interaction"])
    min_gap = float(prereg["identifiability"]["min_relative_gap_vs_second"])
    evidence = json.loads((_ROOT / "artifacts/mrsm/q_run_001/evidence.json").read_text(encoding="utf-8"))
    predictions = json.loads((_ROOT / "artifacts/mrsm/q_run_001/predictions.json").read_text(encoding="utf-8"))
    saved = scores_from_saved(evidence["records"], predictions["baselines"]["B2"])
    buckets = h2_buckets(evidence["records"], predictions["baselines"]["B2"])

    manifest = json.loads((_ROOT / "artifacts/mrsm/q_run_001/artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != FROZEN_REVISION or manifest.get("substitution") is not False:
        raise RuntimeError("Q artifact manifest does not match the pinned revision")
    weight_path = SNAPSHOT / "model.safetensors"
    weight_sha = sha256_file(weight_path)
    if weight_sha != manifest.get("weight_sha256"):
        raise RuntimeError("weight hash does not match the frozen Q manifest")

    records = prompts_by_role(frozen_prompts())["discovery"]
    prompt_ids = [record.prompt_id for record in records]
    regimes = [record.regime_id for record in records]
    texts = [record.text for record in records]
    bins = assign_partitions(prompt_ids)
    confounded = partitions_are_regime_confounded(
        [{"prompt_id": record.prompt_id, "regime_id": record.regime_id} for record in records],
        bins,
    )
    if list(bins) != ["D1", "D2", "D3", "D4"]:
        raise RuntimeError("partition names drifted")

    model = _load_model()
    tokens_path = SNAPSHOT / "config.json"
    from src.cognitive_self_model.m22_1.outcome import resolve_outcome_tokens

    token_ids = resolve_outcome_tokens(model.tokenizer, " yes", " no")
    positive_id = int(token_ids["positive_token_id"])
    negative_id = int(token_ids["negative_token_id"])
    catalog = _catalog()
    _singles, _joints, per_example, _base = _effects(model, texts, catalog, positive_id, negative_id)
    if len(next(iter(per_example.values()))) != len(prompt_ids):
        raise RuntimeError("per-example width does not match discovery prompts")

    index = {prompt_id: position for position, prompt_id in enumerate(prompt_ids)}
    qf1_rows = []
    tracked = []
    for name in ("D1", "D2", "D3", "D4"):
        indices = [index[prompt_id] for prompt_id in bins[name]]
        singles, joints = subset_means(per_example, indices)
        core = fit_core(singles, joints, min_abs=min_abs, min_gap=min_gap)
        if core.scope != SCOPE:
            raise RuntimeError("scope drifted")
        qf1_rows.append(partition_row(name, core, bins[name]))
        tracked.append(tracked_edge_row(name, core))

    null_drops: dict[str, list[float]] = {}
    base = _margins(model, texts, [], positive_id, negative_id)
    for item in catalog:
        nulled = dict(item)
        nulled["alpha"] = 0.0
        treated = _margins(model, texts, [nulled], positive_id, negative_id)
        null_drops[item["name"]] = [b - t for b, t in zip(base, treated)]
    null_abs = max(abs(value) for drops in null_drops.values() for value in drops)
    floor = effect_floor(null_abs)
    signal_rows = []
    for name in [item["name"] for item in catalog]:
        signal_rows.append(intervention_signal_row(
            name,
            per_example[f"single:{name}"],
            null_drops[name],
            regimes,
            floor,
        ))

    loo = loo_scores(per_example, min_abs=min_abs, min_gap=min_gap)
    b2_parts = partition_b2_stability(per_example, prompt_ids, bins)
    environment = _environment(before, weight_sha)
    payload = build_payload(
        verification=verification,
        reproducibility=environment,
        qf1_rows=qf1_rows,
        tracked_rows=tracked,
        confounded=confounded,
        signal_rows=signal_rows,
        signal_floor=floor,
        loo=loo,
        saved=saved,
        buckets=buckets,
        b2_partitions=b2_parts,
    )
    out = _ROOT / "artifacts" / "q_failure"
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "qf1_discovery_stability.json", payload["qf1"])
    dump_json(out / "qf2_intervention_signal.json", payload["qf2"])
    dump_json(out / "qf3_representation_audit.json", payload["qf3"])
    dump_json(out / "qf4_oracle_prediction.json", payload["qf4"])
    dump_json(out / "qf5_h2_decomposition.json", payload["qf5"])
    dump_json(out / "qf6_baseline_ceiling.json", payload["qf6"])
    dump_json(out / "diagnosis.json", {
        "primary_bottleneck": payload["primary_bottleneck"],
        "next_intervention": payload["next_intervention"],
        "matrix": payload["matrix"],
        "scientific_status": payload["scientific_status"],
        "leakage": payload["leakage"],
    })
    write_manifest(out)
    report = render_report(payload)
    (_ROOT / "reports" / "Q_FAILURE_ANALYSIS.md").write_text(report, encoding="utf-8")
    after = _hashes()
    if after != before:
        changed = [rel for rel in FROZEN_FILES if after[rel] != before[rel]]
        raise RuntimeError(f"frozen files changed during QF: {changed}")
    print(json.dumps({
        "phase_status": payload["phase_status"],
        "primary_bottleneck": payload["primary_bottleneck"],
        "qf1": payload["qf1"]["label"],
        "qf2": payload["qf2"]["label"],
        "qf3": payload["qf3"]["status"],
        "qf4": payload["qf4"]["status"],
        "qf5": payload["qf5"]["status"],
        "qf6": payload["qf6"]["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
