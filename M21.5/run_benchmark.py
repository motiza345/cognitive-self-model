"""Reproducible entry point for M21.5-env-v1.0.

The module path is a script because the directory name contains a dot.
From the repository root:

    python M21.5/run_benchmark.py --config M21.5/configs/m21_5_v1.yaml --mode primary
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from agents import ABLATION_NAMES, PRIMARY_AGENTS  # noqa: E402
from environment.env import load_ground_truth  # noqa: E402
from environment.state import ACTIONS, BENCHMARK_VERSION, PRIMARY_SEEDS  # noqa: E402
from evaluation.experiment import (  # noqa: E402
    iter_seed_runs,
    replay_fingerprint,
    run_agent,
    run_condition,
    write_run,
)
from evaluation.leakage_audit import audit_saved_run, scan_agent_sources  # noqa: E402
from evaluation.report import build_report_document, decide_claim, render_markdown  # noqa: E402
from evaluation.scoring import score_ablations, score_condition  # noqa: E402
from agents import build_agent  # noqa: E402
from environment.env import BenchmarkEnvironment  # noqa: E402

RESULTS = PACKAGE / "results"
AUDITS = RESULTS / "audits"
MANIFEST_PATH = RESULTS / "manifest" / "freeze_manifest.json"
REPORT_NAMES = ("M21.5_v1_report.json", "M21.5_v1_report.md")
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".md"}
MODES = {
    "primary": {
        "schedule": "primary",
        "regime": "primary",
        "agents": PRIMARY_AGENTS,
        "dynamic": True,
    },
    "static": {
        "schedule": "static",
        "regime": "primary",
        "agents": PRIMARY_AGENTS,
        "dynamic": False,
    },
    "ood": {
        "schedule": "primary",
        "regime": "ood",
        "agents": PRIMARY_AGENTS,
        "dynamic": True,
    },
    "ablation": {
        "schedule": "primary",
        "regime": "primary",
        "agents": ABLATION_NAMES,
        "dynamic": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["primary", "static", "ood", "ablation", "report", "integration"],
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ground_truth = load_ground_truth()
    _assert_config_matches_ground_truth(config, ground_truth)
    if args.mode == "integration":
        _run_integration(config, ground_truth)
        return
    if args.mode == "report":
        _run_report(config, config_path, ground_truth)
        return
    _run_mode(args.mode, config, config_path, ground_truth)


def _run_integration(config: dict, ground_truth: dict) -> None:
    destination = PACKAGE / "logs" / "integration"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    hparams = _hparams(config)
    env = BenchmarkEnvironment(ground_truth, PRIMARY_SEEDS[0], "primary", "primary")
    for name in PRIMARY_AGENTS:
        result = run_agent(build_agent(name, hparams), env, max_episodes=5)
        if result["leakage"]:
            raise SystemExit("integration leakage: " + "; ".join(result["leakage"]))
        write_run(destination / name, result)
    (destination / "README.json").write_text(
        json.dumps(
            {
                "scientific_result": False,
                "purpose": "pipeline check only",
                "episodes": 5,
                "seed": PRIMARY_SEEDS[0],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("INTEGRATION_COMPLETE scientific_result=false")


def _run_mode(mode: str, config: dict, config_path: Path, ground_truth: dict) -> None:
    spec = MODES[mode]
    final = RESULTS / "raw" / mode
    if final.exists():
        raise SystemExit(f"REFUSING_OVERWRITE {final}")
    if mode == "primary":
        if MANIFEST_PATH.exists():
            raise SystemExit(f"REFUSING_OVERWRITE {MANIFEST_PATH}")
        dirty = _uncommitted_sources()
        if dirty:
            raise SystemExit("STOP: M21.5 sources are not committed:\n" + "\n".join(dirty))
        _write_manifest(config, config_path)
    elif not MANIFEST_PATH.exists():
        raise SystemExit("STOP: freeze manifest is missing; run primary first")
    partial = RESULTS / "raw" / f"{mode}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    leakage = run_condition(
        ground_truth,
        _hparams(config),
        list(config["seeds"]),
        spec["agents"],
        spec["schedule"],
        spec["regime"],
        partial,
    )
    partial.rename(final)
    _write_json(AUDITS / f"{mode}_raw_hashes.json", _hash_tree(final))
    _write_json(AUDITS / f"{mode}_runtime_leakage.json", {"hits": leakage})
    print(f"{mode.upper()}_COMPLETE seeds={len(config['seeds'])} leakage_hits={len(leakage)}")


def _run_report(config: dict, config_path: Path, ground_truth: dict) -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("STOP: freeze manifest is missing")
    for name in REPORT_NAMES:
        if (RESULTS / name).exists() or (PACKAGE / name).exists():
            raise SystemExit(f"REFUSING_OVERWRITE {name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    choices = _hparams(config)
    loaded = {}
    for mode, spec in MODES.items():
        directory = RESULTS / "raw" / mode
        if not directory.exists():
            raise SystemExit(f"STOP: missing raw logs for {mode}")
        loaded[mode] = list(iter_seed_runs(directory))
        loaded[mode] = [
            {"seed": seed, "fingerprint": fingerprint, "agents": agents}
            for seed, fingerprint, agents in loaded[mode]
        ]
        loaded[mode].sort(key=lambda row: row["seed"])
    primary = score_condition(loaded["primary"], choices, True)
    static_summary = score_condition(loaded["static"], choices, False)
    ood_summary = score_condition(loaded["ood"], choices, True)
    ablations = score_ablations(loaded["ablation"], primary, choices)
    integrity = _integrity(config, config_path, ground_truth, manifest)
    claim = decide_claim(primary, static_summary, ood_summary, ablations, integrity)
    public_integrity = {
        "leakage": "PASS" if integrity["leakage_passed"] else "FAIL",
        "compute_fairness": "PASS" if not integrity["major_compute_confound"] else "FAIL",
        "reproducibility": "PASS" if integrity["fingerprints_match"] else "FAIL",
        "historical_immutability": "PASS"
        if integrity["raw_hashes_match"] and integrity["source_hashes_match"]
        else "FAIL",
        "compute_note": config["compute"]["asymmetry_note"].strip(),
    }
    integrity["public"] = public_integrity
    document = build_report_document(
        config, manifest, primary, static_summary, ood_summary, ablations, integrity, claim
    )
    _write_json(RESULTS / "metrics" / "primary.json", primary)
    _write_json(RESULTS / "metrics" / "static.json", static_summary)
    _write_json(RESULTS / "metrics" / "ood.json", ood_summary)
    _write_json(RESULTS / "metrics" / "ablation.json", ablations)
    _write_json(RESULTS / "statistics" / "primary.json", primary["comparisons"])
    _write_json(RESULTS / "statistics" / "static.json", static_summary["comparisons"])
    _write_json(RESULTS / "statistics" / "ood.json", ood_summary["comparisons"])
    _write_json(AUDITS / "leakage.json", integrity["leakage_report"])
    _write_json(AUDITS / "immutability.json", integrity["immutability_report"])
    _write_json(
        AUDITS / "compute.json",
        {
            "model_revision": config["compute"]["model_revision"],
            "prompt_revision": config["compute"]["prompt_revision"],
            "max_model_calls_per_episode": config["compute"]["max_model_calls_per_episode"],
            "max_output_tokens": config["compute"]["max_output_tokens"],
            "max_memory_tokens": config["compute"]["max_memory_tokens"],
            "temperature": config["compute"]["temperature"],
            "sampling_parameters": config["compute"]["sampling_parameters"],
            "major_compute_confound": config["compute"]["major_compute_confound"],
            "asymmetry_note": config["compute"]["asymmetry_note"].strip(),
        },
    )
    text = render_markdown(document)
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    for directory in (RESULTS, PACKAGE):
        (directory / "M21.5_v1_report.json").write_text(encoded, encoding="utf-8")
        (directory / "M21.5_v1_report.md").write_text(text, encoding="utf-8")
    print(claim["verdict"])
    print(f"DeltaU {primary['comparisons']['B3-BestSimple']['mean']}")
    print(f"CI {primary['comparisons']['B3-BestSimple']['low']} {primary['comparisons']['B3-BestSimple']['high']}")


def _integrity(config, config_path, ground_truth, manifest) -> dict:
    leakage_hits = []
    leakage_hits.extend(scan_agent_sources(PACKAGE / "agents"))
    for mode in MODES:
        runtime_path = AUDITS / f"{mode}_runtime_leakage.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        leakage_hits.extend(runtime["hits"])
        for seed in config["seeds"]:
            leakage_hits.extend(audit_saved_run(RESULTS / "raw" / mode / f"seed_{seed}", int(seed)))
    raw_problems = []
    for mode in MODES:
        saved = json.loads((AUDITS / f"{mode}_raw_hashes.json").read_text(encoding="utf-8"))
        current = _hash_tree(RESULTS / "raw" / mode)
        if current["files"] != saved["files"]:
            raw_problems.append(mode)
    source_digest, source_files = _source_hashes()
    source_match = source_files == manifest["source_files"] and source_digest == manifest["source_tree_sha256"]
    fingerprint_problems = []
    for mode, spec in MODES.items():
        for seed, fingerprint, _agents in iter_seed_runs(RESULTS / "raw" / mode):
            expected = replay_fingerprint(
                ground_truth, seed, spec["schedule"], spec["regime"]
            )
            if expected != fingerprint:
                fingerprint_problems.append(f"{mode}:{seed}")
    environment_match = _sha256(PACKAGE / "environment" / "ground_truth.json") == manifest["environment_hash"]
    config_match = _sha256(config_path) == manifest["config_hash"]
    leakage_report = {
        "passed": not leakage_hits,
        "hits": leakage_hits,
        "checks": {
            "hidden_self_state_in_agent_events": not any("hidden" in hit for hit in leakage_hits),
            "ground_truth_file_opened_by_agent": not any("ground_truth" in hit for hit in leakage_hits),
            "future_outcomes": not any("future" in hit for hit in leakage_hits),
            "evaluator_metadata": not any("banned key" in hit or "unexpected" in hit for hit in leakage_hits),
            "environment_seed": not any("environment seed" in hit for hit in leakage_hits),
            "transition_schedule": not any("schedule" in hit for hit in leakage_hits),
            "prompt_or_memory_reveals_hidden_state": not any("hidden state name" in hit for hit in leakage_hits),
        },
    }
    immutability_report = {
        "raw_hashes_match": not raw_problems,
        "raw_mismatches": raw_problems,
        "source_hashes_match": source_match,
        "environment_hash_match": environment_match,
        "config_hash_match": config_match,
        "fingerprints_match": not fingerprint_problems,
        "fingerprint_mismatches": fingerprint_problems,
    }
    return {
        "leakage_passed": leakage_report["passed"],
        "major_compute_confound": bool(config["compute"]["major_compute_confound"]),
        "raw_hashes_match": immutability_report["raw_hashes_match"] and environment_match and config_match,
        "source_hashes_match": source_match,
        "fingerprints_match": immutability_report["fingerprints_match"],
        "leakage_report": leakage_report,
        "immutability_report": immutability_report,
    }


def _write_manifest(config: dict, config_path: Path) -> None:
    source_digest, source_files = _source_hashes()
    agent_versions = {
        "B0": _sha256(PACKAGE / "agents" / "b0_raw.py"),
        "B1": _sha256(PACKAGE / "agents" / "b1_confidence.py"),
        "B2": _sha256(PACKAGE / "agents" / "b2_memory_reflection.py"),
        "B3": _sha256(PACKAGE / "agents" / "b3_self_model.py"),
        "base": _sha256(PACKAGE / "agents" / "base.py"),
    }
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "implementation_version": config["implementation_version"],
        "environment_hash": _sha256(PACKAGE / "environment" / "ground_truth.json"),
        "config_hash": _sha256(config_path),
        "prompt_hash": hashlib.sha256(config["compute"]["prompt_hash_material"].encode("utf-8")).hexdigest(),
        "model_revision": config["compute"]["model_revision"],
        "code_commit": _git_head(),
        "seed_list": list(config["seeds"]),
        "agent_versions": agent_versions,
        "source_tree_sha256": source_digest,
        "source_files": source_files,
        "compute": {
            "base_model": config["compute"]["base_model"],
            "model_revision": config["compute"]["model_revision"],
            "prompt_revision": config["compute"]["prompt_revision"],
            "max_model_calls_per_episode": config["compute"]["max_model_calls_per_episode"],
            "max_output_tokens": config["compute"]["max_output_tokens"],
            "max_memory_tokens": config["compute"]["max_memory_tokens"],
            "temperature": config["compute"]["temperature"],
            "major_compute_confound": config["compute"]["major_compute_confound"],
        },
    }
    _write_json(MANIFEST_PATH, manifest)


def _assert_config_matches_ground_truth(config: dict, ground_truth: dict) -> None:
    if config["benchmark_version"] != BENCHMARK_VERSION:
        raise SystemExit("STOP: config benchmark version drifted")
    if list(config["seeds"]) != list(PRIMARY_SEEDS) or list(ground_truth["seeds"]) != list(PRIMARY_SEEDS):
        raise SystemExit("STOP: seed list drifted")
    if int(config["episodes"]) != 100:
        raise SystemExit("STOP: episode count drifted")
    choices = config["implementation_choices"]
    if list(choices["tie_break"]) != list(ACTIONS):
        raise SystemExit("STOP: tie break drifted from the frozen action order")
    if float(choices["b1_ucb_c"]) != float(choices["b3_ucb_c"]):
        raise SystemExit("STOP: B1 and B3 exploration coefficients differ")
    if list(choices["latent_states"]) != ["S0", "S1", "S2"]:
        raise SystemExit("STOP: latent hypothesis labels drifted")
    if choices["agent_reset_seed"] != 0:
        raise SystemExit("STOP: agent reset seed would expose a varying environment seed")


def _hparams(config: dict) -> dict:
    return dict(config["implementation_choices"])


def _source_hashes() -> tuple[str, dict]:
    files = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(PACKAGE)
        if "results" in relative.parts or "logs" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.name in REPORT_NAMES:
            continue
        files.append(path)
    digest = hashlib.sha256()
    listing = {}
    for path in files:
        data = path.read_bytes()
        relative = str(path.relative_to(PACKAGE))
        listing[relative] = hashlib.sha256(data).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
    return digest.hexdigest(), listing


def _hash_tree(directory: Path) -> dict:
    files = {}
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        files[str(path.relative_to(directory))] = _sha256(path)
    return {"files": files}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE.parent, text=True).strip()


def _uncommitted_sources() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "M21.5"],
        cwd=PACKAGE.parent,
        text=True,
    )
    bad = []
    for line in output.splitlines():
        path = line[3:].strip().strip('"')
        if path.startswith("M21.5/results/") or path.startswith("M21.5/logs/"):
            continue
        if path.endswith("M21.5_v1_report.json") or path.endswith("M21.5_v1_report.md"):
            continue
        bad.append(line)
    return bad


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
