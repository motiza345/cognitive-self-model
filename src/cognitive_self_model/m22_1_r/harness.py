"""M22.1-R harness.

The default invocation checks prerequisites and writes an execution record.
It does not call the model. ``execute_forward=True`` is the only path that
loads weights, and that path still refuses an unpinned revision.

Replay tiers are the names in ``configs/m22_1_r_replay.yaml``. This module
does not emit ``REPLAY_FAIL``.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

import yaml

from ..m22_1.runtime_info import git_identity
from .comparator import RESULTS, compare_bundles
from .config import REQUIRED_REVISION, ReplayConfig, load_replay_config
from .hashing import file_sha256
from .record import build_record
from .references import MANIFEST_PATH, verify_reference_manifest
from .strict_loader import ReplayBlockedError, load_pinned_revision
from .weights import (
    EXPECTED_WEIGHT_HASHES_ABSENT,
    default_hub_cache,
    hash_weight_files,
    snapshot_directory,
    weight_identity_mismatch,
)

ENTRYPOINT = "scripts/run_m22_1_r_harness.py"
INVOCATION = "PYTHONPATH=src python3 scripts/run_m22_1_r_harness.py"
EXECUTION_RECORD_PATH = "artifacts/m22_1_r/execution_record.json"
HUB_RECHECK_NOTE = (
    "The live huggingface_hub.model_info recheck described in "
    "configs/m22_1_r_replay.yaml revision_resolution.precondition_recheck "
    "was not executed. This phase requires an offline snapshot of the "
    "recorded revision and forbids network-fetched weights. The local-cache "
    "clause of REPLAY_BLOCKED was evaluated instead."
)


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _cuda_facts() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "cuda_available": False,
            "cuda_version": "N/A",
            "cuda_version_reason": "torch is not installed, so CUDA was not queried",
            "gpu_identity": "N/A",
            "gpu_identity_reason": "torch is not installed, so no GPU was queried",
        }
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_version": "N/A",
            "cuda_version_reason": "torch.cuda.is_available() is false",
            "gpu_identity": "N/A",
            "gpu_identity_reason": "cuda_available is false",
        }
    return {
        "cuda_available": True,
        "cuda_version": str(torch.version.cuda),
        "cuda_version_reason": "torch.version.cuda",
        "gpu_identity": str(torch.cuda.get_device_name(0)),
        "gpu_identity_reason": "torch.cuda.get_device_name(0)",
    }


def _environment(block_reasons: list[str]) -> dict[str, Any]:
    mapping = {
        "numpy": "numpy",
        "torch": "torch",
        "transformers": "transformers",
        "transformer_lens": "transformer-lens",
    }
    versions: dict[str, str] = {"python": platform.python_version()}
    for field, distribution in mapping.items():
        found = _package_version(distribution)
        if found is None:
            versions[field] = "package_not_installed"
            block_reasons.append(f"environment prerequisite missing: {distribution}")
        else:
            versions[field] = found
    versions.update(_cuda_facts())
    versions["os"] = platform.platform()
    versions["machine"] = platform.machine()
    versions["cpu_count"] = os_cpu_count()
    return versions


def os_cpu_count() -> int:
    import os

    count = os.cpu_count()
    if count is None or count < 1:
        raise RuntimeError("os.cpu_count() did not return a positive integer")
    return int(count)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ReplayBlockedError(f"{path} is not a JSON object")
    return payload


def load_reference_bundle(repo: Path) -> dict[str, Any]:
    directory = repo / "reports" / "M22_1"
    bundle: dict[str, Any] = {}
    for name in (
        RESULTS,
        "frozen_candidate.json",
        "certificate.json",
        "manifest.json",
        "intervention_not_validated.json",
    ):
        bundle[name] = _load_json(directory / name)
    return bundle


def _git_fields(repo: Path, identity: dict[str, Any] | None, block_reasons: list[str]) -> dict[str, Any]:
    current = identity if identity is not None else git_identity(repo)
    commit = current.get("commit")
    dirty = current.get("dirty")
    branch = current.get("branch")
    fields: dict[str, Any] = {}
    if not isinstance(commit, str) or not commit:
        fields["git_commit"] = "not_read_git_rev_parse_failed"
        fields["git_dirty"] = True
        fields["dirty_reason"] = "git HEAD could not be read, so the commit was not established"
        fields["git_branch"] = "not_read_git_branch_failed"
        block_reasons.append(fields["dirty_reason"])
        return fields
    fields["git_commit"] = commit
    if branch is None or branch == "":
        fields["git_branch"] = "detached_HEAD"
    else:
        fields["git_branch"] = str(branch)
    if dirty is None:
        fields["git_dirty"] = True
        fields["dirty_reason"] = "git status could not be read, so cleanliness was not established"
        block_reasons.append(fields["dirty_reason"])
    elif dirty:
        fields["git_dirty"] = True
        fields["dirty_reason"] = "git status --porcelain was non-empty at harness start"
        block_reasons.append("worktree is dirty")
    else:
        fields["git_dirty"] = False
    return fields


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def splice_execution_records(text: str, record: dict[str, Any]) -> str:
    """Replace the top-level execution_records block and leave milestones unchanged."""
    start = text.index("execution_records:")
    end = text.index("\nmilestones:", start)
    dumped = yaml.safe_dump([record], sort_keys=True)
    indented = "\n".join(("  " + line) if line else "" for line in dumped.splitlines())
    return text[:start] + "execution_records:\n" + indented + "\n" + text[end + 1 :]


def update_recovery_execution_record(repo: Path, record: dict[str, Any]) -> None:
    """Insert this attempt into the top-level DEC-009 list without reformatting milestones."""
    path = repo / "control_plane" / "RECOVERY.yaml"
    path.write_text(splice_execution_records(path.read_text(), record))


def run_harness(
    repo: Path,
    *,
    output_path: Path | None = None,
    cache_root: Path | None = None,
    identity: dict[str, Any] | None = None,
    execute_forward: bool = False,
    from_pretrained: Callable[..., Any] | None = None,
    expected_weight_hashes: dict[str, str] | None = None,
    replay_bundle: dict[str, Any] | None = None,
    update_recovery: bool = False,
    clock: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Run the prerequisite check and, only when asked, a pinned forward.

    ``expected_weight_hashes=None`` records that no reference weight hash
    exists. It does not invent one. A supplied map that disagrees with the
    local files is ``REPLAY_DIVERGENT`` and does not load the model.
    """
    repo = repo.resolve()
    config = load_replay_config(repo)
    block_reasons: list[str] = []
    warnings = [HUB_RECHECK_NOTE, "Tolerance values were read and were not modified."]
    if not config.revision_matches_required_pin():
        block_reasons.append(
            "configs/m22_1_r_replay.yaml recorded_revision does not match "
            f"the required pin {REQUIRED_REVISION}"
        )

    git_fields = _git_fields(repo, identity, block_reasons)
    environment = _environment(block_reasons)
    reference_rows, reference_errors = verify_reference_manifest(repo)
    block_reasons.extend(reference_errors)

    cache = cache_root if cache_root is not None else default_hub_cache()
    revision = config.recorded_revision or REQUIRED_REVISION
    snapshot = snapshot_directory(cache, config.model_id, REQUIRED_REVISION)
    weight_rows: list[dict[str, str]] = []
    weight_status = "local_snapshot_absent"
    weight_mismatch = None
    if not config.revision_matches_required_pin():
        weight_status = "not_hashed_revision_pin_disagrees"
    elif not snapshot.is_dir():
        block_reasons.append(
            "local snapshot for pinned revision "
            f"{REQUIRED_REVISION} is absent at {snapshot}"
        )
    else:
        weight_rows = hash_weight_files(snapshot)
        if not weight_rows:
            weight_status = "snapshot_has_no_weight_files"
            block_reasons.append(f"local snapshot {snapshot} contains no weight files")
        else:
            weight_status = "hashed"
            weight_mismatch = weight_identity_mismatch(weight_rows, expected_weight_hashes)

    forward_executed = False
    comparison: dict[str, Any] | None = None
    tier = "REPLAY_BLOCKED"
    failure_class = "prerequisite"

    if weight_mismatch is not None and not block_reasons:
        tier = "REPLAY_DIVERGENT"
        failure_class = "weight_identity"
        warnings.append(weight_mismatch)
    elif block_reasons:
        tier = "REPLAY_BLOCKED"
        failure_class = "prerequisite"
    elif replay_bundle is not None:
        comparison = compare_bundles(load_reference_bundle(repo), replay_bundle, config)
        tier = comparison["replay_tier"]
        failure_class = comparison["failure_class"]
    elif execute_forward:
        try:
            _run_pinned_forward(
                repo=repo,
                config=config,
                from_pretrained=from_pretrained,
            )
            forward_executed = True
        except ReplayBlockedError as exc:
            block_reasons.append(exc.reason)
            tier = "REPLAY_BLOCKED"
            failure_class = "prerequisite"
            forward_executed = False
        else:
            warnings.append(
                "The pinned revision loaded, and this invocation did not run "
                "the M22.1 preflight scoring pass."
            )
            block_reasons.append(
                "pinned load was requested without a replay bundle; "
                "preflight scoring was not run"
            )
            tier = "REPLAY_BLOCKED"
            failure_class = "prerequisite"
    else:
        block_reasons.append(
            "forward execution was not requested and no replay bundle was supplied"
        )
        tier = "REPLAY_BLOCKED"
        failure_class = "prerequisite"

    preflight_path = repo / "configs" / "m22_1_preflight.json"
    preflight_hash = file_sha256(preflight_path) if preflight_path.is_file() else "not_computed_file_absent"
    if preflight_hash == "not_computed_file_absent":
        block_reasons.append("configs/m22_1_preflight.json is absent")
        if tier != "REPLAY_DIVERGENT":
            tier = "REPLAY_BLOCKED"
            failure_class = "prerequisite"

    success_names = set((config.document.get("dec007_success_tiers") or {}).get("tiers") or [])
    dec007_success = tier in success_names

    timestamp = clock() if clock is not None else datetime.now(timezone.utc).isoformat()
    expected_weight_record: Any
    if expected_weight_hashes is None:
        expected_weight_record = EXPECTED_WEIGHT_HASHES_ABSENT
    else:
        expected_weight_record = [
            {"relative_path": key, "sha256": expected_weight_hashes[key]}
            for key in sorted(expected_weight_hashes)
        ]

    record = {
        "record_id": "M22.1-R-harness",
        "milestone": "M22.1-R",
        "entrypoint": ENTRYPOINT,
        "invocation": INVOCATION,
        "timestamp": timestamp,
        "replay_tier": tier,
        "failure_class": failure_class,
        "dec007_success": dec007_success,
        "scientific_failure": False,
        "scientific_claim_updated": False,
        "forward_executed": forward_executed,
        "execute_forward_requested": execute_forward,
        "block_reasons": block_reasons,
        "block_reason": block_reasons[0] if block_reasons else "none",
        "warnings": warnings,
        "model_id": config.model_id,
        "model_revision": REQUIRED_REVISION,
        "recorded_revision_in_config": config.recorded_revision,
        "revision_source": "configs/m22_1_r_replay.yaml revision_pin.recorded_revision",
        "hub_recheck": "not_executed_offline_phase",
        "local_snapshot_present": snapshot.is_dir(),
        "local_snapshot_identity": snapshot.as_posix(),
        "huggingface_cache_root": cache.as_posix(),
        "weight_status": weight_status,
        "weight_files": weight_rows,
        "expected_weight_hashes": expected_weight_record,
        "weight_mismatch": weight_mismatch if weight_mismatch is not None else "none",
        "reference_manifest": MANIFEST_PATH,
        "reference_manifest_sha256": file_sha256(repo / MANIFEST_PATH)
        if (repo / MANIFEST_PATH).is_file()
        else "not_computed_file_absent",
        "reference_artifacts": reference_rows,
        "holdout_identity": (
            "M22.1 preflight has no separate holdout file. "
            "The frozen prompt manifest recorded in reports/M22_1/manifest.json "
            "is the split identity."
        ),
        "config_path": config.path,
        "config_sha256": config.file_sha256,
        "preflight_config_path": "configs/m22_1_preflight.json",
        "preflight_config_sha256": preflight_hash,
        "tolerance_label": str((config.document.get("tolerances") or {}).get("label", "missing")),
        "tolerance_keys": list(config.tolerances.keys()),
        "tolerance_values": {key: config.tolerances[key] for key in config.tolerances},
        "seed": {
            "forward_seed_consumed": False,
            "note": "No replay forward was executed, so no additional seed was drawn.",
        },
        "nondeterministic_fields": ["timestamp"] if clock is None else [],
        "comparison_performed": comparison is not None,
        "comparison": comparison if comparison is not None else {"performed": False},
        **git_fields,
        **environment,
    }
    built = build_record(record)
    destination = output_path if output_path is not None else repo / EXECUTION_RECORD_PATH
    if destination.resolve() == (repo / "reports" / "M22_1").resolve():
        raise ReplayBlockedError("refusing to write the execution record onto the reference directory")
    _write_json(destination, built)
    if update_recovery:
        update_recovery_execution_record(repo, built)
    return built


def _run_pinned_forward(
    *,
    repo: Path,
    config: ReplayConfig,
    from_pretrained: Callable[..., Any] | None,
) -> None:
    """Load the pinned revision. The historical unpinned retry is not used.

    Producing a full preflight bundle is intentionally not done unless a
    caller both requests ``execute_forward`` and supplies a loader that
    returns a model. This function fails closed on load failure.
    """
    del repo
    if not config.revision_matches_required_pin():
        raise ReplayBlockedError("refusing to load because the recorded pin disagrees")
    if from_pretrained is None:
        import torch
        from transformer_lens import HookedTransformer

        from_pretrained = HookedTransformer.from_pretrained
        dtype = torch.float32
    else:
        dtype = None
    load_pinned_revision(
        config.model_id,
        REQUIRED_REVISION,
        from_pretrained=from_pretrained,
        recorded_revision=REQUIRED_REVISION,
        dtype=dtype,
    )
