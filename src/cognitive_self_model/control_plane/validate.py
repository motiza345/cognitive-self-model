"""Validate the project control-plane YAML ledgers."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

HEX64 = re.compile(r"^[0-9a-f]{40,64}$")
COMMIT40 = re.compile(r"^[0-9a-f]{40}$")
PYTHON_VERSION = re.compile(r"^(?:UNKNOWN|\d+\.\d+(?:\.\d+)?(?:(?:a|b|rc)\d+)?)$")
ENV_VERSION = re.compile(r"^(?:UNKNOWN|\d+\.\d+(?:\.\d+)?(?:\+\w+)?)$")
RECOVERY_FILE = "control_plane/RECOVERY.yaml"
REPLAY_CONFIG = "configs/m22_1_r_replay.yaml"
REQUIRED_ENV_KEYS = ("python", "torch", "transformers", "transformer_lens", "numpy")
ENV_VERSION_KEYS = ("torch", "transformers", "transformer_lens", "numpy")
REPLAY_TIER_NAMES = (
    "REPLAY_INVALID",
    "REPLAY_BLOCKED",
    "REPLAY_DIVERGENT",
    "REPLAY_EXACT",
    "REPLAY_NUMERIC_EQUIVALENT",
    "REPLAY_BEHAVIORAL_EQUIVALENT",
)
REPLAY_TOLERANCE_KEYS = ("score_atol", "score_rtol", "delta_atol", "mean_atol", "null_atol")
REPLAY_RESOLUTION_KEYS = ("mechanism", "drift_risk", "mitigation_required", "precondition_recheck")
EXECUTION_SENTINELS = {"UNKNOWN", "NOT_APPLICABLE", "NOT_RECORDED", ""}
EXECUTION_REQUIRED_STRINGS = ("entrypoint", "invocation")
FROZEN_TOLERANCE_ALIASES = ("s_abs", "s_rel", "delta_abs", "mean_abs", "null_delta_abs")

STATUS_ENUM = {
    "CANDIDATE",
    "VALIDATED_FOR_M22",
    "INSUFFICIENT_EVIDENCE",
    "READOUT_RECONSTRUCTION_EXACT",
    "CONDITIONAL_SUPPORT_LIMITED",
    "NOT_AUTHORIZED",
    "UNSPECIFIED_IN_REPORT",
    "NOT_STARTED",
    "PREREGISTERED",
    "COMPLETED_REINTERPRETATION_DESCRIPTIVELY_SUPPORTED",
    "COMPLETED_PROJECTION_TERM_NON_NEGLIGIBLE",
    "COMPLETED_INSUFFICIENT_EVIDENCE",
    "GATE_FAILED_G0",
    "GATE_FAILED_G1",
    "GATE_FAILED_G2",
    "GATE_FAILED_G3",
    "GATE_FAILED_G4",
    "POINTER_ONLY",
    "ABSENT",
}
CLAIM_STATUS = {"ACTIVE", "REINTERPRETED", "SUPERSEDED", "RETRACTED"}
CLAIM_KIND = {"finding", "observation"}
FILE_CATEGORY = {
    "ACTIVE",
    "AUTHORITATIVE",
    "REUSABLE",
    "BENCHMARK",
    "REPORT",
    "ARTIFACT",
    "CONTRACT",
    "LEGACY",
    "DUPLICATE",
    "OBSOLETE",
}
RECOVERY_GRADE = {"EXACT", "PARTIAL", "POINTER_ONLY", "ABSENT"}
GRADE_RANK = {"ABSENT": 0, "POINTER_ONLY": 1, "PARTIAL": 2, "EXACT": 3}
CHAIN_STATUS = {"COMPLETE", "INCOMPLETE"}
GATE_REASON_IDS = re.compile(r"\b(?:F|OBS)-[A-Z0-9][A-Z0-9._-]*\b")
OPEN_FLAG_STATUSES = {"open", "OPEN"}
WORSE_THAN_PARTIAL = {"POINTER_ONLY", "ABSENT"}


def repo_root_from(start: Path) -> Path:
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("repository root not found")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit_exists(repo: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", commit],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def collect_commits(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"commit", "as_of_commit", "tip", "merge_base_with_main", "run_recorded_commit", "preflight_commit"}:
                if isinstance(value, str) and COMMIT40.match(value):
                    found.append(value)
            found.extend(collect_commits(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_commits(item))
    return found


def collect_paths(obj: Any) -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"path", "config_path", "report"} and isinstance(value, str):
                if value and value != "UNKNOWN":
                    found.append(value)
            found.extend(collect_paths(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_paths(item))
    return found


def flatten_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(flatten_strings(value))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(flatten_strings(item))
    return out


def validate(control_plane_dir: Path, repo_root: Path | None = None) -> list[str]:
    errors: list[str] = []
    control_plane_dir = control_plane_dir.resolve()
    repo = repo_root.resolve() if repo_root is not None else repo_root_from(control_plane_dir)

    required = ["STATE.yaml", "CLAIMS.yaml", "FILE_MAP.yaml", "RECOVERY.yaml"]
    docs: dict[str, Any] = {}
    for name in required:
        path = control_plane_dir / name
        if not path.is_file():
            errors.append(f"missing {name}")
            continue
        try:
            docs[name] = load_yaml(path)
        except yaml.YAMLError as exc:
            errors.append(f"{name} failed to parse: {exc}")
    if len(docs) != len(required):
        return _with_replay_config(repo, errors)

    state = docs["STATE.yaml"]
    claims_doc = docs["CLAIMS.yaml"]
    file_map = docs["FILE_MAP.yaml"]
    recovery = docs["RECOVERY.yaml"]

    if not isinstance(state, dict):
        errors.append("STATE.yaml must be a mapping")
    if not isinstance(claims_doc, dict) or not isinstance(claims_doc.get("entries"), list):
        errors.append("CLAIMS.yaml must contain an entries list")
    if not isinstance(file_map, dict) or not isinstance(file_map.get("files"), list):
        errors.append("FILE_MAP.yaml must contain a files list")
    if not isinstance(recovery, dict) or not isinstance(recovery.get("milestones"), dict):
        errors.append("RECOVERY.yaml must contain a milestones mapping")
    if errors:
        return _with_replay_config(repo, errors)

    for field in ("as_of_commit", "as_of_date", "branch_topology", "milestones", "gates", "decisions", "next_milestone"):
        if field not in state:
            errors.append(f"STATE missing required field {field}")

    milestones = state.get("milestones") or {}
    if not isinstance(milestones, dict):
        errors.append("STATE.milestones must be a mapping")
        return _with_replay_config(repo, errors)

    for mid, rec in milestones.items():
        if not isinstance(rec, dict):
            errors.append(f"STATE.milestones.{mid} must be a mapping")
            continue
        for field in ("status", "report", "claim_ids"):
            if field not in rec:
                errors.append(f"STATE.milestones.{mid} missing {field}")
        status = rec.get("status")
        if status not in STATUS_ENUM:
            errors.append(f"STATE.milestones.{mid}.status invalid: {status}")

    gates = state.get("gates") or {}
    if "M22.2" not in gates:
        errors.append("STATE.gates.M22.2 missing")
    else:
        if not isinstance(gates["M22.2"], dict) or "authorized" not in gates["M22.2"]:
            errors.append("STATE.gates.M22.2.authorized missing")

    decisions = state.get("decisions") or []
    decision_ids: list[str] = []
    if not isinstance(decisions, list):
        errors.append("STATE.decisions must be a list")
    else:
        for item in decisions:
            if not isinstance(item, dict) or "id" not in item:
                errors.append("STATE.decisions entry missing id")
                continue
            decision_ids.append(str(item["id"]))
        if len(decision_ids) != len(set(decision_ids)):
            errors.append("STATE.decisions ids are not unique")

    entries = claims_doc["entries"]
    claim_ids: list[str] = []
    by_id: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("CLAIMS entry is not a mapping")
            continue
        for field in (
            "id",
            "kind",
            "milestone",
            "statement",
            "scope",
            "evidence",
            "preregistration",
            "labels",
            "status",
            "non_claims",
            "recorded_on",
        ):
            if field not in entry:
                errors.append(f"CLAIMS {entry.get('id', '<unknown>')} missing {field}")
        cid = entry.get("id")
        if cid:
            claim_ids.append(cid)
            by_id[cid] = entry
        if entry.get("kind") not in CLAIM_KIND:
            errors.append(f"CLAIMS {cid} kind invalid: {entry.get('kind')}")
        if entry.get("status") not in CLAIM_STATUS:
            errors.append(f"CLAIMS {cid} status invalid: {entry.get('status')}")
        if entry.get("kind") == "observation":
            if entry.get("evidentiary_weight") != "not a claim; not usable for status decisions":
                errors.append(f"CLAIMS {cid} observation missing required evidentiary_weight")
        scope = entry.get("scope") or {}
        if isinstance(scope, dict):
            for field in ("model", "revision", "site", "outcome", "intervention_family", "observation_contract"):
                if field not in scope:
                    errors.append(f"CLAIMS {cid} scope missing {field}")
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("CLAIMS ids are not unique")

    fmap_paths: list[str] = []
    for entry in file_map["files"]:
        if not isinstance(entry, dict):
            errors.append("FILE_MAP entry is not a mapping")
            continue
        for field in ("path", "category", "immutable", "milestone", "role"):
            if field not in entry:
                errors.append(f"FILE_MAP entry missing {field}: {entry.get('path')}")
        path = entry.get("path")
        if path:
            fmap_paths.append(path)
        if entry.get("category") not in FILE_CATEGORY:
            errors.append(f"FILE_MAP {path} category invalid: {entry.get('category')}")
        if entry.get("immutable") and not entry.get("sha256"):
            errors.append(f"FILE_MAP {path} immutable but sha256 missing")
    if len(fmap_paths) != len(set(fmap_paths)):
        errors.append("FILE_MAP paths are not unique")

    for mid, rec in (recovery.get("milestones") or {}).items():
        if not isinstance(rec, dict):
            errors.append(f"RECOVERY {mid} must be a mapping")
            continue
        for field in (
            "recovery_grade_local",
            "recovery_grade_chain",
            "depends_on",
            "depends_on_verified",
            "depends_on_provenance",
            "chain_status",
            "branch",
            "commit",
            "entrypoint",
            "config_path",
            "config_sha256",
            "environment",
            "model_id",
            "model_revision",
            "expected_labels",
        ):
            if field not in rec:
                errors.append(f"RECOVERY {mid} missing {field}")
        local = rec.get("recovery_grade_local")
        chain = rec.get("recovery_grade_chain")
        if local not in RECOVERY_GRADE:
            errors.append(f"RECOVERY {mid} recovery_grade_local invalid: {local}")
        if chain not in RECOVERY_GRADE:
            errors.append(f"RECOVERY {mid} recovery_grade_chain invalid: {chain}")
        if rec.get("chain_status") not in CHAIN_STATUS:
            errors.append(f"RECOVERY {mid} chain_status invalid: {rec.get('chain_status')}")
        if not _depends_on_is_valid(rec.get("depends_on")):
            errors.append(f"RECOVERY {mid} depends_on must be a list or UNKNOWN")
        if not isinstance(rec.get("depends_on_provenance"), list):
            errors.append(f"RECOVERY {mid} depends_on_provenance must be a list")
        _check_environment(str(mid), rec.get("environment"), errors)
        if "execution_records" in rec:
            errors.append(
                f"{RECOVERY_FILE}: RECOVERY {mid} must not nest execution_records; "
                "use the top-level list"
            )
    _check_execution_records(recovery, errors)

    # 2. Every path in FILE_MAP/CLAIMS/RECOVERY exists in the working tree.
    for label, doc in (("FILE_MAP", file_map), ("CLAIMS", claims_doc), ("RECOVERY", recovery)):
        for rel in collect_paths(doc):
            target = repo / rel
            if not target.exists():
                errors.append(f"{label} path does not exist: {rel}")

    # 3. Immutable sha256 matches; referenced commits exist.
    for entry in file_map["files"]:
        if not entry.get("immutable"):
            continue
        rel = entry["path"]
        target = repo / rel
        if not target.is_file():
            continue
        actual = sha256_file(target)
        expected = str(entry.get("sha256") or "")
        if actual != expected:
            errors.append(f"immutable sha256 mismatch: {rel}")

    for rel, expected in _recovery_hashes(recovery):
        target = repo / rel
        if target.is_file() and HEX64.match(expected) and sha256_file(target) != expected:
            errors.append(f"RECOVERY sha256 mismatch: {rel}")

    for commit in sorted(set(collect_commits(state) + collect_commits(claims_doc) + collect_commits(recovery))):
        if not commit_exists(repo, commit):
            errors.append(f"referenced commit does not exist: {commit}")

    # 4. Evidence resolves; reinterpreted_by / supersedes bidirectional.
    for entry in entries:
        cid = entry.get("id")
        evidence = entry.get("evidence") or []
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"CLAIMS {cid} evidence missing")
            continue
        for item in evidence:
            if not isinstance(item, dict):
                errors.append(f"CLAIMS {cid} evidence item is not a mapping")
                continue
            for field in ("path", "sha256", "commit"):
                if field not in item:
                    errors.append(f"CLAIMS {cid} evidence missing {field}")
            rel = item.get("path")
            if not rel:
                continue
            target = repo / str(rel)
            if not target.is_file():
                errors.append(f"CLAIMS {cid} evidence path missing: {rel}")
                continue
            if HEX64.match(str(item.get("sha256") or "")) and sha256_file(target) != item["sha256"]:
                errors.append(f"CLAIMS {cid} evidence sha256 mismatch: {rel}")
            commit = item.get("commit")
            if isinstance(commit, str) and COMMIT40.match(commit) and not commit_exists(repo, commit):
                errors.append(f"CLAIMS {cid} evidence commit missing: {commit}")

        pointer = entry.get("reinterpreted_by")
        if pointer:
            if pointer not in by_id:
                errors.append(f"CLAIMS {cid} reinterpreted_by missing: {pointer}")
            else:
                supersedes = by_id[pointer].get("supersedes") or []
                if cid not in supersedes:
                    errors.append(f"CLAIMS {cid} reinterpreted_by {pointer} is not bidirectional")
        for other in entry.get("supersedes") or []:
            if other not in by_id:
                errors.append(f"CLAIMS {cid} supersedes missing: {other}")
            elif by_id[other].get("reinterpreted_by") != cid:
                errors.append(f"CLAIMS {cid} supersedes {other} is not bidirectional")

    # 5. No STATE status or gate references a kind=observation entry.
    observation_ids = {entry["id"] for entry in entries if entry.get("kind") == "observation" and entry.get("id")}
    gate_and_status = {
        "gates": state.get("gates"),
        "statuses": {mid: rec.get("status") for mid, rec in milestones.items() if isinstance(rec, dict)},
    }
    for text in flatten_strings(gate_and_status):
        for match in GATE_REASON_IDS.findall(text):
            if match in observation_ids:
                errors.append(f"STATE status/gate references observation {match}")
        if text in observation_ids:
            errors.append(f"STATE status/gate references observation {text}")

    # 6. Tracked reports/configs/artifacts must appear in FILE_MAP.
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).splitlines()
    mapped = set(fmap_paths)
    for rel in tracked:
        if not (rel.startswith("reports/") or rel.startswith("configs/") or rel.startswith("artifacts/")):
            continue
        if rel.endswith("/.gitkeep"):
            if rel not in mapped:
                errors.append(f"FILE_MAP missing tracked path {rel}")
            continue
        if rel in mapped:
            continue
        if _is_m22_path(rel):
            errors.append(f"FILE_MAP missing M22 tracked path {rel}")
        else:
            print(f"warning: FILE_MAP missing older tracked path {rel}", file=sys.stderr)

    # 7. Status differing from report-extracted label must cite a decision.
    for mid, rec in milestones.items():
        if not isinstance(rec, dict):
            continue
        status = rec.get("status")
        extracted = rec.get("report_extracted_label")
        if status == "UNSPECIFIED_IN_REPORT":
            continue
        if extracted is None:
            continue
        if status != extracted:
            cited = rec.get("status_decision_id")
            if cited not in decision_ids:
                errors.append(f"STATE.milestones.{mid} status differs from report without a decision id")

    # 8. Guard M22.1 CANDIDATE and M22.2 unauthorized unless user decision.
    m22_1_status = (milestones.get("M22.1") or {}).get("status")
    m22_2_auth = ((state.get("gates") or {}).get("M22.2") or {}).get("authorized")
    # The two gates are independent: an override for one must not disarm the other.
    override_m22_1 = any(_decision_grants(item, "M22.1") for item in decisions)
    override_m22_2 = any(_decision_grants(item, "M22.2") for item in decisions)
    if m22_1_status != "CANDIDATE" and not override_m22_1:
        errors.append("STATE.milestones.M22.1.status must be CANDIDATE unless a user-approved decision authorizes otherwise")
    if m22_2_auth is not False and not override_m22_2:
        errors.append("STATE.gates.M22.2.authorized must be false unless a user-approved decision authorizes otherwise")

    # 9. Open integrity flags on claim evidence must be listed in STATE.known_integrity_issues.
    open_issues = _open_integrity_issues(state)
    issue_ids = {item.get("id") for item in open_issues}
    issue_paths = {item.get("path") for item in open_issues}
    for entry in entries:
        for flag in entry.get("integrity_flags") or []:
            if not isinstance(flag, dict):
                continue
            if str(flag.get("resolution_status")) not in OPEN_FLAG_STATUSES:
                continue
            fid = flag.get("id")
            fpath = flag.get("path")
            if fid not in issue_ids:
                errors.append(f"open integrity flag {fid} missing from STATE.known_integrity_issues")
            if fpath and fpath not in issue_paths:
                errors.append(f"open integrity flag path {fpath} missing from STATE.known_integrity_issues")

    # 10. Recovery grade criteria consistency.
    rec_ms = recovery.get("milestones") or {}
    if "recovery_grade_criteria" not in recovery:
        errors.append("RECOVERY missing recovery_grade_criteria")
    for mid, rec in rec_ms.items():
        if not isinstance(rec, dict):
            continue
        local = rec.get("recovery_grade_local")
        missing = rec.get("missing") or []
        if not isinstance(missing, list):
            errors.append(f"RECOVERY {mid} missing must be a list")
            continue
        if local == "EXACT" and missing:
            errors.append(f"RECOVERY {mid} EXACT must have no missing items")
        if local == "PARTIAL" and len(missing) < 1:
            errors.append(f"RECOVERY {mid} PARTIAL must have a non-empty missing list")
        deps = rec.get("depends_on")
        if isinstance(deps, list):
            for dep in deps:
                if dep not in rec_ms:
                    errors.append(f"RECOVERY {mid} depends_on unknown milestone {dep}")
            expected_chain = _computed_chain_grade(mid, rec_ms)
            if expected_chain is not None and rec.get("recovery_grade_chain") != expected_chain:
                errors.append(
                    f"RECOVERY {mid} recovery_grade_chain {rec.get('recovery_grade_chain')} "
                    f"!= computed {expected_chain}"
                )

    # 11. depends_on UNKNOWN → chain_status INCOMPLETE; chain not better than local.
    #     Empty depends_on [] allowed only with depends_on_verified true and a cited source.
    for mid, rec in rec_ms.items():
        if not isinstance(rec, dict):
            continue
        deps = rec.get("depends_on")
        local = rec.get("recovery_grade_local")
        chain = rec.get("recovery_grade_chain")
        if deps == "UNKNOWN":
            if rec.get("chain_status") != "INCOMPLETE":
                errors.append(f"RECOVERY {mid} depends_on UNKNOWN requires chain_status INCOMPLETE")
            if local in GRADE_RANK and chain in GRADE_RANK and GRADE_RANK[chain] > GRADE_RANK[local]:
                errors.append(
                    f"RECOVERY {mid} depends_on UNKNOWN: recovery_grade_chain {chain} "
                    f"must not be better than recovery_grade_local {local}"
                )
        elif isinstance(deps, list) and deps == []:
            if rec.get("depends_on_verified") is not True:
                errors.append(f"RECOVERY {mid} empty depends_on requires depends_on_verified true")
            if not _has_cited_provenance(rec.get("depends_on_provenance")):
                errors.append(f"RECOVERY {mid} empty depends_on requires a cited source in depends_on_provenance")

    # 12. Claims whose milestone chain is worse than PARTIAL need scope_limitation.
    for entry in entries:
        if entry.get("kind") != "finding":
            continue
        cid = entry.get("id")
        mid = entry.get("milestone")
        rec = rec_ms.get(mid) if isinstance(mid, str) else None
        if not isinstance(rec, dict):
            continue
        chain = rec.get("recovery_grade_chain")
        if chain not in WORSE_THAN_PARTIAL:
            continue
        limitation = entry.get("scope_limitation")
        if not (isinstance(limitation, str) and limitation.strip()):
            errors.append(
                f"CLAIMS {cid} milestone chain {chain} requires non-empty "
                f"scope_limitation referring to the upstream gap"
            )

    return _with_replay_config(repo, errors)


def _recovery_hashes(recovery: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for rec in (recovery.get("milestones") or {}).values():
        if not isinstance(rec, dict):
            continue
        cfg = rec.get("config_path")
        digest = rec.get("config_sha256")
        if isinstance(cfg, str) and cfg != "UNKNOWN" and isinstance(digest, str) and HEX64.match(digest):
            pairs.append((cfg, digest))
        for bucket in ("input_artifacts", "output_artifacts"):
            for item in rec.get(bucket) or []:
                if isinstance(item, dict) and isinstance(item.get("path"), str) and HEX64.match(str(item.get("sha256") or "")):
                    pairs.append((item["path"], item["sha256"]))
    return pairs


def _is_m22_path(rel: str) -> bool:
    name = rel.lower()
    return "m22" in name or "/M22" in rel or rel.startswith("reports/M22")


def _open_integrity_issues(state: dict) -> list[dict]:
    issues = state.get("known_integrity_issues")
    if isinstance(issues, dict):
        return [item for item in (issues.get("open") or []) if isinstance(item, dict)]
    if isinstance(issues, list):
        return [
            item
            for item in issues
            if isinstance(item, dict) and str(item.get("resolution_status", "open")) in OPEN_FLAG_STATUSES
        ]
    return []


def _is_execution_sentinel(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in EXECUTION_SENTINELS or value.strip() == ""
    return False


def _check_execution_records(recovery: dict, errors: list[str]) -> None:
    """DEC-009. Milestone entries may still use UNKNOWN. Execution records may not."""
    records = recovery.get("execution_records")
    if not isinstance(records, list):
        errors.append(f"{RECOVERY_FILE}: execution_records must be a top-level list")
        return
    for index, record in enumerate(records):
        label = f"execution_records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{RECOVERY_FILE}: {label} must be a mapping")
            continue
        for field in EXECUTION_REQUIRED_STRINGS:
            value = record.get(field)
            if field not in record or not isinstance(value, str) or _is_execution_sentinel(value):
                errors.append(
                    f"{RECOVERY_FILE}: {label} {field} is required and must not be a sentinel"
                )
        git_dirty = record.get("git_dirty")
        if "git_dirty" not in record or not isinstance(git_dirty, bool):
            errors.append(f"{RECOVERY_FILE}: {label} git_dirty is required and must be a boolean")
            continue
        if git_dirty:
            reason = record.get("dirty_reason")
            if "dirty_reason" not in record or not isinstance(reason, str) or _is_execution_sentinel(reason):
                errors.append(
                    f"{RECOVERY_FILE}: {label} dirty_reason is required when git_dirty is true "
                    "and must be a non-empty non-sentinel string"
                )


def _check_environment(mid: str, environment: Any, errors: list[str]) -> None:
    """Require the five recorded environment keys. Optional keys are checked only when present.

    A present key whose value is UNKNOWN means the value was checked and is not known.
    A missing key is a separate error. Historical milestones keep UNKNOWN rather than
    dropping the key.
    """
    if not isinstance(environment, dict):
        errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment must be a mapping")
        return
    for key in REQUIRED_ENV_KEYS:
        if key not in environment:
            errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment missing required key: {key}")
    if "python" in environment:
        value = environment["python"]
        if not isinstance(value, str) or PYTHON_VERSION.match(value) is None:
            errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment.python invalid: {value}")
    for key in ENV_VERSION_KEYS:
        if key not in environment:
            continue
        value = environment[key]
        if not isinstance(value, str) or ENV_VERSION.match(value) is None:
            errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment.{key} invalid: {value}")
    if "threads" in environment:
        value = environment["threads"]
        if value != "UNKNOWN" and not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment.threads invalid: {value}")
    if "os" in environment:
        value = environment["os"]
        if not isinstance(value, str) or value == "":
            errors.append(f"{RECOVERY_FILE}: RECOVERY {mid} environment.os invalid: {value}")


def _with_replay_config(repo: Path, errors: list[str]) -> list[str]:
    _check_m22_1_r_replay_config(repo, errors)
    return errors


def _check_m22_1_r_replay_config(repo_root: Path, errors: list[str]) -> None:
    """Schema-check configs/m22_1_r_replay.yaml. Skip when that file is absent."""
    path = repo_root / REPLAY_CONFIG
    if not path.exists():
        return
    try:
        document = load_yaml(path)
    except yaml.YAMLError as exc:
        errors.append(f"{REPLAY_CONFIG}: failed to parse: {exc}")
        return
    if not isinstance(document, dict):
        errors.append(f"{REPLAY_CONFIG}: document must be a mapping")
        return

    resolution = document.get("revision_resolution")
    if not isinstance(resolution, dict):
        errors.append(f"{REPLAY_CONFIG}: revision_resolution must be a mapping")
    else:
        for field in REPLAY_RESOLUTION_KEYS:
            value = resolution.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{REPLAY_CONFIG}: revision_resolution missing required key: {field}")

    tolerances = document.get("tolerances")
    if not isinstance(tolerances, dict):
        errors.append(f"{REPLAY_CONFIG}: tolerances must be a mapping")
    else:
        for key in REPLAY_TOLERANCE_KEYS:
            value = tolerances.get(key)
            if isinstance(value, bool) or not isinstance(value, float) or value <= 0:
                errors.append(f"{REPLAY_CONFIG}: tolerances.{key} invalid: {value}")
        for alias in FROZEN_TOLERANCE_ALIASES:
            if alias in tolerances:
                errors.append(
                    f"{REPLAY_CONFIG}: tolerances.{alias} is not a frozen tolerance key; "
                    "use score_atol, score_rtol, delta_atol, mean_atol, null_atol"
                )

    tiers = document.get("tiers")
    if not isinstance(tiers, dict):
        errors.append(f"{REPLAY_CONFIG}: tiers must be a mapping")
        tier_names: set[str] = set()
    else:
        tier_names = {str(name) for name in tiers}
        if tier_names != set(REPLAY_TIER_NAMES):
            errors.append(f"{REPLAY_CONFIG}: tiers must define exactly {list(REPLAY_TIER_NAMES)}")

    precedence = document.get("tier_precedence")
    if not isinstance(precedence, list):
        errors.append(f"{REPLAY_CONFIG}: tier_precedence must be a list")
    elif len(precedence) != len(set(precedence)) or set(precedence) != tier_names or set(precedence) != set(REPLAY_TIER_NAMES):
        errors.append(
            f"{REPLAY_CONFIG}: tier_precedence must list each tier name once and match tiers"
        )

    success_block = document.get("dec007_success_tiers")
    if not isinstance(success_block, dict):
        errors.append(f"{REPLAY_CONFIG}: dec007_success_tiers must be a mapping")
    else:
        success = success_block.get("tiers")
        if not isinstance(success, list) or len(success) == 0:
            errors.append(f"{REPLAY_CONFIG}: dec007_success_tiers.tiers must be a non-empty list")
        else:
            if "REPLAY_BEHAVIORAL_EQUIVALENT" in success:
                errors.append(
                    f"{REPLAY_CONFIG}: dec007_success_tiers.tiers must not contain "
                    "REPLAY_BEHAVIORAL_EQUIVALENT"
                )
            for name in success:
                if name not in tier_names:
                    errors.append(
                        f"{REPLAY_CONFIG}: dec007_success_tiers.tiers contains unknown tier: {name}"
                    )

    code = document.get("code_under_test")
    if not isinstance(code, dict):
        errors.append(f"{REPLAY_CONFIG}: code_under_test must be a mapping")
        return
    commit = code.get("commit")
    if not isinstance(commit, str) or COMMIT40.match(commit) is None:
        errors.append(f"{REPLAY_CONFIG}: code_under_test.commit invalid: {commit}")
    branch = code.get("branch")
    if not isinstance(branch, str) or not branch.strip():
        errors.append(f"{REPLAY_CONFIG}: code_under_test.branch invalid: {branch}")


def _depends_on_is_valid(value: Any) -> bool:
    return value == "UNKNOWN" or isinstance(value, list)


def _has_cited_provenance(provenance: Any) -> bool:
    if not isinstance(provenance, list):
        return False
    for item in provenance:
        if isinstance(item, dict) and item.get("source_path"):
            return True
    return False


def _computed_chain_grade(mid: str, milestones: dict, seen: set[str] | None = None) -> str | None:
    rec = milestones.get(mid)
    if not isinstance(rec, dict):
        return None
    local = rec.get("recovery_grade_local")
    if local not in GRADE_RANK:
        return None
    seen = set() if seen is None else set(seen)
    if mid in seen:
        return local
    seen.add(mid)
    rank = GRADE_RANK[local]
    deps = rec.get("depends_on")
    if not isinstance(deps, list):
        return local
    for dep in deps:
        dep_grade = _computed_chain_grade(str(dep), milestones, seen)
        if dep_grade in GRADE_RANK:
            rank = min(rank, GRADE_RANK[dep_grade])
    for grade, value in GRADE_RANK.items():
        if value == rank:
            return grade
    return local


_GATE_MARKERS = {
    "M22.1": ("m22.1", "validated"),
    "M22.2": ("m22.2", "authoriz"),
}


def _decision_grants(item: Any, gate: str) -> bool:
    """True only if `item` is an in-force, user-approved override for `gate`.

    Each gate is evaluated separately; granting one never grants the other.
    Prose matching is a heuristic and is negation-blind; it is scheduled to be
    superseded by an explicit structured field (DEC-009 follow-up, Commit 5).
    """
    if gate not in _GATE_MARKERS:
        raise ValueError(f"unknown gate: {gate!r}")
    if not isinstance(item, dict):
        return False
    if item.get("approved_by") != "user":
        return False
    if item.get("status") != "ACCEPTED":
        return False
    subject, verb = _GATE_MARKERS[gate]
    text = " ".join(flatten_strings(item)).lower()
    return subject in text and verb in text


def validate_or_raise(control_plane_dir: Path, repo_root: Path | None = None) -> None:
    errors = validate(control_plane_dir, repo_root)
    if errors:
        raise AssertionError("\n".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the project control plane.")
    parser.add_argument(
        "--control-plane",
        default=None,
        help="Path to the control_plane directory (defaults to <repo>/control_plane).",
    )
    parser.add_argument("--repo-root", default=None, help="Repository root (defaults to git ancestor).")
    args = parser.parse_args(argv)
    if args.repo_root:
        repo = Path(args.repo_root)
    else:
        repo = repo_root_from(Path.cwd())
    control = Path(args.control_plane) if args.control_plane else repo / "control_plane"
    errors = validate(control, repo)
    if errors:
        for item in errors:
            print(item)
        return 1
    print("control plane OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
