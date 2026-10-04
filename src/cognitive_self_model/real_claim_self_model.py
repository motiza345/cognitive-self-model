"""Claim registry for one real Qwen capability.

The decision reads a claim status. It does not read margins, effects, or
task labels. Qwen is not loaded here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scipy.stats import beta

from src.cognitive_self_model.evaluation.catalog import (
    historical_prompts,
    normalize_prompt,
    prompt_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "REAL_QWEN_CLAIM_SELF_MODEL.1"
EXPECTED_PROTOCOL_SHA256 = "55a14f9e84465ee20bdcb4347201624c54e9def9c2d05465e5e52d0fb3492249"
EXPECTED_CATALOG_SHA256 = "298136e56f6dddcc7fbb4015f9940eb40f71e096555494588d8e14dd435f4e21"
CATALOG_VERSION = "REAL_QWEN_CLAIM_SELF_MODEL_CATALOG.1"
CLAIM_ID = "QWEN_SELF_MARGIN_GAIN_D1_L0_A1"
CLAIM_TYPE = "CAPABILITY"
PROPOSITION = (
    "Qwen can reliably increase its own yes/no margin under D1-L0-alpha1 "
    "within the declared scope."
)
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DIRECTION_SEED = 22101
DIRECTION_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
HOOK = "blocks.0.hook_resid_post"
LAYER = 0
ALPHA = 1
M24_PROTOCOL = "reports/M24_CONSUMPTION_PROTOCOL.md"
M24_RESULTS = "reports/M24_CONSUMPTION_RESULTS.json"
M24_VERDICT = "CONSUMPTION_SUPPORTED"

STATUS_SUPPORTED = "SUPPORTED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_CONTRADICTED = "CONTRADICTED"
STATUS_NONE = "NO_APPLICABLE_CLAIM"
ACTION_INTERVENE = "INTERVENE"
ACTION_ABSTAIN = "ABSTAIN"
YES = "YES"
NO = "NO"

SCOPE = {
    "alpha": ALPHA,
    "direction_id": "D1",
    "direction_seed": DIRECTION_SEED,
    "direction_sha256": DIRECTION_SHA256,
    "hook": HOOK,
    "layer": LAYER,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "target": "yes/no margin",
}

EVIDENCE_COUNT = 12
VALIDATION_COUNT = 12
HOLDOUT_COUNT = 24
SERIALS = range(1, EVIDENCE_COUNT + VALIDATION_COUNT + HOLDOUT_COUNT + 1)

SUITE_CATALOGS = (
    "reports/SELF_MODEL_EVALUATION_CATALOG_V1.json",
    "reports/SELF_MODEL_EVALUATION_CATALOG_V2.json",
)

BANNED_PREDICTION_KEYS = {
    "correct",
    "ground_truth",
    "intervened_margin",
    "label",
    "observed_effect",
    "outcome",
    "utility",
}


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clopper_pearson(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact interval for a binomial proportion."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("binomial counts are outside the evidence set")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return lower, upper


def status_from_effects(effects: list[float]) -> dict[str, object]:
    """Freeze the evidence rule. Exact zero is neither positive nor negative."""
    if len(effects) != EVIDENCE_COUNT:
        raise ValueError("the evidence rule requires the sealed set of 12")
    positive = sum(1 for value in effects if value > 0.0)
    negative = sum(1 for value in effects if value < 0.0)
    zeros = sum(1 for value in effects if value == 0.0)
    if positive + negative + zeros != EVIDENCE_COUNT:
        raise ValueError("an evidence effect is not a real number")
    lower, upper = clopper_pearson(positive, EVIDENCE_COUNT)
    if lower > 0.5:
        status = STATUS_SUPPORTED
    elif upper < 0.5:
        status = STATUS_CONTRADICTED
    else:
        status = STATUS_UNCERTAIN
    return {
        "alpha": 0.05,
        "interval": "clopper_pearson",
        "lower": lower,
        "n": EVIDENCE_COUNT,
        "negative": negative,
        "positive": positive,
        "positive_fraction": positive / EVIDENCE_COUNT,
        "status": status,
        "uncertainty": upper - lower,
        "upper": upper,
        "zeros": zeros,
    }


def decide(status: str) -> str:
    """External policy. The only epistemic input is the claim status."""
    if status == STATUS_SUPPORTED:
        return ACTION_INTERVENE
    if status in {STATUS_UNCERTAIN, STATUS_CONTRADICTED, STATUS_NONE}:
        return ACTION_ABSTAIN
    raise ValueError("status is outside the sealed policy")


def applicable_status(claim: dict[str, object], execution_scope: dict[str, object]) -> str:
    if claim.get("scope") != execution_scope:
        return STATUS_NONE
    return str(claim["status"])


def choice_from_margin(margin: float) -> str:
    if margin > 0.0:
        return YES
    return NO


def answer_correct(margin: float, label: str) -> int:
    return int(choice_from_margin(margin) == label)


def episode_utility(decision: str, baseline_margin: float, intervened_margin: float, label: str) -> int:
    if decision == ACTION_INTERVENE:
        return answer_correct(intervened_margin, label)
    if decision == ACTION_ABSTAIN:
        return answer_correct(baseline_margin, label)
    raise ValueError("decision is outside the sealed action set")


def _partition(serial: int) -> str:
    if 1 <= serial <= EVIDENCE_COUNT:
        return "EVIDENCE"
    if EVIDENCE_COUNT < serial <= EVIDENCE_COUNT + VALIDATION_COUNT:
        return "VALIDATION"
    return "HOLDOUT"


def _prompt_row(serial: int) -> dict[str, object]:
    # Odd serials ask a false comparison. Even serials ask a true one.
    if serial % 2 == 1:
        left = 3000 + serial
        right = 4000 + serial
    else:
        left = 4000 + serial
        right = 3000 + serial
    label = YES if left > right else NO
    text = (
        f"RC1-{serial:04d} ledger. First sealed integer {left}. "
        f"Second sealed integer {right}. Is the first integer greater than the second?"
    )
    return {
        "family": "RC1",
        "label": label,
        "left": left,
        "partition": _partition(serial),
        "prompt_id": f"RC1-{serial:04d}",
        "prompt_sha256": prompt_sha256(text),
        "right": right,
        "serial": serial,
        "text": text,
    }


def _blocked_texts() -> set[str]:
    blocked = {normalize_prompt(row["text"]) for row in historical_prompts()}
    for relative in SUITE_CATALOGS:
        path = ROOT / relative
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        for episode in document.get("episodes", []):
            text = episode.get("prompt")
            if isinstance(text, str):
                blocked.add(normalize_prompt(text))
    return blocked


def build_catalog() -> dict[str, object]:
    prompts = [_prompt_row(serial) for serial in SERIALS]
    blocked = _blocked_texts()
    for row in prompts:
        if normalize_prompt(str(row["text"])) in blocked:
            raise RuntimeError(f"prompt overlaps a historical catalog: {row['prompt_id']}")
    normalized = [normalize_prompt(str(row["text"])) for row in prompts]
    if len(set(normalized)) != len(normalized):
        raise RuntimeError("catalog prompts are not unique")
    counts = {"EVIDENCE": 0, "VALIDATION": 0, "HOLDOUT": 0}
    for row in prompts:
        counts[str(row["partition"])] += 1
    if counts != {"EVIDENCE": 12, "VALIDATION": 12, "HOLDOUT": 24}:
        raise RuntimeError("catalog partitions drifted")
    return {
        "assignment": "serial order 1-12 EVIDENCE, 13-24 VALIDATION, 25-48 HOLDOUT",
        "catalog_version": CATALOG_VERSION,
        "claim_id": CLAIM_ID,
        "execution_authorized_by_catalog": False,
        "family": "RC1",
        "prompts": prompts,
        "validation_role": "frozen and not executed; not an input to the claim or the verdict",
    }


def historical_prior() -> dict[str, object]:
    results_path = ROOT / M24_RESULTS
    protocol_path = ROOT / M24_PROTOCOL
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("verdict") != M24_VERDICT:
        raise RuntimeError("M24 provenance verdict changed")
    if results.get("direction_sha256") != DIRECTION_SHA256:
        raise RuntimeError("M24 direction hash does not match D1")
    return {
        "evidence_id": "M24-HISTORICAL-PRIOR",
        "m24_protocol_sha256": file_sha256(protocol_path),
        "m24_results_sha256": file_sha256(results_path),
        "m24_verdict": M24_VERDICT,
        "note": "Historical candidate only. Its outcomes are not in the current positive fraction.",
        "source": "historical_prior",
        "source_is_current_test_evidence": False,
        "source_protocol": "M24",
    }


def initial_claim() -> dict[str, object]:
    prior = historical_prior()
    return {
        "claim_id": CLAIM_ID,
        "claim_type": CLAIM_TYPE,
        "evidence_history": [prior],
        "proposition": PROPOSITION,
        "revision_history": [],
        "scope": SCOPE,
        "source": "historical_prior",
        "source_is_current_test_evidence": False,
        "source_protocol": "M24",
        "status": STATUS_SUPPORTED,
        "uncertainty": None,
        "version": 1,
    }


def update_claim(claim: dict[str, object], evidence_rows: list[dict[str, object]]) -> dict[str, object]:
    """Apply the frozen rule. The previous claim record is copied, not edited."""
    if claim["version"] != 1 or claim["status"] != STATUS_SUPPORTED:
        raise RuntimeError("the update expects the historical prior")
    if any(row["partition"] != "EVIDENCE" for row in evidence_rows):
        raise RuntimeError("a non-evidence row entered the update")
    summary = status_from_effects([float(row["observed_effect"]) for row in evidence_rows])
    prior_snapshot = {
        "evidence_ids": [row["evidence_id"] for row in claim["evidence_history"]],
        "source": claim["source"],
        "status": claim["status"],
        "uncertainty": claim["uncertainty"],
        "version": claim["version"],
    }
    current_rows = []
    for row in evidence_rows:
        current_rows.append(
            {
                "evidence_id": row["prompt_id"],
                "observed_effect": row["observed_effect"],
                "source": "current_qwen_evidence",
                "source_is_current_test_evidence": True,
            }
        )
    updated = {
        "claim_id": claim["claim_id"],
        "claim_type": claim["claim_type"],
        "evidence_history": list(claim["evidence_history"]) + current_rows,
        "evidence_summary": summary,
        "proposition": claim["proposition"],
        "revision_history": list(claim["revision_history"]) + [prior_snapshot],
        "scope": claim["scope"],
        "source": "current_qwen_evidence",
        "source_is_current_test_evidence": True,
        "source_protocol": PROTOCOL_VERSION,
        "status": summary["status"],
        "uncertainty": summary["uncertainty"],
        "version": 2,
    }
    if claim["evidence_history"][0] != updated["evidence_history"][0]:
        raise RuntimeError("historical prior was rewritten")
    return updated


def holdout_decisions(claim: dict[str, object], prompt_ids: list[str]) -> list[dict[str, object]]:
    status = applicable_status(claim, SCOPE)
    action = decide(status)
    return [
        {
            "claim_id": claim["claim_id"],
            "claim_version": claim["version"],
            "decision": action,
            "prompt_id": prompt_id,
            "status": status,
        }
        for prompt_id in prompt_ids
    ]


def counterfactual_changes_decision(status: str) -> dict[str, object]:
    """Software check. No model call."""
    alternate = STATUS_UNCERTAIN if status == STATUS_SUPPORTED else STATUS_SUPPORTED
    return {
        "actual_decision": decide(status),
        "actual_status": status,
        "alternate_decision": decide(alternate),
        "alternate_status": alternate,
        "changed": decide(status) != decide(alternate),
    }


def verdict(
    *,
    qwen_executed: bool,
    claim_exact: bool,
    evidence_before_holdout: bool,
    scope_pass: bool,
    history_pass: bool,
    decision_from_claim: bool,
    causality_pass: bool,
    leakage_pass: bool,
    immutable_pass: bool,
    utility_better: bool,
    revision_from_qwen: bool,
    stores_evidence: bool,
    decisions_equal_always_intervene: bool,
) -> str:
    supported = all(
        (
            qwen_executed,
            claim_exact,
            evidence_before_holdout,
            scope_pass,
            history_pass,
            decision_from_claim,
            causality_pass,
            leakage_pass,
            immutable_pass,
            utility_better,
            revision_from_qwen,
        )
    )
    if supported:
        return "REAL_CLAIM_SELF_MODEL_SUPPORTED"
    not_supported = (
        (not stores_evidence)
        or (not decision_from_claim)
        or (not leakage_pass)
        or ((not utility_better) and decisions_equal_always_intervene)
    )
    if not_supported:
        return "REAL_CLAIM_SELF_MODEL_NOT_SUPPORTED"
    return "INCONCLUSIVE"
