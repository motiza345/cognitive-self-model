"""One Qwen run of the sealed claim-registry decision test.

Hashes are checked before the model loads. An existing result is not rerun.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cognitive_self_model.real_claim_self_model import (  # noqa: E402
    ACTION_INTERVENE,
    BANNED_PREDICTION_KEYS,
    EXPECTED_CATALOG_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SCOPE,
    STATUS_SUPPORTED,
    PROPOSITION,
    applicable_status,
    build_catalog,
    canonical_sha256,
    choice_from_margin,
    counterfactual_changes_decision,
    decide,
    episode_utility,
    file_sha256,
    holdout_decisions,
    initial_claim,
    status_from_effects,
    update_claim,
    verdict,
)

PROTOCOL = ROOT / "reports" / "REAL_QWEN_CLAIM_SELF_MODEL_PROTOCOL.md"
CATALOG = ROOT / "reports" / "REAL_QWEN_CLAIM_SELF_MODEL_CATALOG.json"
RESULTS = ROOT / "reports" / "REAL_QWEN_CLAIM_SELF_MODEL_RESULTS.json"
EXECUTION = ROOT / "reports" / "REAL_QWEN_CLAIM_SELF_MODEL_EXECUTION.md"
EPISODES = ROOT / "reports" / "REAL_QWEN_CLAIM_SELF_MODEL_EPISODES.csv"
RAW = ROOT / "reports" / "real_qwen_claim_raw"
CLAIM_ID = "QWEN_SELF_MARGIN_GAIN_D1_L0_A1"


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bump_after(path: Path, previous: Path) -> None:
    prev = previous.stat().st_mtime_ns
    if path.stat().st_mtime_ns <= prev:
        new_ns = prev + 1_000_000
        os.utime(path, ns=(new_ns, new_ns))


def _keys(value: object, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).casefold())
            _keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _keys(child, found)


def _assert_prediction(payload: object) -> None:
    found: set[str] = set()
    _keys(payload, found)
    if found & BANNED_PREDICTION_KEYS:
        raise RuntimeError(f"prediction artifact contains {sorted(found & BANNED_PREDICTION_KEYS)}")


def _git_state() -> str:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return "CLEAN" if completed.stdout.strip() == "" else "DIRTY"


def _load_catalog() -> dict[str, object]:
    if file_sha256(PROTOCOL) != EXPECTED_PROTOCOL_SHA256:
        raise SystemExit("protocol hash mismatch; Qwen was not loaded")
    document = json.loads(CATALOG.read_text(encoding="utf-8"))
    if canonical_sha256(document) != EXPECTED_CATALOG_SHA256:
        raise SystemExit("catalog hash mismatch; Qwen was not loaded")
    if canonical_sha256(build_catalog()) != EXPECTED_CATALOG_SHA256:
        raise SystemExit("catalog rebuild mismatch; Qwen was not loaded")
    return document


def _rows(document: dict[str, object], partition: str) -> list[dict[str, object]]:
    return [row for row in document["prompts"] if row["partition"] == partition]


class Probe:
    def __init__(self) -> None:
        import torch
        from transformer_lens import HookedTransformer

        from src.cognitive_self_model.m23.m22_reuse import (
            logit_margin,
            make_resid_hook,
            primary_direction,
            resolve_outcome_tokens,
            vector_sha256,
        )

        torch.set_num_threads(os.cpu_count() or 1)
        self.torch = torch
        self.logit_margin = logit_margin
        self.make_resid_hook = make_resid_hook
        self.counts = {"clean_forwards": 0, "backwards": 0, "intervention_forwards": 0, "plain_forwards": 0}
        self.model = HookedTransformer.from_pretrained(
            MODEL_ID,
            device="cpu",
            dtype=torch.float32,
            revision=MODEL_REVISION,
        )
        self.model.eval()
        tokens = resolve_outcome_tokens(self.model.tokenizer, " yes", " no")
        if (tokens["positive_token_id"], tokens["negative_token_id"]) != (9834, 902):
            raise RuntimeError("token ids drifted")
        self.positive_id = 9834
        self.negative_id = 902
        direction = primary_direction(896, 22101)
        if vector_sha256(direction) != SCOPE["direction_sha256"]:
            raise RuntimeError("direction hash drifted")
        self.direction = torch.tensor(direction, dtype=torch.float32)
        self.direction_64 = self.direction.detach().to(dtype=torch.float64)

    def evidence_prediction(self, text: str) -> tuple[float, float]:
        hook_name = "blocks.0.hook_resid_post"
        tokens = self.model.to_tokens(text, prepend_bos=True)
        saved: dict[str, object] = {}

        def capture(residual, hook):
            if getattr(hook, "name", hook_name) not in {hook_name, ""}:
                raise RuntimeError("unexpected measurement hook")
            saved["residual"] = residual
            return residual

        self.model.zero_grad(set_to_none=True)
        logits = self.model.run_with_hooks(tokens, fwd_hooks=[(hook_name, capture)])
        margin_tensor = logits[0, -1, self.positive_id] - logits[0, -1, self.negative_id]
        gradient = self.torch.autograd.grad(margin_tensor, saved["residual"])[0]
        g_value = float(self.torch.dot(gradient[0, -1].detach().to(dtype=self.torch.float64), self.direction_64).item())
        margin = float(margin_tensor.detach().item())
        self.counts["clean_forwards"] += 1
        self.counts["backwards"] += 1
        self.model.zero_grad(set_to_none=True)
        return g_value, margin

    def baseline_margin(self, text: str) -> float:
        tokens = self.model.to_tokens(text, prepend_bos=True)
        with self.torch.no_grad():
            logits = self.model(tokens)
            margin = float(self.logit_margin(logits, self.positive_id, self.negative_id))
        self.counts["plain_forwards"] += 1
        return margin

    def intervened_margin(self, text: str) -> float:
        hook_name = "blocks.0.hook_resid_post"
        tokens = self.model.to_tokens(text, prepend_bos=True)
        recorder: dict[str, object] = {}
        hook_fn = self.make_resid_hook(1.0, self.direction, recorder)
        with self.torch.no_grad():
            logits = self.model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
            margin = float(self.logit_margin(logits, self.positive_id, self.negative_id))
        if int(recorder.get("fired", 0)) != 1 or not recorder.get("last_modified", False):
            raise RuntimeError("intervention hook did not modify the last token")
        self.counts["intervention_forwards"] += 1
        return margin


def _render(result: dict[str, object]) -> str:
    lines = [
        "# Real Qwen claim-based self-model execution",
        "",
        "One run. The protocol and catalog were not edited after the first outcome.",
        "",
        f"Protocol SHA-256: `{result['protocol_sha256']}`",
        "",
        f"Catalog SHA-256: `{result['catalog_sha256']}`",
        "",
        f"Model revision: `{result['model_revision']}`",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        f"Initial status: `{result['initial_status']}`",
        "",
        f"Evidence status: `{result['evidence_status']}`",
        "",
        f"Final status: `{result['final_status']}`",
        "",
        f"SelfModel decision: `{result['self_model_decision']}`",
        "",
        f"Baseline decision: `{result['baseline_decision']}`",
        "",
        f"SelfModel utility: `{result['self_model_utility']}`",
        "",
        f"Baseline utility: `{result['baseline_utility']}`",
        "",
        f"Utility difference: `{result['utility_difference']}`",
        "",
        f"Claim revision: `{result['claim_revision']}`",
        "",
        f"Decision causality: `{result['decision_causality']}`",
        "",
        f"Scope: `{result['scope']}`",
        "",
        f"Historical immutability: `{result['historical_immutability']}`",
        "",
        f"Leakage: `{result['leakage']}`",
        "",
        "Validation prompts were not executed.",
        "",
        "A supported verdict would not mean that Qwen has a general self-model.",
        "",
        "```json",
        json.dumps(result["evidence_summary"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if RESULTS.exists() or (RAW / "evidence_predictions.json").exists() or (RAW / "evidence_outcomes.json").exists():
        raise SystemExit("a claim-test artifact already exists; the run was not repeated")
    document = _load_catalog()
    claim = initial_claim()
    initial_path = RAW / "initial_claim.json"
    _dump(initial_path, claim)
    initial_hash = file_sha256(initial_path)
    print("REAL_QWEN_CLAIM_SELF_MODEL_TEST", flush=True)
    print(f"MODEL:\n{MODEL_ID}", flush=True)
    print(f"REVISION:\n{MODEL_REVISION}", flush=True)
    print(f"CLAIM:\n{CLAIM_ID}", flush=True)
    print("INTERVENTION:\nD1 / L0 / alpha=+1", flush=True)
    print(f"PROTOCOL_HASH:\n{EXPECTED_PROTOCOL_SHA256}", flush=True)
    print(f"CATALOG_HASH:\n{EXPECTED_CATALOG_SHA256}", flush=True)
    print("RUN:\nONE_RUN_ONLY", flush=True)

    evidence = _rows(document, "EVIDENCE")
    holdout = _rows(document, "HOLDOUT")
    probe = Probe()
    predictions = []
    for row in evidence:
        g_value, margin = probe.evidence_prediction(str(row["text"]))
        predictions.append(
            {
                "baseline_margin": margin,
                "g": g_value,
                "partition": "EVIDENCE",
                "prompt_id": row["prompt_id"],
            }
        )
    _assert_prediction(predictions)
    prediction_path = RAW / "evidence_predictions.json"
    _dump(prediction_path, predictions)
    _bump_after(prediction_path, initial_path)
    prediction_hash = file_sha256(prediction_path)

    by_prediction = {row["prompt_id"]: row for row in predictions}
    outcomes = []
    for row in evidence:
        baseline = float(by_prediction[row["prompt_id"]]["baseline_margin"])
        intervened = probe.intervened_margin(str(row["text"]))
        outcomes.append(
            {
                "baseline_margin": baseline,
                "intervened_margin": intervened,
                "observed_effect": intervened - baseline,
                "partition": "EVIDENCE",
                "prompt_id": row["prompt_id"],
            }
        )
    outcome_path = RAW / "evidence_outcomes.json"
    _dump(outcome_path, outcomes)
    _bump_after(outcome_path, prediction_path)
    if file_sha256(prediction_path) != prediction_hash or file_sha256(initial_path) != initial_hash:
        raise RuntimeError("a sealed file changed after evidence outcomes")

    updated = update_claim(claim, outcomes)
    summary = status_from_effects([float(row["observed_effect"]) for row in outcomes])
    if updated["evidence_summary"] != summary:
        raise RuntimeError("claim summary drifted from the frozen rule")
    claim_path = RAW / "updated_claim.json"
    _dump(claim_path, updated)
    _bump_after(claim_path, outcome_path)

    decisions = holdout_decisions(updated, [str(row["prompt_id"]) for row in holdout])
    _assert_prediction(decisions)
    decision_path = RAW / "holdout_decisions.json"
    _dump(decision_path, decisions)
    _bump_after(decision_path, claim_path)
    decision_by_id = {row["prompt_id"]: row for row in decisions}
    labels = {str(row["prompt_id"]): str(row["label"]) for row in holdout}
    texts = {str(row["prompt_id"]): str(row["text"]) for row in holdout}

    holdout_rows = []
    b2_utilities = []
    b0_utilities = []
    for row in decisions:
        prompt_id = str(row["prompt_id"])
        baseline = probe.baseline_margin(texts[prompt_id])
        intervened = probe.intervened_margin(texts[prompt_id])
        label = labels[prompt_id]
        b2_utility = episode_utility(str(row["decision"]), baseline, intervened, label)
        b0_utility = episode_utility(ACTION_INTERVENE, baseline, intervened, label)
        b2_utilities.append(b2_utility)
        b0_utilities.append(b0_utility)
        holdout_rows.append(
            {
                "baseline_margin": baseline,
                "choice_baseline": choice_from_margin(baseline),
                "choice_intervened": choice_from_margin(intervened),
                "decision_b0": ACTION_INTERVENE,
                "decision_b2": row["decision"],
                "intervened_margin": intervened,
                "label": label,
                "prompt_id": prompt_id,
                "utility_b0": b0_utility,
                "utility_b2": b2_utility,
            }
        )
    holdout_outcome_path = RAW / "holdout_outcomes.json"
    _dump(holdout_outcome_path, holdout_rows)
    _bump_after(holdout_outcome_path, decision_path)

    immutable = file_sha256(prediction_path) == prediction_hash and file_sha256(initial_path) == initial_hash
    order_ok = (
        initial_path.stat().st_mtime_ns
        < prediction_path.stat().st_mtime_ns
        < outcome_path.stat().st_mtime_ns
        < claim_path.stat().st_mtime_ns
        < decision_path.stat().st_mtime_ns
        < holdout_outcome_path.stat().st_mtime_ns
    )
    found: set[str] = set()
    _keys(json.loads(prediction_path.read_text(encoding="utf-8")), found)
    _keys(json.loads(decision_path.read_text(encoding="utf-8")), found)
    leakage_pass = order_ok and immutable and not (found & BANNED_PREDICTION_KEYS)
    status = applicable_status(updated, SCOPE)
    decision_from_claim = all(
        row["decision"] == decide(status) and row["status"] == status for row in decisions
    )
    causal = counterfactual_changes_decision(status)
    scope_pass = updated["scope"] == SCOPE and status != "NO_APPLICABLE_CLAIM"
    history_pass = (
        updated["version"] == 2
        and updated["revision_history"][0]["status"] == STATUS_SUPPORTED
        and updated["evidence_history"][0]["source_is_current_test_evidence"] is False
        and len(updated["evidence_history"]) == 13
    )
    if updated["status"] != summary["status"]:
        claim_revision = "FAIL"
    elif updated["status"] != STATUS_SUPPORTED:
        claim_revision = "PASS"
    else:
        claim_revision = "NOT_OBSERVED"
    revision_from_qwen = claim_revision != "FAIL"
    mean_b2 = sum(b2_utilities) / len(b2_utilities)
    mean_b0 = sum(b0_utilities) / len(b0_utilities)
    utility_better = mean_b2 > mean_b0
    decisions_equal = all(row["decision"] == ACTION_INTERVENE for row in decisions)
    final_status = str(updated["status"])
    label = verdict(
        qwen_executed=True,
        claim_exact=updated["claim_id"] == CLAIM_ID and updated["proposition"] == PROPOSITION,
        evidence_before_holdout=order_ok,
        scope_pass=scope_pass,
        history_pass=history_pass,
        decision_from_claim=decision_from_claim,
        causality_pass=bool(causal["changed"]),
        leakage_pass=leakage_pass,
        immutable_pass=immutable,
        utility_better=utility_better,
        revision_from_qwen=revision_from_qwen,
        stores_evidence=history_pass,
        decisions_equal_always_intervene=decisions_equal,
    )
    result = {
        "baseline_decision": ACTION_INTERVENE,
        "baseline_utility": mean_b0,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "claim": CLAIM_ID,
        "claim_revision": claim_revision,
        "compute": probe.counts,
        "decision_causality": "PASS" if causal["changed"] else "FAIL",
        "evidence_before_holdout": order_ok,
        "evidence_status": final_status,
        "evidence_summary": summary,
        "final_status": final_status,
        "git": _git_state(),
        "historical_immutability": "PASS" if immutable else "FAIL",
        "initial_status": STATUS_SUPPORTED,
        "leakage": "PASS" if leakage_pass else "FAIL",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "scope": "PASS" if scope_pass else "FAIL",
        "self_model_decision": decide(status),
        "self_model_utility": mean_b2,
        "utility_difference": mean_b2 - mean_b0,
        "validation_executed": False,
        "verdict": label,
    }
    _dump(RESULTS, result)
    EXECUTION.write_text(_render(result), encoding="utf-8")
    fieldnames = [
        "prompt_id",
        "partition",
        "label",
        "baseline_margin",
        "intervened_margin",
        "observed_effect",
        "g",
        "decision_b2",
        "decision_b0",
        "utility_b2",
        "utility_b0",
    ]
    outcome_by_id = {row["prompt_id"]: row for row in outcomes}
    holdout_by_id = {row["prompt_id"]: row for row in holdout_rows}
    with EPISODES.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in document["prompts"]:
            prompt_id = str(row["prompt_id"])
            record = {
                "baseline_margin": "",
                "decision_b0": "",
                "decision_b2": "",
                "g": "",
                "intervened_margin": "",
                "label": row["label"],
                "observed_effect": "",
                "partition": row["partition"],
                "prompt_id": prompt_id,
                "utility_b0": "",
                "utility_b2": "",
            }
            if prompt_id in outcome_by_id:
                item = outcome_by_id[prompt_id]
                record.update(
                    {
                        "baseline_margin": item["baseline_margin"],
                        "g": by_prediction[prompt_id]["g"],
                        "intervened_margin": item["intervened_margin"],
                        "observed_effect": item["observed_effect"],
                    }
                )
            if prompt_id in holdout_by_id:
                item = holdout_by_id[prompt_id]
                record.update(
                    {
                        "baseline_margin": item["baseline_margin"],
                        "decision_b0": item["decision_b0"],
                        "decision_b2": item["decision_b2"],
                        "intervened_margin": item["intervened_margin"],
                        "utility_b0": item["utility_b0"],
                        "utility_b2": item["utility_b2"],
                    }
                )
            writer.writerow(record)
    if decision_by_id.keys() != {row["prompt_id"] for row in holdout}:
        raise RuntimeError("holdout decisions do not match the catalog")
    print("REAL_QWEN_CLAIM_SELF_MODEL_COMPLETE", flush=True)
    print(f"Protocol hash:\n{EXPECTED_PROTOCOL_SHA256}", flush=True)
    print(f"Catalog hash:\n{EXPECTED_CATALOG_SHA256}", flush=True)
    print(f"Model revision:\n{MODEL_REVISION}", flush=True)
    print(f"Claim:\n{CLAIM_ID}", flush=True)
    print(f"Initial status:\n{STATUS_SUPPORTED}", flush=True)
    print(f"Evidence status:\n{final_status}", flush=True)
    print(f"Final status:\n{final_status}", flush=True)
    print(f"SelfModel decision:\n{decide(status)}", flush=True)
    print(f"Baseline decision:\n{ACTION_INTERVENE}", flush=True)
    print(f"SelfModel utility:\n{mean_b2}", flush=True)
    print(f"Baseline utility:\n{mean_b0}", flush=True)
    print(f"Claim revision:\n{claim_revision}", flush=True)
    print(f"Decision causality:\n{'PASS' if causal['changed'] else 'FAIL'}", flush=True)
    print(f"Scope:\n{'PASS' if scope_pass else 'FAIL'}", flush=True)
    print(f"Historical immutability:\n{'PASS' if immutable else 'FAIL'}", flush=True)
    print(f"Leakage:\n{'PASS' if leakage_pass else 'FAIL'}", flush=True)
    print(f"Verdict:\n{label}", flush=True)
    print(f"Git:\n{result['git']}", flush=True)


if __name__ == "__main__":
    main()
