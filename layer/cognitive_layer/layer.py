"""CognitiveLayer v0.2: before/after loop around any agent step.

    pred = layer.before(context, confidence=0.83)   # seals p_hat, EU and the action
    ... the agent acts according to pred.action ...
    layer.after(pred.prediction_id, success=..., provenance=...)

`success` is whether the agent's own unverified answer was correct. After
ABSTAIN or DELEGATE it is unknown and must be None.

Modes (for control arms; same code path, one switch each):
  mode="scoped"          full layer (default)
  mode="raw"             no calibration: p_hat = raw confidence (naive use of confidence)
  mode="base_rate_only"  ignores confidence: v0.1-style track record per scope
shuffle_evidence=True   evidence goes to a random declared scope (breaks self-specificity)
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Mapping

from .core import OBSERVING_ACTIONS, Action, Evidence, Prediction, Status, canonical_sha256, reject_forbidden
from .ledger import Ledger
from .policy import ExpectedUtilityPolicy
from .registry import CalibrationRegistry

MODES = {"scoped", "raw", "base_rate_only"}


class CognitiveLayer:
    def __init__(
        self,
        registry: CalibrationRegistry,
        policy: ExpectedUtilityPolicy | None = None,
        ledger: Ledger | None = None,
        mode: str = "scoped",
        shuffle_evidence: bool = False,
        seed: int = 0,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}")
        self.registry = registry
        self.policy = policy or ExpectedUtilityPolicy(seed=seed)
        self.ledger = ledger or Ledger()
        self.mode = mode
        self.shuffle_evidence = shuffle_evidence
        self.frozen = False
        self._rng = random.Random(seed + 7919)
        self._open: dict[str, Prediction] = {}
        self._ids = itertools.count(1)
        self.ledger.append("LAYER_CONFIG", {
            "mode": mode, "shuffle_evidence": shuffle_evidence, "rule": vars(registry.rule),
            "utility": vars(self.policy.utility), "epsilon": self.policy.epsilon,
            "claims": [c.snapshot() for c in registry.claims.values()],
        })

    # -- decision ---------------------------------------------------------
    def _probability(self, claim, confidence: float | None) -> tuple[float, str]:
        if self.mode == "raw":
            return (confidence, "raw") if confidence is not None else (0.5, "prior")
        if claim is None:
            return 0.5, "prior"
        if self.mode == "scoped" and claim.status == Status.SUPPORTED and confidence is not None:
            return claim.calibrated(confidence), "calibrated"
        return claim.base_rate(), "base_rate"

    def before(self, context: Mapping[str, Any], confidence: float | None = None) -> Prediction:
        reject_forbidden(context)
        claim, learning, chain = self.registry.resolve(context)
        if self.mode == "raw":
            learning = False
        p_hat, source = self._probability(claim, confidence)
        action, greedy, eu, explored, propensity = self.policy.choose(p_hat, learning, self.frozen)
        pred = Prediction(
            prediction_id=f"p{next(self._ids)}",
            context_sha256=canonical_sha256(dict(context)),
            raw_confidence=confidence,
            matching_claim_ids=tuple(c.claim_id for c in chain),
            used_claim_id=claim.claim_id if claim else None,
            used_claim_version=claim.version if claim else None,
            used_status=claim.status if claim else Status.NO_APPLICABLE_CLAIM,
            learning=learning,
            p_hat=float(p_hat),
            p_source=source,
            expected_utility=tuple((a.value, float(v)) for a, v in eu.items()),
            greedy_action=greedy,
            action=action,
            explored=explored,
            propensity=propensity,
        )
        self.ledger.append("PREDICTION_SEALED", {"seal": pred.seal_sha256, **{
            k: (v.value if hasattr(v, "value") else v) for k, v in vars(pred).items() if k != "seal_sha256"}})
        self._open[pred.prediction_id] = pred
        return pred

    # -- evidence ---------------------------------------------------------
    def after(self, prediction_id: str, success: bool | None, provenance: str,
              source_is_current: bool = True) -> Evidence:
        if prediction_id not in self._open:
            raise ValueError("outcome cites no open sealed prediction")
        pred = self._open[prediction_id]
        if success is not None and pred.action not in OBSERVING_ACTIONS:
            raise ValueError("the outcome of a non-observing action is unknown; pass success=None")
        self._open.pop(prediction_id)
        targets = list(pred.matching_claim_ids)
        if self.shuffle_evidence and targets:
            pool = sorted(self.registry.claims)
            targets = [self._rng.choice(pool)]
            provenance += "|shuffled"
        if self.frozen or self.mode == "raw":
            targets = []
        evidence = Evidence(
            evidence_id=f"e-{prediction_id}",
            prediction_id=prediction_id,
            prediction_seal_sha256=pred.seal_sha256,
            success=success,
            raw_confidence=pred.raw_confidence,
            weight=1.0 / pred.propensity,
            applied_to=tuple(targets),
            provenance=provenance,
            source_is_current=source_is_current,
        )
        self.ledger.append("EVIDENCE", evidence.payload())
        for claim_id in targets:
            updated = self.registry.apply(claim_id, evidence)
            self.ledger.append("CLAIM_REVISED", updated.snapshot())
        return evidence

    def freeze(self) -> None:
        """Holdout scoring: no exploration, no updates (score once)."""
        self.frozen = True
        self.ledger.append("FROZEN", {"claims": [c.snapshot() for c in self.registry.claims.values()]})


def replay(ledger: Ledger, rule=None) -> CalibrationRegistry:
    """Rebuild the registry from the ledger alone. State is a replay of evidence."""
    from .core import Scope
    from .registry import CalibrationRegistry as R

    config = next(ledger.of_kind("LAYER_CONFIG"))["payload"]
    reg = R(rule, with_global=False)
    for snap in config["claims"]:
        reg.declare(snap["claim_id"], snap["proposition"], Scope.of(**snap["scope"]))
    for event in ledger.of_kind("EVIDENCE"):
        p = dict(event["payload"])
        p["applied_to"] = tuple(p["applied_to"])
        ev = Evidence(**p)
        for claim_id in ev.applied_to:
            reg.apply(claim_id, ev)
    return reg
