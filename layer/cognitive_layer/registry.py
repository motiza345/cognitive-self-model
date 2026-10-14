"""Scoped calibration claims.

A claim is no longer "I succeed in scope S" (v0.1). It is:

    "In scope S, my confidence signal is (or is not) informative about my own
     correctness, and this is the map from confidence to probability."

Origins: minimal_self_model.ClaimRegistry (scope, status, uncertainty,
revision_history) and real_claim_self_model.update_claim (prior is copied,
historical evidence is recorded but does not count).

Status rule (frozen per registry, see CalibrationRule):
  - fewer than `min_each` correct OR fewer than `min_each` wrong outcomes -> UNCERTAIN
    (the M34a v3 gate: at least 15 of each)
  - AUROC lower bound > 0.5 + margin                                     -> SUPPORTED
  - AUROC upper bound < 0.5 + margin                                     -> CONTRADICTED
  - otherwise                                                            -> UNCERTAIN
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

from .calibration import IDENTITY, PlattFit, auroc, ece, fit_platt
from .core import GLOBAL, Evidence, Scope, Status, canonical_sha256


@dataclass(frozen=True)
class CalibrationRule:
    min_each: int = 15
    auroc_margin: float = 0.05
    alpha: float = 0.05
    prior_strength: float = 2.0
    name: str = "scoped_platt_auroc.v1"

    def status(self, n_pos: int, n_neg: int, lower: float | None, upper: float | None) -> Status:
        if n_pos < self.min_each or n_neg < self.min_each or lower is None:
            return Status.UNCERTAIN
        if lower > 0.5 + self.auroc_margin:
            return Status.SUPPORTED
        if upper < 0.5 + self.auroc_margin:
            return Status.CONTRADICTED
        return Status.UNCERTAIN


@dataclass(frozen=True)
class Observation:
    evidence_id: str
    confidence: float | None
    success: bool
    weight: float


@dataclass(frozen=True)
class CalibrationClaim:
    claim_id: str
    proposition: str
    scope: Scope
    status: Status = Status.UNCERTAIN
    platt: PlattFit = IDENTITY
    auroc: float | None = None
    auroc_ci: tuple[float | None, float | None] = (None, None)
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    observations: tuple[Observation, ...] = ()
    recorded_only: tuple[str, ...] = ()   # evidence ids recorded but not counted
    version: int = 1
    revision_history: tuple[str, ...] = field(default=())

    @property
    def n_pos(self) -> int:
        return sum(1 for o in self.observations if o.success)

    @property
    def n_neg(self) -> int:
        return sum(1 for o in self.observations if not o.success)

    def base_rate(self) -> float:
        """Weighted Beta(1, 1) posterior mean of own success."""
        s = sum(o.weight for o in self.observations if o.success)
        n = sum(o.weight for o in self.observations)
        return (s + 1.0) / (n + 2.0)

    def calibrated(self, confidence: float) -> float:
        return self.platt.apply(confidence)

    def snapshot(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "proposition": self.proposition,
            "scope": self.scope.as_dict(),
            "status": self.status.value,
            "platt": {"a": self.platt.a, "b": self.platt.b},
            "auroc": self.auroc,
            "auroc_ci": list(self.auroc_ci),
            "ece_raw": self.ece_raw,
            "ece_calibrated": self.ece_calibrated,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "base_rate": self.base_rate(),
            "evidence_ids": [o.evidence_id for o in self.observations],
            "recorded_only": list(self.recorded_only),
            "version": self.version,
            "revision_history": list(self.revision_history),
        }


class CalibrationRegistry:
    def __init__(self, rule: CalibrationRule | None = None, with_global: bool = True) -> None:
        self.rule = rule or CalibrationRule()
        self.claims: dict[str, CalibrationClaim] = {}
        if with_global:
            self.declare("GLOBAL", "My confidence is informative about my own correctness", GLOBAL)

    def declare(self, claim_id: str, proposition: str, scope: Scope) -> CalibrationClaim:
        if claim_id in self.claims:
            raise ValueError("claim already declared")
        if any(c.scope == scope for c in self.claims.values()):
            raise ValueError("a claim with this scope already exists")
        claim = CalibrationClaim(claim_id, proposition, scope)
        self.claims[claim_id] = claim
        return claim

    def matching(self, context: Mapping[str, Any]) -> list[CalibrationClaim]:
        """All claims whose scope matches, most specific first. Equal specificity is refused."""
        found = sorted((c for c in self.claims.values() if c.scope.matches(context)),
                       key=lambda c: c.scope.specificity, reverse=True)
        for a, b in zip(found, found[1:]):
            if a.scope.specificity == b.scope.specificity:
                raise ValueError(f"ambiguous scope: {a.claim_id} vs {b.claim_id}")
        return found

    def resolve(self, context: Mapping[str, Any]) -> tuple[CalibrationClaim | None, bool, list[CalibrationClaim]]:
        """Backoff: the most specific claim that has left UNCERTAIN.
        Returns (claim_used, learning, chain). learning is True while the most
        specific claim is still UNCERTAIN; the policy then keeps choosing
        actions that reveal the outcome."""
        chain = self.matching(context)
        if not chain:
            return None, True, chain
        learning = chain[0].status == Status.UNCERTAIN
        for claim in chain:
            if claim.status != Status.UNCERTAIN:
                return claim, learning, chain
        return chain[0], learning, chain

    def apply(self, claim_id: str, evidence: Evidence) -> CalibrationClaim:
        current = self.claims[claim_id]
        seen = {o.evidence_id for o in current.observations} | set(current.recorded_only)
        if evidence.evidence_id in seen:
            raise ValueError("evidence already applied")
        if evidence.success is None or not evidence.source_is_current:
            updated = replace(current, recorded_only=current.recorded_only + (evidence.evidence_id,))
        else:
            obs = current.observations + (
                Observation(evidence.evidence_id, evidence.raw_confidence, bool(evidence.success), evidence.weight),
            )
            updated = self._refit(replace(current, observations=obs))
        updated = replace(
            updated,
            version=current.version + 1,
            revision_history=current.revision_history + (canonical_sha256(current.snapshot()),),
        )
        self.claims[claim_id] = updated
        return updated

    def _refit(self, claim: CalibrationClaim) -> CalibrationClaim:
        rows = [o for o in claim.observations if o.confidence is not None]
        if not rows:
            status = self.rule.status(claim.n_pos, claim.n_neg, None, None)
            return replace(claim, status=status)
        c = np.array([o.confidence for o in rows])
        y = np.array([o.success for o in rows])
        w = np.array([o.weight for o in rows])
        disc = auroc(c, y, self.rule.alpha)
        status = self.rule.status(disc.n_pos, disc.n_neg, disc.lower, disc.upper)
        platt = fit_platt(c, y, w, prior_strength=self.rule.prior_strength)
        cal = np.array([platt.apply(x) for x in c])
        return replace(
            claim,
            status=status,
            platt=platt,
            auroc=disc.auroc,
            auroc_ci=(disc.lower, disc.upper),
            ece_raw=ece(c, y),
            ece_calibrated=ece(cal, y),
        )
