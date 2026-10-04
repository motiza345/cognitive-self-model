"""Synthetic claim-registry self-model.

The objects below are the sealed minimal architecture test. They do not
load a language model and they do not read earlier milestone artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PROTOCOL_VERSION = "MINIMAL_SELF_MODEL_ARCHITECTURE_TEST.1"
EXPECTED_PROTOCOL_SHA256 = "f9880f59e7e4103dc9ad9dd2a525745482dac2d74a50436186f289956b97e014"
RESERVED_SEED = 32001

STATE_A = "STATE_A"
STATE_B = "STATE_B"
STATE_C = "STATE_C"
STATES = (STATE_A, STATE_B, STATE_C)

ACTION_X = "ACTION_X"
ABSTAIN = "ABSTAIN"
GLOBAL_SCOPE = "GLOBAL"
CLAIM_TYPE = "MECHANISM"
PROPOSITION = "ACTION_X produces a positive self-effect"

CLAIM_USE_LINEAR = "USE_LINEAR"
CLAIM_WITHHOLD = "WITHHOLD"
STATUS_SUPPORTED = "SUPPORTED"
STATUS_UNCERTAIN = "UNCERTAIN"
STATUS_CONTRADICTED = "CONTRADICTED"

GROUND_TRUTH_EFFECT = {
    STATE_A: 1.0,
    STATE_B: 0.25,
    STATE_C: -1.0,
}
ABSTAIN_EFFECT = 0.0

PACKET_KEYS = ("action", "evidence_id", "measured_state", "outcome")

# Order is the sealed episode list. Probes carry no evidence outcome.
EPISODES = (
    ("U1", "update", STATE_A, 1.0),
    ("P1", "probe", STATE_A, None),
    ("P2", "probe", STATE_C, None),
    ("U2", "update", STATE_A, 0.0),
    ("P3", "probe", STATE_A, None),
    ("U3", "update", STATE_A, -1.0),
    ("P4", "probe", STATE_A, None),
    ("U4", "update", STATE_B, 0.25),
    ("P5", "probe", STATE_B, None),
    ("P5b", "probe", STATE_B, None),
    ("P6", "probe", STATE_A, None),
    ("P7", "probe", STATE_C, None),
)


def classify_evidence(outcome: float) -> tuple[str, str]:
    """Return (class, strength) from the evidence outcome alone."""
    if isinstance(outcome, bool) or not isinstance(outcome, (int, float)):
        raise TypeError("outcome must be a real number")
    value = float(outcome)
    if value > 0.5:
        return "CONFIRMING", "strong"
    if value > 0.0:
        return "CONFIRMING", "weak"
    if value == 0.0:
        return "AMBIGUOUS", "none"
    return "CONTRADICTORY", "strong"


def target_status(kind: str, strength: str) -> tuple[str, float]:
    table = {
        ("CONFIRMING", "strong"): (STATUS_SUPPORTED, 0.15),
        ("CONFIRMING", "weak"): (STATUS_SUPPORTED, 0.35),
        ("AMBIGUOUS", "none"): (STATUS_UNCERTAIN, 0.70),
        ("CONTRADICTORY", "strong"): (STATUS_CONTRADICTED, 0.90),
    }
    try:
        return table[(kind, strength)]
    except KeyError as exc:
        raise ValueError("evidence class is outside the sealed table") from exc


def correct_action(measured_state: str) -> str:
    effect = GROUND_TRUTH_EFFECT[measured_state]
    if effect > 0.0:
        return ACTION_X
    return ABSTAIN


def realized_effect(measured_state: str, decision: str) -> float:
    if decision == ACTION_X:
        return GROUND_TRUTH_EFFECT[measured_state]
    if decision == ABSTAIN:
        return ABSTAIN_EFFECT
    raise ValueError("decision is not in the sealed action set")


def primary_episode_ids() -> list[str]:
    last_update_state = None
    selected = []
    for episode_id, role, measured_state, _outcome in EPISODES:
        if role == "update":
            last_update_state = measured_state
        elif measured_state != last_update_state:
            selected.append(episode_id)
    return selected


def _freeze(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _check_packet(packet: dict) -> None:
    if set(packet) != set(PACKET_KEYS):
        raise ValueError("packet fields are not the sealed measurement set")
    if packet["action"] != ACTION_X:
        raise ValueError("action is not ACTION_X")
    if packet["measured_state"] not in GROUND_TRUTH_EFFECT:
        raise ValueError("measured state is outside the environment")
    classify_evidence(packet["outcome"])


def _packet_copy(episode_id: str, measured_state: str, outcome: float) -> dict:
    return {
        "action": ACTION_X,
        "evidence_id": episode_id,
        "measured_state": measured_state,
        "outcome": outcome,
    }


@dataclass(frozen=True)
class Claim:
    claim_type: str
    proposition: str
    scope: str
    status: str
    uncertainty: float
    evidence_ids: tuple[str, ...]
    version: int
    revision_history: tuple[str, ...]


class FixedPolicy:
    """B0. The action is fixed. Measurements are checked and discarded."""

    name = "B0"

    def __init__(self) -> None:
        self.predictions: tuple[str, ...] = ()

    def observe(self, packet: dict) -> None:
        _check_packet(packet)

    def decide(self, measured_state: str) -> str:
        if measured_state not in GROUND_TRUTH_EFFECT:
            raise ValueError("measured state is outside the environment")
        return ACTION_X

    def seal(self, episode_id: str, measured_state: str, decision: str) -> str:
        record = _freeze(
            {
                "decision": decision,
                "episode_id": episode_id,
                "measured_state": measured_state,
                "stored": False,
            }
        )
        self.predictions = self.predictions + (record,)
        return record


class LegacyTwoState:
    """B1. One absorbing flag. No scope and no uncertainty."""

    name = "B1"

    def __init__(self) -> None:
        self.claim = CLAIM_USE_LINEAR
        self.evidence_history: tuple[str, ...] = ()
        self.predictions: tuple[str, ...] = ()

    def observe(self, packet: dict) -> None:
        _check_packet(packet)
        kind, strength = classify_evidence(packet["outcome"])
        record = _freeze(
            {
                "evidence_class": kind,
                "evidence_id": packet["evidence_id"],
                "measured_state": packet["measured_state"],
                "outcome": packet["outcome"],
                "strength": strength,
            }
        )
        self.evidence_history = self.evidence_history + (record,)
        if self.claim == CLAIM_WITHHOLD:
            return
        if kind in {"AMBIGUOUS", "CONTRADICTORY"}:
            self.claim = CLAIM_WITHHOLD

    def decide(self, measured_state: str) -> str:
        if measured_state not in GROUND_TRUTH_EFFECT:
            raise ValueError("measured state is outside the environment")
        if self.claim == CLAIM_USE_LINEAR:
            return ACTION_X
        if self.claim == CLAIM_WITHHOLD:
            return ABSTAIN
        raise RuntimeError("legacy claim left the sealed pair")

    def seal(self, episode_id: str, measured_state: str, decision: str) -> str:
        record = _freeze(
            {
                "claim": self.claim,
                "decision": decision,
                "episode_id": episode_id,
                "measured_state": measured_state,
            }
        )
        self.predictions = self.predictions + (record,)
        return record


class ClaimRegistry:
    """B2, or B2_NO_SCOPE when use_scope is false."""

    def __init__(self, use_scope: bool) -> None:
        self.use_scope = use_scope
        self.name = "B2" if use_scope else "B2_NO_SCOPE"
        self.claims: dict[tuple[str, str, str], Claim] = {}
        self.version = 0
        self.evidence_history: tuple[str, ...] = ()
        self.revision_history: tuple[str, ...] = ()
        self.predictions: tuple[str, ...] = ()

    def scope_for(self, measured_state: str) -> str:
        if self.use_scope:
            return measured_state
        return GLOBAL_SCOPE

    def applicable(self, measured_state: str) -> Claim | None:
        if measured_state not in GROUND_TRUTH_EFFECT:
            raise ValueError("measured state is outside the environment")
        key = (CLAIM_TYPE, PROPOSITION, self.scope_for(measured_state))
        return self.claims.get(key)

    def decide(self, measured_state: str) -> str:
        claim = self.applicable(measured_state)
        if claim is None:
            return ABSTAIN
        if claim.status == STATUS_SUPPORTED:
            return ACTION_X
        if claim.status in {STATUS_UNCERTAIN, STATUS_CONTRADICTED}:
            return ABSTAIN
        raise RuntimeError("claim status left the sealed set")

    def observe(self, packet: dict) -> None:
        _check_packet(packet)
        kind, strength = classify_evidence(packet["outcome"])
        status, uncertainty = target_status(kind, strength)
        evidence_id = str(packet["evidence_id"])
        scope = self.scope_for(str(packet["measured_state"]))
        record = _freeze(
            {
                "evidence_class": kind,
                "evidence_id": evidence_id,
                "measured_state": packet["measured_state"],
                "outcome": packet["outcome"],
                "scope": scope,
                "strength": strength,
            }
        )
        self.evidence_history = self.evidence_history + (record,)
        key = (CLAIM_TYPE, PROPOSITION, scope)
        current = self.claims.get(key)
        if current is not None and current.status == STATUS_CONTRADICTED:
            return
        if current is None:
            claim = Claim(
                claim_type=CLAIM_TYPE,
                proposition=PROPOSITION,
                scope=scope,
                status=status,
                uncertainty=uncertainty,
                evidence_ids=(evidence_id,),
                version=1,
                revision_history=(),
            )
        else:
            prior = _freeze(
                {
                    "evidence_ids": list(current.evidence_ids),
                    "proposition": current.proposition,
                    "scope": current.scope,
                    "status": current.status,
                    "uncertainty": current.uncertainty,
                    "version": current.version,
                }
            )
            claim = Claim(
                claim_type=current.claim_type,
                proposition=current.proposition,
                scope=current.scope,
                status=status,
                uncertainty=uncertainty,
                evidence_ids=current.evidence_ids + (evidence_id,),
                version=current.version + 1,
                revision_history=current.revision_history + (prior,),
            )
        replaced = dict(self.claims)
        replaced[key] = claim
        self.claims = replaced
        self.version += 1
        self.revision_history = self.revision_history + (
            _freeze(
                {
                    "evidence_id": evidence_id,
                    "scope": scope,
                    "status": status,
                    "uncertainty": uncertainty,
                    "version": claim.version,
                }
            ),
        )

    def seal(self, episode_id: str, measured_state: str, decision: str) -> str:
        claim = self.applicable(measured_state)
        record = _freeze(
            {
                "claim_version": None if claim is None else claim.version,
                "decision": decision,
                "episode_id": episode_id,
                "measured_state": measured_state,
                "registry_version": self.version,
                "scope": None if claim is None else claim.scope,
                "status": None if claim is None else claim.status,
                "uncertainty": None if claim is None else claim.uncertainty,
            }
        )
        self.predictions = self.predictions + (record,)
        return record

    def add_claim(self, scope: str, status: str, uncertainty: float) -> None:
        """Insert one claim. Used only by the sealed bypass probe."""
        if status not in {STATUS_SUPPORTED, STATUS_UNCERTAIN, STATUS_CONTRADICTED}:
            raise ValueError("status left the sealed set")
        key = (CLAIM_TYPE, PROPOSITION, scope)
        self.claims = dict(self.claims)
        self.claims[key] = Claim(
            claim_type=CLAIM_TYPE,
            proposition=PROPOSITION,
            scope=scope,
            status=status,
            uncertainty=uncertainty,
            evidence_ids=(),
            version=1,
            revision_history=(),
        )


def claim_view(model: ClaimRegistry) -> list[dict]:
    rows = []
    for claim in model.claims.values():
        rows.append(
            {
                "claim_type": claim.claim_type,
                "evidence_ids": list(claim.evidence_ids),
                "proposition": claim.proposition,
                "revision_history": [json.loads(item) for item in claim.revision_history],
                "scope": claim.scope,
                "status": claim.status,
                "uncertainty": claim.uncertainty,
                "version": claim.version,
            }
        )
    rows.sort(key=lambda row: str(row["scope"]))
    return rows


def _row_by_id(rows: list[dict], episode_id: str) -> dict:
    for row in rows:
        if row["episode_id"] == episode_id:
            return row
    raise KeyError(episode_id)


def _claim_by_scope(snapshot: list[dict], scope: str) -> dict | None:
    for row in snapshot:
        if row["scope"] == scope:
            return row
    return None


def run_architecture_test(protocol_sha256: str) -> dict[str, object]:
    """Execute the sealed episode list once in memory."""
    agents = {
        "B0": FixedPolicy(),
        "B1": LegacyTwoState(),
        "B2_NO_SCOPE": ClaimRegistry(use_scope=False),
        "B2": ClaimRegistry(use_scope=True),
    }
    b2 = agents["B2"]
    b1 = agents["B1"]
    rows: list[dict] = []
    snapshots: dict[str, list[dict]] = {}
    b1_after: dict[str, str] = {}
    last_update_state = None
    sealed_predictions: dict[str, tuple[str, ...]] = {name: () for name in agents}
    sealed_evidence: dict[str, tuple[str, ...]] = {
        "B1": (),
        "B2": (),
        "B2_NO_SCOPE": (),
    }
    history_immutable = True
    information_parity = True

    for episode_id, role, measured_state, outcome in EPISODES:
        if role == "update":
            packet = _packet_copy(episode_id, measured_state, float(outcome))
            if set(packet) != set(PACKET_KEYS):
                information_parity = False
            kind, strength = classify_evidence(packet["outcome"])
            for name in ("B1", "B2_NO_SCOPE", "B2"):
                before_predictions = agents[name].predictions if name != "B1" else ()
                before_evidence = agents[name].evidence_history
                agents[name].observe(dict(packet))
                if not before_evidence == agents[name].evidence_history[: len(before_evidence)]:
                    history_immutable = False
                if name != "B1" and before_predictions != agents[name].predictions[: len(before_predictions)]:
                    history_immutable = False
            agents["B0"].observe(dict(packet))
            last_update_state = measured_state
            snapshots[episode_id] = claim_view(b2)
            b1_after[episode_id] = b1.claim
            for name in sealed_evidence:
                sealed_evidence[name] = agents[name].evidence_history
            rows.append(
                {
                    "correct_action": None,
                    "decision_B0": None,
                    "decision_B1": None,
                    "decision_B2": None,
                    "decision_B2_NO_SCOPE": None,
                    "episode_id": episode_id,
                    "evidence_class": kind,
                    "evidence_outcome": outcome,
                    "evidence_strength": strength,
                    "ground_truth_effect": GROUND_TRUTH_EFFECT[measured_state],
                    "measured_state": measured_state,
                    "primary": False,
                    "role": role,
                    "b2_scope": None,
                    "b2_status": None,
                }
            )
            continue

        decisions = {name: agent.decide(measured_state) for name, agent in agents.items()}
        for name, agent in agents.items():
            record = agent.seal(episode_id, measured_state, decisions[name])
            if not sealed_predictions[name] == agent.predictions[: len(sealed_predictions[name])]:
                history_immutable = False
            if agent.predictions[-1] != record:
                history_immutable = False
            sealed_predictions[name] = agent.predictions
        claim = b2.applicable(measured_state)
        primary = measured_state != last_update_state
        expected = correct_action(measured_state)
        rows.append(
            {
                "b2_scope": None if claim is None else claim.scope,
                "b2_status": None if claim is None else claim.status,
                "correct_action": expected,
                "decision_B0": decisions["B0"],
                "decision_B1": decisions["B1"],
                "decision_B2": decisions["B2"],
                "decision_B2_NO_SCOPE": decisions["B2_NO_SCOPE"],
                "episode_id": episode_id,
                "evidence_class": None,
                "evidence_outcome": None,
                "evidence_strength": None,
                "ground_truth_effect": GROUND_TRUTH_EFFECT[measured_state],
                "measured_state": measured_state,
                "primary": primary,
                "role": role,
            }
        )

    for name in sealed_evidence:
        if sealed_evidence[name] != agents[name].evidence_history:
            history_immutable = False

    primary_rows = [row for row in rows if row["primary"]]
    primary_ids = [row["episode_id"] for row in primary_rows]
    if primary_ids != primary_episode_ids():
        raise RuntimeError("primary set drifted from the sealed rule")

    def _count(key: str) -> int:
        return sum(1 for row in primary_rows if row[key] == row["correct_action"])

    counts = {
        "B0": _count("decision_B0"),
        "B1": _count("decision_B1"),
        "B2_NO_SCOPE": _count("decision_B2_NO_SCOPE"),
        "B2": _count("decision_B2"),
    }
    primary_n = len(primary_rows)

    def _utility(key: str) -> float:
        return sum(realized_effect(row["measured_state"], row[key]) for row in primary_rows)

    utilities = {
        "B0": _utility("decision_B0"),
        "B1": _utility("decision_B1"),
        "B2_NO_SCOPE": _utility("decision_B2_NO_SCOPE"),
        "B2": _utility("decision_B2"),
    }

    scope_failures = []
    for row in rows:
        if row["role"] != "probe":
            continue
        if row["decision_B2"] == ACTION_X and (
            row["b2_scope"] != row["measured_state"] or row["b2_status"] != STATUS_SUPPORTED
        ):
            scope_failures.append(row["episode_id"])
        if row["measured_state"] == STATE_C and (row["b2_scope"] is not None or row["decision_B2"] != ABSTAIN):
            scope_failures.append(row["episode_id"])
    before_b = snapshots["U3"]
    after_b = snapshots["U4"]
    claim_a_before = _claim_by_scope(before_b, STATE_A)
    claim_a_after = _claim_by_scope(after_b, STATE_A)
    if claim_a_before != claim_a_after:
        scope_failures.append("U4_changed_STATE_A")
    scope_pass = not scope_failures

    revision_failures = []
    u1 = _claim_by_scope(snapshots["U1"], STATE_A)
    u2 = _claim_by_scope(snapshots["U2"], STATE_A)
    u3 = _claim_by_scope(snapshots["U3"], STATE_A)
    u4b = _claim_by_scope(snapshots["U4"], STATE_B)
    p1 = _row_by_id(rows, "P1")
    p3 = _row_by_id(rows, "P3")
    p4 = _row_by_id(rows, "P4")
    if u1 is None or u1["status"] != STATUS_SUPPORTED or u1["uncertainty"] != 0.15:
        revision_failures.append("U1")
    if p1["decision_B2"] != ACTION_X:
        revision_failures.append("P1")
    if (
        u2 is None
        or u2["status"] != STATUS_UNCERTAIN
        or u2["uncertainty"] != 0.70
        or u2["proposition"] != PROPOSITION
        or u2["status"] == STATUS_CONTRADICTED
    ):
        revision_failures.append("U2")
    if p3["decision_B2"] != ABSTAIN:
        revision_failures.append("P3")
    if u3 is None or u3["status"] != STATUS_CONTRADICTED or u3["uncertainty"] != 0.90:
        revision_failures.append("U3")
    if p4["decision_B2"] != ABSTAIN or p4["decision_B2"] == p1["decision_B2"]:
        revision_failures.append("P4")
    if u4b is None or u4b["status"] != STATUS_SUPPORTED or u4b["uncertainty"] != 0.35:
        revision_failures.append("U4")
    revision_pass = not revision_failures

    statuses = {row["status"] for snapshot in snapshots.values() for row in snapshot}
    epistemic_distinction = statuses == {STATUS_SUPPORTED, STATUS_UNCERTAIN, STATUS_CONTRADICTED}

    state_a_final = _claim_by_scope(snapshots["U4"], STATE_A)
    history_statuses = [] if state_a_final is None else [item["status"] for item in state_a_final["revision_history"]]
    representation_pass = (
        state_a_final is not None
        and state_a_final["status"] == STATUS_CONTRADICTED
        and STATUS_SUPPORTED in history_statuses
        and u4b is not None
        and u4b["status"] == STATUS_SUPPORTED
        and _claim_by_scope(snapshots["U4"], STATE_C) is None
        and state_a_final["scope"] != u4b["scope"]
    )

    causality_failures = []
    if not (p1["b2_status"] == STATUS_SUPPORTED and p1["decision_B2"] == ACTION_X):
        causality_failures.append("P1")
    if not (p3["b2_status"] == STATUS_UNCERTAIN and p3["decision_B2"] == ABSTAIN):
        causality_failures.append("P3")
    if not (p4["b2_status"] == STATUS_CONTRADICTED and p4["decision_B2"] == ABSTAIN):
        causality_failures.append("P4")
    if p1["decision_B2"] == p3["decision_B2"]:
        causality_failures.append("supported_to_uncertain_did_not_change_decision")
    if p3["decision_B2"] != p4["decision_B2"]:
        causality_failures.append("non_distinct_statuses_changed_decision")
    p5 = _row_by_id(rows, "P5")
    p5b = _row_by_id(rows, "P5b")
    if p5["decision_B2"] != p5b["decision_B2"] or p5["b2_status"] != p5b["b2_status"]:
        causality_failures.append("P5b")
    causality_pass = not causality_failures

    probe_ids = {row["episode_id"] for row in rows if row["role"] == "probe"}
    leakage_failures = []
    for name in ("B1", "B2", "B2_NO_SCOPE"):
        for item in agents[name].evidence_history:
            evidence_id = json.loads(item)["evidence_id"]
            if evidence_id in probe_ids:
                leakage_failures.append(evidence_id)
    if not history_immutable or not information_parity:
        leakage_failures.append("record")
    leakage_pass = not leakage_failures

    bypass = ClaimRegistry(use_scope=True)
    bypass_failures = []
    for state in STATES:
        if bypass.decide(state) != ABSTAIN:
            bypass_failures.append(f"empty:{state}")
    bypass.add_claim(STATE_C, STATUS_SUPPORTED, 0.15)
    if bypass.decide(STATE_C) != ACTION_X:
        bypass_failures.append("supported_state_c")
    if bypass.decide(STATE_A) != ABSTAIN or bypass.decide(STATE_B) != ABSTAIN:
        bypass_failures.append("other_states")
    bypass_pass = not bypass_failures

    probe_rows = [row for row in rows if row["role"] == "probe"]
    b2_equals_b1 = all(row["decision_B2"] == row["decision_B1"] for row in probe_rows)
    b2_equals_no_scope = all(row["decision_B2"] == row["decision_B2_NO_SCOPE"] for row in primary_rows)

    falsification = []
    if not representation_pass:
        falsification.append("cannot_represent_distinct_state_claims")
    if b2_equals_b1:
        falsification.append("identical_to_b1")
    if b2_equals_no_scope:
        falsification.append("scope_removal_same_primary_decisions")
    if not bypass_pass:
        falsification.append("hardcoded_bypass")
    if not information_parity:
        falsification.append("extra_information")
    if not epistemic_distinction:
        falsification.append("epistemic_states_not_distinct")
    if not history_immutable:
        falsification.append("historical_record_changed")

    rates = {name: counts[name] / primary_n for name in counts}
    no_benefit = not (counts["B2"] > counts["B2_NO_SCOPE"])
    not_supported = bool(falsification) or no_benefit or not causality_pass
    supported = (
        counts["B2"] > counts["B2_NO_SCOPE"]
        and counts["B2"] > counts["B1"]
        and revision_pass
        and scope_pass
        and leakage_pass
        and bypass_pass
        and representation_pass
        and causality_pass
        and history_immutable
        and epistemic_distinction
        and information_parity
        and not b2_equals_b1
        and not b2_equals_no_scope
        and not falsification
    )
    if supported:
        verdict = "ARCHITECTURE_SUPPORTED"
    elif not_supported:
        verdict = "ARCHITECTURE_NOT_SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"

    result: dict[str, object] = {
        "b1_claim_after_updates": b1_after,
        "b2_equals_b1": b2_equals_b1,
        "b2_equals_b2_no_scope_on_primary": b2_equals_no_scope,
        "bypass_failures": bypass_failures,
        "bypass_pass": bypass_pass,
        "causality_failures": causality_failures,
        "causality_pass": causality_pass,
        "epistemic_distinction": epistemic_distinction,
        "episodes": rows,
        "falsification": falsification,
        "ground_truth_effect": GROUND_TRUTH_EFFECT,
        "history_immutable": history_immutable,
        "information_parity": information_parity,
        "leakage_failures": leakage_failures,
        "leakage_pass": leakage_pass,
        "measured_state_is_observable": True,
        "model": None,
        "primary_correct_count": counts,
        "primary_correct_rate": rates,
        "primary_ids": primary_ids,
        "primary_n": primary_n,
        "primary_realized_effect": utilities,
        "protocol_sha256": protocol_sha256,
        "protocol_version": PROTOCOL_VERSION,
        "representation_pass": representation_pass,
        "reserved_seed": RESERVED_SEED,
        "reserved_seed_draws": 0,
        "revision_failures": revision_failures,
        "revision_pass": revision_pass,
        "scope_failures": scope_failures,
        "scope_pass": scope_pass,
        "snapshots": snapshots,
        "verdict": verdict,
    }
    result["answers"] = _answers(result)
    return result


def _fmt_rate(result: dict[str, object], name: str) -> str:
    count = int(result["primary_correct_count"][name])
    total = int(result["primary_n"])
    return f"{count}/{total}"


def _answers(result: dict[str, object]) -> dict[str, str]:
    episodes = result["episodes"]
    p1 = _row_by_id(episodes, "P1")
    p2 = _row_by_id(episodes, "P2")
    p3 = _row_by_id(episodes, "P3")
    snapshots = result["snapshots"]
    final_claims = snapshots["U4"]
    state_a = _claim_by_scope(final_claims, STATE_A)
    history = [] if state_a is None else [item["status"] for item in state_a["revision_history"]]
    current = None if state_a is None else state_a["status"]
    scopes = [row["scope"] for row in final_claims]
    b1_final = result["b1_claim_after_updates"]["U4"]
    misses = [
        row["episode_id"]
        for row in episodes
        if row["primary"] and row["decision_B2"] != row["correct_action"]
    ]
    rates = ", ".join(f"{name} {_fmt_rate(result, name)}" for name in ("B0", "B1", "B2_NO_SCOPE", "B2"))
    effects = result["primary_realized_effect"]
    effect_text = ", ".join(f"{name} {effects[name]}" for name in ("B0", "B1", "B2_NO_SCOPE", "B2"))
    higher = (
        result["primary_correct_count"]["B2"] > result["primary_correct_count"]["B1"]
        and result["primary_correct_count"]["B2"] > result["primary_correct_count"]["B2_NO_SCOPE"]
    )
    return {
        "1": (
            f"B1 ends as the single flag {b1_final}. "
            f"B2 ends with current claims at scopes {scopes}, each carrying status, uncertainty, evidence ids, and revision history. "
            f"The registry {'does' if result['representation_pass'] else 'does not'} hold STATE_A contradicted, STATE_B separate, and STATE_C absent at the same time."
        ),
        "2": (
            f"On P2 the measured state is STATE_C. B2 decided {p2['decision_B2']} with cited scope {p2['b2_scope']}. "
            f"B2_NO_SCOPE decided {p2['decision_B2_NO_SCOPE']}. B1 decided {p2['decision_B1']}. "
            f"Scope checks {'passed' if result['scope_pass'] else 'failed'}."
        ),
        "3": (
            f"The STATE_A claim history is {history}, and the current status is {current}. "
            f"Prior statuses remain in the revision history. B1 stores only {b1_final}."
        ),
        "4": (
            f"P1 read status {p1['b2_status']} and decided {p1['decision_B2']}. "
            f"P3 read status {p3['b2_status']} and decided {p3['decision_B2']}. "
            f"The decision {'changed' if p1['decision_B2'] != p3['decision_B2'] else 'did not change'} when the applicable status changed."
        ),
        "5": (
            f"Primary correct decisions: {rates}. Primary realized effects: {effect_text}. "
            f"B2 was incorrect on primary episodes: {misses or 'none'}. "
            f"The primary correct-decision rate for B2 {'was' if higher else 'was not'} strictly higher than both B1 and B2_NO_SCOPE."
        ),
        "6": (
            f"Verdict {result['verdict']}. "
            f"Scope handling {'matched' if result['scope_pass'] else 'did not match'} the sealed checks. "
            f"Revision {'matched' if result['revision_pass'] else 'did not match'} the sealed status sequence. "
            f"Decision causality {'held' if result['causality_pass'] else 'failed'}. "
            f"Falsification flags: {result['falsification'] or 'none'}. "
            f"B2 primary misses: {misses or 'none'}."
        ),
    }


def render_execution(result: dict[str, object]) -> str:
    answers = result["answers"]
    lines = [
        "# Minimal self-model architecture test execution",
        "",
        "One execution of the sealed protocol. No language model was loaded.",
        "",
        f"Protocol version: `{result['protocol_version']}`",
        "",
        f"Protocol SHA-256: `{result['protocol_sha256']}`",
        "",
        f"Reserved seed: `{result['reserved_seed']}` draws `{result['reserved_seed_draws']}`",
        "",
        f"Verdict: `{result['verdict']}`",
        "",
        "## Primary endpoint",
        "",
        "Correct decisions on held-out state transitions "
        f"({', '.join(result['primary_ids'])}).",
        "",
        "| Agent | Correct | Rate | Realized effect |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("B0", "B1", "B2_NO_SCOPE", "B2"):
        lines.append(
            f"| {name} | {result['primary_correct_count'][name]}/{result['primary_n']} | "
            f"{result['primary_correct_rate'][name]} | {result['primary_realized_effect'][name]} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- Scope: `{result['scope_pass']}` failures `{result['scope_failures'] or 'none'}`",
            f"- Revision: `{result['revision_pass']}` failures `{result['revision_failures'] or 'none'}`",
            f"- Decision causality: `{result['causality_pass']}` failures `{result['causality_failures'] or 'none'}`",
            f"- Representation: `{result['representation_pass']}`",
            f"- Historical immutability: `{result['history_immutable']}`",
            f"- Leakage: `{result['leakage_pass']}` failures `{result['leakage_failures'] or 'none'}`",
            f"- Bypass probe: `{result['bypass_pass']}` failures `{result['bypass_failures'] or 'none'}`",
            f"- Epistemic distinction: `{result['epistemic_distinction']}`",
            f"- B2 probe decisions identical to B1: `{result['b2_equals_b1']}`",
            f"- B2 primary decisions identical to B2_NO_SCOPE: `{result['b2_equals_b2_no_scope_on_primary']}`",
            "",
            "## Answers",
            "",
            "1. Did the Claim Registry representation add information that B1 did not have?",
            "",
            answers["1"],
            "",
            "2. Did scope prevent invalid transfer?",
            "",
            answers["2"],
            "",
            "3. Did evidence update the claim rather than merely change a flag?",
            "",
            answers["3"],
            "",
            "4. Did the changed claim alter a decision?",
            "",
            answers["4"],
            "",
            "5. Did that decision change improve held-out outcomes?",
            "",
            answers["5"],
            "",
            "6. Which exact part of the proposed theory was supported or falsified?",
            "",
            answers["6"],
            "",
            "Measured state was an observable given to every agent. The ground-truth effect was not an agent input.",
            "",
            "`measured_state` is observable. No agent received the ground-truth effect or the correct action.",
            "",
        ]
    )
    return "\n".join(lines)
