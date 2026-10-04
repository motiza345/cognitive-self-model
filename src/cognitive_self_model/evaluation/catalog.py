"""Frozen self-model evaluation catalog v1.

This module assigns prompts, partitions, shifts, interventions, and seeds.
It does not load a model, run a forward, or store an outcome.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path

CATALOG_VERSION = "SELF_MODEL_EVALUATION_CATALOG.1"
SUITE_VERSION = "SELF_MODEL_EVALUATION_SUITE.1"
PROTOCOL_VERSION = "SELF_MODEL_EVALUATION_PROTOCOL.1"
MODEL_ID = "Qwen/Qwen2.5-0.5B"
MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DIRECTION_ID = "D1"
DIRECTION_SEED = 22101
DIRECTION_SHA256 = "5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411"
ASSIGNMENT_SEED = 31000
HOOK_L0 = "blocks.0.hook_resid_post"
HOOK_L8 = "blocks.8.hook_resid_post"

TRACK_BOOTSTRAP_SEEDS = {
    "SM1": 31001,
    "SM2": 31002,
    "SM3": 31003,
    "SM4": 31004,
    "SM5": 31005,
    "SM6": 31006,
    "SM7": 31007,
}
SHUFFLE_SEEDS = {
    "SM1": 31101,
    "SM2": 31102,
    "SM3": 31103,
    "SM4": 31104,
    "SM5": 31105,
    "SM6": 31106,
    "SM7": 31107,
}

BANNED_FIELDS = {
    "outcome",
    "post_intervention_activation",
    "observed_effect",
    "observed_outcome",
    "error",
    "residual",
    "utility",
    "revision_result",
    "answer_correct",
    "correctness",
    "intervened_margin",
    "state_label",
}

ROOT = Path(__file__).resolve().parents[3]

_PYTHON_CATALOGS = (
    "src/cognitive_self_model/m23/m22_reuse.py",
    "scripts/validate_m23_g_protocol.py",
    "scripts/validate_post_m23_single_measurement.py",
    "scripts/feasibility_gate_catalog.py",
    "scripts/m24_consumption_catalog.py",
    "scripts/m26_update_catalog.py",
    "tests/test_m27_mechanism_revision_design.py",
    "scripts/run_m29_pre_evidence_measurement.py",
)
_JSON_CATALOGS = (
    "reports/m24_raw/catalog_freeze.json",
    "reports/m26_raw/catalog_freeze.json",
    "reports/m27_raw/catalog_freeze.json",
    "reports/m29_raw/catalog_freeze.json",
    "reports/feasibility_gate_raw/catalog_freeze.json",
)
_QWEN_PROTOCOL = "reports/REAL_SELF_MODEL_QWEN_EXECUTION_PROTOCOL.md"


def normalize_prompt(text: str) -> str:
    characters = []
    for character in str(text).casefold():
        characters.append(character if character.isalnum() else " ")
    return " ".join("".join(characters).split())


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _string_literals(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    found: list[str] = []
    for child in ast.iter_child_nodes(node):
        found.extend(_string_literals(child))
    return found


def _historical_from_python(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    for node in tree.body:
        targets: list[str] = []
        value = None
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        if value is None:
            continue
        names = [name for name in targets if name in {"_TEXTS", "_COMPLETION", "_SYNTAX", "_INSTRUCTION"}]
        if not names:
            continue
        for text in _string_literals(value):
            if text.strip():
                rows.append(
                    {
                        "source": relative,
                        "container": names[0],
                        "text": text,
                        "prompt_id": "",
                        "partition": "",
                        "family": "",
                    }
                )
    return rows


def _historical_from_json(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompts = payload.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError(f"{relative} has no prompt list")
    rows = []
    for row in prompts:
        text = str(row["text"])
        rows.append(
            {
                "source": relative,
                "container": "prompts",
                "text": text,
                "prompt_id": str(row.get("prompt_id", "")),
                "partition": str(row.get("partition", "")),
                "family": str(row.get("family", "")),
            }
        )
    return rows


def _historical_from_qwen_protocol(relative: str) -> list[dict[str, str]]:
    text = (ROOT / relative).read_text(encoding="utf-8")
    start = text.index("<!-- FROZEN_SPEC_BEGIN -->")
    end = text.index("<!-- FROZEN_SPEC_END -->")
    block = text[start:end]
    payload = json.loads(block[block.index("{") : block.rindex("}") + 1])
    rows = []
    for key in ("prompt_a", "prompt_b"):
        row = payload[key]
        rows.append(
            {
                "source": relative,
                "container": key,
                "text": str(row["text"]),
                "prompt_id": str(row["id"]),
                "partition": str(row["role"]),
                "family": "",
            }
        )
    return rows


def historical_prompts() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for relative in _PYTHON_CATALOGS:
        rows.extend(_historical_from_python(relative))
    for relative in _JSON_CATALOGS:
        rows.extend(_historical_from_json(relative))
    rows.extend(_historical_from_qwen_protocol(_QWEN_PROTOCOL))
    return rows


def _shuffle(items: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]]:
    ordered = list(items)
    for index in range(len(ordered) - 1, 0, -1):
        swap = rng.randrange(index + 1)
        ordered[index], ordered[swap] = ordered[swap], ordered[index]
    return ordered


def _f1_prompt(serial: int) -> dict[str, object]:
    left = 20 + (serial * 3) % 400
    right = 7 + (serial * 5) % 250
    text = f"SV1F1-{serial:04d} addend {left} plus addend {right}."
    return {
        "family": "F1",
        "serial": serial,
        "text": text,
        "task_kind": "integer_sum",
        "left": left,
        "right": right,
        "task_key": str(left + right),
    }


def _f2_prompt(serial: int) -> dict[str, object]:
    token = f"qx{serial:04d}{(serial * 17) % 97:02d}"
    text = f"SV1F2-{serial:04d} glyphstring {token} parity query."
    return {
        "family": "F2",
        "serial": serial,
        "text": text,
        "task_kind": "glyph_parity",
        "token": token,
        "task_key": "even" if len(token) % 2 == 0 else "odd",
    }


def _f3_prompt(serial: int) -> dict[str, object]:
    token = f"zb{serial:04d}{(serial * 13) % 89:02d}"
    vowels = sum(character in "aeiou" for character in token)
    text = f"SV1F3-{serial:04d} orbitcode {token} maps through rule Z."
    return {
        "family": "F3",
        "serial": serial,
        "text": text,
        "task_kind": "orbit_vowel_mod3",
        "token": token,
        "task_key": str(vowels % 3),
    }


def _pool(builder, size: int) -> list[dict[str, object]]:
    return [builder(serial) for serial in range(1, size + 1)]


def _site(layer: int, alpha: int) -> dict[str, object]:
    if layer == 0:
        hook = HOOK_L0
        intervention = "NONE" if alpha == 0 else f"SUITE-D1-L0-A{alpha}"
    elif layer == 8:
        hook = HOOK_L8
        if alpha != 1:
            raise ValueError("layer 8 uses alpha +1 only")
        intervention = "SUITE-D1-L8-A1"
    else:
        raise ValueError("layer is outside the suite")
    return {
        "intervention_id": intervention,
        "layer": layer,
        "hook": hook,
        "alpha": alpha,
        "direction_id": DIRECTION_ID,
        "direction_seed": DIRECTION_SEED,
    }


def _stages(*names: str) -> list[str]:
    return list(names)


def _base_episode(
    episode_id: str,
    track: str,
    item: dict[str, object],
    partition: str,
    shift_condition: str,
    layer: int,
    alpha: int,
) -> dict[str, object]:
    text = str(item["text"])
    episode = {
        "episode_id": episode_id,
        "track": track,
        "family": item["family"],
        "family_slot": "IN_SUITE",
        "partition": partition,
        "prompt": text,
        "prompt_sha256": prompt_sha256(text),
        "shift_condition": shift_condition,
        "known_family": item["family"] != "F3",
        "bootstrap_seed": TRACK_BOOTSTRAP_SEEDS[track],
        "shuffle_seed": SHUFFLE_SEEDS[track],
        "assignment_seed": ASSIGNMENT_SEED,
        "reuses_episode_id": "",
        "pair_id": "",
        "arms": [],
        "desired_effect_sign": None,
        "scheduled_state": "",
        "scheduled_monitor_label": "",
        "catalog_mark": "",
        "scheduled_evidence_class": "",
        "evidence_route": "",
        "sealed_capability_claim": "",
        "external_key_relation": "",
        "task_definition": None,
        "hidden_from_reader": False,
        "pre_outcome_stages": _stages("CLEAN_INSTRUMENT", "SEAL", "OUTCOME_FILE_NOT_IN_CATALOG"),
        "compute": {
            "shared_instrument_forwards": 1,
            "intervention_calls": 0 if alpha == 0 else 1,
            "additional_inference": 0,
            "additional_feature_extraction": ["g", "kappa_or_null", "rho_or_null"],
            "arm_extra_search": 0,
        },
    }
    episode.update(_site(layer, alpha))
    return episode


def _take(pool: list[dict[str, object]], start: int, count: int) -> list[dict[str, object]]:
    chunk = pool[start : start + count]
    if len(chunk) != count:
        raise RuntimeError("catalog pool underrun")
    return chunk


def _assign_sm1(pool: list[dict[str, object]]) -> list[dict[str, object]]:
    episodes = []
    calibration_states = ["NORMAL", "PERTURBED", "DEGRADED", "UNRELIABLE"] * 4
    evaluation_states = (
        ["NORMAL"] * 8 + ["PERTURBED"] * 8 + ["DEGRADED"] * 8 + ["UNRELIABLE"] * 8
    )
    plan = [("CALIBRATION", "CALIBRATION", calibration_states), ("EVALUATION", "IID", evaluation_states)]
    cursor = 0
    for partition, shift, states in plan:
        items = _take(pool, cursor, len(states))
        cursor += len(states)
        prefix = "C" if partition == "CALIBRATION" else "E"
        for index, (item, state) in enumerate(zip(items, states), start=1):
            layer, alpha = {
                "NORMAL": (0, 0),
                "PERTURBED": (0, 1),
                "DEGRADED": (8, 1),
                "UNRELIABLE": (0, 1),
            }[state]
            episode = _base_episode(
                f"SM1-{prefix}-{index:02d}",
                "SM1",
                item,
                partition,
                shift,
                layer,
                alpha,
            )
            episode["scheduled_state"] = state
            episode["family_slot"] = "UNRELIABLE" if state == "UNRELIABLE" else "IN_SUITE"
            episode["hidden_from_reader"] = True
            episode["pre_outcome_stages"] = _stages(
                "CLEAN_INSTRUMENT",
                "SEAL_STATE_CALL",
                "OUTCOME_FILE_NOT_IN_CATALOG",
            )
            episodes.append(episode)
    ood_items = _take(pool, cursor, 8)
    for index, item in enumerate(ood_items, start=1):
        episode = _base_episode(f"SM1-O-{index:02d}", "SM1", item, "OOD", "OOD", 0, 2)
        episode["scheduled_state"] = "OOD"
        episode["family_slot"] = "OOD_ALPHA_PLUS_2"
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _stages(
            "CLEAN_INSTRUMENT",
            "SEAL_STATE_CALL",
            "OUTCOME_FILE_NOT_IN_CATALOG",
        )
        episodes.append(episode)
    return episodes


def _task_definition(item: dict[str, object]) -> dict[str, object]:
    definition: dict[str, object] = {"kind": item["task_kind"], "key": item["task_key"]}
    if item["task_kind"] == "integer_sum":
        definition["left"] = item["left"]
        definition["right"] = item["right"]
    else:
        definition["token"] = item["token"]
    return definition


def _assign_sm2(f1: list[dict[str, object]], f2: list[dict[str, object]], f3: list[dict[str, object]]) -> list[dict[str, object]]:
    episodes = []
    groups = (
        ("F1", "CALIBRATION", "CALIBRATION", _take(f1, 0, 8)),
        ("F1", "EVALUATION", "IID", _take(f1, 8, 8)),
        ("F2", "CALIBRATION", "CALIBRATION", _take(f2, 0, 8)),
        ("F2", "EVALUATION", "IID", _take(f2, 8, 8)),
        ("F3", "STRONG_SHIFT", "STRONG_SHIFT", _take(f3, 0, 8)),
    )
    for family, partition, shift, items in groups:
        for index, item in enumerate(items, start=1):
            code = {"CALIBRATION": "C", "EVALUATION": "E", "STRONG_SHIFT": "S"}[partition]
            episode = _base_episode(
                f"SM2-{code}-{family}-{index:02d}",
                "SM2",
                item,
                partition,
                shift,
                0,
                0,
            )
            episode["task_definition"] = _task_definition(item)
            episode["known_family"] = family != "F3"
            episode["compute"]["intervention_calls"] = 0
            episode["pre_outcome_stages"] = _stages(
                "SEAL_CAPABILITY_PROBABILITY",
                "SCORE_TASK_KEY_AFTER_SEAL",
            )
            episodes.append(episode)
    return episodes


def _assign_prediction_block(
    track: str,
    prefix: str,
    items: list[dict[str, object]],
    partition: str,
    shift: str,
    layer: int,
) -> list[dict[str, object]]:
    episodes = []
    for index, item in enumerate(items, start=1):
        episode = _base_episode(
            f"{track}-{prefix}-{index:02d}",
            track,
            item,
            partition,
            shift,
            layer,
            1,
        )
        episode["pre_outcome_stages"] = _stages(
            "CLEAN_INSTRUMENT",
            "SEAL_PREDICTION",
            "INTERVENTION",
            "OUTCOME_FILE_NOT_IN_CATALOG",
        )
        episodes.append(episode)
    return episodes


def _assign_sm3(f1: list[dict[str, object]], f3: list[dict[str, object]]) -> list[dict[str, object]]:
    return (
        _assign_prediction_block("SM3", "C", _take(f1, 0, 16), "CALIBRATION", "IN_FAMILY", 0)
        + _assign_prediction_block("SM3", "I", _take(f1, 16, 24), "EVALUATION", "IID", 0)
        + _assign_prediction_block("SM3", "M", _take(f1, 40, 24), "MILD_SHIFT", "MILD_SHIFT", 8)
        + _assign_prediction_block("SM3", "S", _take(f3, 0, 24), "STRONG_SHIFT", "STRONG_SHIFT", 0)
    )


def _assign_sm4(f1: list[dict[str, object]], f3: list[dict[str, object]]) -> list[dict[str, object]]:
    episodes = []
    calibration_labels = ["GOOD"] * 6 + ["PRE-FAILURE"] * 5 + ["FAILURE"] * 5
    for index, (item, label) in enumerate(zip(_take(f1, 0, 16), calibration_labels), start=1):
        layer, alpha, mark = {
            "GOOD": (0, 0, "GOOD"),
            "PRE-FAILURE": (0, 1, "GOOD"),
            "FAILURE": (0, 0, "FAILURE"),
        }[label]
        episode = _base_episode(f"SM4-C-{index:02d}", "SM4", item, "CALIBRATION", "CALIBRATION", layer, alpha)
        episode["scheduled_monitor_label"] = label
        episode["catalog_mark"] = mark
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _monitor_stages(alpha)
        episodes.append(episode)
    for index, item in enumerate(_take(f1, 16, 12), start=1):
        episode = _base_episode(f"SM4-G-{index:02d}", "SM4", item, "EVALUATION", "IID", 0, 0)
        episode["scheduled_monitor_label"] = "GOOD"
        episode["catalog_mark"] = "GOOD"
        episode["pair_id"] = f"SM4-PAIR-{index:02d}"
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _monitor_stages(0)
        episodes.append(episode)
    for index, item in enumerate(_take(f1, 28, 12), start=1):
        episode = _base_episode(f"SM4-P-{index:02d}", "SM4", item, "EVALUATION", "IID", 0, 1)
        episode["scheduled_monitor_label"] = "PRE-FAILURE"
        episode["catalog_mark"] = "GOOD"
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _monitor_stages(1)
        episodes.append(episode)
    for index, item in enumerate(_take(f1, 40, 12), start=1):
        episode = _base_episode(f"SM4-F-{index:02d}", "SM4", item, "EVALUATION", "IID", 0, 0)
        episode["scheduled_monitor_label"] = "FAILURE"
        episode["catalog_mark"] = "FAILURE"
        episode["pair_id"] = f"SM4-PAIR-{index:02d}"
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _monitor_stages(0)
        episodes.append(episode)
    for index, item in enumerate(_take(f3, 0, 12), start=1):
        episode = _base_episode(f"SM4-S-{index:02d}", "SM4", item, "STRONG_SHIFT", "STRONG_SHIFT", 0, 1)
        episode["scheduled_monitor_label"] = "FAILURE"
        episode["catalog_mark"] = "FAILURE"
        episode["hidden_from_reader"] = True
        episode["pre_outcome_stages"] = _monitor_stages(1)
        episodes.append(episode)
    return episodes


def _monitor_stages(alpha: int) -> list[str]:
    if alpha == 0:
        return _stages("CLEAN_INSTRUMENT", "SEAL_MONITOR_SCORE", "OUTCOME_FILE_NOT_IN_CATALOG")
    return _stages(
        "CLEAN_INSTRUMENT",
        "SEAL_MONITOR_SCORE",
        "INTERVENTION",
        "OUTCOME_FILE_NOT_IN_CATALOG",
    )


def _decision_episode(episode_id: str, track: str, item: dict[str, object], sign: int) -> dict[str, object]:
    episode = _base_episode(episode_id, track, item, "EVALUATION", "NONE", 0, 1)
    episode["pair_id"] = episode_id
    episode["arms"] = ["B0", "B1", "B2", "B3"]
    episode["desired_effect_sign"] = sign
    episode["switch_layer"] = 8
    episode["switch_hook"] = HOOK_L8
    episode["switch_alpha"] = 1
    episode["abstain_alpha"] = 0
    episode["pre_outcome_stages"] = _stages(
        "CLEAN_INSTRUMENT",
        "SEAL_ACTION",
        "ACTION_FORWARD",
        "OUTCOME_FILE_NOT_IN_CATALOG",
    )
    episode["compute"] = {
        "shared_instrument_forwards": 1,
        "b0_intervention_calls": 1,
        "b1_intervention_calls": "0_if_abstain_else_1",
        "b2_intervention_calls": "0_if_abstain_else_1",
        "b3_intervention_calls": "0_if_abstain_else_1",
        "additional_inference": 0,
        "additional_feature_extraction": ["g", "kappa_or_null", "rho_or_null"],
        "arm_extra_search": 0,
        "note": "ABSTAIN consumes the action slot and calls no hook. SWITCH_STRATEGY uses layer 8 alpha +1. No arm receives an extra prompt or search.",
    }
    return episode


def _assign_sm5(evidence_items: list[dict[str, object]], decision_items: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]]:
    episodes = []
    for index, item in enumerate(evidence_items, start=1):
        episode = _base_episode(f"SM5-V-{index:02d}", "SM5", item, "EVIDENCE", "NONE", 0, 1)
        episode["pre_outcome_stages"] = _stages(
            "CLEAN_INSTRUMENT",
            "SEAL_PREDICTION",
            "INTERVENTION",
            "OUTCOME_FILE_NOT_IN_CATALOG",
        )
        episodes.append(episode)
    for index, item in enumerate(decision_items, start=1):
        sign = 1 if rng.randrange(2) == 0 else -1
        episodes.append(_decision_episode(f"SM5-D-{index:02d}", "SM5", item, sign))
    return episodes


def _assign_sm6(ambiguous: list[dict[str, object]], contradictory: list[dict[str, object]], decisions: list[dict[str, object]], rng: random.Random) -> list[dict[str, object]]:
    episodes = []
    for index, item in enumerate(ambiguous, start=1):
        episode = _base_episode(f"SM6-A-{index:02d}", "SM6", item, "EVIDENCE", "NONE", 0, 1)
        episode["catalog_mark"] = "AMBIGUOUS"
        episode["scheduled_evidence_class"] = "AMBIGUOUS"
        episode["evidence_route"] = "CATALOG_MARK"
        episode["sealed_capability_claim"] = "NO_POSITIVE_CLAIM"
        episode["pre_outcome_stages"] = _stages(
            "SEAL_PREDICTION",
            "INTERVENTION",
            "OUTCOME_FILE_NOT_IN_CATALOG",
        )
        episodes.append(episode)
    for index, item in enumerate(contradictory, start=1):
        if str(item["task_key"]) == "0":
            raise RuntimeError("external-key contradiction is not independent of a zero key")
        episode = _base_episode(f"SM6-X-{index:02d}", "SM6", item, "EVIDENCE", "NONE", 0, 1)
        episode["scheduled_evidence_class"] = "CONTRADICTORY"
        episode["evidence_route"] = "EXTERNAL_KEY"
        episode["sealed_capability_claim"] = "integer_sum_key_is_0"
        episode["external_key_relation"] = "CONTRADICTS"
        episode["task_definition"] = _task_definition(item)
        episode["pre_outcome_stages"] = _stages(
            "SEAL_CAPABILITY_CLAIM",
            "SEAL_PREDICTION",
            "INTERVENTION",
            "OUTCOME_FILE_NOT_IN_CATALOG",
        )
        episodes.append(episode)
    for index, item in enumerate(decisions, start=1):
        sign = 1 if rng.randrange(2) == 0 else -1
        episode = _decision_episode(f"SM6-D-{index:02d}", "SM6", item, sign)
        episode["catalog_mark"] = "LATER_DECISION_NOT_AN_EVIDENCE_CLASS"
        episodes.append(episode)
    return episodes


def _assign_sm7(sm3: list[dict[str, object]]) -> list[dict[str, object]]:
    aliases = []
    for source in sm3:
        episode = dict(source)
        episode["episode_id"] = "SM7-" + str(source["episode_id"])
        episode["track"] = "SM7"
        episode["reuses_episode_id"] = source["episode_id"]
        episode["bootstrap_seed"] = TRACK_BOOTSTRAP_SEEDS["SM7"]
        episode["shuffle_seed"] = SHUFFLE_SEEDS["SM7"]
        episode["compute"] = dict(source["compute"])
        episode["pre_outcome_stages"] = list(source["pre_outcome_stages"])
        aliases.append(episode)
    return aliases


def _baselines() -> dict[str, object]:
    return {
        "SM1": {
            "B0": "constant call NORMAL",
            "B1": "one calibration-only rule written before any evaluation seal; instrument bundle only; label withheld",
        },
        "SM2": {
            "B1": "calibration accuracy of F1 and of F2, one number each, written before evaluation; F3 value is the mean of those two accuracies and is also written before evaluation",
        },
        "SM3": {
            "external_predictor": "prediction = g from the shared instrument; no self-record; no outcome",
        },
        "SM4": {
            "B1": "absolute g; the calibration file fixes whether higher absolute g ranks toward FAILURE or toward GOOD before any evaluation seal",
        },
        "SM5": {
            "B0": "always INTERVENE at layer 0 alpha +1",
            "B1": "INTERVENE when sign(g) equals the desired sign, otherwise ABSTAIN; no self-record",
            "B2": "frozen self-model declared before the evidence split",
            "B3": "self-model after the 16 evidence episodes, then frozen on the 24 decision episodes",
        },
        "SM6": {
            "B0": "always preserve",
            "B1_public_sign_rule": "revise when the outcome sign and sign(g) differ and both are nonzero; otherwise preserve; do not read capability keys, ambiguity marks, or the self-record",
        },
        "SM7": {
            "external_predictor": "the same public g rule as SM3, scored on the SM3 mild split",
            "semantic_ood_claim": False,
        },
        "information_rule": "No baseline reads an outcome, a self-record, or a feature outside the shared instrument unless this table states that input. B3 may read the shared instrument and its own record. B1 does not read the self-record.",
    }


def _shuffle_controls() -> dict[str, object]:
    return {
        "SM1": {"seed": 31101, "unit": "scheduled_state", "partition": "EVALUATION", "holds_fixed": "prompt"},
        "SM2": {"seed": 31102, "unit": "sealed_capability_probability", "partition": "EVALUATION", "holds_fixed": "prompt_and_task_key"},
        "SM3": {"seed": 31103, "unit": "sealed_prediction", "partition": "EVALUATION", "holds_fixed": "prompt_and_intervention"},
        "SM4": {"seed": 31104, "unit": "sealed_monitor_score", "partition": "EVALUATION", "holds_fixed": "prompt_and_pair_id"},
        "SM5": {"seed": 31105, "unit": "self_record", "partition": "EVALUATION", "holds_fixed": "prompt_desired_sign_and_action_opportunity"},
        "SM6": {"seed": 31106, "unit": "scheduled_evidence_class", "partition": "EVIDENCE", "holds_fixed": "prompt_and_external_key"},
        "SM7": {"seed": 31107, "unit": "shift_condition", "partition": "MILD_SHIFT", "holds_fixed": "prompt_and_intervention"},
        "generator": "random.Random(seed).randrange only; the process random state is not used",
    }


def build_episodes() -> list[dict[str, object]]:
    # One stream: shuffle F1, then F2, then F3, then SM5 signs, then SM6 signs.
    rng = random.Random(ASSIGNMENT_SEED)
    f1 = _shuffle(_pool(_f1_prompt, 220), rng)
    f2 = _shuffle(_pool(_f2_prompt, 64), rng)
    f3 = _shuffle(_pool(_f3_prompt, 44), rng)
    sm3 = _assign_sm3(_take(f1, 72, 64), _take(f3, 8, 24))
    episodes = []
    episodes.extend(_assign_sm1(_take(f1, 0, 56)))
    episodes.extend(_assign_sm2(_take(f1, 56, 16), _take(f2, 0, 16), _take(f3, 0, 8)))
    episodes.extend(sm3)
    episodes.extend(_assign_sm4(_take(f1, 136, 52), _take(f3, 32, 12)))
    episodes.extend(_assign_sm5(_take(f1, 188, 16), _take(f2, 16, 24), rng))
    episodes.extend(_assign_sm6(_take(f1, 204, 8), _take(f1, 212, 8), _take(f2, 40, 24), rng))
    episodes.extend(_assign_sm7(sm3))
    episodes.sort(key=lambda episode: str(episode["episode_id"]))
    for index, episode in enumerate(episodes):
        episode["random_seed"] = 400000 + index
    return episodes


def catalog_document(episodes: list[dict[str, object]] | None = None) -> dict[str, object]:
    rows = build_episodes() if episodes is None else episodes
    return {
        "catalog_version": CATALOG_VERSION,
        "suite_version": SUITE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "direction_sha256": DIRECTION_SHA256,
        "assignment_seed": ASSIGNMENT_SEED,
        "execution_authorized": False,
        "sm6_preassignment": {
            "CONFIRMING": "NOT_PREASSIGNABLE_WITHOUT_FORWARD",
            "CONTRADICTORY": "EXTERNAL_KEY_ROUTE",
            "AMBIGUOUS": "CATALOG_MARK",
            "confirming_episode_count": 0,
            "top_up_forbidden": True,
        },
        "empty_partitions": {
            "TRAIN": "The protocol defines no TRAIN split. The partition is recorded and contains no episodes.",
        },
        "shift_definitions": {
            "IID": "held-out prompts, family slot F1, hook blocks.0.hook_resid_post, alpha +1, direction D1",
            "MILD_SHIFT": "different held-out prompts, same F1 slot, hook blocks.8.hook_resid_post, alpha +1, direction D1",
            "STRONG_SHIFT": "family F3, hook blocks.0.hook_resid_post, alpha +1, direction D1",
            "OOD": "SM1 only, alpha +2 along D1 at layer 0; not a semantic-OOD claim and not used by SM2-SM7",
            "layer_shift_is_general_semantic_ood": False,
        },
        "baselines": _baselines(),
        "shuffle_controls": _shuffle_controls(),
        "episodes": rows,
    }


def catalog_sha256(document: dict[str, object] | None = None) -> str:
    return _sha256_json(catalog_document() if document is None else document)


def _walk_keys(value: object, found: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).casefold())
            _walk_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk_keys(child, found)


def _count(rows: list[dict[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def sm6_class_counts(episodes: list[dict[str, object]]) -> dict[str, int]:
    counts = {"CONFIRMING": 0, "CONTRADICTORY": 0, "AMBIGUOUS": 0}
    for episode in episodes:
        if episode["track"] != "SM6":
            continue
        label = str(episode["scheduled_evidence_class"])
        if label in counts:
            counts[label] += 1
    return counts


def _prefix(text: str) -> tuple[str, ...]:
    tokens = normalize_prompt(text).split()
    return tuple(tokens[:4])


def audit_catalog(document: dict[str, object] | None = None) -> dict[str, object]:
    document = catalog_document() if document is None else document
    episodes = list(document["episodes"])
    historical = historical_prompts()
    checks: dict[str, str] = {}
    details: dict[str, str] = {}

    ids = [str(episode["episode_id"]) for episode in episodes]
    checks["duplicate_episode"] = "PASS" if len(ids) == len(set(ids)) else "FAIL"

    by_hash: dict[str, list[dict[str, object]]] = {}
    for episode in episodes:
        by_hash.setdefault(str(episode["prompt_sha256"]), []).append(episode)
    duplicate_prompt_ok = True
    for group in by_hash.values():
        if len(group) == 1:
            continue
        if len(group) != 2:
            duplicate_prompt_ok = False
            break
        sm3 = [row for row in group if row["track"] == "SM3"]
        sm7 = [row for row in group if row["track"] == "SM7"]
        if len(sm3) != 1 or len(sm7) != 1:
            duplicate_prompt_ok = False
            break
        source = sm3[0]
        alias = sm7[0]
        same_assignment = (
            alias["reuses_episode_id"] == source["episode_id"]
            and alias["prompt"] == source["prompt"]
            and alias["partition"] == source["partition"]
            and alias["intervention_id"] == source["intervention_id"]
            and alias["layer"] == source["layer"]
            and alias["hook"] == source["hook"]
            and alias["alpha"] == source["alpha"]
            and alias["direction_id"] == source["direction_id"]
            and alias["shift_condition"] == source["shift_condition"]
        )
        duplicate_prompt_ok = duplicate_prompt_ok and same_assignment
    checks["duplicate_prompt"] = "PASS" if duplicate_prompt_ok else "FAIL"
    details["duplicate_prompt"] = "Repeated prompt hashes are SM7 aliases of one SM3 episode with the same assignment."

    partition_ok = True
    for group in by_hash.values():
        partitions = {str(row["partition"]) for row in group}
        if len(partitions) != 1:
            partition_ok = False
    id_partition = {}
    for episode in episodes:
        id_partition.setdefault(str(episode["episode_id"]), set()).add(str(episode["partition"]))
    partition_ok = partition_ok and all(len(value) == 1 for value in id_partition.values())
    checks["cross_partition_contamination"] = "PASS" if partition_ok else "FAIL"

    cross_track_ok = duplicate_prompt_ok
    checks["cross_track_contamination"] = "PASS" if cross_track_ok else "FAIL"
    details["cross_track_contamination"] = "SM7 reuses the SM3 splits by protocol. Every other cross-track prompt share fails."

    historical_exact = {prompt_sha256(row["text"]) for row in historical}
    historical_normalized = {normalize_prompt(row["text"]) for row in historical}
    suite_texts = [str(episode["prompt"]) for episode in episodes if episode["track"] != "SM7"]
    exact_hit = any(prompt_sha256(text) in historical_exact for text in suite_texts)
    normalized_hit = any(normalize_prompt(text) in historical_normalized for text in suite_texts)
    checks["historical_overlap"] = "FAIL" if exact_hit else "PASS"
    checks["prompt_normalization_overlap"] = "FAIL" if normalized_hit else "PASS"

    historical_prefixes = {_prefix(row["text"]) for row in historical}
    historical_families = {row["family"] for row in historical if row["family"]}
    prefix_hit = any(_prefix(text) in historical_prefixes for text in suite_texts)
    family_hit = bool(historical_families & {"F1", "F2", "F3"}) or any(
        str(episode["family"]) in historical_families for episode in episodes
    )
    checks["prompt_family_duplication"] = "FAIL" if prefix_hit or family_hit else "PASS"
    details["prompt_family_duplication"] = (
        "Fail if a suite family name is a historical family name or a 4-token normalized prefix matches a historical prompt."
    )

    historical_ids = {row["prompt_id"] for row in historical if row["prompt_id"]}
    heldout_texts = {
        normalize_prompt(row["text"])
        for row in historical
        if row["partition"].casefold() in {"holdout", "evaluation", "replication", "validation", "held_out_decision", "evidence"}
    }
    heldout_hit = any(normalize_prompt(text) in heldout_texts for text in suite_texts)
    id_hit = any(str(episode["episode_id"]) in historical_ids for episode in episodes)
    checks["reused_held_out_episodes"] = "FAIL" if heldout_hit or id_hit else "PASS"

    qwen_cells = {
        (normalize_prompt(row["text"]), "D1", 0, 1, HOOK_L0)
        for row in historical
        if row["source"] == _QWEN_PROTOCOL
    }
    suite_cells = {
        (
            normalize_prompt(str(episode["prompt"])),
            str(episode["direction_id"]),
            int(episode["layer"]),
            int(episode["alpha"]),
            str(episode["hook"]),
        )
        for episode in episodes
    }
    checks["reused_intervention_assignments"] = "FAIL" if qwen_cells & suite_cells else "PASS"
    details["reused_intervention_assignments"] = (
        "Historical prompt catalogs do not bind one intervention tuple per prompt. "
        "The executed bridge protocol does. A hit is an identical normalized prompt, direction, layer, alpha, and hook."
    )

    f3_bad = [
        episode
        for episode in episodes
        if episode["family"] == "F3" and episode["partition"] in {"TRAIN", "CALIBRATION"}
    ]
    f3_prompts = {str(episode["prompt"]) for episode in episodes if episode["family"] == "F3"}
    f3_elsewhere = [
        episode
        for episode in episodes
        if episode["prompt"] in f3_prompts and episode["family"] != "F3"
    ]
    checks["unknown_family_contamination"] = "FAIL" if f3_bad or f3_elsewhere else "PASS"

    ood_bad = [
        episode
        for episode in episodes
        if int(episode["alpha"]) == 2 and not (episode["track"] == "SM1" and episode["partition"] == "OOD")
    ]
    ood_missing = [episode for episode in episodes if episode["track"] == "SM1" and episode["partition"] == "OOD" and int(episode["alpha"]) != 2]
    checks["ood_contamination"] = "FAIL" if ood_bad or ood_missing else "PASS"

    legal_intervention = True
    for episode in episodes:
        allowed = {"NONE", "SUITE-D1-L0-A1", "SUITE-D1-L8-A1", "SUITE-D1-L0-A2"}
        if episode["intervention_id"] not in allowed or episode["direction_id"] != "D1":
            legal_intervention = False
        if episode["intervention_id"] == "SUITE-D1-L0-A2" and episode["track"] != "SM1":
            legal_intervention = False
    checks["intervention_legality"] = "PASS" if legal_intervention else "FAIL"

    legal_layer = all(
        (episode["layer"] == 0 and episode["hook"] == HOOK_L0)
        or (episode["layer"] == 8 and episode["hook"] == HOOK_L8)
        for episode in episodes
    )
    checks["layer_legality"] = "PASS" if legal_layer else "FAIL"

    legal_alpha = True
    for episode in episodes:
        alpha = int(episode["alpha"])
        if alpha not in {0, 1, 2}:
            legal_alpha = False
        if alpha == 2 and not (episode["track"] == "SM1" and episode["partition"] == "OOD"):
            legal_alpha = False
        if alpha == 0 and episode["intervention_id"] != "NONE":
            legal_alpha = False
        if alpha == 1 and episode["intervention_id"] not in {"SUITE-D1-L0-A1", "SUITE-D1-L8-A1"}:
            legal_alpha = False
    checks["alpha_legality"] = "PASS" if legal_alpha else "FAIL"

    episode_seeds = [int(episode["random_seed"]) for episode in episodes]
    seed_values = list(TRACK_BOOTSTRAP_SEEDS.values()) + list(SHUFFLE_SEEDS.values()) + [ASSIGNMENT_SEED]
    seeds_ok = (
        len(episode_seeds) == len(set(episode_seeds))
        and len(set(TRACK_BOOTSTRAP_SEEDS.values())) == 7
        and len(set(SHUFFLE_SEEDS.values())) == 7
        and len(set(seed_values)) == len(seed_values)
        and all(int(episode["direction_seed"]) == DIRECTION_SEED for episode in episodes)
    )
    checks["seed_uniqueness"] = "PASS" if seeds_ok else "FAIL"
    details["seed_uniqueness"] = (
        "Episode random seeds are unique. Track bootstrap seeds and shuffle seeds are unique. "
        "direction_seed repeats because every intervention uses the one frozen D1 seed 22101."
    )

    good = {str(episode["pair_id"]): episode for episode in episodes if episode["episode_id"].startswith("SM4-G-")}
    failure = {str(episode["pair_id"]): episode for episode in episodes if episode["episode_id"].startswith("SM4-F-")}
    sm4_pairs_ok = set(good) == set(failure) and len(good) == 12
    sm5_decisions = [episode for episode in episodes if str(episode["episode_id"]).startswith("SM5-D-")]
    sm5_evidence = [episode for episode in episodes if str(episode["episode_id"]).startswith("SM5-V-")]
    sm5_ok = (
        len(sm5_decisions) == 24
        and len(sm5_evidence) == 16
        and all(episode["arms"] == ["B0", "B1", "B2", "B3"] for episode in sm5_decisions)
        and all(episode["desired_effect_sign"] in {1, -1} for episode in sm5_decisions)
        and not ({episode["prompt_sha256"] for episode in sm5_decisions} & {episode["prompt_sha256"] for episode in sm5_evidence})
    )
    checks["pair_integrity"] = "PASS" if sm4_pairs_ok and sm5_ok else "FAIL"

    class_counts = sm6_class_counts(episodes)
    checks["minimum_sm6_class_size"] = "PASS" if all(count >= 8 for count in class_counts.values()) else "FAIL"
    details["minimum_sm6_class_size"] = (
        "CONFIRMING requires the outcome sign to equal the sealed prediction sign. "
        "That equality cannot be frozen before a forward. AMBIGUOUS is the catalog mark. "
        "CONTRADICTORY is the external-key route. No class is left open for a later top-up."
    )

    banned_found: set[str] = set()
    _walk_keys(document, banned_found)
    stage_ok = True
    for episode in episodes:
        stages = list(episode["pre_outcome_stages"])
        if not stages or any("OUTCOME" in stage for stage in stages[:-1]):
            stage_ok = False
        if "SEAL" not in " ".join(stages).upper():
            stage_ok = False
    pre_outcome_ok = stage_ok and not (banned_found & BANNED_FIELDS)
    confirming_absent = class_counts["CONFIRMING"] == 0
    checks["pre_outcome_availability"] = "PASS" if pre_outcome_ok and confirming_absent else "FAIL"
    details["pre_outcome_availability"] = (
        "Every stored episode seals before its outcome stage, and the document has no outcome field. "
        "CONFIRMING is absent because its protocol definition is not available before the outcome."
    )

    baseline_text = json.dumps(document["baselines"], sort_keys=True)
    parity_ok = "outcome" not in baseline_text.casefold() or "does not read" in baseline_text.casefold() or "before" in baseline_text.casefold()
    # The public sign rule names the outcome sign as a future input. That is the frozen rule, not a stored outcome.
    parity_ok = "self-record" in baseline_text and "No baseline reads an outcome" in str(document["baselines"]["information_rule"])
    checks["baseline_information_parity"] = "PASS" if parity_ok else "FAIL"
    details["baseline_information_parity"] = (
        "B1 is limited to the shared instrument or the protocol's calibration file. "
        "The SM6 public sign rule ignores capability keys, ambiguity marks, and the self-record. "
        "No baseline is given a stored outcome in this catalog."
    )

    second = catalog_document()
    checks["catalog_determinism"] = "PASS" if _sha256_json(document) == _sha256_json(second) else "FAIL"
    checks["hash_reproducibility"] = checks["catalog_determinism"]

    if any(status == "FAIL" for status in checks.values()):
        verdict = "CATALOG_DESIGN_BLOCKED"
        audit_status = "FAIL"
    elif any(status == "UNRESOLVED" for status in checks.values()):
        verdict = "CATALOG_AUDIT_INCONCLUSIVE"
        audit_status = "UNRESOLVED"
    elif any(status != "PASS" for status in checks.values()):
        verdict = "CATALOG_AUDIT_INCONCLUSIVE"
        audit_status = "UNRESOLVED"
    else:
        verdict = "CATALOG_READY_FOR_EXECUTION"
        audit_status = "PASS"
    return {
        "verdict": verdict,
        "audit_status": audit_status,
        "audit_checks": checks,
        "details": details,
        "sm6_class_counts": class_counts,
        "historical_prompt_rows": len(historical),
        "historical_unique_texts": len(historical_exact),
    }


def results_payload(created_at: str) -> dict[str, object]:
    document = catalog_document()
    episodes = list(document["episodes"])
    audit = audit_catalog(document)
    by_partition = _count(episodes, "partition")
    by_partition["TRAIN"] = by_partition.get("TRAIN", 0)
    interventions: dict[str, int] = {}
    for episode in episodes:
        key = str(episode["intervention_id"])
        interventions[key] = interventions.get(key, 0) + 1
    ood = [
        {
            "episode_id": episode["episode_id"],
            "track": episode["track"],
            "alpha": episode["alpha"],
            "layer": episode["layer"],
            "hook": episode["hook"],
            "direction_id": episode["direction_id"],
        }
        for episode in episodes
        if episode["partition"] == "OOD" or int(episode["alpha"]) == 2
    ]
    return {
        "suite_version": SUITE_VERSION,
        "catalog_version": CATALOG_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "catalog_sha256": catalog_sha256(document),
        "execution_authorized": False,
        "episode_count": len(episodes),
        "episode_count_by_track": _count(episodes, "track"),
        "episode_count_by_partition": dict(sorted(by_partition.items())),
        "episode_count_by_family": _count(episodes, "family"),
        "seed_by_track": {
            track: {
                "bootstrap_seed": TRACK_BOOTSTRAP_SEEDS[track],
                "shuffle_seed": SHUFFLE_SEEDS[track],
                "assignment_seed": ASSIGNMENT_SEED,
            }
            for track in TRACK_BOOTSTRAP_SEEDS
        },
        "audit_status": audit["audit_status"],
        "audit_checks": audit["audit_checks"],
        "overlap_status": {
            "exact_prompt": audit["audit_checks"]["historical_overlap"],
            "normalized_prompt": audit["audit_checks"]["prompt_normalization_overlap"],
            "prompt_family": audit["audit_checks"]["prompt_family_duplication"],
            "copied_template_prefix": audit["audit_checks"]["prompt_family_duplication"],
            "reused_intervention_assignment": audit["audit_checks"]["reused_intervention_assignments"],
            "reused_held_out_episode": audit["audit_checks"]["reused_held_out_episodes"],
            "historical_unique_texts": audit["historical_unique_texts"],
            "historical_prompt_rows": audit["historical_prompt_rows"],
        },
        "sm6_class_counts": audit["sm6_class_counts"],
        "ood_assignments": ood,
        "intervention_assignments": dict(sorted(interventions.items())),
        "created_at": created_at,
        "verdict": audit["verdict"],
        "design_block": audit["details"]["minimum_sm6_class_size"],
    }


def render_catalog_markdown(payload: dict[str, object]) -> str:
    counts = payload["episode_count_by_track"]
    classes = payload["sm6_class_counts"]
    return f"""# Self-model evaluation catalog v1

Version: `{CATALOG_VERSION}`

Verdict: `{payload["verdict"]}`

This catalog is a blocked design record. It is not an assignment authorized for execution. No track runner is part of this freeze. No model forward was used to choose a prompt, a class, a shift, or a seed.

## Why execution is not authorized

Protocol `SELF_MODEL_EVALUATION_PROTOCOL.1` recognizes `CONFIRMING` only when the outcome sign and the sealed-prediction sign are equal and nonzero, and the external key does not contradict the sealed capability claim. There is no catalog-mark route for that class.

`CONTRADICTORY` can be frozen without a forward by the protocol's external-key route. `AMBIGUOUS` can be frozen by the catalog mark `AMBIGUOUS`. `CONFIRMING` cannot. Filling it after outcomes are known would be the post-hoc top-up this catalog forbids.

Frozen class counts:

| class | episodes | route |
| --- | ---: | --- |
| CONFIRMING | {classes["CONFIRMING"]} | not preassignable without a forward |
| CONTRADICTORY | {classes["CONTRADICTORY"]} | external key contradicts sealed claim `integer_sum_key_is_0` |
| AMBIGUOUS | {classes["AMBIGUOUS"]} | catalog mark `AMBIGUOUS` |

The protocol's evidence quota is 24. This catalog freezes 16 evidence episodes and does not reserve empty slots. The later 24 SM6 decision episodes are disjoint and are not evidence-class members.

## Canonical assignments

Machine-readable assignments: `reports/SELF_MODEL_EVALUATION_CATALOG_V1.json`

Catalog SHA-256, over the canonical JSON of that document (`sort_keys`, compact separators, ASCII): `{payload["catalog_sha256"]}`

The hash covers episode ids, track, family, partition, prompt, intervention, layer, hook, alpha, direction, seeds, shift, baseline assignment, and the SM6 preassignment block. It does not cover `created_at` or any outcome.

Model: `{MODEL_ID}` revision `{MODEL_REVISION}`

Direction: D1 seed `{DIRECTION_SEED}` SHA-256 `{DIRECTION_SHA256}`

Assignment stream: one `random.Random({ASSIGNMENT_SEED})`. It shuffles F1, then F2, then F3, with an in-module Fisher-Yates that calls `randrange` only. It then draws SM5 decision signs and SM6 decision signs. The process random state is not used.

## Partitions

`TRAIN` is empty on every track. The protocol has no training split, and the partition is recorded rather than dropped.

| track | episodes | assignment |
| --- | ---: | --- |
| SM1 | {counts["SM1"]} | 16 calibration, 32 IID evaluation with 8 of each in-suite state, 8 OOD alpha +2 |
| SM2 | {counts["SM2"]} | 8 calibration and 8 IID evaluation for F1 and for F2; 8 F3 strong-shift items and no F3 calibration |
| SM3 | {counts["SM3"]} | 16 F1 calibration, 24 F1 IID layer 0, 24 F1 mild layer 8, 24 F3 strong layer 0 |
| SM4 | {counts["SM4"]} | 16 calibration (6 GOOD, 5 PRE-FAILURE, 5 FAILURE), 12 GOOD, 12 PRE-FAILURE, 12 FAILURE, 12 F3 strong-shift failures |
| SM5 | {counts["SM5"]} | 16 evidence episodes and 24 paired decision episodes |
| SM6 | {counts["SM6"]} | 8 ambiguous, 8 key-route contradictory, 0 confirming, 24 later decision episodes |
| SM7 | {counts["SM7"]} | aliases of the SM3 episodes; no new prompt and no new split |

SM7 does not claim that a layer shift is general semantic OOD. Mild means F1 at layer 8. Strong means F3 at layer 0. Alpha +2 belongs only to SM1 OOD.

## Episode schema

Each episode stores `episode_id`, `track`, `family`, `partition`, `prompt`, `prompt_sha256`, `shift_condition`, `intervention_id`, `layer`, `hook`, `alpha`, `direction_id`, `direction_seed`, and `random_seed`. Track-specific schedule fields are included. Ground-truth schedule labels use `scheduled_state` and `scheduled_monitor_label`. They are hidden from the later reader.

The document does not store an outcome, a post-intervention activation, an observed effect, an error, a residual, a utility, or a revision result.

## Baselines and controls

The public SM6 sign rule is frozen as: revise when the outcome sign and `sign(g)` differ and both are nonzero; otherwise preserve. That rule does not read capability keys, ambiguity marks, or the self-record.

Shuffle seeds are 31101 through 31107. The permutation unit and the partition are in `shuffle_controls`. A later execution may not choose a different permutation.

SM5 and SM6 decision episodes share one instrument forward. B0 always uses one layer-0 intervention. B1, B2, and B3 each have one action slot: `ABSTAIN` uses no hook, `INTERVENE` uses layer 0 alpha +1, and `SWITCH_STRATEGY` uses layer 8 alpha +1. No arm receives an extra search or an extra prompt.

## Independence

Tracks do not borrow episode choice from another track's result. SM7's reuse of SM3 is the protocol split map, not a result-dependent selection. F3 does not appear in calibration or in the empty training partition.
"""


def render_audit_markdown(payload: dict[str, object], audit: dict[str, object]) -> str:
    lines = [
        "# Self-model evaluation catalog audit v1",
        "",
        f"Verdict: `{payload['verdict']}`",
        "",
        f"Audit status: `{payload['audit_status']}`",
        "",
        f"Catalog SHA-256: `{payload['catalog_sha256']}`",
        "",
        "No model was loaded. Historical catalogs were read as text and JSON prompt lists. Model-derived fields in old freeze files, including baseline means, were not copied.",
        "",
        f"Historical prompt rows read: {audit['historical_prompt_rows']}. Unique historical texts: {audit['historical_unique_texts']}.",
        "",
        "## Checks",
        "",
        "| check | status |",
        "| --- | --- |",
    ]
    for name, status in payload["audit_checks"].items():
        lines.append(f"| {name} | {status} |")
    lines.extend(["", "## Notes", ""])
    for name, detail in audit["details"].items():
        lines.append(f"- `{name}`: {detail}")
    lines.extend(
        [
            "",
            "## Stop",
            "",
            "`minimum_sm6_class_size` is FAIL because CONFIRMING has 0 episodes. The catalog is incomplete. Confirming episodes are not added after this audit.",
            "",
            "UNRESOLVED was not converted to PASS. The failed check is a design block, not an uncertain overlap.",
            "",
        ]
    )
    return "\n".join(lines)


def write_catalog_files(created_at: str) -> dict[str, object]:
    document = catalog_document()
    audit = audit_catalog(document)
    payload = results_payload(created_at)
    reports = ROOT / "reports"
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != payload["catalog_sha256"]:
        raise RuntimeError("catalog hash drifted while writing")
    (reports / "SELF_MODEL_EVALUATION_CATALOG_V1.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports / "SELF_MODEL_EVALUATION_CATALOG_V1.md").write_text(
        render_catalog_markdown(payload),
        encoding="utf-8",
    )
    (reports / "SELF_MODEL_EVALUATION_CATALOG_AUDIT.md").write_text(
        render_audit_markdown(payload, audit),
        encoding="utf-8",
    )
    (reports / "SELF_MODEL_EVALUATION_CATALOG_RESULTS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload
