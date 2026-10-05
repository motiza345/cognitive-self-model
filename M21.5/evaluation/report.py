"""Claim decision and report rendering. Numbers come from sealed logs."""

from __future__ import annotations

import json

PRIMARY_AGENTS = ("B0", "B1", "B2", "B3")


def decide_claim(primary: dict, static_summary: dict, ood_summary: dict, ablations: dict, integrity: dict) -> dict:
    delta = primary["comparisons"]["B3-BestSimple"]
    b3_dsr = primary["per_agent"]["B3"]["dsr"]["dsr"]
    baseline_dsrs = [primary["per_agent"][name]["dsr"]["dsr"] for name in ("B0", "B1", "B2")]
    if b3_dsr is None or any(value is None for value in baseline_dsrs):
        dsr_improves = False
        best_baseline_dsr = None
    else:
        best_baseline_dsr = max(baseline_dsrs)
        dsr_improves = b3_dsr > best_baseline_dsr
    gains = primary["per_agent"]["B3"]["mean_detection_update_gain"]
    high_low = gains.get("HIGH->LOW")
    low_medium = gains.get("LOW->MEDIUM")
    gains_positive = (
        high_low is not None and low_medium is not None and high_low > 0.0 and low_medium > 0.0
    )
    ood_delta = ood_summary["comparisons"]["B3-BestSimple"]
    causal = ablations["B3-no-causal-structure"]["full_minus_ablation"]
    static_delta = static_summary["comparisons"]["B3-BestSimple"]
    criteria = {
        "no_leakage": bool(integrity["leakage_passed"]),
        "no_major_compute_confound": not bool(integrity["major_compute_confound"]),
        "b3_beats_best_simple": delta["mean"] > 0.0,
        "delta_ci_above_zero": delta["low"] > 0.0 and int(delta["n"]) == 10,
        "decision_separation_improves": dsr_improves,
        "advantage_after_transition": primary["post_transition_delta"] > 0.0,
        "update_gain_positive": gains_positive,
        "replicates_across_seeds": delta["low"] > 0.0 and int(delta["n"]) == 10,
        "ood_advantage": ood_delta["mean"] > 0.0 and ood_delta["low"] > 0.0 and int(ood_delta["n"]) == 10,
        "causal_ablation_reduces_advantage": causal["mean"] > 0.0
        and causal["low"] > 0.0
        and int(causal["n"]) == 10,
    }
    integrity_ok = bool(
        integrity["raw_hashes_match"]
        and integrity["source_hashes_match"]
        and integrity["fingerprints_match"]
    )
    static_blocks = static_delta["low"] > 0.0
    accepted = all(criteria.values()) and integrity_ok and not static_blocks
    if not integrity["leakage_passed"]:
        verdict = "BENCHMARK_INVALID"
    elif not integrity_ok:
        verdict = "INTEGRITY_FAILURE"
    elif accepted:
        verdict = "M21.5_V1_CLAIM_ACCEPTED"
    else:
        verdict = "M21.5_V1_CLAIM_NOT_ACCEPTED"
    interpretation = _interpret(
        primary,
        ood_summary,
        accepted=accepted,
        leakage_passed=bool(integrity["leakage_passed"]),
        static_blocks=static_blocks,
    )
    return {
        "criteria": criteria,
        "best_baseline_dsr": best_baseline_dsr,
        "b3_dsr": b3_dsr,
        "static_confound_suspected": static_blocks,
        "integrity_ok": integrity_ok,
        "accepted": accepted,
        "verdict": verdict,
        "interpretation": interpretation,
    }


def _interpret(primary: dict, ood_summary: dict, accepted: bool, leakage_passed: bool, static_blocks: bool) -> list[str]:
    if not leakage_passed:
        return ["Leakage audit failed. No scientific interpretation is offered."]
    notes = []
    briers = {
        name: primary["per_agent"][name]["prediction"]["brier"] for name in PRIMARY_AGENTS
    }
    delta = primary["comparisons"]["B3-BestSimple"]
    if briers["B3"] < min(briers["B0"], briers["B1"], briers["B2"]) and delta["mean"] <= 0.0:
        notes.append("Predictive value without demonstrated decision value.")
    b3_b0 = primary["comparisons"]["B3-B0"]
    b3_b2 = primary["comparisons"]["B3-B2"]
    if b3_b0["low"] > 0.0 and b3_b2["low"] <= 0.0 <= b3_b2["high"]:
        notes.append(
            "The observed advantage may be explained by memory/reflection rather than causal Self-Modeling."
        )
    ood_delta = ood_summary["comparisons"]["B3-BestSimple"]
    ood_b2 = ood_summary["comparisons"]["B3-B2"]
    ood_advantage = ood_delta["mean"] > 0.0 and ood_delta["low"] > 0.0
    if b3_b2["low"] > 0.0 and ood_b2["low"] <= 0.0:
        notes.append("In-distribution advantage without demonstrated generalization.")
    if static_blocks:
        notes.append(
            "The static negative control also shows an interval above zero. "
            "That pattern is a confound check, not evidence for self-model utility."
        )
    if accepted:
        notes.append(
            "On this frozen benchmark, a causal updateable model of the agent's own changing "
            "behavior produced higher decision utility than the best simple baseline under the "
            "predeclared criteria."
        )
        notes.append(
            "This does not show that a self-model is useful for every task, that it understands "
            "a neural network, that it is conscious, or that self-improvement of model weights occurred."
        )
    elif ood_advantage and b3_b2["low"] > 0.0:
        notes.append(
            "Some contrasts favor the causal model, but the full predeclared acceptance list is not met."
        )
    if not notes:
        notes.append(
            "The predeclared acceptance list is not met. Improved prediction, a win against one "
            "baseline, or a gain on one seed is not sufficient."
        )
    return notes


def render_markdown(report: dict) -> str:
    env = report["environment"]
    performance = report["performance"]
    stats = report["statistics"]
    decision = report["decision_metrics"]
    controls = report["controls"]
    integrity = report["integrity"]
    claim = report["claim"]
    lines = [
        "# M21.5 Self-Model Utility & Necessity Benchmark v1.0",
        "",
        f"Verdict: `{claim['verdict']}`",
        "",
        "The dynamic schedule is the positive control and the primary comparison.",
        "A passing verdict would not mean that a self-model is generally useful, conscious, or able to edit model weights.",
        "",
        "## Environment",
        "",
        f"- Version: `{env['benchmark_version']}`",
        f"- Implementation version: `{env['implementation_version']}`",
        f"- Environment hash: `{env['environment_hash']}`",
        f"- Config hash: `{env['config_hash']}`",
        f"- Prompt hash: `{env['prompt_hash']}`",
        f"- Model revision: `{env['model_revision']}`",
        f"- Code commit: `{env['code_commit']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in env['seed_list'])}`",
        "",
        "## Performance",
        "",
        _utility_table(performance),
        "",
        f"BestSimple utility (mean of per-seed maxima): { _fmt(performance['best_simple_mean_utility']) }",
        "",
        f"DeltaU (B3 - BestSimple): { _fmt(performance['delta_u']) }",
        "",
        "## Statistics",
        "",
        _comparison_table(stats["primary"]),
        "",
        "The primary comparison is B3 - BestSimple. The other rows are secondary and are always reported.",
        "",
        "## Decision metrics",
        "",
        _dsr_table(decision["dsr"]),
        "",
        _regret_table(decision["regret"]),
        "",
        "### Prediction metrics",
        "",
        _prediction_table(decision["prediction"]),
        "",
        "Prediction metrics are secondary. A better score here is not decision utility.",
        "",
        "### Mismatch detection",
        "",
        _mismatch_table(decision["mismatch"]),
        "",
        "### Update gain",
        "",
        _gain_table(decision["update_gain"]),
        "",
        f"Post-transition utility difference (episodes 41-100, B3 minus per-seed best simple): { _fmt(decision['post_transition_delta']) }",
        "",
        "## Controls",
        "",
        "### Dynamic positive control",
        "",
        "The primary run uses HIGH for episodes 1-40, LOW for 41-60, and MEDIUM for 61-100.",
        "",
        "### Static negative control",
        "",
        _utility_table(controls["static"]["performance"]),
        "",
        _comparison_table(controls["static"]["comparisons"]),
        "",
        f"Static confound suspected: `{controls['static']['confound_suspected']}`",
        "",
        "### OOD",
        "",
        "OOD changes only the task-regime distribution to 0.70 / 0.20 / 0.10. The state-action table is unchanged.",
        "",
        _utility_table(controls["ood"]["performance"]),
        "",
        _comparison_table(controls["ood"]["comparisons"]),
        "",
        "### Ablations",
        "",
        _ablation_table(controls["ablations"]),
        "",
        "## Integrity",
        "",
        f"- Leakage audit: `{integrity['leakage']}`",
        f"- Compute fairness: `{integrity['compute_fairness']}`",
        f"- Reproducibility: `{integrity['reproducibility']}`",
        f"- Historical immutability: `{integrity['historical_immutability']}`",
        "",
        integrity["compute_note"],
        "",
        "## Acceptance criteria",
        "",
    ]
    for name, passed in claim["criteria"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(["", "## Interpretation", ""])
    for note in claim["interpretation"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Implementation choices", "", "These choices are not part of the frozen probability table, schedule, seeds, or reward. They were fixed in the config before the primary run.", ""])
    for key, value in report["implementation_choices"].items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    lines.append("")
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def _utility_table(performance: dict) -> str:
    lines = ["| Agent | Mean utility |", "| --- | ---: |"]
    for name in list(PRIMARY_AGENTS) + [key for key in performance.get("means", {}) if key not in PRIMARY_AGENTS]:
        if name in performance["means"]:
            lines.append(f"| {name} | {_fmt(performance['means'][name])} |")
    return "\n".join(lines)


def _comparison_table(comparisons: dict) -> str:
    lines = ["| Contrast | Mean | 95% low | 95% high |", "| --- | ---: | ---: | ---: |"]
    for name in ("B3-B0", "B3-B1", "B3-B2", "B3-BestSimple"):
        row = comparisons[name]
        lines.append(f"| {name} | {_fmt(row['mean'])} | {_fmt(row['low'])} | {_fmt(row['high'])} |")
    return "\n".join(lines)


def _dsr_table(dsr: dict) -> str:
    lines = ["| Agent | Opportunities | Correct changes | DSR |", "| --- | ---: | ---: | ---: |"]
    for name in PRIMARY_AGENTS:
        row = dsr[name]
        lines.append(
            f"| {name} | {row['opportunities']} | {row['correct_changes']} | {_fmt(row['dsr'])} |"
        )
    return "\n".join(lines)


def _regret_table(regret: dict) -> str:
    lines = ["| Agent | Mean regret |", "| --- | ---: |"]
    for name in PRIMARY_AGENTS:
        lines.append(f"| {name} | {_fmt(regret[name])} |")
    return "\n".join(lines)


def _prediction_table(prediction: dict) -> str:
    lines = ["| Agent | MAE | Brier | Log loss | Calibration error |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in PRIMARY_AGENTS:
        row = prediction[name]
        lines.append(
            f"| {name} | {_fmt(row['mae'])} | {_fmt(row['brier'])} | {_fmt(row['log_loss'])} | {_fmt(row['calibration_error'])} |"
        )
    return "\n".join(lines)


def _mismatch_table(mismatch: dict) -> str:
    lines = [
        "| Agent | Precision | Recall | Mean delay | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in PRIMARY_AGENTS:
        row = mismatch[name]
        lines.append(
            f"| {name} | {_fmt(row['precision'])} | {_fmt(row['recall'])} | {_fmt(row['mean_delay'])} | "
            f"{row['true_positive']} | {row['false_positive']} | {row['false_negative']} |"
        )
    return "\n".join(lines)


def _gain_table(gains: dict) -> str:
    lines = [
        "| Agent | HIGH→LOW at transition | LOW→MEDIUM at transition | HIGH→LOW at detection | LOW→MEDIUM at detection |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in PRIMARY_AGENTS:
        row = gains[name]
        lines.append(
            f"| {name} | {_fmt(row['transition'].get('HIGH->LOW'))} | {_fmt(row['transition'].get('LOW->MEDIUM'))} | "
            f"{_fmt(row['detection'].get('HIGH->LOW'))} | {_fmt(row['detection'].get('LOW->MEDIUM'))} |"
        )
    return "\n".join(lines)


def _ablation_table(ablations: dict) -> str:
    lines = [
        "| Ablation | Mean utility | Full B3 minus ablation | 95% low | 95% high |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in ablations.items():
        contrast = row["full_minus_ablation"]
        lines.append(
            f"| {name} | {_fmt(row['mean_utility'])} | {_fmt(contrast['mean'])} | {_fmt(contrast['low'])} | {_fmt(contrast['high'])} |"
        )
    return "\n".join(lines)


def performance_block(summary: dict) -> dict:
    means = {name: summary["per_agent"][name]["mean_utility"] for name in summary["agent_names"]}
    for name in PRIMARY_AGENTS:
        if name in summary["per_agent"]:
            means[name] = summary["per_agent"][name]["mean_utility"]
    block = {"means": means}
    if "best_simple_mean_utility" in summary:
        block["best_simple_mean_utility"] = summary["best_simple_mean_utility"]
        block["delta_u"] = summary["comparisons"]["B3-BestSimple"]["mean"]
    return block


def build_report_document(config: dict, manifest: dict, primary, static_summary, ood_summary, ablations, integrity, claim) -> dict:
    return {
        "environment": {
            "benchmark_version": manifest["benchmark_version"],
            "implementation_version": manifest["implementation_version"],
            "environment_hash": manifest["environment_hash"],
            "config_hash": manifest["config_hash"],
            "prompt_hash": manifest["prompt_hash"],
            "model_revision": manifest["model_revision"],
            "code_commit": manifest["code_commit"],
            "seed_list": manifest["seed_list"],
        },
        "performance": performance_block(primary),
        "statistics": {"primary": primary["comparisons"]},
        "decision_metrics": {
            "dsr": {name: primary["per_agent"][name]["dsr"] for name in PRIMARY_AGENTS},
            "regret": {name: primary["per_agent"][name]["mean_regret"] for name in PRIMARY_AGENTS},
            "prediction": {name: primary["per_agent"][name]["prediction"] for name in PRIMARY_AGENTS},
            "mismatch": {name: primary["per_agent"][name]["mismatch"] for name in PRIMARY_AGENTS},
            "update_gain": {
                name: {
                    "transition": primary["per_agent"][name]["mean_transition_update_gain"],
                    "detection": primary["per_agent"][name]["mean_detection_update_gain"],
                }
                for name in PRIMARY_AGENTS
            },
            "post_transition_delta": primary["post_transition_delta"],
        },
        "controls": {
            "static": {
                "performance": performance_block(static_summary),
                "comparisons": static_summary["comparisons"],
                "confound_suspected": claim["static_confound_suspected"],
            },
            "ood": {
                "performance": performance_block(ood_summary),
                "comparisons": ood_summary["comparisons"],
            },
            "ablations": ablations,
        },
        "integrity": integrity["public"],
        "claim": claim,
        "implementation_choices": config["implementation_choices"],
        "primary_details": _compact_details(primary),
        "static_details": _compact_details(static_summary),
        "ood_details": _compact_details(ood_summary),
    }


def _compact_details(summary: dict) -> list[dict]:
    rows = []
    for row in summary["per_seed"]:
        agents = {}
        for name, record in row["agents"].items():
            agents[name] = {
                "utility": record["utility"],
                "post_utility": record["post_utility"],
                "regret": record["regret"],
                "dsr": record["decision_separation"]["dsr"],
                "transition_update_gain": record["transition_update_gain"],
                "detection_update_gain": record["detection_update_gain"],
                "mismatch": {
                    "precision": record["mismatch"]["precision"],
                    "recall": record["mismatch"]["recall"],
                    "delays": record["mismatch"]["delays"],
                },
            }
        rows.append({"seed": row["seed"], "agents": agents})
    return rows
