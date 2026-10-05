"""Benchmark agents."""

from agents.b0_raw import RawAgent
from agents.b1_confidence import ConfidenceAgent
from agents.b2_memory_reflection import MemoryAgent
from agents.b3_self_model import SelfModelAgent

ABLATION_NAMES = (
    "B3-no-causal-structure",
    "B3-no-update",
    "B3-no-uncertainty",
    "B3-no-intervention-prediction",
)

ABLATION_CODES = {
    "B3-no-causal-structure": "no_causal_structure",
    "B3-no-update": "no_update",
    "B3-no-uncertainty": "no_uncertainty",
    "B3-no-intervention-prediction": "no_intervention_prediction",
}


def build_agent(name: str, hparams: dict):
    if name == "B0":
        return RawAgent(hparams)
    if name == "B1":
        return ConfidenceAgent(hparams)
    if name == "B2":
        return MemoryAgent(hparams)
    if name == "B3":
        return SelfModelAgent(hparams)
    if name in ABLATION_CODES:
        return SelfModelAgent(hparams, ablation=ABLATION_CODES[name])
    raise KeyError(name)


PRIMARY_AGENTS = ("B0", "B1", "B2", "B3")
