"""Black-box confidence sensors. They turn what any model API returns into a
probability in [0, 1] that the agent's own answer is correct.

Origins: M34a v2 `answer_logprob` = sum of generated-token log probabilities,
end-of-sequence excluded; within-level AUROC 0.934. The sensors here are
deliberately not privileged: anyone with log-probability access can compute
them (see TECHNICAL_REPORT v1.1, section 1, object 3).
"""

from __future__ import annotations

import math
import re
from typing import Sequence


def from_token_logprobs(token_logprobs: Sequence[float], mode: str = "sum") -> float:
    """sum: probability of the whole answer string (M34a definition).
    mean: geometric-mean token probability (less length-biased)."""
    lps = [float(x) for x in token_logprobs]
    if not lps or any(not math.isfinite(x) or x > 1e-9 for x in lps):
        raise ValueError("token log probabilities must be finite and <= 0")
    total = sum(lps) if mode == "sum" else sum(lps) / len(lps)
    if mode not in {"sum", "mean"}:
        raise ValueError("mode is 'sum' or 'mean'")
    return math.exp(total)


_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%?")


def from_verbal(text: str) -> float | None:
    """Parse '83', '83%', '0.83' from a verbalized confidence. Returns None if absent.
    Verbal confidence is weaker than log probabilities in the literature; keep it
    in a separate scope or a separate arm, not mixed with logprob confidence."""
    m = _PCT.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    if "%" in text or v > 1.0:
        v /= 100.0
    return min(1.0, max(0.0, v))
