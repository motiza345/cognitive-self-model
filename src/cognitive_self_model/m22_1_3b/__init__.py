"""M22.1.3b first-order structure check (no model forward)."""

from .compute import (
    degenerate,
    leading_energy,
    leading_singular_vectors,
    least_squares_scale,
    run_audit,
)

__all__ = [
    "degenerate",
    "leading_energy",
    "leading_singular_vectors",
    "least_squares_scale",
    "run_audit",
]
