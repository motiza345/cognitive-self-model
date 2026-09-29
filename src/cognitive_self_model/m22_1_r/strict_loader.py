"""Fail-closed loader used only by the M22.1-R replay path.

This module does not query the Hub for a revision and does not retry
``from_pretrained`` without a revision. The historical loader fallback in
``m22_1.loader`` is left unchanged for other callers.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import REQUIRED_REVISION


class ReplayBlockedError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def load_pinned_revision(
    model_id: str,
    revision: str,
    *,
    from_pretrained: Callable[..., Any],
    recorded_revision: str = REQUIRED_REVISION,
    dtype: Any = None,
) -> Any:
    """Load ``model_id`` at ``revision`` or raise ``ReplayBlockedError``.

    A different revision is rejected before ``from_pretrained`` is called.
    A failure of the pinned call is not followed by a second, unpinned call.
    """
    if revision != recorded_revision:
        raise ReplayBlockedError(
            "requested revision does not match the recorded pin "
            f"{recorded_revision}"
        )
    if dtype is None:
        import torch

        dtype = torch.float32
    try:
        return from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device="cpu",
            dtype=dtype,
        )
    except ReplayBlockedError:
        raise
    except Exception as exc:
        raise ReplayBlockedError(
            "pinned revision load failed without an unpinned retry: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
