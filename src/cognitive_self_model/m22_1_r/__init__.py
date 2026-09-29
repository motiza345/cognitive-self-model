"""M22.1-R replay harness.

Engineering replay of the recorded M22.1 preflight. This package does not
implement MRSM, does not change tolerance values, and does not assign a
scientific claim.
"""

__all__ = ["run_harness"]


def __getattr__(name: str):
    if name == "run_harness":
        from .harness import run_harness

        return run_harness
    raise AttributeError(name)
