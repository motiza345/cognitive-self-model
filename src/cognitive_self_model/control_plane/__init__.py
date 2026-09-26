"""Machine-checkable project control plane."""

from .validate import validate, validate_or_raise

__all__ = ["validate", "validate_or_raise"]
