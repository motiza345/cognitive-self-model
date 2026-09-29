"""Minimal MRSM benchmark. Not a cognitive operating system."""

__all__ = ["HEADS"]

HEADS = [f"L{layer}H{head}" for layer in range(2) for head in range(4)]
