"""Schoolbook carry counts for add_nn and mul_n1 (descriptive only)."""

from __future__ import annotations


def n_carries_mul_n1(a: int, b: int) -> int:
    if not (0 <= int(b) <= 9):
        raise ValueError(f"b must be one digit, got {b}")
    n_carries = 0
    carry = 0
    n = abs(int(a))
    while True:
        digit = n % 10
        total = digit * int(b) + carry
        carry = total // 10
        if carry != 0:
            n_carries += 1
        n //= 10
        if n == 0:
            break
    return n_carries


def n_carries_add_nn(a: int, b: int) -> int:
    """Digit-wise schoolbook addition from LSD; count nonzero carry-outs."""
    n_carries = 0
    carry = 0
    x, y = abs(int(a)), abs(int(b))
    while x > 0 or y > 0:
        total = (x % 10) + (y % 10) + carry
        carry = total // 10
        if carry != 0:
            n_carries += 1
        x //= 10
        y //= 10
    return n_carries


def n_carries_for_row(row: dict) -> int:
    op = str(row.get("op", "x"))
    a, b = int(row["a"]), int(row["b"])
    if op == "+":
        return n_carries_add_nn(a, b)
    return n_carries_mul_n1(a, b)
