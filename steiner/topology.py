"""Counting Steiner topologies."""

from __future__ import annotations


def double_factorial(k: int) -> int:
    """k!! = k (k-2) (k-4) ... ; by convention (-1)!! = 0!! = 1."""
    if k < -1:
        raise ValueError("double factorial defined for k >= -1")
    out = 1
    while k > 1:
        out *= k
        k -= 2
    return out


def num_full_topologies(n: int) -> int:
    """Number of full Steiner topologies on ``n >= 2`` labelled terminals.

    A full topology has every terminal as a leaf and exactly n - 2 Steiner
    points of degree 3. The count is (2n - 5)!! = 1, 3, 15, 105, ... for
    n = 3, 4, 5, 6 (n = 2 gives the single edge, count 1).
    """
    if n < 2:
        raise ValueError("need at least 2 terminals")
    return double_factorial(2 * n - 5)
