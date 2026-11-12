"""Point-set generators used in the experiments and figures."""

from __future__ import annotations

import math

import numpy as np


def random_uniform(n: int, seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """``n`` points uniform in the unit square [0, 1]^2 (reproducible by seed)."""
    rng = np.random.default_rng(seed) if rng is None else rng
    return rng.random((n, 2))


def equilateral_triangle(side: float = 1.0) -> np.ndarray:
    """Equilateral triangle with a horizontal base starting at the origin."""
    return np.array([[0.0, 0.0], [side, 0.0], [side / 2.0, side * math.sqrt(3) / 2.0]])


def unit_square(side: float = 1.0) -> np.ndarray:
    """Square corners, counter-clockwise from the origin."""
    return np.array([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]])


def regular_polygon(k: int, circumradius: float = 1.0, side: float | None = None) -> np.ndarray:
    """Vertices of a regular ``k``-gon centred at the origin.

    If ``side`` is given the polygon is scaled to that side length instead.
    """
    if side is not None:
        circumradius = side / (2.0 * math.sin(math.pi / k))
    t = np.pi / 2 + 2 * np.pi * np.arange(k) / k
    return circumradius * np.column_stack([np.cos(t), np.sin(t)])


def triangular_lattice_patch(rows: int, spacing: float = 1.0) -> np.ndarray:
    """Triangle-shaped patch of the triangular lattice with ``rows`` rows.

    rows = 2 gives the equilateral triangle (3 points), rows = 3 gives 6
    points, rows = 4 gives 10, ... (triangular numbers).
    """
    pts = []
    h = spacing * math.sqrt(3) / 2.0
    for r in range(rows):
        count = rows - r
        for c in range(count):
            pts.append((c * spacing + r * spacing / 2.0, r * h))
    return np.array(pts, dtype=float)


def hexagonal_lattice_patch(radius: int = 1, spacing: float = 1.0) -> np.ndarray:
    """Hexagon-shaped patch of the triangular lattice (centre plus rings)."""
    pts = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) <= radius:
                pts.append((spacing * (q + r / 2.0), spacing * r * math.sqrt(3) / 2.0))
    return np.array(pts, dtype=float)


def square_grid(k: int, spacing: float = 1.0) -> np.ndarray:
    """``k`` x ``k`` square grid."""
    xs, ys = np.meshgrid(np.arange(k) * spacing, np.arange(k) * spacing)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
