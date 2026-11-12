"""Basic planar geometry: distances, angles and the Fermat (Torricelli) point.

The Fermat point of a triangle is the point minimising the sum of distances
to the three corners. If every angle of the triangle is below 120 degrees it is
the unique interior point where the three corners are seen at 120 degrees from
each other; otherwise it is the corner whose angle is at least 120 degrees.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

Point = Sequence[float]

_EPS = 1e-15
_TWO_THIRDS_PI = 2.0 * math.pi / 3.0


def dist(p: Point, q: Point) -> float:
    """Euclidean (L2) distance between two points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def l1_dist(p: Point, q: Point) -> float:
    """Rectilinear (L1, 'Manhattan') distance between two points."""
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def angle_at(v: Point, a: Point, b: Point) -> float:
    """Angle a-v-b at vertex ``v``, in degrees (0..180).

    Returns ``nan`` if ``a`` or ``b`` coincides with ``v`` (the angle is
    undefined), so comparisons such as ``angle < 120`` are False.
    """
    ax, ay = a[0] - v[0], a[1] - v[1]
    bx, by = b[0] - v[0], b[1] - v[1]
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na < _EPS or nb < _EPS:
        return math.nan
    c = (ax * bx + ay * by) / (na * nb)
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def _clamped_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, x)))


def fermat_point(a: Point, b: Point, c: Point) -> tuple[float, float]:
    """Closed-form Fermat/Torricelli point of triangle ``abc``.

    Uses the barycentric coordinates ``a*csc(A+60deg) : b*csc(B+60deg) :
    c*csc(C+60deg)`` (side lengths opposite each vertex) of the first isogonic
    centre. If some angle is >= 120 degrees, that vertex is returned. Coincident
    or collinear inputs are handled (the duplicated / middle point is returned).
    """
    la = dist(b, c)  # side opposite a
    lb = dist(a, c)
    lc = dist(a, b)
    if lc < _EPS:  # a == b: f(p) = 2|pa| + |pc| is minimised at a
        return (float(a[0]), float(a[1]))
    if lb < _EPS:
        return (float(a[0]), float(a[1]))
    if la < _EPS:
        return (float(b[0]), float(b[1]))
    A = _clamped_acos((lb * lb + lc * lc - la * la) / (2.0 * lb * lc))
    B = _clamped_acos((la * la + lc * lc - lb * lb) / (2.0 * la * lc))
    C = math.pi - A - B
    if A >= _TWO_THIRDS_PI:
        return (float(a[0]), float(a[1]))
    if B >= _TWO_THIRDS_PI:
        return (float(b[0]), float(b[1]))
    if C >= _TWO_THIRDS_PI:
        return (float(c[0]), float(c[1]))
    third = math.pi / 3.0
    wa = la / math.sin(A + third)
    wb = lb / math.sin(B + third)
    wc = lc / math.sin(C + third)
    s = wa + wb + wc
    return (
        (wa * a[0] + wb * b[0] + wc * c[0]) / s,
        (wa * a[1] + wb * b[1] + wc * c[1]) / s,
    )


def weiszfeld(
    points: Sequence[Point],
    x0: Point | None = None,
    tol: float = 1e-12,
    max_iter: int = 100_000,
) -> tuple[float, float]:
    """Geometric median of ``points`` (minimiser of the sum of distances).

    Classic Weiszfeld iteration with a fixed tolerance on the step size. For
    three points this converges to the same point as :func:`fermat_point`; it
    is provided as an independent check and for more than three points.
    If an iterate lands on an input point, that point is tested for optimality
    (Kuhn's condition) and returned if optimal.
    """
    P = np.asarray(points, dtype=float)
    x = P.mean(axis=0) if x0 is None else np.asarray(x0, dtype=float)
    for _ in range(max_iter):
        d = np.hypot(*(P - x).T)
        hit = d < 1e-14
        if hit.any():
            # Kuhn: an input point is optimal iff the pull of the others is <= 1
            others = ~hit
            g = ((P[others] - x) / d[others, None]).sum(axis=0)
            if np.hypot(*g) <= hit.sum():
                return (float(x[0]), float(x[1]))
            x = x + 1e-9 * g / np.hypot(*g)
            continue
        w = 1.0 / d
        x_new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.hypot(*(x_new - x)) < tol:
            return (float(x_new[0]), float(x_new[1]))
        x = x_new
    return (float(x[0]), float(x[1]))


def tree_length(points, edges, metric: str = "euclidean") -> float:
    """Total edge length of a tree given as point coordinates and index pairs."""
    f = dist if metric == "euclidean" else l1_dist
    return float(sum(f(points[i], points[j]) for i, j in edges))
