"""Rectilinear (L1) Steiner trees: MST and exact RSMT for small n.

Hanan's theorem (1966): some rectilinear Steiner minimal tree uses only
Steiner points on the Hanan grid (intersections of the horizontal and vertical
lines through the terminals), and at most n - 2 of them. So for small n we can
brute-force every subset of at most n - 2 Hanan-grid points, take the L1 MST
of terminals + subset, and keep the shortest. That is exact but exponential,
so it is limited to n <= 6.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .mst import kruskal
from .smt_heuristic import SteinerTree


def rectilinear_mst(points) -> tuple[list[tuple[int, int]], float]:
    """L1 minimum spanning tree: ``(edges, length)``."""
    return kruskal(points, metric="l1")


def hanan_grid(points) -> np.ndarray:
    """Hanan-grid points that are not terminals."""
    P = np.asarray(points, dtype=float)
    xs = np.unique(P[:, 0])
    ys = np.unique(P[:, 1])
    term = {(float(x), float(y)) for x, y in P}
    cand = [(float(x), float(y)) for x in xs for y in ys if (float(x), float(y)) not in term]
    return np.array(cand, dtype=float).reshape(-1, 2)


def exact_rsmt(points, max_n: int = 6) -> SteinerTree:
    """Exact rectilinear Steiner minimal tree by Hanan-grid brute force (n <= max_n)."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if n > max_n:
        raise ValueError(f"exact_rsmt is brute force; n={n} > max_n={max_n}")
    edges, L_mst = rectilinear_mst(P)
    best_L, best_pts, best_edges, best_k = L_mst, P.copy(), edges, 0
    H = hanan_grid(P)
    for k in range(1, max(0, n - 2) + 1):
        for subset in combinations(range(len(H)), k):
            pts = np.vstack([P, H[list(subset)]])
            e, L = kruskal(pts, metric="l1")
            if L < best_L - 1e-12:
                best_L, best_pts, best_edges, best_k = L, pts, e, k
    # drop Steiner leaves (cannot be in a strictly-better tree, but be safe)
    return SteinerTree(best_pts, sorted(best_edges), list(range(n, n + best_k)), best_L, n,
                       method="exact-hanan", mst_length=L_mst)


def l_shape_path(p, q, horizontal_first: bool = True) -> np.ndarray:
    """Polyline for drawing an L1 edge as an axis-parallel 'L'."""
    corner = (q[0], p[1]) if horizontal_first else (p[0], q[1])
    return np.array([p, corner, q], dtype=float)
