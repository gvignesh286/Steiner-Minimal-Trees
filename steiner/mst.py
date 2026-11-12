"""Minimum spanning trees by Kruskal's algorithm with a hand-written union-find."""

from __future__ import annotations

import numpy as np


class UnionFind:
    """Disjoint-set forest with union by rank and path halving."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of ``a`` and ``b``. Returns False if already joined."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def distance_matrix(points, metric: str = "euclidean") -> np.ndarray:
    """All pairwise distances, ``metric`` is 'euclidean' (L2) or 'l1'."""
    P = np.asarray(points, dtype=float)
    diff = P[:, None, :] - P[None, :, :]
    if metric == "euclidean":
        return np.hypot(diff[..., 0], diff[..., 1])
    if metric == "l1":
        return np.abs(diff).sum(axis=-1)
    raise ValueError(f"unknown metric {metric!r}")


def kruskal(points, metric: str = "euclidean") -> tuple[list[tuple[int, int]], float]:
    """Minimum spanning tree of the complete graph on ``points``.

    Returns ``(edges, total_length)`` where ``edges`` is a list of index pairs
    ``(i, j)`` with ``i < j``. Edges are scanned in increasing length (stable
    sort, so ties are broken by input order) and kept when they join two
    different components of the union-find structure.
    """
    if len(points) < 2:
        return [], 0.0
    return kruskal_matrix(distance_matrix(points, metric))


def kruskal_matrix(D: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Kruskal on a symmetric weight matrix ``D``; returns ``(edges, total_weight)``."""
    n = len(D)
    if n < 2:
        return [], 0.0
    iu, ju = np.triu_indices(n, 1)
    w = D[iu, ju]
    order = np.argsort(w, kind="stable")
    uf = UnionFind(n)
    edges: list[tuple[int, int]] = []
    total = 0.0
    for k in order:
        i, j = int(iu[k]), int(ju[k])
        if uf.union(i, j):
            edges.append((i, j))
            total += float(w[k])
            if len(edges) == n - 1:
                break
    return edges, total


def mst_length(points, metric: str = "euclidean") -> float:
    """Total length of the minimum spanning tree."""
    return kruskal(points, metric)[1]
