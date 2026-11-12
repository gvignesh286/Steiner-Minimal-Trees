"""Exact Euclidean Steiner minimal trees for n = 3 and n = 4 terminals.

Used to validate the heuristic. Every Steiner minimal tree is a union of
full Steiner components (each a tree whose leaves are terminals and whose
internal nodes are degree-3 Steiner points). For n <= 4 the possibilities are:

* no Steiner point: the MST;
* one Steiner point joining three terminals (their Fermat point), with the
  fourth terminal attached by an edge to its nearest of those three;
* a full topology with two Steiner points: 3 ways to pair the 4 terminals,
  {a,b | c,d}, with Steiner points s1 ~ (a, b, s2) and s2 ~ (c, d, s1).

For a fixed topology the tree length is a convex function of the Steiner
point positions, so a stationary point in which no Steiner point collapses
is the global optimum for that topology. We reach it by alternating Fermat
point updates and keep it only if it is non-degenerate. The minimum over all
candidates is the exact SMT (up to floating-point tolerance).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .geometry import dist, fermat_point
from .mst import kruskal
from .smt_heuristic import SteinerTree


def _full_two_steiner(a, b, c, d, tol: float = 1e-14, max_iter: int = 100_000):
    s1 = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    s2 = ((c[0] + d[0]) / 2, (c[1] + d[1]) / 2)
    for _ in range(max_iter):
        n1 = fermat_point(a, b, s2)
        n2 = fermat_point(c, d, n1)
        move = max(dist(n1, s1), dist(n2, s2))
        s1, s2 = n1, n2
        if move < tol:
            break
    eps = 1e-7
    if min(dist(s1, a), dist(s1, b), dist(s1, s2), dist(s2, c), dist(s2, d)) < eps:
        return None  # degenerate: covered by another candidate
    L = dist(a, s1) + dist(b, s1) + dist(s1, s2) + dist(s2, c) + dist(s2, d)
    return L, s1, s2


def exact_smt(points) -> SteinerTree:
    """Exact SMT for 2 <= n <= 4 points."""
    P = np.asarray(points, dtype=float)
    n = len(P)
    if not 2 <= n <= 4:
        raise ValueError("exact_smt handles 2..4 terminals")
    mst_edges, L_mst = kruskal(P)
    best = SteinerTree(P.copy(), mst_edges, [], L_mst, n, method="exact", mst_length=L_mst)
    T = [tuple(p) for p in P]

    def consider(L, steiner_pts, edges):
        nonlocal best
        if L < best.length - 1e-13:
            pts = np.vstack([P, np.array(steiner_pts).reshape(-1, 2)])
            best = SteinerTree(pts, sorted(edges), list(range(n, n + len(steiner_pts))),
                               L, n, method="exact", mst_length=L_mst)

    # one Steiner point on a triple (+ an edge for the remaining terminal if n == 4)
    for tri in combinations(range(n), 3):
        f = fermat_point(*(T[i] for i in tri))
        if min(dist(f, T[i]) for i in tri) < 1e-9:
            continue  # Fermat point is a vertex: no Steiner point needed
        L = sum(dist(f, T[i]) for i in tri)
        edges = [(i, n) for i in tri]
        rest = [k for k in range(n) if k not in tri]
        if rest:
            r = rest[0]
            near = min(tri, key=lambda i: dist(T[r], T[i]))
            L += dist(T[r], T[near])
            edges.append((min(r, near), max(r, near)))
        consider(L, [f], edges)

    if n == 4:
        for (i, j), (k, l) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            res = _full_two_steiner(T[i], T[j], T[k], T[l])
            if res is None:
                continue
            L, s1, s2 = res
            consider(L, [s1, s2], [(i, 4), (j, 4), (4, 5), (k, 5), (l, 5)])
    return best
