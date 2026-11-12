"""Euclidean Steiner tree HEURISTIC (MST + greedy Steiner-point insertion).

IMPORTANT: this is a heuristic, not an exact solver. The tree it returns is a
valid Steiner tree for the input terminals, so its length is an UPPER BOUND on
the length of the true Steiner minimal tree (SMT). Consequently the ratio
L(heuristic) / L(MST) it reports is an upper bound on the true Steiner ratio
of that point set. For exact SMTs install GeoSteiner (see ``geosteiner.py``).

Algorithm
---------
1. Start from the Euclidean minimum spanning tree.
2. Repeatedly look for a vertex ``v`` with two neighbours ``a`` and ``b`` such
   that the angle a-v-b is below 120 degrees. Replacing edges ``va`` and ``vb``
   with a new Steiner point ``s`` at the Fermat point of (v, a, b) and edges
   ``sv, sa, sb`` shortens the tree. Each round performs the insertion with
   the largest length gain.
3. After each insertion the Steiner points are smoothed: each one is moved to
   the Fermat point of its three current neighbours (the fixed point of the
   Weiszfeld iteration for three points, computed in closed form), sweeping
   until nothing moves by more than ``1e-11``. Only the connected cluster of
   Steiner points containing the new point can move, so only that cluster is
   swept.
4. Steiner points that end with degree 2 are removed (their two neighbours
   are joined directly, which is never longer), and Steiner points that
   collapse onto a neighbour (distance < 1e-9) are merged into it.
5. Steps 2-4 repeat until no insertion helps. The whole procedure is run
   ``restarts`` times; restart 0 is purely greedy from the true MST. Later
   restarts shuffle the terminal order, start from the MST of randomly
   inflated distances (a slightly different starting topology; plain MST
   greedy cannot reach every optimal topology) and pick randomly among the
   three best insertions each round. The shortest tree is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import angle_at, dist, fermat_point
from .mst import distance_matrix, kruskal, kruskal_matrix

START_NOISE = 0.3  # restarts >= 1 start from the MST of distances inflated by up to 30%
ANGLE_TOL_DEG = 1e-7  # an angle must be below 120 - this to trigger insertion
GAIN_TOL = 1e-12
COLLAPSE_TOL = 1e-9
SMOOTH_TOL = 1e-11
MAX_SWEEPS = 5000


@dataclass
class SteinerTree:
    """A Steiner tree: terminals first (indices 0..n-1), then Steiner points."""

    points: np.ndarray
    edges: list[tuple[int, int]]
    steiner: list[int]
    length: float
    n_terminals: int
    method: str = "heuristic"
    mst_length: float = field(default=float("nan"))

    @property
    def num_steiner(self) -> int:
        return len(self.steiner)

    @property
    def ratio(self) -> float:
        return self.length / self.mst_length

    def degrees(self) -> dict[int, int]:
        deg: dict[int, int] = {i: 0 for i in range(len(self.points))}
        for i, j in self.edges:
            deg[i] += 1
            deg[j] += 1
        return deg

    def steiner_angles(self) -> list[list[float]]:
        """For each Steiner point, the angles (deg) between consecutive edges."""
        nbrs: dict[int, list[int]] = {s: [] for s in self.steiner}
        for i, j in self.edges:
            if i in nbrs:
                nbrs[i].append(j)
            if j in nbrs:
                nbrs[j].append(i)
        out = []
        for s, ns in nbrs.items():
            p = self.points[s]
            th = sorted(math.atan2(self.points[q][1] - p[1], self.points[q][0] - p[0]) for q in ns)
            gaps = [math.degrees(th[(k + 1) % len(th)] - th[k]) % 360.0 for k in range(len(th))]
            out.append(gaps)
        return out


class _Tree:
    """Mutable tree used internally: positions, adjacency sets, alive flags."""

    def __init__(self, terminals: np.ndarray, edges):
        self.n = len(terminals)
        self.pos: list[tuple[float, float]] = [(float(x), float(y)) for x, y in terminals]
        self.adj: list[set[int]] = [set() for _ in range(self.n)]
        self.alive: list[bool] = [True] * self.n
        for i, j in edges:
            self.adj[i].add(j)
            self.adj[j].add(i)

    # -- basic edits ------------------------------------------------------
    def is_steiner(self, v: int) -> bool:
        return v >= self.n and self.alive[v]

    def add_point(self, p) -> int:
        self.pos.append((float(p[0]), float(p[1])))
        self.adj.append(set())
        self.alive.append(True)
        return len(self.pos) - 1

    def link(self, u: int, v: int) -> None:
        self.adj[u].add(v)
        self.adj[v].add(u)

    def unlink(self, u: int, v: int) -> None:
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def kill(self, s: int) -> None:
        for w in list(self.adj[s]):
            self.unlink(s, w)
        self.alive[s] = False

    def length(self) -> float:
        pos = self.pos
        total = 0.0
        for u in range(len(pos)):
            if not self.alive[u]:
                continue
            for w in self.adj[u]:
                if w > u:
                    total += dist(pos[u], pos[w])
        return total

    # -- heuristic steps -------------------------------------------------
    def candidates(self):
        """All improving insertions as (gain, v, a, b, steiner_position)."""
        pos = self.pos
        out = []
        limit = 120.0 - ANGLE_TOL_DEG
        for v in range(len(pos)):
            if not self.alive[v] or len(self.adj[v]) < 2:
                continue
            nb = list(self.adj[v])
            pv = pos[v]
            for x in range(len(nb)):
                a = nb[x]
                pa = pos[a]
                for y in range(x + 1, len(nb)):
                    b = nb[y]
                    pb = pos[b]
                    if not angle_at(pv, pa, pb) < limit:
                        continue
                    s = fermat_point(pv, pa, pb)
                    gain = dist(pv, pa) + dist(pv, pb) - (dist(s, pv) + dist(s, pa) + dist(s, pb))
                    if gain > GAIN_TOL:
                        out.append((gain, v, a, b, s))
        return out

    def insert(self, v: int, a: int, b: int, s_pos) -> int:
        s = self.add_point(s_pos)
        self.unlink(v, a)
        self.unlink(v, b)
        self.link(s, v)
        self.link(s, a)
        self.link(s, b)
        return s

    def steiner_cluster(self, seeds) -> list[int]:
        """Steiner points connected to ``seeds`` through Steiner-Steiner edges."""
        seen = set()
        stack = [s for s in seeds if self.is_steiner(s)]
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            for w in self.adj[s]:
                if self.is_steiner(w) and w not in seen:
                    stack.append(w)
        return sorted(seen)

    def smooth(self, cluster) -> None:
        """Gauss-Seidel sweeps moving each Steiner point to its neighbours' Fermat point."""
        pos = self.pos
        for _ in range(MAX_SWEEPS):
            biggest = 0.0
            for s in cluster:
                if not self.alive[s] or len(self.adj[s]) != 3:
                    continue
                a, b, c = (pos[w] for w in self.adj[s])
                new = fermat_point(a, b, c)
                move = dist(new, pos[s])
                if move > biggest:
                    biggest = move
                pos[s] = new
            if biggest < SMOOTH_TOL:
                break

    def cleanup(self) -> set[int]:
        """Remove degree<=2 Steiner points and merge collapsed ones. Returns touched nodes."""
        touched: set[int] = set()
        changed = True
        while changed:
            changed = False
            for s in range(self.n, len(self.pos)):
                if not self.alive[s]:
                    continue
                nb = list(self.adj[s])
                if len(nb) <= 1:
                    touched.update(nb)
                    self.kill(s)
                    changed = True
                    continue
                if len(nb) == 2:
                    u, w = nb
                    self.kill(s)
                    self.link(u, w)
                    touched.update((u, w))
                    changed = True
                    continue
                for u in nb:
                    if dist(self.pos[s], self.pos[u]) < COLLAPSE_TOL:
                        others = [w for w in nb if w != u]
                        self.kill(s)
                        for w in others:
                            self.link(u, w)
                        touched.add(u)
                        touched.update(others)
                        changed = True
                        break
        return touched


def _run_once(terminals: np.ndarray, rng: np.random.Generator | None, max_rounds: int) -> _Tree:
    if rng is None:
        edges, _ = kruskal(terminals)
    else:
        # Diversify the starting topology: MST of randomly inflated distances.
        D = distance_matrix(terminals)
        noise = 1.0 + START_NOISE * rng.random(D.shape)
        edges, _ = kruskal_matrix(D * np.triu(noise) + (D * np.triu(noise)).T)
    tree = _Tree(terminals, edges)
    for _ in range(max_rounds):
        cands = tree.candidates()
        if not cands:
            break
        cands.sort(key=lambda c: -c[0])
        if rng is None:
            pick = cands[0]
        else:
            pick = cands[int(rng.integers(min(3, len(cands))))]
        _, v, a, b, s_pos = pick
        s = tree.insert(v, a, b, s_pos)
        tree.smooth(tree.steiner_cluster([s]))
        touched = tree.cleanup()
        while touched:
            seeds = set(touched)
            for t in touched:
                seeds.update(tree.adj[t])
            tree.smooth(tree.steiner_cluster(seeds))
            touched = tree.cleanup()
    return tree


def steiner_tree(points, restarts: int = 16, seed: int = 0, max_rounds: int | None = None) -> SteinerTree:
    """Heuristic Euclidean Steiner tree for ``points`` (an upper bound on the SMT).

    Parameters
    ----------
    points : (n, 2) array-like of terminal coordinates.
    restarts : number of runs (>= 1); the best tree is kept.
    seed : seed for the shuffles / randomised choices in restarts 1, 2, ...
    max_rounds : safety cap on insertions per run (default 10 n + 10).

    Returns
    -------
    SteinerTree with ``points`` (terminals in input order, then Steiner points),
    ``edges``, ``steiner`` (indices of Steiner points), ``length`` and
    ``mst_length``.
    """
    P = np.asarray(points, dtype=float)
    n = len(P)
    _, L_mst = kruskal(P)
    if n <= 2:
        edges = [(0, 1)] if n == 2 else []
        return SteinerTree(P.copy(), edges, [], L_mst, n, mst_length=L_mst)
    max_rounds = 10 * n + 10 if max_rounds is None else max_rounds
    rng = np.random.default_rng(seed)

    best: tuple[float, _Tree, np.ndarray] | None = None
    for r in range(max(1, restarts)):
        order = np.arange(n) if r == 0 else rng.permutation(n)
        tree = _run_once(P[order], None if r == 0 else rng, max_rounds)
        L = tree.length()
        if best is None or L < best[0] - 1e-13:
            best = (L, tree, order)

    L, tree, order = best
    # Map permuted terminal j -> original index order[j]; Steiner points follow.
    index = {j: int(order[j]) for j in range(n)}
    steiner_ids = [s for s in range(n, len(tree.pos)) if tree.alive[s]]
    for k, s in enumerate(steiner_ids):
        index[s] = n + k
    pts = np.vstack([P, np.array([tree.pos[s] for s in steiner_ids]).reshape(-1, 2)])
    edges = []
    for u in range(len(tree.pos)):
        if not tree.alive[u]:
            continue
        for w in tree.adj[u]:
            if w > u:
                i, j = index[u], index[w]
                edges.append((min(i, j), max(i, j)))
    edges.sort()
    if L > L_mst:  # cannot happen (every move shortens), but guarantee it
        mst_edges, _ = kruskal(P)
        return SteinerTree(P.copy(), mst_edges, [], L_mst, n, mst_length=L_mst)
    return SteinerTree(pts, edges, list(range(n, n + len(steiner_ids))), L, n, mst_length=L_mst)
