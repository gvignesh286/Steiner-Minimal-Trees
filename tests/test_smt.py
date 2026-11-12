import math

import pytest

from steiner.exact_small import exact_smt
from steiner.mst import kruskal
from steiner.pointsets import equilateral_triangle, random_uniform, unit_square
from steiner.smt_heuristic import steiner_tree

SQRT3 = math.sqrt(3)


@pytest.mark.parametrize("solver", [steiner_tree, exact_smt])
def test_equilateral_triangle(solver):
    P = equilateral_triangle()
    t = solver(P)
    assert kruskal(P)[1] == pytest.approx(2, abs=1e-6)
    assert t.length == pytest.approx(SQRT3, abs=1e-6)
    assert t.ratio == pytest.approx(SQRT3 / 2, abs=1e-6)
    assert t.num_steiner == 1


@pytest.mark.parametrize("solver", [steiner_tree, exact_smt])
def test_unit_square(solver):
    P = unit_square()
    t = solver(P)
    assert kruskal(P)[1] == pytest.approx(3, abs=1e-6)
    assert t.length == pytest.approx(1 + SQRT3, abs=1e-6)
    assert t.ratio == pytest.approx((1 + SQRT3) / 3, abs=1e-6)
    assert t.num_steiner == 2


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("n", [5, 10, 30])
def test_heuristic_invariants(n, seed):
    P = random_uniform(n, seed)
    t = steiner_tree(P, restarts=3, seed=seed)
    # never longer than the MST, at most n - 2 Steiner points
    assert t.length <= t.mst_length + 1e-12
    assert t.num_steiner <= n - 2
    # it is a tree on all points
    assert len(t.edges) == len(t.points) - 1
    # every Steiner point has degree 3 with 120-degree angles (within 1 degree)
    deg = t.degrees()
    assert all(deg[s] == 3 for s in t.steiner)
    for gaps in t.steiner_angles():
        assert all(abs(g - 120) < 1.0 for g in gaps), gaps
    # reported length matches the edges
    assert t.length == pytest.approx(sum(math.dist(t.points[i], t.points[j]) for i, j in t.edges))


def test_heuristic_matches_exact_on_four_points():
    trials = 50
    mismatches = []
    for seed in range(trials):
        P = random_uniform(4, 10_000 + seed)
        h = steiner_tree(P, restarts=16, seed=seed)
        e = exact_smt(P)
        assert h.length >= e.length - 1e-9  # the exact solver is a lower bound
        if h.length - e.length > 1e-6:
            mismatches.append((seed, h.length, e.length))
    rate = 1 - len(mismatches) / trials
    print(f"\nheuristic == exact on {trials - len(mismatches)}/{trials} random 4-point sets "
          f"({rate:.0%}); mismatches: {mismatches}")
    assert rate >= 0.9, mismatches


def test_exact_small_three_points_obtuse():
    P = [(0, 0), (4, 0), (1, 0.5)]  # angle >= 120: SMT is the MST
    t = exact_smt(P)
    assert t.num_steiner == 0
    assert t.length == pytest.approx(kruskal(P)[1])
