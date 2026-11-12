import numpy as np
import pytest
from scipy.sparse.csgraph import minimum_spanning_tree

from steiner.mst import UnionFind, distance_matrix, kruskal
from steiner.pointsets import random_uniform


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("n", [2, 5, 17, 60])
def test_kruskal_matches_scipy(n, seed):
    P = random_uniform(n, seed)
    edges, L = kruskal(P)
    ref = minimum_spanning_tree(distance_matrix(P)).sum()
    assert len(edges) == n - 1
    assert L == pytest.approx(ref, abs=1e-9)


def test_kruskal_l1_matches_scipy():
    P = random_uniform(30, 7)
    _, L = kruskal(P, metric="l1")
    assert L == pytest.approx(minimum_spanning_tree(distance_matrix(P, "l1")).sum(), abs=1e-9)


def test_kruskal_is_spanning_tree():
    P = random_uniform(40, 3)
    edges, _ = kruskal(P)
    uf = UnionFind(len(P))
    for i, j in edges:
        assert uf.union(i, j)  # no cycles
    assert len({uf.find(i) for i in range(len(P))}) == 1  # connected


def test_union_find():
    uf = UnionFind(5)
    assert uf.union(0, 1) and uf.union(3, 4)
    assert not uf.union(1, 0)
    assert uf.find(0) == uf.find(1) and uf.find(0) != uf.find(3)


def test_trivial_inputs():
    assert kruskal(np.zeros((1, 2))) == ([], 0.0)
    assert kruskal([[0, 0], [3, 4]])[1] == pytest.approx(5.0)
