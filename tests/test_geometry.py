import math

import numpy as np
import pytest

from steiner.geometry import angle_at, dist, fermat_point, l1_dist, weiszfeld
from steiner.topology import num_full_topologies


def test_distances():
    assert dist((0, 0), (3, 4)) == 5
    assert l1_dist((0, 0), (3, -4)) == 7


def test_angle():
    assert angle_at((0, 0), (1, 0), (0, 1)) == pytest.approx(90)
    assert angle_at((0, 0), (1, 0), (-1, 0)) == pytest.approx(180)
    assert math.isnan(angle_at((0, 0), (0, 0), (1, 1)))


@pytest.mark.parametrize("seed", range(30))
def test_fermat_closed_form_matches_weiszfeld(seed):
    tri = np.random.default_rng(seed).random((3, 2))
    f = fermat_point(*tri)
    w = weiszfeld(tri)
    cost = lambda p: sum(dist(p, q) for q in tri)
    assert cost(f) == pytest.approx(cost(w), abs=1e-9)
    assert cost(f) <= cost(w) + 1e-9


def test_fermat_point_has_120_degree_angles():
    a, b, c = (0, 0), (4, 0), (1, 3)
    f = fermat_point(a, b, c)
    for p, q in ((a, b), (b, c), (a, c)):
        assert angle_at(f, p, q) == pytest.approx(120, abs=1e-6)


def test_fermat_point_obtuse_vertex():
    a, b, c = (0, 0), (4, 0), (1, 0.5)  # angle at c is about 144 degrees
    assert angle_at(c, a, b) > 120
    assert fermat_point(a, b, c) == pytest.approx(c)


def test_fermat_degenerate_inputs():
    assert fermat_point((1, 1), (1, 1), (5, 2)) == pytest.approx((1, 1))
    assert fermat_point((0, 0), (1, 0), (2, 0)) == pytest.approx((1, 0))


def test_topology_counts():
    assert [num_full_topologies(n) for n in (3, 4, 5, 6)] == [1, 3, 15, 105]
    assert num_full_topologies(10) == 2_027_025
