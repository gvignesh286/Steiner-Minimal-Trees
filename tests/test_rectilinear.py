import pytest

from steiner.pointsets import random_uniform
from steiner.rectilinear import exact_rsmt, hanan_grid, rectilinear_mst

PLUS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def test_plus_shape():
    _, L_mst = rectilinear_mst(PLUS)
    t = exact_rsmt(PLUS)
    assert L_mst == pytest.approx(6)
    assert t.length == pytest.approx(4)
    assert t.ratio == pytest.approx(2 / 3)
    assert t.points[t.steiner[0]] == pytest.approx((0, 0))


def test_hanan_grid_size():
    assert len(hanan_grid(PLUS)) == 9 - 4


@pytest.mark.parametrize("seed", range(10))
def test_rsmt_bounds(seed):
    P = random_uniform(5, seed)
    _, L_mst = rectilinear_mst(P)
    t = exact_rsmt(P)
    # Hwang (1976): the rectilinear Steiner ratio is 2/3
    assert 2 / 3 * L_mst - 1e-12 <= t.length <= L_mst + 1e-12
    assert t.num_steiner <= len(P) - 2


def test_too_many_points_rejected():
    with pytest.raises(ValueError):
        exact_rsmt(random_uniform(7, 0))
