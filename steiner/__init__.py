"""Steiner minimal trees vs minimum spanning trees (Math 453 final project)."""

from .exact_small import exact_smt
from .geometry import angle_at, dist, fermat_point, l1_dist, tree_length, weiszfeld
from .mst import UnionFind, kruskal, mst_length
from .rectilinear import exact_rsmt, rectilinear_mst
from .smt_heuristic import SteinerTree, steiner_tree
from .topology import num_full_topologies

STEINER_RATIO_CONJECTURE = 3 ** 0.5 / 2  # Gilbert-Pollak: rho >= sqrt(3)/2
RECTILINEAR_RATIO = 2 / 3  # Hwang 1976

__all__ = [
    "SteinerTree", "UnionFind", "angle_at", "dist", "exact_rsmt", "exact_smt",
    "fermat_point", "kruskal", "l1_dist", "mst_length", "num_full_topologies",
    "rectilinear_mst", "steiner_tree", "tree_length", "weiszfeld",
    "STEINER_RATIO_CONJECTURE", "RECTILINEAR_RATIO",
]
