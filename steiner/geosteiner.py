"""Optional wrapper around the GeoSteiner exact solver.

GeoSteiner (http://www.geosteiner.com/) computes exact Euclidean SMTs for
hundreds of points. It is not on PyPI; build it from source and put the
``efst`` and ``bb`` programs on your PATH. The pipeline is::

    efst < points.txt | bb

If the binaries are missing (or their output cannot be parsed) we fall back to
the heuristic in :mod:`steiner.smt_heuristic` and say so with a warning, and
the result's ``method`` field is ``"heuristic"`` so results files record which
solver produced each number.

Note: the output parser looks for a "length" figure in bb's text output and
was written against GeoSteiner 5.x documentation; it has not been tested here
because GeoSteiner was not installed when this repo was built.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import warnings

import numpy as np

from .mst import kruskal
from .smt_heuristic import SteinerTree, steiner_tree

_LENGTH_RE = re.compile(r"length\s*[=:]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.IGNORECASE)


def available() -> bool:
    """True if both GeoSteiner programs are on PATH."""
    return shutil.which("efst") is not None and shutil.which("bb") is not None


def method_name() -> str:
    return "geosteiner" if available() else "heuristic"


def _geosteiner_length(points: np.ndarray, timeout: float = 600.0) -> float | None:
    text = "\n".join(f"{x:.17g} {y:.17g}" for x, y in points) + "\n"
    efst = subprocess.run(["efst"], input=text, capture_output=True, text=True, timeout=timeout, check=True)
    bb = subprocess.run(["bb"], input=efst.stdout, capture_output=True, text=True, timeout=timeout, check=True)
    m = None
    for m in _LENGTH_RE.finditer(bb.stdout):
        pass
    return float(m.group(1)) if m else None


def smt(points, restarts: int = 16, seed: int = 0, warn: bool = True) -> SteinerTree:
    """Exact SMT length via GeoSteiner if installed, otherwise the heuristic.

    With GeoSteiner only the length is returned (``edges``/``steiner`` are
    empty and ``num_steiner`` is 0) because only the length is parsed.
    """
    P = np.asarray(points, dtype=float)
    if available():
        try:
            L = _geosteiner_length(P)
            if L is not None:
                _, L_mst = kruskal(P)
                return SteinerTree(P.copy(), [], [], L, len(P), method="geosteiner", mst_length=L_mst)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            if warn:
                warnings.warn(f"GeoSteiner failed ({exc}); falling back to the heuristic")
    elif warn:
        warnings.warn("GeoSteiner (efst, bb) not found on PATH; using the heuristic (upper bound)",
                      stacklevel=2)
    return steiner_tree(P, restarts=restarts, seed=seed)
