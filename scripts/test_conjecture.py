"""Search for point sets with the smallest Steiner ratio rho = L(SMT) / L(MST).

Gilbert-Pollak (1968) conjectured rho >= sqrt(3)/2 ~ 0.866 for every planar
point set, with equality for the equilateral triangle. We evaluate structured
families (regular polygons, triangular-lattice patches, square grids) and run a
random search (random sets for n = 3..8 followed by local hill-climbing on the
best ones).

The default solver is a heuristic, which can only OVER-estimate L(SMT) and so
rho; a heuristic value below sqrt(3)/2 would therefore indicate a bug, not a
counterexample.

Writes results/conjecture.csv with columns: family, n, rho, reduction_pct.
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings

import numpy as np

from _common import RESULTS  # noqa: F401
from steiner import geosteiner
from steiner.pointsets import (equilateral_triangle, hexagonal_lattice_patch, random_uniform,
                               regular_polygon, square_grid, triangular_lattice_patch, unit_square)

BOUND = math.sqrt(3) / 2


def rho(P, restarts=16, seed=0) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = geosteiner.smt(P, restarts=restarts, seed=seed, warn=False)
    return t.length / t.mst_length


def structured():
    yield "equilateral triangle", equilateral_triangle()
    yield "square", unit_square()
    yield "regular hexagon", regular_polygon(6)
    for rows in (3, 4, 5, 6):
        yield "triangular lattice patch", triangular_lattice_patch(rows)
    for radius in (1, 2):
        yield "triangular lattice patch", hexagonal_lattice_patch(radius)
    for k in (3, 4, 5):
        yield "square grid", square_grid(k)


def random_search(ns, samples_per_n, climb_steps, rng):
    """Best random point set per n: random sampling, then hill-climbing."""
    best = {}
    for n in ns:
        cands = []
        for i in range(samples_per_n):
            P = random_uniform(n, rng=rng)
            cands.append((rho(P, restarts=8, seed=i), P))
        cands.sort(key=lambda c: c[0])
        r_best, P_best = cands[0]
        for r0, P in cands[:3]:  # climb from the three best random starts
            r_cur, P_cur = r0, P
            sigma = 0.05
            for step in range(climb_steps):
                Q = P_cur + rng.normal(0, sigma, P_cur.shape)
                r_new = rho(Q, restarts=8, seed=step)
                if r_new < r_cur:
                    r_cur, P_cur = r_new, Q
                sigma = max(0.002, sigma * 0.995)
            if r_cur < r_best:
                r_best, P_best = r_cur, P_cur
        best[n] = (rho(P_best, restarts=32, seed=1), P_best)  # final, more thorough evaluation
        print(f"  random search n={n}: best rho = {best[n][0]:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=453)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for fam, P in structured():
        r = rho(P)
        rows.append(dict(family=fam, n=len(P), rho=r))
        print(f"  {fam:<26} n={len(P):>3}  rho = {r:.4f}")
    samples, steps = (40, 30) if args.quick else (400, 250)
    for n, (r, _) in random_search(range(3, 9), samples, steps, rng).items():
        rows.append(dict(family="random search", n=n, rho=r))

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "conjecture.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "n", "rho", "reduction_pct"])
        w.writeheader()
        for r in rows:
            w.writerow(dict(family=r["family"], n=r["n"], rho=f"{r['rho']:.10f}",
                            reduction_pct=f"{100 * (1 - r['rho']):.4f}"))
    print(f"wrote {RESULTS / 'conjecture.csv'}")

    print(f"\nSmallest rho per family (conjectured lower bound sqrt(3)/2 = {BOUND:.4f}):")
    fams = {}
    for r in rows:
        if r["family"] not in fams or r["rho"] < fams[r["family"]]["rho"]:
            fams[r["family"]] = r
    for fam, r in sorted(fams.items(), key=lambda kv: kv[1]["rho"]):
        print(f"  {fam:<26} n={r['n']:>3}  rho = {r['rho']:.4f}  ({100 * (1 - r['rho']):.1f}% shorter)")
    worst = min(r["rho"] for r in rows)
    print("no point set beat sqrt(3)/2" if worst >= BOUND - 1e-9 else
          f"WARNING: rho = {worst} < sqrt(3)/2 -- this would be a bug in the solver")


if __name__ == "__main__":
    main()
