"""Random-trial experiment: Steiner ratio rho = L(SMT) / L(MST) for uniform random points.

Writes
  results/random_trials.csv  (n, seed, L_mst, L_smt, rho, num_steiner_points, method)
  results/summary.json       (per-n mean / median / min / percentiles, trial counts, method)

Usage:  python scripts/run_experiment.py [--quick] [--workers K]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from _common import RESULTS  # noqa: F401  (also puts the repo root on sys.path)
from steiner import geosteiner
from steiner.pointsets import random_uniform

# n -> number of random trials (fewer for large n so the run stays short)
TRIALS = {5: 1000, 10: 1000, 20: 500, 30: 300, 50: 200, 75: 100, 100: 100}
QUICK_TRIALS = {5: 50, 10: 100, 20: 30, 30: 20, 50: 10, 75: 5, 100: 5}
RESTARTS = 16
HEADLINE_N = 10  # the "1,000 random trials" headline experiment
PLATEAU_MIN_N = 50  # pooled trials with n >= this define the large-n plateau


def trial(args):
    n, seed = args
    # seed the point set by (n, seed) so every (n, seed) pair is reproducible
    P = random_uniform(n, rng=np.random.default_rng([n, seed]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = geosteiner.smt(P, restarts=RESTARTS, seed=seed, warn=False)
    return dict(n=n, seed=seed, L_mst=t.mst_length, L_smt=t.length, rho=t.length / t.mst_length,
                num_steiner_points=t.num_steiner, method=t.method)


def summarise(rows):
    out = {}
    for n in sorted({r["n"] for r in rows}):
        rho = np.array([r["rho"] for r in rows if r["n"] == n])
        ns = np.array([r["num_steiner_points"] for r in rows if r["n"] == n])
        methods = sorted({r["method"] for r in rows if r["n"] == n})
        out[str(n)] = dict(
            n=n, trials=int(len(rho)), mean=float(rho.mean()), median=float(np.median(rho)),
            min=float(rho.min()), max=float(rho.max()), std=float(rho.std(ddof=1)) if len(rho) > 1 else 0.0,
            p10=float(np.percentile(rho, 10)), p90=float(np.percentile(rho, 90)),
            mean_reduction_pct=float(100 * (1 - rho.mean())),
            mean_steiner_points=float(ns.mean()), method="+".join(methods),
        )
    big = np.array([r["rho"] for r in rows if r["n"] >= PLATEAU_MIN_N])
    return dict(
        description="Steiner ratio rho = L(SMT)/L(MST) for n points uniform in the unit square. "
                    "With method 'heuristic', L(SMT) is an upper bound, so rho is an upper bound too.",
        restarts=RESTARTS,
        headline_n=HEADLINE_N,
        plateau=dict(min_n=PLATEAU_MIN_N, trials=int(len(big)), mean=float(big.mean()) if len(big) else None),
        by_n=out,
    )


def print_table(summary):
    print(f"\n{'n':>4} {'trials':>7} {'mean rho':>9} {'median':>8} {'min':>8} {'p10':>8} {'p90':>8} "
          f"{'saving':>7} {'#Steiner':>8}  method")
    for s in summary["by_n"].values():
        print(f"{s['n']:>4} {s['trials']:>7} {s['mean']:>9.4f} {s['median']:>8.4f} {s['min']:>8.4f} "
              f"{s['p10']:>8.4f} {s['p90']:>8.4f} {s['mean_reduction_pct']:>6.2f}% "
              f"{s['mean_steiner_points']:>8.2f}  {s['method']}")
    p = summary["plateau"]
    if p["mean"] is not None:
        print(f"plateau (all trials with n >= {p['min_n']}, {p['trials']} trials): mean rho = {p['mean']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="few trials, for a smoke test")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    trials = QUICK_TRIALS if args.quick else TRIALS
    method = geosteiner.method_name()
    print(f"solver: {method}" + ("" if method == "geosteiner" else
          "  (GeoSteiner not found; heuristic values are upper bounds on the true SMT)"))
    jobs = [(n, seed) for n, k in trials.items() for seed in range(k)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(trial, jobs, chunksize=8))
    print(f"{len(rows)} trials in {time.time() - t0:.1f} s using {args.workers} worker(s)")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "random_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "seed", "L_mst", "L_smt", "rho", "num_steiner_points", "method"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "L_mst": f"{r['L_mst']:.12f}", "L_smt": f"{r['L_smt']:.12f}", "rho": f"{r['rho']:.12f}"})
    summary = summarise(rows)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RESULTS / 'random_trials.csv'} and {RESULTS / 'summary.json'}")
    print_table(summary)


if __name__ == "__main__":
    main()
