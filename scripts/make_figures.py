"""Regenerate every figure into figures/ (200 dpi PNG, white background).

Figures 08-10 read results/*.csv, so run scripts/run_experiment.py and
scripts/test_conjecture.py first (``make all`` does this in order).
"""

from __future__ import annotations

import csv
import math

import matplotlib.pyplot as plt
import numpy as np

from _common import FIGURES, RESULTS
from steiner import plotting as sty
from steiner.exact_small import exact_smt
from steiner.geometry import angle_at, dist, fermat_point
from steiner.mst import kruskal
from steiner.pointsets import equilateral_triangle, random_uniform, unit_square
from steiner.rectilinear import exact_rsmt, l_shape_path, rectilinear_mst
from steiner.smt_heuristic import steiner_tree
from steiner.topology import num_full_topologies

SQ3 = math.sqrt(3)
BOUND = SQ3 / 2
UNITS = "length units"


# ----------------------------------------------------------------- helpers
def arcs_at(ax, p, nbrs, r=0.1, label=True, color=sty.ACCENT, fontsize=10, text_r=None):
    """Draw the angles between consecutive edges at point p (with degree labels)."""
    th = sorted(math.degrees(math.atan2(q[1] - p[1], q[0] - p[0])) for q in nbrs)
    for k in range(len(th)):
        a1, a2 = th[k], th[(k + 1) % len(th)]
        if a2 <= a1:
            a2 += 360
        sty.angle_arc(ax, p, a1, a2, inset=16, r=r, text=f"{a2 - a1:.0f}°" if label else None,
                      color=color, fontsize=fontsize, text_r=text_r)


def neighbours(tree, s):
    return [tree.points[j if i == s else i] for i, j in tree.edges if s in (i, j)]


def mst_vs_smt(points, title, fname, edge_fmt, pad=0.18, arrow_offsets=None):
    P = np.asarray(points, float)
    mst_edges, L_mst = kruskal(P)
    smt = exact_smt(P)
    saving = 100 * (1 - smt.length / L_mst)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, kind in zip(axes, ("mst", "smt")):
        if kind == "mst":
            sty.draw_edges(ax, P, mst_edges, kind="mst", label="MST edge")
            sty.draw_terminals(ax, P)
            for i, j in mst_edges:
                sty.label_edge(ax, P[i], P[j], edge_fmt(dist(P[i], P[j])), side=-1 if i == 0 and j == 1 else 1)
            ax.set_title(f"Minimum spanning tree (MST)\ntotal length = {L_mst:.3f}")
        else:
            sty.draw_tree(ax, smt, kind="smt", label="Steiner tree edge")
            for i, j in smt.edges:
                sty.label_edge(ax, smt.points[i], smt.points[j], edge_fmt(dist(smt.points[i], smt.points[j])),
                               offset=0.07)
            for k, s in enumerate(smt.steiner):
                off = (arrow_offsets or [(0.25, -0.3)] * 4)[k]
                ax.annotate("added\nSteiner point", xy=smt.points[s], xytext=smt.points[s] + np.array(off),
                            color=sty.STEINER, fontsize=10, ha="left", va="center", fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color=sty.STEINER, lw=1.5, shrinkB=7))
            ax.set_title(f"Steiner minimal tree (SMT)\ntotal length = {smt.length:.3f}  "
                         f"({saving:.1f}% shorter)", color="black")
        sty.equal_axes(ax, P, pad=pad, units=UNITS)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9)
    fig.suptitle(title.format(saving=saving, L_mst=L_mst, L_smt=smt.length))
    fig.tight_layout()
    sty.save(fig, FIGURES / fname)
    return L_mst, smt.length


# ----------------------------------------------------------------- figures
def fig01_triangle():
    mst_vs_smt(equilateral_triangle(), "Equilateral triangle (side 1): adding one junction makes the "
               "network {saving:.1f}% shorter\nMST = 2 vs SMT = √3 ≈ {L_smt:.3f}", "01_triangle.png",
               lambda d: f"{d:.3f}", arrow_offsets=[(0.18, 0.3)])


def fig02_square():
    mst_vs_smt(unit_square(), "Unit square: two Steiner points make the network {saving:.1f}% shorter\n"
               "MST = 3 vs SMT = 1 + √3 ≈ {L_smt:.3f}", "02_square.png", lambda d: f"{d:.3f}",
               arrow_offsets=[(0.3, 0.12), (0.3, -0.12)])


def fig03_120_rule():
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    s = np.array([0.0, 0.0])
    T = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (90, 210, 330)])
    sty.draw_edges(ax, np.vstack([T, s]), [(0, 3), (1, 3), (2, 3)], kind="smt", label="Steiner tree edge")
    sty.draw_terminals(ax, T)
    sty.draw_steiner_points(ax, s, size=140)
    arcs_at(ax, s, T, r=0.22, fontsize=13, text_r=0.38)
    ax.annotate("Steiner point:\nexactly 3 edges", xy=s, xytext=(0.55, -0.05), fontsize=10,
                color=sty.STEINER, arrowprops=dict(arrowstyle="->", color=sty.STEINER, shrinkB=9))
    ax.text(0, -0.88, "If any angle were below 120°, sliding the junction\nwould shorten the tree, "
            "so at the optimum all three are 120°.", ha="center", fontsize=9.5, color=sty.MUTED)
    sty.equal_axes(ax, np.vstack([T, [[0, -1.0]]]), pad=0.12, units=UNITS)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("The 120° rule: every Steiner point has\nthree edges meeting at 120°")
    sty.save(fig, FIGURES / "03_120_rule.png")


def fig04_why_120():
    v, a, b = np.array([0.0, 0.0]), np.array([-0.5, 1.0]), np.array([0.5, 1.0])
    ang = angle_at(v, a, b)
    L0 = dist(v, a) + dist(v, b)
    s = np.array(fermat_point(v, a, b))
    L1 = dist(s, v) + dist(s, a) + dist(s, b)
    P = np.vstack([v, a, b])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    ax = axes[0]
    sty.draw_edges(ax, P, [(0, 1), (0, 2)], kind="mst", label="edge")
    sty.draw_terminals(ax, P)
    th_a, th_b = (math.degrees(math.atan2(*(q - v)[::-1])) for q in (a, b))
    sty.angle_arc(ax, v, th_b, th_a, r=0.22, text=f"{ang:.0f}° < 120°", text_r=0.42, color="#c0392b")
    for q in (a, b):
        sty.label_edge(ax, v, q, f"{dist(v, q):.3f}", offset=0.09, side=1 if q[0] > 0 else -1)
    ax.set_title(f"Junction at a given point, angle {ang:.0f}°\ntotal length = {L0:.3f}")
    ax = axes[1]
    Q = np.vstack([v, a, b, s])
    sty.draw_edges(ax, Q, [(0, 3), (1, 3), (2, 3)], kind="smt", label="Steiner tree edge")
    sty.draw_terminals(ax, P)
    sty.draw_steiner_points(ax, s)
    arcs_at(ax, s, P, r=0.1, fontsize=9, text_r=0.2)
    ax.annotate("added\nSteiner point", xy=s, xytext=s + np.array([0.12, -0.32]), color=sty.STEINER,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=sty.STEINER, shrinkB=7))
    ax.set_title(f"Add a Steiner point (all angles 120°)\ntotal length = {L1:.3f}  "
                 f"({100 * (1 - L1 / L0):.1f}% shorter)")
    for ax in axes:
        sty.equal_axes(ax, P, pad=0.15, units=UNITS)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Why 120°: an angle below 120° can always be shortened by adding a junction")
    fig.tight_layout()
    sty.save(fig, FIGURES / "04_why_120.png")


def fig05_soap_film():
    pins = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.6], [0.0, 1.6]])
    smt = exact_smt(pins)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.add_patch(plt.Rectangle((-0.45, -0.45), 3.9, 2.5, fc="#dbe7f3", ec="#7a9cc6", lw=2, zorder=0,
                               label="glass plate"))
    for i, j in smt.edges:  # film: wide translucent band + bright core
        p, q = smt.points[i], smt.points[j]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#8ec5ff", lw=11, alpha=0.55, solid_capstyle="round", zorder=2)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=sty.STEINER, lw=2, zorder=3)
    ax.plot([], [], color="#8ec5ff", lw=8, alpha=0.6, label="soap film")
    ax.scatter(pins[:, 0], pins[:, 1], s=180, color=sty.TERMINAL, zorder=5, label="pin (terminal)")
    sty.draw_steiner_points(ax, smt.points[smt.steiner], size=90, label="film junction (Steiner point)")
    for s, off in zip(smt.steiner, ((0.0, 0.3), (0.0, 0.3))):
        arcs_at(ax, smt.points[s], neighbours(smt, s), r=0.16, fontsize=9, text_r=0.33)
    ax.text(1.5, -0.3, f"film length = {smt.length:.2f} cm  vs  MST = {smt.mst_length:.2f} cm", ha="center",
            fontsize=10, color=sty.MUTED)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(-0.6, 2.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9)
    ax.set_title("Soap film between 4 pins (stylized): surface tension finds\n"
                 "the Steiner tree, with two junctions at 120°")
    sty.save(fig, FIGURES / "05_soap_film.png")


def _sci(x: int) -> str:
    e = int(math.floor(math.log10(x)))
    return f"{x / 10 ** e:.1f} × $10^{{{e}}}$"


def fig06_topology_growth():
    ns = np.arange(3, 21)
    counts = [num_full_topologies(int(n)) for n in ns]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.semilogy(ns, counts, "o-", color=sty.STEINER, lw=2, ms=6)
    notes = {5: f"n = 5 → {num_full_topologies(5)}",
             10: f"n = 10 → {num_full_topologies(10):,}\n(≈ {num_full_topologies(10) / 1e6:.0f} million)",
             20: f"n = 20 → ≈ {_sci(num_full_topologies(20))}"}
    offsets = {5: (0.6, 3e3), 10: (0.8, 3e8), 20: (-6.2, 2e1)}
    for n, txt in notes.items():
        c = num_full_topologies(n)
        ax.annotate(txt, xy=(n, c), xytext=(n + offsets[n][0], c * offsets[n][1]) if n != 20 else
                    (n + offsets[n][0], c * 1.5), fontsize=10,
                    arrowprops=dict(arrowstyle="->", color=sty.MUTED), va="center")
    ax.set_xlabel("number of terminals n")
    ax.set_ylabel("number of full Steiner topologies, (2n − 5)!!  (log scale)")
    ax.set_xticks(ns)
    ax.set_ylim(0.3, 1e23)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.set_title("Possible tree layouts explode as n grows\n(one reason the Euclidean SMT problem is NP-hard)")
    sty.save(fig, FIGURES / "06_topology_growth.png")


def fig07_random_overlay(seed=7):
    P = random_uniform(10, seed)
    mst_edges, L_mst = kruskal(P)
    t = steiner_tree(P, restarts=32, seed=seed)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    sty.draw_edges(ax, P, mst_edges, kind="mst", label=f"MST (length {L_mst:.3f})")
    sty.draw_tree(ax, t, kind="smt", label=f"Steiner tree (length {t.length:.3f})",
                  steiner_label=f"Steiner points ({t.num_steiner} added)")
    sty.equal_axes(ax, P, pad=0.08, units=UNITS)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9.5)
    ax.set_title(f"10 random points (seed {seed}): MST = {L_mst:.3f}, Steiner = {t.length:.3f}\n"
                 f"ρ = {t.ratio:.4f}  ({100 * (1 - t.ratio):.1f}% shorter; Steiner tree from heuristic)")
    sty.save(fig, FIGURES / "07_random_overlay.png")


def _read_csv(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def fig08_ratio_histogram(n=10):
    rows = [r for r in _read_csv("random_trials.csv") if int(r["n"]) == n]
    rho = np.array([float(r["rho"]) for r in rows])
    mean = rho.mean()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.hist(rho, bins=40, color=sty.STEINER, alpha=0.8, edgecolor="white")
    ax.axvline(BOUND, ls="--", color=sty.TERMINAL, lw=1.5)
    ax.axvline(mean, ls="--", color=sty.ACCENT, lw=1.5)
    ytop = ax.get_ylim()[1]
    ax.text(BOUND + 0.002, ytop * 0.93, f"conjectured limit\n√3/2 ≈ {BOUND:.3f}", fontsize=10, va="top")
    ax.text(mean - 0.002, ytop * 0.93, f"mean ρ = {mean:.3f}", color=sty.ACCENT, fontsize=10, va="top", ha="right")
    ax.annotate(f"min ρ = {rho.min():.3f}", xy=(rho.min(), 1), xytext=(rho.min() - 0.012, ytop * 0.2),
                fontsize=9, color=sty.MUTED, ha="center", arrowprops=dict(arrowstyle="->", color=sty.MUTED))
    ax.set_xlim(BOUND - 0.015, 1.005)
    ax.set_xlabel("Steiner ratio ρ = L(Steiner tree) / L(MST)   (lower = bigger savings)")
    ax.set_ylabel("number of trials")
    method = rows[0]["method"]
    ax.set_title(f"{len(rho):,} random {n}-point sets: Steiner trees are about {100 * (1 - mean):.1f}% shorter "
                 f"on average\n(uniform points in the unit square; solver: {method})")
    sty.save(fig, FIGURES / "08_ratio_histogram.png")


def fig09_ratio_vs_n():
    import json
    summary = json.loads((RESULTS / "summary.json").read_text())["by_n"]
    ns = np.array([s["n"] for s in summary.values()])
    mean = np.array([s["mean"] for s in summary.values()])
    p10 = np.array([s["p10"] for s in summary.values()])
    p90 = np.array([s["p90"] for s in summary.values()])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.fill_between(ns, p10, p90, color=sty.STEINER, alpha=0.2, label="10th–90th percentile")
    ax.plot(ns, mean, "o-", color=sty.STEINER, lw=2, label="mean ρ")
    ax.axhline(BOUND, ls="--", color=sty.TERMINAL, lw=1.5, label=f"conjectured limit √3/2 ≈ {BOUND:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{s['n']}\n({s['trials']})" for s in summary.values()], fontsize=8.5)
    ax.minorticks_off()
    ax.set_ylim(BOUND - 0.02, 1.005)
    ax.set_xlabel("number of random points n, log scale  (trials per n in parentheses)")
    ax.set_ylabel("Steiner ratio ρ  (lower = bigger savings)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.25)
    plateau = np.mean(mean[ns >= 50]) if (ns >= 50).any() else mean[-1]
    ax.set_title(f"The average saving levels off near {100 * (1 - plateau):.0f}% as n grows\n"
                 f"(mean ρ ≈ {plateau:.3f} for n ≥ 50, far above the worst case √3/2)")
    sty.save(fig, FIGURES / "09_ratio_vs_n.png")


def fig10_conjecture_test():
    rows = _read_csv("conjecture.csv")
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], []).append((float(r["rho"]), int(r["n"])))
    order = sorted(fams, key=lambda f: -min(fams[f])[0])  # smallest at the top
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for y, f in enumerate(order):
        vals = fams[f]
        ax.scatter([v[0] for v in vals], [y] * len(vals), s=25, color=sty.MST, zorder=2,
                   label="other instances" if y == 0 else None)
        best, n = min(vals)
        ax.scatter([best], [y], s=90, color=sty.STEINER, zorder=3, label="smallest ρ in family" if y == 0 else None)
        extra = ", hexagon + centre" if f == "triangular lattice patch" and n == 7 else ""
        ax.text(best + 0.004, y + 0.18, f"ρ = {best:.4f} (n = {n}{extra})", fontsize=9, va="bottom")
        if f == "equilateral triangle":
            ax.annotate("extremal case", xy=(best, y), xytext=(best + 0.035, y - 0.42), fontsize=10,
                        fontweight="bold", color=sty.ACCENT,
                        arrowprops=dict(arrowstyle="->", color=sty.ACCENT, shrinkB=6))
    ax.axvline(BOUND, ls="--", color=sty.TERMINAL, lw=1.5)
    ax.text(BOUND - 0.003, len(order) - 0.45, f"0.866 = √3/2\n(conjectured limit)", ha="right", va="top", fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.3)
    ax.set_xlim(BOUND - 0.04, 1.02)
    ax.set_xlabel("Steiner ratio ρ = L(SMT) / L(MST)   (lower = bigger savings)")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_title("Testing the Gilbert–Pollak conjecture: no point set beats √3/2\n"
                 "(smallest ρ found in each family of point sets)")
    sty.save(fig, FIGURES / "10_conjecture_test.png")


def fig11_rectilinear():
    P = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]])
    mst_edges, L_mst = rectilinear_mst(P)
    t = exact_rsmt(P)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax in axes:
        ax.set_xticks(np.arange(-1.5, 1.6, 0.5))
        ax.set_yticks(np.arange(-1.5, 1.6, 0.5))
        ax.grid(True, color="#d1d5db", lw=0.8)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x ({UNITS})")
        ax.set_ylabel(f"y ({UNITS})")
    ax = axes[0]
    for k, (i, j) in enumerate(mst_edges):
        # draw each L1 edge as an 'L', bending away from the centre
        paths = [l_shape_path(P[i], P[j], h) for h in (True, False)]
        path = max(paths, key=lambda pth: np.hypot(*pth[1]))
        ax.plot(path[:, 0], path[:, 1], color=sty.MST, lw=5, alpha=0.75, solid_capstyle="round",
                label="L1 MST edge (length 2)" if k == 0 else None)
    sty.draw_terminals(ax, P)
    ax.set_title(f"Rectilinear MST\ntotal length = {L_mst:.0f}")
    ax = axes[1]
    sty.draw_tree(ax, t, kind="smt", label="RSMT edge (length 1)")
    ax.annotate("Steiner point on\nthe Hanan grid", xy=t.points[t.steiner[0]], xytext=(0.55, 0.75),
                color=sty.STEINER, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=sty.STEINER, shrinkB=7))
    ax.set_title(f"Rectilinear Steiner minimal tree\ntotal length = {t.length:.0f}   →   "
                 f"ratio = {t.length:.0f}/{L_mst:.0f} = 2/3")
    for ax in axes:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9)
    fig.suptitle("Grid (L1, 'Manhattan') distance, as on a chip: Steiner points can cut length by 1/3")
    fig.tight_layout()
    sty.save(fig, FIGURES / "11_rectilinear.png")


def fig12_fermat_point():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    # (a) all angles < 120 degrees
    ax = axes[0]
    T = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 3.0]])
    f = np.array(fermat_point(*T))
    ax.fill(T[:, 0], T[:, 1], color="#f3f4f6", zorder=0)
    ax.plot(*np.vstack([T, T[:1]]).T, color=sty.MST, lw=1.2, ls=":", zorder=1)
    Q = np.vstack([T, f])
    sty.draw_edges(ax, Q, [(0, 3), (1, 3), (2, 3)], kind="smt", label="shortest network")
    sty.draw_terminals(ax, T, label="triangle corner")
    sty.draw_steiner_points(ax, f, label="Fermat point")
    arcs_at(ax, f, T, r=0.3, fontsize=10, text_r=0.58)
    ax.annotate("Fermat point: sees every\npair of corners at 120°", xy=f, xytext=(2.1, -0.25), fontsize=10,
                va="top",
                color=sty.STEINER, arrowprops=dict(arrowstyle="->", color=sty.STEINER, shrinkB=7))
    angles = [angle_at(T[k], T[(k + 1) % 3], T[(k + 2) % 3]) for k in range(3)]
    total = sum(dist(f, p) for p in T)
    ax.set_title(f"All angles < 120° (largest {max(angles):.0f}°)\n"
                 f"Fermat point inside: total = {total:.3f}")
    # (b) one angle >= 120 degrees
    ax = axes[1]
    T2 = np.array([[0.0, 0.0], [4.0, 0.0], [1.2, 0.6]])
    f2 = np.array(fermat_point(*T2))
    big = angle_at(T2[2], T2[0], T2[1])
    ax.fill(T2[:, 0], T2[:, 1], color="#f3f4f6", zorder=0)
    ax.plot(*np.vstack([T2, T2[:1]]).T, color=sty.MST, lw=1.2, ls=":", zorder=1)
    sty.draw_edges(ax, T2, [(0, 2), (1, 2)], kind="smt", label="shortest network")
    sty.draw_terminals(ax, T2, label="triangle corner")
    sty.draw_steiner_points(ax, f2, size=200, label="Fermat point (= the obtuse corner)")
    th0, th1 = (math.degrees(math.atan2(*(q - T2[2])[::-1])) for q in (T2[1], T2[0]))
    sty.angle_arc(ax, T2[2], th0, th1, r=0.3, color="#c0392b")
    ax.annotate(f"angle {big:.0f}° ≥ 120°", xy=(T2[2][0], T2[2][1] - 0.3), xytext=(T2[2][0], -0.45),
                ha="center", va="top", color="#c0392b", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#c0392b"))
    ax.annotate("Fermat point is this corner:\nno Steiner point helps,\nthe two short sides are best",
                xy=T2[2], xytext=(1.9, 2.2), fontsize=10, color=sty.STEINER,
                arrowprops=dict(arrowstyle="->", color=sty.STEINER, shrinkB=9))
    total2 = dist(T2[2], T2[0]) + dist(T2[2], T2[1])
    ax.set_title(f"One angle ≥ 120° ({big:.0f}°)\nFermat point = that corner: total = {total2:.3f}")
    for ax in axes:
        sty.equal_axes(ax, np.vstack([T, [[0, -0.3]]]), pad=0.08, units=UNITS)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=9)
    fig.suptitle("The Fermat point: where to put one junction joining three points")
    fig.tight_layout()
    sty.save(fig, FIGURES / "12_fermat_point.png")


def main():
    sty.setup()
    FIGURES.mkdir(exist_ok=True)
    for fn in (fig01_triangle, fig02_square, fig03_120_rule, fig04_why_120, fig05_soap_film,
               fig06_topology_growth, fig07_random_overlay, fig08_ratio_histogram, fig09_ratio_vs_n,
               fig10_conjecture_test, fig11_rectilinear, fig12_fermat_point):
        fn()


if __name__ == "__main__":
    main()
