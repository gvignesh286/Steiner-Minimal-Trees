"""Shared plot style for every figure.

* terminals: dark filled dots
* Steiner points: hollow orange circles
* MST: thick gray lines
* Steiner tree: orange lines
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TERMINAL = "#1f2933"
STEINER = "#e8730c"
MST = "#9aa0a6"
ACCENT = "#2563eb"
MUTED = "#6b7280"
DPI = 200


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    })


def draw_terminals(ax, pts, size: float = 60, label: str | None = "terminal (given point)", zorder=5):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, color=TERMINAL, zorder=zorder, label=label)


def draw_steiner_points(ax, pts, size: float = 80, label: str | None = "Steiner point (added)", zorder=6):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=size, facecolor="white", edgecolor=STEINER,
                   linewidth=2.2, zorder=zorder, label=label)


def draw_edges(ax, points, edges, kind: str = "mst", label: str | None = None, **kw):
    """kind = 'mst' (thick gray) or 'smt' (orange)."""
    style = dict(color=MST, lw=5, alpha=0.75, solid_capstyle="round", zorder=2) if kind == "mst" \
        else dict(color=STEINER, lw=2.6, solid_capstyle="round", zorder=3)
    style.update(kw)
    for k, (i, j) in enumerate(edges):
        ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]],
                label=label if k == 0 else None, **style)


def draw_tree(ax, tree, kind: str = "smt", label: str | None = None, steiner_label="Steiner point (added)",
              terminal_label="terminal (given point)"):
    draw_edges(ax, tree.points, tree.edges, kind=kind, label=label)
    draw_terminals(ax, tree.points[: tree.n_terminals], label=terminal_label)
    draw_steiner_points(ax, tree.points[tree.steiner], label=steiner_label)


def equal_axes(ax, pts, pad: float = 0.15, units: str = "length units"):
    pts = np.asarray(pts, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = max(hi - lo) or 1.0
    c = (lo + hi) / 2
    half = span / 2 + pad * span
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")


def label_edge(ax, p, q, text, offset=0.06, color=MUTED, fontsize=10, side=1):
    """Write ``text`` next to the midpoint of segment pq, offset perpendicular to it."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    d = q - p
    nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1.0)
    pos = m + side * offset * nrm
    ax.text(pos[0], pos[1], text, ha="center", va="center", fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=8)


def angle_arc(ax, center, a1_deg, a2_deg, r=0.12, text=None, color=ACCENT, text_r=None, fontsize=10,
              inset=0.0):
    """Arc from angle a1 to a2 (degrees, counter-clockwise) around ``center``.

    ``inset`` trims that many degrees off each end so adjacent arcs stay distinct.
    """
    t = np.radians(np.linspace(a1_deg + inset, a2_deg - inset, 60))
    ax.plot(center[0] + r * np.cos(t), center[1] + r * np.sin(t), color=color, lw=1.4, zorder=7)
    if text:
        mid = np.radians((a1_deg + a2_deg) / 2)
        tr = text_r if text_r is not None else r * 1.75
        ax.text(center[0] + tr * np.cos(mid), center[1] + tr * np.sin(mid), text, color=color,
                ha="center", va="center", fontsize=fontsize, zorder=8)


def save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
