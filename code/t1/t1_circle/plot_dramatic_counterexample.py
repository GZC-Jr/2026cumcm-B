"""Compare the existing subtle hexagon with a dramatic counterexample.

Two side-by-side panels:

  Left  - the existing 6-vertex hexagon from 3 detection points:
          r*/(D/2) ≈ 1.010  (subtle failure, only ~0.2 m gap).

  Right - an equilateral triangle (constructed counterexample):
          r*/(D/2) = 2/√3 ≈ 1.155  (Jung bound tight, dramatic failure).

Why the triangle is the worst case (Jung's theorem, tight):

  When a polygon contains an equilateral triangle of side D, the diameter
  D is achieved by one of the triangle's edges.  The minimum cover circle
  is the circumscribed circle of that triangle, with radius D/√3.

  The apex sits at perpendicular distance D·√3/2 from the diameter edge,
  while the diameter circle reaches only D/2.  The apex is uncovered by
  D·(√3/2 − 1/2) ≈ 0.366 D — the largest possible gap.

Run::

    python scratch/t1_circle/plot_dramatic_counterexample.py

Output:  scratch/t1_circle/dramatic_counterexamples.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------
def min_enclosing_circle(pts: np.ndarray) -> tuple[np.ndarray, float, list[int]]:
    """Brute-force minimum enclosing circle for small m (<= ~20).

    Returns (center, radius, support_indices).  Tries all vertex pairs
    and all non-collinear triples; picks the smallest valid candidate.
    """
    m = len(pts)
    best_c = pts[0].copy()
    best_r = np.inf
    best_supports: list[int] = [0]

    def consider(c: np.ndarray, r: float, supports: list[int]) -> None:
        nonlocal best_c, best_r, best_supports
        # All points must lie inside or on the circle.
        if np.all(np.linalg.norm(pts - c, axis=1) <= r + 1e-9) and r < best_r - 1e-12:
            best_c = c
            best_r = r
            best_supports = list(supports)

    for i in range(m):
        for j in range(i + 1, m):
            c = 0.5 * (pts[i] + pts[j])
            r = 0.5 * np.linalg.norm(pts[i] - pts[j])
            consider(c, r, [i, j])

    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                # Circumcenter of the triangle.
                a = pts[j] - pts[i]
                b = pts[k] - pts[i]
                det = a[0] * b[1] - a[1] * b[0]
                if abs(det) < 1e-12:
                    continue
                M = np.array([2 * a, 2 * b])
                rhs = np.array(
                    [
                        np.dot(pts[j], pts[j]) - np.dot(pts[i], pts[i]),
                        np.dot(pts[k], pts[k]) - np.dot(pts[i], pts[i]),
                    ]
                )
                c = np.linalg.solve(M, rhs)
                r = float(np.linalg.norm(pts[i] - c))
                consider(c, r, [i, j, k])

    return best_c, best_r, best_supports


def farthest_pair(pts: np.ndarray) -> tuple[int, int, float]:
    m = len(pts)
    best_d = -1.0
    best_i, best_j = 0, 0
    for i in range(m):
        diffs = pts - pts[i]
        dists = np.einsum("ij,ij->i", diffs, diffs)
        j = int(np.argmax(dists))
        d = float(np.sqrt(dists[j]))
        if d > best_d:
            best_d = d
            best_i, best_j = i, j
    return best_i, best_j, best_d


def order_ccw(pts: np.ndarray) -> np.ndarray:
    """Return points sorted counterclockwise around their centroid."""
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)]


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
def make_equilateral_triangle(D: float) -> np.ndarray:
    """Equilateral triangle with one edge on the x-axis (length D)."""
    return np.array(
        [
            [-D / 2, 0.0],
            [D / 2, 0.0],
            [0.0, D * np.sqrt(3.0) / 2.0],
        ]
    )


def make_extreme_hexagon(D: float) -> np.ndarray:
    """Six-vertex convex polygon that attains Jung's bound.

    Three vertices form an equilateral triangle with one side of length D.
    Three additional vertices sit just inside the triangle's edges so that
    they remain convex hull vertices (small outward bumps), but small
    enough not to enlarge D or r_*.
    """
    tri = make_equilateral_triangle(D)

    # Mid-edge points (slightly outside the triangle, by ``epsilon``).
    epsilon = 0.6
    m01 = 0.5 * (tri[0] + tri[1])
    m12 = 0.5 * (tri[1] + tri[2])
    m20 = 0.5 * (tri[2] + tri[0])
    # Push outward by epsilon along the perpendicular direction.
    def outward(p, a, b):
        edge = b - a
        normal = np.array([-edge[1], edge[0]])
        normal /= np.linalg.norm(normal)
        # Make sure the bump keeps the polygon convex.
        return p + epsilon * normal

    bump01 = outward(m01, tri[0], tri[1])  # along base, push downward
    bump12 = outward(m12, tri[1], tri[2])  # along right edge, push rightward
    bump20 = outward(m20, tri[2], tri[0])  # along left edge, push leftward

    pts = np.vstack([tri, bump01, bump12, bump20])
    return order_ccw(pts)


def load_original_hexagon() -> np.ndarray:
    json_path = WORKSPACE / "outputs" / "t1_demo_result.json"
    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return np.array(data["vertices"]), data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_case(
    ax: plt.Axes,
    pts: np.ndarray,
    title: str,
    show_labels: bool = True,
    label_offset: tuple[int, int] = (6, 6),
) -> dict:
    m = len(pts)
    p_idx, q_idx, D = farthest_pair(pts)

    p_star = pts[p_idx]
    q_star = pts[q_idx]
    c_D = 0.5 * (p_star + q_star)
    r_D = D / 2.0

    c_star, r_star, supports = min_enclosing_circle(pts)

    poly = np.vstack([pts, pts[:1]])
    ax.add_patch(
        Polygon(poly, closed=True, fill=False, edgecolor="black", linewidth=1.8)
    )

    ax.add_patch(
        Circle(
            c_D,
            radius=r_D,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=1.8,
            linestyle="--",
        )
    )

    ax.add_patch(
        Circle(
            c_star,
            radius=r_star,
            facecolor="#1f77b4",
            alpha=0.20,
            edgecolor="#1f77b4",
            linewidth=2.2,
        )
    )

    ax.plot(
        [p_star[0], q_star[0]],
        [p_star[1], q_star[1]],
        color="#ff7f0e",
        linewidth=2.2,
        linestyle=":",
    )

    support_set = set(supports)
    for idx, v in enumerate(pts):
        is_support = idx in support_set
        ax.scatter(
            v[0],
            v[1],
            s=80 if is_support else 40,
            color="#2ca02c" if is_support else "#444444",
            zorder=5,
        )
        if show_labels:
            ax.annotate(
                f"$v_{idx}$",
                xy=(v[0], v[1]),
                xytext=label_offset,
                textcoords="offset points",
                fontsize=11,
                fontweight="bold" if is_support else "normal",
            )

    ax.scatter(
        c_D[0],
        c_D[1],
        marker="x",
        color="#d62728",
        s=120,
        linewidths=2.0,
        zorder=6,
    )
    ax.scatter(
        c_star[0],
        c_star[1],
        marker="+",
        color="#1f77b4",
        s=220,
        linewidths=2.4,
        zorder=6,
    )

    # Custom legend with the metric numbers.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0], [0],
            color="black",
            linewidth=1.8,
            label="Polygon boundary",
        ),
        Line2D(
            [0], [0],
            color="#ff7f0e",
            linewidth=2.2,
            linestyle=":",
            label=f"Diameter  $D = {D:.3f}$",
        ),
        Line2D(
            [0], [0],
            color="#d62728",
            linewidth=1.8,
            linestyle="--",
            label=f"Diameter circle  $r = D/2 = {r_D:.3f}$  (fails)",
        ),
        Line2D(
            [0], [0],
            color="#1f77b4",
            linewidth=2.2,
            label=f"Min cover circle  $r_* = {r_star:.3f}$",
        ),
        Line2D(
            [0], [0],
            marker="+",
            color="#1f77b4",
            markersize=10,
            linewidth=0,
            label=f"$c_* = ({c_star[0]:.3f},\\,{c_star[1]:.3f})$",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)

    ratio = r_star / r_D
    # Inside title we report ratio and the visual gap.
    gap = r_star - r_D
    ax.set_title(
        f"{title}\n"
        f"$r_*/(D/2) = {ratio:.4f}$   "
        f"(gap $= r_* - D/2 = {gap:.3f}$)   "
        f"supports = {supports}",
        fontsize=10,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")

    return {
        "D": D,
        "r_D": r_D,
        "r_star": r_star,
        "ratio": ratio,
        "supports": supports,
    }


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    D = 40.0

    # --- (a) Original 6-vertex hexagon: subtle failure ---
    hex_verts, _ = load_original_hexagon()
    res_orig = plot_case(
        axes[0],
        hex_verts,
        "(a) Existing 6-vertex hexagon  (subtle)",
    )

    # --- (b) Equilateral triangle: worst case (Jung tight) ---
    tri = make_equilateral_triangle(D)
    res_tri = plot_case(
        axes[1],
        tri,
        "(b) Equilateral triangle  (worst case, Jung tight)",
    )

    # --- (c) Constructed 6-vertex hexagon hitting the Jung bound ---
    extreme_hex = make_extreme_hexagon(D)
    res_ext = plot_case(
        axes[2],
        extreme_hex,
        "(c) 6-vertex polygon with $r_* = D/\\sqrt{3}$",
    )

    fig.suptitle(
        "Counterexample gallery: the diameter circle can fail to cover $P$",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = HERE / "dramatic_counterexamples.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_path}")
    print()
    print(f"(a) Hexagon:        D={res_orig['D']:.3f},  D/2={res_orig['r_D']:.3f},  "
          f"r*={res_orig['r_star']:.3f},  ratio={res_orig['ratio']:.4f}")
    print(f"(b) Triangle:       D={res_tri['D']:.3f},  D/2={res_tri['r_D']:.3f},  "
          f"r*={res_tri['r_star']:.3f},  ratio={res_tri['ratio']:.4f}  (Jung bound 2/√3={2/np.sqrt(3):.4f})")
    print(f"(c) Extreme hex:    D={res_ext['D']:.3f},  D/2={res_ext['r_D']:.3f},  "
          f"r*={res_ext['r_star']:.3f},  ratio={res_ext['ratio']:.4f}")


if __name__ == "__main__":
    main()
