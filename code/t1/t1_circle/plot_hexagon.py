"""Plot the hexagon counterexample and its minimum covering circle.

Produces a visualization that shows:
  - the six vertices of the localization hexagon (counterexample),
  - the convex hexagon outline,
  - the minimum covering circle (centered near the origin),
  - the diameter circle (centered at the midpoint of the farthest pair)
    for reference, even though it fails to cover the hexagon.

Run from the workspace root::

    python scratch/t1_circle/plot_hexagon.py

Outputs are written to ``scratch/t1_circle/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RESULT_JSON = WORKSPACE / "outputs" / "t1_demo_result.json"
VERTICES_CSV = WORKSPACE / "outputs" / "t1_demo_vertices.csv"


def load_data() -> dict:
    """Load the result JSON and CSV, returning a merged dict."""
    with RESULT_JSON.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    import csv

    vertices: list[tuple[float, float]] = []
    with VERTICES_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vertices.append((float(row["x"]), float(row["y"])))
    result["vertices"] = vertices
    return result


def plot_hexagon(data: dict, out_path: Path) -> None:
    vertices = np.asarray(data["vertices"])
    mcc = data["minimum_cover_circle"]
    dcc = data["diameter_circle"]
    diameter = data["diameter"]
    p_star, q_star = data["diameter_endpoints"]["p"], data["diameter_endpoints"]["q"]

    # Close the polygon by appending the first vertex.
    polygon = np.vstack([vertices, vertices[:1]])

    fig, ax = plt.subplots(figsize=(8, 8))

    # Minimum covering circle (the answer).
    mcc_circle = Circle(
        (mcc["center"][0], mcc["center"][1]),
        radius=mcc["radius"],
        facecolor="#1f77b4",
        alpha=0.18,
        edgecolor="#1f77b4",
        linewidth=2.0,
        linestyle="-",
        label=f"Minimum cover circle  $r_* \\approx {mcc['radius']:.3f}$",
    )
    ax.add_patch(mcc_circle)

    # Diameter circle (which fails to cover).
    dcc_circle = Circle(
        (dcc["center"][0], dcc["center"][1]),
        radius=dcc["radius"],
        facecolor="none",
        edgecolor="#d62728",
        linewidth=1.8,
        linestyle="--",
        label=(
            f"Diameter circle  $D/2 \\approx {dcc['radius']:.3f}$  (does NOT cover)"
        ),
    )
    ax.add_patch(dcc_circle)

    # Convex hexagon outline.
    ax.plot(
        polygon[:, 0],
        polygon[:, 1],
        color="black",
        linewidth=1.6,
        label="Localization hexagon",
    )

    # Vertices: fill and label.
    support = set(mcc["support_indices"])
    for idx, (x, y) in enumerate(vertices):
        is_support = idx in support
        ax.scatter(
            x,
            y,
            s=70 if is_support else 40,
            color="#2ca02c" if is_support else "#444444",
            zorder=5,
        )
        ax.annotate(
            f"$v_{idx}$",
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold" if is_support else "normal",
        )

    # Mark the centers of both circles.
    ax.scatter(
        mcc["center"][0],
        mcc["center"][1],
        marker="+",
        color="#1f77b4",
        s=200,
        linewidths=2.2,
        label=f"$c_* = (${mcc['center'][0]:.3f}, {mcc['center'][1]:.3f}$)$",
    )
    ax.scatter(
        dcc["center"][0],
        dcc["center"][1],
        marker="x",
        color="#d62728",
        s=120,
        linewidths=2.0,
        label=f"$c_D = (${dcc['center'][0]:.3f}, {dcc['center'][1]:.3f}$)$",
    )

    # Diameter segment.
    ax.plot(
        [p_star[0], q_star[0]],
        [p_star[1], q_star[1]],
        color="#ff7f0e",
        linewidth=2.0,
        linestyle=":",
        label=f"Diameter  $D \\approx {diameter:.3f}$",
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title(
        "Counterexample: hexagon whose diameter circle fails to cover it\n"
        "Six vertices, three supports of the minimum cover circle ($v_1, v_3, v_5$)"
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Pad the view so labels do not collide with the axes.
    margin = 4.0
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = load_data()
    out_path = HERE / "hexagon_with_min_cover.png"
    plot_hexagon(data, out_path)

    print(f"Wrote figure: {out_path}")
    print(f"  Diameter        D = {data['diameter']:.6f}")
    print(f"  Diameter circle : center={data['diameter_circle']['center']}, "
          f"r={data['diameter_circle']['radius']:.6f}, "
          f"covers={data['diameter_circle']['covers']}")
    print(f"  Min cover circle: center={data['minimum_cover_circle']['center']}, "
          f"r={data['minimum_cover_circle']['radius']:.6f}, "
          f"supports={data['minimum_cover_circle']['support_indices']}")


if __name__ == "__main__":
    main()
