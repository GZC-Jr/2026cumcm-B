"""Reproducible convergence, search, boundary and Monte-Carlo experiments.

The production solver in :mod:`t2.solve_q2` intentionally keeps its public
output compact.  This module is the experiment layer for the paper: it
re-optimises every numerical setting, compares local/global search over
multiple seeds, extracts continuous near-optimal boundaries with marching
squares, and evaluates two symmetric optimum branches by Monte Carlo.

Run from the workspace root::

    python code/t2/enhanced_experiments.py --demo --output-dir outputs/t2/enhanced

``--quick`` uses a small but structurally identical design for CI/smoke checks.
The default design is intended for the final paper tables and can take a few
minutes on a laptop because each convergence row is re-optimised.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from t2.solve_q2 import Q2Config, Question2Model, load_input, parse_input  # noqa: E402


try:  # ``scikit-image`` is a small optional dependency for the contour step.
    from skimage.measure import find_contours
except ImportError:  # pragma: no cover - exercised only on minimal installs
    find_contours = None


DEFAULT_CIRCLE_VALUES = (32, 64, 128, 180, 360)
DEFAULT_QUADRATURE_VALUES = (3, 4, 5, 6, 8)
DEFAULT_GRID_VALUES = (31, 51, 81, 121, 161)
DEFAULT_SEEDS = (0, 1, 2, 20260911)
QUICK_CIRCLE_VALUES = (32, 64, 128)
QUICK_QUADRATURE_VALUES = (3, 4, 5)
QUICK_GRID_VALUES = (31, 51, 81)
QUICK_SEEDS = (0, 1)


def _clone_config(config: Q2Config, **updates: Any) -> Q2Config:
    values = asdict(config)
    values.update(updates)
    result = Q2Config(**values)
    result.validate()
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return number


def _parse_int_list(raw: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(value) for value in default)
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("integer list must contain at least one value")
    return values


def _branch_label(point: Sequence[float], s1: np.ndarray, bearing_deg: float) -> str:
    direction = np.asarray(
        (math.cos(math.radians(bearing_deg)), math.sin(math.radians(bearing_deg))),
        dtype=float,
    )
    signed = float(direction[0] * (float(point[1]) - s1[1]) - direction[1] * (float(point[0]) - s1[0]))
    if abs(signed) <= 1e-7:
        return "axis"
    return "positive_side" if signed > 0 else "negative_side"


def reflect_across_bearing(
    point: Sequence[float], s1: Sequence[float], bearing_deg: float
) -> np.ndarray:
    """Reflect a detector position in the measured-bearing axis."""

    origin = np.asarray(s1, dtype=float)
    value = np.asarray(point, dtype=float)
    direction = np.asarray(
        (math.cos(math.radians(bearing_deg)), math.sin(math.radians(bearing_deg))),
        dtype=float,
    )
    relative = value - origin
    return origin + 2.0 * float(relative @ direction) * direction - relative


def _optimise(
    s1: np.ndarray,
    bearing_deg: float,
    config: Q2Config,
    *,
    seed: int,
    use_global_search: bool,
    max_local_starts: int,
) -> tuple[Question2Model, dict[str, Any], float]:
    local_config = _clone_config(
        config,
        seed=int(seed),
        use_global_search=bool(use_global_search),
    )
    model = Question2Model(s1, bearing_deg, local_config)
    started = time.perf_counter()
    optimum = model.optimize(seed=int(seed), max_local_starts=max_local_starts)
    elapsed = time.perf_counter() - started
    return model, optimum, float(elapsed)


def _row_from_optimum(
    model: Question2Model,
    optimum: dict[str, Any],
    *,
    seed: int,
    use_global_search: bool,
    elapsed_s: float,
    experiment: str,
    numerical_value: int | None = None,
) -> dict[str, Any]:
    point = np.asarray(optimum["point"], dtype=float)
    row: dict[str, Any] = {
        "experiment": experiment,
        "setting": numerical_value,
        "seed": int(seed),
        "use_global_search": bool(use_global_search),
        "circle_sides": int(model.config.circle_sides),
        "quadrature_order": int(model.quadrature_order),
        "candidate_grid": int(model.config.candidate_grid),
        "region_area": float(model.region.area),
        "inradius": float(model.region.d_star),
        "point_x": float(point[0]),
        "point_y": float(point[1]),
        "j_star": float(optimum["j_star"]),
        "distance_from_s1": float(np.linalg.norm(point - model.s1)),
        "reception_h": float(optimum["reception_h"]),
        "h_tolerance": float(optimum["reception_h_tolerance"]),
        "strict_feasible": bool(optimum["strict_feasible"]),
        "feasible_with_tolerance": bool(optimum["feasible_with_tolerance"]),
        "source": str(optimum["source"]),
        "branch": _branch_label(point, model.s1, model.bearing_deg),
        "local_run_count": int(optimum["local_run_count"]),
        "local_success_count": int(sum(bool(run.get("success")) for run in optimum.get("local_runs", []))),
        "local_feasible_count": int(sum(bool(run.get("feasible")) for run in optimum.get("local_runs", []))),
        "selected_iterations": optimum.get("iterations"),
        "selected_function_evaluations": optimum.get("function_evaluations"),
        "elapsed_s": float(elapsed_s),
    }
    if optimum.get("global") is not None:
        global_result = optimum["global"]
        row.update(
            {
                "global_objective": _finite_float(global_result.get("objective")),
                "global_h": _finite_float(global_result.get("reception_h")),
                "global_feasible": bool(global_result.get("feasible_with_tolerance", False)),
                "global_iterations": int(global_result.get("iterations", 0)),
                "global_function_evaluations": int(global_result.get("function_evaluations", 0)),
                "global_branch": _branch_label(global_result.get("point", point), model.s1, model.bearing_deg),
            }
        )
    else:
        row.update(
            {
                "global_objective": None,
                "global_h": None,
                "global_feasible": None,
                "global_iterations": None,
                "global_function_evaluations": None,
                "global_branch": None,
            }
        )
    return row


def run_convergence(
    s1: np.ndarray,
    bearing_deg: float,
    base_config: Q2Config,
    *,
    circle_values: Sequence[int],
    quadrature_values: Sequence[int],
    grid_values: Sequence[int],
    tau: float,
    seed: int,
    max_local_starts: int,
) -> dict[str, Any]:
    """Re-optimise every circle/quadrature setting and classify each grid."""

    circle_rows: list[dict[str, Any]] = []
    for sides in circle_values:
        config = _clone_config(base_config, circle_sides=int(sides), use_global_search=False)
        model, optimum, elapsed = _optimise(
            s1,
            bearing_deg,
            config,
            seed=seed,
            use_global_search=False,
            max_local_starts=max_local_starts,
        )
        point = np.asarray(optimum["point"], dtype=float)
        high_order = max(int(max(quadrature_values)), int(config.validation_quadrature_order))
        row = _row_from_optimum(
            model,
            optimum,
            seed=seed,
            use_global_search=False,
            elapsed_s=elapsed,
            experiment="circle_sides",
            numerical_value=int(sides),
        )
        row.update(
            {
                "native_j_star": float(optimum["j_star"]),
                "reference_order": int(high_order),
                "j_at_reference_order": float(model.objective(point, quadrature_order=high_order)),
            }
        )
        circle_rows.append(row)
    circle_reference = (
        max(circle_rows, key=lambda row: int(row["setting"])) if circle_rows else None
    )
    for row in circle_rows:
        row["delta_j_native_vs_last"] = None if circle_reference is None else float(row["native_j_star"] - circle_reference["native_j_star"])
        row["relative_j_native_vs_last"] = None if circle_reference is None else float(abs(row["delta_j_native_vs_last"]) / max(1.0, abs(circle_reference["native_j_star"])))

    final_sides = int(max(circle_values))
    quadrature_rows: list[dict[str, Any]] = []
    reference_order = int(max(quadrature_values))
    for order in quadrature_values:
        config = _clone_config(
            base_config,
            circle_sides=final_sides,
            quadrature_order=int(order),
            validation_quadrature_order=reference_order,
            use_global_search=False,
        )
        model, optimum, elapsed = _optimise(
            s1,
            bearing_deg,
            config,
            seed=seed,
            use_global_search=False,
            max_local_starts=max_local_starts,
        )
        point = np.asarray(optimum["point"], dtype=float)
        row = _row_from_optimum(
            model,
            optimum,
            seed=seed,
            use_global_search=False,
            elapsed_s=elapsed,
            experiment="quadrature_order",
            numerical_value=int(order),
        )
        row.update(
            {
                "native_j_star": float(optimum["j_star"]),
                "reference_order": int(reference_order),
                "j_at_reference_order": float(model.objective(point, quadrature_order=reference_order)),
            }
        )
        quadrature_rows.append(row)
    quadrature_reference = (
        max(quadrature_rows, key=lambda row: int(row["setting"])) if quadrature_rows else None
    )
    for row in quadrature_rows:
        row["delta_j_native_vs_last"] = None if quadrature_reference is None else float(row["native_j_star"] - quadrature_reference["native_j_star"])
        row["relative_j_native_vs_last"] = None if quadrature_reference is None else float(abs(row["delta_j_native_vs_last"]) / max(1.0, abs(quadrature_reference["native_j_star"])))

    final_config = _clone_config(
        base_config,
        circle_sides=final_sides,
        quadrature_order=reference_order,
        validation_quadrature_order=reference_order,
        use_global_search=False,
    )
    final_model, final_optimum, final_elapsed = _optimise(
        s1,
        bearing_deg,
        final_config,
        seed=seed,
        use_global_search=False,
        max_local_starts=max_local_starts,
    )
    grid_rows: list[dict[str, Any]] = []
    contour_records: list[dict[str, Any]] = []
    for grid_size in grid_values:
        level = marching_square_level_set(final_model, final_optimum, tau=tau, grid_size=int(grid_size))
        row = {
            "experiment": "candidate_grid",
            "grid_size": int(grid_size),
            "tau": float(tau),
            "threshold": float(level["threshold"]),
            "cell_center_area": float(level["cell_center_area"]),
            "marching_area": float(level["marching_area"]),
            "accepted_cell_count": int(level["accepted_cell_count"]),
            "contour_count": int(level["contour_count"]),
            "closed_contour_count": int(level["closed_contour_count"]),
            "open_contour_count": int(level["open_contour_count"]),
            "boundary_length": float(level["boundary_length"]),
            "finite_node_count": int(level["finite_node_count"]),
            "reception_feasible_node_count": int(level["reception_feasible_node_count"]),
        }
        grid_rows.append(row)
        for contour_id, contour in enumerate(level["contours"]):
            for vertex_id, point in enumerate(contour):
                contour_records.append(
                    {
                        "surface": "objective_level",
                        "tau": float(tau),
                        "grid_size": int(grid_size),
                        "contour_id": int(contour_id),
                        "vertex_id": int(vertex_id),
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                )
    return {
        "circle_sides": circle_rows,
        "quadrature_order": quadrature_rows,
        "candidate_grid": grid_rows,
        "final_model": final_model,
        "final_optimum": final_optimum,
        "final_elapsed_s": final_elapsed,
        "boundary_records": contour_records,
    }


def _extract_contours(field: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> list[np.ndarray]:
    """Extract zero contours and convert array indices to world coordinates."""

    if find_contours is None:
        raise RuntimeError(
            "marching-squares extraction requires scikit-image; install code/t2/requirements.txt"
        )
    contours = find_contours(field, level=0.0, fully_connected="high")
    result: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        rows = np.clip(contour[:, 0], 0.0, len(y_values) - 1.0)
        columns = np.clip(contour[:, 1], 0.0, len(x_values) - 1.0)
        points = np.column_stack(
            (
                np.interp(columns, np.arange(len(x_values)), x_values),
                np.interp(rows, np.arange(len(y_values)), y_values),
            )
        )
        result.append(points)
    return result


def _polygon_area(path: np.ndarray, close_tolerance: float) -> tuple[float, bool]:
    if len(path) < 3:
        return 0.0, False
    closed = bool(np.linalg.norm(path[0] - path[-1]) <= close_tolerance)
    if not closed:
        return 0.0, False
    vertices = path[:-1] if len(path) > 1 else path
    next_vertices = np.roll(vertices, -1, axis=0)
    area = 0.5 * abs(
        float(
            np.sum(vertices[:, 0] * next_vertices[:, 1] - next_vertices[:, 0] * vertices[:, 1])
        )
    )
    return area, True


def _downsample_path(path: np.ndarray, max_points: int = 300) -> np.ndarray:
    if len(path) <= max_points:
        return path
    indices = np.linspace(0, len(path) - 1, max_points).round().astype(int)
    return path[np.unique(indices)]


def evaluate_surface(
    model: Question2Model,
    optimum: dict[str, Any],
    *,
    grid_size: int,
    tau: float,
) -> dict[str, Any]:
    """Evaluate J, H and candidate masks on a square node grid."""

    count = int(grid_size)
    if count < 11:
        raise ValueError("surface grid must contain at least 11 nodes per axis")
    x_values = np.linspace(model.search_bounds[0][0], model.search_bounds[0][1], count)
    y_values = np.linspace(model.search_bounds[1][0], model.search_bounds[1][1], count)
    xx, yy = np.meshgrid(x_values, y_values, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    h_values = model.receive_h_many(points).reshape(count, count)
    j_values = model.objective_many(points).reshape(count, count)
    h_tolerance = model.reception_h_tolerance()
    threshold = (1.0 + float(tau)) * float(optimum["j_star"])
    finite = np.isfinite(j_values)
    reception_feasible = np.isfinite(h_values) & (h_values <= h_tolerance)
    candidate = finite & reception_feasible & (j_values <= threshold)
    return {
        "x_values": x_values,
        "y_values": y_values,
        "xx": xx,
        "yy": yy,
        "h_values": h_values,
        "j_values": j_values,
        "finite": finite,
        "reception_feasible": reception_feasible,
        "candidate": candidate,
        "threshold": float(threshold),
        "h_tolerance": float(h_tolerance),
    }


def marching_square_level_set(
    model: Question2Model,
    optimum: dict[str, Any],
    *,
    tau: float,
    grid_size: int,
) -> dict[str, Any]:
    """Extract the continuous objective boundary and grid diagnostics."""

    surface = evaluate_surface(model, optimum, grid_size=grid_size, tau=tau)
    x_values = surface["x_values"]
    y_values = surface["y_values"]
    j_values = surface["j_values"]
    finite = surface["finite"]
    threshold = float(surface["threshold"])
    # Infeasible/singular nodes are assigned a positive level-set value.  The
    # resulting zero contour is the objective boundary clipped by the search
    # domain and reception-feasible mask; the H=0 boundary is exported
    # separately by ``plot_geometry_overlay``.
    valid_level = finite & surface["reception_feasible"]
    field = np.where(valid_level, j_values - threshold, max(1.0, abs(threshold)))
    contours = _extract_contours(field, x_values, y_values)
    dx = float(x_values[1] - x_values[0])
    dy = float(y_values[1] - y_values[0])
    close_tolerance = 1.5 * max(abs(dx), abs(dy))
    areas: list[float] = []
    closed_count = 0
    boundary_length = 0.0
    for path in contours:
        area, closed = _polygon_area(path, close_tolerance)
        areas.append(area)
        closed_count += int(closed)
        if len(path) > 1:
            boundary_length += float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    center_x = (x_values[:-1] + x_values[1:]) / 2.0
    center_y = (y_values[:-1] + y_values[1:]) / 2.0
    center_xx, center_yy = np.meshgrid(center_x, center_y, indexing="xy")
    center_points = np.column_stack((center_xx.ravel(), center_yy.ravel()))
    center_h = model.receive_h_many(center_points).reshape(count := len(center_y), len(center_x))
    center_j = model.objective_many(center_points).reshape(count, len(center_x))
    center_candidate = (
        np.isfinite(center_j)
        & np.isfinite(center_h)
        & (center_h <= surface["h_tolerance"])
        & (center_j <= threshold)
    )
    cell_center_area = float(np.sum(center_candidate) * abs(dx * dy))
    # Closed objective contours provide a true continuous area estimate.  If a
    # contour is clipped/open, retain the raster estimate and report the fact
    # explicitly instead of claiming a closed polygon area.
    marching_area = float(sum(areas)) if contours and closed_count == len(contours) else cell_center_area
    return {
        "tau": float(tau),
        "threshold": threshold,
        "grid_size": int(grid_size),
        "contours": [_downsample_path(path).astype(float).tolist() for path in contours],
        "contour_count": int(len(contours)),
        "closed_contour_count": int(closed_count),
        "open_contour_count": int(len(contours) - closed_count),
        "marching_area": marching_area,
        "cell_center_area": cell_center_area,
        "accepted_cell_count": int(np.sum(center_candidate)),
        "boundary_length": float(boundary_length),
        "finite_node_count": int(np.sum(surface["finite"])),
        "reception_feasible_node_count": int(np.sum(surface["reception_feasible"])),
        "x_min": float(x_values[0]),
        "x_max": float(x_values[-1]),
        "y_min": float(y_values[0]),
        "y_max": float(y_values[-1]),
        "x_step": dx,
        "y_step": dy,
        "surface": surface,
    }


def _find_opposite_branch(
    model: Question2Model,
    optimum: dict[str, Any],
) -> tuple[np.ndarray, str]:
    primary = np.asarray(optimum["point"], dtype=float)
    reflected = reflect_across_bearing(primary, model.s1, model.bearing_deg)
    if (
        model.geometry_is_finite(reflected)
        and model.receive_h(reflected) <= model.reception_h_tolerance()
        and math.isfinite(model.objective(reflected))
    ):
        return reflected, "geometric_reflection"
    candidates: list[tuple[float, np.ndarray]] = []
    primary_branch = _branch_label(primary, model.s1, model.bearing_deg)
    for run in optimum.get("local_runs", []):
        point = np.asarray(run.get("point", []), dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            continue
        if _branch_label(point, model.s1, model.bearing_deg) == primary_branch:
            continue
        value = float(run.get("objective", math.inf))
        if math.isfinite(value) and model.receive_h(point) <= model.reception_h_tolerance():
            candidates.append((value, point))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1], "opposite_local_branch"
    return reflected, "geometric_reflection_unverified"


def _sample_uniform_polygon(
    model: Question2Model,
    count: int,
    rng: np.random.Generator,
    *,
    max_distance: float | None = None,
) -> np.ndarray:
    """Draw area-uniform points, optionally conditioning on a radius bound.

    The target polygon is an external approximation, so a very thin sliver can
    lie outside the physical first-reception disk.  Rejection sampling keeps
    the Monte-Carlo scenario conditional on ``||G-S1|| <= max_distance``
    instead of silently sampling an impossible receive radius.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if max_distance is not None and (not math.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be positive and finite")
    center = np.asarray(model.region.centroid, dtype=float)
    vertices = model.region.vertices
    triangles: list[np.ndarray] = []
    areas: list[float] = []
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        triangle = np.asarray((center, first, second), dtype=float)
        area = abs(0.5 * ((first[0] - center[0]) * (second[1] - center[1]) - (first[1] - center[1]) * (second[0] - center[0])))
        if area > 1e-14:
            triangles.append(triangle)
            areas.append(area)
    if not triangles:
        raise ValueError("target polygon cannot be sampled")
    probabilities = np.asarray(areas, dtype=float)
    probabilities /= np.sum(probabilities)
    def draw(batch_count: int) -> np.ndarray:
        selected = rng.choice(len(triangles), size=batch_count, p=probabilities)
        points = np.empty((batch_count, 2), dtype=float)
        for index, triangle_index in enumerate(selected):
            first, second = triangles[int(triangle_index)][1:]
            u, v = rng.random(2)
            root = math.sqrt(u)
            points[index] = (1.0 - root) * center + root * ((1.0 - v) * first + v * second)
        return points

    if max_distance is None:
        return draw(count)

    accepted: list[np.ndarray] = []
    accepted_count = 0
    attempts = 0
    # The external polygon overshoot is normally tiny; the generous cap keeps
    # a malformed geometry from turning a typo into an infinite loop.
    attempt_cap = max(10_000, 100 * count)
    while accepted_count < count and attempts < attempt_cap:
        batch = max(256, 2 * (count - accepted_count))
        candidates = draw(batch)
        distances = np.linalg.norm(candidates - model.s1[None, :], axis=1)
        keep = candidates[distances <= float(max_distance) + 1e-9]
        if len(keep):
            accepted.append(keep[: count - accepted_count])
            accepted_count += min(len(keep), count - accepted_count)
        attempts += batch
    if accepted_count < count:
        raise RuntimeError("could not sample enough first-reception-compatible target points")
    return np.vstack(accepted)[:count]


def _intersect_bearing_lines(
    s1: np.ndarray,
    p: np.ndarray,
    theta1: np.ndarray,
    theta2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect two noisy bearing lines in a vectorised form."""

    direction1 = np.column_stack((np.cos(theta1), np.sin(theta1)))
    direction2 = np.column_stack((np.cos(theta2), np.sin(theta2)))
    delta = p[None, :] - s1[None, :]
    determinants = direction1[:, 0] * direction2[:, 1] - direction1[:, 1] * direction2[:, 0]
    numerators = delta[:, 0] * direction2[:, 1] - delta[:, 1] * direction2[:, 0]
    finite = np.abs(determinants) > 1e-10
    scales = np.divide(numerators, determinants, out=np.full_like(numerators, np.nan), where=finite)
    estimates = s1[None, :] + scales[:, None] * direction1
    return estimates, finite


def run_monte_carlo(
    model: Question2Model,
    optimum: dict[str, Any],
    *,
    sample_count: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Simulate bounded bearing errors conditional on first reception."""

    primary = np.asarray(optimum["point"], dtype=float)
    opposite, opposite_source = _find_opposite_branch(model, optimum)
    branch_points = {"branch_primary": primary, "branch_opposite": opposite}
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    delta = math.radians(model.config.angle_error_deg)
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        targets = _sample_uniform_polygon(
            model,
            int(sample_count),
            rng,
            max_distance=model.config.max_receive_radius,
        )
        true_theta1 = np.arctan2(targets[:, 1] - model.s1[1], targets[:, 0] - model.s1[0])
        first_distances = np.linalg.norm(targets - model.s1[None, :], axis=1)
        # Conditioning on the already observed first detection: the radius is
        # sampled from the part of [r_min, r_max] that can receive G.
        lower_radius = np.maximum(model.config.min_receive_radius, first_distances)
        receive_radius = lower_radius + rng.random(sample_count) * (
            model.config.max_receive_radius - lower_radius
        )
        first_reception = receive_radius + 1e-12 >= first_distances
        errors_by_branch: dict[str, np.ndarray] = {}
        for branch_name, point in branch_points.items():
            point = np.asarray(point, dtype=float)
            second_distances = np.linalg.norm(targets - point[None, :], axis=1)
            second_reception = receive_radius + 1e-12 >= second_distances
            noisy_theta1 = true_theta1 + rng.uniform(-delta, delta, sample_count)
            noisy_theta2 = np.arctan2(targets[:, 1] - point[1], targets[:, 0] - point[0]) + rng.uniform(-delta, delta, sample_count)
            estimates, finite = _intersect_bearing_lines(model.s1, point, noisy_theta1, noisy_theta2)
            errors = np.linalg.norm(estimates - targets, axis=1)
            usable = first_reception & second_reception & finite
            errors[~usable] = np.inf
            finite_errors = np.isfinite(errors)
            errors_by_branch[branch_name] = errors
            summary_rows.append(
                {
                    "seed": int(seed),
                    "branch": branch_name,
                    "detector_x": float(point[0]),
                    "detector_y": float(point[1]),
                    "branch_source": opposite_source if branch_name == "branch_opposite" else "optimiser",
                    "sample_count": int(sample_count),
                    "first_reception_rate": float(np.mean(first_reception)),
                    "second_reception_rate": float(np.mean(second_reception)),
                    "joint_reception_rate": float(np.mean(first_reception & second_reception)),
                    "finite_intersection_rate": float(np.mean(finite)),
                    "rmse_m": float(np.sqrt(np.mean(errors[finite_errors] ** 2))) if np.any(finite_errors) else None,
                    "mae_m": float(np.mean(errors[finite_errors])) if np.any(finite_errors) else None,
                    "median_error_m": float(np.median(errors[finite_errors])) if np.any(finite_errors) else None,
                    "p95_error_m": float(np.quantile(errors[finite_errors], 0.95)) if np.any(finite_errors) else None,
                    "max_error_m": float(np.max(errors[finite_errors])) if np.any(finite_errors) else None,
                    "hit_rate_5m": float(np.mean(finite_errors & (errors <= 5.0))),
                    "hit_rate_10m": float(np.mean(finite_errors & (errors <= 10.0))),
                    "hit_rate_20m": float(np.mean(finite_errors & (errors <= 20.0))),
                    "hit_rate_50m": float(np.mean(finite_errors & (errors <= 50.0))),
                }
            )
            # Keep a compact sample table for reproducibility and plotting.
            for index in range(sample_count):
                sample_rows.append(
                    {
                        "seed": int(seed),
                        "sample_id": int(index),
                        "branch": branch_name,
                        "target_x": float(targets[index, 0]),
                        "target_y": float(targets[index, 1]),
                        "receive_radius": float(receive_radius[index]),
                        "first_distance": float(first_distances[index]),
                        "second_distance": float(second_distances[index]),
                        "finite_intersection": bool(finite[index]),
                        "error_m": None if not math.isfinite(float(errors[index])) else float(errors[index]),
                        "second_reception": bool(second_reception[index]),
                    }
                )
    # Aggregate seed-wise rows so the paper can report uncertainty across
    # random seeds rather than a single lucky Monte-Carlo run.
    aggregate_rows: list[dict[str, Any]] = []
    for branch in branch_points:
        selected = [row for row in summary_rows if row["branch"] == branch]
        aggregate: dict[str, Any] = {"branch": branch, "seed_count": len(selected)}
        for field in (
            "second_reception_rate",
            "joint_reception_rate",
            "finite_intersection_rate",
            "rmse_m",
            "mae_m",
            "median_error_m",
            "p95_error_m",
            "hit_rate_5m",
            "hit_rate_10m",
            "hit_rate_20m",
            "hit_rate_50m",
        ):
            values = np.asarray([row[field] for row in selected if row[field] is not None], dtype=float)
            aggregate[f"{field}_mean"] = None if not len(values) else float(np.mean(values))
            aggregate[f"{field}_std"] = None if not len(values) else float(np.std(values, ddof=0))
        aggregate_rows.append(aggregate)
    return {
        "protocol": {
            "target_sampling": "uniform area sampling by triangle decomposition of Omega_1",
            "first_reception_condition": "R is sampled uniformly from [max(min_receive_radius, ||G-S1||), max_receive_radius]",
            "bearing_errors": "independent uniform errors in [-angle_error_deg, angle_error_deg] at both detectors",
            "intersection": "two infinite bearing lines; near-parallel determinant <= 1e-10 is a failed finite intersection",
            "hit_rate_denominator": "all Monte-Carlo trials; first detection is conditioned to be successful",
            "thresholds_m": [5.0, 10.0, 20.0, 50.0],
        },
        "branch_points": {
            name: [float(value) for value in point]
            for name, point in branch_points.items()
        },
        "opposite_branch_source": opposite_source,
        "seed_summary": summary_rows,
        "aggregate": aggregate_rows,
        "samples": sample_rows,
    }


def run_search_comparison(
    s1: np.ndarray,
    bearing_deg: float,
    base_config: Q2Config,
    *,
    seeds: Sequence[int],
    max_local_starts: int,
    global_maxiter: int,
    global_popsize: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for use_global in (False, True):
        for seed in seeds:
            config = _clone_config(
                base_config,
                use_global_search=use_global,
                global_maxiter=int(global_maxiter),
                global_popsize=int(global_popsize),
            )
            model, optimum, elapsed = _optimise(
                s1,
                bearing_deg,
                config,
                seed=int(seed),
                use_global_search=use_global,
                max_local_starts=max_local_starts,
            )
            row = _row_from_optimum(
                model,
                optimum,
                seed=int(seed),
                use_global_search=use_global,
                elapsed_s=elapsed,
                experiment="search_comparison",
            )
            rows.append(row)
    return rows


def plot_convergence(payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    from mathmodel_viz.styles import get_theme, save_figure

    theme = get_theme()
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.4))
    circles = payload["convergence"]["circle_sides"]
    axes[0, 0].plot([row["setting"] for row in circles], [row["native_j_star"] for row in circles], "o-", color=theme.primary_blue, label="J* native")
    axes[0, 0].plot([row["setting"] for row in circles], [row["j_at_reference_order"] for row in circles], "s--", color=theme.primary_red, label="J* at reference quadrature")
    axes[0, 0].set_xlabel("circle_sides")
    axes[0, 0].set_ylabel("objective / m")
    axes[0, 0].set_title("Circle polygon convergence")
    axes[0, 0].legend()
    axes[0, 1].plot([row["setting"] for row in circles], [row["region_area"] for row in circles], "o-", color=theme.transition)
    axes[0, 1].set_xlabel("circle_sides")
    axes[0, 1].set_ylabel("target area / m$^2$")
    axes[0, 1].set_title("Conservative target-region area")
    quadrature = payload["convergence"]["quadrature_order"]
    axes[1, 0].plot([row["setting"] for row in quadrature], [row["native_j_star"] for row in quadrature], "o-", color=theme.primary_blue, label="native")
    axes[1, 0].plot([row["setting"] for row in quadrature], [row["j_at_reference_order"] for row in quadrature], "s--", color=theme.primary_red, label="reference order")
    axes[1, 0].set_xlabel("quadrature_order")
    axes[1, 0].set_ylabel("objective / m")
    axes[1, 0].set_title("Quadrature convergence")
    axes[1, 0].legend()
    grids = payload["convergence"]["candidate_grid"]
    axes[1, 1].plot([row["grid_size"] for row in grids], [row["cell_center_area"] for row in grids], "o-", color=theme.primary_blue, label="cell centres")
    axes[1, 1].plot([row["grid_size"] for row in grids], [row["marching_area"] for row in grids], "s--", color=theme.primary_red, label="marching squares")
    axes[1, 1].set_xlabel("grid nodes per axis")
    axes[1, 1].set_ylabel("candidate area / m$^2$")
    axes[1, 1].set_title("Near-optimal area resolution")
    axes[1, 1].legend()
    figure.suptitle("Question 2 numerical convergence diagnostics")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(figure, output_path)
    plt.close(figure)


def plot_search_stability(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    from mathmodel_viz.styles import get_theme, save_figure

    theme = get_theme()
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    labels = [f"{('global' if row['use_global_search'] else 'local')}\nseed={row['seed']}" for row in rows]
    x_values = np.arange(len(rows))
    axes[0].scatter(x_values, [row["j_star"] for row in rows], c=[theme.primary_red if row["use_global_search"] else theme.primary_blue for row in rows], s=55)
    axes[0].set_xticks(x_values)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("J* / m")
    axes[0].set_title("Search result by seed")
    axes[1].scatter([row["point_x"] for row in rows], [row["point_y"] for row in rows], c=[theme.primary_red if row["use_global_search"] else theme.primary_blue for row in rows], s=60)
    for row in rows:
        axes[1].annotate(f"{('G' if row['use_global_search'] else 'L')}-{row['seed']}", (row["point_x"], row["point_y"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("detector x / m")
    axes[1].set_ylabel("detector y / m")
    axes[1].set_title("Returned optimum branches")
    figure.suptitle("Local versus global search stability")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_path)
    plt.close(figure)


def plot_monte_carlo(monte_carlo: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    from mathmodel_viz.styles import get_theme, save_figure

    theme = get_theme()
    aggregate = monte_carlo["aggregate"]
    branches = [row["branch"] for row in aggregate]
    x_values = np.arange(len(branches))
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.8))
    rmse = [row.get("rmse_m_mean") or np.nan for row in aggregate]
    p95 = [row.get("p95_error_m_mean") or np.nan for row in aggregate]
    width = 0.34
    axes[0].bar(x_values - width / 2, rmse, width, color=theme.primary_blue, label="RMSE")
    axes[0].bar(x_values + width / 2, p95, width, color=theme.primary_red, label="P95")
    axes[0].set_xticks(x_values)
    axes[0].set_xticklabels(branches, rotation=20)
    axes[0].set_ylabel("localization error / m")
    axes[0].set_title("Error summaries")
    axes[0].legend()
    hit_fields = ("hit_rate_5m_mean", "hit_rate_10m_mean", "hit_rate_20m_mean", "hit_rate_50m_mean")
    labels = ("5 m", "10 m", "20 m", "50 m")
    for index, (field, label) in enumerate(zip(hit_fields, labels)):
        axes[1].plot(x_values, [row.get(field) or np.nan for row in aggregate], "o-", label=label, color=theme.series_colors[index])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x_values)
    axes[1].set_xticklabels(branches, rotation=20)
    axes[1].set_ylabel("hit rate")
    axes[1].set_title("Position hit rate")
    axes[1].legend()
    reception = [row.get("joint_reception_rate_mean") or np.nan for row in aggregate]
    finite = [row.get("finite_intersection_rate_mean") or np.nan for row in aggregate]
    axes[2].bar(x_values - width / 2, reception, width, color=theme.transition, label="joint reception")
    axes[2].bar(x_values + width / 2, finite, width, color=theme.primary_blue, label="finite intersection")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(x_values)
    axes[2].set_xticklabels(branches, rotation=20)
    axes[2].set_ylabel("rate")
    axes[2].set_title("Reception and intersection")
    axes[2].legend()
    figure.suptitle("Monte-Carlo positioning validation")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(figure, output_path)
    plt.close(figure)


def plot_geometry_overlay(
    model: Question2Model,
    optimum: dict[str, Any],
    *,
    output_path: Path,
    grid_size: int,
    taus: Sequence[float],
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    from mathmodel_viz.styles import get_theme, save_figure

    theme = get_theme()
    surfaces = {float(tau): evaluate_surface(model, optimum, grid_size=grid_size, tau=float(tau)) for tau in taus}
    reference_surface = surfaces[float(taus[0])]
    xx = reference_surface["xx"]
    yy = reference_surface["yy"]
    h_values = reference_surface["h_values"]
    h_tol = reference_surface["h_tolerance"]
    figure, axis = plt.subplots(figsize=(8.5, 7.3))
    feasible = np.where(h_values <= h_tol, 1.0, np.nan)
    axis.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=[theme.color("blue", 0.10)], alpha=0.55)
    axis.contour(xx, yy, h_values, levels=[h_tol], colors=[theme.primary_blue], linewidths=1.4, linestyles="--")
    vertices = np.vstack((model.region.vertices, model.region.vertices[0]))
    axis.fill(vertices[:, 0], vertices[:, 1], color=theme.color("transition", 0.18), label="target region Omega_1")
    axis.plot(vertices[:, 0], vertices[:, 1], color=theme.transition, linewidth=1.5)
    contour_data: list[dict[str, Any]] = []
    level_colors = (theme.primary_red, theme.transition, theme.auxiliary, theme.primary_blue)
    for index, tau in enumerate(taus):
        surface = surfaces[float(tau)]
        valid_level = surface["finite"] & surface["reception_feasible"]
        field = np.where(
            valid_level,
            surface["j_values"] - surface["threshold"],
            max(1.0, abs(surface["threshold"])),
        )
        contours = _extract_contours(field, surface["x_values"], surface["y_values"])
        color = level_colors[index % len(level_colors)]
        for contour_id, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            axis.plot(contour[:, 0], contour[:, 1], color=color, linewidth=1.5, label=f"J=(1+{float(tau):g})J*")
            contour_data.append({"tau": float(tau), "contour_id": int(contour_id), "points": _downsample_path(contour).astype(float).tolist()})
    primary = np.asarray(optimum["point"], dtype=float)
    opposite, opposite_source = _find_opposite_branch(model, optimum)
    axis.scatter([primary[0], opposite[0]], [primary[1], opposite[1]], c=[theme.primary_red, theme.primary_blue], marker="*", s=120, zorder=5, label="two optimum branches")
    axis.scatter([model.s1[0]], [model.s1[1]], color=theme.foreground, marker="+", s=90, zorder=5, label="S1")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_title("Target region, reception-feasible domain and near-optimal levels")
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    axis.legend(list(unique.values()), list(unique.keys()), loc="best", fontsize=8)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    save_figure(figure, output_path)
    plt.close(figure)
    return {
        "grid_size": int(grid_size),
        "taus": [float(value) for value in taus],
        "contours": contour_data,
        "opposite_branch_source": opposite_source,
        "reception_boundary_level": float(h_tol),
    }


def _prepare_matplotlib() -> None:
    visualization_src = Path(__file__).resolve().parents[2] / "tools" / "visualization" / "src"
    if str(visualization_src) not in sys.path:
        sys.path.insert(0, str(visualization_src))
    from mathmodel_viz.styles import configure_matplotlib

    configure_matplotlib(font_candidates=("Microsoft YaHei", "Noto Sans SC"), figure_dpi=120, savefig_dpi=300)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Question 2 JSON input path")
    source.add_argument("--demo", action="store_true", help="use the documented S1=(0,0), bearing=32 degree case")
    parser.add_argument("--output-dir", default="outputs/t2/enhanced", help="directory for tables, JSON and figures")
    parser.add_argument("--quick", action="store_true", help="small structurally identical smoke design")
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--samples", type=int, default=2000, help="Monte-Carlo samples per seed and branch")
    parser.add_argument("--seeds", help="comma-separated search/Monte-Carlo seeds")
    parser.add_argument("--circle-values", help="comma-separated circle_sides convergence values")
    parser.add_argument("--quadrature-values", help="comma-separated quadrature orders")
    parser.add_argument("--grid-values", help="comma-separated candidate grid node counts")
    parser.add_argument("--max-local-starts", type=int, default=24)
    parser.add_argument("--local-maxiter", type=int)
    parser.add_argument("--global-maxiter", type=int)
    parser.add_argument("--global-popsize", type=int)
    parser.add_argument("--boundary-grid", type=int, help="grid nodes per axis for the overlay figure")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        s1, bearing_deg, base_config = load_input(args.input)
    else:
        s1, bearing_deg, base_config = parse_input(
            {"s1": {"x": 0.0, "y": 0.0}, "svd_deg": 32.0}
        )
    if args.quick:
        base_config = _clone_config(
            base_config,
            circle_sides=64,
            quadrature_order=4,
            validation_quadrature_order=6,
            candidate_grid=31,
            local_maxiter=args.local_maxiter or 60,
            global_maxiter=args.global_maxiter or 10,
            global_popsize=args.global_popsize or 5,
        )
        circle_defaults = QUICK_CIRCLE_VALUES
        quadrature_defaults = QUICK_QUADRATURE_VALUES
        grid_defaults = QUICK_GRID_VALUES
        seed_defaults = QUICK_SEEDS
    else:
        base_config = _clone_config(
            base_config,
            local_maxiter=args.local_maxiter or min(base_config.local_maxiter, 120),
            global_maxiter=args.global_maxiter or min(base_config.global_maxiter, 16),
            global_popsize=args.global_popsize or min(base_config.global_popsize, 6),
        )
        circle_defaults = DEFAULT_CIRCLE_VALUES
        quadrature_defaults = DEFAULT_QUADRATURE_VALUES
        grid_defaults = DEFAULT_GRID_VALUES
        seed_defaults = DEFAULT_SEEDS
    circle_values = _parse_int_list(args.circle_values, circle_defaults)
    quadrature_values = _parse_int_list(args.quadrature_values, quadrature_defaults)
    grid_values = _parse_int_list(args.grid_values, grid_defaults)
    seeds = _parse_int_list(args.seeds, seed_defaults)
    if args.samples < 1:
        raise ValueError("samples must be positive")
    boundary_grid = int(args.boundary_grid or (81 if args.quick else 161))
    if args.max_local_starts < 1:
        raise ValueError("max-local-starts must be positive")
    out_dir = Path(args.output_dir)
    figure_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    _prepare_matplotlib()

    convergence = run_convergence(
        s1,
        bearing_deg,
        base_config,
        circle_values=circle_values,
        quadrature_values=quadrature_values,
        grid_values=grid_values,
        tau=base_config.tau,
        seed=int(seeds[0]),
        max_local_starts=int(args.max_local_starts),
    )
    search_rows = run_search_comparison(
        s1,
        bearing_deg,
        base_config,
        seeds=seeds,
        max_local_starts=int(args.max_local_starts),
        global_maxiter=int(base_config.global_maxiter),
        global_popsize=int(base_config.global_popsize),
    )
    final_model: Question2Model = convergence["final_model"]
    final_optimum: dict[str, Any] = convergence["final_optimum"]
    overlay = plot_geometry_overlay(
        final_model,
        final_optimum,
        output_path=figure_dir / "q2_geometry_candidate_overlay.png",
        grid_size=boundary_grid,
        taus=(0.01, base_config.tau, 0.10),
    )
    monte_carlo: dict[str, Any]
    if args.skip_monte_carlo:
        monte_carlo = {"skipped": True}
    else:
        monte_carlo = run_monte_carlo(
            final_model,
            final_optimum,
            sample_count=int(args.samples),
            seeds=seeds,
        )
        _write_csv(out_dir / "q2_monte_carlo_seed_summary.csv", monte_carlo["seed_summary"])
        _write_csv(out_dir / "q2_monte_carlo_aggregate.csv", monte_carlo["aggregate"])
        _write_csv(out_dir / "q2_monte_carlo_samples.csv", monte_carlo["samples"])
        plot_monte_carlo(monte_carlo, figure_dir / "q2_monte_carlo_error.png")

    payload: dict[str, Any] = {
        "model": "question_2_continuous_robust_dop",
        "input": {"s1": [float(s1[0]), float(s1[1])], "svd_deg": float(bearing_deg)},
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(base_config).items()
        },
        "design": {
            "circle_values": [int(value) for value in circle_values],
            "quadrature_values": [int(value) for value in quadrature_values],
            "grid_values": [int(value) for value in grid_values],
            "search_seeds": [int(value) for value in seeds],
            "max_local_starts": int(args.max_local_starts),
            "boundary_grid": boundary_grid,
            "quick": bool(args.quick),
        },
        "convergence": {
            key: value
            for key, value in convergence.items()
            if key not in {"final_model", "final_optimum", "boundary_records"}
        },
        "search_comparison": search_rows,
        "geometry_overlay": overlay,
        "monte_carlo": monte_carlo,
        "metadata": {
            "strict_vs_tolerance": "strict_feasible means H<=0; feasible_with_tolerance uses the configured distance-scale residual tolerance.",
            "circle_convergence": "Every circle_sides row reconstructs the target polygon and re-optimises the detector position.",
            "quadrature_convergence": "Every quadrature_order row re-optimises; j_at_reference_order removes the native-order comparison bias.",
            "grid_convergence": "Cell-centre area is a raster estimate; marching_area is the sum of closed zero contours, with raster fallback for clipped/open contours.",
            "boundary_method": "skimage.measure.find_contours (marching squares) on J(P)-(1+tau)J*; H(P)=tolerance is plotted as a separate reception boundary.",
            "search_comparison": "The same local-start budget is used for every seed; global rows add differential evolution before SLSQP polishing.",
            "monte_carlo": "Uniform area target sampling and bounded uniform bearing errors are a validation scenario, not a statistical assumption in the mathematical model.",
        },
    }
    _write_json(out_dir / "q2_enhanced_experiments.json", payload)
    _write_csv(out_dir / "q2_circle_convergence.csv", convergence["circle_sides"])
    _write_csv(out_dir / "q2_quadrature_convergence.csv", convergence["quadrature_order"])
    _write_csv(out_dir / "q2_grid_convergence.csv", convergence["candidate_grid"])
    _write_csv(out_dir / "q2_search_comparison.csv", search_rows)
    _write_csv(out_dir / "q2_boundary_contours.csv", convergence["boundary_records"])
    plot_convergence(payload, figure_dir / "q2_convergence_tables.png")
    plot_search_stability(search_rows, figure_dir / "q2_search_stability.png")
    print(f"saved enhanced experiment JSON: {out_dir / 'q2_enhanced_experiments.json'}")
    print(f"saved tables and figures below: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
