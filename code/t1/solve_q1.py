"""Solve Question 1 of CUMCM 2026 B.

The input consists of detector coordinates and measured bearings.  Each bearing
is treated as a bounded error interval (default +/-1 degree), which gives two
linear half-plane constraints.  The script classifies their intersection,
constructs the boundary polygon when it is bounded, and computes:

* the region diameter (rotating calipers for a 2-D polygon),
* whether the diameter circle can cover the region, and
* the exact minimum enclosing circle of the boundary vertices.

No artificial bounding box is introduced.  SciPy is used only for feasibility
and coordinate-boundedness checks; all geometric output is computed explicitly.

Accepted JSON input formats::

    {"measurements": [{"x": 10, "y": 20, "svd_deg": 35.2}, ...],
     "angle_error_deg": 1.0}
    {"points": [[10, 20, 35.2], ...], "angle_error_deg": 1.0}

CSV input must contain x, y and one of svd_deg/theta_deg/theta/angle_deg.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linprog


DEFAULT_ANGLE_ERROR_DEG = 1.0
_ANGLE_KEYS = ("svd_deg", "theta_deg", "theta", "angle_deg", "bearing_deg")


def _cross2(left: np.ndarray, right: np.ndarray) -> float:
    """Return the scalar two-dimensional cross product."""

    return float(left[0] * right[1] - left[1] * right[0])


@dataclass(frozen=True)
class HalfPlane:
    """A half-plane represented by a directed boundary line and its normal."""

    a: np.ndarray
    b: float
    p: np.ndarray
    d: np.ndarray
    angle: float
    source_index: int


def _as_finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _extract_row(row: Any, index: int) -> tuple[float, float, float]:
    if isinstance(row, dict):
        try:
            x = row["x"]
            y = row["y"]
        except KeyError as exc:
            raise ValueError(f"measurement {index} must contain x and y") from exc
        theta_key = next((key for key in _ANGLE_KEYS if key in row), None)
        if theta_key is None:
            raise ValueError(
                f"measurement {index} must contain one of {', '.join(_ANGLE_KEYS)}"
            )
        theta = row[theta_key]
    elif isinstance(row, (list, tuple)) and len(row) >= 3:
        x, y, theta = row[:3]
    else:
        raise ValueError(
            f"measurement {index} must be an object or [x, y, svd_deg] sequence"
        )
    return (
        _as_finite_float(x, f"measurement {index}.x"),
        _as_finite_float(y, f"measurement {index}.y"),
        _as_finite_float(theta, f"measurement {index}.svd_deg"),
    )


def parse_measurements(source: Any) -> tuple[list[tuple[float, float, float]], float]:
    """Parse a JSON-like object into measurements and angular error in degrees."""

    if isinstance(source, list):
        rows = source
        error_deg = DEFAULT_ANGLE_ERROR_DEG
    elif isinstance(source, dict):
        rows = source.get("measurements", source.get("points"))
        if rows is None:
            raise ValueError("input must contain a 'measurements' or 'points' array")
        error_deg = source.get("angle_error_deg", DEFAULT_ANGLE_ERROR_DEG)
    else:
        raise ValueError("input must be a JSON object or an array")

    if not isinstance(rows, list) or not rows:
        raise ValueError("measurements must be a non-empty array")
    error_deg = _as_finite_float(error_deg, "angle_error_deg")
    if not (0.0 < error_deg < 90.0):
        raise ValueError("angle_error_deg must be in (0, 90)")
    return [_extract_row(row, i) for i, row in enumerate(rows)], error_deg


def load_input(path: str | Path) -> tuple[list[tuple[float, float, float]], float]:
    """Load JSON or CSV input from *path* (or '-' for stdin)."""

    if str(path) == "-":
        text = sys.stdin.read()
        return parse_measurements(json.loads(text))
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return parse_measurements(rows)
    with input_path.open("r", encoding="utf-8") as handle:
        return parse_measurements(json.load(handle))


def build_constraints(
    measurements: Sequence[tuple[float, float, float]], angle_error_deg: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A, b`` for the constraints ``A @ z <= b``.

    For a reading theta at s, the feasible bearing is [theta-delta, theta+delta].
    The two rows are the half-plane form in equations (1)-(2) of the design note.
    """

    angle_error_deg = _as_finite_float(angle_error_deg, "angle_error_deg")
    if not (0.0 < angle_error_deg < 90.0):
        raise ValueError("angle_error_deg must be in (0, 90)")
    delta = math.radians(angle_error_deg)
    normalised_measurements = [
        _extract_row(row, index) for index, row in enumerate(measurements)
    ]
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for x, y, theta_deg in normalised_measurements:
        theta = math.radians(theta_deg % 360.0)
        lower = theta - delta
        upper = theta + delta
        a_lower = np.array([math.sin(lower), -math.cos(lower)], dtype=float)
        a_upper = np.array([-math.sin(upper), math.cos(upper)], dtype=float)
        s = np.array([x, y], dtype=float)
        rows.extend((a_lower, a_upper))
        rhs.extend((float(a_lower @ s), float(a_upper @ s)))
    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def _intersection(h1: HalfPlane, h2: HalfPlane) -> np.ndarray | None:
    denominator = _cross2(h1.d, h2.d)
    if abs(denominator) <= 1e-14:
        return None
    parameter = _cross2(h2.p - h1.p, h2.d) / denominator
    return h1.p + parameter * h1.d


def _outside(h: HalfPlane, point: np.ndarray, tol: float) -> bool:
    return _cross2(h.d, point - h.p) < -tol


def _make_halfplanes(a: np.ndarray, b: np.ndarray) -> list[HalfPlane]:
    halfplanes: list[HalfPlane] = []
    for index, (normal, bound) in enumerate(zip(a, b)):
        # The input normals have unit length, but normalisation keeps this helper
        # stable if build_constraints is reused with externally supplied rows.
        norm = float(np.linalg.norm(normal))
        if norm == 0.0:
            raise ValueError("a constraint normal cannot be zero")
        unit_a = np.asarray(normal, dtype=float) / norm
        unit_b = float(bound) / norm
        direction = np.array([-unit_a[1], unit_a[0]], dtype=float)
        angle = math.atan2(direction[1], direction[0]) % (2.0 * math.pi)
        halfplanes.append(
            HalfPlane(
                a=unit_a,
                b=unit_b,
                p=unit_b * unit_a,
                d=direction,
                angle=angle,
                source_index=index,
            )
        )
    return halfplanes


def _deduplicate_directions(halfplanes: list[HalfPlane], tol: float) -> list[HalfPlane]:
    """Sort by direction and retain the strictest constraint per direction."""

    ordered = sorted(halfplanes, key=lambda h: h.angle)
    result: list[HalfPlane] = []
    for current in ordered:
        if result:
            previous = result[-1]
            if abs(_cross2(previous.d, current.d)) <= 1e-13 and float(
                previous.d @ current.d
            ) > 0:
                if current.b < previous.b - tol:
                    result[-1] = current
                continue
        result.append(current)
    # The first and last bins meet at angle 0/2pi and need the same treatment.
    if len(result) > 1:
        first, last = result[0], result[-1]
        if abs(_cross2(first.d, last.d)) <= 1e-13 and float(first.d @ last.d) > 0:
            result[0] = first if first.b <= last.b else last
            result.pop()
    return result


def halfplane_intersection(
    a: np.ndarray, b: np.ndarray, tol: float = 1e-8
) -> list[np.ndarray]:
    """Construct a full-dimensional bounded intersection using a deque HPI.

    A pairwise feasible-intersection fallback is retained for lower-dimensional
    bounded sets, where a standard deque HPI has fewer than three boundary lines.
    """

    halfplanes = _deduplicate_directions(_make_halfplanes(a, b), tol)
    deque: list[HalfPlane] = []
    for halfplane in halfplanes:
        while len(deque) >= 2:
            point = _intersection(deque[-2], deque[-1])
            if point is None or not _outside(halfplane, point, tol):
                break
            deque.pop()
        while len(deque) >= 2:
            point = _intersection(deque[0], deque[1])
            if point is None or not _outside(halfplane, point, tol):
                break
            deque.pop(0)
        deque.append(halfplane)

    while len(deque) >= 3:
        point = _intersection(deque[-2], deque[-1])
        if point is None or not _outside(deque[0], point, tol):
            break
        deque.pop()
    while len(deque) >= 3:
        point = _intersection(deque[0], deque[1])
        if point is None or not _outside(deque[-1], point, tol):
            break
        deque.pop(0)

    if len(deque) >= 3 and all(
        _intersection(deque[i], deque[(i + 1) % len(deque)]) is not None
        for i in range(len(deque))
    ):
        vertices = [
            _intersection(deque[i], deque[(i + 1) % len(deque)])
            for i in range(len(deque))
        ]
        return [point for point in vertices if point is not None]
    return []


def _pairwise_feasible_vertices(a: np.ndarray, b: np.ndarray, tol: float) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            matrix = np.asarray((a[i], a[j]), dtype=float)
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) <= 1e-13:
                continue
            point = np.linalg.solve(matrix, np.asarray((b[i], b[j]), dtype=float))
            if np.all(a @ point <= b + tol):
                candidates.append(point)
    return _deduplicate_points(candidates, tol)


def _deduplicate_points(points: Iterable[np.ndarray], tol: float) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(float(np.linalg.norm(point - other)) <= tol for other in unique):
            unique.append(np.asarray(point, dtype=float))
    return unique


def convex_hull(points: Sequence[np.ndarray], tol: float = 1e-8) -> list[np.ndarray]:
    """Return a counter-clockwise monotonic-chain hull."""

    if len(points) == 0:
        return []
    # Cross-product tolerances must be translation invariant.  Using absolute
    # coordinates here would incorrectly collapse a small polygon near (1e12,
    # 1e12) into a line even though its edge lengths are well resolved.
    anchor = np.asarray(points[0], dtype=float)
    scale = max(1.0, max(float(np.max(np.abs(np.asarray(point) - anchor))) for point in points))
    cross_tol = tol * scale * scale
    ordered = sorted(
        _deduplicate_points((np.asarray(p, dtype=float) for p in points), tol),
        key=lambda p: (p[0], p[1]),
    )
    if len(ordered) == 1:
        return ordered

    def cross(o: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
        return _cross2(p - o, q - o)

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= cross_tol:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= cross_tol:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _feasible_and_bounded(a: np.ndarray, b: np.ndarray) -> tuple[str, np.ndarray | None]:
    if len(a) == 0:
        return "unbounded", None
    feasibility = linprog(
        np.zeros(2),
        A_ub=a,
        b_ub=b,
        bounds=[(None, None), (None, None)],
        method="highs",
    )
    if feasibility.status == 2:
        return "empty", None
    if feasibility.status != 0:
        raise RuntimeError(f"linear feasibility check failed: {feasibility.message}")
    for objective in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        bound_check = linprog(
            np.asarray(objective),
            A_ub=a,
            b_ub=b,
            bounds=[(None, None), (None, None)],
            method="highs",
        )
        if bound_check.status == 3:
            return "unbounded", np.asarray(feasibility.x, dtype=float)
        if bound_check.status not in (0, 4):
            raise RuntimeError(f"boundedness check failed: {bound_check.message}")
    return "bounded", np.asarray(feasibility.x, dtype=float)


def _distance_sq(p: np.ndarray, q: np.ndarray) -> float:
    difference = p - q
    return float(difference @ difference)


def diameter_rotating_calipers(vertices: Sequence[np.ndarray], tol: float = 1e-10) -> tuple[float, int, int]:
    """Return diameter and endpoint indices of a CCW convex polygon in O(m)."""

    count = len(vertices)
    if count < 2:
        return 0.0, 0, 0
    if count == 2:
        return math.sqrt(_distance_sq(vertices[0], vertices[1])), 0, 1
    points = [np.asarray(point, dtype=float) for point in vertices]
    best_sq, best_pair = 0.0, (0, 0)

    def update(i: int, j: int) -> None:
        nonlocal best_sq, best_pair
        value = _distance_sq(points[i], points[j])
        if value > best_sq + tol:
            best_sq, best_pair = value, (i, j)

    j = 1
    for i in range(count):
        ni = (i + 1) % count
        edge = points[ni] - points[i]
        while True:
            nj = (j + 1) % count
            current_area = _cross2(edge, points[j] - points[i])
            next_area = _cross2(edge, points[nj] - points[i])
            if next_area > current_area + tol:
                j = nj
                continue
            break
        update(i, j)
        update(ni, j)
        if abs(
            _cross2(edge, points[(j + 1) % count] - points[i])
            - _cross2(edge, points[j] - points[i])
        ) <= tol:
            update(i, (j + 1) % count)
            update(ni, (j + 1) % count)
    return math.sqrt(best_sq), best_pair[0], best_pair[1]


def _circle_from_three(p: np.ndarray, q: np.ndarray, r: np.ndarray, tol: float) -> tuple[np.ndarray, float] | None:
    """Return the circumcircle of three points, or ``None`` if near-collinear.

    The equations are translated by ``p`` before solving.  This avoids the
    cancellation in ``q @ q - p @ p`` when coordinates are large but the
    triangle itself is small.
    """

    q_relative = q - p
    r_relative = r - p
    determinant = 2.0 * _cross2(q_relative, r_relative)
    local_scale = max(1.0, float(np.max(np.abs(np.vstack((q_relative, r_relative))))))
    if abs(determinant) <= tol * local_scale * local_scale:
        return None
    matrix = 2.0 * np.asarray((q_relative, r_relative), dtype=float)
    rhs = np.asarray((q_relative @ q_relative, r_relative @ r_relative), dtype=float)
    center = p + np.linalg.solve(matrix, rhs)
    return center, math.sqrt(_distance_sq(center, p))


def _circle_from_two(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    difference = q - p
    # Translate before averaging; (p + q) / 2 loses the low bits when both
    # coordinates share a large common offset.
    center = p + difference / 2.0
    return center, math.sqrt(float(difference @ difference)) / 2.0


def _contains_circle(
    circle: tuple[np.ndarray, float, list[int]], point: np.ndarray, tolerance: float
) -> bool:
    center, radius, _ = circle
    return _distance_sq(point, center) <= (radius + tolerance) ** 2


def _minimum_circle_for_support(
    points: Sequence[np.ndarray], support: Sequence[int], tolerance: float
) -> tuple[np.ndarray, float, list[int]]:
    """Find the smallest circle covering one, two, or three support points."""

    indices = list(dict.fromkeys(support))
    if not indices:
        return np.zeros(2, dtype=float), 0.0, []
    if len(indices) == 1:
        return np.asarray(points[indices[0]], dtype=float), 0.0, indices

    candidates: list[tuple[np.ndarray, float, list[int]]] = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            i, j = indices[left], indices[right]
            center, radius = _circle_from_two(points[i], points[j])
            candidates.append((center, radius, [i, j]))
    if len(indices) >= 3:
        i, j, k = indices[:3]
        circle = _circle_from_three(points[i], points[j], points[k], tolerance)
        if circle is not None:
            center, radius = circle
            candidates.append((center, radius, [i, j, k]))

    feasible = []
    for center, radius, boundary in candidates:
        if all(_distance_sq(points[index], center) <= (radius + tolerance) ** 2 for index in indices):
            feasible.append((center, radius, boundary))
    if not feasible:
        raise RuntimeError("failed to construct a circle for support points")
    return min(feasible, key=lambda item: item[1])


def _minimum_enclosing_circle_enumeration(
    points: Sequence[np.ndarray], tolerance: float
) -> tuple[np.ndarray, float]:
    """Exact 2/3-point candidate enumeration used as a numerical fallback."""

    candidates: list[tuple[np.ndarray, float]] = []
    count = len(points)
    for i in range(count):
        for j in range(i + 1, count):
            candidates.append(_circle_from_two(points[i], points[j]))
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                circle = _circle_from_three(points[i], points[j], points[k], tolerance)
                if circle is not None:
                    candidates.append(circle)

    for center, radius in sorted(candidates, key=lambda item: item[1]):
        if all(_distance_sq(point, center) <= (radius + tolerance) ** 2 for point in points):
            return center, radius
    raise RuntimeError("failed to construct a covering circle")


def minimum_enclosing_circle(vertices: Sequence[np.ndarray], tol: float = 1e-8) -> dict[str, Any]:
    """Find the minimum enclosing circle with a randomized incremental solver.

    In the plane, the active support has at most three points.  The nested
    boundary updates therefore implement Welzl's randomized incremental method
    in expected linear time.  A candidate enumeration fallback is retained for
    rare floating-point inconsistencies.
    """

    points = [np.asarray(point, dtype=float) for point in vertices]
    if not points:
        raise ValueError("minimum enclosing circle requires at least one point")
    if any(point.shape != (2,) or not np.all(np.isfinite(point)) for point in points):
        raise ValueError("minimum enclosing circle points must be finite 2-D vectors")
    anchor = points[0]
    local_scale = max(1.0, max(float(np.max(np.abs(point - anchor))) for point in points))
    absolute_scale = max(1.0, max(float(np.max(np.abs(point))) for point in points))
    roundoff_tolerance = 32.0 * np.finfo(float).eps * absolute_scale
    tolerance = max(abs(float(tol)), 1e-12 * local_scale, roundoff_tolerance)
    if len(points) == 1:
        return {"center": points[0], "radius": 0.0, "support_indices": [0]}

    order = list(range(len(points)))
    random.Random(0xC0DEC0DE).shuffle(order)
    circle = _minimum_circle_for_support(points, [], tolerance)
    for outer, index in enumerate(order):
        if _contains_circle(circle, points[index], tolerance):
            continue
        circle = _minimum_circle_for_support(points, [index], tolerance)
        for middle in range(outer):
            middle_index = order[middle]
            if _contains_circle(circle, points[middle_index], tolerance):
                continue
            circle = _minimum_circle_for_support(points, [index, middle_index], tolerance)
            for inner in range(middle):
                inner_index = order[inner]
                if _contains_circle(circle, points[inner_index], tolerance):
                    continue
                circle = _minimum_circle_for_support(
                    points, [index, middle_index, inner_index], tolerance
                )

    center, radius, _ = circle
    distances = np.sqrt([_distance_sq(point, center) for point in points])
    maximum_distance = float(np.max(distances))
    if maximum_distance > radius + 10.0 * tolerance:
        center, radius = _minimum_enclosing_circle_enumeration(points, tolerance)
        distances = np.sqrt([_distance_sq(point, center) for point in points])
    radius = max(float(radius), float(np.max(distances)))
    support_tolerance = max(10.0 * tolerance, 1e-10 * max(1.0, radius))
    support_indices = [
        index for index, distance in enumerate(distances) if abs(float(distance) - radius) <= support_tolerance
    ]
    return {
        "center": np.asarray(center, dtype=float),
        "radius": radius,
        "support_indices": support_indices,
    }


def _point_list(points: Sequence[np.ndarray]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in points]


def solve_question1(
    measurements: Sequence[tuple[float, float, float]],
    angle_error_deg: float = DEFAULT_ANGLE_ERROR_DEG,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Solve Question 1 and return a JSON-serialisable result dictionary."""

    measurements = [
        _extract_row(row, index) for index, row in enumerate(measurements)
    ]
    if not measurements:
        raise ValueError("measurements must be a non-empty sequence")
    a, b = build_constraints(measurements, angle_error_deg)
    coordinate_array = np.asarray([row[:2] for row in measurements], dtype=float)
    anchor = coordinate_array[0]
    extent = max(
        1.0,
        float(np.max(np.ptp(coordinate_array, axis=0))),
        float(np.max(np.abs(coordinate_array - anchor))),
    )
    origin_scale = max(
        1.0,
        float(np.max(np.abs(coordinate_array))),
        float(np.max(np.abs(b))) if len(b) else 1.0,
    )
    roundoff_scale = 32.0 * np.finfo(float).eps * origin_scale
    tol = float(tolerance) if tolerance is not None else 1e-9 * extent + roundoff_scale
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be a finite non-negative number")
    status, feasible_point = _feasible_and_bounded(a, b)
    result: dict[str, Any] = {
        "status": status,
        "angle_error_deg": float(angle_error_deg),
        "measurement_count": len(measurements),
        "constraint_count": int(len(a)),
        "vertices": [],
        "diameter": None,
        "diameter_endpoints": None,
        "diameter_circle": None,
        "minimum_cover_circle": None,
    }
    if status == "empty" or status == "unbounded":
        if status == "unbounded" and feasible_point is not None:
            result["feasible_point"] = [float(feasible_point[0]), float(feasible_point[1])]
        return result

    hpi_vertices = halfplane_intersection(a, b, tol=tol)
    if hpi_vertices and all(np.all(a @ point <= b + tol) for point in hpi_vertices):
        candidates = hpi_vertices
    else:
        candidates = _pairwise_feasible_vertices(a, b, tol)
    vertices = convex_hull(candidates, tol=tol)
    result["vertices"] = _point_list(vertices)
    if not vertices:
        # A bounded non-empty polyhedron always has at least one extreme point;
        # reaching this branch indicates a numerical inconsistency in the input.
        raise RuntimeError("bounded feasible region has no recoverable vertices")

    dimension_status = {1: "point", 2: "segment", 3: "polygon"}.get(len(vertices), "polygon")
    result["status"] = dimension_status
    diameter, first, second = diameter_rotating_calipers(vertices, tol=tol * tol)
    result["diameter"] = float(diameter)
    result["diameter_endpoints"] = {
        "p": result["vertices"][first],
        "q": result["vertices"][second],
    }
    diameter_center = vertices[first] + (vertices[second] - vertices[first]) / 2.0
    diameter_radius = diameter / 2.0
    max_distance = max(
        (math.sqrt(_distance_sq(point, diameter_center)) for point in vertices),
        default=0.0,
    )
    result["diameter_circle"] = {
        "center": [float(diameter_center[0]), float(diameter_center[1])],
        "radius": float(diameter_radius),
        "max_vertex_distance": float(max_distance),
        "covers": bool(max_distance <= diameter_radius + tol),
    }
    mec = minimum_enclosing_circle(vertices, tol=tol)
    result["minimum_cover_circle"] = {
        "center": [float(mec["center"][0]), float(mec["center"][1])],
        "radius": float(mec["radius"]),
        "support_indices": mec["support_indices"],
        "max_vertex_distance": float(
            max(math.sqrt(_distance_sq(point, mec["center"])) for point in vertices)
        ),
    }
    return result


def _demo_input() -> dict[str, Any]:
    return {
        "angle_error_deg": 1.0,
        "measurements": [
            {"x": 1000.0, "y": 0.0, "svd_deg": 180.0},
            {"x": -500.0, "y": 500.0 * math.sqrt(3.0), "svd_deg": 300.0},
            {"x": -500.0, "y": -500.0 * math.sqrt(3.0), "svd_deg": 60.0},
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON/CSV input path; omit when using --demo")
    parser.add_argument("--output", help="write result JSON to this path")
    parser.add_argument("--vertices-output", help="write boundary vertices as CSV")
    parser.add_argument("--angle-error-deg", type=float, help="override input angular error")
    parser.add_argument(
        "--tolerance",
        type=float,
        help="non-negative geometric tolerance in metres (default: scale-aware)",
    )
    parser.add_argument("--demo", action="store_true", help="run the six-vertex counterexample")
    args = parser.parse_args(argv)
    if args.demo:
        measurements, error_deg = parse_measurements(_demo_input())
    elif args.input:
        measurements, error_deg = load_input(args.input)
    else:
        parser.error("provide --input or --demo")
    if args.angle_error_deg is not None:
        error_deg = args.angle_error_deg
    result = solve_question1(measurements, error_deg, tolerance=args.tolerance)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if args.vertices_output:
        with Path(args.vertices_output).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("vertex_index", "x", "y"))
            for index, point in enumerate(result["vertices"]):
                writer.writerow((index, point[0], point[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
