"""Continuous second-detector selection model for Question 2.

The first detector and its measured bearing define a conservative continuous
target region.  This module then solves the robust second-detector problem
described in ``docs/问题二求解思路.md``:

* the target disk and the first-detector reception disk are represented by
  circumscribed regular polygons, so the target region remains conservative;
* the weighted DOP is integrated over the polygon by deterministic triangle
  quadrature (the nodes are integration points, not target representatives);
* the semi-infinite reception constraint is evaluated with a geometric
  separation oracle.  For the inner disk part the maximum is attained at
  polygon/circle extreme points; for the outer part the constraint is affine
  in the target position and the circle support point is also checked;
* a multi-start constrained optimization gives a numerical reference optimum;
  the requested candidate region is reported as a grid approximation of the
  continuous near-optimal level set, together with area and connected-component
  summaries.

Example::

    python solve_q2.py --demo --output ../../outputs/t2_demo_result.json

JSON input accepts either::

    {"s1": {"x": 0, "y": 0}, "svd_deg": 35}
    {"detector": [0, 0], "bearing_deg": 35}

Optional model parameters can be placed in the same object or supplied on the
command line.  See ``--help`` for the complete list.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize


TAU_DEFAULTS = (0.01, 0.03, 0.05, 0.10, 0.20)
EPSILON_DEFAULTS = (0.1, 0.2, 0.3, 0.5)
P_DEFAULTS = (1.0, 2.0, 3.0)


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite_int(value: Any, name: str) -> int:
    number = _finite_float(value, name)
    if number != round(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _as_point(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, dict):
        if "x" not in value or "y" not in value:
            raise ValueError(f"{name} must contain x and y")
        return _finite_float(value["x"], f"{name}.x"), _finite_float(
            value["y"], f"{name}.y"
        )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _finite_float(value[0], f"{name}[0]"), _finite_float(
            value[1], f"{name}[1]"
        )
    raise ValueError(f"{name} must be an object with x/y or a two-element array")


@dataclass
class Q2Config:
    """Numerical and model parameters for Question 2."""

    target_radius: float = 1800.0
    min_receive_radius: float = 1000.0
    max_receive_radius: float = 1500.0
    angle_error_deg: float = 1.0
    circle_sides: int = 360
    epsilon_w: float = 0.2
    p_w: float = 2.0
    tau: float = 0.05
    quadrature_order: int = 6
    validation_quadrature_order: int = 8
    candidate_grid: int = 101
    seed: int = 20260911
    use_global_search: bool = True
    global_maxiter: int = 24
    global_popsize: int = 7
    local_maxiter: int = 240
    feasibility_tolerance_m: float = 1e-6
    angle_sin_tolerance: float = 1e-10
    sensitivity_epsilons: tuple[float, ...] = EPSILON_DEFAULTS
    sensitivity_powers: tuple[float, ...] = P_DEFAULTS
    sensitivity_taus: tuple[float, ...] = TAU_DEFAULTS

    def validate(self) -> None:
        for name in (
            "target_radius",
            "min_receive_radius",
            "max_receive_radius",
            "angle_error_deg",
            "epsilon_w",
            "p_w",
            "tau",
            "feasibility_tolerance_m",
            "angle_sin_tolerance",
        ):
            value = _finite_float(getattr(self, name), name)
            setattr(self, name, value)
        self.circle_sides = _finite_int(self.circle_sides, "circle_sides")
        self.quadrature_order = _finite_int(self.quadrature_order, "quadrature_order")
        self.validation_quadrature_order = _finite_int(
            self.validation_quadrature_order, "validation_quadrature_order"
        )
        self.candidate_grid = _finite_int(self.candidate_grid, "candidate_grid")
        self.seed = _finite_int(self.seed, "seed")
        self.global_maxiter = _finite_int(self.global_maxiter, "global_maxiter")
        self.global_popsize = _finite_int(self.global_popsize, "global_popsize")
        self.local_maxiter = _finite_int(self.local_maxiter, "local_maxiter")
        if not isinstance(self.use_global_search, (bool, np.bool_)):
            raise ValueError("use_global_search must be boolean")
        self.use_global_search = bool(self.use_global_search)
        if self.target_radius <= 0:
            raise ValueError("target_radius must be positive")
        if not (0 < self.min_receive_radius <= self.max_receive_radius):
            raise ValueError("receive-radius bounds must satisfy 0 < min <= max")
        if not (0 < self.angle_error_deg < 90):
            raise ValueError("angle_error_deg must be in (0, 90)")
        if not (0 < self.epsilon_w < 1):
            raise ValueError("epsilon_w must be in (0, 1)")
        if self.p_w < 1:
            raise ValueError("p_w must be at least 1")
        if self.tau < 0:
            raise ValueError("tau must be non-negative")
        if self.feasibility_tolerance_m < 0:
            raise ValueError("feasibility_tolerance_m must be non-negative")
        if not (0 <= self.angle_sin_tolerance < 1):
            raise ValueError("angle_sin_tolerance must be in [0, 1)")
        if self.circle_sides < 16:
            raise ValueError("circle_sides must be at least 16")
        if self.quadrature_order < 2 or self.validation_quadrature_order < 2:
            raise ValueError("quadrature orders must be at least 2")
        if self.candidate_grid < 11:
            raise ValueError("candidate_grid must be at least 11")
        if self.global_maxiter < 1 or self.global_popsize < 2 or self.local_maxiter < 1:
            raise ValueError("optimizer iteration/population parameters are too small")

        def validate_sweep(name: str, values: Any, predicate: Any, description: str) -> tuple[float, ...]:
            if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
                raise ValueError(f"{name} must be a non-empty sequence")
            result = tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values))
            if not all(predicate(value) for value in result):
                raise ValueError(f"{name} contains values outside {description}")
            return result

        self.sensitivity_epsilons = validate_sweep(
            "sensitivity_epsilons", self.sensitivity_epsilons, lambda value: 0 < value < 1, "(0, 1)"
        )
        self.sensitivity_powers = validate_sweep(
            "sensitivity_powers", self.sensitivity_powers, lambda value: value >= 1, "[1, +inf)"
        )
        self.sensitivity_taus = validate_sweep(
            "sensitivity_taus", self.sensitivity_taus, lambda value: value >= 0, "[0, +inf)"
        )


@dataclass
class PolygonRegion:
    """A counter-clockwise convex polygon and its half-plane description."""

    vertices: np.ndarray
    a: np.ndarray
    b: np.ndarray
    area: float
    centroid: np.ndarray
    d_star: float
    incenter: np.ndarray


def _cross(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _circle_halfplanes(center: np.ndarray, radius: float, sides: int) -> list[tuple[np.ndarray, float]]:
    """Return tangent half-planes of an external regular polygon."""

    result: list[tuple[np.ndarray, float]] = []
    for index in range(sides):
        angle = 2.0 * math.pi * index / sides
        normal = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
        result.append((normal, radius + float(normal @ center)))
    return result


def _wedge_halfplanes(
    s1: np.ndarray, bearing_deg: float, error_deg: float
) -> list[tuple[np.ndarray, float]]:
    lower = math.radians(bearing_deg - error_deg)
    upper = math.radians(bearing_deg + error_deg)
    lower_direction = np.asarray((math.cos(lower), math.sin(lower)))
    upper_direction = np.asarray((math.cos(upper), math.sin(upper)))
    lower_left_normal = np.asarray((-lower_direction[1], lower_direction[0]))
    upper_left_normal = np.asarray((-upper_direction[1], upper_direction[0]))
    # cross(lower_direction, G-S1) >= 0 and
    # cross(upper_direction, G-S1) <= 0.
    return [
        (-lower_left_normal, float((-lower_left_normal) @ s1)),
        (upper_left_normal, float(upper_left_normal @ s1)),
    ]


def _external_circle_vertices(center: np.ndarray, radius: float, sides: int) -> np.ndarray:
    half_step = math.pi / sides
    vertex_radius = radius / math.cos(half_step)
    angles = 2.0 * math.pi * np.arange(sides, dtype=float) / sides + half_step
    return center + vertex_radius * np.column_stack((np.cos(angles), np.sin(angles)))


def _clip_polygon(
    polygon: np.ndarray,
    normal: np.ndarray,
    bound: float,
    tolerance: float = 1e-10,
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = float(bound - normal @ previous)
    previous_inside = previous_value >= -tolerance
    for current in polygon:
        current_value = float(bound - normal @ current)
        current_inside = current_value >= -tolerance
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if abs(denominator) > np.finfo(float).eps:
                fraction = previous_value / denominator
                output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current.copy())
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not output:
        return np.empty((0, 2), dtype=float)
    cleaned: list[np.ndarray] = []
    for point in output:
        if not cleaned or np.linalg.norm(point - cleaned[-1]) > 1e-8:
            cleaned.append(point)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= 1e-8:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def _polygon_area_centroid(vertices: np.ndarray) -> tuple[float, np.ndarray]:
    if len(vertices) < 3:
        return 0.0, np.mean(vertices, axis=0) if len(vertices) else np.zeros(2)
    next_vertices = np.roll(vertices, -1, axis=0)
    cross_values = vertices[:, 0] * next_vertices[:, 1] - next_vertices[:, 0] * vertices[:, 1]
    signed_twice_area = float(np.sum(cross_values))
    area = abs(signed_twice_area) / 2.0
    if abs(signed_twice_area) <= 1e-14:
        return area, np.mean(vertices, axis=0)
    centroid = np.sum((vertices + next_vertices) * cross_values[:, None], axis=0) / (
        3.0 * signed_twice_area
    )
    if signed_twice_area < 0:
        vertices[:] = vertices[::-1]
    return area, centroid


def _compute_incenter(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(a, axis=1)
    if len(a) == 0:
        raise ValueError("target region has no half-planes")
    result = linprog(
        np.asarray((0.0, 0.0, -1.0)),
        A_ub=np.column_stack((a, norms)),
        b_ub=b,
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if result.status != 0 or result.x is None:
        raise ValueError(f"failed to compute target-region inradius: {result.message}")
    return np.asarray(result.x[:2], dtype=float), float(result.x[2])


def build_target_region(
    s1: Sequence[float] | np.ndarray,
    bearing_deg: float,
    config: Q2Config | None = None,
) -> PolygonRegion:
    """Construct the conservative continuous target polygon ``Omega_1``."""

    config = config or Q2Config()
    config.validate()
    s1_array = np.asarray(s1, dtype=float)
    if s1_array.shape != (2,) or not np.all(np.isfinite(s1_array)):
        raise ValueError("s1 must be a finite two-dimensional point")
    bearing_deg = _finite_float(bearing_deg, "bearing_deg") % 360.0

    circle_target = _circle_halfplanes(
        np.zeros(2), config.target_radius, config.circle_sides
    )
    circle_reception = _circle_halfplanes(
        s1_array, config.max_receive_radius, config.circle_sides
    )
    wedge = _wedge_halfplanes(s1_array, bearing_deg, config.angle_error_deg)
    halfplanes = circle_target + circle_reception + wedge
    polygon = _external_circle_vertices(
        np.zeros(2), config.target_radius, config.circle_sides
    )
    # Clipping an external target polygon by every half-plane preserves the
    # continuous half-plane model while avoiding pairwise line intersections.
    for normal, bound in circle_reception + wedge:
        polygon = _clip_polygon(polygon, normal, bound)
        if len(polygon) == 0:
            raise ValueError("the first measurement yields an empty target region")
    area, centroid = _polygon_area_centroid(polygon)
    if area <= 1e-8:
        raise ValueError("the first measurement yields a zero-area target region")
    a = np.asarray([entry[0] for entry in halfplanes], dtype=float)
    b = np.asarray([entry[1] for entry in halfplanes], dtype=float)
    incenter, d_star = _compute_incenter(a, b)
    if d_star <= 1e-8:
        raise ValueError("target region has no positive inradius")
    return PolygonRegion(
        vertices=polygon,
        a=a,
        b=b,
        area=float(area),
        centroid=np.asarray(centroid, dtype=float),
        d_star=float(d_star),
        incenter=incenter,
    )


def _triangle_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre/Duffy quadrature on the unit reference triangle."""

    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    nodes_1d = (nodes_1d + 1.0) / 2.0
    weights_1d = weights_1d / 2.0
    barycentric: list[tuple[float, float, float]] = []
    weights: list[float] = []
    for u, wu in zip(nodes_1d, weights_1d):
        for v, wv in zip(nodes_1d, weights_1d):
            # Reference coordinates are (u, (1-u)*v); the Jacobian is 1-u.
            barycentric.append(((1.0 - u) * (1.0 - v), u, (1.0 - u) * v))
            weights.append(float(wu * wv * (1.0 - u)))
    # The temporary tuple above is (lambda_0, lambda_1, lambda_2) after
    # interpreting the first coordinate as 1-u and the second as u.
    return np.asarray(barycentric, dtype=float), np.asarray(weights, dtype=float)


def _integration_nodes(
    vertices: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray, float]:
    barycentric, reference_weights = _triangle_quadrature(order)
    center = np.mean(vertices, axis=0)
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        determinant = abs(_cross(second - center, first - center))
        if determinant <= 1e-14:
            continue
        triangle = (
            barycentric[:, 0, None] * center
            + barycentric[:, 1, None] * first
            + barycentric[:, 2, None] * second
        )
        points.append(triangle)
        weights.append(determinant * reference_weights)
    if not points:
        raise ValueError("failed to triangulate target polygon")
    points_array = np.vstack(points)
    weights_array = np.concatenate(weights)
    return points_array, weights_array, float(np.sum(weights_array))


def _point_inside(region: PolygonRegion, point: np.ndarray, tolerance: float = 1e-8) -> bool:
    slack = region.b - region.a @ point
    scale = max(1.0, float(np.max(np.abs(region.b))))
    return bool(np.all(slack >= -tolerance * scale))


def _segment_circle_intersections(
    first: np.ndarray, second: np.ndarray, center: np.ndarray, radius: float
) -> list[np.ndarray]:
    direction = second - first
    offset = first - center
    aa = float(direction @ direction)
    if aa <= 1e-20:
        return []
    bb = 2.0 * float(offset @ direction)
    cc = float(offset @ offset) - radius * radius
    discriminant = bb * bb - 4.0 * aa * cc
    if discriminant < -1e-8 * max(1.0, bb * bb, abs(aa * cc)):
        return []
    discriminant = max(0.0, discriminant)
    root = math.sqrt(discriminant)
    result: list[np.ndarray] = []
    for parameter in ((-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)):
        if -1e-9 <= parameter <= 1.0 + 1e-9:
            clipped = min(1.0, max(0.0, parameter))
            point = first + clipped * direction
            if not result or np.linalg.norm(point - result[-1]) > 1e-7:
                result.append(point)
    return result


class Question2Model:
    """Numerical model with reusable geometry, quadrature and objective state."""

    def __init__(
        self,
        s1: Sequence[float] | np.ndarray,
        bearing_deg: float,
        config: Q2Config,
        quadrature_order: int | None = None,
    ) -> None:
        self.s1 = np.asarray(s1, dtype=float)
        self.bearing_deg = float(bearing_deg) % 360.0
        self.config = config
        self.config.validate()
        self.region = build_target_region(self.s1, self.bearing_deg, config)
        self._quadrature_cache: dict[int, dict[str, Any]] = {}
        order = config.quadrature_order if quadrature_order is None else _finite_int(
            quadrature_order, "quadrature_order"
        )
        if order < 2:
            raise ValueError("quadrature order must be at least 2")
        self.quadrature_order = order
        default_data = self._quadrature_data(order)
        self.nodes = default_data["nodes"]
        self.quad_weights = default_data["quad_weights"]
        self.quad_area = default_data["quad_area"]
        self.depths = default_data["depths"]
        self.depth_quantiles = default_data["depth_quantiles"]
        self.r1_vectors = default_data["r1_vectors"]
        self.r1_sq = default_data["r1_sq"]
        self.r1 = default_data["r1"]
        self.search_center = self.region.centroid.copy()
        self.search_radius = max(
            config.min_receive_radius,
            float(np.linalg.norm(self.search_center - self.s1)),
        )
        self.search_bounds = [
            (
                float(self.search_center[0] - self.search_radius),
                float(self.search_center[0] + self.search_radius),
            ),
            (
                float(self.search_center[1] - self.search_radius),
                float(self.search_center[1] + self.search_radius),
            ),
        ]

    def _quadrature_data(self, order: int) -> dict[str, Any]:
        """Build and cache integration data for one quadrature order."""

        order = _finite_int(order, "quadrature_order")
        if order < 2:
            raise ValueError("quadrature order must be at least 2")
        cached = self._quadrature_cache.get(order)
        if cached is not None:
            return cached
        nodes, quad_weights, quad_area = _integration_nodes(self.region.vertices, order)
        slacks = self.region.b[:, None] - self.region.a @ nodes.T
        depths = np.maximum(
            0.0,
            np.min(slacks / np.linalg.norm(self.region.a, axis=1)[:, None], axis=0),
        )
        depth_quantiles = np.minimum(1.0, depths / self.region.d_star)
        r1_vectors = nodes - self.s1[None, :]
        r1_sq = np.einsum("ij,ij->i", r1_vectors, r1_vectors)
        data = {
            "nodes": nodes,
            "quad_weights": quad_weights,
            "quad_area": float(quad_area),
            "depths": depths,
            "depth_quantiles": depth_quantiles,
            "r1_vectors": r1_vectors,
            "r1_sq": r1_sq,
            "r1": np.sqrt(r1_sq),
        }
        self._quadrature_cache[order] = data
        return data

    def _weights(
        self,
        epsilon_w: float | None = None,
        p_w: float | None = None,
        depth_quantiles: np.ndarray | None = None,
    ) -> np.ndarray:
        epsilon = self.config.epsilon_w if epsilon_w is None else float(epsilon_w)
        power = self.config.p_w if p_w is None else float(p_w)
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon_w must be in (0, 1)")
        if power < 1.0:
            raise ValueError("p_w must be at least 1")
        quantiles = self.depth_quantiles if depth_quantiles is None else depth_quantiles
        return epsilon + (1.0 - epsilon) * np.power(quantiles, power)

    def _circle_boundary_candidates(self, point: np.ndarray) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        vertices = self.region.vertices
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            result.extend(
                _segment_circle_intersections(
                    first, second, self.s1, self.config.min_receive_radius
                )
            )
        direction = point - self.s1
        norm = float(np.linalg.norm(direction))
        if norm > 1e-12:
            # This is simultaneously the farthest point for the inner
            # radius-1000 disk and the support point for the outer affine
            # constraint.
            support = self.s1 - self.config.min_receive_radius * direction / norm
            if _point_inside(self.region, support, 2e-8):
                result.append(support)
        return result

    def _reception_candidates(self, point: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        inner: list[np.ndarray] = []
        outer: list[np.ndarray] = []
        radius = self.config.min_receive_radius
        for vertex in self.region.vertices:
            distance = float(np.linalg.norm(vertex - self.s1))
            if distance <= radius + 1e-7:
                inner.append(vertex)
            if distance >= radius - 1e-7:
                outer.append(vertex)
        intersections = self._circle_boundary_candidates(point)
        for candidate in intersections:
            inner.append(candidate)
            outer.append(candidate)
        # Remove duplicate points while preserving deterministic order.
        def unique(points: list[np.ndarray]) -> list[np.ndarray]:
            output: list[np.ndarray] = []
            for candidate in points:
                if not _point_inside(self.region, candidate, 5e-7):
                    continue
                if not any(np.linalg.norm(candidate - prior) <= 1e-7 for prior in output):
                    output.append(candidate)
            return output

        return unique(inner), unique(outer)

    def receive_h(self, point: Sequence[float] | np.ndarray) -> float:
        """Return the exact polygon/circle separation value H(P), up to roundoff."""

        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
            return math.inf
        inner, outer = self._reception_candidates(point_array)
        radius_sq = self.config.min_receive_radius**2
        values: list[float] = []
        for target in inner:
            distance_sq = float(np.sum((point_array - target) ** 2))
            values.append(distance_sq - radius_sq)
        for target in outer:
            distance_sq = float(np.sum((point_array - target) ** 2))
            target_radius_sq = float(np.sum((target - self.s1) ** 2))
            values.append(distance_sq - target_radius_sq)
        if not values:
            # A full-dimensional polygon always has at least one of the two
            # parts; this branch is only a defensive numerical fallback.
            target = self.region.centroid
            return float(
                np.sum((point_array - target) ** 2)
                - max(radius_sq, np.sum((target - self.s1) ** 2))
            )
        return float(max(values))

    def reception_h_tolerance(self) -> float:
        """Return the numerical H tolerance used for feasibility acceptance.

        ``H`` is a squared-distance residual (m^2), while the public
        configuration keeps the more interpretable distance-scale parameter
        ``feasibility_tolerance_m``.  The search radius converts that parameter
        to the residual scale used by the optimizer and grid classifier.
        """

        return float(self.config.feasibility_tolerance_m * self.search_radius)

    def reception_status(self, point: Sequence[float] | np.ndarray) -> dict[str, Any]:
        """Return strict and tolerance-aware reception feasibility flags."""

        h_value = float(self.receive_h(point))
        finite = math.isfinite(h_value)
        h_tolerance = self.reception_h_tolerance()
        return {
            "h": h_value,
            "h_tolerance": h_tolerance,
            "strict_feasible": bool(finite and h_value <= 0.0),
            "feasible_with_tolerance": bool(finite and h_value <= h_tolerance),
        }

    def _line_intersects_target(self, point: np.ndarray) -> bool:
        """Check the DOP singularity condition for the line S1--P."""

        direction = point - self.s1
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-10:
            return True
        direction /= norm
        lower, upper = -math.inf, math.inf
        scale = max(1.0, float(np.max(np.abs(self.region.b))))
        coefficient_tolerance = 1e-13
        for normal, bound in zip(self.region.a, self.region.b):
            coefficient = float(normal @ direction)
            residual = float(bound - normal @ self.s1)
            if abs(coefficient) <= coefficient_tolerance:
                if residual < -1e-9 * scale:
                    return False
                continue
            candidate = residual / coefficient
            if coefficient > 0:
                upper = min(upper, candidate)
            else:
                lower = max(lower, candidate)
        if lower > upper + 1e-10:
            return False
        if lower < -1e-8 or upper > 1e-8 or upper - lower > 1e-8:
            return True
        # The line only touches the polygon at S1.  A direction in the local
        # tangent cone still creates a logarithmic DOP singularity at the
        # wedge apex, so reject it explicitly.
        active = [
            normal
            for normal, bound in zip(self.region.a, self.region.b)
            if abs(float(bound - normal @ self.s1)) <= 1e-7 * scale
        ]
        if not active:
            return True
        for signed_direction in (direction, -direction):
            if all(float(normal @ signed_direction) <= 1e-9 for normal in active):
                return True
        return False

    def geometry_is_finite(self, point: Sequence[float] | np.ndarray) -> bool:
        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
            return False
        return not self._line_intersects_target(point_array)

    def objective(
        self,
        point: Sequence[float] | np.ndarray,
        epsilon_w: float | None = None,
        p_w: float | None = None,
        quadrature_order: int | None = None,
    ) -> float:
        """Evaluate the weighted position-error DOP J(P)."""

        point_array = np.asarray(point, dtype=float)
        if point_array.shape != (2,) or not np.all(np.isfinite(point_array)):
            return math.inf
        if not self.geometry_is_finite(point_array):
            return math.inf
        order = self.quadrature_order if quadrature_order is None else _finite_int(
            quadrature_order, "quadrature_order"
        )
        data = self._quadrature_data(order)
        nodes = data["nodes"]
        quad_weights = data["quad_weights"]
        r1_vectors = data["r1_vectors"]
        r1_sq = data["r1_sq"]
        r1 = data["r1"]
        vectors_2 = nodes - point_array[None, :]
        r2_sq = np.einsum("ij,ij->i", vectors_2, vectors_2)
        r2 = np.sqrt(r2_sq)
        determinants = np.abs(
            r1_vectors[:, 0] * vectors_2[:, 1]
            - r1_vectors[:, 1] * vectors_2[:, 0]
        )
        denominator = r1 * r2
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sine = np.divide(determinants, denominator, out=np.zeros_like(denominator), where=denominator > 0)
            if np.any(sine <= self.config.angle_sin_tolerance):
                return math.inf
            dop = r1 * r2 * np.sqrt(r1_sq + r2_sq) / determinants
        if not np.all(np.isfinite(dop)):
            return math.inf
        weights = self._weights(epsilon_w, p_w, data["depth_quantiles"])
        normalization = float(np.sum(quad_weights * weights))
        if normalization <= 0:
            return math.inf
        value = float(np.sum(quad_weights * weights * dop) / normalization)
        return value if math.isfinite(value) else math.inf

    def constrained_objective(self, point: Sequence[float] | np.ndarray) -> float:
        """Finite objective used by global and local numerical optimizers.

        The strict public objective intentionally returns ``+inf`` at a
        collinearity singularity.  Derivative-based optimizers cannot finite-
        difference an infinite value, so this wrapper substitutes a large
        finite barrier while retaining the exact objective for all valid
        points.
        """

        value = self.objective(point)
        violation = max(0.0, self.receive_h(point)) / max(1.0, self.search_radius**2)
        if not math.isfinite(value):
            value = 1.0e12
        return float(value + 2.0e7 * violation * violation + 2.0e5 * violation)

    def _candidate_starts(self, rng: np.random.Generator) -> list[np.ndarray]:
        starts: list[np.ndarray] = [self.region.centroid.copy(), self.region.incenter.copy()]
        radii = (0.25, 0.5, 0.75, 1.0)
        for fraction in radii:
            for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
                starts.append(
                    self.s1
                    + fraction
                    * self.search_radius
                    * np.asarray((math.cos(angle), math.sin(angle)))
                )
        for _ in range(18):
            starts.append(
                np.asarray(
                    [
                        rng.uniform(*self.search_bounds[0]),
                        rng.uniform(*self.search_bounds[1]),
                    ]
                )
            )
        clipped: list[np.ndarray] = []
        for start in starts:
            point = np.asarray(start, dtype=float)
            point = np.clip(
                point,
                np.asarray([self.search_bounds[0][0], self.search_bounds[1][0]]),
                np.asarray([self.search_bounds[0][1], self.search_bounds[1][1]]),
            )
            if not any(np.linalg.norm(point - previous) <= 1e-9 for previous in clipped):
                clipped.append(point)
        return clipped

    def optimize(self) -> dict[str, Any]:
        """Run global penalized search followed by constrained local polishing."""

        rng = np.random.default_rng(self.config.seed)
        candidates: list[tuple[float, np.ndarray, str]] = []
        global_result: Any = None
        if self.config.use_global_search:
            global_result = differential_evolution(
                self.constrained_objective,
                self.search_bounds,
                seed=self.config.seed,
                maxiter=self.config.global_maxiter,
                popsize=self.config.global_popsize,
                polish=False,
                updating="immediate",
                workers=1,
                tol=1e-5,
            )
            global_point = np.asarray(global_result.x, dtype=float)
            global_value = self.objective(global_point)
            global_h = self.receive_h(global_point)
            global_feasible = math.isfinite(global_value) and (
                global_h <= self.reception_h_tolerance()
            )
            if global_feasible:
                candidates.append((global_value, global_point, "global"))
        starts = self._candidate_starts(rng)
        if global_result is not None:
            starts.insert(0, np.asarray(global_result.x, dtype=float))
        scale = max(1.0, self.search_radius**2)

        def feasibility_constraint(point: np.ndarray) -> float:
            return -self.receive_h(point) / scale

        local_results: list[dict[str, Any]] = []
        for start in starts:
            result = minimize(
                self.constrained_objective,
                np.asarray(start, dtype=float),
                method="SLSQP",
                bounds=self.search_bounds,
                constraints=[{"type": "ineq", "fun": feasibility_constraint}],
                options={"maxiter": self.config.local_maxiter, "ftol": 1e-7, "disp": False},
            )
            point = np.asarray(result.x, dtype=float)
            value = self.objective(point)
            h_value = self.receive_h(point)
            feasible = math.isfinite(value) and h_value <= self.reception_h_tolerance()
            local_results.append(
                {
                    "success": bool(result.success),
                    "message": str(result.message),
                    "point": point,
                    "objective": value,
                    "objective_with_penalty": float(self.constrained_objective(point)),
                    "reception_h": h_value,
                    "reception_h_tolerance": self.reception_h_tolerance(),
                    "strict_feasible": bool(math.isfinite(h_value) and h_value <= 0.0),
                    "feasible_with_tolerance": bool(
                        math.isfinite(h_value)
                        and h_value <= self.reception_h_tolerance()
                    ),
                    "feasible": feasible,
                    "iterations": int(getattr(result, "nit", 0) or 0),
                }
            )
            if feasible:
                candidates.append((value, point, "local"))
        if not any(
            math.isfinite(value)
            and self.receive_h(point)
            <= self.reception_h_tolerance()
            for value, point, _ in candidates
        ):
            # A finite point can be hard for SLSQP to reach if most starts lie
            # on a bearing ray.  Probe directions outside that ray from S1;
            # S1 itself is reception-feasible but geometrically singular.
            fallback_angles = [
                self.bearing_deg + 90.0,
                self.bearing_deg - 90.0,
                self.bearing_deg + 45.0,
                self.bearing_deg - 45.0,
            ]
            fallback_fractions = (1e-4, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 0.75, 1.0)
            for angle_deg in fallback_angles:
                direction = np.asarray(
                    (
                        math.cos(math.radians(angle_deg)),
                        math.sin(math.radians(angle_deg)),
                    ),
                    dtype=float,
                )
                for fraction in fallback_fractions:
                    raw_point = self.s1 + fraction * self.search_radius * direction
                    point = np.clip(
                        raw_point,
                        np.asarray([self.search_bounds[0][0], self.search_bounds[1][0]]),
                        np.asarray([self.search_bounds[0][1], self.search_bounds[1][1]]),
                    )
                    value = self.objective(point)
                    h_value = self.receive_h(point)
                    if (
                        math.isfinite(value)
                        and h_value <= self.reception_h_tolerance()
                    ):
                        candidates.append((value, point, "direction-fallback"))
                        break
                if any(
                    math.isfinite(value)
                    and self.receive_h(candidate)
                    <= self.reception_h_tolerance()
                    for value, candidate, _ in candidates
                ):
                    break
        feasible_candidates = [
            item
            for item in candidates
            if math.isfinite(item[0]) and self.receive_h(item[1]) <= self.reception_h_tolerance()
        ]
        if not feasible_candidates:
            raise RuntimeError(
                "no finite reception-feasible second detector was found; "
                "increase circle_sides/search iterations or inspect the input bearing"
            )
        best_value, best_point, source = min(feasible_candidates, key=lambda item: item[0])
        best_point = np.asarray(best_point, dtype=float)
        bearing_from_s1 = math.degrees(math.atan2(best_point[1] - self.s1[1], best_point[0] - self.s1[0])) % 360.0
        bearing_difference = ((bearing_from_s1 - self.bearing_deg + 180.0) % 360.0) - 180.0
        return {
            "point": best_point,
            "j_star": float(best_value),
            "reception_h": float(self.receive_h(best_point)),
            "reception_h_tolerance": self.reception_h_tolerance(),
            "strict_feasible": bool(
                math.isfinite(self.receive_h(best_point))
                and self.receive_h(best_point) <= 0.0
            ),
            "feasible_with_tolerance": bool(
                math.isfinite(self.receive_h(best_point))
                and self.receive_h(best_point) <= self.reception_h_tolerance()
            ),
            "distance_from_s1": float(np.linalg.norm(best_point - self.s1)),
            "bearing_from_s1_deg": float(bearing_from_s1),
            "bearing_difference_deg": float(bearing_difference),
            "source": source,
            "global": None
            if global_result is None
            else {
                "objective_with_penalty": float(global_result.fun),
                "objective": float(self.objective(global_result.x)),
                "reception_h": float(self.receive_h(global_result.x)),
                "reception_h_tolerance": self.reception_h_tolerance(),
                "strict_feasible": bool(
                    math.isfinite(self.receive_h(global_result.x))
                    and self.receive_h(global_result.x) <= 0.0
                ),
                "feasible_with_tolerance": bool(
                    math.isfinite(self.receive_h(global_result.x))
                    and self.receive_h(global_result.x) <= self.reception_h_tolerance()
                ),
                "feasible": bool(
                    math.isfinite(self.objective(global_result.x))
                    and self.receive_h(global_result.x) <= self.reception_h_tolerance()
                ),
                "point": [float(value) for value in global_result.x],
                "iterations": int(global_result.nit),
                "function_evaluations": int(global_result.nfev),
            },
            "local_runs": local_results,
        }

    def candidate_region(
        self,
        optimum: dict[str, Any],
        tau: float | None = None,
        grid_size: int | None = None,
        include_values: bool = False,
    ) -> dict[str, Any]:
        """Approximate the continuous near-optimal level set on a regular grid."""

        tau_value = self.config.tau if tau is None else _finite_float(tau, "tau")
        if tau_value < 0:
            raise ValueError("tau must be non-negative")
        count = (
            self.config.candidate_grid
            if grid_size is None
            else _finite_int(grid_size, "grid_size")
        )
        if count < 11:
            raise ValueError("candidate grid must contain at least 11 points per axis")
        try:
            optimum_point = np.asarray(optimum["point"], dtype=float)
            optimum_j = _finite_float(optimum["j_star"], "optimum.j_star")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("optimum must contain a finite point and j_star") from exc
        if optimum_point.shape != (2,) or not np.all(np.isfinite(optimum_point)):
            raise ValueError("optimum.point must be a finite two-dimensional point")
        threshold = (1.0 + tau_value) * optimum_j
        x_edges = np.linspace(self.search_bounds[0][0], self.search_bounds[0][1], count + 1)
        y_edges = np.linspace(self.search_bounds[1][0], self.search_bounds[1][1], count + 1)
        x_values = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_values = (y_edges[:-1] + y_edges[1:]) / 2.0
        mask = np.zeros((count, count), dtype=bool)
        values = np.full((count, count), np.nan, dtype=float)
        for iy, y_value in enumerate(y_values):
            for ix, x_value in enumerate(x_values):
                point = np.asarray((x_value, y_value), dtype=float)
                if self.receive_h(point) > self.reception_h_tolerance():
                    continue
                if not self.geometry_is_finite(point):
                    continue
                value = self.objective(point)
                if not math.isfinite(value):
                    continue
                values[iy, ix] = value
                mask[iy, ix] = value <= threshold
        step_x = float(x_edges[1] - x_edges[0])
        step_y = float(y_edges[1] - y_edges[0])
        area_per_cell = abs(step_x * step_y)
        components = _connected_components(mask)
        component_stats: list[dict[str, Any]] = []
        accepted_indices = np.argwhere(mask)
        for component in components:
            indices = np.asarray(component, dtype=int)
            yy = indices[:, 0]
            xx = indices[:, 1]
            points = np.column_stack((x_values[xx], y_values[yy]))
            distances = np.linalg.norm(points - self.s1[None, :], axis=1)
            component_stats.append(
                {
                    "cell_count": int(len(indices)),
                    "area": float(len(indices) * area_per_cell),
                    "bbox": {
                        "x_min": float(np.min(x_values[xx])),
                        "x_max": float(np.max(x_values[xx])),
                        "y_min": float(np.min(y_values[yy])),
                        "y_max": float(np.max(y_values[yy])),
                    },
                    "distance_from_s1_min": float(np.min(distances)),
                    "distance_from_s1_max": float(np.max(distances)),
                }
            )
        if len(accepted_indices):
            accepted_points = np.column_stack(
                (x_values[accepted_indices[:, 1]], y_values[accepted_indices[:, 0]])
            )
            distances = np.linalg.norm(accepted_points - self.s1[None, :], axis=1)
            distance_min = float(np.min(distances))
            distance_max = float(np.max(distances))
        else:
            distance_min = distance_max = math.nan
        result: dict[str, Any] = {
            "tau": tau_value,
            "threshold": float(threshold),
            "grid_shape": [count, count],
            "x_min": float(x_edges[0]),
            "x_max": float(x_edges[-1]),
            "y_min": float(y_edges[0]),
            "y_max": float(y_edges[-1]),
            "x_step": step_x,
            "y_step": step_y,
            "area": float(len(accepted_indices) * area_per_cell),
            "distance_from_s1_min": distance_min,
            "distance_from_s1_max": distance_max,
            "component_count": len(component_stats),
            "components": component_stats,
            "sample_x": x_values.astype(float).tolist(),
            "sample_y": y_values.astype(float).tolist(),
            "optimum_point": [float(value) for value in optimum_point],
            "optimum_j": float(optimum_j),
            "reception_h_tolerance": float(self.reception_h_tolerance()),
            "feasibility_rule": (
                "A grid cell center is accepted only when H(P) <= "
                "reception_h_tolerance, the objective is finite, and the "
                "S1--P line does not intersect the target polygon."
            ),
            # Row iy corresponds to the cell centered at sample_y[iy], and
            # each value is a 0/1 cell-center classification.  This is
            # intentionally a grid approximation of the continuous level set;
            # the optimum is reported separately and is never injected into
            # an infeasible or singular grid cell.
            "mask": mask.astype(np.uint8).tolist(),
        }
        if include_values:
            result["objective_grid"] = [
                [None if not math.isfinite(float(value)) else float(value) for value in row]
                for row in values
            ]
        return result

    def quadrature_check(self, point: Sequence[float], high_order: int | None = None) -> dict[str, Any]:
        order = (
            self.config.validation_quadrature_order
            if high_order is None
            else _finite_int(high_order, "validation_quadrature_order")
        )
        if order < 2:
            raise ValueError("validation quadrature order must be at least 2")
        high_model = Question2Model(
            self.s1,
            self.bearing_deg,
            self.config,
            quadrature_order=order,
        )
        low_value = self.objective(point)
        high_value = high_model.objective(point)
        return {
            "order_low": int(self.quadrature_order),
            "order_high": int(order),
            "j_low": float(low_value),
            "j_high": float(high_value),
            "absolute_difference": float(abs(low_value - high_value)),
            "relative_difference": float(abs(low_value - high_value) / max(1.0, abs(high_value))),
        }

    def sensitivity(self, grid_size: int | None = None) -> dict[str, Any]:
        """Run the documented epsilon/power/tau sensitivity sweep."""

        rows: list[dict[str, Any]] = []
        compact_grid = grid_size or min(self.config.candidate_grid, 61)
        reference_eps, reference_power = self.config.epsilon_w, self.config.p_w
        for epsilon in self.config.sensitivity_epsilons:
            for power in self.config.sensitivity_powers:
                local_config = Q2Config(**asdict(self.config))
                local_config.epsilon_w = float(epsilon)
                local_config.p_w = float(power)
                local_config.use_global_search = False
                local_model = Question2Model(
                    self.s1,
                    self.bearing_deg,
                    local_config,
                    quadrature_order=self.config.quadrature_order,
                )
                optimum = local_model.optimize()
                point = np.asarray(optimum["point"], dtype=float)
                reference_j = self.objective(point, reference_eps, reference_power)
                areas: dict[str, float] = {}
                # Candidate masks are computed with the same continuous model
                # and only a smaller reporting grid to keep the sweep tractable.
                for tau in self.config.sensitivity_taus:
                    region = local_model.candidate_region(optimum, tau=tau, grid_size=compact_grid)
                    areas[f"{float(tau):g}"] = float(region["area"])
                rows.append(
                    {
                        "epsilon_w": float(epsilon),
                        "p_w": float(power),
                        "point": [float(point[0]), float(point[1])],
                        "j_star": float(optimum["j_star"]),
                        "j_under_reference_weight": float(reference_j),
                        "distance_from_s1": float(np.linalg.norm(point - self.s1)),
                        "bearing_difference_deg": float(optimum["bearing_difference_deg"]),
                        "candidate_area_by_tau": areas,
                    }
                )
        return {
            "epsilon_values": [float(value) for value in self.config.sensitivity_epsilons],
            "p_values": [float(value) for value in self.config.sensitivity_powers],
            "tau_values": [float(value) for value in self.config.sensitivity_taus],
            "rows": rows,
        }


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for iy in range(height):
        for ix in range(width):
            if not mask[iy, ix] or visited[iy, ix]:
                continue
            stack = [(iy, ix)]
            visited[iy, ix] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    return components


def parse_input(source: Any) -> tuple[np.ndarray, float, Q2Config]:
    """Parse JSON-like Question 2 input and return ``(S1, bearing, config)``."""

    if not isinstance(source, dict):
        raise ValueError("Question 2 input must be a JSON object")
    if "s1" in source:
        s1 = np.asarray(_as_point(source["s1"], "s1"), dtype=float)
    elif "detector" in source:
        s1 = np.asarray(_as_point(source["detector"], "detector"), dtype=float)
    elif "x1" in source and "y1" in source:
        s1 = np.asarray(
            (_finite_float(source["x1"], "x1"), _finite_float(source["y1"], "y1")),
            dtype=float,
        )
    else:
        raise ValueError("input must contain s1/detector or x1 and y1")
    bearing_keys = ("svd_deg", "bearing_deg", "theta_deg", "theta", "angle_deg")
    bearing_key = next((key for key in bearing_keys if key in source), None)
    if bearing_key is None:
        raise ValueError("input must contain svd_deg or bearing_deg")
    bearing = _finite_float(source[bearing_key], bearing_key) % 360.0
    config = Q2Config()
    # Keep the JSON names identical to the dataclass fields.  In particular,
    # expose optimizer, numerical-protection, and sensitivity parameters so a
    # paper run can be reproduced without editing the solver source.
    config_fields = set(asdict(config))
    for source_key in config_fields:
        if source_key in source:
            setattr(config, source_key, source[source_key])
    config.validate()
    return s1, bearing, config


def load_input(path: str | Path) -> tuple[np.ndarray, float, Q2Config]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return parse_input(json.load(handle))


def solve_question2(
    s1: Sequence[float] | np.ndarray,
    bearing_deg: float,
    config: Q2Config | None = None,
    *,
    run_sensitivity: bool = False,
    include_grid_values: bool = False,
) -> dict[str, Any]:
    """Solve Question 2 and return a JSON-serialisable result dictionary."""

    config = config or Q2Config()
    config.validate()
    model = Question2Model(s1, bearing_deg, config)
    optimum = model.optimize()
    candidate = model.candidate_region(
        optimum,
        tau=config.tau,
        grid_size=config.candidate_grid,
        include_values=include_grid_values,
    )
    quadrature = model.quadrature_check(optimum["point"])
    result: dict[str, Any] = {
        "model": "question_2_continuous_robust_dop",
        "input": {
            "s1": [float(model.s1[0]), float(model.s1[1])],
            "svd_deg": float(model.bearing_deg),
        },
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in asdict(config).items()
        },
        "target_region": {
            "vertices": model.region.vertices.astype(float).tolist(),
            "halfplane_count": int(len(model.region.a)),
            "area": float(model.region.area),
            "centroid": model.region.centroid.astype(float).tolist(),
            "incenter": model.region.incenter.astype(float).tolist(),
            "inradius": float(model.region.d_star),
            "circle_approximation": "external circumscribed regular polygons",
        },
        "reception_constraint": {
            "description": (
                "H(P)=sup_G [||P-G||^2-max("
                f"{config.min_receive_radius:g}^2,||G-S1||^2)] <= 0"
            ),
            "min_receive_radius": float(config.min_receive_radius),
            "search_center": model.search_center.astype(float).tolist(),
            "search_radius": float(model.search_radius),
            "search_bounds": [[float(low), float(high)] for low, high in model.search_bounds],
            "h_at_optimum": float(optimum["reception_h"]),
            "h_tolerance": float(model.reception_h_tolerance()),
            "strict_feasible_at_optimum": bool(optimum["strict_feasible"]),
            "feasible_with_tolerance_at_optimum": bool(
                optimum["feasible_with_tolerance"]
            ),
            "tolerance_semantics": (
                "feasibility_tolerance_m is converted to an H residual tolerance "
                "by multiplying by search_radius; H itself has squared-distance units"
            ),
        },
        "quadrature": {
            "order": int(config.quadrature_order),
            "node_count": int(len(model.nodes)),
            "integrated_area": float(model.quad_area),
            "target_polygon_area": float(model.region.area),
            "area_absolute_difference": float(abs(model.quad_area - model.region.area)),
            "check_at_optimum": quadrature,
        },
        "optimum": {
            key: (
                [float(value[0]), float(value[1])]
                if key == "point"
                else value
            )
            for key, value in optimum.items()
            if key != "local_runs"
        },
        "candidate_region": candidate,
        "assumptions": [
            "The target and first-reception disks are replaced by external circumscribed polygons; this keeps the feasible target region conservative.",
            (
                "The first bearing error is the closed interval "
                f"[svd-{config.angle_error_deg:g}°, "
                f"svd+{config.angle_error_deg:g}°]."
            ),
            (
                f"The {config.min_receive_radius:g} m lower reception radius is "
                "used in the robust minimum-radius constraint exactly as "
                "specified in the model document."
            ),
            "A detector position whose S1--P line meets the target polygon is assigned J=+infinity to exclude the logarithmic collinearity singularity.",
            "The reported candidate area is a regular-grid approximation of the continuous near-optimal level set; its threshold is continuous J(P)<= (1+tau)J*.",
        ],
    }
    if run_sensitivity:
        result["sensitivity"] = model.sensitivity()
    return result


def _demo_input() -> dict[str, Any]:
    return {
        "s1": {"x": 0.0, "y": 0.0},
        "svd_deg": 32.0,
        "angle_error_deg": 1.0,
        "epsilon_w": 0.2,
        "p_w": 2.0,
        "tau": 0.05,
    }


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Question 2 JSON input path")
    parser.add_argument("--output", help="write result JSON to this path")
    parser.add_argument("--demo", action="store_true", help="run a deterministic demonstration case")
    parser.add_argument("--s1-x", type=float, help="override first detector x")
    parser.add_argument("--s1-y", type=float, help="override first detector y")
    parser.add_argument("--svd-deg", type=float, help="override first measured bearing")
    parser.add_argument("--target-radius", type=float)
    parser.add_argument("--min-receive-radius", type=float)
    parser.add_argument("--max-receive-radius", type=float)
    parser.add_argument("--angle-error-deg", type=float, help="bearing error half-width")
    parser.add_argument("--epsilon-w", type=float, help="boundary base weight")
    parser.add_argument("--p-w", type=float, help="boundary-depth power")
    parser.add_argument("--tau", type=float, help="relative near-optimal tolerance")
    parser.add_argument("--circle-sides", type=int, help="sides used for external circle polygons")
    parser.add_argument("--quadrature-order", type=int, help="Gauss-Duffy order per triangle axis")
    parser.add_argument("--validation-quadrature-order", type=int)
    parser.add_argument("--candidate-grid", type=int, help="candidate-region grid points per axis")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--global-maxiter", type=int)
    parser.add_argument("--global-popsize", type=int)
    parser.add_argument("--local-maxiter", type=int)
    parser.add_argument("--feasibility-tolerance-m", type=float)
    parser.add_argument("--angle-sin-tolerance", type=float)
    parser.add_argument(
        "--global-search",
        dest="use_global_search",
        action="store_true",
        default=None,
        help="enable differential-evolution global search",
    )
    parser.add_argument(
        "--no-global",
        dest="use_global_search",
        action="store_false",
        help="skip differential-evolution global search",
    )
    parser.add_argument("--sensitivity", action="store_true", help="run epsilon/power/tau sensitivity sweep")
    parser.add_argument("--include-grid-values", action="store_true", help="include objective values for every grid cell")
    args = parser.parse_args(argv)
    if args.demo:
        s1, bearing, config = parse_input(_demo_input())
    elif args.input:
        s1, bearing, config = load_input(args.input)
    else:
        parser.error("provide --input or --demo")
    overrides = {
        "target_radius": args.target_radius,
        "min_receive_radius": args.min_receive_radius,
        "max_receive_radius": args.max_receive_radius,
        "angle_error_deg": args.angle_error_deg,
        "epsilon_w": args.epsilon_w,
        "p_w": args.p_w,
        "tau": args.tau,
        "circle_sides": args.circle_sides,
        "quadrature_order": args.quadrature_order,
        "validation_quadrature_order": args.validation_quadrature_order,
        "candidate_grid": args.candidate_grid,
        "seed": args.seed,
        "global_maxiter": args.global_maxiter,
        "global_popsize": args.global_popsize,
        "local_maxiter": args.local_maxiter,
        "feasibility_tolerance_m": args.feasibility_tolerance_m,
        "angle_sin_tolerance": args.angle_sin_tolerance,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    if args.s1_x is not None:
        s1[0] = args.s1_x
    if args.s1_y is not None:
        s1[1] = args.s1_y
    if args.svd_deg is not None:
        bearing = args.svd_deg
    if args.use_global_search is not None:
        config.use_global_search = args.use_global_search
    config.validate()
    result = solve_question2(
        s1,
        bearing,
        config,
        run_sensitivity=args.sensitivity,
        include_grid_values=args.include_grid_values,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        _write_json(args.output, result)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
