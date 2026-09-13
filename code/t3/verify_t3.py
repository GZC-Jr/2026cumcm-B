#!/usr/bin/env python3
"""Deterministic certificates and one local end-to-end smoke test for t3.py.

This file deliberately does not contact the official simulator.  It exercises
the hard geometric invariants directly and uses LocalBackend only for a
repeatable accounting/rollout check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import t3


TOL = 1e-6


class ProtocolBackend:
    """Small deterministic backend for protocol-boundary tests."""

    def __init__(self, enter=None, exit_response=None, clear_result="no_target_in_range",
                 measure_response=None):
        self.enter = enter or {
            "accepted": True,
            "real_timestamp_ms": 0,
            "virtual_time_s": 0.0,
            "remaining_real_duration_s": 1200.0,
            "max_virtual_duration_s": 360000.0,
        }
        self.exit_response = exit_response or {
            "accepted": True,
            "real_timestamp_ms": 0,
            "virtual_time_s": 0.0,
            "exit_reason": "user_exit",
        }
        self.clear_result = clear_result
        self.measure_response = measure_response
        self.paths = []

    def post(self, path, payload, timeout):
        self.paths.append(path)
        if path == "/enter":
            return dict(self.enter)
        if path == "/exit":
            return dict(self.exit_response)
        if path == "/clear":
            return {
                "accepted": True,
                "real_timestamp_ms": 0,
                "virtual_time_s": 0.0,
                "clear_result": self.clear_result,
            }
        if path == "/measure":
            response = {
                "accepted": True,
                "real_timestamp_ms": 0,
                "virtual_time_s": 5.0,
                "measure_result": "no_signal",
            }
            if self.measure_response is not None:
                response.update(self.measure_response)
            return response
        raise AssertionError(f"unexpected protocol path: {path}")


def _json_value(value):
    """Convert numpy/scalar values so the report stays strict JSON.

    Python's default encoder emits ``NaN``/``Infinity`` extensions.  Boundary
    checks intentionally exercise an infinite MEC for an empty region, so
    non-finite numbers are represented as JSON ``null`` instead.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    return value


def _check(name, fn):
    try:
        detail = fn()
        return {"name": name, "passed": True, "detail": _json_value(detail)}
    except Exception as exc:  # The report should identify the first failed invariant.
        return {"name": name, "passed": False,
                "error": f"{type(exc).__name__}: {exc}"}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_layout_certificates():
    records = []
    for name in t3.LAYOUT_PRESETS:
        layout = t3.resolve_layout(name)
        cert = layout["certificate"]
        _require(cert["certified"], f"{name} is not certified")
        _require(cert["whole_domain_bound_m"] <= 1000.0 + TOL,
                 f"{name} exceeds the 1000 m bound")
        _require(len(t3.scan_points(layout["ring_points"], layout["ring_radius"])) ==
                 layout["ring_points"] + 1, f"{name} scan-point count mismatch")
        records.append({
            "layout": name,
            "ring_points": layout["ring_points"],
            "ring_radius_m": layout["ring_radius"],
            "outer_bound_m": cert["outer_annulus_distance_bound_m"],
            "whole_domain_bound_m": cert["whole_domain_bound_m"],
        })

    try:
        t3.coverage_certificate(5, 1000.0)
    except ValueError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    _require(invalid_rejected, "an obviously infeasible five-point layout was accepted")
    return {"preset_count": len(records), "presets": records,
            "infeasible_layout_rejected": invalid_rejected}


def check_scan_boundary_reception():
    records = []
    for name in t3.LAYOUT_PRESETS:
        layout = t3.resolve_layout(name)
        n = layout["ring_points"]
        radius = layout["ring_radius"]
        scans = np.asarray(t3.scan_points(n, radius), dtype=float)
        # The sector midpoint is the worst angular direction for the nearest
        # ring station.  Test both annulus endpoints and the origin disk edge.
        midpoint = math.pi / n
        endpoint_rows = []
        for radial in (1000.0, 1800.0):
            target = np.array([radial * math.cos(midpoint),
                               radial * math.sin(midpoint)])
            nearest = float(np.min(np.linalg.norm(scans[1:] - target, axis=1)))
            _require(nearest <= 1000.0 + TOL,
                     f"{name} misses annulus endpoint at r={radial:g}")
            endpoint_rows.append({"radial_m": radial, "nearest_distance_m": nearest})
        center_edge = float(np.linalg.norm(scans[0] - np.array([1000.0, 0.0])))
        _require(center_edge <= 1000.0 + TOL, f"{name} misses center-disk edge")
        records.append({"layout": name, "annulus_endpoints": endpoint_rows,
                        "center_edge_distance_m": center_edge})
    return {"layouts": records, "tested_radius_boundary_m": 1000.0}


def check_polygon_contains_truth():
    true_position = np.array([1100.0, 400.0])
    stations = [(0.0, 0.0), (939.0, 0.0), (0.0, 939.0)]
    errors = (0.90, -0.90, 0.45)
    poly = t3.initial_poly()
    rows = []
    for station, error in zip(stations, errors):
        _require(t3.dist(station, true_position) <= 1500.0,
                 "synthetic station is outside the 1500 m receive disk")
        true_angle = math.degrees(math.atan2(true_position[1] - station[1],
                                             true_position[0] - station[0])) % 360.0
        observed = (true_angle + error) % 360.0
        poly = t3.update_poly(poly, station, observed)
        _require(t3.contains(poly, tuple(true_position)),
                 f"true position left the conservative polygon after {station}")
        rows.append({"station": station, "angle_error_deg": error,
                     "vertex_count": len(poly), "contains_truth": True})
    return {"true_position": true_position.tolist(), "updates": rows}


def check_mec_jung_certificates():
    polygons = {
        "equilateral_triangle": [(1.0, 0.0), (-0.5, math.sqrt(3) / 2),
                                  (-0.5, -math.sqrt(3) / 2)],
        "square": [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
        "rectangle": [(-3.0, -1.0), (3.0, -1.0), (3.0, 1.0), (-3.0, 1.0)],
    }
    records = []
    for name, polygon in polygons.items():
        center, radius = t3.mec(polygon)
        cert = t3.problem1_geometry_certificate(polygon, center, radius)
        _require(cert["jung_certified"], f"Jung bound failed for {name}")
        _require(radius + TOL >= cert["diameter_m"] / 2.0,
                 f"diameter lower bound failed for {name}")
        _require(radius <= cert["jung_upper_m"] + TOL,
                 f"Jung upper bound failed for {name}")
        _require(all(t3.dist(center, vertex) <= radius + TOL for vertex in polygon),
                 f"MEC does not cover {name}")
        records.append({"name": name, "certificate": cert})
    return {"polygons": records}


def check_failed_clear_cache_invalidation():
    poly = t3.initial_poly()
    target = {
        "poly": poly,
        "stations": [],
        "failed_clears": [],
        "planned": (1.0, 2.0),
        "mean": np.array([250.0, 100.0]),
        "covariance": np.eye(2) * 300.0 ** 2,
    }
    before = t3.probe_distribution(target)
    _require(before is not None, "pre-failure posterior quadrature is empty")
    _require("probe_base" in target, "probe cache was not created")
    failed_point = (250.0, 100.0)
    detail = t3.condition_failed_clear(target, failed_point)
    _require(target["planned"] is None, "old measurement plan survived a failed clear")
    _require("probe_base" not in target,
             "stale probe cache survived failed-clear conditioning")
    _require(target["failed_clears"] == [failed_point],
             "failed-clear history was not recorded")
    after = t3.probe_distribution(target)
    _require(after is not None, "posterior became empty after a non-degenerate failure")
    points, weights = after
    near_mass = float(weights[np.linalg.norm(points - np.asarray(failed_point), axis=1) <= 20.0].sum())
    _require(near_mass <= 1e-12, "conditioned posterior still assigns mass inside failed disk")
    return {"conditioning": detail, "quadrature_points": len(points),
            "mass_inside_failed_20m_disk": near_mass,
            "cache_rebuilt_lazily": "probe_base" in target}


def check_local_time_accounting(output_root, seed):
    layout = t3.resolve_layout()
    case_dir = Path(output_root) / "local_case"
    row, truth = t3.run_local_case(seed, 1, layout, case_dir)
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    expected = (summary["distance_m"] / 5.0 + summary["switch_count"] +
                5.0 * summary["measure_count"] +
                3.0 * summary["clear_attempts"] +
                2.0 * summary["cleared_count"])
    _require(summary["status"] == "complete", f"local case status={summary['status']}")
    _require(row["success"], "local case evaluator did not mark success")
    _require(summary["cleared_count"] == truth["total"],
             "local case did not clear every generated source")
    _require(abs(summary["virtual_time_s"] - expected) <= 1e-5,
             "virtual time does not match movement/action accounting")
    _require(abs(summary["accounting_delta_s"]) <= 1e-5,
             "reported accounting delta is non-zero")
    absent_channels = summary.get("absent_channels", [])
    _require(len(absent_channels) == len(set(absent_channels)),
             "duplicate absent channel state")
    return {"seed": seed, "true_total": truth["total"], "row": row,
            "summary_accounting_delta_s": summary["accounting_delta_s"],
            "expected_virtual_time_s": expected,
            "absent_channel_count": len(absent_channels)}


def check_protocol_boundaries():
    # A rejected response may carry virtual_time_s=0, but that is not a clock
    # reset. The last accepted timestamp must remain intact.
    rejected = ProtocolBackend(enter={
        "accepted": False,
        "real_timestamp_ms": 0,
        "virtual_time_s": 0.0,
    })
    robot = t3.Robot(rejected, "TEST")
    robot.vt = 17.25
    try:
        robot.call("/enter")
    except RuntimeError as exc:
        _require("accepted=false" in str(exc), "rejected response was not surfaced")
    else:
        raise AssertionError("accepted=false unexpectedly succeeded")
    _require(robot.vt == 17.25, "accepted=false reset the virtual clock")

    # With no real time left, the scheduler must stop before sending an action
    # and must not issue /exit after the deadline has elapsed.
    expired = ProtocolBackend(enter={
        "accepted": True,
        "real_timestamp_ms": 0,
        "virtual_time_s": 0.0,
        "remaining_real_duration_s": 0.0,
        "max_virtual_duration_s": 360000.0,
    })
    stopped = t3.Robot(expired, "TEST").run()
    _require(stopped["status"] == "budget_stop",
             f"short-deadline status={stopped['status']}")
    _require(expired.paths == ["/enter"],
             f"expired run sent unexpected requests: {expired.paths}")

    # A normal run still closes the session explicitly. Avoid the full
    # scheduler here; this isolates the /enter -> /exit lifecycle.
    normal = ProtocolBackend()
    complete_robot = t3.Robot(normal, "TEST")
    # Simulate an already-advanced accepted action before /exit.  The
    # protocol says /exit does not advance the clock, and a stale/zero exit
    # timestamp must not erase that accounting value.
    complete_robot.schedule_run = lambda: setattr(complete_robot, "vt", 19.0)
    complete = complete_robot.run()
    _require(complete["status"] == "complete", "normal lifecycle did not complete")
    _require(complete["virtual_time_s"] == 19.0,
             "/exit response incorrectly rewound the virtual clock")
    _require(normal.paths == ["/enter", "/exit"],
             f"normal lifecycle paths={normal.paths}")
    return {
        "rejected_clock_preserved": robot.vt,
        "expired_status": stopped["status"],
        "expired_paths": expired.paths,
        "normal_paths": normal.paths,
        "exit_clock_preserved": complete["virtual_time_s"],
    }


def check_enter_budget_guards():
    """Malformed /enter budgets must not mutate local clocks or deadlines."""
    base = {
        "accepted": True,
        "real_timestamp_ms": 0,
        "virtual_time_s": 0.0,
        "remaining_real_duration_s": 1200.0,
        "max_virtual_duration_s": 360000.0,
    }
    cases = [
        ("remaining_nan", "remaining_real_duration_s", math.nan),
        ("remaining_inf", "remaining_real_duration_s", math.inf),
        ("remaining_negative", "remaining_real_duration_s", -1.0),
        ("remaining_fractional", "remaining_real_duration_s", 1200.5),
        ("remaining_too_large", "remaining_real_duration_s", 1200.1),
        ("vmax_nan", "max_virtual_duration_s", math.nan),
        ("vmax_inf", "max_virtual_duration_s", math.inf),
        ("vmax_zero", "max_virtual_duration_s", 0.0),
        ("vmax_negative", "max_virtual_duration_s", -1.0),
    ]
    rejected = []
    for name, field, value in cases:
        enter = dict(base)
        enter[field] = value
        backend = ProtocolBackend(enter=enter)
        robot = t3.Robot(backend, "TEST")
        robot.vt = 17.25
        # Keep the synthetic backend inside the request's real-time guard;
        # the malformed field under test must be the first rejection.
        robot.deadline = math.inf
        robot.vmax = 99.0
        try:
            robot.call("/enter")
        except RuntimeError as exc:
            message = str(exc)
            _require(field in message, f"{name} error did not identify {field}")
            rejected.append(name)
        else:
            raise AssertionError(f"{name} malformed budget was accepted")
        _require(robot.vt == 17.25, f"{name} changed the virtual clock")
        _require(math.isinf(robot.deadline), f"{name} changed the deadline")
        _require(robot.vmax == 99.0, f"{name} changed the virtual budget")
    return {"rejected_cases": rejected,"case_count": len(rejected)}


def check_measure_response_guards():
    """Invalid direction angles or virtual timestamps are rejected safely."""
    base_enter = {
        "accepted": True,
        "real_timestamp_ms": 0,
        "virtual_time_s": 0.0,
        "remaining_real_duration_s": 1200.0,
        "max_virtual_duration_s": 360000.0,
    }
    cases = [
        ("svd_nan", {"measure_result": "direction", "svd_deg": math.nan}, "svd_deg"),
        ("svd_inf", {"measure_result": "direction", "svd_deg": math.inf}, "svd_deg"),
        ("svd_negative", {"measure_result": "direction", "svd_deg": -0.01}, "svd_deg"),
        ("svd_360", {"measure_result": "direction", "svd_deg": 360.0}, "svd_deg"),
        ("svd_missing", {"measure_result": "direction"}, "svd_deg"),
        ("clock_negative", {"virtual_time_s": -1.0}, "virtual_time_s"),
        ("clock_nan", {"virtual_time_s": math.nan}, "virtual_time_s"),
    ]
    rejected = []
    for name, response, field in cases:
        backend = ProtocolBackend(enter=base_enter, measure_response=response)
        robot = t3.Robot(backend, "TEST")
        robot.call("/enter")
        try:
            robot.call("/measure", (0.0, 0.0), 1)
        except RuntimeError as exc:
            _require(field in str(exc), f"{name} error did not identify {field}")
            rejected.append(name)
        else:
            raise AssertionError(f"{name} malformed response was accepted")
        _require(robot.measures == 0, f"{name} incremented measure count")
    return {"rejected_cases": rejected, "case_count": len(rejected)}


def check_fallback_failed_clear_feedback():
    backend = ProtocolBackend()
    robot = t3.Robot(backend, "TEST")
    robot.beliefs = {1: {
        "mean": np.zeros(2),
        "covariance": np.eye(2) * 80.0 ** 2,
    }}
    target = {
        "poly": [(-100.0, -100.0), (100.0, -100.0),
                 (100.0, 100.0), (-100.0, 100.0)],
        "stations": [(200.0, 0.0)],
        "failed_clears": [],
        "planned": (0.0, 0.0),
        "mean": np.zeros(2),
        "covariance": np.eye(2) * 80.0 ** 2,
        "fallback": [(0.0, 0.0), (60.0, 0.0)],
    }
    info = robot.record_failed_clear(1, target, (0.0, 0.0), "fallback")
    _require(target["planned"] is None, "fallback failure kept the old plan")
    _require(target["failed_clears"] == [(0.0, 0.0)],
             "fallback failure was not recorded")
    _require(target["fallback"] == [(60.0, 0.0)],
             "fallback candidates inside the failed disk were retained")
    _require(info["source"] == "fallback", "failure source was not logged")
    return {"failure_info": info,
            "remaining_fallback": target["fallback"]}


def check_discovered_negative_conditioning():
    """A known source stays known when a later scan point has no signal."""
    backend = ProtocolBackend()
    robot = t3.Robot(backend, "TEST", quiet=True)
    robot.beliefs = {1: {
        "mean": np.array([600.0, 600.0]),
        "covariance": np.eye(2) * 120.0 ** 2,
        "existence": 1.0,
        "negatives": [],
    }}
    target = {
        "poly": [(500.0, 500.0), (700.0, 500.0),
                 (700.0, 700.0), (500.0, 700.0)],
        "stations": [(0.0, 0.0)],
        "negatives": [],
        "planned": (620.0, 610.0),
        "mean": np.array([600.0, 600.0]),
        "covariance": np.eye(2) * 120.0 ** 2,
        "probe_base": {"version": 1},
    }
    robot.targets = {1: target}
    before_poly = list(target["poly"])
    before_mean = target["mean"].copy()
    posterior = robot.condition_discovered_negative(1, (1800.0, 0.0))
    _require(target["planned"] is None,
             "known-target negative observation kept the stale plan")
    _require("probe_base" not in target,
             "known-target negative observation kept a stale probe cache")
    _require(target["negatives"] == [(1800.0, 0.0)],
             "known-target negative observation was not recorded")
    _require(robot.beliefs[1]["negatives"] == [],
             "helper unexpectedly duplicated the global negative history")
    _require(robot.beliefs[1]["existence"] == 1.0,
             "known-target negative observation changed existence probability")
    _require(target["poly"] == before_poly,
             "known-target negative observation changed the hard polygon")
    _require(np.all(np.isfinite(target["mean"])) and
             np.all(np.isfinite(target["covariance"])),
             "known-target negative posterior is not finite")
    _require(not np.allclose(target["mean"], before_mean),
             "known-target negative observation did not update planning posterior")
    return {
        "posterior_note": posterior["note"],
        "posterior_mean": target["mean"].tolist(),
        "existence_probability": robot.beliefs[1]["existence"],
        "hard_polygon_unchanged": target["poly"] == before_poly,
    }


def check_degenerate_geometry_regressions():
    """Empty/one-dimensional regions must fall back without false certificates."""
    cases = {
        "empty": [],
        "single_point": [(12.0, -7.0)],
        "line_segment": [(0.0, 0.0), (80.0, 0.0)],
        "collinear_polygon": [(-100.0, 0.0), (0.0, 0.0),
                              (100.0, 0.0), (40.0, 0.0)],
    }
    records = []
    for name, polygon in cases.items():
        triangles, areas = t3.mesh(polygon, 3)
        _require(triangles.shape == (0, 3, 2),
                 f"{name} unexpectedly produced positive-area triangles")
        _require(areas.size == 0, f"{name} has non-empty mesh areas")
        center, radius = t3.mec(polygon)
        if polygon:
            _require(math.isfinite(radius), f"{name} MEC radius is not finite")
            _require(all(t3.dist(center, p) <= radius + TOL for p in polygon),
                     f"{name} MEC does not cover its points")
        else:
            _require(math.isinf(radius), "empty region received a finite MEC")
        target = {"mean": np.zeros(2), "covariance": np.eye(2) * 100.0 ** 2}
        selected, detail = t3.bayes_select(
            polygon, [], (0.0, 0.0), target, math.inf)
        _require(selected is None,
                 f"{name} was offered a measurement candidate")
        _require(detail.get("reason") == "degenerate_or_empty_polygon",
                 f"{name} did not report a degenerate planning reason")
        score = t3.mandatory_stop_measurement_score(
            {**target, "poly": polygon, "stations": []}, (0.0, 0.0))
        _require(score is None, f"{name} was treated as a mandatory stop")
        records.append({"name": name, "mec_radius_m": radius,
                        "planner_reason": detail.get("reason")})
    return {"cases": records}


def check_empty_station_and_narrow_polygon():
    """An empty station history and a thin but valid polygon remain plannable."""
    narrow = [(-0.5, -100.0), (0.5, -100.0),
              (0.5, 100.0), (-0.5, 100.0)]
    triangles, areas = t3.mesh(narrow, 2)
    _require(len(triangles) > 0 and np.all(areas > 0),
             "valid narrow polygon was discarded by the mesh")
    target = {
        "poly": narrow,
        "stations": [],
        "negatives": [],
        "mean": np.array([0.0, 0.0]),
        "covariance": np.eye(2) * 250.0 ** 2,
    }
    posterior = t3.update_gaussian(target, (-1000.0, 0.0), 0.0)
    _require(np.all(np.isfinite(target["mean"])) and
             np.all(np.isfinite(target["covariance"])),
             "narrow-polygon posterior is not finite")
    selected, detail = t3.bayes_select(
        narrow, [], (0.0, 0.0), target, math.inf)
    _require(selected is not None,
             f"empty-station narrow polygon was not plannable: {detail}")
    _require(detail["reception_certificate"]["certified"],
             "narrow-polygon candidate lacks a reception certificate")
    _require(detail["problem2_dop_tiebreak"]["original_dop"] is None,
             "empty station set unexpectedly triggered a DOP calculation")
    return {
        "triangle_count": len(triangles),
        "posterior_note": posterior["note"],
        "selected_point": selected,
        "empty_station_dop": detail["problem2_dop_tiebreak"]["original_dop"],
    }


def check_strict_reception_boundary():
    """The shared squared-distance margin rejects equality at 1000 m."""
    theta = 1e-3
    triangles = np.asarray([[(1000.0, 0.0),
                             (1000.0 * math.cos(theta),
                              1000.0 * math.sin(theta)),
                             (1000.0 * math.cos(theta),
                              -1000.0 * math.sin(theta))]], dtype=float)
    exact = t3.reception_upper_bound_m2(triangles, (0.0, 0.0), [])
    inward = t3.reception_upper_bound_m2(triangles, (2e-6, 0.0), [])
    outward = t3.reception_upper_bound_m2(triangles, (-2e-6, 0.0), [])
    _require(abs(exact) <= 1e-6, f"boundary certificate was not zero: {exact}")
    _require(not t3.reception_certified(exact),
             "exact 1000 m boundary was accepted")
    _require(not t3.reception_certified(outward),
             "outside candidate was accepted")
    _require(inward < -t3.RECEIVE_MARGIN_M2 and
             t3.reception_certified(inward),
             "candidate inside the strict margin was rejected")
    _require(not t3.reception_certified(-0.5 * t3.RECEIVE_MARGIN_M2),
             "half-margin certificate was accepted")
    _require(t3.reception_certified(-1.5 * t3.RECEIVE_MARGIN_M2),
             "more-than-margin certificate was rejected")
    return {"exact_upper_m2": exact, "inward_upper_m2": inward,
            "outward_upper_m2": outward,
            "required_margin_m2": t3.RECEIVE_MARGIN_M2}


def check_optimizer_failure_fallbacks():
    """clear_point must use the MEC fallback for every unsafe SLSQP outcome."""
    backend = ProtocolBackend()
    robot = t3.Robot(backend, "TEST", quiet=True)
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    fallback = (5.0, 5.0)
    original_minimize = t3.minimize
    outcomes = {}
    try:
        def raise_optimizer(*args, **kwargs):
            raise RuntimeError("synthetic SLSQP failure")

        t3.minimize = raise_optimizer
        outcomes["exception"] = robot.clear_point(polygon, fallback)

        def unsuccessful_optimizer(*args, **kwargs):
            return SimpleNamespace(success=False, x=np.asarray([0.0, 0.0]))

        t3.minimize = unsuccessful_optimizer
        outcomes["success_false"] = robot.clear_point(polygon, fallback)

        def nonfinite_optimizer(*args, **kwargs):
            return SimpleNamespace(success=True, x=np.asarray([np.nan, np.inf]))

        t3.minimize = nonfinite_optimizer
        outcomes["nonfinite_solution"] = robot.clear_point(polygon, fallback)
    finally:
        t3.minimize = original_minimize
    for name, point in outcomes.items():
        _require(np.allclose(point, fallback),
                 f"{name} did not return the deterministic MEC fallback")
    return {"outcomes": outcomes, "fallback": fallback,
            "minimize_restored": t3.minimize is original_minimize}


def check_posterior_weight_degeneracy():
    """Degenerate quadrature weights are rejected and callers stay finite."""
    try:
        t3.gaussian_moments([(0.0, 0.0), (1.0, 1.0)], [0.0, 0.0])
    except ValueError as exc:
        _require("退化" in str(exc),
                 "zero posterior weights raised the wrong diagnostic")
        zero_weight_error = str(exc)
    else:
        raise AssertionError("zero posterior weights were silently accepted")

    degenerate_target = {"poly": [(0.0, 0.0), (1.0, 0.0)],
                         "stations": [], "negatives": []}
    record = t3.update_gaussian(degenerate_target, (0.0, 0.0), 0.0)
    _require(record["note"].startswith("degenerate_mesh_fallback"),
             "degenerate posterior did not use the documented fallback")
    _require(np.all(np.isfinite(degenerate_target["mean"])) and
             np.all(np.isfinite(degenerate_target["covariance"])),
             "degenerate posterior fallback is not finite")
    empty_map = SimpleNamespace(centers=np.empty((0, 2)),
                                remaining=np.empty((20, 0), dtype=bool))
    belief = {"existence": 0.65, "negatives": []}
    negative_record = t3.update_unknown_belief(belief, empty_map, 1)
    _require(np.all(np.isfinite(negative_record["mean"])) and
             np.all(np.isfinite(negative_record["covariance"])),
             "empty-map belief fallback is not finite")
    _require(t3.detection_information(belief, empty_map, (0.0, 0.0)) == (0.0, 0.0),
             "empty-map detection information was non-zero")
    return {"zero_weight_error": zero_weight_error,
            "degenerate_update_note": record["note"],
            "empty_map_update_note": negative_record["note"]}


def check_direct_planner_input_guards():
    """Public planner helpers reject malformed inputs without raising."""
    polygon = [(-50.0, -50.0), (50.0, -50.0),
               (50.0, 50.0), (-50.0, 50.0)]
    base = {
        "poly": polygon,
        "stations": [],
        "failed_clears": [],
        "planned": (0.0, 0.0),
        "center": (0.0, 0.0),
        "mean": np.zeros(2),
        "covariance": np.eye(2) * 100.0 ** 2,
        "radius": 5.0,
        "probe_attempts": 0,
    }

    invalid_targets = {}
    for field, value in (("mean", "bad"),
                         ("center", [0.0]),
                         ("planned", [np.nan, 0.0]),
                         ("failed_clears", [["bad", 0.0]])):
        target = dict(base)
        target[field] = value
        try:
            result = t3.joint_probe_plan(
                target, (0.0, 0.0), 1, 1, (1.0, 0.0), True, None)
        except Exception as exc:
            raise AssertionError(f"invalid {field} raised {type(exc).__name__}: {exc}")
        _require(result is None, f"invalid {field} was accepted by joint_probe_plan")
        invalid_targets[field] = result

    negative_radius = dict(base, radius=-1.0)
    _require(t3.joint_probe_plan(
        negative_radius, (0.0, 0.0), 1, 1, (1.0, 0.0), True, None) is None,
             "negative probe radius was accepted")

    old_lookahead = t3.SCHEDULE_CONFIG["lookahead"]
    try:
        t3.SCHEDULE_CONFIG["lookahead"] = math.nan
        _require(t3.joint_probe_plan(
            dict(base), (0.0, 0.0), 1, 1, (1.0, 0.0), True, None) is None,
                 "non-finite lookahead was accepted")
    finally:
        t3.SCHEDULE_CONFIG["lookahead"] = old_lookahead

    invalid_dop = {}
    for name, station1, candidate in (("station", "bad", (0.0, 0.0)),
                                      ("candidate", (0.0, 0.0), [0.0]),
                                      ("nonfinite", (0.0, 0.0), [np.nan, 0.0])):
        try:
            score = t3.depth_weighted_dop(polygon, station1, candidate)
        except Exception as exc:
            raise AssertionError(f"invalid DOP {name} raised {type(exc).__name__}: {exc}")
        _require(math.isinf(score), f"invalid DOP {name} was not rejected")
        invalid_dop[name] = score

    return {"invalid_joint_targets": sorted(invalid_targets),
            "negative_radius_rejected": True,
            "nonfinite_lookahead_rejected": True,
            "invalid_dop_scores": invalid_dop}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=t3.OUTPUT_ROOT / "verification",
                        help="directory for the verification report and local case")
    parser.add_argument("--seed", type=int, default=2026091399,
                        help="LocalBackend seed for the end-to-end smoke case")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checks = [
        _check("layout_certificates", check_layout_certificates),
        _check("scan_boundary_reception", check_scan_boundary_reception),
        _check("polygon_contains_truth", check_polygon_contains_truth),
        _check("mec_jung_certificates", check_mec_jung_certificates),
        _check("failed_clear_cache_invalidation", check_failed_clear_cache_invalidation),
        _check("local_time_accounting",
               lambda: check_local_time_accounting(args.output, args.seed)),
        _check("protocol_boundaries", check_protocol_boundaries),
        _check("enter_budget_guards", check_enter_budget_guards),
        _check("measure_response_guards", check_measure_response_guards),
        _check("fallback_failed_clear_feedback", check_fallback_failed_clear_feedback),
        _check("discovered_negative_conditioning",
               check_discovered_negative_conditioning),
        _check("degenerate_geometry_regressions",
               check_degenerate_geometry_regressions),
        _check("empty_station_narrow_polygon",
               check_empty_station_and_narrow_polygon),
        _check("strict_reception_boundary", check_strict_reception_boundary),
        _check("optimizer_failure_fallbacks", check_optimizer_failure_fallbacks),
        _check("posterior_weight_degeneracy", check_posterior_weight_degeneracy),
        _check("direct_planner_input_guards", check_direct_planner_input_guards),
    ]
    report = {"script": "code/t3/verify_t3.py",
              "official_simulator_test": False,
              "checks": checks,
              "passed": all(item["passed"] for item in checks)}
    report_path = args.output / "verification_results.json"
    report_path.write_text(json.dumps(_json_value(report), ensure_ascii=False,
                                      indent=2), encoding="utf-8")
    print(json.dumps(_json_value(report), ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
