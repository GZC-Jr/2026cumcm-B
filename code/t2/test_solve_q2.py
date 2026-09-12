"""Regression tests for the continuous Question 2 solver.

The tests intentionally use a small external polygon and low-order quadrature.
They exercise the model contracts without depending on the expensive default
global-search configuration used for final experiments.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


# ``code`` is a directory in this workspace, not a Python package.  Adding it
# to the import path avoids colliding with Python's standard-library ``code``.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from t2.solve_q2 import (
    Q2Config,
    Question2Model,
    build_target_region,
    parse_input,
    solve_question2,
)


class Question2ModelTests(unittest.TestCase):
    @staticmethod
    def small_config(**overrides: object) -> Q2Config:
        values: dict[str, object] = {
            "circle_sides": 32,
            "quadrature_order": 3,
            "validation_quadrature_order": 4,
            "candidate_grid": 11,
            "use_global_search": False,
            "local_maxiter": 40,
        }
        values.update(overrides)
        return Q2Config(**values)

    def test_parse_input_accepts_reproducible_configuration(self) -> None:
        source = {
            "detector": [10, -20],
            "bearing_deg": 32,
            "circle_sides": 32,
            "quadrature_order": 3,
            "validation_quadrature_order": 4,
            "candidate_grid": 11,
            "use_global_search": False,
            "sensitivity_epsilons": [0.2],
            "sensitivity_powers": [2],
            "sensitivity_taus": [0.05],
        }
        s1, bearing, config = parse_input(source)
        np.testing.assert_allclose(s1, [10.0, -20.0])
        self.assertEqual(bearing, 32.0)
        self.assertFalse(config.use_global_search)
        self.assertEqual(config.circle_sides, 32)
        self.assertEqual(config.sensitivity_powers, (2.0,))

    def test_config_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            Q2Config(circle_sides=8).validate()
        with self.assertRaises(ValueError):
            Q2Config(use_global_search="false").validate()  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            parse_input({"s1": [0, 0], "svd_deg": 20, "tau": -0.1})

    def test_target_region_has_area_and_positive_inradius(self) -> None:
        config = self.small_config()
        region = build_target_region([0.0, 0.0], 32.0, config)
        self.assertGreater(region.area, 0.0)
        self.assertGreater(region.d_star, 0.0)
        self.assertEqual(region.vertices.shape[1], 2)
        self.assertTrue(np.all(region.b[:, None] - region.a @ region.vertices.T >= -1e-5))

    def test_quadrature_conserves_polygon_area(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        self.assertAlmostEqual(model.quad_area, model.region.area, places=7)
        self.assertEqual(len(model.nodes), len(model.quad_weights))
        self.assertTrue(np.all(model.quad_weights > 0.0))

    def test_reception_oracle_at_first_detector_is_zero(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        self.assertLessEqual(model.receive_h(model.s1), 1e-5)
        status = model.reception_status(model.s1)
        # The analytic value is zero; the polygon/circle subtraction can leave
        # a tiny positive floating-point residual, which must not be relabeled
        # as strict feasibility.
        self.assertEqual(status["strict_feasible"], status["h"] <= 0.0)
        self.assertTrue(status["feasible_with_tolerance"])
        self.assertEqual(status["h_tolerance"], model.reception_h_tolerance())

    def test_reception_oracle_dominates_dense_boundary_sampling(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        point = np.asarray((1000.0, 0.0))
        oracle = model.receive_h(point)
        sampled_values: list[float] = []
        vertices = model.region.vertices
        for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
            for fraction in np.linspace(0.0, 1.0, 401):
                target = first + fraction * (second - first)
                sampled_values.append(
                    float(
                        np.sum((point - target) ** 2)
                        - max(
                            model.config.min_receive_radius**2,
                            np.sum((target - model.s1) ** 2),
                        )
                    )
                )
        self.assertGreaterEqual(oracle + 1e-5, max(sampled_values))

    def test_finite_optimizer_barrier_at_singular_point(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        singular = 1000.0 * np.asarray((math.cos(math.radians(32.0)), math.sin(math.radians(32.0))))
        self.assertTrue(math.isinf(model.objective(singular)))
        self.assertTrue(math.isfinite(model.constrained_objective(singular)))

    def test_collinear_point_is_singular_but_noncollinear_point_is_finite(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        # The measured bearing ray intersects the target wedge, hence the
        # second detector on that line has a logarithmic collinearity singularity.
        singular = 1000.0 * np.asarray((math.cos(math.radians(32.0)), math.sin(math.radians(32.0))))
        self.assertFalse(model.geometry_is_finite(singular))
        self.assertTrue(math.isinf(model.objective(singular)))
        finite = np.asarray((1000.0, 0.0))
        self.assertTrue(model.geometry_is_finite(finite))
        self.assertTrue(math.isfinite(model.objective(finite)))

    def test_objective_changes_with_quadrature_order(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        point = np.asarray((1000.0, 0.0))
        low = model.objective(point, quadrature_order=3)
        high = model.objective(point, quadrature_order=4)
        self.assertTrue(math.isfinite(low))
        self.assertTrue(math.isfinite(high))
        self.assertGreater(low, 0.0)
        self.assertGreater(high, 0.0)
        # The orders are independent cached continuous integrations; they need
        # not agree exactly, but a large discrepancy indicates a broken map.
        self.assertLess(abs(low - high) / high, 0.2)

    def test_optimization_and_candidate_grid_contracts(self) -> None:
        model = Question2Model([0.0, 0.0], 32.0, self.small_config())
        optimum = model.optimize()
        point = np.asarray(optimum["point"], dtype=float)
        self.assertTrue(np.isfinite(point).all())
        self.assertTrue(math.isfinite(optimum["j_star"]))
        self.assertAlmostEqual(optimum["j_star"], model.objective(point), places=8)
        self.assertLessEqual(
            optimum["reception_h"],
            model.reception_h_tolerance() + 1e-6,
        )
        self.assertEqual(
            optimum["feasible_with_tolerance"],
            optimum["reception_h"] <= model.reception_h_tolerance(),
        )
        self.assertEqual(
            optimum["strict_feasible"],
            optimum["reception_h"] <= 0.0,
        )

        candidate = model.candidate_region(optimum, grid_size=11, include_values=True)
        self.assertEqual(candidate["grid_shape"], [11, 11])
        self.assertEqual(len(candidate["mask"]), 11)
        self.assertEqual(len(candidate["mask"][0]), 11)
        for iy, row in enumerate(candidate["mask"]):
            for ix, accepted in enumerate(row):
                if not accepted:
                    continue
                grid_point = np.asarray(
                    (candidate["sample_x"][ix], candidate["sample_y"][iy]),
                    dtype=float,
                )
                self.assertLessEqual(
                    model.receive_h(grid_point),
                    model.reception_h_tolerance() + 1e-6,
                )
                self.assertTrue(model.geometry_is_finite(grid_point))
                self.assertLessEqual(
                    model.objective(grid_point), candidate["threshold"] + 1e-8
                )

        wider = model.candidate_region(optimum, tau=0.20, grid_size=11)
        self.assertGreaterEqual(wider["area"], candidate["area"])

    def test_result_metadata_tracks_custom_constraint_parameters(self) -> None:
        config = self.small_config(
            angle_error_deg=2.5,
            min_receive_radius=850.0,
            candidate_grid=11,
            local_maxiter=24,
        )
        result = solve_question2([0.0, 0.0], 32.0, config)
        description = result["reception_constraint"]["description"]
        self.assertIn("850^2", description)
        self.assertIn("2.5°", result["assumptions"][1])
        self.assertIn("850 m", result["assumptions"][2])
        candidate = result["candidate_region"]
        self.assertEqual(
            candidate["reception_h_tolerance"],
            result["reception_constraint"]["h_tolerance"],
        )
        self.assertIn("H(P) <= reception_h_tolerance", candidate["feasibility_rule"])


if __name__ == "__main__":
    unittest.main()
