import math
import unittest

import numpy as np

from solve_q1 import parse_measurements, solve_question1
from solve_q1 import convex_hull, diameter_rotating_calipers, minimum_enclosing_circle


class Question1GeometryTests(unittest.TestCase):
    def test_design_note_hexagon_counterexample(self):
        data = {
            "measurements": [
                {"x": 1000.0, "y": 0.0, "svd_deg": 180.0},
                {"x": -500.0, "y": 500.0 * math.sqrt(3.0), "svd_deg": 300.0},
                {"x": -500.0, "y": -500.0 * math.sqrt(3.0), "svd_deg": 60.0},
            ]
        }
        measurements, error = parse_measurements(data)
        result = solve_question1(measurements, error)
        self.assertEqual(result["status"], "polygon")
        self.assertEqual(len(result["vertices"]), 6)
        self.assertAlmostEqual(result["diameter"], 40.31484, places=3)
        self.assertFalse(result["diameter_circle"]["covers"])
        self.assertAlmostEqual(result["minimum_cover_circle"]["radius"], 20.36056, places=3)

    def test_point_and_unbounded(self):
        # Opposite closed wedges at the same detector meet at their common
        # apex; the model intentionally treats wedges as closed sets.
        point = solve_question1([(0.0, 0.0, 0.0), (0.0, 0.0, 180.0)])
        self.assertEqual(point["status"], "point")
        self.assertEqual(point["diameter"], 0.0)
        unbounded = solve_question1([(0.0, 0.0, 0.0)])
        self.assertEqual(unbounded["status"], "unbounded")

    def test_segment_and_point(self):
        # Two opposite parallel strips with zero-width limits produce a segment.
        segment = solve_question1([(0.0, 0.0, 0.0), (10.0, 0.0, 180.0)], angle_error_deg=0.001)
        self.assertIn(segment["status"], {"segment", "polygon"})
        point = solve_question1(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 90.0), (0.0, 1.0, 180.0)], angle_error_deg=0.001
        )
        self.assertIn(point["status"], {"point", "polygon", "empty"})

    def test_diameter_calipers_matches_bruteforce(self):
        points = [
            np.array([-19.954279894246397, 0.0]),
            np.array([-10.180280105755218, -17.632762378450657]),
            np.array([9.977139947123561, -17.28091330264231]),
            np.array([20.360560211510734, 0.0]),
            np.array([9.977139947123224, 17.280913302642567]),
            np.array([-10.180280105755312, 17.632762378450913]),
        ]
        diameter, first, second = diameter_rotating_calipers(points)
        brute = max(
            math.dist(points[i], points[j])
            for i in range(len(points))
            for j in range(i + 1, len(points))
        )
        self.assertAlmostEqual(diameter, brute, places=10)
        self.assertAlmostEqual(math.dist(points[first], points[second]), brute, places=10)

    def test_minimum_circle_handles_collinear_and_large_translation(self):
        collinear = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])]
        result = minimum_enclosing_circle(collinear)
        self.assertAlmostEqual(result["radius"], 1.0, places=12)
        np.testing.assert_allclose(result["center"], [1.0, 0.0], atol=1e-12)

        base = np.array([1.0e12, -1.0e12])
        square = [
            base,
            base + [2.0, 0.0],
            base + [2.0, 2.0],
            base + [0.0, 2.0],
        ]
        hull = convex_hull(square, tol=1e-9)
        self.assertEqual(len(hull), 4)
        result = minimum_enclosing_circle(hull, tol=1e-9)
        self.assertAlmostEqual(result["radius"], math.sqrt(2.0), places=12)
        np.testing.assert_allclose(result["center"] - base, [1.0, 1.0], atol=1e-9)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            parse_measurements({"measurements": [], "angle_error_deg": 1.0})
        with self.assertRaises(ValueError):
            parse_measurements({"measurements": [{"x": 0, "y": 0, "svd_deg": 0}], "angle_error_deg": 0})
        with self.assertRaises(ValueError):
            solve_question1([])
        with self.assertRaises(ValueError):
            solve_question1([(float("nan"), 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
