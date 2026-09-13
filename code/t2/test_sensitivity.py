"""Regression tests for the Question 2 sensitivity/export/plot pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from t2 import advanced_sensitivity_visualization as advanced_viz  # noqa: E402
from t2 import plot_sensitivity, run_sensitivity  # noqa: E402
from t2.solve_q2 import Q2Config, Question2Model  # noqa: E402


class Question2SensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = Q2Config(
            circle_sides=16,
            quadrature_order=2,
            validation_quadrature_order=2,
            candidate_grid=11,
            use_global_search=False,
            local_maxiter=12,
            sensitivity_epsilons=(0.2,),
            sensitivity_powers=(2.0,),
            sensitivity_taus=(0.0, 0.05, 0.20),
        )
        cls.model = Question2Model([0.0, 0.0], 32.0, config)
        cls.sensitivity = cls.model.sensitivity(grid_size=11, use_global_search=False)

    def test_one_weight_pair_is_reoptimised_and_has_all_tau_rows(self) -> None:
        rows = self.sensitivity["rows"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["epsilon_w"], row["p_w"]), (0.2, 2.0))
        self.assertGreater(row["local_run_count"], 0)
        self.assertEqual(len(row["candidate_by_tau"]), 3)
        self.assertEqual(
            [record["tau"] for record in row["candidate_by_tau"]],
            [0.0, 0.05, 0.2],
        )

    def test_candidate_area_and_cell_count_are_monotone_in_tau(self) -> None:
        records = self.sensitivity["rows"][0]["candidate_by_tau"]
        areas = [float(record["area"]) for record in records]
        counts = [int(record["accepted_cell_count"]) for record in records]
        self.assertEqual(areas, sorted(areas))
        self.assertEqual(counts, sorted(counts))
        for record in records:
            self.assertAlmostEqual(
                record["area"],
                record["accepted_cell_count"] * record["x_step"] * record["y_step"],
            )

    def test_empty_grid_level_set_uses_none_distance_range(self) -> None:
        region = self.model.candidate_region(
            {"point": [0.0, 0.0], "j_star": 1.0e-12},
            tau=0.0,
            grid_size=11,
        )
        self.assertEqual(region["area"], 0.0)
        self.assertEqual(region["component_count"], 0)
        self.assertIsNone(region["distance_from_s1_min"])
        self.assertIsNone(region["distance_from_s1_max"])
        json.dumps(region, allow_nan=False)

    def test_advanced_metric_maps_candidate_area_to_tau_record_area(self) -> None:
        """Regression guard for the public ``candidate_area`` metric name.

        Sensitivity rows expose candidate records with the shorter ``area``
        key.  The advanced tornado chart uses the public name
        ``candidate_area`` and must translate it rather than returning NaN.
        """

        row = {
            "candidate_by_tau": [
                {"tau": 0.0, "area": 9.0, "component_count": 1},
                {"tau": 0.05, "area": 1234.5, "component_count": 2},
            ]
        }
        self.assertAlmostEqual(
            advanced_viz._metric_value(row, "candidate_area", tau=0.0),
            9.0,
        )
        self.assertAlmostEqual(
            advanced_viz._metric_value(row, "candidate_area", tau=0.05),
            1234.5,
        )
        self.assertAlmostEqual(
            advanced_viz._candidate_value(row, 0.05, "candidate_area"),
            1234.5,
        )
        self.assertEqual(
            advanced_viz._metric_value(row, "component_count", tau=0.05),
            2.0,
        )

    def test_json_csv_round_trip_and_blank_distance_plot(self) -> None:
        payload = {"model": "question_2_continuous_robust_dop", "sensitivity": self.sensitivity}
        records = run_sensitivity._flatten_rows(self.sensitivity)
        self.assertEqual(len(records), 3)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            json_path = directory / "sensitivity.json"
            csv_path = directory / "sensitivity.csv"
            run_sensitivity._write_json(json_path, payload)
            run_sensitivity._write_csv(csv_path, records)

            json_payload = plot_sensitivity.load_payload(json_path)
            csv_payload = plot_sensitivity.load_payload(csv_path)
            self.assertEqual(len(json_payload["sensitivity"]["rows"]), 1)
            self.assertEqual(len(csv_payload["sensitivity"]["rows"]), 1)
            self.assertEqual(
                len(csv_payload["sensitivity"]["rows"][0]["candidate_by_tau"]),
                3,
            )

            plot_sensitivity.configure_matplotlib(
                font_candidates=("Microsoft YaHei", "Noto Sans SC"),
                figure_dpi=50,
                savefig_dpi=50,
            )
            output = plot_sensitivity.plot_tau_effect(csv_payload, directory / "figures")
            self.assertTrue(output.is_file())

    def test_all_missing_distance_ranges_are_plot_safe(self) -> None:
        payload = {
            "sensitivity": {
                "reference_weight": {"epsilon_w": 0.2, "p_w": 2.0},
                "epsilon_values": [0.2],
                "p_values": [2.0],
                "rows": [
                    {
                        "epsilon_w": 0.2,
                        "p_w": 2.0,
                        "point": [0.0, 0.0],
                        "j_star": 1.0,
                        "j_under_reference_weight": 1.0,
                        "distance_from_s1": 0.0,
                        "bearing_difference_deg": 0.0,
                        "candidate_by_tau": [
                            {
                                "tau": 0.0,
                                "area": 0.0,
                                "component_count": 0,
                                "distance_from_s1_min": None,
                                "distance_from_s1_max": None,
                            }
                        ],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            plot_sensitivity.configure_matplotlib(figure_dpi=50, savefig_dpi=50)
            output = plot_sensitivity.plot_tau_effect(payload, Path(temporary))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
