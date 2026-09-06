from __future__ import annotations

import unittest

from mathmodel_viz.catalog import CATALOG
from mathmodel_viz.registry import DEFAULT_REGISTRY, VisualizationRegistry


class VisualizationRegistryTests(unittest.TestCase):
    def test_catalog_has_unique_ids_and_broad_coverage(self) -> None:
        identifiers = [spec.id for spec in CATALOG]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(len(CATALOG), 50)
        self.assertIn("opt-pareto", identifiers)
        self.assertIn("geo-choropleth", identifiers)
        self.assertIn("ml-calibration", identifiers)

    def test_list_filters_by_workflow_stage(self) -> None:
        results = DEFAULT_REGISTRY.list(stage="optimize")
        self.assertTrue(results)
        self.assertTrue(all("optimize" in [stage.value for stage in spec.stages] for spec in results))

    def test_recommendation_prefers_matching_shapes(self) -> None:
        results = DEFAULT_REGISTRY.recommend(stage="optimize", data_shapes=("multi-objective",))
        self.assertEqual(results[0].spec.id, "opt-pareto")
        self.assertGreaterEqual(results[0].score, 8)

    def test_recommendation_requires_a_signal(self) -> None:
        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.recommend()

    def test_invalid_catalog_rejects_duplicate_ids(self) -> None:
        with self.assertRaises(ValueError):
            VisualizationRegistry((CATALOG[0], CATALOG[0]))


if __name__ == "__main__":
    unittest.main()
