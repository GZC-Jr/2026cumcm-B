from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from mathmodel_viz.styles import configure_matplotlib, save_figure


class StyleTests(unittest.TestCase):
    def test_save_figure_creates_parent_and_file(self) -> None:
        configure_matplotlib()
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as directory:
            output = save_figure(figure, Path(directory) / "nested" / "figure.png")
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
        plt.close(figure)

    def test_save_figure_rejects_unknown_extension(self) -> None:
        figure, _ = plt.subplots()
        with self.assertRaises(ValueError):
            save_figure(figure, "figure.invalid")
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
