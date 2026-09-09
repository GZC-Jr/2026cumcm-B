from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib as mpl
import matplotlib.pyplot as plt

import mathmodel_viz.styles as style_module
from mathmodel_viz.styles import (
    AUXILIARY,
    FONT_CJK,
    FONT_LATIN,
    PRIMARY_BLUE,
    PRIMARY_RED,
    TRANSITION,
    VisualizationTheme,
    blend_colors,
    configure_matplotlib,
    configure_theme,
    get_theme,
    normalize_hex_color,
    rgba,
    rgba_css,
    save_figure,
)


class StyleTests(unittest.TestCase):
    def test_default_theme_constants_and_five_digit_normalization(self) -> None:
        self.assertEqual(PRIMARY_RED, "#B41B20")
        self.assertEqual(PRIMARY_BLUE, "#266AA0")
        self.assertEqual(TRANSITION, "#EF8F67")
        self.assertEqual(AUXILIARY, "#FFE181")
        self.assertEqual(normalize_hex_color("#B41B2"), PRIMARY_RED)
        self.assertEqual(normalize_hex_color(" #266AA "), PRIMARY_BLUE)
        self.assertEqual(normalize_hex_color("#abc"), "#AABBCC")
        self.assertEqual(normalize_hex_color("#abcd"), "#AABBCCDD")
        with self.assertRaises(ValueError):
            normalize_hex_color("#12345G")

    def test_color_helpers_and_theme_palette(self) -> None:
        self.assertEqual(blend_colors(PRIMARY_RED, PRIMARY_BLUE, 0), PRIMARY_RED)
        self.assertEqual(blend_colors(PRIMARY_RED, PRIMARY_BLUE, 1), PRIMARY_BLUE)
        self.assertEqual(rgba("#112233", 0.4), (17 / 255, 34 / 255, 51 / 255, 0.4))
        self.assertEqual(rgba_css("#112233", 0.4), "rgba(17, 34, 51, 0.4)")
        theme = VisualizationTheme()
        self.assertEqual(theme.main_colors, (PRIMARY_RED, PRIMARY_BLUE, TRANSITION))
        self.assertEqual(theme.series_colors[:3], (PRIMARY_BLUE, PRIMARY_RED, TRANSITION))
        self.assertEqual(theme.color("red", 0.25)[-1], 0.25)
        self.assertEqual(theme.css_color("blue", 0.25), "rgba(38, 106, 160, 0.25)")
        self.assertEqual(theme.colormap("blue").N, 256)
        self.assertEqual(theme.plotly_colorscale("accent")[1][1], AUXILIARY)
        with self.assertRaises(ValueError):
            theme.colormap("unknown")

    def test_configure_matplotlib_applies_typography_and_theme(self) -> None:
        theme = VisualizationTheme(
            primary_red="#AA1122",
            primary_blue="#1122AA",
            transition="#CC8844",
            auxiliary="#EEDD77",
        )
        configure_matplotlib(font_candidates=("备用字体",), theme=theme, figure_dpi=123, savefig_dpi=234)
        self.assertEqual(mpl.rcParams["font.family"][0], FONT_LATIN)
        self.assertIn(FONT_CJK, mpl.rcParams["font.family"])
        self.assertIn("备用字体", mpl.rcParams["font.family"])
        self.assertEqual(mpl.rcParams["axes.prop_cycle"].by_key()["color"], list(theme.series_colors))
        self.assertEqual(mpl.rcParams["grid.color"], theme.transition)
        self.assertEqual(mpl.rcParams["figure.dpi"], 123)
        self.assertEqual(mpl.rcParams["savefig.dpi"], 234)

    def test_configure_theme_updates_global_theme_and_can_restore(self) -> None:
        original = get_theme()
        try:
            updated = configure_theme(
                primary_red="#123456",
                primary_blue="#654321",
                transition="#AA7733",
                auxiliary="#DDBB66",
                apply=False,
            )
            self.assertEqual(updated.primary_red, "#123456")
            self.assertEqual(updated.primary_blue, "#654321")
            self.assertEqual(get_theme(), updated)
        finally:
            style_module._ACTIVE_THEME = original
            configure_matplotlib(theme=original)

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
