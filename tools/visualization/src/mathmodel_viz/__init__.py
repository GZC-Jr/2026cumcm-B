"""Structured utilities for selecting and producing mathematical-modeling visuals."""

from .catalog import CATALOG
from .registry import DEFAULT_REGISTRY, VisualizationRegistry
from .styles import (
    AUXILIARY,
    DEFAULT_THEME,
    FONT_CJK,
    FONT_LATIN,
    PRIMARY_BLUE,
    PRIMARY_RED,
    TRANSITION,
    VisualizationTheme,
    blend_colors,
    configure_matplotlib,
    configure_plotly,
    configure_theme,
    get_theme,
    normalize_hex_color,
    rgba,
    rgba_css,
    save_figure,
)

__all__ = [
    "AUXILIARY",
    "CATALOG",
    "DEFAULT_REGISTRY",
    "DEFAULT_THEME",
    "FONT_CJK",
    "FONT_LATIN",
    "PRIMARY_BLUE",
    "PRIMARY_RED",
    "TRANSITION",
    "VisualizationRegistry",
    "VisualizationTheme",
    "blend_colors",
    "configure_matplotlib",
    "configure_plotly",
    "configure_theme",
    "get_theme",
    "normalize_hex_color",
    "rgba",
    "rgba_css",
    "save_figure",
]
