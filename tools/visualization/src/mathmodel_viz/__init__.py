"""Structured utilities for selecting and producing mathematical-modeling visuals."""

from .catalog import CATALOG
from .registry import DEFAULT_REGISTRY, VisualizationRegistry
from .styles import configure_matplotlib, save_figure

__all__ = [
    "CATALOG",
    "DEFAULT_REGISTRY",
    "VisualizationRegistry",
    "configure_matplotlib",
    "save_figure",
]
