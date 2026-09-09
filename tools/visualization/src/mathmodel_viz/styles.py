"""Shared typography, color theme, and file-output helpers.

The theme is intentionally kept in this module so a project can change the
visual identity once and have static (Matplotlib/Seaborn) and interactive
(Plotly) figures follow the same settings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


# The two five-digit values in the original brief are accepted as input and
# normalized to valid six-digit colors by appending ``0`` (``#B41B2`` ->
# ``#B41B20`` and ``#266AA`` -> ``#266AA0``).
FONT_CJK = "SimSun"
FONT_LATIN = "Times New Roman"
PRIMARY_RED = "#B41B20"
PRIMARY_BLUE = "#266AA0"
TRANSITION = "#EF8F67"
AUXILIARY = "#FFE181"

# Backwards-compatible name retained for callers that supplied CJK candidates
# to ``configure_matplotlib`` in the first version of the package.
DEFAULT_CJK_FONTS = (FONT_CJK,)
DEFAULT_FONT_FAMILY = (FONT_LATIN, FONT_CJK)
SUPPORTED_OUTPUT_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}


def normalize_hex_color(value: str) -> str:
    """Return a normalized ``#RRGGBB``/``#RRGGBBAA`` color.

    Standard three- and four-digit CSS forms are expanded.  Five-digit input
    is accepted for the two colors from the project brief and receives a
    trailing zero so Matplotlib and Plotly can consume it consistently.
    """

    if not isinstance(value, str):
        raise TypeError("color must be a hexadecimal string")
    color = value.strip().upper()
    if not color.startswith("#"):
        raise ValueError(f"color must start with '#': {value!r}")
    digits = color[1:]
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    elif len(digits) == 4:
        digits = "".join(character * 2 for character in digits)
    elif len(digits) == 5:
        digits = f"{digits}0"
    if len(digits) not in (6, 8) or any(character not in "0123456789ABCDEF" for character in digits):
        raise ValueError(f"color must be #RGB, #RGBA, #RRGGBB, or #RRGGBBAA: {value!r}")
    return f"#{digits}"


def _rgb(color: str) -> tuple[int, int, int]:
    """Parse a normalized color into RGB integer channels."""

    normalized = normalize_hex_color(color)
    return tuple(int(normalized[index : index + 2], 16) for index in (1, 3, 5))  # type: ignore[return-value]


def blend_colors(start: str, end: str, amount: float) -> str:
    """Linearly blend two colors in RGB space.

    ``amount=0`` returns ``start`` and ``amount=1`` returns ``end``.  This is
    used for the intermediate palette entries between the red and blue anchors.
    """

    if not 0 <= amount <= 1:
        raise ValueError("amount must be between 0 and 1")
    first = _rgb(start)
    second = _rgb(end)
    channels = [round(left + (right - left) * amount) for left, right in zip(first, second)]
    return "#" + "".join(f"{channel:02X}" for channel in channels)


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    """Return a Matplotlib-compatible RGBA tuple for a theme color."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    red, green, blue = _rgb(color)
    return red / 255, green / 255, blue / 255, alpha


def rgba_css(color: str, alpha: float) -> str:
    """Return a CSS/Plotly ``rgba(...)`` value for a theme color."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    red, green, blue = _rgb(color)
    return f"rgba({red}, {green}, {blue}, {alpha:g})"


@dataclass(frozen=True)
class VisualizationTheme:
    """Global visual identity shared by all rendering backends."""

    primary_red: str = PRIMARY_RED
    primary_blue: str = PRIMARY_BLUE
    transition: str = TRANSITION
    auxiliary: str = AUXILIARY

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_red", normalize_hex_color(self.primary_red))
        object.__setattr__(self, "primary_blue", normalize_hex_color(self.primary_blue))
        object.__setattr__(self, "transition", normalize_hex_color(self.transition))
        object.__setattr__(self, "auxiliary", normalize_hex_color(self.auxiliary))

    @property
    def main_colors(self) -> tuple[str, str, str]:
        """Return the red, blue, and transition anchors in stable order."""

        return self.primary_red, self.primary_blue, self.transition

    @property
    def foreground(self) -> str:
        """A dark readable neutral derived from both primary anchors."""

        return blend_colors(self.primary_red, self.primary_blue, 0.5)

    @property
    def series_colors(self) -> tuple[str, ...]:
        """Colors for categorical series, including transparent variants."""

        return (
            self.primary_blue,
            self.primary_red,
            self.transition,
            blend_colors(self.primary_blue, self.transition, 0.5),
            blend_colors(self.transition, self.primary_red, 0.5),
            self.auxiliary,
        )

    def color(self, name: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
        """Resolve a named theme color as a Matplotlib RGBA tuple."""

        colors = {
            "red": self.primary_red,
            "blue": self.primary_blue,
            "transition": self.transition,
            "auxiliary": self.auxiliary,
            "foreground": self.foreground,
        }
        try:
            value = colors[name]
        except KeyError as exc:
            valid = ", ".join(sorted(colors))
            raise KeyError(f"Unknown theme color {name!r}. Choose one of: {valid}.") from exc
        return rgba(value, alpha)

    def css_color(self, name: str, alpha: float = 1.0) -> str:
        """Resolve a named theme color as a CSS/Plotly RGBA value."""

        colors = {
            "red": self.primary_red,
            "blue": self.primary_blue,
            "transition": self.transition,
            "auxiliary": self.auxiliary,
            "foreground": self.foreground,
        }
        try:
            value = colors[name]
        except KeyError as exc:
            valid = ", ".join(sorted(colors))
            raise KeyError(f"Unknown theme color {name!r}. Choose one of: {valid}.") from exc
        return rgba_css(value, alpha)

    def colormap(self, name: str = "diverging") -> Any:
        """Build a Matplotlib colormap from the theme anchors.

        Supported names are ``diverging``, ``blue``, ``red`` and ``accent``.
        The low-opacity endpoints are RGBA variants of the same theme colors,
        so a heatmap never introduces an unrelated palette.
        """

        from matplotlib.colors import LinearSegmentedColormap

        low_blue = rgba(self.primary_blue, 0.12)
        low_red = rgba(self.primary_red, 0.12)
        palettes = {
            "diverging": [low_blue, self.primary_blue, self.transition, self.primary_red, low_red],
            "blue": [low_blue, self.primary_blue],
            "red": [low_red, self.primary_red],
            "accent": [low_blue, self.auxiliary],
        }
        try:
            colors = palettes[name]
        except KeyError as exc:
            valid = ", ".join(sorted(palettes))
            raise ValueError(f"Unknown theme colormap {name!r}. Choose one of: {valid}.") from exc
        return LinearSegmentedColormap.from_list(f"mathmodel-{name}", colors)

    def plotly_colorscale(self, name: str = "diverging") -> list[list[object]]:
        """Return a Plotly colorscale matching :meth:`colormap`."""

        anchors = {
            "diverging": (
                (0.0, self.css_color("blue", 0.12)),
                (0.25, self.primary_blue),
                (0.5, self.transition),
                (0.75, self.primary_red),
                (1.0, self.css_color("red", 0.12)),
            ),
            "blue": ((0.0, self.css_color("blue", 0.12)), (1.0, self.primary_blue)),
            "red": ((0.0, self.css_color("red", 0.12)), (1.0, self.primary_red)),
            "accent": ((0.0, self.css_color("blue", 0.12)), (1.0, self.auxiliary)),
        }
        try:
            return [[position, color] for position, color in anchors[name]]
        except KeyError as exc:
            valid = ", ".join(sorted(anchors))
            raise ValueError(f"Unknown theme colorscale {name!r}. Choose one of: {valid}.") from exc


DEFAULT_THEME = VisualizationTheme()
_ACTIVE_THEME = DEFAULT_THEME


def get_theme() -> VisualizationTheme:
    """Return the currently active global theme."""

    return _ACTIVE_THEME


def configure_theme(
    *,
    primary_red: str | None = None,
    primary_blue: str | None = None,
    transition: str | None = None,
    auxiliary: str | None = None,
    apply: bool = True,
) -> VisualizationTheme:
    """Update the global theme and optionally apply it to installed backends.

    Values omitted from the call retain their current setting.  Passing
    ``apply=False`` is useful when a caller wants to inspect or serialize the
    new theme before applying it with :func:`configure_matplotlib`.
    """

    global _ACTIVE_THEME
    updates = {
        key: value
        for key, value in {
            "primary_red": primary_red,
            "primary_blue": primary_blue,
            "transition": transition,
            "auxiliary": auxiliary,
        }.items()
        if value is not None
    }
    _ACTIVE_THEME = replace(_ACTIVE_THEME, **updates)
    if apply:
        configure_matplotlib(theme=_ACTIVE_THEME)
    return _ACTIVE_THEME


def configure_plotly(*, theme: VisualizationTheme | None = None) -> None:
    """Install a Plotly template using the active typography and palette."""

    import plotly.graph_objects as go
    import plotly.io as pio

    active = theme or get_theme()
    template = go.layout.Template(
        layout=go.Layout(
            font={"family": f"{FONT_LATIN}, {FONT_CJK}", "color": active.foreground},
            colorway=list(active.series_colors),
            paper_bgcolor=active.css_color("blue", 0.02),
            plot_bgcolor=active.css_color("blue", 0.02),
            xaxis={"gridcolor": active.css_color("transition", 0.18), "zerolinecolor": active.css_color("red", 0.35)},
            yaxis={"gridcolor": active.css_color("transition", 0.18), "zerolinecolor": active.css_color("red", 0.35)},
        )
    )
    pio.templates["mathmodel"] = template
    pio.templates.default = "mathmodel"


def configure_matplotlib(
    *,
    font_candidates: tuple[str, ...] | None = None,
    figure_dpi: int = 150,
    savefig_dpi: int = 300,
    theme: VisualizationTheme | None = None,
) -> None:
    """Apply shared fonts, color cycle, grid, and export defaults.

    ``font_candidates`` remains available for callers that need a platform
    fallback.  English and digits use Times New Roman first; Chinese glyphs
    use SimSun (or the explicitly supplied CJK fallback list).
    """

    import matplotlib as mpl
    from cycler import cycler

    active = theme or get_theme()
    cjk_fonts = tuple(dict.fromkeys((FONT_CJK, *(font_candidates or DEFAULT_CJK_FONTS))))
    font_family = tuple(dict.fromkeys((FONT_LATIN, *cjk_fonts)))
    mpl.rcParams.update(
        {
            "font.family": list(font_family),
            "font.serif": list(font_family),
            "font.sans-serif": list(font_family),
            "axes.unicode_minus": False,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": active.transition,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": rgba(active.primary_blue, 0.55),
            "axes.labelcolor": active.foreground,
            "xtick.color": active.foreground,
            "ytick.color": active.foreground,
            "text.color": active.foreground,
            "patch.edgecolor": active.foreground,
            "patch.facecolor": active.primary_blue,
            "patch.force_edgecolor": True,
            "figure.facecolor": rgba(active.primary_blue, 0.02),
            "axes.facecolor": rgba(active.primary_blue, 0.02),
            "legend.frameon": False,
            "axes.prop_cycle": cycler(color=list(active.series_colors)),
        }
    )

    # Seaborn is optional at import time but part of the core package.  Keep
    # this guarded so the style helper remains usable with Matplotlib alone.
    try:
        import seaborn as sns
    except ImportError:
        pass
    else:
        sns.set_palette(list(active.series_colors))
        sns.set_style(
            "whitegrid",
            {
                "axes.grid": True,
                "grid.color": active.transition,
                "grid.alpha": 0.25,
                "axes.edgecolor": rgba(active.primary_blue, 0.55),
                "axes.facecolor": rgba(active.primary_blue, 0.02),
                "text.color": active.foreground,
                "axes.labelcolor": active.foreground,
                "xtick.color": active.foreground,
                "ytick.color": active.foreground,
                "patch.edgecolor": active.foreground,
                "patch.facecolor": active.primary_blue,
                "patch.force_edgecolor": True,
            },
        )

        # ``set_style`` resets Matplotlib's font family and may replace the
        # color cycle.  Reassert the package-level typography and palette
        # after Seaborn has installed its style defaults.
        mpl.rcParams.update(
            {
                "font.family": list(font_family),
                "font.serif": list(font_family),
                "font.sans-serif": list(font_family),
                "axes.prop_cycle": cycler(color=list(active.series_colors)),
            }
        )

    try:
        configure_plotly(theme=active)
    except ImportError:
        pass


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    dpi: int = 300,
    transparent: bool = False,
) -> Path:
    """Write a figure with a validated extension and create only its direct parent."""

    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in SUPPORTED_OUTPUT_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_SUFFIXES))
        raise ValueError(f"Unsupported figure extension {suffix!r}. Use one of: {supported}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, transparent=transparent, bbox_inches="tight")
    return target


__all__ = [
    "AUXILIARY",
    "DEFAULT_CJK_FONTS",
    "DEFAULT_FONT_FAMILY",
    "DEFAULT_THEME",
    "FONT_CJK",
    "FONT_LATIN",
    "PRIMARY_BLUE",
    "PRIMARY_RED",
    "SUPPORTED_OUTPUT_SUFFIXES",
    "TRANSITION",
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
