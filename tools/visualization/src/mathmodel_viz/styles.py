"""Shared visual defaults and file output helpers for reproducible figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
)
SUPPORTED_OUTPUT_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}


def configure_matplotlib(
    *,
    font_candidates: tuple[str, ...] = DEFAULT_CJK_FONTS,
    figure_dpi: int = 150,
    savefig_dpi: int = 300,
) -> None:
    """Configure conservative, report-friendly Matplotlib defaults once per process."""

    import matplotlib as mpl

    existing_fonts = tuple(mpl.rcParams["font.sans-serif"])
    fonts = list(dict.fromkeys((*font_candidates, *existing_fonts)))
    mpl.rcParams.update(
        {
            "font.sans-serif": fonts,
            "axes.unicode_minus": False,
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


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
