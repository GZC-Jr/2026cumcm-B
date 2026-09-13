"""Advanced Question 2 sensitivity and validation visualisations.

This module is the paper-facing visualisation layer for the continuous robust
second-detector model.  It combines the formal ``q2_sensitivity`` sweep with
the optional experiment bundle produced by ``enhanced_experiments.py``.

The plotting functions intentionally accept partially populated JSON/CSV
files.  This is useful when a long experiment is interrupted: the figures
still render the completed rows and mark unavailable measurements as gaps.

Example
-------
From the repository root::

    python code/t2/advanced_sensitivity_visualization.py \
        --input outputs/t2/q2_sensitivity.json \
        --enhanced-dir outputs/t2/enhanced_smoke \
        --output-dir outputs/t2/figures/advanced

The enhanced smoke bundle is a workflow check, not evidence of a formal
global optimum.  Every figure derived from it carries an explicit note.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np


# Keep the shared project style as the single source of truth for typography,
# colours and high-DPI export.
VISUALIZATION_SRC = Path(__file__).resolve().parents[2] / "tools" / "visualization" / "src"
if str(VISUALIZATION_SRC) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_SRC))

from mathmodel_viz.styles import configure_matplotlib, get_theme, save_figure  # noqa: E402


DEFAULT_INPUT = "outputs/t2/q2_sensitivity.json"
DEFAULT_ENHANCED_DIR = "outputs/t2/enhanced_smoke"
DEFAULT_OUTPUT_DIR = "outputs/t2/figures/advanced"


def _float(value: Any, default: float = np.nan) -> float:
    """Convert a JSON/CSV value to a finite-or-NaN float."""

    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "strict", "feasible"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _unique_sorted(values: Iterable[Any]) -> list[float]:
    finite = [_float(value) for value in values]
    return sorted({float(value) for value in finite if math.isfinite(value)})


def _point(row: Mapping[str, Any]) -> list[float]:
    raw = row.get("point")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return [_float(raw[0]), _float(raw[1])]
    return [_float(row.get("point_x")), _float(row.get("point_y"))]


def _normalise_tau_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for key in (
        "tau",
        "threshold",
        "area",
        "candidate_area",
        "distance_from_s1_min",
        "distance_from_s1_max",
        "x_step",
        "y_step",
    ):
        if key in result:
            result[key] = _float(result[key])
    if "area" not in result:
        result["area"] = _float(result.get("candidate_area"))
    for key in ("accepted_cell_count", "component_count", "grid_size"):
        if key in result:
            result[key] = _int(result[key], None)
    return result


def _normalise_sensitivity_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable row schema shared by JSON and flattened CSV inputs."""

    result = dict(row)
    result["epsilon_w"] = _float(row.get("epsilon_w"))
    result["p_w"] = _float(row.get("p_w"))
    result["point"] = _point(row)
    for key in (
        "j_star",
        "j_under_reference_weight",
        "distance_from_s1",
        "bearing_from_s1_deg",
        "bearing_difference_deg",
        "reception_h",
        "reception_h_tolerance",
        "candidate_area",
    ):
        if key in row:
            result[key] = _float(row.get(key))
    for key in ("strict_feasible", "feasible_with_tolerance", "use_global_search"):
        if key in row:
            result[key] = _bool(row.get(key))
    candidate_records = row.get("candidate_by_tau", [])
    if not isinstance(candidate_records, list):
        candidate_records = []
    result["candidate_by_tau"] = [_normalise_tau_record(item) for item in candidate_records if isinstance(item, Mapping)]
    return result


def _load_flat_sensitivity_csv(path: Path) -> dict[str, Any]:
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    tau_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            epsilon = _float(record.get("epsilon_w"))
            power = _float(record.get("p_w"))
            if not (math.isfinite(epsilon) and math.isfinite(power)):
                continue
            key = (epsilon, power)
            row = grouped.setdefault(
                key,
                _normalise_sensitivity_row(
                    {
                        "epsilon_w": epsilon,
                        "p_w": power,
                        "point_x": record.get("point_x"),
                        "point_y": record.get("point_y"),
                        "j_star": record.get("j_star"),
                        "j_under_reference_weight": record.get("j_under_reference_weight"),
                        "distance_from_s1": record.get("distance_from_s1"),
                        "bearing_from_s1_deg": record.get("bearing_from_s1_deg"),
                        "bearing_difference_deg": record.get("bearing_difference_deg"),
                        "reception_h": record.get("reception_h"),
                        "reception_h_tolerance": record.get("reception_h_tolerance"),
                        "strict_feasible": record.get("strict_feasible"),
                        "feasible_with_tolerance": record.get("feasible_with_tolerance"),
                        "optimization_source": record.get("optimization_source", ""),
                        "use_global_search": record.get("use_global_search"),
                    }
                ),
            )
            tau = _float(record.get("tau"))
            if math.isfinite(tau):
                tau_values.append(tau)
                row["candidate_by_tau"].append(
                    _normalise_tau_record(
                        {
                            "tau": tau,
                            "threshold": record.get("threshold"),
                            "area": record.get("candidate_area"),
                            "accepted_cell_count": record.get("accepted_cell_count"),
                            "grid_size": record.get("grid_size"),
                            "x_step": record.get("x_step"),
                            "y_step": record.get("y_step"),
                            "distance_from_s1_min": record.get("distance_from_s1_min"),
                            "distance_from_s1_max": record.get("distance_from_s1_max"),
                            "component_count": record.get("component_count"),
                        }
                    )
                )
    rows = list(grouped.values())
    return {
        "model": "question_2_continuous_robust_dop",
        "input": {},
        "config": {},
        "geometry": {},
        "sensitivity": {
            "reference_weight": {"epsilon_w": 0.2, "p_w": 2.0},
            "epsilon_values": _unique_sorted(row["epsilon_w"] for row in rows),
            "p_values": _unique_sorted(row["p_w"] for row in rows),
            "tau_values": _unique_sorted(tau_values),
            "rows": rows,
        },
        "metadata": {"source_format": "flattened_csv"},
    }


def load_sensitivity(path: str | Path) -> dict[str, Any]:
    """Load the formal sensitivity JSON or its flattened CSV companion."""

    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        payload = _load_flat_sensitivity_csv(input_path)
    else:
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    sensitivity = payload.get("sensitivity", {}) if isinstance(payload, Mapping) else {}
    rows = sensitivity.get("rows", []) if isinstance(sensitivity, Mapping) else []
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{input_path} does not contain a non-empty sensitivity.rows array")
    payload = dict(payload)
    sensitivity = dict(sensitivity)
    sensitivity["rows"] = [_normalise_sensitivity_row(row) for row in rows if isinstance(row, Mapping)]
    if not sensitivity["rows"]:
        raise ValueError(f"{input_path} contains no valid sensitivity rows")
    sensitivity["epsilon_values"] = _unique_sorted(
        sensitivity.get("epsilon_values", [row["epsilon_w"] for row in sensitivity["rows"]])
    )
    sensitivity["p_values"] = _unique_sorted(
        sensitivity.get("p_values", [row["p_w"] for row in sensitivity["rows"]])
    )
    sensitivity["tau_values"] = _unique_sorted(
        sensitivity.get(
            "tau_values",
            [item.get("tau") for row in sensitivity["rows"] for item in row.get("candidate_by_tau", [])],
        )
    )
    payload["sensitivity"] = sensitivity
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _boundary_from_csv(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, int, int], list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        tau = _float(row.get("tau"))
        contour_id = _int(row.get("contour_id"), 0) or 0
        grid_size = _int(row.get("grid_size"), 0) or 0
        vertex_id = _int(row.get("vertex_id"), len(grouped[(tau, grid_size, contour_id)])) or 0
        x = _float(row.get("x"))
        y = _float(row.get("y"))
        if math.isfinite(tau) and math.isfinite(x) and math.isfinite(y):
            grouped[(tau, grid_size, contour_id)].append((vertex_id, x, y))
    contours: list[dict[str, Any]] = []
    for (tau, grid_size, contour_id), points in sorted(grouped.items()):
        points.sort(key=lambda item: item[0])
        contours.append(
            {
                "tau": float(tau),
                "grid_size": int(grid_size),
                "contour_id": int(contour_id),
                "points": [[float(x), float(y)] for _, x, y in points],
            }
        )
    return contours


def load_enhanced(
    enhanced_dir: str | Path | None = None,
    enhanced_json: str | Path | None = None,
) -> dict[str, Any]:
    """Load enhanced JSON and fill missing sections from companion CSV files."""

    directory = Path(enhanced_dir) if enhanced_dir else None
    json_path = Path(enhanced_json) if enhanced_json else None
    if json_path is None and directory is not None:
        candidate = directory / "q2_enhanced_experiments.json"
        if candidate.exists():
            json_path = candidate
    if json_path is None or not json_path.exists():
        if directory is None:
            return {}
        json_path = directory / "q2_enhanced_experiments.json"
    payload: dict[str, Any] = {}
    if json_path.exists():
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                payload = dict(loaded)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.warn(f"could not read enhanced JSON {json_path}: {exc}")
    if directory is None:
        directory = json_path.parent

    # CSV files are intentionally consulted even when JSON exists: interrupted
    # runs can leave richer row-level tables than the compact JSON snapshot.
    convergence = dict(payload.get("convergence", {}))
    for key, filename in (
        ("circle_sides", "q2_circle_convergence.csv"),
        ("quadrature_order", "q2_quadrature_convergence.csv"),
        ("candidate_grid", "q2_grid_convergence.csv"),
    ):
        if not convergence.get(key):
            records = _read_csv_rows(directory / filename)
            normalised: list[dict[str, Any]] = []
            for record in records:
                item = dict(record)
                for name in item:
                    if name not in {"experiment", "source", "branch", "setting"}:
                        item[name] = _float(item[name]) if _finite(item[name]) else item[name]
                if "setting" in item:
                    item["setting"] = _float(item["setting"])
                normalised.append(item)
            convergence[key] = normalised
    payload["convergence"] = convergence

    if not payload.get("search_comparison"):
        search = []
        for record in _read_csv_rows(directory / "q2_search_comparison.csv"):
            item = dict(record)
            for key in item:
                if key not in {"experiment", "setting", "source", "branch"}:
                    item[key] = _float(item[key]) if _finite(item[key]) else item[key]
            item["use_global_search"] = _bool(item.get("use_global_search"))
            search.append(item)
        payload["search_comparison"] = search

    boundary_rows = _read_csv_rows(directory / "q2_boundary_contours.csv")
    overlay = dict(payload.get("geometry_overlay", {}))
    if not overlay.get("contours") and boundary_rows:
        overlay["contours"] = _boundary_from_csv(boundary_rows)
    payload["geometry_overlay"] = overlay

    monte = dict(payload.get("monte_carlo", {}))
    if not monte.get("aggregate"):
        records = _read_csv_rows(directory / "q2_monte_carlo_aggregate.csv")
        aggregate: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            for key in item:
                if key != "branch":
                    item[key] = _float(item[key]) if _finite(item[key]) else item[key]
            aggregate.append(item)
        monte["aggregate"] = aggregate
    if not monte.get("seed_summary"):
        records = _read_csv_rows(directory / "q2_monte_carlo_seed_summary.csv")
        summary: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            for key in item:
                if key not in {"branch", "branch_source"}:
                    item[key] = _float(item[key]) if _finite(item[key]) else item[key]
            summary.append(item)
        monte["seed_summary"] = summary
    if not monte.get("samples"):
        records = _read_csv_rows(directory / "q2_monte_carlo_samples.csv")
        samples: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            for key in item:
                if key not in {"branch"}:
                    item[key] = _float(item[key]) if _finite(item[key]) else item[key]
            item["finite_intersection"] = _bool(item.get("finite_intersection"))
            item["second_reception"] = _bool(item.get("second_reception"))
            samples.append(item)
        monte["samples"] = samples
    payload["monte_carlo"] = monte

    # ``quick`` is the authoritative flag for current bundles.  The directory
    # name remains a compatibility fallback: older low-cost bundles were
    # sometimes produced with ``quick=False`` despite being written to an
    # ``enhanced_smoke`` directory.  Keeping that fallback prevents those
    # figures from being presented as formal evidence.
    design = payload.get("design", {})
    smoke = bool(isinstance(design, Mapping) and design.get("quick"))
    if not smoke:
        smoke = "smoke" in str(directory).lower() or "quick" in str(directory).lower()
    payload["_source_dir"] = str(directory)
    payload["_is_smoke"] = smoke
    return payload


def _sensitivity(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("sensitivity", {})
    if not isinstance(value, Mapping):
        raise ValueError("sensitivity section is missing")
    rows = value.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError("sensitivity.rows is empty")
    return dict(value)


def _axes_values(sensitivity: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    rows = sensitivity.get("rows", [])
    eps = _unique_sorted(sensitivity.get("epsilon_values", [row.get("epsilon_w") for row in rows]))
    powers = _unique_sorted(sensitivity.get("p_values", [row.get("p_w") for row in rows]))
    return eps, powers


def _reference_row(sensitivity: Mapping[str, Any]) -> dict[str, Any]:
    reference = sensitivity.get("reference_weight", {})
    if not isinstance(reference, Mapping):
        reference = {}
    epsilon = _float(reference.get("epsilon_w"), 0.2)
    power = _float(reference.get("p_w"), 2.0)
    rows = sensitivity.get("rows", [])
    return min(
        rows,
        key=lambda row: abs(_float(row.get("epsilon_w")) - epsilon) + abs(_float(row.get("p_w")) - power),
    )


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    epsilons: Sequence[float],
    powers: Sequence[float],
    field: str,
    *,
    tau: float | None = None,
) -> np.ndarray:
    matrix = np.full((len(epsilons), len(powers)), np.nan, dtype=float)
    for row in rows:
        epsilon = _float(row.get("epsilon_w"))
        power = _float(row.get("p_w"))
        if not (math.isfinite(epsilon) and math.isfinite(power)) or not epsilons or not powers:
            continue
        i = int(np.argmin(np.abs(np.asarray(epsilons) - epsilon)))
        j = int(np.argmin(np.abs(np.asarray(powers) - power)))
        value: Any = row.get(field)
        if field == "point_x":
            value = _point(row)[0]
        elif field == "point_y":
            value = _point(row)[1]
        elif field in {"candidate_area", "component_count", "accepted_cell_count"}:
            record = _candidate_record(row, tau)
            if record is not None:
                value = record.get("area" if field == "candidate_area" else field)
        if _finite(value):
            matrix[i, j] = float(value)
    return matrix


def _candidate_record(row: Mapping[str, Any], tau: float | None = None) -> dict[str, Any] | None:
    records = row.get("candidate_by_tau", [])
    if not isinstance(records, list) or not records:
        return None
    valid = [item for item in records if isinstance(item, Mapping) and _finite(item.get("tau"))]
    if not valid:
        return None
    target = tau if tau is not None and math.isfinite(float(tau)) else None
    if target is None:
        return dict(valid[0])
    return dict(min(valid, key=lambda item: abs(_float(item.get("tau")) - float(target))))


def _reference_tau(payload: Mapping[str, Any], sensitivity: Mapping[str, Any]) -> float:
    config = payload.get("config", {})
    if isinstance(config, Mapping) and _finite(config.get("tau")):
        return _float(config.get("tau"))
    values = _unique_sorted(sensitivity.get("tau_values", []))
    return values[len(values) // 2] if values else 0.05


def _smoke_note(enhanced: Mapping[str, Any] | None) -> str:
    if not enhanced or not enhanced.get("_is_smoke"):
        return ""
    return "注：增强实验为低成本 smoke 数据，仅用于流程验证，不代表正式全局最优或高精度收敛结果。"


def _title(base: str, note: str = "") -> str:
    return f"{base}（低成本 smoke 数据）" if note else base


def _safe_rel_span(values: Iterable[Any]) -> float:
    numbers = np.asarray([_float(value) for value in values], dtype=float)
    numbers = numbers[np.isfinite(numbers)]
    if len(numbers) < 2:
        return 0.0
    scale = max(1.0, abs(float(np.nanmedian(numbers))))
    return 100.0 * float(np.nanmax(numbers) - np.nanmin(numbers)) / scale


def _set_grid(axis: plt.Axes) -> None:
    axis.grid(True, axis="y", alpha=0.25)
    axis.grid(False, axis="x")


def _set_distance_formatter(axis: plt.Axes) -> None:
    """Render metre-scale distances without an offset/scientific prefix.

    The robust reception constraint pins most optima very close to 1000 m.
    Matplotlib's default ``1e-7 + 1e3`` offset notation is technically
    correct but visually obscures that all curves are on the same boundary.
    Keeping the absolute unit on the ticks makes the plot auditable.
    """

    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))


def _finish(figure: plt.Figure, path: Path, note: str = "") -> Path:
    if note:
        figure.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8, color=get_theme().foreground, alpha=0.86)
        rect = (0.0, 0.045, 1.0, 0.91)
    else:
        rect = (0.0, 0.0, 1.0, 0.93)
    figure.tight_layout(rect=rect)
    saved = save_figure(figure, path)
    plt.close(figure)
    return saved


def _fmt_number(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _csv_value(value: Any) -> Any:
    """Keep text/bools intact while converting non-finite numeric values to blank."""

    if isinstance(value, str) or isinstance(value, (bool, np.bool_)):
        return value
    if value is None:
        return ""
    try:
        return value if math.isfinite(float(value)) else ""
    except (TypeError, ValueError):
        return value


def _annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, *, percent: bool = False) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isfinite(matrix[i, j]):
                continue
            value = matrix[i, j]
            if percent:
                label = f"{value:.0%}" if 0.0 <= value <= 1.0 else f"{value:.1f}"
            else:
                magnitude = abs(value)
                digits = 1 if magnitude >= 100 else 2
                label = f"{value:.{digits}f}"
            axis.text(j, i, label, ha="center", va="center", fontsize=8, color=get_theme().foreground)


def _format_parameter_axes(axis: plt.Axes, epsilons: Sequence[float], powers: Sequence[float]) -> None:
    axis.set_xticks(np.arange(len(powers)))
    axis.set_xticklabels([f"{value:g}" for value in powers])
    axis.set_yticks(np.arange(len(epsilons)))
    axis.set_yticklabels([f"{value:g}" for value in epsilons])
    # Keep parameter names in plain text.  Mixing CJK glyphs with mathtext
    # makes Matplotlib select its ``default`` math font for the whole label on
    # some platforms, which produces missing-glyph boxes in exported figures.
    axis.set_xlabel("深度权重幂次 p_w")
    axis.set_ylabel("边界基础权重 epsilon_w")


def _plot_empty(axis: plt.Axes, message: str = "无可用数据") -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center", color=get_theme().foreground)
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    epsilons: Sequence[float],
    powers: Sequence[float],
    title: str,
    cmap: Any,
    *,
    percent: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> Any:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        _plot_empty(axis)
        axis.set_title(title)
        return None
    if vmin is None:
        vmin = float(np.nanmin(finite))
    if vmax is None:
        vmax = float(np.nanmax(finite))
    if abs(vmax - vmin) <= 1e-12:
        # A constant image still needs a non-degenerate colour scale for a
        # readable colourbar.
        pad = max(1.0, abs(vmin) * 0.01)
        vmin, vmax = vmin - pad, vmax + pad
    image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    _format_parameter_axes(axis, epsilons, powers)
    axis.set_title(title)
    if annotate:
        _annotate_heatmap(axis, matrix, percent=percent)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=8)
    # Avoid the ``1e-7 + 1e3`` offset notation for distance maps whose
    # values are constrained very close to 1000 m.  Absolute units remain
    # visible, while tiny deviations are still encoded by the colour.
    colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f" if np.nanmax(np.abs(finite)) < 10000 else "%.0f"))
    colorbar.update_ticks()
    return image


def _records_by_tau(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = row.get("candidate_by_tau", [])
    if not isinstance(records, list):
        return []
    valid = [dict(item) for item in records if isinstance(item, Mapping) and _finite(item.get("tau"))]
    return sorted(valid, key=lambda item: _float(item.get("tau")))


def _value(row: Mapping[str, Any], key: str, default: float = np.nan) -> float:
    return _float(row.get(key), default)


def _candidate_value(row: Mapping[str, Any], tau: float, key: str, default: float = np.nan) -> float:
    record = _candidate_record(row, tau)
    if record is None:
        return default
    # Keep the helper tolerant of both schemas used in the project: the
    # flattened summary calls the metric ``candidate_area`` while a
    # ``candidate_by_tau`` record stores it as ``area``.
    lookup_key = "area" if key == "candidate_area" else key
    value = record.get(lookup_key)
    # Accept an un-normalised record as well; this makes the private helper
    # safe for callers that construct a row directly instead of going through
    # ``load_sensitivity`` first.
    if key == "candidate_area" and not _finite(value):
        value = record.get("candidate_area")
    return _float(value, default)


def _draw_reference_line(axis: plt.Axes, value: float, label: str, color: Any, *, orientation: str = "h") -> None:
    if not math.isfinite(value):
        return
    if orientation == "v":
        axis.axvline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.75, label=label)
    else:
        axis.axhline(value, color=color, linestyle="--", linewidth=1.0, alpha=0.75, label=label)


def plot_parameter_sweep(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Four-panel sweep with a common reference objective.

    ``j_under_reference_weight`` is deliberately used in all cross-weight
    comparisons.  The native ``j_star`` is shown only in the heatmap figure.
    """

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    reference_row = _reference_row(sensitivity)
    tau = _reference_tau(payload, sensitivity)
    theme = get_theme()
    colors = theme.series_colors
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(2, 2, figsize=(13.4, 9.1), constrained_layout=False)

    for index, power in enumerate(powers):
        selected = sorted(
            [row for row in rows if abs(_value(row, "p_w") - power) <= 1e-9],
            key=lambda row: _value(row, "epsilon_w"),
        )
        if not selected:
            continue
        color = colors[index % len(colors)]
        x = np.asarray([_value(row, "epsilon_w") for row in selected])
        series = (
            (["j_under_reference_weight", "J0(P*) / m", axes[0, 0]], "o-"),
            (["distance_from_s1", "distance from S1 / m", axes[0, 1]], "s-"),
            (["bearing_difference_deg", "bearing difference / deg", axes[1, 0]], "^-"),
        )
        for (field, ylabel, axis), style in series:
            y = np.asarray([_value(row, field) for row in selected])
            finite = np.isfinite(y)
            if np.any(finite):
                axis.plot(x[finite], y[finite], style, color=color, linewidth=1.7, markersize=5, label=f"p_w={power:g}")
        area = np.asarray([_candidate_value(row, tau, "area") for row in selected])
        finite_area = np.isfinite(area)
        if np.any(finite_area):
            axes[1, 1].plot(x[finite_area], area[finite_area], "o-", color=color, linewidth=1.7, markersize=5, label=f"p_w={power:g}")

    labels = (
        (axes[0, 0], "统一参考权重下的目标值", "epsilon_w", "J0(P*) / m"),
        (axes[0, 1], "最优点距离", "epsilon_w", "distance from S1 / m"),
        (axes[1, 0], "最优点方位偏差", "epsilon_w", "bearing difference / deg"),
        (axes[1, 1], f"近优候选面积（tau={tau:g}）", "epsilon_w", "grid / marching area / m^2"),
    )
    for axis, title, xlabel, ylabel in labels:
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if epsilons:
            axis.set_xticks(epsilons)
        _set_grid(axis)
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8, ncol=min(3, len(handles)))
    if math.isfinite(_value(reference_row, "epsilon_w")):
        for axis in axes.ravel():
            axis.axvline(_value(reference_row, "epsilon_w"), color=theme.foreground, linestyle=":", linewidth=0.9, alpha=0.7)
    _set_distance_formatter(axes[0, 1])
    figure.suptitle("问题二参数扫描：位置、性能与候选区域的联动", fontsize=14)
    return _finish(figure, output_dir / "q2_advanced_parameter_sweep.png", note)


def plot_weight_heatmaps(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Six aligned heatmaps for native/reference performance and feasibility."""

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    tau = _reference_tau(payload, sensitivity)
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.2))
    matrices = (
        (_matrix(rows, epsilons, powers, "j_star"), "各组自身权重 J*", theme.colormap("blue"), False),
        (_matrix(rows, epsilons, powers, "j_under_reference_weight"), "统一参考权重 J0(P*)", theme.colormap("red"), False),
        (_matrix(rows, epsilons, powers, "distance_from_s1"), "最优距离 / m", theme.colormap("accent"), False),
        (_matrix(rows, epsilons, powers, "bearing_difference_deg"), "方位偏差 / deg", theme.colormap("diverging"), False),
        (_matrix(rows, epsilons, powers, "candidate_area", tau=tau), f"候选面积 / m^2 (tau={tau:g})", theme.colormap("blue"), False),
        (_matrix(rows, epsilons, powers, "component_count", tau=tau), f"连通分量数 (tau={tau:g})", theme.colormap("accent"), False),
    )
    for axis, (matrix, title, cmap, percent) in zip(axes.ravel(), matrices):
        _plot_heatmap(axis, matrix, epsilons, powers, title, cmap, percent=percent)
    figure.suptitle("epsilon_w × p_w 参数平面：性能、位置与候选拓扑", fontsize=14)
    return _finish(figure, output_dir / "q2_weight_parameter_heatmaps.png", note)


def plot_feasibility_heatmap(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Strict and tolerance feasibility displayed side-by-side."""

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    strict = _matrix(rows, epsilons, powers, "strict_feasible")
    tolerant = _matrix(rows, epsilons, powers, "feasible_with_tolerance")
    # Boolean matrices enter ``_matrix`` as 0/1; retain NaN for missing rows.
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for axis, matrix, title in zip(axes, (strict, tolerant), ("严格可行 H≤0", "允许数值容差 H≤tol")):
        _plot_heatmap(axis, matrix, epsilons, powers, title, theme.colormap("blue"), vmin=0.0, vmax=1.0, annotate=False)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    mark = "✓" if matrix[i, j] >= 0.5 else "×"
                    axis.text(j, i, mark, ha="center", va="center", fontsize=13, color=theme.foreground)
    figure.suptitle("接收约束可行性：严格判定与数值容差分开报告", fontsize=13)
    return _finish(figure, output_dir / "q2_feasibility_heatmaps.png", note)


def plot_tau_evolution(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Show how the near-optimal level set changes as tau increases."""

    sensitivity = _sensitivity(payload)
    row = _reference_row(sensitivity)
    records = _records_by_tau(row)
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    if not records:
        for axis in axes.ravel():
            _plot_empty(axis, "主敏感度文件未提供 candidate_by_tau")
        figure.suptitle("tau 候选区域演化")
        return _finish(figure, output_dir / "q2_tau_candidate_evolution.png", note)
    tau = np.asarray([_float(item.get("tau")) for item in records])
    area = np.asarray([_float(item.get("area")) for item in records])
    dmin = np.asarray([_float(item.get("distance_from_s1_min")) for item in records])
    dmax = np.asarray([_float(item.get("distance_from_s1_max")) for item in records])
    comp = np.asarray([_float(item.get("component_count")) for item in records])
    threshold = np.asarray([_float(item.get("threshold")) for item in records])
    accepted = np.asarray([_float(item.get("accepted_cell_count")) for item in records])

    finite = np.isfinite(area)
    if np.any(finite):
        axes[0, 0].plot(tau[finite], area[finite], "o-", color=theme.primary_blue, linewidth=2, label="候选面积")
        axes[0, 0].plot(tau[finite], accepted[finite] / max(1.0, np.nanmax(accepted[finite])) * np.nanmax(area[finite]), "s--", color=theme.transition, linewidth=1.3, label="接受单元数（归一化）")
        axes[0, 0].set_ylabel("面积 / m^2")
        axes[0, 0].legend(fontsize=8)
    else:
        _plot_empty(axes[0, 0])
    axes[0, 0].set_title("候选区域面积随 tau 扩张")

    valid_range = np.isfinite(dmin) & np.isfinite(dmax)
    if np.any(valid_range):
        axes[0, 1].fill_between(tau[valid_range], dmin[valid_range], dmax[valid_range], color=theme.color("transition", 0.22), label="距离范围")
        axes[0, 1].plot(tau[valid_range], dmin[valid_range], "o-", color=theme.primary_blue, label="L_min")
        axes[0, 1].plot(tau[valid_range], dmax[valid_range], "s-", color=theme.primary_red, label="L_max")
        axes[0, 1].legend(fontsize=8)
    else:
        _plot_empty(axes[0, 1], "无有效距离范围")
    axes[0, 1].set_title("候选点到 S1 的距离范围")
    axes[0, 1].set_ylabel("distance / m")

    if np.any(np.isfinite(comp)):
        axes[1, 0].step(tau, comp, where="mid", color=theme.primary_red, linewidth=2, marker="o")
        axes[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        _plot_empty(axes[1, 0])
    axes[1, 0].set_title("连通分量数（网格拓扑近似）")
    axes[1, 0].set_ylabel("component count")

    valid_threshold = np.isfinite(threshold)
    if np.any(valid_threshold):
        axes[1, 1].plot(tau[valid_threshold], threshold[valid_threshold], "o-", color=theme.primary_red, linewidth=2, label="(1+tau)J*")
        j_star = _value(row, "j_star")
        if math.isfinite(j_star):
            axes[1, 1].axhline(j_star, color=theme.primary_blue, linestyle="--", linewidth=1.2, label="J*")
        axes[1, 1].legend(fontsize=8)
    else:
        _plot_empty(axes[1, 1])
    axes[1, 1].set_title("目标阈值随 tau 的增长")
    axes[1, 1].set_ylabel("objective / m")

    for axis in axes.ravel():
        axis.set_xlabel("近优容差 tau")
        _set_grid(axis)
    figure.suptitle(
        "候选区域演化：参考权重 "
        f"epsilon_w={_value(row, 'epsilon_w'):g}, p_w={_value(row, 'p_w'):g}"
    )
    return _finish(figure, output_dir / "q2_tau_candidate_evolution.png", note)


def plot_tradeoff(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Scatter view of objective-position trade-offs and branch changes."""

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    tau = _reference_tau(payload, sensitivity)
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    eps_values = np.asarray([_value(row, "epsilon_w") for row in rows])
    finite_eps = eps_values[np.isfinite(eps_values)]
    norm = Normalize(vmin=float(np.nanmin(finite_eps)) if finite_eps.size else 0.0, vmax=float(np.nanmax(finite_eps)) if finite_eps.size else 1.0)
    cmap = theme.colormap("diverging")
    markers = ["o", "s", "^", "D", "P", "X"]
    powers = _axes_values(sensitivity)[1]
    for row in rows:
        x = _value(row, "j_under_reference_weight")
        y_distance = _value(row, "distance_from_s1")
        y_bearing = _value(row, "bearing_difference_deg")
        epsilon = _value(row, "epsilon_w")
        power = _value(row, "p_w")
        size_area = _candidate_value(row, tau, "area")
        marker = markers[int(np.argmin(np.abs(np.asarray(powers) - power))) % len(markers)] if powers else "o"
        size = 42.0 if not math.isfinite(size_area) else 36.0 + 90.0 * math.sqrt(max(0.0, size_area) / max(1.0, np.nanmax([_candidate_value(item, tau, "area") for item in rows if math.isfinite(_candidate_value(item, tau, "area"))] or [1.0])))
        color = cmap(norm(epsilon)) if math.isfinite(epsilon) else theme.primary_blue
        if math.isfinite(x) and math.isfinite(y_distance):
            axes[0].scatter(x, y_distance, s=size, marker=marker, color=color, alpha=0.82, edgecolors=theme.foreground, linewidths=0.35)
        if math.isfinite(x) and math.isfinite(y_bearing):
            axes[1].scatter(x, y_bearing, s=size, marker=marker, color=color, alpha=0.82, edgecolors=theme.foreground, linewidths=0.35)
    reference = _reference_row(sensitivity)
    ref_x = _value(reference, "j_under_reference_weight")
    if math.isfinite(ref_x):
        for axis, y in ((axes[0], _value(reference, "distance_from_s1")), (axes[1], _value(reference, "bearing_difference_deg"))):
            if math.isfinite(y):
                axis.scatter([ref_x], [y], s=160, facecolors="none", edgecolors=theme.primary_red, linewidths=1.8, marker="*")
    axes[0].set_xlabel("统一参考目标 J0(P*) / m")
    axes[0].set_ylabel("distance from S1 / m")
    axes[0].set_title("性能—距离权衡")
    axes[1].set_xlabel("统一参考目标 J0(P*) / m")
    axes[1].set_ylabel("bearing difference / deg")
    axes[1].set_title("性能—方位权衡")
    for axis in axes:
        _set_grid(axis)
    legend_handles = [
        Line2D([0], [0], marker=markers[i % len(markers)], color="none", markerfacecolor=theme.primary_blue, markeredgecolor=theme.foreground, markersize=6, label=f"p_w={power:g}")
        for i, power in enumerate(powers)
    ]
    if legend_handles:
        axes[1].legend(handles=legend_handles, title="点形状：p_w", fontsize=8, title_fontsize=8, loc="best")
    figure.suptitle("位置与性能的多目标权衡（点色：epsilon_w；点大小：候选面积）")
    return _finish(figure, output_dir / "q2_objective_position_tradeoff.png", note)


def _metric_value(row: Mapping[str, Any], field: str, *, tau: float | None = None) -> float:
    """Read a sensitivity metric, including branch-invariant derived fields."""

    if field == "bearing_difference_abs_deg":
        value = _value(row, "bearing_difference_deg")
        return abs(value) if math.isfinite(value) else np.nan
    # The top-level sensitivity schema calls this metric ``candidate_area``
    # while each ``candidate_by_tau`` record stores the same quantity under
    # the shorter field name ``area``.  Keep the public metric name explicit
    # here and translate it at the record boundary; passing
    # ``candidate_area`` through to ``_candidate_value`` would silently
    # return NaN for JSON payloads and make the tornado bar appear as 0%.
    if field == "candidate_area":
        return _candidate_value(row, float(0.05 if tau is None else tau), field)
    if field == "component_count":
        return _candidate_value(row, float(0.05 if tau is None else tau), "component_count")
    return _value(row, field)


def _sensitivity_score(rows: Sequence[Mapping[str, Any]], field: str, *, tau: float | None = None) -> float:
    values: list[float] = []
    for row in rows:
        value = _metric_value(row, field, tau=tau)
        if math.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return np.nan
    baseline = max(1.0, abs(float(np.nanmedian(values))))
    return 100.0 * (float(np.nanmax(values)) - float(np.nanmin(values))) / baseline


def plot_tornado(payload: Mapping[str, Any], output_dir: Path, enhanced: Mapping[str, Any] | None = None) -> Path:
    """Normalised range (tornado) summary for model parameters."""

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    tau = _reference_tau(payload, sensitivity)
    metrics = [
        ("统一参考目标 J0", "j_under_reference_weight", "性能"),
        ("最优距离", "distance_from_s1", "位置"),
        # Use the absolute angular deviation for the tornado score.  A sign
        # flip in the signed panel is often just a switch between the two
        # mirror-image optima, and should not be reported as a 200% physical
        # sensitivity effect.
        ("方位偏差绝对值", "bearing_difference_abs_deg", "位置"),
        ("候选面积", "candidate_area", "拓扑"),
        ("连通分量", "component_count", "拓扑"),
    ]
    scores = np.asarray([_sensitivity_score(rows, field, tau=tau) for _, field, _ in metrics], dtype=float)
    order = np.argsort(np.nan_to_num(scores, nan=-1.0))
    labels = [metrics[i][0] for i in order]
    values = scores[order]
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    colors = [theme.primary_blue if metrics[i][2] == "性能" else theme.transition if metrics[i][2] == "位置" else theme.primary_red for i in order]
    bars = axis.barh(np.arange(len(labels)), np.nan_to_num(values, nan=0.0), color=colors, alpha=0.82)
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_xlabel("跨扫描相对范围 (%) = (max−min)/median")
    axis.set_title("参数敏感度龙卷风图（同一指标内归一化）")
    for bar, value in zip(bars, values):
        if math.isfinite(value):
            axis.text(float(bar.get_width()) + max(0.2, np.nanmax(np.nan_to_num(values, nan=0.0)) * 0.012), bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=9)
    _set_grid(axis)
    axis.grid(True, axis="x", alpha=0.25)
    axis.grid(False, axis="y")
    axis.text(
        0.99,
        0.02,
        "方位偏差取绝对值；符号翻转按镜像分支切换处理",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=theme.foreground,
    )
    figure.suptitle("epsilon_w、p_w 对结果指标的相对敏感度")
    return _finish(figure, output_dir / "q2_parameter_sensitivity_tornado.png", note)


def _enhanced_section(enhanced: Mapping[str, Any] | None, key: str) -> list[dict[str, Any]]:
    if not enhanced:
        return []
    convergence = enhanced.get("convergence", {})
    if key == "search_comparison":
        raw = enhanced.get("search_comparison", [])
    else:
        raw = convergence.get(key, []) if isinstance(convergence, Mapping) else []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _setting(item: Mapping[str, Any]) -> float:
    return _float(item.get("setting"), _float(item.get("grid_size")))


def _series_from_rows(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [(_setting(row), _float(row.get(field))) for row in rows]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    pairs.sort(key=lambda pair: pair[0])
    if not pairs:
        return np.asarray([]), np.asarray([])
    return np.asarray([pair[0] for pair in pairs]), np.asarray([pair[1] for pair in pairs])


def _plot_convergence_panel(axis: plt.Axes, rows: Sequence[Mapping[str, Any]], x_label: str, title: str, fields: Sequence[tuple[str, str, Any, str]]) -> None:
    plotted = False
    for field, label, color, style in fields:
        x, y = _series_from_rows(rows, field)
        if len(x):
            axis.plot(x, y, style, color=color, linewidth=1.8, markersize=5, label=label)
            plotted = True
    if not plotted:
        _plot_empty(axis, "无可用收敛记录")
    axis.set_xlabel(x_label)
    axis.set_title(title)
    if plotted:
        axis.legend(fontsize=8)
        _set_grid(axis)


def plot_convergence_suite(enhanced: Mapping[str, Any] | None, output_dir: Path) -> Path:
    """Six-panel numerical convergence diagnostics."""

    theme = get_theme()
    note = _smoke_note(enhanced)
    circles = _enhanced_section(enhanced, "circle_sides")
    quadrature = _enhanced_section(enhanced, "quadrature_order")
    grids = _enhanced_section(enhanced, "candidate_grid")
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
    _plot_convergence_panel(
        axes[0, 0], circles, "circle_sides", "圆边数：目标值收敛",
        (("native_j_star", "native J*", theme.primary_blue, "o-"), ("j_at_reference_order", "reference-order J", theme.primary_red, "s--")),
    )
    _plot_convergence_panel(
        axes[0, 1], circles, "circle_sides", "圆边数：区域几何收敛",
        (("region_area", "target area", theme.transition, "o-"), ("inradius", "inradius", theme.primary_blue, "s--")),
    )
    _plot_convergence_panel(
        axes[0, 2], quadrature, "quadrature_order", "积分阶数：目标值收敛",
        (("native_j_star", "native J*", theme.primary_blue, "o-"), ("j_at_reference_order", "reference-order J", theme.primary_red, "s--")),
    )
    _plot_convergence_panel(
        axes[1, 0], quadrature, "quadrature_order", "积分阶数：相对差异",
        (("relative_j_native_vs_last", "|ΔJ|/J_ref", theme.primary_red, "o-"),),
    )
    # The fourth panel above intentionally uses one field; relabel its y axis
    # explicitly and avoid duplicating a legend entry.
    axes[1, 0].set_ylabel("relative difference")
    axes[1, 0].set_yscale("linear")
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1%}" if value >= 0 else f"{value:.1%}"))
    axes[1, 0].set_ylabel("relative difference")
    _plot_convergence_panel(
        axes[1, 1], grids, "grid size", "网格分辨率：候选面积",
        (("cell_center_area", "cell-centre area", theme.primary_blue, "o-"), ("marching_area", "marching-squares area", theme.primary_red, "s--")),
    )
    _plot_convergence_panel(
        axes[1, 2], grids, "grid size", "网格分辨率：边界与拓扑",
        (("boundary_length", "boundary length", theme.transition, "o-"), ("contour_count", "contour count", theme.primary_red, "s--")),
    )
    for axis in axes.ravel():
        axis.set_ylabel(axis.get_ylabel() or "value")
    figure.suptitle("第二问数值收敛诊断：圆边数、积分阶数与网格分辨率", fontsize=14)
    return _finish(figure, output_dir / "q2_convergence_suite.png", note)


def _table_rows(rows: Sequence[Mapping[str, Any]], experiment: str) -> list[list[str]]:
    output: list[list[str]] = []
    for row in sorted(rows, key=_setting):
        setting = _setting(row)
        if not math.isfinite(setting):
            continue
        output.append(
            [
                experiment,
                f"{setting:g}",
                _fmt_number(_float(row.get("native_j_star")), 2),
                _fmt_number(_float(row.get("j_at_reference_order")), 2),
                _fmt_number(100.0 * _float(row.get("relative_j_native_vs_last")), 3),
                _fmt_number(_float(row.get("region_area" if experiment == "circle_sides" else "cell_center_area")), 1),
                _fmt_number(_float(row.get("marching_area")), 1),
            ]
        )
    return output


def write_convergence_summary(enhanced: Mapping[str, Any] | None, output_dir: Path) -> Path:
    """Write a compact, paper-ready CSV in addition to the source tables."""

    path = output_dir / "q2_convergence_formal_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "setting", "native_j_star", "j_at_reference_order", "relative_delta_percent", "area_or_region", "marching_area"]
    records: list[dict[str, Any]] = []
    for experiment, key in (("circle_sides", "circle_sides"), ("quadrature_order", "quadrature_order"), ("candidate_grid", "candidate_grid")):
        for row in _enhanced_section(enhanced, key):
            records.append(
                {
                    "experiment": experiment,
                    "setting": _setting(row),
                    "native_j_star": _float(row.get("native_j_star")),
                    "j_at_reference_order": _float(row.get("j_at_reference_order")),
                    "relative_delta_percent": 100.0 * _float(row.get("relative_j_native_vs_last")),
                    "area_or_region": _float(row.get("region_area" if experiment == "circle_sides" else "cell_center_area")),
                    "marching_area": _float(row.get("marching_area")),
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(value) for key, value in record.items()})
    return path


def plot_convergence_table(enhanced: Mapping[str, Any] | None, output_dir: Path) -> Path:
    """Render the convergence records as three compact formal tables."""

    theme = get_theme()
    note = _smoke_note(enhanced)
    figure = plt.figure(figsize=(15.5, 10.0))
    grid = figure.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 1.0), hspace=0.38)
    specs = (("circle_sides", "circle_sides 收敛表"), ("quadrature_order", "quadrature_order 收敛表"), ("candidate_grid", "candidate_grid / marching-squares 收敛表"))
    headers = ["实验", "设置", "native J*", "参考阶 J", "相对差异(%)", "面积/区域", "marching 面积"]
    for slot, (key, title) in enumerate(specs):
        axis = figure.add_subplot(grid[slot, 0])
        axis.axis("off")
        rows = _table_rows(_enhanced_section(enhanced, key), key)
        if not rows:
            axis.text(0.5, 0.5, "无可用记录", ha="center", va="center", color=theme.foreground)
            continue
        table = axis.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", colLoc="center", colWidths=[0.14, 0.10, 0.13, 0.13, 0.14, 0.17, 0.19])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.55)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor(theme.color("blue", 0.35))
            cell.set_linewidth(0.45)
            if row_index == 0:
                cell.set_facecolor(theme.color("blue", 0.14))
                cell.set_text_props(weight="bold", color=theme.foreground)
            else:
                cell.set_facecolor(theme.color("blue", 0.025 if row_index % 2 else 0.06))
        axis.set_title(title, loc="left", pad=6, color=theme.foreground)
    # Keep the provenance in the title itself.  The previous wording called
    # every table a "formal" table even when a quick/smoke bundle was passed,
    # which made it too easy to copy the figure into a paper with the wrong
    # evidence label.
    table_title = "收敛表（smoke 数据，仅作流程验证）" if note else "正式收敛表"
    figure.suptitle(table_title, fontsize=14, y=0.985)
    if note:
        figure.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8, color=theme.foreground)
        figure.subplots_adjust(top=0.95, bottom=0.055, left=0.03, right=0.98)
    else:
        figure.subplots_adjust(top=0.95, bottom=0.03, left=0.03, right=0.98)
    saved = save_figure(figure, output_dir / "q2_convergence_formal_table.png")
    plt.close(figure)
    return saved


def _search_rows(enhanced: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    rows = _enhanced_section(enhanced, "search_comparison")
    for row in rows:
        row["seed"] = _int(row.get("seed"), 0) or 0
        row["use_global_search"] = _bool(row.get("use_global_search"))
        row["point_x"] = _float(row.get("point_x"))
        row["point_y"] = _float(row.get("point_y"))
        row["j_star"] = _float(row.get("j_star"))
        row["distance_from_s1"] = _float(row.get("distance_from_s1"))
        row["selected_iterations"] = _float(row.get("selected_iterations"))
        row["selected_function_evaluations"] = _float(row.get("selected_function_evaluations"))
    return rows


def _mc_aggregate(enhanced: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not enhanced:
        return []
    monte = enhanced.get("monte_carlo", {})
    rows = monte.get("aggregate", []) if isinstance(monte, Mapping) else []
    result: list[dict[str, Any]] = []
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        for key, value in list(row.items()):
            if key != "branch":
                row[key] = _float(value)
        result.append(row)
    return result


def _mc_summary(enhanced: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not enhanced:
        return []
    monte = enhanced.get("monte_carlo", {})
    rows = monte.get("seed_summary", []) if isinstance(monte, Mapping) else []
    result: list[dict[str, Any]] = []
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        for key, value in list(row.items()):
            if key not in {"branch", "branch_source"}:
                row[key] = _float(value)
        result.append(row)
    return result


def _mc_samples(enhanced: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not enhanced:
        return []
    monte = enhanced.get("monte_carlo", {})
    rows = monte.get("samples", []) if isinstance(monte, Mapping) else []
    result: list[dict[str, Any]] = []
    for item in rows if isinstance(rows, list) else []:
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row["error_m"] = _float(row.get("error_m"))
        row["branch"] = str(row.get("branch", "branch"))
        result.append(row)
    return result


def plot_search_and_mc(enhanced: Mapping[str, Any] | None, output_dir: Path) -> Path:
    """Combined search stability, error distribution and hit-rate dashboard."""

    theme = get_theme()
    note = _smoke_note(enhanced)
    search = _search_rows(enhanced)
    aggregate = _mc_aggregate(enhanced)
    summary = _mc_summary(enhanced)
    samples = _mc_samples(enhanced)
    figure, axes = plt.subplots(2, 3, figsize=(16.2, 9.2))
    if not search:
        _plot_empty(axes[0, 0], "无 search_comparison 记录")
        _plot_empty(axes[0, 1], "无 search_comparison 记录")
        _plot_empty(axes[0, 2], "无 search_comparison 记录")
    else:
        strategies = [(False, "local", theme.primary_blue), (True, "global + local", theme.primary_red)]
        # Offset each strategy slightly so equal-seed local/global points remain
        # visible in the smoke experiment.
        seed_values = sorted({int(row.get("seed", 0)) for row in search})
        seed_index = {seed: index for index, seed in enumerate(seed_values)}
        for use_global, label, color in strategies:
            selected = [row for row in search if bool(row.get("use_global_search")) == use_global]
            if not selected:
                continue
            x = np.asarray([seed_index.get(int(row.get("seed", 0)), 0) + (0.12 if use_global else -0.12) for row in selected])
            y = np.asarray([_float(row.get("j_star")) for row in selected])
            finite = np.isfinite(y)
            axes[0, 0].scatter(x[finite], y[finite], s=68, color=color, marker="s" if use_global else "o", label=label)
            axes[0, 1].scatter(np.asarray([_float(row.get("point_x")) for row in selected])[finite], np.asarray([_float(row.get("point_y")) for row in selected])[finite], s=70, color=color, marker="s" if use_global else "o", label=label)
            axes[0, 2].scatter(np.asarray([_float(row.get("selected_function_evaluations")) for row in selected])[finite], y[finite], s=62, color=color, marker="s" if use_global else "o", label=label)
        axes[0, 0].set_xticks(np.arange(len(seed_values)))
        axes[0, 0].set_xticklabels([str(seed) for seed in seed_values])
        axes[0, 0].set_xlabel("random seed")
        axes[0, 0].set_ylabel("J* / m")
        axes[0, 0].set_title("局部搜索与全局+局部搜索")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set_xlabel("detector x / m")
        axes[0, 1].set_ylabel("detector y / m")
        axes[0, 1].set_title("返回点与对称分支")
        axes[0, 1].legend(fontsize=8)
        axes[0, 2].set_xlabel("function evaluations")
        axes[0, 2].set_ylabel("J* / m")
        axes[0, 2].set_title("计算预算—目标值")
        axes[0, 2].legend(fontsize=8)
        for axis in axes[0]:
            _set_grid(axis)
        _set_distance_formatter(axes[0, 1])

    if not aggregate:
        for axis in axes[1]:
            _plot_empty(axis, "无 Monte Carlo 汇总")
    else:
        branches = [str(row.get("branch", "branch")) for row in aggregate]
        x = np.arange(len(branches))
        branch_colors = [theme.primary_blue, theme.primary_red]
        rmse = np.asarray([_float(row.get("rmse_m_mean")) for row in aggregate])
        rmse_std = np.asarray([_float(row.get("rmse_m_std"), 0.0) for row in aggregate])
        p95 = np.asarray([_float(row.get("p95_error_m_mean")) for row in aggregate])
        p95_std = np.asarray([_float(row.get("p95_error_m_std"), 0.0) for row in aggregate])
        width = 0.32
        axes[1, 0].bar(x - width / 2, rmse, width, yerr=rmse_std, capsize=3, color=theme.primary_blue, alpha=0.84, label="RMSE")
        axes[1, 0].bar(x + width / 2, p95, width, yerr=p95_std, capsize=3, color=theme.primary_red, alpha=0.84, label="P95")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(branches, rotation=15)
        axes[1, 0].set_ylabel("localization error / m")
        axes[1, 0].set_title("Monte Carlo 误差（均值±跨种子标准差）")
        axes[1, 0].legend(fontsize=8)

        hit_fields = [("hit_rate_5m_mean", "5 m"), ("hit_rate_10m_mean", "10 m"), ("hit_rate_20m_mean", "20 m"), ("hit_rate_50m_mean", "50 m")]
        for index, (field, label) in enumerate(hit_fields):
            values = np.asarray([_float(row.get(field)) for row in aggregate])
            axes[1, 1].plot(x, values, "o-", color=theme.series_colors[index % len(theme.series_colors)], linewidth=1.8, label=label)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(branches, rotation=15)
        axes[1, 1].set_ylim(0.0, 1.05)
        axes[1, 1].set_ylabel("hit rate")
        axes[1, 1].set_title("定位命中率")
        axes[1, 1].legend(fontsize=8, ncol=2)

        reception = np.asarray([_float(row.get("joint_reception_rate_mean")) for row in aggregate])
        finite_intersection = np.asarray([_float(row.get("finite_intersection_rate_mean")) for row in aggregate])
        axes[1, 2].bar(x - width / 2, reception, width, color=theme.transition, label="joint reception")
        axes[1, 2].bar(x + width / 2, finite_intersection, width, color=theme.primary_blue, label="finite intersection")
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(branches, rotation=15)
        axes[1, 2].set_ylim(0.0, 1.05)
        axes[1, 2].set_ylabel("rate")
        axes[1, 2].set_title("接收与交会成功率")
        axes[1, 2].legend(fontsize=8)
        for axis in axes[1]:
            _set_grid(axis)

    # If sample-level records exist, add their empirical error distribution as
    # a translucent inset in the error panel; this retains the aggregate bars
    # while exposing heavy tails.
    if samples and aggregate:
        inset = axes[1, 0].inset_axes([0.53, 0.50, 0.44, 0.43])
        for index, branch in enumerate(sorted({row["branch"] for row in samples})):
            errors = np.asarray([row["error_m"] for row in samples if row["branch"] == branch and math.isfinite(row["error_m"])])
            if len(errors):
                sorted_errors = np.sort(errors)
                ecdf = np.arange(1, len(errors) + 1) / len(errors)
                inset.plot(sorted_errors, ecdf, color=branch_colors[index % len(branch_colors)], linewidth=1.1, label=branch)
        inset.set_xlabel("error", fontsize=7)
        inset.set_ylabel("ECDF", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.grid(True, alpha=0.18)
        inset.legend(fontsize=6)
    figure.suptitle("搜索稳定性与 Monte Carlo 定位误差验证", fontsize=14)
    return _finish(figure, output_dir / "q2_search_monte_carlo_dashboard.png", note)


def _candidate_geometry_files(input_path: Path, enhanced: Mapping[str, Any] | None) -> list[Path]:
    candidates: list[Path] = []
    for parent in (input_path.parent, Path(str(enhanced.get("_source_dir", ""))) if enhanced else None):
        if parent is None or not str(parent):
            continue
        parent = Path(parent)
        for name in ("q2_default_noglobal.json", "t2_default_noglobal.json", "t2_smoke.json", "q2_result.json"):
            candidates.append(parent / name)
        candidates.append(parent.parent / "q2_default_noglobal.json")
        candidates.append(parent.parent / "t2_default_noglobal.json")
    # Preserve order but remove duplicates.
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _read_target_vertices(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        vertices = payload.get("target_region", {}).get("vertices", [])
        array = np.asarray(vertices, dtype=float)
        if array.ndim == 2 and array.shape[1] == 2 and len(array) >= 3 and np.all(np.isfinite(array)):
            return array
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return None


def _build_model_for_overlay(payload: Mapping[str, Any], geometry_input: Path | None = None) -> tuple[Any, np.ndarray, float] | None:
    """Best-effort construction of the production model for H(P) contours."""

    try:
        # ``solve_q2`` lives one directory above this script; importing it
        # lazily keeps CSV-only use functional on minimal environments.
        code_root = Path(__file__).resolve().parents[1]
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        from t2.solve_q2 import Q2Config, Question2Model, parse_input

        source_payload: Mapping[str, Any] = payload
        if geometry_input is not None and geometry_input.exists():
            with geometry_input.open("r", encoding="utf-8") as handle:
                source_payload = json.load(handle)
        input_section = source_payload.get("input", {}) if isinstance(source_payload, Mapping) else {}
        config_section = source_payload.get("config", {}) if isinstance(source_payload, Mapping) else {}
        if not isinstance(input_section, Mapping):
            input_section = {}
        if not isinstance(config_section, Mapping):
            config_section = {}
        merged: dict[str, Any] = dict(config_section)
        merged.update(dict(input_section))
        # Some result files put s1/svd_deg in top-level ``input`` while an
        # explicit geometry input may only expose target-region metadata.
        if "s1" not in merged:
            merged["s1"] = payload.get("input", {}).get("s1", [0.0, 0.0]) if isinstance(payload.get("input"), Mapping) else [0.0, 0.0]
        if not any(key in merged for key in ("svd_deg", "bearing_deg", "theta_deg", "theta", "angle_deg")):
            merged["svd_deg"] = payload.get("input", {}).get("svd_deg", 32.0) if isinstance(payload.get("input"), Mapping) else 32.0
        s1, bearing, config = parse_input(merged)
        model = Question2Model(s1, bearing, config)
        return model, np.asarray(s1, dtype=float), float(bearing)
    except Exception as exc:  # pragma: no cover - defensive fallback path
        warnings.warn(f"could not construct Question2Model for geometry overlay: {exc}")
        return None


def _reflect_across_bearing(point: Sequence[float], s1: Sequence[float], bearing_deg: float) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    origin = np.asarray(s1, dtype=float)
    direction = np.asarray((math.cos(math.radians(float(bearing_deg))), math.sin(math.radians(float(bearing_deg)))))
    relative = p - origin
    return origin + 2.0 * float(relative @ direction) * direction - relative


def _overlay_branch_points(payload: Mapping[str, Any], enhanced: Mapping[str, Any] | None, model: Any | None, s1: np.ndarray, bearing: float) -> tuple[np.ndarray, np.ndarray]:
    sensitivity = _sensitivity(payload)
    primary = np.asarray(_point(_reference_row(sensitivity)), dtype=float)
    if primary.shape != (2,) or not np.all(np.isfinite(primary)):
        primary = np.asarray((0.0, 0.0), dtype=float)
    opposite: np.ndarray | None = None
    if enhanced:
        monte = enhanced.get("monte_carlo", {})
        points = monte.get("branch_points", {}) if isinstance(monte, Mapping) else {}
        if isinstance(points, Mapping):
            # The enhanced contour bundle is generated from the final model
            # optimum.  Prefer its primary point too, otherwise a formal
            # sensitivity row can belong to the reflected branch and the star
            # marker would be visually detached from its own level set.
            raw_primary = points.get("branch_primary")
            if isinstance(raw_primary, (list, tuple)) and len(raw_primary) >= 2:
                candidate_primary = np.asarray(raw_primary[:2], dtype=float)
                if np.all(np.isfinite(candidate_primary)):
                    primary = candidate_primary
            raw = points.get("branch_opposite")
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                candidate = np.asarray(raw[:2], dtype=float)
                if np.all(np.isfinite(candidate)):
                    opposite = candidate
    if opposite is None:
        opposite = _reflect_across_bearing(primary, s1, bearing)
    return primary, opposite


def _contour_records(enhanced: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not enhanced:
        return []
    overlay = enhanced.get("geometry_overlay", {})
    records = overlay.get("contours", []) if isinstance(overlay, Mapping) else []
    result: list[dict[str, Any]] = []
    overlay_grid = _int(overlay.get("grid_size"), None) if isinstance(overlay, Mapping) else None
    for item in records if isinstance(records, list) else []:
        if not isinstance(item, Mapping):
            continue
        points = np.asarray(item.get("points", []), dtype=float)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.all(np.isfinite(points)):
            grid_size = _int(item.get("grid_size"), None)
            # The compact JSON overlay predates the row-level CSV and stores
            # the grid only once at overlay level.  Use that value as a
            # fallback so records from a mixed/interrupted run can still be
            # compared on a common resolution.
            if grid_size is None:
                grid_size = overlay_grid
            result.append(
                {
                    "tau": _float(item.get("tau")),
                    "grid_size": grid_size,
                    "contour_id": _int(item.get("contour_id"), 0),
                    "points": points,
                }
            )

    # A convergence bundle may contain contours from several candidate-grid
    # resolutions.  Overlaying all of them produces nearly coincident lines
    # and makes the legend imply more independent levels than are present.
    # Retain every connected component at the *highest available* resolution
    # for each tau; the lower-resolution alternatives remain represented by
    # the dedicated convergence table/plot.
    by_tau: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in result:
        tau = record["tau"]
        if math.isfinite(tau):
            by_tau[float(tau)].append(record)
    filtered: list[dict[str, Any]] = []
    for tau in sorted(by_tau):
        candidates = by_tau[tau]
        finite_grids = [int(item["grid_size"]) for item in candidates if item.get("grid_size") is not None]
        if finite_grids:
            highest = max(finite_grids)
            candidates = [item for item in candidates if item.get("grid_size") is not None and int(item["grid_size"]) == highest]
        filtered.extend(sorted(candidates, key=lambda item: (int(item.get("contour_id") or 0), len(item["points"]))))
    return filtered


def plot_geometry_overlay(
    payload: Mapping[str, Any],
    enhanced: Mapping[str, Any] | None,
    output_dir: Path,
    *,
    geometry_input: Path | None = None,
    input_path: Path | None = None,
    grid_size: int = 81,
) -> Path:
    """Overlay target polygon, H(P) feasible set, branches and level contours."""

    sensitivity = _sensitivity(payload)
    theme = get_theme()
    note = _smoke_note(enhanced)
    base_tau = _reference_tau(payload, sensitivity)
    built = _build_model_for_overlay(payload, geometry_input)
    model = built[0] if built else None
    if built:
        s1, bearing = built[1], built[2]
    else:
        input_section = payload.get("input", {})
        s1 = np.asarray(input_section.get("s1", [0.0, 0.0]) if isinstance(input_section, Mapping) else [0.0, 0.0], dtype=float)
        bearing = _float(input_section.get("svd_deg"), 32.0) if isinstance(input_section, Mapping) else 32.0
    vertices = _read_target_vertices(geometry_input)
    if vertices is None and model is not None:
        vertices = np.asarray(model.region.vertices, dtype=float)
    if vertices is None:
        # Prefer the actual sensitivity input supplied by the caller.  Using
        # DEFAULT_INPUT here made custom CSV/JSON runs silently fall back to
        # the wrong geometry file when no explicit --geometry-input was set.
        geometry_search_input = input_path if input_path is not None else Path(DEFAULT_INPUT)
        for candidate in _candidate_geometry_files(geometry_search_input, enhanced):
            vertices = _read_target_vertices(candidate)
            if vertices is not None:
                break
    if vertices is None:
        # Conservative schematic fallback: clipped wedge-sector polygon.
        radius = _float(payload.get("config", {}).get("target_radius"), 1800.0) if isinstance(payload.get("config"), Mapping) else 1800.0
        angle = math.radians(bearing)
        half = math.radians(_float(payload.get("config", {}).get("angle_error_deg"), 1.0)) if isinstance(payload.get("config"), Mapping) else math.radians(1.0)
        vertices = np.vstack(
            (
                s1,
                s1 + radius * np.asarray((math.cos(angle - half), math.sin(angle - half))),
                s1 + radius * np.asarray((math.cos(angle + half), math.sin(angle + half))),
            )
        )

    config = payload.get("config", {})
    if isinstance(config, Mapping):
        min_radius = _float(config.get("min_receive_radius"), 1000.0)
        angle_error = _float(config.get("angle_error_deg"), 1.0)
    else:
        min_radius, angle_error = 1000.0, 1.0
    search_bounds = payload.get("geometry", {}).get("search_bounds") if isinstance(payload.get("geometry"), Mapping) else None
    if not (isinstance(search_bounds, (list, tuple)) and len(search_bounds) == 2):
        if model is not None:
            search_bounds = model.search_bounds
        else:
            center = np.nanmean(vertices, axis=0)
            search_bounds = [[float(center[0] - min_radius), float(center[0] + min_radius)], [float(center[1] - min_radius), float(center[1] + min_radius)]]
    x_min, x_max = map(float, search_bounds[0])
    y_min, y_max = map(float, search_bounds[1])
    n = max(31, int(grid_size))
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    h_values = np.full_like(xx, np.nan, dtype=float)
    h_tolerance = _float(payload.get("reception_constraint", {}).get("h_tolerance"), 1e-3) if isinstance(payload.get("reception_constraint"), Mapping) else 1e-3
    if model is not None:
        try:
            h_values = np.asarray(model.receive_h_many(np.column_stack((xx.ravel(), yy.ravel()))), dtype=float).reshape(xx.shape)
            h_tolerance = float(model.reception_h_tolerance())
        except Exception as exc:  # pragma: no cover - fallback for version mismatch
            warnings.warn(f"receive_h_many failed for overlay: {exc}")
    if not np.any(np.isfinite(h_values)):
        # The exact oracle is unavailable; use the conservative first-circle
        # approximation and label it in the annotation.
        h_values = (xx - s1[0]) ** 2 + (yy - s1[1]) ** 2 - min_radius**2

    primary, opposite = _overlay_branch_points(payload, enhanced, model, s1, bearing)
    figure, axis = plt.subplots(figsize=(10.8, 8.2))
    feasible = np.where(h_values <= h_tolerance, 1.0, np.nan)
    axis.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=[theme.color("blue", 0.14)], alpha=0.8)
    finite_h = np.isfinite(h_values)
    if np.any(finite_h) and float(np.nanmin(h_values[finite_h])) <= h_tolerance <= float(np.nanmax(h_values[finite_h])):
        try:
            axis.contour(xx, yy, h_values, levels=[h_tolerance], colors=[theme.primary_blue], linewidths=1.5, linestyles="--")
        except (ValueError, UserWarning):
            pass

    closed_vertices = np.vstack((vertices, vertices[0]))
    axis.fill(closed_vertices[:, 0], closed_vertices[:, 1], color=theme.color("transition", 0.18), label="目标区域 Omega_1")
    axis.plot(closed_vertices[:, 0], closed_vertices[:, 1], color=theme.transition, linewidth=1.4)
    circle = plt.Circle((s1[0], s1[1]), min_radius, fill=False, color=theme.color("foreground", 0.55), linewidth=1.0, linestyle=":", label=f"S1 接收半径 {min_radius:g} m")
    axis.add_patch(circle)

    # First bearing ray and its +/- error wedge.
    ray_length = max(math.hypot(x_max - x_min, y_max - y_min), 1.5 * min_radius)
    for offset, style, label in ((0.0, "-", "首次示向度"), (-angle_error, "--", "示向度误差边界"), (angle_error, "--", None)):
        theta = math.radians(bearing + offset)
        end = s1 + ray_length * np.asarray((math.cos(theta), math.sin(theta)))
        axis.plot([s1[0], end[0]], [s1[1], end[1]], color=theme.primary_red if offset == 0 else theme.color("red", 0.55), linewidth=1.1, linestyle=style, label=label)

    # Continuous candidate boundaries from marching squares.
    contour_colors = (theme.primary_red, theme.transition, theme.auxiliary, theme.primary_blue)
    # Proxy handles keep fallback contours visible in the legend across
    # Matplotlib releases.  ``QuadContourSet`` itself is not a supported
    # legend handle in Matplotlib 3.10+.
    fallback_contour_handles: list[Line2D] = []
    contour_records = _contour_records(enhanced)
    seen_tau: set[float] = set()
    for index, record in enumerate(contour_records):
        tau = record["tau"]
        label = f"候选水平集 tau={tau:g}" if math.isfinite(tau) and tau not in seen_tau else None
        if math.isfinite(tau):
            seen_tau.add(tau)
        points = record["points"]
        axis.plot(points[:, 0], points[:, 1], color=contour_colors[index % len(contour_colors)], linewidth=1.55, alpha=0.9, label=label)

    # If the enhanced bundle is unavailable, compute a continuous candidate
    # boundary directly from the production model on an adaptive-sized
    # regular grid.  Matplotlib's contour implementation uses the marching
    # squares algorithm internally; this fallback keeps the standalone
    # visualisation useful with only q2_sensitivity.json.
    if not contour_records and model is not None:
        try:
            grid_x = np.linspace(x_min, x_max, n)
            grid_y = np.linspace(y_min, y_max, n)
            grid_xx, grid_yy = np.meshgrid(grid_x, grid_y, indexing="xy")
            grid_points = np.column_stack((grid_xx.ravel(), grid_yy.ravel()))
            grid_h = np.asarray(model.receive_h_many(grid_points), dtype=float).reshape(grid_xx.shape)
            grid_j = np.asarray(model.objective_many(grid_points), dtype=float).reshape(grid_xx.shape)
            candidate_row = _reference_row(sensitivity)
            candidate_point = np.asarray(_point(candidate_row), dtype=float)
            base_j = _value(candidate_row, "j_star")
            if not math.isfinite(base_j):
                base_j = float(model.objective(candidate_point))
            valid = np.isfinite(grid_j) & np.isfinite(grid_h) & (grid_h <= h_tolerance)
            tau_values = _unique_sorted((0.01, base_tau, 0.10))
            for index, level_tau in enumerate(tau_values):
                level = (1.0 + float(level_tau)) * base_j
                masked = np.where(valid, grid_j, np.nan)
                finite_values = masked[np.isfinite(masked)]
                if finite_values.size == 0 or not (float(np.nanmin(finite_values)) <= level <= float(np.nanmax(finite_values))):
                    continue
                contour_set = axis.contour(
                    grid_xx,
                    grid_yy,
                    masked,
                    levels=[level],
                    colors=[contour_colors[index % len(contour_colors)]],
                    linewidths=1.55,
                    alpha=0.9,
                )
                del contour_set  # geometry is already drawn; use a proxy below
                fallback_contour_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=contour_colors[index % len(contour_colors)],
                        linewidth=1.55,
                        label=f"候选水平集 tau={float(level_tau):g}",
                    )
                )
        except Exception as exc:  # pragma: no cover - defensive numerical fallback
            warnings.warn(f"fallback marching-squares contour failed: {exc}")

    # All weight-pair optima are shown faintly to reveal branch switching; the
    # two selected branches receive larger star markers.
    all_points = np.asarray([_point(row) for row in sensitivity["rows"]], dtype=float)
    if all_points.ndim == 2 and all_points.shape[1] == 2 and np.all(np.isfinite(all_points)):
        axis.scatter(all_points[:, 0], all_points[:, 1], s=24, color=theme.color("transition", 0.5), alpha=0.55, label="各权重组最优点")
    axis.scatter([primary[0]], [primary[1]], s=190, color=theme.primary_red, marker="*", edgecolors=theme.foreground, linewidths=0.5, zorder=8, label="主最优分支")
    axis.scatter([opposite[0]], [opposite[1]], s=190, color=theme.primary_blue, marker="*", edgecolors=theme.foreground, linewidths=0.5, zorder=8, label="对称最优分支")
    axis.plot([primary[0], opposite[0]], [primary[1], opposite[1]], color=theme.color("foreground", 0.35), linestyle=":", linewidth=1.0, zorder=4)
    axis.scatter([s1[0]], [s1[1]], color=theme.foreground, marker="+", s=130, linewidths=1.8, zorder=9, label="S1")
    axis.annotate("主分支", xy=primary, xytext=(8, 8), textcoords="offset points", color=theme.primary_red, fontsize=9)
    axis.annotate("对称分支", xy=opposite, xytext=(8, -14), textcoords="offset points", color=theme.primary_blue, fontsize=9)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x / m")
    axis.set_ylabel("y / m")
    axis.set_title("目标区域、接收可行域、对称最优分支与 marching-squares 候选边界")
    handles, labels = axis.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label:
            unique.setdefault(label, handle)
    for handle in fallback_contour_handles:
        unique.setdefault(handle.get_label(), handle)
    axis.legend(list(unique.values()), list(unique.keys()), fontsize=8, loc="upper left", ncol=2)
    method_note = "H(P) 由连续接收分离预言机计算" if model is not None else "H(P) 采用接收圆保守近似（未找到模型输入）"
    annotation = f"{method_note}; 蓝色填充：H(P)≤tol；虚线：H(P)=tol\n候选边界为规则网格上的 marching-squares 近似"
    axis.text(0.99, 0.02, annotation, transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color=theme.foreground, bbox={"facecolor": theme.color("blue", 0.06), "edgecolor": "none", "pad": 3})
    return _finish(figure, output_dir / "q2_geometry_advanced_overlay.png", note)


def _draw_sweep_lines(axis: plt.Axes, rows: Sequence[Mapping[str, Any]], field: str, powers: Sequence[float], theme: Any, *, tau: float | None = None) -> None:
    for index, power in enumerate(powers):
        selected = sorted([row for row in rows if abs(_value(row, "p_w") - power) <= 1e-9], key=lambda row: _value(row, "epsilon_w"))
        x = np.asarray([_value(row, "epsilon_w") for row in selected])
        if field in {"candidate_area", "component_count"}:
            # ``tau=0`` is a valid level set; do not let truthiness replace it
            # with the default 0.05 reference level.
            level_tau = float(0.05 if tau is None else tau)
            candidate_field = "area" if field == "candidate_area" else field
            y = np.asarray([_candidate_value(row, level_tau, candidate_field) for row in selected])
        else:
            y = np.asarray([_value(row, field) for row in selected])
        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            axis.plot(x[finite], y[finite], "o-", color=theme.series_colors[index % len(theme.series_colors)], linewidth=1.5, markersize=4, label=f"p_w={power:g}")


def plot_summary_dashboard(payload: Mapping[str, Any], enhanced: Mapping[str, Any] | None, output_dir: Path) -> Path:
    """A compact overview for quick paper/meeting review."""

    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    tau = _reference_tau(payload, sensitivity)
    theme = get_theme()
    note = _smoke_note(enhanced)
    figure, axes = plt.subplots(3, 3, figsize=(16.2, 13.0))
    _draw_sweep_lines(axes[0, 0], rows, "j_under_reference_weight", powers, theme)
    axes[0, 0].set_title("统一参考目标")
    axes[0, 0].set_ylabel("J0 / m")
    axes[0, 0].set_xlabel("epsilon_w")
    axes[0, 0].legend(fontsize=7, ncol=2)
    _set_grid(axes[0, 0])
    _plot_heatmap(axes[0, 1], _matrix(rows, epsilons, powers, "j_under_reference_weight"), epsilons, powers, "J0 参数热力图", theme.colormap("red"))
    reference = _reference_row(sensitivity)
    records = _records_by_tau(reference)
    if records:
        tx = np.asarray([_float(item.get("tau")) for item in records])
        ta = np.asarray([_float(item.get("area")) for item in records])
        axes[0, 2].plot(tx, ta, "o-", color=theme.primary_blue, linewidth=1.7)
        axes[0, 2].set_xlabel("tau")
        axes[0, 2].set_ylabel("candidate area / m^2")
        axes[0, 2].set_title("tau—候选面积")
        _set_grid(axes[0, 2])
    else:
        _plot_empty(axes[0, 2], "无 tau 记录")
        axes[0, 2].set_title("tau—候选面积")

    _draw_sweep_lines(axes[1, 0], rows, "distance_from_s1", powers, theme)
    axes[1, 0].set_title("最优距离")
    axes[1, 0].set_xlabel("epsilon_w")
    axes[1, 0].set_ylabel("distance / m")
    axes[1, 0].legend(fontsize=7, ncol=2)
    _set_grid(axes[1, 0])
    _set_distance_formatter(axes[1, 0])
    _draw_sweep_lines(axes[1, 1], rows, "bearing_difference_deg", powers, theme)
    axes[1, 1].axhline(0.0, color=theme.color("foreground", 0.5), linewidth=0.9)
    axes[1, 1].set_title("方位偏差")
    axes[1, 1].set_xlabel("epsilon_w")
    axes[1, 1].set_ylabel("difference / deg")
    axes[1, 1].legend(fontsize=7, ncol=2)
    _set_grid(axes[1, 1])

    grids = _enhanced_section(enhanced, "candidate_grid")
    _plot_convergence_panel(axes[1, 2], grids, "grid size", "网格面积收敛", (("cell_center_area", "cell centre", theme.primary_blue, "o-"), ("marching_area", "marching", theme.primary_red, "s--")))

    search = _search_rows(enhanced)
    if search:
        for use_global, label, color, marker in ((False, "local", theme.primary_blue, "o"), (True, "global + local", theme.primary_red, "s")):
            selected = [row for row in search if bool(row.get("use_global_search")) == use_global]
            if selected:
                axes[2, 0].scatter([_value(row, "point_x") for row in selected], [_value(row, "point_y") for row in selected], s=56, color=color, marker=marker, label=label)
        axes[2, 0].set_title("搜索返回分支")
        axes[2, 0].set_xlabel("x / m")
        axes[2, 0].set_ylabel("y / m")
        axes[2, 0].legend(fontsize=7)
        _set_grid(axes[2, 0])
    else:
        _plot_empty(axes[2, 0], "无搜索对照记录")
        axes[2, 0].set_title("搜索返回分支")

    aggregate = _mc_aggregate(enhanced)
    if aggregate:
        branches = [str(row.get("branch", "branch")) for row in aggregate]
        x = np.arange(len(branches))
        for index, (field, label) in enumerate((("hit_rate_5m_mean", "5 m"), ("hit_rate_20m_mean", "20 m"), ("hit_rate_50m_mean", "50 m"))):
            axes[2, 1].plot(x, [_float(row.get(field)) for row in aggregate], "o-", color=theme.series_colors[index], label=label)
        axes[2, 1].set_xticks(x)
        axes[2, 1].set_xticklabels(branches, rotation=15)
        axes[2, 1].set_ylim(0, 1.05)
        axes[2, 1].set_ylabel("hit rate")
        axes[2, 1].set_title("Monte Carlo 命中率")
        axes[2, 1].legend(fontsize=7)
        _set_grid(axes[2, 1])
    else:
        _plot_empty(axes[2, 1], "无 Monte Carlo 汇总")
        axes[2, 1].set_title("Monte Carlo 命中率")

    # A compact text panel records the provenance and comparison rule; it is
    # deliberately not a KPI card, so the dashboard remains a figure rather
    # than a second report page.
    axes[2, 2].axis("off")
    metadata = payload.get("metadata", {})
    lines = [
        "数据口径",
        "• 跨权重比较使用 J0(P*)",
        "• J* 仅在自身权重内解释",
        "• strict_feasible 与容差可行性分开",
        "• 候选面积为网格/等值线近似",
        f"• 参考组：epsilon_w={_value(reference, 'epsilon_w'):g}, p_w={_value(reference, 'p_w'):g}",
    ]
    if enhanced:
        lines.append(f"• 增强实验来源：{Path(str(enhanced.get('_source_dir', ''))).name}")
    axes[2, 2].text(0.03, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10, color=theme.foreground, linespacing=1.55)
    figure.suptitle("第二问高级敏感度总览", fontsize=15)
    return _finish(figure, output_dir / "q2_advanced_summary_dashboard.png", note)


def write_sensitivity_summary(payload: Mapping[str, Any], output_dir: Path) -> Path:
    """Flatten each weight pair to one auditable summary row."""

    sensitivity = _sensitivity(payload)
    tau = _reference_tau(payload, sensitivity)
    path = output_dir / "q2_sensitivity_summary.csv"
    fieldnames = [
        "epsilon_w", "p_w", "j_star", "j_under_reference_weight", "distance_from_s1", "bearing_difference_deg",
        "reception_h", "strict_feasible", "feasible_with_tolerance", "candidate_tau", "candidate_area",
        "accepted_cell_count", "component_count", "point_x", "point_y",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sensitivity["rows"]:
            candidate = _candidate_record(row, tau) or {}
            output = {
                "epsilon_w": _value(row, "epsilon_w"),
                "p_w": _value(row, "p_w"),
                "j_star": _value(row, "j_star"),
                "j_under_reference_weight": _value(row, "j_under_reference_weight"),
                "distance_from_s1": _value(row, "distance_from_s1"),
                "bearing_difference_deg": _value(row, "bearing_difference_deg"),
                "reception_h": _value(row, "reception_h"),
                "strict_feasible": bool(row.get("strict_feasible", False)),
                "feasible_with_tolerance": bool(row.get("feasible_with_tolerance", False)),
                "candidate_tau": _float(candidate.get("tau")),
                "candidate_area": _float(candidate.get("area")),
                "accepted_cell_count": _int(candidate.get("accepted_cell_count"), None),
                "component_count": _int(candidate.get("component_count"), None),
                "point_x": _point(row)[0],
                "point_y": _point(row)[1],
            }
            writer.writerow({key: _csv_value(value) for key, value in output.items()})
    return path


def _manifest(
    payload: Mapping[str, Any],
    enhanced: Mapping[str, Any] | None,
    files: Sequence[Path],
    output_dir: Path,
    *,
    input_path: str | Path = DEFAULT_INPUT,
) -> Path:
    path = output_dir / "q2_advanced_visualization_manifest.json"
    manifest = {
        "input": str(input_path),
        "enhanced_source": None if not enhanced else enhanced.get("_source_dir"),
        "is_smoke_data": bool(enhanced and enhanced.get("_is_smoke")),
        "comparison_rule": payload.get("metadata", {}).get("comparison_rule", "j_under_reference_weight is the cross-weight comparison metric"),
        "files": [str(file) for file in files],
        "notes": [
            "Candidate areas and connected components are numerical grid/marching-squares estimates.",
            "Strict and tolerance feasibility are reported separately.",
            "Smoke enhanced data must not be described as a formal global optimum.",
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="formal sensitivity JSON or flattened CSV")
    parser.add_argument("--enhanced-dir", default=DEFAULT_ENHANCED_DIR, help="enhanced experiment directory")
    parser.add_argument("--enhanced-json", help="explicit enhanced experiment JSON (overrides --enhanced-dir JSON)")
    parser.add_argument("--geometry-input", help="optional solver result JSON containing target_region.vertices")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for PNGs, summaries and manifest")
    parser.add_argument("--geometry-grid", type=int, default=81, help="H(P) grid nodes per axis for geometry overlay")
    parser.add_argument("--no-dashboard", action="store_true", help="skip the compact overview dashboard")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_matplotlib(font_candidates=("Microsoft YaHei", "Noto Sans SC"), figure_dpi=150, savefig_dpi=320)
    payload = load_sensitivity(args.input)
    enhanced = load_enhanced(args.enhanced_dir, args.enhanced_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.geometry_grid < 31:
        raise ValueError("--geometry-grid must be at least 31")
    geometry_input = Path(args.geometry_input) if args.geometry_input else None
    saved: list[Path] = []
    saved.append(plot_parameter_sweep(payload, output_dir, enhanced))
    saved.append(plot_weight_heatmaps(payload, output_dir, enhanced))
    saved.append(plot_feasibility_heatmap(payload, output_dir, enhanced))
    saved.append(plot_tau_evolution(payload, output_dir, enhanced))
    saved.append(plot_tradeoff(payload, output_dir, enhanced))
    saved.append(plot_tornado(payload, output_dir, enhanced))
    saved.append(plot_convergence_suite(enhanced, output_dir))
    saved.append(plot_convergence_table(enhanced, output_dir))
    saved.append(plot_search_and_mc(enhanced, output_dir))
    saved.append(
        plot_geometry_overlay(
            payload,
            enhanced,
            output_dir,
            geometry_input=geometry_input,
            input_path=Path(args.input),
            grid_size=int(args.geometry_grid),
        )
    )
    if not args.no_dashboard:
        saved.append(plot_summary_dashboard(payload, enhanced, output_dir))
    saved.append(write_sensitivity_summary(payload, output_dir))
    saved.append(write_convergence_summary(enhanced, output_dir))
    saved.append(_manifest(payload, enhanced, saved, output_dir, input_path=args.input))
    for path in saved:
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
