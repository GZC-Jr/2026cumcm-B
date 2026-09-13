"""Render publication figures for a Question 2 sensitivity sweep.

The script consumes the JSON written by ``run_sensitivity.py``.  A flattened
CSV from the same run is also accepted when a compact tabular input is more
convenient.  All figures use the shared ``mathmodel_viz`` typography and
colour theme and are written below ``outputs/t2/figures`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np


VISUALIZATION_SRC = Path(__file__).resolve().parents[2] / "tools" / "visualization" / "src"
if str(VISUALIZATION_SRC) not in sys.path:
    sys.path.insert(0, str(VISUALIZATION_SRC))

from mathmodel_viz.styles import configure_matplotlib, get_theme, save_figure  # noqa: E402


def _float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    return float(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _unique_sorted(values: Iterable[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _load_csv(path: Path) -> dict[str, Any]:
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    tau_values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            epsilon = float(record["epsilon_w"])
            power = float(record["p_w"])
            key = (epsilon, power)
            row = grouped.setdefault(
                key,
                {
                    "epsilon_w": epsilon,
                    "p_w": power,
                    "point": [float(record["point_x"]), float(record["point_y"])],
                    "j_star": float(record["j_star"]),
                    "j_under_reference_weight": float(record["j_under_reference_weight"]),
                    "distance_from_s1": float(record["distance_from_s1"]),
                    "bearing_from_s1_deg": float(record["bearing_from_s1_deg"]),
                    "bearing_difference_deg": float(record["bearing_difference_deg"]),
                    "reception_h": float(record["reception_h"]),
                    "reception_h_tolerance": float(record["reception_h_tolerance"]),
                    "strict_feasible": _bool(record["strict_feasible"]),
                    "feasible_with_tolerance": _bool(record["feasible_with_tolerance"]),
                    "optimization_source": record["optimization_source"],
                    "use_global_search": _bool(record["use_global_search"]),
                    "selected_iterations": _float(record["selected_iterations"]),
                    "selected_function_evaluations": _float(record["selected_function_evaluations"]),
                    "local_run_count": int(float(record["local_run_count"])),
                    "local_success_count": int(float(record["local_success_count"])),
                    "local_feasible_count": int(float(record["local_feasible_count"])),
                    "candidate_by_tau": [],
                },
            )
            tau = float(record["tau"])
            tau_values.append(tau)
            row["candidate_by_tau"].append(
                {
                    "tau": tau,
                    "threshold": float(record["threshold"]),
                    "area": float(record["candidate_area"]),
                    "accepted_cell_count": int(float(record["accepted_cell_count"])),
                    "grid_shape": [int(float(record["grid_size"])), int(float(record["grid_size"]))],
                    "x_step": float(record["x_step"]),
                    "y_step": float(record["y_step"]),
                    "distance_from_s1_min": _float(record["distance_from_s1_min"], np.nan),
                    "distance_from_s1_max": _float(record["distance_from_s1_max"], np.nan),
                    "component_count": int(float(record["component_count"])),
                    "components": [],
                }
            )
    rows = list(grouped.values())
    epsilon_values = _unique_sorted(row["epsilon_w"] for row in rows)
    p_values = _unique_sorted(row["p_w"] for row in rows)
    return {
        "model": "question_2_continuous_robust_dop",
        "sensitivity": {
            "reference_weight": {
                "epsilon_w": 0.2,
                "p_w": 2.0,
            },
            "epsilon_values": epsilon_values,
            "p_values": p_values,
            "tau_values": _unique_sorted(tau_values),
            "rows": rows,
        },
    }


def load_payload(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        return _load_csv(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "sensitivity" not in payload or "rows" not in payload["sensitivity"]:
        raise ValueError("input JSON must contain sensitivity.rows")
    return payload


def _sensitivity(payload: dict[str, Any]) -> dict[str, Any]:
    sensitivity = payload["sensitivity"]
    if not sensitivity.get("rows"):
        raise ValueError("sensitivity.rows is empty")
    return sensitivity


def _axes_values(sensitivity: dict[str, Any]) -> tuple[list[float], list[float]]:
    rows = sensitivity["rows"]
    epsilons = _unique_sorted(sensitivity.get("epsilon_values", [row["epsilon_w"] for row in rows]))
    powers = _unique_sorted(sensitivity.get("p_values", [row["p_w"] for row in rows]))
    return epsilons, powers


def _matrix(rows: Sequence[dict[str, Any]], epsilons: Sequence[float], powers: Sequence[float], field: str) -> np.ndarray:
    result = np.full((len(epsilons), len(powers)), np.nan, dtype=float)
    for row in rows:
        i = int(np.argmin(np.abs(np.asarray(epsilons) - float(row["epsilon_w"]))))
        j = int(np.argmin(np.abs(np.asarray(powers) - float(row["p_w"]))))
        if field == "point_x":
            value = row.get("point_x", (row.get("point") or [None, None])[0])
        elif field == "point_y":
            value = row.get("point_y", (row.get("point") or [None, None])[1])
        else:
            value = row.get(field)
        if value is not None:
            result[i, j] = float(value)
    return result


def _format_ticks(axis: plt.Axes, epsilons: Sequence[float], powers: Sequence[float]) -> None:
    axis.set_xticks(np.arange(len(powers)))
    axis.set_xticklabels([f"{value:g}" for value in powers])
    axis.set_yticks(np.arange(len(epsilons)))
    axis.set_yticklabels([f"{value:g}" for value in epsilons])
    # Keep CJK labels out of mathtext: its fallback font does not provide
    # Chinese glyphs on all platforms.  Plain parameter names retain the
    # same meaning without triggering missing-glyph warnings.
    axis.set_xlabel("深度权重幂次 p_w")
    axis.set_ylabel("边界基础权重 epsilon_w")


def _annotate_matrix(axis: plt.Axes, matrix: np.ndarray, fmt: str = ".2f") -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                text_color = get_theme().foreground
                axis.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", color=text_color, fontsize=9)


def _plot_heatmap(
    axis: plt.Axes,
    matrix: np.ndarray,
    epsilons: Sequence[float],
    powers: Sequence[float],
    title: str,
    color_map: Any,
    value_label: str,
    fmt: str,
) -> None:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError(f"no finite values available for {value_label}")
    image = axis.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest", cmap=color_map)
    _format_ticks(axis, epsilons, powers)
    axis.set_title(title)
    _annotate_matrix(axis, matrix, fmt)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(value_label)


def plot_position(payload: dict[str, Any], output_dir: Path) -> Path:
    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    theme = get_theme()
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    _plot_heatmap(
        axes[0],
        _matrix(rows, epsilons, powers, "point_x"),
        epsilons,
        powers,
        "数值最优点横坐标",
        theme.colormap("blue"),
        "x* / m",
        ".1f",
    )
    _plot_heatmap(
        axes[1],
        _matrix(rows, epsilons, powers, "point_y"),
        epsilons,
        powers,
        "数值最优点纵坐标",
        theme.colormap("accent"),
        "y* / m",
        ".1f",
    )
    figure.suptitle("权重参数对第二检测点数值最优位置的影响")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "q2_weight_optimal_position.png"
    saved = save_figure(figure, path)
    plt.close(figure)
    return saved


def plot_objective_heatmaps(payload: dict[str, Any], output_dir: Path) -> Path:
    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    theme = get_theme()
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    _plot_heatmap(
        axes[0],
        _matrix(rows, epsilons, powers, "j_star"),
        epsilons,
        powers,
        "各组自身权重下的 J*",
        theme.colormap("blue"),
        "J* / m",
        ".2f",
    )
    _plot_heatmap(
        axes[1],
        _matrix(rows, epsilons, powers, "j_under_reference_weight"),
        epsilons,
        powers,
        "统一参考权重下重算的 J0(P*)",
        theme.colormap("red"),
        "J0(P*) / m",
        ".2f",
    )
    figure.suptitle("权重参数敏感性：目标值与统一标准重算值")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "q2_weight_reference_objective.png"
    saved = save_figure(figure, path)
    plt.close(figure)
    return saved


def _reference_row(sensitivity: dict[str, Any]) -> dict[str, Any]:
    reference = sensitivity.get("reference_weight", {"epsilon_w": 0.2, "p_w": 2.0})
    epsilon = float(reference.get("epsilon_w", 0.2))
    power = float(reference.get("p_w", 2.0))
    rows = sensitivity["rows"]
    return min(
        rows,
        key=lambda row: abs(float(row["epsilon_w"]) - epsilon) + abs(float(row["p_w"]) - power),
    )


def _tau_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(row.get("candidate_by_tau", []), key=lambda item: float(item["tau"]))


def plot_tau_effect(payload: dict[str, Any], output_dir: Path) -> Path:
    sensitivity = _sensitivity(payload)
    row = _reference_row(sensitivity)
    records = _tau_records(row)
    if not records:
        raise ValueError("reference row contains no candidate_by_tau records")
    tau = np.asarray([float(item["tau"]) for item in records])
    area = np.asarray([float(item["area"]) for item in records])
    distance_min = np.asarray([_float(item.get("distance_from_s1_min"), np.nan) for item in records], dtype=float)
    distance_max = np.asarray([_float(item.get("distance_from_s1_max"), np.nan) for item in records], dtype=float)
    components = np.asarray([float(item["component_count"]) for item in records])
    theme = get_theme()
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7))
    axes[0].plot(tau, area, "o-", color=theme.primary_blue, linewidth=1.8, markersize=5, label="候选面积")
    axes[0].set_xlabel("近优容差 tau")
    axes[0].set_ylabel("网格近似面积 / m^2")
    axes[0].set_title("候选区域面积")
    axes[0].legend()
    valid_distance = np.isfinite(distance_min) & np.isfinite(distance_max)
    if np.any(valid_distance):
        # Empty grid estimates are represented by None in JSON and blank CSV
        # cells.  Plot only paired finite ranges so missing taus remain gaps.
        axes[1].fill_between(
            tau[valid_distance],
            distance_min[valid_distance],
            distance_max[valid_distance],
            color=theme.color("transition", 0.22),
            label=r"$L_{\min}$--$L_{\max}$",
        )
        axes[1].plot(
            tau[valid_distance],
            distance_min[valid_distance],
            "o-",
            color=theme.primary_blue,
            linewidth=1.5,
            markersize=4,
            label=r"$L_{\min}$",
        )
        axes[1].plot(
            tau[valid_distance],
            distance_max[valid_distance],
            "s-",
            color=theme.primary_red,
            linewidth=1.5,
            markersize=4,
            label=r"$L_{\max}$",
        )
    else:
        axes[1].text(
            0.5,
            0.5,
            "无有效候选网格单元",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    axes[1].set_xlabel("近优容差 tau")
    axes[1].set_ylabel("距 S1 的距离 / m")
    axes[1].set_title("候选区域距离范围")
    if np.any(valid_distance):
        axes[1].legend()
    axes[2].step(tau, components, where="mid", color=theme.primary_red, linewidth=1.8, marker="o", label="连通分量数")
    axes[2].set_xlabel("近优容差 tau")
    axes[2].set_ylabel("连通分量数")
    axes[2].set_title("候选区域拓扑统计")
    axes[2].set_yticks(np.arange(0, max(1, int(np.nanmax(components))) + 1))
    axes[2].legend()
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(
        "近优容差对候选区域的影响 "
        f"(参考行: epsilon_w={float(row['epsilon_w']):g}, "
        f"p_w={float(row['p_w']):g}; 网格近似)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path = output_dir / "q2_tau_candidate_region.png"
    saved = save_figure(figure, path)
    plt.close(figure)
    return saved


def plot_location_metrics(payload: dict[str, Any], output_dir: Path) -> Path:
    sensitivity = _sensitivity(payload)
    rows = sensitivity["rows"]
    epsilons, powers = _axes_values(sensitivity)
    theme = get_theme()
    figure, axes = plt.subplots(1, 2, figsize=(12.1, 4.8))
    colors = theme.series_colors
    for index, power in enumerate(powers):
        selected = sorted(
            [row for row in rows if abs(float(row["p_w"]) - power) <= 1e-9],
            key=lambda row: float(row["epsilon_w"]),
        )
        if not selected:
            continue
        x_values = [float(row["epsilon_w"]) for row in selected]
        color = colors[index % len(colors)]
        axes[0].plot(
            x_values,
            [float(row["distance_from_s1"]) for row in selected],
            "o-",
            color=color,
            linewidth=1.7,
            markersize=5,
            label=fr"$p_w={power:g}$",
        )
        axes[1].plot(
            x_values,
            [float(row["bearing_difference_deg"]) for row in selected],
            "o-",
            color=color,
            linewidth=1.7,
            markersize=5,
            label=fr"$p_w={power:g}$",
        )
    axes[0].set_xlabel("边界基础权重 epsilon_w")
    axes[0].set_ylabel("距 S1 的最优距离 / m")
    axes[0].set_title("最优距离变化")
    axes[1].axhline(0.0, color=theme.color("foreground", 0.65), linewidth=0.9)
    axes[1].set_xlabel("边界基础权重 epsilon_w")
    axes[1].set_ylabel("相对首次示向度偏差 / deg")
    axes[1].set_title("最优方位偏差变化")
    for axis in axes:
        axis.set_xticks(epsilons)
        axis.legend()
        axis.grid(True, axis="y", alpha=0.25)
    # Distances are close to 1000 m in this experiment.  An automatic
    # offset formatter would display ticks as ``1e-7 + 9.999e2``, which
    # obscures the actual scale in a paper figure.  Format the absolute
    # metre values explicitly and keep the full baseline visible.
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    figure.suptitle("权重参数对最优点几何位置的影响")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "q2_weight_location_metrics.png"
    saved = save_figure(figure, path)
    plt.close(figure)
    return saved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/t2/q2_sensitivity.json", help="sensitivity JSON or CSV")
    parser.add_argument("--output-dir", default="outputs/t2/figures", help="figure output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_matplotlib(
        font_candidates=("Microsoft YaHei", "Noto Sans SC"),
        figure_dpi=150,
        savefig_dpi=300,
    )
    payload = load_payload(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        plot_position(payload, output_dir),
        plot_objective_heatmaps(payload, output_dir),
        plot_tau_effect(payload, output_dir),
        plot_location_metrics(payload, output_dir),
    ]
    for path in saved:
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
