#!/usr/bin/env python3
"""Common-random-number sensitivity analysis for Question 3.

Each perturbation reuses the seeds from ``simulate_t3.py``.  This makes the
differences paired by scenario, reducing Monte-Carlo noise when comparing the
default strategy with changes in layout, measurement weighting, probe budget,
and route detour allowance.  The script also renders paper-ready figures
using the shared visualization style package.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
import t3  # noqa: E402
from simulate_t3 import _aggregate, _case_row, _source_rows  # noqa: E402

VIS_SRC = PROJECT_ROOT / "tools" / "visualization" / "src"
if str(VIS_SRC) not in sys.path:
    sys.path.insert(0, str(VIS_SRC))
from mathmodel_viz.styles import configure_matplotlib, get_theme, save_figure  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "t3" / "simulation" / "simulation_results.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "t3" / "sensitivity"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flat(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else math.nan
    except (TypeError, ValueError):
        return math.nan


@contextmanager
def _parameters(values: dict[str, Any]) -> Iterator[None]:
    """Temporarily mutate the existing strategy configuration for one sweep."""
    old_schedule = dict(t3.SCHEDULE_CONFIG)
    old_bayes = dict(t3.BAYES_CONFIG)
    old_probe = dict(t3.PROBE_CONFIG)
    old_joint = dict(t3.JOINT_CONFIG)
    try:
        for key in ("search_gain", "lookahead", "cell_size"):
            if key in values:
                t3.SCHEDULE_CONFIG[key] = values[key]
        for key in ("noise_deg", "prior_exists", "quad_depth"):
            if key in values:
                t3.BAYES_CONFIG[key] = values[key]
        for key in ("max_attempts", "min_probability", "max_radius"):
            if key in values:
                t3.PROBE_CONFIG[key] = values[key]
        for key in ("max_detour_m", "horizon", "min_saving_s"):
            if key in values:
                t3.JOINT_CONFIG[key] = values[key]
        yield
    finally:
        t3.SCHEDULE_CONFIG.clear(); t3.SCHEDULE_CONFIG.update(old_schedule)
        t3.BAYES_CONFIG.clear(); t3.BAYES_CONFIG.update(old_bayes)
        t3.PROBE_CONFIG.clear(); t3.PROBE_CONFIG.update(old_probe)
        t3.JOINT_CONFIG.clear(); t3.JOINT_CONFIG.update(old_joint)


def _run_setting(
    setting: dict[str, Any], seeds: list[int], output_dir: Path, layout_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    layout = t3.resolve_layout(layout_name, setting.get("ring_points"), setting.get("ring_radius"))
    safe_name = str(setting["setting_id"]).replace("/", "_")
    # A rerun with fewer paired seeds must not leave stale case directories
    # from an earlier, larger sweep.  This directory is script-owned output.
    setting_root = output_dir / "runs" / safe_name
    if setting_root.exists():
        shutil.rmtree(setting_root)
    with _parameters(setting.get("parameters", {})):
        for index, seed in enumerate(seeds, start=1):
            case_dir = output_dir / "runs" / safe_name / f"case_{index:03d}"
            backend = t3.LocalBackend(seed)
            truth = t3.scenario_record(backend)
            robot = t3.Robot(backend, "LOCAL-SENS", case_dir, quiet=True, layout=layout)
            result = robot.run()
            src = _source_rows(backend, robot, case_dir, index, seed)
            case_row = _case_row(result, truth, index, seed, src)
            # Keep the paired case table self-describing when all settings are
            # concatenated into sensitivity_case_metrics.csv.
            case_row.update({
                "setting_id": setting["setting_id"],
                "parameter": setting["parameter"],
                "setting_value": setting["value"],
            })
            case_rows.append(case_row)
            source_rows.extend(src)
    aggregate = _aggregate(case_rows, source_rows)
    row = {
        "setting_id": setting["setting_id"],
        "parameter": setting["parameter"],
        "value": setting["value"],
        "layout_name": layout["layout_name"],
        "case_count": aggregate["case_count"],
        "success_rate": aggregate["success_rate"],
        "mean_clear_ratio": aggregate["clear_ratio"]["mean"],
        "mean_virtual_time_s": aggregate["virtual_time_s"]["mean"],
        "p95_virtual_time_s": aggregate["virtual_time_s"]["p95"],
        "mean_time_per_source_s": aggregate["time_per_cleared_source_s"]["mean"],
        "p95_clear_distance_m": aggregate["clear_distance_m"]["p95"],
        "error_hit_rate_20m": aggregate["hit_rate_clear_distance"]["le_20m"],
        "failed_clear_attempts": aggregate["failed_clear_attempts"],
    }
    return row, case_rows


def _settings(layout_name: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for parameter, candidates in (
        ("search_gain", (0.05, 0.2, 0.5, 1.0)),
        ("lookahead", (0.0, 0.5, 1.0)),
        ("max_probes", (0, 2, 5, 8)),
        ("max_detour_m", (0.0, 2.5, 5.0, 10.0)),
        ("layout_ring_points", (6, 7, 8, 9, 10)),
    ):
        for value in candidates:
            setting_id = f"{parameter}={value:g}"
            if parameter == "max_probes":
                parameters = {"max_attempts": int(value)}
            elif parameter == "max_detour_m":
                parameters = {"max_detour_m": float(value)}
            elif parameter == "layout_ring_points":
                parameters = {}
            else:
                parameters = {parameter: float(value)}
            values.append({
                "setting_id": setting_id,
                "parameter": parameter,
                "value": value,
                "parameters": parameters,
                "ring_points": int(value) if parameter == "layout_ring_points" else None,
                "ring_radius": {6: 1123.0, 7: 998.0, 8: 939.0, 9: 904.0, 10: 881.0}.get(int(value))
                if parameter == "layout_ring_points" else None,
            })
    return values


def _baseline_from_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    aggregate = payload.get("aggregate", {})
    return {
        "setting_id": "baseline_simulation",
        "parameter": "baseline",
        "value": "default",
        "layout_name": payload.get("layout", {}).get("layout_name", t3.DEFAULT_LAYOUT),
        "case_count": aggregate.get("case_count"),
        "success_rate": aggregate.get("success_rate"),
        "mean_clear_ratio": aggregate.get("clear_ratio", {}).get("mean"),
        "mean_virtual_time_s": aggregate.get("virtual_time_s", {}).get("mean"),
        "p95_virtual_time_s": aggregate.get("virtual_time_s", {}).get("p95"),
        "mean_time_per_source_s": aggregate.get("time_per_cleared_source_s", {}).get("mean"),
        "p95_clear_distance_m": aggregate.get("clear_distance_m", {}).get("p95"),
        "error_hit_rate_20m": aggregate.get("hit_rate_clear_distance", {}).get("le_20m"),
        "failed_clear_attempts": aggregate.get("failed_clear_attempts"),
    }


def _plot(rows: list[dict[str, Any]], baseline: dict[str, Any], output_dir: Path) -> None:
    theme = get_theme()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("success_rate", "Success rate", "rate", (0.0, 1.05)),
        ("mean_virtual_time_s", "Mean virtual time", "s", None),
        ("p95_clear_distance_m", "P95 clear distance", "m", None),
    ]
    parameters = ["search_gain", "lookahead", "max_probes", "max_detour_m", "layout_ring_points"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    for axis, parameter in zip(axes.flat, parameters + [None]):
        if parameter is None:
            axis.axis("off")
            axis.text(0.02, 0.88, "Paired sensitivity design", transform=axis.transAxes, fontsize=13)
            axis.text(0.02, 0.70, "Every setting reuses the same seeds.\nLower time/error is better; higher success is better.",
                      transform=axis.transAxes, va="top", fontsize=10)
            continue
        subset = [r for r in rows if r["parameter"] == parameter]
        x = np.asarray([float(r["value"]) for r in subset])
        order = np.argsort(x)
        x = x[order]
        success = np.asarray([_flat(subset[i]["success_rate"]) for i in order])
        times = np.asarray([_flat(subset[i]["mean_virtual_time_s"]) for i in order])
        distance = np.asarray([_flat(subset[i]["p95_clear_distance_m"]) for i in order])
        axis.plot(x, success, "o-", color=theme.primary_blue, label="success rate")
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel(parameter)
        axis.set_ylabel("success rate")
        axis.grid(alpha=0.25)
        twin = axis.twinx()
        twin.plot(x, times, "s--", color=theme.primary_red, label="mean virtual time")
        twin.set_ylabel("mean virtual time / s", color=theme.primary_red)
        twin.tick_params(axis="y", labelcolor=theme.primary_red)
        axis.set_title(parameter)
        # Mark the strategy default where it is meaningful.  The aggregate
        # baseline row stores the literal value ``default`` rather than a
        # numeric x-coordinate, so attempting to cast it caused plotting to
        # fail before any sensitivity artifact was written.
        defaults = {
            "search_gain": float(t3.SCHEDULE_CONFIG["search_gain"]),
            "lookahead": float(t3.SCHEDULE_CONFIG["lookahead"]),
            "max_probes": float(t3.PROBE_CONFIG["max_attempts"]),
            "max_detour_m": float(t3.JOINT_CONFIG["max_detour_m"]),
            "layout_ring_points": float(t3.resolve_layout(t3.DEFAULT_LAYOUT)["ring_points"]),
        }
        default_x = defaults.get(parameter)
        if default_x is not None and np.isfinite(default_x):
            axis.axvline(default_x, color=theme.foreground, linestyle=":", linewidth=1.0, alpha=0.55)
        if parameter == "layout_ring_points":
            axis.set_xticks(x)
    fig.suptitle("Question 3 paired sensitivity analysis", fontsize=16)
    save_figure(fig, output_dir / "q3_sensitivity_analysis.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    case_rows = payload_cases = []
    # The distribution chart is based on the baseline simulation file and is
    # written by ``main``; this function only handles the sweep figure.
    for axis, (field, title, unit, ylim) in zip(axes, metrics):
        values = [_flat(row.get(field)) for row in rows if row.get("parameter") != "baseline"]
        values = [v for v in values if math.isfinite(v)]
        axis.hist(values, bins=min(12, max(4, len(values) // 3)), color=theme.transition, alpha=0.8)
        axis.set_title(f"Sensitivity distribution: {title}")
        axis.set_xlabel(unit)
        axis.set_ylabel("setting count")
        if ylim is not None:
            axis.set_xlim(*ylim)
    save_figure(fig, output_dir / "q3_sensitivity_metric_distributions.png")
    plt.close(fig)

    # Error sensitivity is kept in a dedicated small-multiple figure so the
    # localization metric remains readable instead of competing with the
    # success/time twin axes above.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    for axis, parameter in zip(axes.flat, parameters + [None]):
        if parameter is None:
            axis.axis("off")
            continue
        subset = [r for r in rows if r["parameter"] == parameter]
        x = np.asarray([float(r["value"]) for r in subset])
        order = np.argsort(x)
        x = x[order]
        p95 = np.asarray([_flat(subset[i]["p95_clear_distance_m"]) for i in order])
        hit = np.asarray([_flat(subset[i]["error_hit_rate_20m"]) for i in order])
        axis.plot(x, p95, "o-", color=theme.primary_red, label="P95 clear distance")
        axis.set_xlabel(parameter)
        axis.set_ylabel("P95 distance / m")
        axis.grid(alpha=0.25)
        twin = axis.twinx()
        twin.plot(x, hit, "s--", color=theme.primary_blue, label="≤20 m hit rate")
        twin.set_ylabel("≤20 m hit rate", color=theme.primary_blue)
        twin.set_ylim(0.0, 1.05)
        twin.tick_params(axis="y", labelcolor=theme.primary_blue)
        axis.set_title(parameter)
        defaults = {
            "search_gain": float(t3.SCHEDULE_CONFIG["search_gain"]),
            "lookahead": float(t3.SCHEDULE_CONFIG["lookahead"]),
            "max_probes": float(t3.PROBE_CONFIG["max_attempts"]),
            "max_detour_m": float(t3.JOINT_CONFIG["max_detour_m"]),
            "layout_ring_points": float(t3.resolve_layout(t3.DEFAULT_LAYOUT)["ring_points"]),
        }
        default_x = defaults.get(parameter)
        if default_x is not None and np.isfinite(default_x):
            axis.axvline(default_x, color=theme.foreground, linestyle=":", linewidth=1.0, alpha=0.55)
    fig.suptitle("Question 3 localization-error sensitivity")
    save_figure(fig, output_dir / "q3_sensitivity_error_distance.png")
    plt.close(fig)


def _plot_baseline_distribution(payload: dict[str, Any], output_dir: Path, input_path: Path | None = None) -> None:
    theme = get_theme()
    cases = payload.get("cases", [])
    source_path = Path(payload.get("source_metrics_csv", ""))
    if not source_path.is_absolute():
        source_path = (input_path.parent if input_path is not None else DEFAULT_INPUT.parent) / source_path
    source_rows: list[dict[str, Any]] = []
    if source_path.exists():
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
    times = [_flat(row.get("virtual_time_s")) for row in cases]
    ratios = [_flat(row.get("clear_ratio")) for row in cases]
    errors = sorted(_flat(row.get("clear_distance_m")) for row in source_rows)
    errors = [value for value in errors if math.isfinite(value)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].hist([v for v in times if math.isfinite(v)], bins=12, color=theme.primary_blue, alpha=0.85)
    axes[0, 0].set(title="Virtual time distribution", xlabel="virtual time / s", ylabel="cases")
    axes[0, 1].hist([v for v in ratios if math.isfinite(v)], bins=np.linspace(0, 1, 11), color=theme.transition, alpha=0.85)
    axes[0, 1].set(title="Clear ratio distribution", xlabel="clear ratio", ylabel="cases")
    if errors:
        y = np.arange(1, len(errors) + 1) / len(errors)
        axes[1, 0].plot(errors, y, color=theme.primary_red, linewidth=2)
    axes[1, 0].set(title="Empirical clear-distance CDF", xlabel="clear distance / m", ylabel="CDF")
    axes[1, 0].set_ylim(0, 1.02)
    axes[1, 1].scatter(
        [_flat(row.get("true_total")) for row in cases],
        times,
        c=[theme.primary_blue if row.get("success") else theme.primary_red for row in cases],
        alpha=0.85,
    )
    axes[1, 1].set(title="Time by source count", xlabel="true source count", ylabel="virtual time / s")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Question 3 simulation distributions", fontsize=16)
    save_figure(fig, output_dir / "q3_simulation_distributions.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=8, help="paired cases per sensitivity setting")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="use one perturbation per parameter")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    seeds = payload.get("seeds", [])
    if args.seed_start is not None:
        seeds = list(range(args.seed_start, args.seed_start + args.cases))
    else:
        seeds = [int(seed) for seed in seeds[: args.cases]]
    if not seeds:
        raise ValueError("input simulation_results.json has no seeds")
    configure_matplotlib(font_candidates=("Microsoft YaHei", "Noto Sans SC"), figure_dpi=120, savefig_dpi=300)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _baseline_from_simulation(payload)
    layout_name = payload.get("layout", {}).get("layout_name", t3.DEFAULT_LAYOUT)
    settings = _settings(layout_name)
    if args.quick:
        settings = [item for item in settings if item["value"] in {0.2, 1.0, 5, 8}]
    rows: list[dict[str, Any]] = []
    all_case_rows: list[dict[str, Any]] = []
    for setting in settings:
        row, case_rows = _run_setting(setting, seeds, args.output_dir, layout_name)
        rows.append(row)
        all_case_rows.extend(case_rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    _write_csv(args.output_dir / "sensitivity_summary.csv", rows)
    _write_csv(args.output_dir / "sensitivity_case_metrics.csv", all_case_rows)
    _plot(rows, baseline, args.output_dir / "figures")
    _plot_baseline_distribution(payload, args.output_dir / "figures", args.input)
    result = {
        "schema_version": 1,
        "experiment": "q3_common_random_number_sensitivity",
        "official_simulator_test": False,
        "input_simulation": str(args.input.resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed_count": len(seeds),
        "seeds": seeds,
        "baseline": baseline,
        "settings": rows,
        "figures": [
            "figures/q3_sensitivity_analysis.png",
            "figures/q3_sensitivity_metric_distributions.png",
            "figures/q3_sensitivity_error_distance.png",
            "figures/q3_simulation_distributions.png",
        ],
        "limitations": [
            "Sensitivity is paired on LocalBackend synthetic cases, not official simulator cases.",
            "The baseline distribution comes from the input simulation results; perturbed settings use the same seed list.",
        ],
    }
    (args.output_dir / "sensitivity_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
