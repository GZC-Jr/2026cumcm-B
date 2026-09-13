#!/usr/bin/env python3
"""Run reproducible synthetic simulations for Question 3.

The official simulator hides the source truth, so this module uses the
existing ``LocalBackend`` only for offline strategy validation.  It records
case-level success/time distributions and source-level localization/clear
errors.  No simulator result or official case code is inferred from these
synthetic runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
import t3  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "t3" / "simulation"


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    """Convert non-finite numeric values to JSON ``null`` recursively.

    CSV files intentionally retain ``nan`` for unavailable metrics, while the
    machine-readable JSON contract must remain strict (``allow_nan=False``).
    This matters for unsuccessful runs where no successful clear distance is
    available.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    values = [float(v) for v in values if _finite(v) is not None]
    if not values:
        return {"mean": None, "std": None, "p05": None, "median": None, "p95": None}
    q05, q50, q95 = np.quantile(np.asarray(values, dtype=float), [0.05, 0.50, 0.95])
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p05": float(q05),
        "median": float(q50),
        "p95": float(q95),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clear_records(case_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Read clear request/response pairs from the existing JSONL audit log."""
    records: dict[int, list[dict[str, Any]]] = {}
    path = case_dir / "actions.jsonl"
    if not path.exists():
        return records
    pending: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "request" and event.get("path") == "/clear":
            payload = event.get("payload", {})
            request_id = str(payload.get("request_id", ""))
            pending[request_id] = payload
        elif event.get("event") == "response" and event.get("path") == "/clear":
            request_id = str(event.get("request_id", ""))
            payload = pending.get(request_id, {})
            response = event.get("response", {})
            try:
                ch = int(payload["channel"])
                position = payload["position"]
                point = (float(position["x"]), float(position["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            records.setdefault(ch, []).append(
                {"point": point, "result": response.get("clear_result")}
            )
    return records


def _source_rows(
    backend: Any,
    robot: Any,
    case_dir: Path,
    case_number: int,
    seed: int,
) -> list[dict[str, Any]]:
    clears = _clear_records(case_dir)
    rows: list[dict[str, Any]] = []
    for channel, (truth, reception_radius) in sorted(backend.targets.items()):
        target = robot.targets.get(channel, {})
        center = target.get("center")
        mean = target.get("mean")
        center_error = t3.dist(truth, center) if center is not None else math.nan
        mean_error = t3.dist(truth, mean) if mean is not None else math.nan
        attempts = clears.get(channel, [])
        successful = [item for item in attempts if item.get("result") == "success"]
        clear_distance = (
            t3.dist(truth, successful[-1]["point"]) if successful else math.nan
        )
        rows.append(
            {
                "case": case_number,
                "seed": seed,
                "channel": channel,
                "reception_radius_m": float(reception_radius),
                "center_error_m": center_error,
                "mean_error_m": mean_error,
                "clear_distance_m": clear_distance,
                "clear_attempts": len(attempts),
                "clear_success": bool(successful),
                "station_count": len(target.get("stations", [])),
                "certified": bool(target.get("radius", math.inf) <= 20.0),
            }
        )
    return rows


def _case_row(
    result: dict[str, Any],
    truth: dict[str, Any],
    case_number: int,
    seed: int,
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = int(truth["total"])
    clear_count = int(result.get("cleared_count", 0))
    errors = [row["clear_distance_m"] for row in source_rows if _finite(row["clear_distance_m"]) is not None]
    center_errors = [row["center_error_m"] for row in source_rows if _finite(row["center_error_m"]) is not None]
    return {
        "case": case_number,
        "seed": seed,
        "true_total": total,
        "cleared_count": clear_count,
        "clear_ratio": clear_count / total if total else 0.0,
        "success": bool(result.get("status") == "complete" and clear_count == total),
        "status": result.get("status"),
        "virtual_time_s": float(result.get("virtual_time_s", math.nan)),
        "time_per_cleared_source_s": (
            float(result["virtual_time_s"]) / clear_count if clear_count else math.nan
        ),
        "distance_m": float(result.get("distance_m", math.nan)),
        "measure_count": int(result.get("measure_count", 0)),
        "clear_attempts": int(result.get("clear_attempts", 0)),
        "failed_clear_attempts": int(result.get("failed_clear_attempts", 0)),
        "program_runtime_s": float(result.get("program_runtime_s", math.nan)),
        "mean_clear_distance_m": float(statistics.fmean(errors)) if errors else math.nan,
        "p95_clear_distance_m": float(np.quantile(errors, 0.95)) if errors else math.nan,
        "mean_center_error_m": float(statistics.fmean(center_errors)) if center_errors else math.nan,
        "error_hit_rate_20m": float(sum(v <= 20.0 for v in errors) / len(errors)) if errors else math.nan,
        "error_hit_rate_50m": float(sum(v <= 50.0 for v in errors) / len(errors)) if errors else math.nan,
        "error_hit_rate_100m": float(sum(v <= 100.0 for v in errors) / len(errors)) if errors else math.nan,
        "probe_attempts": int(result.get("scheduler", {}).get("probe_attempts", 0)),
        "probe_successes": int(result.get("scheduler", {}).get("probe_successes", 0)),
        "error": result.get("error"),
    }


def _aggregate(case_rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = [bool(row["success"]) for row in case_rows]
    times = [row["virtual_time_s"] for row in case_rows]
    per_source = [row["time_per_cleared_source_s"] for row in case_rows]
    clear_ratios = [row["clear_ratio"] for row in case_rows]
    clear_errors = [row["clear_distance_m"] for row in source_rows if _finite(row["clear_distance_m"]) is not None]
    center_errors = [row["center_error_m"] for row in source_rows if _finite(row["center_error_m"]) is not None]
    return {
        "case_count": len(case_rows),
        "source_count": len(source_rows),
        "success_count": int(sum(success)),
        "success_rate": float(sum(success) / len(success)) if success else 0.0,
        "clear_ratio": _quantiles(clear_ratios),
        "virtual_time_s": _quantiles(times),
        "time_per_cleared_source_s": _quantiles(per_source),
        "clear_distance_m": _quantiles(clear_errors),
        "center_error_m": _quantiles(center_errors),
        "hit_rate_clear_distance": {
            f"le_{limit:g}m": float(sum(v <= limit for v in clear_errors) / len(clear_errors))
            if clear_errors else None
            for limit in (5, 20, 50, 100)
        },
        "failed_clear_attempts": int(sum(int(row["failed_clear_attempts"]) for row in case_rows)),
        "mean_measure_count": float(statistics.fmean(row["measure_count"] for row in case_rows)) if case_rows else None,
        "mean_program_runtime_s": float(statistics.fmean(row["program_runtime_s"] for row in case_rows)) if case_rows else None,
    }


def run_simulation(
    cases: int,
    seed_start: int,
    output_dir: Path,
    layout_name: str = t3.DEFAULT_LAYOUT,
    edge_fraction: float = 0.0,
) -> dict[str, Any]:
    if cases < 1:
        raise ValueError("cases must be positive")
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError("edge_fraction must be in [0, 1]")
    layout = t3.resolve_layout(layout_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    seeds = list(range(int(seed_start), int(seed_start) + int(cases)))
    edge_cases = int(math.ceil(cases * edge_fraction)) if edge_fraction > 0 else 0
    for case_number, seed in enumerate(seeds, start=1):
        case_dir = output_dir / "cases" / f"case_{case_number:03d}"
        # Deterministically assign the requested fraction to the first cases;
        # unlike modulo rounding this still exercises an edge case for small
        # smoke runs and never silently rounds a nonzero fraction to zero.
        backend = t3.LocalBackend(seed, edge=(case_number <= edge_cases))
        truth = t3.scenario_record(backend)
        robot = t3.Robot(backend, "LOCAL-SIM", case_dir, quiet=True, layout=layout)
        result = robot.run()
        truth_rows = _source_rows(backend, robot, case_dir, case_number, seed)
        case_rows.append(_case_row(result, truth, case_number, seed, truth_rows))
        source_rows.extend(truth_rows)
        (case_dir / "scenario.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(case_rows[-1], ensure_ascii=False), flush=True)
    aggregate = _aggregate(case_rows, source_rows)
    _write_csv(output_dir / "case_metrics.csv", case_rows)
    _write_csv(output_dir / "source_metrics.csv", source_rows)
    payload = {
        "schema_version": 1,
        "experiment": "q3_localbackend_monte_carlo",
        "official_simulator_test": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "layout": layout,
        "seed_start": int(seed_start),
        "seeds": seeds,
        "edge_fraction": float(edge_fraction),
        "cases": case_rows,
        "aggregate": aggregate,
        "source_metrics_csv": "source_metrics.csv",
        "case_metrics_csv": "case_metrics.csv",
        "limitations": [
            "LocalBackend is a synthetic offline evaluator and not the official simulator.",
            "The source truth is used only after each local run to score errors.",
            "Program runtime is local Python wall time; virtual_time_s is the task cost metric.",
        ],
    }
    (output_dir / "simulation_results.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=2026091300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layout", choices=list(t3.LAYOUT_PRESETS), default=t3.DEFAULT_LAYOUT)
    parser.add_argument("--edge-fraction", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_simulation(args.cases, args.seed_start, args.output_dir, args.layout, args.edge_fraction)
