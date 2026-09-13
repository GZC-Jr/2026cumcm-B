"""Run the Question 2 weight/tolerance sensitivity sweep.

The continuous model remains implemented in :mod:`t2.solve_q2`.  This file
only handles experiment configuration and reproducible JSON/CSV export, so a
paper run cannot accidentally drift from the solver used for the main result.

Examples from the repository root::

    python code/t2/run_sensitivity.py --demo --no-global \
        --output outputs/t2/q2_sensitivity.json
    python code/t2/run_sensitivity.py --input data/question2.json \
        --global-search --output outputs/t2/q2_sensitivity_global.json
    python code/t2/run_sensitivity.py --demo --smoke --no-global
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


# ``code`` is a workspace directory rather than a Python package.  Import the
# already-tested continuous solver from its parent directory.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from t2.solve_q2 import Q2Config, Question2Model, load_input, parse_input  # noqa: E402


def _demo_input() -> dict[str, Any]:
    return {
        "s1": {"x": 0.0, "y": 0.0},
        "svd_deg": 32.0,
        "angle_error_deg": 1.0,
        "epsilon_w": 0.2,
        "p_w": 2.0,
        "tau": 0.05,
    }


def _parse_values(raw: str | None, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values: list[float] = []
    for index, token in enumerate(raw.split(",")):
        token = token.strip()
        if not token:
            raise ValueError(f"{name} contains an empty value at position {index}")
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-numeric value: {token!r}") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(values)


def _serializable_config(config: Q2Config) -> dict[str, Any]:
    return _serializable_mapping(asdict(config))


def _serializable_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in values.items()
    }


def _smoke_config(config: Q2Config) -> dict[str, Any]:
    """Apply explicit low-cost settings used only by ``--smoke`` validation."""

    overrides = {
        "circle_sides": min(config.circle_sides, 32),
        "quadrature_order": min(config.quadrature_order, 3),
        "validation_quadrature_order": min(config.validation_quadrature_order, 4),
        "candidate_grid": min(config.candidate_grid, 11),
        "local_maxiter": min(config.local_maxiter, 40),
        "sensitivity_epsilons": (0.2,),
        "sensitivity_powers": (2.0,),
        "sensitivity_taus": (0.05,),
        "use_global_search": False,
    }
    for name, value in overrides.items():
        setattr(config, name, value)
    config.validate()
    return overrides


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays if a caller supplied them in config data."""

    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serialisable")


def _flatten_rows(sensitivity: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one weight row into one CSV row per ``tau`` value."""

    records: list[dict[str, Any]] = []
    for row in sensitivity.get("rows", []):
        global_result = row.get("global") or {}
        candidate_by_tau = row.get("candidate_by_tau", [])
        for candidate in candidate_by_tau:
            records.append(
                {
                    "epsilon_w": row.get("epsilon_w"),
                    "p_w": row.get("p_w"),
                    "point_x": row.get("point", [None, None])[0],
                    "point_y": row.get("point", [None, None])[1],
                    "j_star": row.get("j_star"),
                    "j_under_reference_weight": row.get("j_under_reference_weight"),
                    "distance_from_s1": row.get("distance_from_s1"),
                    "bearing_from_s1_deg": row.get("bearing_from_s1_deg"),
                    "bearing_difference_deg": row.get("bearing_difference_deg"),
                    "reception_h": row.get("reception_h"),
                    "reception_h_tolerance": row.get("reception_h_tolerance"),
                    "strict_feasible": row.get("strict_feasible"),
                    "feasible_with_tolerance": row.get("feasible_with_tolerance"),
                    "optimization_source": row.get("optimization_source"),
                    "use_global_search": row.get("use_global_search"),
                    "selected_iterations": row.get("selected_iterations"),
                    "selected_function_evaluations": row.get("selected_function_evaluations"),
                    "local_run_count": row.get("local_run_count"),
                    "local_success_count": row.get("local_success_count"),
                    "local_feasible_count": row.get("local_feasible_count"),
                    "global_used": bool(global_result),
                    "global_iterations": global_result.get("iterations"),
                    "global_function_evaluations": global_result.get("function_evaluations"),
                    "global_feasible_with_tolerance": global_result.get("feasible_with_tolerance"),
                    "tau": candidate.get("tau"),
                    "threshold": candidate.get("threshold"),
                    "candidate_area": candidate.get("area"),
                    "accepted_cell_count": candidate.get("accepted_cell_count"),
                    "grid_size": (candidate.get("grid_shape") or [None])[0],
                    "x_step": candidate.get("x_step"),
                    "y_step": candidate.get("y_step"),
                    "distance_from_s1_min": candidate.get("distance_from_s1_min"),
                    "distance_from_s1_max": candidate.get("distance_from_s1_max"),
                    "component_count": candidate.get("component_count"),
                }
            )
    return records


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epsilon_w",
        "p_w",
        "point_x",
        "point_y",
        "j_star",
        "j_under_reference_weight",
        "distance_from_s1",
        "bearing_from_s1_deg",
        "bearing_difference_deg",
        "reception_h",
        "reception_h_tolerance",
        "strict_feasible",
        "feasible_with_tolerance",
        "optimization_source",
        "use_global_search",
        "selected_iterations",
        "selected_function_evaluations",
        "local_run_count",
        "local_success_count",
        "local_feasible_count",
        "global_used",
        "global_iterations",
        "global_function_evaluations",
        "global_feasible_with_tolerance",
        "tau",
        "threshold",
        "candidate_area",
        "accepted_cell_count",
        "grid_size",
        "x_step",
        "y_step",
        "distance_from_s1_min",
        "distance_from_s1_max",
        "component_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Question 2 JSON input path")
    source.add_argument("--demo", action="store_true", help="use S1=(0,0), bearing=32°")
    parser.add_argument(
        "--output",
        default="outputs/t2/q2_sensitivity.json",
        help="JSON output path (default: outputs/t2/q2_sensitivity.json)",
    )
    parser.add_argument(
        "--csv-output",
        help="flattened CSV output path (default: same stem as --output)",
    )
    parser.add_argument("--grid-size", type=int, help="regular-grid points per axis for candidate statistics")
    parser.add_argument("--epsilon-values", help="comma-separated epsilon_w values")
    parser.add_argument("--p-values", help="comma-separated p_w values")
    parser.add_argument("--tau-values", help="comma-separated tau values")
    parser.add_argument("--circle-sides", type=int)
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--validation-quadrature-order", type=int)
    parser.add_argument("--candidate-grid", type=int)
    parser.add_argument("--local-maxiter", type=int)
    parser.add_argument("--global-maxiter", type=int)
    parser.add_argument("--global-popsize", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-candidate-grids", action="store_true", help="store grid coordinates and masks in JSON")
    parser.add_argument("--smoke", action="store_true", help="run one low-cost combination for validation")
    global_group = parser.add_mutually_exclusive_group()
    global_group.add_argument("--global-search", dest="use_global_search", action="store_true", help="enable differential evolution for every weight pair")
    global_group.add_argument("--no-global", dest="use_global_search", action="store_false", help="skip differential evolution for every weight pair")
    parser.set_defaults(use_global_search=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        s1, bearing, config = load_input(args.input)
    else:
        # The explicit demo flag is retained for symmetry with solve_q2.py;
        # no input also defaults to the documented reproducible demo case.
        s1, bearing, config = parse_input(_demo_input())

    overrides: dict[str, Any] = {
        "circle_sides": args.circle_sides,
        "quadrature_order": args.quadrature_order,
        "validation_quadrature_order": args.validation_quadrature_order,
        "candidate_grid": args.candidate_grid,
        "local_maxiter": args.local_maxiter,
        "global_maxiter": args.global_maxiter,
        "global_popsize": args.global_popsize,
        "seed": args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    epsilon_values = _parse_values(args.epsilon_values, "epsilon-values")
    p_values = _parse_values(args.p_values, "p-values")
    tau_values = _parse_values(args.tau_values, "tau-values")
    if epsilon_values is not None:
        config.sensitivity_epsilons = epsilon_values
    if p_values is not None:
        config.sensitivity_powers = p_values
    if tau_values is not None:
        config.sensitivity_taus = tau_values
    if args.use_global_search is not None:
        config.use_global_search = args.use_global_search
    smoke_overrides: dict[str, Any] = {}
    if args.smoke:
        smoke_overrides = _smoke_config(config)
        if args.use_global_search is not None:
            config.use_global_search = args.use_global_search
            smoke_overrides["use_global_search"] = args.use_global_search
    config.validate()

    model = Question2Model(s1, bearing, config)
    sensitivity = model.sensitivity(
        grid_size=args.grid_size,
        use_global_search=args.use_global_search,
        include_candidate_grids=args.include_candidate_grids,
    )
    payload: dict[str, Any] = {
        "model": "question_2_continuous_robust_dop",
        "input": {
            "s1": [float(model.s1[0]), float(model.s1[1])],
            "svd_deg": float(model.bearing_deg),
        },
        "config": _serializable_config(config),
        "geometry": {
            "target_region_area": float(model.region.area),
            "target_region_halfplane_count": int(len(model.region.a)),
            "target_region_inradius": float(model.region.d_star),
            "target_region_centroid": model.region.centroid.astype(float).tolist(),
            "search_bounds": [[float(low), float(high)] for low, high in model.search_bounds],
            "quadrature_order": int(model.quadrature_order),
            "quadrature_node_count": int(len(model.nodes)),
        },
        "sensitivity": sensitivity,
        "metadata": {
            "purpose": "Question 2 parameter sensitivity for continuous robust second-detector selection",
            "run_mode": "smoke" if args.smoke else "configured",
            "smoke_overrides": _serializable_mapping(smoke_overrides),
            "optimization": {
                "per_weight_pair_reoptimization": True,
                "global_search": bool(sensitivity["use_global_search"]),
                "global_search_parameters": {
                    "maxiter": int(config.global_maxiter),
                    "popsize": int(config.global_popsize),
                    "seed": int(config.seed),
                },
                "local_method": "SLSQP with exact geometric reception oracle and finite singularity barrier",
                "local_maxiter": int(config.local_maxiter),
            },
            "comparison_rule": (
                "j_star values use each row's own epsilon_w and p_w. "
                "Use j_under_reference_weight for cross-row comparison."
            ),
            "candidate_region_rule": (
                "R_tau={P: H(P)<=0, J(P)<= (1+tau)J_star, J(P)<infinity}; "
                "reported area, distance range and component count are grid estimates."
            ),
            "grid_rule": sensitivity["grid_approximation"],
            "algorithm_steps": [
                "Construct the conservative continuous target polygon from the prior circle, first-reception circle and bearing-error wedge.",
                "Compute polygon depth d(G), maximum inradius d_star and weight epsilon_w+(1-epsilon_w)(d/d_star)^p_w at quadrature nodes.",
                "Evaluate the weighted continuous DOP integral with triangle Gauss-Duffy quadrature; reject line-polygon collinearity as J=+infinity.",
                "Use the geometric separation oracle H(P) for the semi-infinite all-target reception constraint.",
                "Re-optimise each weight pair, then classify near-optimal grid cell centres for every tau.",
                "Re-evaluate every selected point under the fixed reference weight before comparing weight rows.",
            ],
        },
    }
    output_path = Path(args.output)
    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    _write_json(output_path, payload)
    records = _flatten_rows(sensitivity)
    _write_csv(csv_path, records)
    print(f"saved JSON: {output_path}")
    print(f"saved CSV:  {csv_path}")
    print(f"weight combinations: {len(sensitivity['rows'])}; flattened rows: {len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
