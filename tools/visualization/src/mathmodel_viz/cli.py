"""Command-line entry point for catalog lookup and environment inspection."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence

from .backends import list_backends
from .catalog import CATEGORY_LABELS
from .models import WorkflowStage
from .registry import DEFAULT_REGISTRY


def _configure_utf8_output() -> None:
    """Keep Chinese catalog metadata legible in Windows terminals and captured logs."""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def _format_specs(specs: Sequence[object], as_json: bool) -> str:
    if as_json:
        return json.dumps([spec.as_dict() for spec in specs], ensure_ascii=False, indent=2)
    lines: list[str] = []
    for spec in specs:
        stages = ", ".join(stage.value for stage in spec.stages)
        shapes = ", ".join(spec.data_shapes)
        lines.extend(
            (
                f"{spec.id} | {spec.name}",
                f"  category: {spec.category} ({CATEGORY_LABELS[spec.category]})",
                f"  question: {spec.question}",
                f"  stages: {stages}; shapes: {shapes}; backends: {', '.join(spec.backends)}",
            )
        )
    return "\n".join(lines)


def _format_recommendations(recommendations: Sequence[object], as_json: bool) -> str:
    if as_json:
        return json.dumps([item.as_dict() for item in recommendations], ensure_ascii=False, indent=2)
    lines: list[str] = []
    for item in recommendations:
        lines.extend(
            (
                f"[{item.score}] {item.spec.id} | {item.spec.name}",
                f"  {item.spec.question}",
                f"  matched: {'; '.join(item.reasons)}",
                f"  backend: {', '.join(item.spec.backends)}",
            )
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mathmodel-viz",
        description="Query the mathematical-modeling visualization catalog.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List catalog entries.")
    list_parser.add_argument("--category", choices=tuple(CATEGORY_LABELS))
    list_parser.add_argument("--stage", choices=tuple(stage.value for stage in WorkflowStage))
    list_parser.add_argument("--format", choices=("text", "json"), default="text")

    recommend_parser = subparsers.add_parser("recommend", help="Recommend visuals for a task.")
    recommend_parser.add_argument("--stage", choices=tuple(stage.value for stage in WorkflowStage))
    recommend_parser.add_argument("--shape", action="append", default=[], help="Data-shape tag; may repeat.")
    recommend_parser.add_argument("--mode", choices=("static", "interactive"))
    recommend_parser.add_argument("--limit", type=int, default=5)
    recommend_parser.add_argument("--format", choices=("text", "json"), default="text")

    backends_parser = subparsers.add_parser("backends", help="Inspect rendering backends.")
    backends_parser.add_argument("--available-only", action="store_true")
    backends_parser.add_argument("--format", choices=("text", "json"), default="text")

    subparsers.add_parser("report", help="Show a compact catalog summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_output()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        specs = DEFAULT_REGISTRY.list(category=args.category, stage=args.stage)
        print(_format_specs(specs, args.format == "json"))
        return 0

    if args.command == "recommend":
        interactive = None
        if args.mode is not None:
            interactive = args.mode == "interactive"
        try:
            results = DEFAULT_REGISTRY.recommend(
                stage=args.stage,
                data_shapes=args.shape,
                interactive=interactive,
                limit=args.limit,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(_format_recommendations(results, args.format == "json"))
        return 0

    if args.command == "backends":
        backends = list_backends(available_only=args.available_only)
        if args.format == "json":
            payload = [
                {
                    "key": backend.key,
                    "package": backend.package,
                    "installed": backend.installed,
                    "use_case": backend.use_case,
                    "optional_group": backend.optional_group,
                }
                for backend in backends
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for backend in backends:
                group = backend.optional_group or "core"
                state = "installed" if backend.installed else "not installed"
                print(f"{backend.key} | {state} | group={group} | {backend.use_case}")
        return 0

    if args.command == "report":
        counts = Counter(spec.category for spec in DEFAULT_REGISTRY.list())
        print(f"Visualization types: {sum(counts.values())}")
        for key in DEFAULT_REGISTRY.categories:
            print(f"- {key} ({CATEGORY_LABELS[key]}): {counts[key]}")
        print(f"Workflow stages: {', '.join(stage.value for stage in WorkflowStage)}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2
