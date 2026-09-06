"""Lookup and explainable recommendation over the visualization catalog."""

from __future__ import annotations

from collections.abc import Iterable

from .catalog import CATALOG, CATEGORY_LABELS
from .models import Recommendation, VisualizationSpec, WorkflowStage


class VisualizationRegistry:
    """Validated index over semantic visualization metadata."""

    def __init__(self, specs: Iterable[VisualizationSpec] = CATALOG) -> None:
        self._specs = tuple(specs)
        if not self._specs:
            raise ValueError("The visualization catalog cannot be empty.")

        ids = [spec.id for spec in self._specs]
        duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
        if duplicates:
            raise ValueError(f"Duplicate visualization ids: {', '.join(duplicates)}")

        unknown_categories = sorted({spec.category for spec in self._specs} - set(CATEGORY_LABELS))
        if unknown_categories:
            raise ValueError(f"Catalog has unknown categories: {', '.join(unknown_categories)}")
        self._by_id = {spec.id: spec for spec in self._specs}

    @property
    def categories(self) -> tuple[str, ...]:
        """Return catalog category keys in their documented order."""

        present = {spec.category for spec in self._specs}
        return tuple(key for key in CATEGORY_LABELS if key in present)

    def get(self, identifier: str) -> VisualizationSpec:
        """Return a visualization by stable identifier."""

        try:
            return self._by_id[identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown visualization id: {identifier}") from exc

    def list(
        self,
        *,
        category: str | None = None,
        stage: WorkflowStage | str | None = None,
    ) -> tuple[VisualizationSpec, ...]:
        """List catalog entries, optionally constrained by category and stage."""

        if category is not None and category not in CATEGORY_LABELS:
            valid = ", ".join(CATEGORY_LABELS)
            raise ValueError(f"Unknown category {category!r}. Choose one of: {valid}.")
        target_stage = WorkflowStage.parse(stage) if stage is not None else None
        return tuple(
            spec
            for spec in self._specs
            if (category is None or spec.category == category)
            and (target_stage is None or target_stage in spec.stages)
        )

    def recommend(
        self,
        *,
        stage: WorkflowStage | str | None = None,
        data_shapes: Iterable[str] = (),
        interactive: bool | None = None,
        limit: int = 5,
    ) -> tuple[Recommendation, ...]:
        """Rank entries by matching workflow stage, data shapes, and output mode."""

        if limit < 1:
            raise ValueError("limit must be at least 1.")
        target_stage = WorkflowStage.parse(stage) if stage is not None else None
        target_shapes = tuple(dict.fromkeys(shape.strip() for shape in data_shapes if shape.strip()))
        if target_stage is None and not target_shapes and interactive is None:
            raise ValueError("Provide at least one of stage, data_shapes, or interactive.")

        recommendations: list[Recommendation] = []
        for spec in self._specs:
            score = 0
            reasons: list[str] = []
            if target_stage is not None and target_stage in spec.stages:
                score += 5
                reasons.append(f"workflow stage: {target_stage.value}")

            matched_shapes = tuple(shape for shape in target_shapes if shape in spec.data_shapes)
            if matched_shapes:
                score += 3 * len(matched_shapes)
                reasons.append(f"data shape: {', '.join(matched_shapes)}")

            if interactive is not None and spec.interactive is interactive:
                score += 1
                reasons.append("output mode matches")

            if score:
                recommendations.append(Recommendation(spec, score, tuple(reasons)))

        return tuple(
            sorted(recommendations, key=lambda item: (-item.score, item.spec.id))[:limit]
        )


DEFAULT_REGISTRY = VisualizationRegistry()
