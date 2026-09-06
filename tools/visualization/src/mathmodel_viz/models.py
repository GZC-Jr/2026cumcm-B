"""Stable data contracts shared by the catalog, CLI, and future recipes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkflowStage(str, Enum):
    """Common stages in a mathematical-modeling workflow."""

    EXPLORE = "explore"
    VALIDATE = "validate"
    COMPARE = "compare"
    FORECAST = "forecast"
    OPTIMIZE = "optimize"
    EXPLAIN = "explain"
    REPORT = "report"

    @classmethod
    def parse(cls, value: str | "WorkflowStage") -> "WorkflowStage":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(stage.value for stage in cls)
            raise ValueError(f"Unknown workflow stage {value!r}. Choose one of: {valid}.") from exc


@dataclass(frozen=True)
class VisualizationSpec:
    """One semantic visualization type, independent from a concrete renderer."""

    id: str
    name: str
    category: str
    question: str
    data_shapes: tuple[str, ...]
    stages: tuple[WorkflowStage, ...]
    backends: tuple[str, ...]
    interactive: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        """Return JSON-safe metadata for CLI and downstream automation."""

        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "question": self.question,
            "data_shapes": list(self.data_shapes),
            "stages": [stage.value for stage in self.stages],
            "backends": list(self.backends),
            "interactive": self.interactive,
            "note": self.note,
        }


@dataclass(frozen=True)
class Recommendation:
    """A scored catalog recommendation with explainable matching reasons."""

    spec: VisualizationSpec
    score: int
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        payload = self.spec.as_dict()
        payload.update({"score": self.score, "reasons": list(self.reasons)})
        return payload
