"""Generate one publication-ready model-diagnostic figure with the shared helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mathmodel_viz import configure_matplotlib, save_figure


def build_demo(output_dir: Path) -> Path:
    rng = np.random.default_rng(20260906)
    observed = np.linspace(15, 120, 80)
    predicted = observed * 0.96 + 4 + rng.normal(0, 5, size=observed.size)
    residual = predicted - observed

    configure_matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(observed, predicted, alpha=0.75, label="observations")
    lower, upper = min(observed.min(), predicted.min()), max(observed.max(), predicted.max())
    axes[0].plot([lower, upper], [lower, upper], "--", color="tab:red", label="ideal")
    axes[0].set(title="Prediction vs. observation", xlabel="Observed", ylabel="Predicted")
    axes[0].legend()

    axes[1].axhline(0, color="tab:red", linestyle="--", linewidth=1)
    axes[1].scatter(predicted, residual, alpha=0.75)
    axes[1].set(title="Residual diagnostic", xlabel="Predicted", ylabel="Residual")

    figure.suptitle("Regression diagnostic example")
    figure.tight_layout()
    return save_figure(figure, output_dir / "regression_diagnostic.png")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    output = root / "outputs" / "figures" / "visualization-demo"
    path = build_demo(output)
    print(f"Saved: {path}")
