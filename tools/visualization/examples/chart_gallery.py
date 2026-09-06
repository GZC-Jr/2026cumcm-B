"""Generate a deterministic gallery that exercises several visualization categories.

Run from ``tools/visualization`` after installing the package, or set
``PYTHONPATH=src`` before executing this file directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.integrate import solve_ivp
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL

from mathmodel_viz import DEFAULT_REGISTRY, configure_matplotlib, save_figure


@dataclass(frozen=True)
class Artifact:
    """One generated artifact with its matching catalog entry."""

    filename: str
    spec_id: str
    description: str


def _save(figure: plt.Figure, output_dir: Path, filename: str) -> Path:
    figure.tight_layout()
    path = save_figure(figure, output_dir / filename)
    plt.close(figure)
    return path


def distribution_figure(rng: np.random.Generator) -> plt.Figure:
    group_a = rng.normal(52, 9, 320)
    group_b = rng.gamma(6, 6, 320) + 20
    data = pd.DataFrame(
        {
            "value": np.concatenate([group_a, group_b]),
            "group": ["Group A"] * len(group_a) + ["Group B"] * len(group_b),
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    sns.histplot(data=data, x="value", hue="group", kde=True, stat="density", common_norm=False, ax=axes[0])
    axes[0].set(title="Histogram and KDE", xlabel="Value", ylabel="Density")
    sns.ecdfplot(data=data, x="value", hue="group", ax=axes[1])
    axes[1].set(title="Empirical CDF", xlabel="Value", ylabel="Cumulative proportion")
    sns.boxplot(data=data, x="group", y="value", hue="group", legend=False, ax=axes[2])
    sns.stripplot(data=data.sample(160, random_state=20260906), x="group", y="value", color="0.2", alpha=0.35, ax=axes[2])
    axes[2].set(title="Boxplot with observations", xlabel="", ylabel="Value")
    figure.suptitle("Distribution and group-comparison diagnostics")
    return figure


def relationship_figure(rng: np.random.Generator) -> plt.Figure:
    x = rng.normal(0, 1.2, 1_400)
    y = 0.8 * x + 0.45 * x**2 + rng.normal(0, 0.85, x.size)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x, y, s=12, alpha=0.28, color="tab:blue")
    axes[0].set(title="Scatter plot", xlabel="Input x", ylabel="Response y")
    hexbin = axes[1].hexbin(x, y, gridsize=28, mincnt=1, cmap="viridis")
    figure.colorbar(hexbin, ax=axes[1], label="Observations per bin")
    axes[1].set(title="Hexbin density", xlabel="Input x", ylabel="Response y")
    figure.suptitle("Relationship diagnostics for dense paired data")
    return figure


def correlation_figure(rng: np.random.Generator) -> plt.Figure:
    base = rng.normal(size=(420, 3))
    frame = pd.DataFrame(
        {
            "demand": base[:, 0],
            "price": -0.65 * base[:, 0] + 0.55 * base[:, 1],
            "inventory": 0.40 * base[:, 0] + 0.30 * base[:, 2],
            "distance": base[:, 1] - 0.25 * base[:, 2],
            "delay": 0.45 * base[:, 1] + 0.35 * base[:, 2],
            "quality": -0.35 * base[:, 0] + 0.50 * base[:, 2],
        }
    )
    figure, axis = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(frame.corr(), annot=True, fmt=".2f", center=0, cmap="coolwarm", square=True, ax=axis)
    axis.set_title("Correlation matrix")
    return figure


def time_series_figure(rng: np.random.Generator) -> plt.Figure:
    periods = 84
    dates = pd.date_range("2024-01-01", periods=periods, freq="W")
    t = np.arange(periods)
    observed = 80 + 0.28 * t + 9 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 3, periods)
    forecast_index = dates[-20:]
    trend = 80 + 0.28 * t[-20:] + 9 * np.sin(2 * np.pi * t[-20:] / 12)
    uncertainty = 4 + np.linspace(0, 4, 20)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(dates, observed, label="Observed", color="tab:blue")
    axes[0].plot(forecast_index, trend, label="Forecast mean", color="tab:orange")
    axes[0].fill_between(forecast_index, trend - 1.96 * uncertainty, trend + 1.96 * uncertainty, color="tab:orange", alpha=0.22, label="95% interval")
    axes[0].axvline(forecast_index[0], color="0.35", linestyle="--", linewidth=1)
    axes[0].set(title="Time series and forecast interval", xlabel="Week", ylabel="Demand")
    axes[0].legend(loc="upper left")
    plot_acf(observed, lags=24, zero=False, ax=axes[1])
    axes[1].set(title="Autocorrelation diagnostic", xlabel="Lag (weeks)", ylabel="Autocorrelation")
    figure.suptitle("Forecast and serial-dependence checks")
    return figure


def optimization_figure(rng: np.random.Generator) -> plt.Figure:
    iterations = np.arange(1, 121)
    runs = []
    for _ in range(12):
        noise = rng.normal(0, 0.025, iterations.size)
        runs.append(1.4 * np.exp(-iterations / 33) + 0.08 + noise)
    runs_array = np.vstack(runs)
    median = np.median(runs_array, axis=0)
    low, high = np.quantile(runs_array, [0.1, 0.9], axis=0)

    front_x = np.linspace(0.9, 5.6, 24)
    front_y = 6.6 / (front_x + 0.5) + 0.35
    dominated_x = rng.uniform(1.1, 6.0, 75)
    dominated_y = 6.6 / (dominated_x + 0.5) + rng.uniform(0.3, 2.8, dominated_x.size)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(iterations, median, color="tab:blue", label="Median objective")
    axes[0].fill_between(iterations, low, high, color="tab:blue", alpha=0.2, label="10%--90% across runs")
    axes[0].set(title="Optimization convergence", xlabel="Iteration", ylabel="Objective value")
    axes[0].legend()
    axes[1].scatter(dominated_x, dominated_y, s=26, alpha=0.55, color="0.55", label="Dominated candidates")
    axes[1].plot(front_x, front_y, "o-", color="tab:red", label="Pareto front")
    axes[1].set(title="Two-objective Pareto front", xlabel="Cost", ylabel="Risk")
    axes[1].legend()
    figure.suptitle("Optimization performance and trade-off")
    return figure


def sensitivity_figure(rng: np.random.Generator) -> plt.Figure:
    parameters = ["Demand", "Price", "Capacity", "Travel time", "Penalty", "Service level"]
    scenarios = ["-20%", "-10%", "Baseline", "+10%", "+20%"]
    base = rng.normal(0, 1, (len(parameters), len(scenarios)))
    direction = np.array([[-1], [-0.6], [0.8], [0.4], [0.6], [0.3]])
    change = np.array([-1.6, -0.8, 0, 0.8, 1.6])
    effects = base * 0.15 + direction * change
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(effects, cmap="RdBu_r", center=0, annot=True, fmt=".2f", xticklabels=scenarios, yticklabels=parameters, cbar_kws={"label": "Relative objective change"}, ax=axis)
    axis.set(title="Parameter sensitivity heatmap", xlabel="Scenario", ylabel="Parameter")
    return figure


def network_figure(rng: np.random.Generator) -> plt.Figure:
    graph = nx.planted_partition_graph(l=3, k=8, p_in=0.55, p_out=0.045, seed=20260906)
    positions = nx.spring_layout(graph, seed=20260906, k=0.45)
    groups = [node // 8 for node in graph.nodes]
    degrees = np.array([graph.degree(node) for node in graph.nodes])
    community_cmap = plt.get_cmap("Set2", 3)
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    nx.draw_networkx_edges(graph, positions, alpha=0.3, width=0.8, ax=axis)
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=groups,
        node_size=90 + 35 * degrees,
        cmap=community_cmap,
        vmin=-0.5,
        vmax=2.5,
        edgecolors="white",
        linewidths=0.8,
        ax=axis,
    )
    figure.colorbar(nodes, ax=axis, ticks=[0, 1, 2], label="Community")
    axis.set_title("Network communities and degree-scaled nodes")
    axis.set_axis_off()
    return figure


def field_figure() -> plt.Figure:
    x = np.linspace(-2.2, 2.2, 31)
    y = np.linspace(-2.2, 2.2, 31)
    xx, yy = np.meshgrid(x, y)
    potential = 0.35 * (xx**4 + yy**4) - 0.9 * (xx**2 + yy**2) + 0.28 * xx * yy
    grad_x = 1.4 * xx**3 - 1.8 * xx + 0.28 * yy
    grad_y = 1.4 * yy**3 - 1.8 * yy + 0.28 * xx
    sample = np.s_[::3, ::3]
    direction_x = -grad_x[sample]
    direction_y = -grad_y[sample]
    magnitude = np.hypot(direction_x, direction_y)
    safe_magnitude = np.where(magnitude > 1e-9, magnitude, 1.0)
    figure, axis = plt.subplots(figsize=(7, 5.5))
    contours = axis.contourf(xx, yy, potential, levels=22, cmap="viridis")
    axis.quiver(
        xx[sample],
        yy[sample],
        direction_x / safe_magnitude,
        direction_y / safe_magnitude,
        color="white",
        alpha=0.78,
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.003,
    )
    figure.colorbar(contours, ax=axis, label="Potential value")
    axis.set(title="Contour and negative-gradient vector field", xlabel="State x", ylabel="State y", aspect="equal")
    return figure


def scheduling_figure() -> plt.Figure:
    tasks = [
        ("Prepare", 0, 3, "tab:blue"),
        ("Process A", 3, 5, "tab:orange"),
        ("Process B", 4, 4, "tab:green"),
        ("Inspect", 8, 2, "tab:red"),
        ("Deliver", 10, 3, "tab:purple"),
    ]
    timeline = np.arange(0, 14)
    capacity = np.full_like(timeline, 2, dtype=float)
    load = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True, height_ratios=(2, 1))
    for row, (name, start, duration, color) in enumerate(tasks):
        axes[0].barh(row, duration, left=start, color=color, height=0.62)
        axes[0].text(start + duration / 2, row, name, ha="center", va="center", color="white", fontsize=9)
    axes[0].set(yticks=range(len(tasks)), yticklabels=[task[0] for task in tasks], title="Gantt schedule", ylabel="Task")
    axes[0].invert_yaxis()
    axes[1].step(timeline, load, where="post", label="Resource load", color="tab:blue")
    axes[1].step(timeline, capacity, where="post", label="Capacity", color="tab:red", linestyle="--")
    axes[1].set(title="Resource load", xlabel="Time period", ylabel="Units", ylim=(0, 2.5))
    axes[1].legend(loc="upper right")
    figure.suptitle("Scheduling and capacity validation")
    return figure


def tornado_figure() -> plt.Figure:
    factors = ["Demand", "Unit cost", "Penalty", "Capacity", "Travel time", "Service target"]
    low = np.array([-19, -13, -10, -7, -5, -3])
    high = np.array([24, 16, 11, 8, 6, 4])
    y = np.arange(len(factors))
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.barh(y, low, color="tab:blue", alpha=0.82, label="Low assumption")
    axis.barh(y, high, color="tab:orange", alpha=0.82, label="High assumption")
    axis.axvline(0, color="0.25", linewidth=1)
    axis.set(yticks=y, yticklabels=factors, title="Tornado sensitivity chart", xlabel="Change in objective relative to baseline (%)")
    axis.legend(loc="lower right")
    return figure


def data_quality_figure(rng: np.random.Generator) -> plt.Figure:
    """Show structured missingness alongside a robust outlier diagnostic."""

    columns = ["Demand", "Price", "Distance", "Lead time", "Quality"]
    missing = rng.random((100, len(columns))) < np.array([0.03, 0.08, 0.12, 0.05, 0.02])
    missing[35:55, 2] = True
    missing[68:80, 1] = True
    observations = np.concatenate([rng.normal(52, 6.5, 220), np.array([76, 79, 83, 87])])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": (1.25, 1)})
    image = axes[0].imshow(missing.T, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    axes[0].set(
        title="Missing-data pattern",
        xlabel="Observation index",
        ylabel="Variable",
        yticks=np.arange(len(columns)),
        yticklabels=columns,
    )
    figure.colorbar(image, ax=axes[0], ticks=[0, 1], label="Missing indicator")

    jitter = rng.normal(0, 0.055, observations.size)
    axes[1].boxplot(observations, vert=False, widths=0.55, patch_artist=True, boxprops={"facecolor": "tab:blue", "alpha": 0.4})
    axes[1].scatter(observations, np.ones_like(observations) + jitter, s=13, alpha=0.36, color="0.2")
    axes[1].set(title="Outlier diagnostic", xlabel="Measured value", yticks=[1], yticklabels=["Sample"])
    figure.suptitle("Data-quality screening before modeling")
    return figure


def decomposition_control_figure(rng: np.random.Generator) -> plt.Figure:
    """Exercise STL decomposition and a process control chart with a shift."""

    periods = 120
    dates = pd.date_range("2024-01-01", periods=periods, freq="W")
    index = np.arange(periods)
    observed = 95 + 0.18 * index + 7 * np.sin(2 * np.pi * index / 12) + rng.normal(0, 2.1, periods)
    observed[91:96] += 8
    decomposition = STL(observed, period=12, robust=True).fit()
    center = np.mean(observed[:60])
    sigma = np.std(observed[:60], ddof=1)

    figure, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    axes[0, 0].plot(dates, observed, color="tab:blue", label="Observed")
    axes[0, 0].plot(dates, decomposition.trend, color="tab:orange", label="STL trend")
    axes[0, 0].set(title="Observed series and trend", ylabel="Demand")
    axes[0, 0].legend(loc="upper left")

    axes[1, 0].plot(dates, decomposition.seasonal, color="tab:green", label="Seasonal")
    axes[1, 0].plot(dates, decomposition.resid, color="tab:purple", alpha=0.8, label="Residual")
    axes[1, 0].axhline(0, color="0.3", linewidth=0.8)
    axes[1, 0].set(title="STL seasonal and residual components", xlabel="Week", ylabel="Component")
    axes[1, 0].legend(loc="upper left")

    axes[0, 1].plot(dates, observed, "o-", markersize=3, color="tab:blue", label="Process value")
    axes[0, 1].axhline(center, color="0.25", linewidth=1, label="Center line")
    axes[0, 1].axhline(center + 3 * sigma, color="tab:red", linestyle="--", label="Upper control limit")
    axes[0, 1].axhline(center - 3 * sigma, color="tab:red", linestyle="--", label="Lower control limit")
    axes[0, 1].set(title="Individuals control chart", ylabel="Demand")
    axes[0, 1].legend(loc="upper left", fontsize=8)

    axes[1, 1].hist(decomposition.resid, bins=18, density=True, color="tab:purple", alpha=0.7)
    axes[1, 1].axvline(0, color="0.25", linewidth=1)
    axes[1, 1].set(title="Residual distribution", xlabel="Residual", ylabel="Density")
    figure.suptitle("Time-series decomposition and process stability checks")
    return figure


def network_adjacency_figure() -> plt.Figure:
    """Compare a block-structured adjacency matrix with degree distribution."""

    sizes = [8, 9, 7]
    graph = nx.stochastic_block_model(
        sizes,
        [[0.55, 0.06, 0.04], [0.06, 0.48, 0.08], [0.04, 0.08, 0.60]],
        seed=20260906,
    )
    adjacency = nx.to_numpy_array(graph, nodelist=range(sum(sizes)))
    degrees = np.array([degree for _, degree in graph.degree()])
    boundaries = np.cumsum(sizes)[:-1] - 0.5

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": (1.1, 0.9)})
    image = axes[0].imshow(adjacency, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    for boundary in boundaries:
        axes[0].axhline(boundary, color="tab:red", linewidth=1)
        axes[0].axvline(boundary, color="tab:red", linewidth=1)
    axes[0].set(title="Community-ordered adjacency matrix", xlabel="Target node", ylabel="Source node")
    figure.colorbar(image, ax=axes[0], ticks=[0, 1], label="Edge present")

    bins = np.arange(degrees.min() - 0.5, degrees.max() + 1.5)
    axes[1].hist(degrees, bins=bins, color="tab:green", rwidth=0.82)
    axes[1].set(title="Degree distribution", xlabel="Degree", ylabel="Node count", xticks=np.arange(degrees.min(), degrees.max() + 1))
    figure.suptitle("Dense-network structure diagnostics")
    return figure


def dynamics_figure() -> plt.Figure:
    """Plot a predator-prey phase plane beside a synthetic spatiotemporal field."""

    alpha, beta, delta, gamma = 1.0, 0.28, 0.17, 0.80

    def dynamics(_: float, state: np.ndarray) -> tuple[float, float]:
        prey, predator = state
        return alpha * prey - beta * prey * predator, delta * prey * predator - gamma * predator

    prey = np.linspace(0.2, 8.0, 25)
    predator = np.linspace(0.2, 6.5, 25)
    xx, yy = np.meshgrid(prey, predator)
    dx, dy = dynamics(0, np.array([xx, yy]))
    speed = np.hypot(dx, dy)
    solution_times = np.linspace(0, 38, 600)

    space = np.linspace(0, 100, 180)
    time = np.linspace(0, 48, 160)
    space_grid, time_grid = np.meshgrid(space, time)
    field = (
        0.95 * np.exp(-((space_grid - (18 + 1.05 * time_grid)) / 13) ** 2)
        + 0.55 * np.exp(-((space_grid - (66 - 0.42 * time_grid)) / 17) ** 2)
    ) * np.exp(-time_grid / 110)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].streamplot(xx, yy, dx, dy, color=speed, cmap="viridis", density=1.1, linewidth=1)
    for initial, color in [((2.0, 1.0), "tab:blue"), ((5.0, 3.5), "tab:orange"), ((7.0, 1.7), "tab:green")]:
        solution = solve_ivp(dynamics, (solution_times[0], solution_times[-1]), initial, t_eval=solution_times)
        axes[0].plot(solution.y[0], solution.y[1], color=color, label=f"Initial {initial}")
        axes[0].plot(initial[0], initial[1], "o", color=color)
    axes[0].plot(gamma / delta, alpha / beta, "*", color="tab:red", markersize=12, label="Equilibrium")
    axes[0].set(title="Phase plane and trajectories", xlabel="Prey population", ylabel="Predator population", xlim=(0, 8), ylim=(0, 6.5))
    axes[0].legend(loc="upper right", fontsize=8)

    image = axes[1].imshow(field, aspect="auto", origin="lower", extent=(space.min(), space.max(), time.min(), time.max()), cmap="magma")
    axes[1].set(title="Spatiotemporal propagation field", xlabel="Space coordinate", ylabel="Simulation time")
    figure.colorbar(image, ax=axes[1], label="State intensity")
    figure.suptitle("Dynamics and simulation visual diagnostics")
    return figure


def feasible_allocation_figure() -> plt.Figure:
    """Show a linear-program feasible region and a capacity allocation matrix."""

    x = np.linspace(0, 5.2, 260)
    y = np.linspace(0, 6.2, 260)
    xx, yy = np.meshgrid(x, y)
    feasible = (3 * xx + yy <= 12) & (xx + 2 * yy <= 10)
    objective = 5 * xx + 4 * yy
    optimal = np.array([2.8, 3.6])
    allocation = np.array(
        [
            [18, 8, 4, 0],
            [7, 15, 5, 6],
            [0, 6, 16, 9],
            [4, 0, 7, 18],
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": (1, 1.05)})
    axes[0].contourf(xx, yy, feasible.astype(int), levels=[-0.5, 0.5, 1.5], colors=["#f3f3f3", "#ccebc5"], alpha=0.9)
    contours = axes[0].contour(xx, yy, np.where(feasible, objective, np.nan), levels=7, cmap="viridis")
    axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    axes[0].plot(x, 12 - 3 * x, color="tab:red", label=r"$3x + y = 12$")
    axes[0].plot(x, (10 - x) / 2, color="tab:blue", label=r"$x + 2y = 10$")
    axes[0].plot(*optimal, "*", color="tab:orange", markersize=14, label="Optimal vertex")
    axes[0].set(title="Feasible region and objective contours", xlabel="Decision x", ylabel="Decision y", xlim=(0, 5.2), ylim=(0, 6.2))
    axes[0].legend(loc="upper right", fontsize=8)

    sns.heatmap(
        allocation,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar_kws={"label": "Allocated units"},
        xticklabels=["North", "East", "South", "West"],
        yticklabels=["Resource A", "Resource B", "Resource C", "Resource D"],
        ax=axes[1],
    )
    axes[1].set(title="Resource-to-region allocation", xlabel="Region", ylabel="Resource")
    figure.suptitle("Optimization feasibility and allocation checks")
    return figure


def queue_metrics_figure(rng: np.random.Generator) -> plt.Figure:
    """Simulate queue length, wait, and utilization over a demand surge."""

    intervals = np.arange(96)
    demand_rate = 1.45 + 0.85 * np.exp(-((intervals - 48) / 15) ** 2)
    arrivals = rng.poisson(demand_rate)
    service_capacity = np.where((intervals >= 42) & (intervals <= 60), 1, 2)
    queue = np.zeros(intervals.size)
    served = np.zeros(intervals.size)
    for index in range(intervals.size):
        incoming = arrivals[index] + (queue[index - 1] if index else 0)
        served[index] = min(incoming, service_capacity[index])
        queue[index] = incoming - served[index]
    wait_estimate = queue / np.maximum(service_capacity, 1)
    utilization = served / service_capacity

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 7), sharex=True)
    axes[0].step(intervals, queue, where="mid", color="tab:blue", label="Queue length")
    axes[0].fill_between(intervals, queue, step="mid", color="tab:blue", alpha=0.2)
    axes[0].set(title="Queue length", ylabel="Customers")
    axes[0].legend(loc="upper left")

    axes[1].plot(intervals, wait_estimate, color="tab:orange", label="Estimated wait")
    axes[1].axhline(3, color="tab:red", linestyle="--", label="Service target")
    axes[1].set(title="Waiting-time proxy", ylabel="Intervals")
    axes[1].legend(loc="upper left")

    axes[2].step(intervals, utilization, where="mid", color="tab:green", label="Utilization")
    axes[2].axhline(0.85, color="tab:red", linestyle="--", label="Preferred ceiling")
    axes[2].set(title="Server utilization", xlabel="15-minute interval", ylabel="Share", ylim=(0, 1.12))
    axes[2].legend(loc="upper left")
    figure.suptitle("Queueing-system performance during a capacity disruption")
    return figure


def scenario_fan_ranking_figure(rng: np.random.Generator) -> plt.Figure:
    """Visualize stochastic forecast uncertainty and ranking stability."""

    periods = np.arange(25)
    shocks = rng.normal(0.025, 0.075, size=(900, periods.size))
    paths = 100 * np.cumprod(1 + shocks, axis=1)
    q05, q25, q50, q75, q95 = np.quantile(paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)

    weights = rng.dirichlet([2.8, 2.1, 1.9], size=3_000)
    alternatives = np.array(
        [
            [0.88, 0.66, 0.57],
            [0.74, 0.84, 0.62],
            [0.65, 0.70, 0.91],
            [0.80, 0.72, 0.72],
        ]
    )
    scores = weights @ alternatives.T
    ranks = np.argsort(np.argsort(-scores, axis=1), axis=1) + 1
    rank_probability = np.vstack([(ranks == rank).mean(axis=0) for rank in range(1, 5)]).T

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": (1.1, 1)})
    axes[0].fill_between(periods, q05, q95, color="tab:blue", alpha=0.16, label="5%--95%")
    axes[0].fill_between(periods, q25, q75, color="tab:blue", alpha=0.32, label="25%--75%")
    axes[0].plot(periods, q50, color="tab:blue", linewidth=2, label="Median")
    axes[0].set(title="Scenario fan chart", xlabel="Forecast period", ylabel="Indexed outcome")
    axes[0].legend(loc="upper left")

    sns.heatmap(
        rank_probability,
        annot=True,
        fmt=".0%",
        cmap="Blues",
        cbar_kws={"label": "Probability"},
        xticklabels=["Rank 1", "Rank 2", "Rank 3", "Rank 4"],
        yticklabels=["Alternative A", "Alternative B", "Alternative C", "Alternative D"],
        ax=axes[1],
    )
    axes[1].set(title="Ranking stability under weight variation", xlabel="Resulting rank", ylabel="Alternative")
    figure.suptitle("Risk scenarios and multi-criteria decision robustness")
    return figure


def machine_learning_diagnostics_figure() -> plt.Figure:
    """Exercise common regression and classification validation views."""

    features, target = make_regression(n_samples=520, n_features=7, noise=20, random_state=20260906)
    train_x, test_x, train_y_reg, test_y_reg = train_test_split(features, target, test_size=0.32, random_state=17)
    regression = Ridge(alpha=3.0).fit(train_x, train_y_reg)
    predicted = regression.predict(test_x)
    residuals = test_y_reg - predicted

    class_features, labels = make_classification(
        n_samples=900,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.62, 0.38],
        class_sep=1.05,
        flip_y=0.04,
        random_state=20260906,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        class_features, labels, test_size=0.30, stratify=labels, random_state=17
    )
    classifier = LogisticRegression(max_iter=1_000).fit(train_x, train_y)
    probability = classifier.predict_proba(test_x)[:, 1]
    predictions = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(test_y, predictions)
    fpr, tpr, _ = roc_curve(test_y, probability)
    precision, recall, _ = precision_recall_curve(test_y, probability)
    calibration_true, calibration_predicted = calibration_curve(test_y, probability, n_bins=10)

    figure, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    low, high = np.percentile(np.concatenate([test_y_reg, predicted]), [1, 99])
    axes[0, 0].scatter(test_y_reg, predicted, color="tab:blue", alpha=0.55, s=20)
    axes[0, 0].plot([low, high], [low, high], color="tab:red", linestyle="--", label="Ideal")
    axes[0, 0].set(title="Predicted vs observed", xlabel="Observed response", ylabel="Predicted response")
    axes[0, 0].legend(loc="upper left")

    axes[0, 1].scatter(predicted, residuals, color="tab:purple", alpha=0.55, s=20)
    axes[0, 1].axhline(0, color="0.25", linewidth=1)
    axes[0, 1].set(title="Regression residuals", xlabel="Predicted response", ylabel="Residual")

    axes[0, 2].hist(residuals, bins=18, color="tab:purple", alpha=0.72)
    axes[0, 2].axvline(0, color="0.25", linewidth=1)
    axes[0, 2].set(title="Residual distribution", xlabel="Residual", ylabel="Count")

    coefficients = np.abs(regression.coef_)
    axes[0, 3].barh(np.arange(coefficients.size), coefficients, color="tab:green")
    axes[0, 3].set(title="Linear feature magnitude", xlabel="Absolute coefficient", ylabel="Feature", yticks=np.arange(coefficients.size), yticklabels=[f"x{i + 1}" for i in range(coefficients.size)])

    image = axes[1, 0].imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1, 0].set(title="Confusion matrix", xlabel="Predicted class", ylabel="True class", xticks=[0, 1], yticks=[0, 1])
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].plot(fpr, tpr, color="tab:blue", label=f"AUC = {auc(fpr, tpr):.2f}")
    axes[1, 1].plot([0, 1], [0, 1], color="0.4", linestyle="--")
    axes[1, 1].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend(loc="lower right")

    axes[1, 2].plot(recall, precision, color="tab:orange", label=f"Area = {auc(recall, precision):.2f}")
    axes[1, 2].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[1, 2].legend(loc="lower left")

    axes[1, 3].plot(calibration_predicted, calibration_true, "o-", color="tab:red", label="Classifier")
    axes[1, 3].plot([0, 1], [0, 1], color="0.4", linestyle="--", label="Perfect calibration")
    axes[1, 3].set(title="Probability calibration", xlabel="Mean predicted probability", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    axes[1, 3].legend(loc="upper left", fontsize=8)
    figure.suptitle("Regression and classification diagnostic suite")
    return figure


def statistical_inference_figure(rng: np.random.Generator) -> plt.Figure:
    """Combine effect-size intervals with residual and influence diagnostics."""

    labels = ["Demand", "Price", "Capacity", "Travel time", "Policy"]
    estimates = np.array([0.42, -0.31, 0.27, -0.18, 0.11])
    lower = estimates - np.array([0.12, 0.10, 0.14, 0.09, 0.08])
    upper = estimates + np.array([0.12, 0.10, 0.14, 0.09, 0.08])

    predictors = rng.normal(size=(180, 3))
    response = 2.0 + predictors @ np.array([1.4, -0.9, 0.6]) + rng.normal(0, 0.8, 180)
    response[[12, 107]] += np.array([4.7, -4.2])
    design = sm.add_constant(predictors)
    fit = sm.OLS(response, design).fit()
    influence = fit.get_influence()
    fitted = fit.fittedvalues
    residuals = fit.resid
    leverage = influence.hat_matrix_diag
    standardized = influence.resid_studentized_internal
    cooks = influence.cooks_distance[0]

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5))
    positions = np.arange(len(labels))
    axes[0].errorbar(
        estimates,
        positions,
        xerr=np.vstack((estimates - lower, upper - estimates)),
        fmt="o",
        color="tab:blue",
        capsize=3,
    )
    axes[0].axvline(0, color="0.25", linewidth=1)
    axes[0].set(title="Forest plot of estimated effects", xlabel="Effect estimate (95% interval)", yticks=positions, yticklabels=labels)

    point_sizes = 24 + 800 * cooks / max(cooks.max(), 1e-9)
    axes[1].scatter(fitted, residuals, s=point_sizes, alpha=0.55, color="tab:purple")
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].set(title="Residual vs fitted", xlabel="Fitted value", ylabel="Residual")

    axes[2].scatter(leverage, standardized, s=point_sizes, alpha=0.55, color="tab:orange")
    axes[2].axhline(0, color="0.25", linewidth=1)
    axes[2].axvline(2 * design.shape[1] / design.shape[0], color="tab:red", linestyle="--", label="Leverage guide")
    axes[2].set(title="Leverage and influence", xlabel="Leverage", ylabel="Studentized residual")
    axes[2].legend(loc="upper right", fontsize=8)
    figure.suptitle("Statistical inference and regression assumption checks")
    return figure


def route_comparison_figure() -> plt.Figure:
    """Compare a baseline route against a shorter order through the same stops."""

    locations = np.array(
        [
            [0.5, 0.6],
            [1.8, 4.9],
            [4.7, 5.4],
            [7.4, 4.4],
            [8.5, 1.7],
            [5.9, 0.8],
            [3.0, 1.6],
        ]
    )
    labels = ["Depot", "A", "B", "C", "D", "E", "F"]
    baseline = np.array([0, 1, 4, 2, 5, 3, 6, 0])
    optimized = np.array([0, 1, 2, 3, 4, 5, 6, 0])

    def distance(order: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(locations[order], axis=0), axis=1).sum())

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, order, color, title in [
        (axes[0], baseline, "tab:red", "Baseline route"),
        (axes[1], optimized, "tab:blue", "Optimized route"),
    ]:
        axis.plot(locations[order, 0], locations[order, 1], "o-", color=color, linewidth=2, markersize=6)
        axis.scatter(locations[0, 0], locations[0, 1], marker="s", s=90, color="tab:green", zorder=3, label="Depot")
        for index, label in enumerate(labels):
            axis.annotate(label, locations[index] + np.array([0.10, 0.12]))
        axis.set(title=f"{title}: {distance(order):.1f} distance units", xlabel="East coordinate", ylabel="North coordinate", xlim=(0, 9.4), ylim=(0, 6.2), aspect="equal")
        axis.legend(loc="upper right")
    figure.suptitle("Route-order comparison for a fixed set of demand nodes")
    return figure


def interactive_pareto_html(output_dir: Path) -> Path:
    import plotly.graph_objects as go

    front_x = np.linspace(0.9, 5.6, 24)
    front_y = 6.6 / (front_x + 0.5) + 0.35
    figure = go.Figure()
    figure.add_scatter(
        x=front_x,
        y=front_y,
        mode="lines+markers",
        name="Pareto front",
        marker={"size": 9},
        hovertemplate="Cost: %{x:.2f}<br>Risk: %{y:.2f}<extra></extra>",
    )
    figure.update_layout(
        title="Interactive Pareto front",
        xaxis_title="Cost",
        yaxis_title="Risk",
        template="plotly_white",
        width=860,
        height=560,
    )
    path = output_dir / "11_interactive_pareto.html"
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def generate_gallery(output_dir: Path) -> list[Artifact]:
    """Generate static and interactive artifacts for major catalog categories."""

    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    rng = np.random.default_rng(20260906)
    tasks: list[tuple[Artifact, Callable[[], plt.Figure]]] = [
        (Artifact("01_distribution_ecdf_boxplot.png", "dist-ecdf", "Histogram/KDE, ECDF, boxplot, and raw observations."), lambda: distribution_figure(rng)),
        (Artifact("02_relationship_scatter_hexbin.png", "rel-hexbin-density", "Scatter and hexbin density diagnostics."), lambda: relationship_figure(rng)),
        (Artifact("03_correlation_heatmap.png", "rel-correlation-heatmap", "Annotated multivariable correlation matrix."), lambda: correlation_figure(rng)),
        (Artifact("04_time_series_forecast_acf.png", "ts-acf-pacf", "Forecast interval and autocorrelation diagnostic."), lambda: time_series_figure(rng)),
        (Artifact("05_optimization_convergence_pareto.png", "opt-pareto", "Repeated-run convergence band and Pareto front."), lambda: optimization_figure(rng)),
        (Artifact("06_parameter_sensitivity_heatmap.png", "opt-sensitivity", "Parameter-by-scenario sensitivity matrix."), lambda: sensitivity_figure(rng)),
        (Artifact("07_network_communities.png", "net-node-link", "Community network with degree-scaled nodes."), lambda: network_figure(rng)),
        (Artifact("08_contour_vector_field.png", "math-vector-field", "Potential contour and negative-gradient field."), field_figure),
        (Artifact("09_schedule_gantt_load.png", "sched-gantt", "Gantt schedule and resource capacity load."), scheduling_figure),
        (Artifact("10_tornado_sensitivity.png", "decision-tornado", "One-way decision sensitivity chart."), tornado_figure),
        (Artifact("12_data_quality_missingness_outliers.png", "quality-missingness", "Structured missingness matrix and robust outlier screening."), lambda: data_quality_figure(rng)),
        (Artifact("13_time_series_decomposition_control.png", "ts-decomposition", "STL trend/seasonal/residual decomposition and control limits."), lambda: decomposition_control_figure(rng)),
        (Artifact("14_network_adjacency_degree.png", "net-adjacency", "Community-ordered adjacency matrix and degree distribution."), network_adjacency_figure),
        (Artifact("15_dynamics_phase_spacetime.png", "dyn-phase-plane", "Phase-plane trajectories with a spatiotemporal propagation field."), dynamics_figure),
        (Artifact("16_optimization_feasible_allocation.png", "opt-feasible-region", "Constraint-feasible region with objective contours and allocation matrix."), feasible_allocation_figure),
        (Artifact("17_queue_length_wait_utilization.png", "sched-queue", "Queue length, waiting-time proxy, utilization, and service targets."), lambda: queue_metrics_figure(rng)),
        (Artifact("18_scenario_fan_ranking_stability.png", "decision-scenario-fan", "Monte Carlo scenario fan chart and ranking stability under weight variation."), lambda: scenario_fan_ranking_figure(rng)),
        (Artifact("19_machine_learning_diagnostics.png", "ml-prediction-residual", "Regression and classification diagnostics: residuals, confusion, ROC/PR, and calibration."), machine_learning_diagnostics_figure),
        (Artifact("20_statistical_inference_forest_influence.png", "stats-forest", "Effect-size intervals with residual, leverage, and influence diagnostics."), lambda: statistical_inference_figure(rng)),
        (Artifact("21_route_baseline_optimized.png", "opt-route", "Baseline versus optimized route order and total travel distance."), route_comparison_figure),
    ]

    artifacts: list[Artifact] = []
    for artifact, factory in tasks:
        DEFAULT_REGISTRY.get(artifact.spec_id)
        _save(factory(), output_dir, artifact.filename)
        artifacts.append(artifact)

    interactive = Artifact("11_interactive_pareto.html", "opt-pareto", "Self-contained Plotly interactive Pareto explorer.")
    DEFAULT_REGISTRY.get(interactive.spec_id)
    interactive_pareto_html(output_dir)
    artifacts.append(interactive)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic mathematical-modeling chart gallery.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated visual artifacts.")
    args = parser.parse_args()

    artifacts = generate_gallery(args.output)
    manifest = []
    for artifact in artifacts:
        spec = DEFAULT_REGISTRY.get(artifact.spec_id)
        path = args.output / artifact.filename
        manifest.append(
            {
                "filename": artifact.filename,
                "spec_id": artifact.spec_id,
                "category": spec.category,
                "name": spec.name,
                "description": artifact.description,
                "bytes": path.stat().st_size,
            }
        )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(artifacts)} artifacts in {args.output.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
