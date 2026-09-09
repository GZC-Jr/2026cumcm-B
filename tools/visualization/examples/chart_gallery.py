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

from mathmodel_viz import DEFAULT_REGISTRY, configure_matplotlib, get_theme, save_figure


def _theme_hex(name: str) -> str:
    """Resolve a theme color at render time so global updates take effect."""

    theme = get_theme()
    names = {
        "red": "primary_red",
        "blue": "primary_blue",
        "transition": "transition",
        "auxiliary": "auxiliary",
        "foreground": "foreground",
    }
    try:
        return getattr(theme, names[name])
    except KeyError as exc:
        valid = ", ".join(sorted(names))
        raise KeyError(f"Unknown theme color {name!r}. Choose one of: {valid}.") from exc


def _theme_rgba(name: str, alpha: float) -> tuple[float, float, float, float]:
    return get_theme().color(name, alpha)


def _theme_cmap(name: str = "diverging"):
    return get_theme().colormap(name)


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
            "group": ["组A"] * len(group_a) + ["组B"] * len(group_b),
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    palette = [_theme_hex("blue"), _theme_hex("red")]
    sns.histplot(data=data, x="value", hue="group", kde=True, stat="density", common_norm=False, palette=palette, ax=axes[0])
    axes[0].set(title="直方图与 KDE", xlabel="取值", ylabel="密度")
    sns.ecdfplot(data=data, x="value", hue="group", palette=palette, ax=axes[1])
    axes[1].set(title="经验 ECDF", xlabel="取值", ylabel="累计比例")
    for axis in axes[:2]:
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("分组")
    sns.boxplot(data=data, x="group", y="value", hue="group", palette=palette, legend=False, ax=axes[2])
    sns.stripplot(data=data.sample(160, random_state=20260906), x="group", y="value", color=_theme_rgba("foreground", 0.55), alpha=0.35, ax=axes[2])
    axes[2].set(title="箱线图与观测点", xlabel="", ylabel="取值")
    figure.suptitle("分布与分组比较诊断")
    return figure


def relationship_figure(rng: np.random.Generator) -> plt.Figure:
    x = rng.normal(0, 1.2, 1_400)
    y = 0.8 * x + 0.45 * x**2 + rng.normal(0, 0.85, x.size)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(x, y, s=12, alpha=0.28, color=_theme_rgba("blue", 0.75))
    axes[0].set(title="散点图", xlabel="输入 x", ylabel="响应 y")
    hexbin = axes[1].hexbin(x, y, gridsize=28, mincnt=1, cmap=_theme_cmap("blue"))
    figure.colorbar(hexbin, ax=axes[1], label="每个网格的观测数")
    axes[1].set(title="六边形密度图", xlabel="输入 x", ylabel="响应 y")
    figure.suptitle("稠密配对数据的关系诊断")
    return figure


def correlation_figure(rng: np.random.Generator) -> plt.Figure:
    base = rng.normal(size=(420, 3))
    frame = pd.DataFrame(
        {
            "需求": base[:, 0],
            "价格": -0.65 * base[:, 0] + 0.55 * base[:, 1],
            "库存": 0.40 * base[:, 0] + 0.30 * base[:, 2],
            "距离": base[:, 1] - 0.25 * base[:, 2],
            "延迟": 0.45 * base[:, 1] + 0.35 * base[:, 2],
            "质量": -0.35 * base[:, 0] + 0.50 * base[:, 2],
        }
    )
    figure, axis = plt.subplots(figsize=(7, 5.6))
    sns.heatmap(frame.corr(), annot=True, fmt=".2f", center=0, cmap=_theme_cmap(), square=True, ax=axis)
    axis.set_title("相关矩阵")
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
    axes[0].plot(dates, observed, label="观测值", color=_theme_hex("blue"))
    axes[0].plot(forecast_index, trend, label="预测均值", color=_theme_hex("transition"))
    axes[0].fill_between(forecast_index, trend - 1.96 * uncertainty, trend + 1.96 * uncertainty, color=_theme_rgba("transition", 0.22), label="95% 预测区间")
    axes[0].axvline(forecast_index[0], color=_theme_rgba("foreground", 0.7), linestyle="--", linewidth=1)
    axes[0].set(title="时序与预测区间", xlabel="周次", ylabel="需求量")
    axes[0].legend(loc="upper left")
    plot_acf(observed, lags=24, zero=False, ax=axes[1])
    axes[1].set(title="自相关诊断", xlabel="滞后（周）", ylabel="自相关系数")
    figure.suptitle("预测与序列相关性检验")
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
    axes[0].plot(iterations, median, color=_theme_hex("blue"), label="目标值中位数")
    axes[0].fill_between(iterations, low, high, color=_theme_rgba("blue", 0.2), label="多次运行的 10%--90% 分位带")
    axes[0].set(title="优化收敛曲线", xlabel="迭代次数", ylabel="目标值")
    axes[0].legend()
    axes[1].scatter(dominated_x, dominated_y, s=26, alpha=0.55, color=_theme_rgba("foreground", 0.45), label="受支配方案")
    axes[1].plot(front_x, front_y, "o-", color=_theme_hex("red"), label="Pareto 前沿")
    axes[1].set(title="双目标 Pareto 前沿", xlabel="成本", ylabel="风险")
    axes[1].legend()
    figure.suptitle("优化性能与目标权衡")
    return figure


def sensitivity_figure(rng: np.random.Generator) -> plt.Figure:
    parameters = ["需求量", "价格", "容量", "运输时间", "惩罚项", "服务水平"]
    scenarios = ["-20%", "-10%", "基准", "+10%", "+20%"]
    base = rng.normal(0, 1, (len(parameters), len(scenarios)))
    direction = np.array([[-1], [-0.6], [0.8], [0.4], [0.6], [0.3]])
    change = np.array([-1.6, -0.8, 0, 0.8, 1.6])
    effects = base * 0.15 + direction * change
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    sns.heatmap(effects, cmap=_theme_cmap(), center=0, annot=True, fmt=".2f", xticklabels=scenarios, yticklabels=parameters, cbar_kws={"label": "相对目标值变化"}, ax=axis)
    axis.set(title="参数敏感性热力图", xlabel="情景", ylabel="参数")
    return figure


def network_figure(rng: np.random.Generator) -> plt.Figure:
    graph = nx.planted_partition_graph(l=3, k=8, p_in=0.55, p_out=0.045, seed=20260906)
    positions = nx.spring_layout(graph, seed=20260906, k=0.45)
    groups = [node // 8 for node in graph.nodes]
    degrees = np.array([graph.degree(node) for node in graph.nodes])
    from matplotlib.colors import ListedColormap

    community_cmap = ListedColormap(list(get_theme().series_colors[:3]))
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    nx.draw_networkx_edges(graph, positions, edge_color=_theme_rgba("transition", 0.3), width=0.8, ax=axis)
    nodes = nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=groups,
        node_size=90 + 35 * degrees,
        cmap=community_cmap,
        vmin=-0.5,
        vmax=2.5,
        edgecolors=_theme_rgba("auxiliary", 0.9),
        linewidths=0.8,
        ax=axis,
    )
    figure.colorbar(nodes, ax=axis, ticks=[0, 1, 2], label="社团")
    axis.set_title("网络社团与度数缩放节点")
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
    contours = axis.contourf(xx, yy, potential, levels=22, cmap=_theme_cmap("accent"))
    axis.quiver(
        xx[sample],
        yy[sample],
        direction_x / safe_magnitude,
        direction_y / safe_magnitude,
        color=_theme_hex("auxiliary"),
        alpha=0.78,
        angles="xy",
        scale_units="xy",
        scale=4.0,
        width=0.003,
    )
    figure.colorbar(contours, ax=axis, label="势能值")
    axis.set(title="等高线与负梯度向量场", xlabel="状态 x", ylabel="状态 y", aspect="equal")
    return figure


def scheduling_figure() -> plt.Figure:
    theme = get_theme()
    tasks = [
        ("准备", 0, 3, theme.primary_blue),
        ("处理 A", 3, 5, theme.transition),
        ("处理 B", 4, 4, theme.primary_red),
        ("检查", 8, 2, theme.auxiliary),
        ("交付", 10, 3, theme.series_colors[3]),
    ]
    timeline = np.arange(0, 14)
    capacity = np.full_like(timeline, 2, dtype=float)
    load = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.3), sharex=True, height_ratios=(2, 1))
    for row, (name, start, duration, color) in enumerate(tasks):
        axes[0].barh(row, duration, left=start, color=color, height=0.62)
        axes[0].text(start + duration / 2, row, name, ha="center", va="center", color=theme.foreground, fontsize=9)
    axes[0].set(yticks=range(len(tasks)), yticklabels=[task[0] for task in tasks], title="甘特图", ylabel="任务")
    axes[0].invert_yaxis()
    axes[1].step(timeline, load, where="post", label="资源负荷", color=theme.primary_blue)
    axes[1].step(timeline, capacity, where="post", label="容量上限", color=theme.primary_red, linestyle="--")
    axes[1].set(title="资源负荷", xlabel="时间段", ylabel="单位数", ylim=(0, 2.5))
    axes[1].legend(loc="upper right")
    figure.suptitle("调度计划与容量校验")
    return figure


def tornado_figure() -> plt.Figure:
    factors = ["需求量", "单位成本", "惩罚项", "容量", "运输时间", "服务目标"]
    low = np.array([-19, -13, -10, -7, -5, -3])
    high = np.array([24, 16, 11, 8, 6, 4])
    y = np.arange(len(factors))
    figure, axis = plt.subplots(figsize=(8.5, 5))
    axis.barh(y, low, color=_theme_rgba("blue", 0.82), alpha=0.82, label="低位假设")
    axis.barh(y, high, color=_theme_rgba("red", 0.82), alpha=0.82, label="高位假设")
    axis.axvline(0, color=_theme_rgba("foreground", 0.8), linewidth=1)
    axis.set(yticks=y, yticklabels=factors, title="龙卷风敏感性图", xlabel="相对基准目标值变化（%）")
    axis.legend(loc="lower right")
    return figure


def data_quality_figure(rng: np.random.Generator) -> plt.Figure:
    """Show structured missingness alongside a robust outlier diagnostic."""

    columns = ["需求量", "价格", "距离", "提前期", "质量"]
    missing = rng.random((100, len(columns))) < np.array([0.03, 0.08, 0.12, 0.05, 0.02])
    missing[35:55, 2] = True
    missing[68:80, 1] = True
    observations = np.concatenate([rng.normal(52, 6.5, 220), np.array([76, 79, 83, 87])])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": (1.25, 1)})
    image = axes[0].imshow(missing.T, aspect="auto", interpolation="nearest", cmap=_theme_cmap("red"), vmin=0, vmax=1)
    axes[0].set(
        title="缺失数据模式",
        xlabel="观测序号",
        ylabel="变量",
        yticks=np.arange(len(columns)),
        yticklabels=columns,
    )
    figure.colorbar(image, ax=axes[0], ticks=[0, 1], label="缺失指示")

    jitter = rng.normal(0, 0.055, observations.size)
    axes[1].boxplot(observations, vert=False, widths=0.55, patch_artist=True, boxprops={"facecolor": _theme_rgba("blue", 0.4), "edgecolor": _theme_hex("blue")})
    axes[1].scatter(observations, np.ones_like(observations) + jitter, s=13, alpha=0.36, color=_theme_rgba("foreground", 0.7))
    axes[1].set(title="异常值诊断", xlabel="测量值", yticks=[1], yticklabels=["样本"])
    figure.suptitle("建模前的数据质量筛查")
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
    axes[0, 0].plot(dates, observed, color=_theme_hex("blue"), label="观测序列")
    axes[0, 0].plot(dates, decomposition.trend, color=_theme_hex("transition"), label="STL 趋势")
    axes[0, 0].set(title="观测序列与趋势", ylabel="需求量")
    axes[0, 0].legend(loc="upper left")

    axes[1, 0].plot(dates, decomposition.seasonal, color=_theme_hex("auxiliary"), label="季节项")
    axes[1, 0].plot(dates, decomposition.resid, color=_theme_hex("red"), alpha=0.8, label="残差")
    axes[1, 0].axhline(0, color=_theme_rgba("foreground", 0.75), linewidth=0.8)
    axes[1, 0].set(title="STL 季节项与残差", xlabel="周次", ylabel="分量")
    axes[1, 0].legend(loc="upper left")

    axes[0, 1].plot(dates, observed, "o-", markersize=3, color=_theme_hex("blue"), label="过程值")
    axes[0, 1].axhline(center, color=_theme_rgba("foreground", 0.8), linewidth=1, label="中心线")
    axes[0, 1].axhline(center + 3 * sigma, color=_theme_hex("red"), linestyle="--", label="上控制限")
    axes[0, 1].axhline(center - 3 * sigma, color=_theme_hex("red"), linestyle="--", label="下控制限")
    axes[0, 1].set(title="单值控制图", ylabel="需求量")
    axes[0, 1].legend(loc="upper left", fontsize=8)

    axes[1, 1].hist(decomposition.resid, bins=18, density=True, color=_theme_rgba("red", 0.7), alpha=0.7)
    axes[1, 1].axvline(0, color=_theme_rgba("foreground", 0.8), linewidth=1)
    axes[1, 1].set(title="残差分布", xlabel="残差", ylabel="密度")
    figure.suptitle("时序分解与过程稳定性检验")
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
    image = axes[0].imshow(adjacency, cmap=_theme_cmap("blue"), vmin=0, vmax=1, interpolation="nearest")
    for boundary in boundaries:
        axes[0].axhline(boundary, color=_theme_hex("red"), linewidth=1)
        axes[0].axvline(boundary, color=_theme_hex("red"), linewidth=1)
    axes[0].set(title="按社团排序的邻接矩阵", xlabel="目标节点", ylabel="源节点")
    figure.colorbar(image, ax=axes[0], ticks=[0, 1], label="是否存在边")

    bins = np.arange(degrees.min() - 0.5, degrees.max() + 1.5)
    axes[1].hist(degrees, bins=bins, color=_theme_hex("transition"), rwidth=0.82)
    axes[1].set(title="度数分布", xlabel="度数", ylabel="节点数", xticks=np.arange(degrees.min(), degrees.max() + 1))
    figure.suptitle("稠密网络结构诊断")
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

    theme = get_theme()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].streamplot(xx, yy, dx, dy, color=speed, cmap=_theme_cmap("accent"), density=1.1, linewidth=1)
    for initial, color in [((2.0, 1.0), theme.primary_blue), ((5.0, 3.5), theme.transition), ((7.0, 1.7), theme.auxiliary)]:
        solution = solve_ivp(dynamics, (solution_times[0], solution_times[-1]), initial, t_eval=solution_times)
        axes[0].plot(solution.y[0], solution.y[1], color=color, label=f"初始状态 {initial}")
        axes[0].plot(initial[0], initial[1], "o", color=color)
    axes[0].plot(gamma / delta, alpha / beta, "*", color=theme.primary_red, markersize=12, label="平衡点")
    axes[0].set(title="相平面与状态轨迹", xlabel="猎物数量", ylabel="捕食者数量", xlim=(0, 8), ylim=(0, 6.5))
    axes[0].legend(loc="upper right", fontsize=8)

    image = axes[1].imshow(field, aspect="auto", origin="lower", extent=(space.min(), space.max(), time.min(), time.max()), cmap=_theme_cmap("diverging"))
    axes[1].set(title="时空传播场", xlabel="空间坐标", ylabel="仿真时间")
    figure.colorbar(image, ax=axes[1], label="状态强度")
    figure.suptitle("动力学与仿真诊断")
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
    axes[0].contourf(xx, yy, feasible.astype(int), levels=[-0.5, 0.5, 1.5], colors=[_theme_rgba("blue", 0.03), _theme_rgba("transition", 0.25)], alpha=0.9)
    contours = axes[0].contour(xx, yy, np.where(feasible, objective, np.nan), levels=7, cmap=_theme_cmap("blue"))
    axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    axes[0].plot(x, 12 - 3 * x, color=_theme_hex("red"), label=r"$3x + y = 12$")
    axes[0].plot(x, (10 - x) / 2, color=_theme_hex("blue"), label=r"$x + 2y = 10$")
    axes[0].plot(*optimal, "*", color=_theme_hex("transition"), markersize=14, label="最优顶点")
    axes[0].set(title="可行域与目标等高线", xlabel="决策 x", ylabel="决策 y", xlim=(0, 5.2), ylim=(0, 6.2))
    axes[0].legend(loc="upper right", fontsize=8)

    sns.heatmap(
        allocation,
        annot=True,
        fmt="d",
        cmap=_theme_cmap("accent"),
        cbar_kws={"label": "分配单位数"},
        xticklabels=["北", "东", "南", "西"],
        yticklabels=["资源 A", "资源 B", "资源 C", "资源 D"],
        ax=axes[1],
    )
    axes[1].set(title="资源到区域的分配", xlabel="区域", ylabel="资源")
    figure.suptitle("优化可行性与分配校验")
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
    axes[0].step(intervals, queue, where="mid", color=_theme_hex("blue"), label="队列长度")
    axes[0].fill_between(intervals, queue, step="mid", color=_theme_rgba("blue", 0.2))
    axes[0].set(title="队列长度", ylabel="顾客数")
    axes[0].legend(loc="upper left")

    axes[1].plot(intervals, wait_estimate, color=_theme_hex("transition"), label="估计等待时间")
    axes[1].axhline(3, color=_theme_hex("red"), linestyle="--", label="服务目标")
    axes[1].set(title="等待时间代理指标", ylabel="时间段数")
    axes[1].legend(loc="upper left")

    axes[2].step(intervals, utilization, where="mid", color=_theme_hex("auxiliary"), label="利用率")
    axes[2].axhline(0.85, color=_theme_hex("red"), linestyle="--", label="建议上限")
    axes[2].set(title="服务台利用率", xlabel="15 分钟时间段", ylabel="占比", ylim=(0, 1.12))
    axes[2].legend(loc="upper left")
    figure.suptitle("容量受限期间的排队系统表现")
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
    axes[0].fill_between(periods, q05, q95, color=_theme_rgba("blue", 0.16), label="5%--95% 分位带")
    axes[0].fill_between(periods, q25, q75, color=_theme_rgba("blue", 0.32), label="25%--75% 分位带")
    axes[0].plot(periods, q50, color=_theme_hex("blue"), linewidth=2, label="中位数")
    axes[0].set(title="情景扇形图", xlabel="预测期", ylabel="指数化结果")
    axes[0].legend(loc="upper left")

    sns.heatmap(
        rank_probability,
        annot=True,
        fmt=".0%",
        cmap=_theme_cmap("blue"),
        cbar_kws={"label": "概率"},
        xticklabels=["排名 1", "排名 2", "排名 3", "排名 4"],
        yticklabels=["方案 A", "方案 B", "方案 C", "方案 D"],
        ax=axes[1],
    )
    axes[1].set(title="权重变化下的排名稳定性", xlabel="结果排名", ylabel="方案")
    figure.suptitle("风险情景与多指标决策稳健性")
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

    theme = get_theme()
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    low, high = np.percentile(np.concatenate([test_y_reg, predicted]), [1, 99])
    axes[0, 0].scatter(test_y_reg, predicted, color=_theme_rgba("blue", 0.55), s=20)
    axes[0, 0].plot([low, high], [low, high], color=_theme_hex("red"), linestyle="--", label="理想线")
    axes[0, 0].set(title="预测值与观测值", xlabel="观测响应", ylabel="预测响应")
    axes[0, 0].legend(loc="upper left")

    axes[0, 1].scatter(predicted, residuals, color=_theme_rgba("transition", 0.55), s=20)
    axes[0, 1].axhline(0, color=_theme_rgba("foreground", 0.75), linewidth=1)
    axes[0, 1].set(title="回归残差", xlabel="预测响应", ylabel="残差")

    axes[0, 2].hist(residuals, bins=18, color=_theme_rgba("transition", 0.72))
    axes[0, 2].axvline(0, color=_theme_rgba("foreground", 0.75), linewidth=1)
    axes[0, 2].set(title="残差分布", xlabel="残差", ylabel="频数")

    coefficients = np.abs(regression.coef_)
    axes[0, 3].barh(np.arange(coefficients.size), coefficients, color=_theme_hex("red"))
    axes[0, 3].set(
        title="线性特征权重",
        xlabel="系数绝对值",
        ylabel="特征",
        yticks=np.arange(coefficients.size),
        yticklabels=[f"特征 {i + 1}" for i in range(coefficients.size)],
    )

    image = axes[1, 0].imshow(matrix, cmap=_theme_cmap("blue"))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axes[1, 0].text(column, row, str(matrix[row, column]), ha="center", va="center", color=theme.foreground)
    axes[1, 0].set(title="混淆矩阵", xlabel="预测类别", ylabel="真实类别", xticks=[0, 1], yticks=[0, 1])
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04, label="样本数")

    axes[1, 1].plot(fpr, tpr, color=_theme_hex("blue"), label=f"AUC = {auc(fpr, tpr):.2f}")
    axes[1, 1].plot([0, 1], [0, 1], color=_theme_rgba("foreground", 0.65), linestyle="--", label="随机基线")
    axes[1, 1].set(title="ROC 曲线", xlabel="假阳性率", ylabel="真阳性率", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].legend(loc="lower right")

    axes[1, 2].plot(recall, precision, color=_theme_hex("red"), label=f"面积 = {auc(recall, precision):.2f}")
    axes[1, 2].set(title="PR 曲线", xlabel="召回率", ylabel="精确率", xlim=(0, 1), ylim=(0, 1))
    axes[1, 2].legend(loc="lower left")

    axes[1, 3].plot(calibration_predicted, calibration_true, "o-", color=_theme_hex("transition"), label="分类器")
    axes[1, 3].plot([0, 1], [0, 1], color=_theme_rgba("foreground", 0.65), linestyle="--", label="完美校准")
    axes[1, 3].set(title="概率校准", xlabel="平均预测概率", ylabel="观测频率", xlim=(0, 1), ylim=(0, 1))
    axes[1, 3].legend(loc="upper left", fontsize=8)
    figure.suptitle("回归与分类诊断")
    return figure


def statistical_inference_figure(rng: np.random.Generator) -> plt.Figure:
    """Combine effect-size intervals with residual and influence diagnostics."""

    labels = ["需求量", "价格", "容量", "运输时间", "策略"]
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
        color=_theme_hex("blue"),
        capsize=3,
    )
    axes[0].axvline(0, color=_theme_rgba("foreground", 0.75), linewidth=1)
    axes[0].set(title="效应估计森林图", xlabel="效应估计（95% 区间）", yticks=positions, yticklabels=labels)

    point_sizes = 24 + 800 * cooks / max(cooks.max(), 1e-9)
    axes[1].scatter(fitted, residuals, s=point_sizes, color=_theme_rgba("transition", 0.55))
    axes[1].axhline(0, color=_theme_rgba("foreground", 0.75), linewidth=1)
    axes[1].set(title="残差与拟合值", xlabel="拟合值", ylabel="残差")

    axes[2].scatter(leverage, standardized, s=point_sizes, color=_theme_rgba("auxiliary", 0.72))
    axes[2].axhline(0, color=_theme_rgba("foreground", 0.75), linewidth=1)
    axes[2].axvline(2 * design.shape[1] / design.shape[0], color=_theme_hex("red"), linestyle="--", label="杠杆参考线")
    axes[2].set(title="杠杆值与影响点", xlabel="杠杆值", ylabel="学生化残差")
    axes[2].legend(loc="upper right", fontsize=8)
    figure.suptitle("统计推断与回归假设检验")
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
    labels = ["仓库", "A", "B", "C", "D", "E", "F"]
    baseline = np.array([0, 1, 4, 2, 5, 3, 6, 0])
    optimized = np.array([0, 1, 2, 3, 4, 5, 6, 0])

    def distance(order: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(locations[order], axis=0), axis=1).sum())

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, order, color, title in [
        (axes[0], baseline, _theme_hex("red"), "基准路线"),
        (axes[1], optimized, _theme_hex("blue"), "优化路线"),
    ]:
        axis.plot(locations[order, 0], locations[order, 1], "o-", color=color, linewidth=2, markersize=6)
        axis.scatter(locations[0, 0], locations[0, 1], marker="s", s=90, color=_theme_hex("auxiliary"), zorder=3, label="仓库")
        for index, label in enumerate(labels):
            axis.annotate(label, locations[index] + np.array([0.10, 0.12]))
        axis.set(title=f"{title}：总距离 {distance(order):.1f} 个距离单位", xlabel="东西坐标", ylabel="南北坐标", xlim=(0, 9.4), ylim=(0, 6.2), aspect="equal")
        axis.legend(loc="upper right")
    figure.suptitle("固定需求点的路线顺序对比")
    return figure


def interactive_pareto_html(output_dir: Path) -> Path:
    import plotly.graph_objects as go
    from mathmodel_viz import configure_plotly

    front_x = np.linspace(0.9, 5.6, 24)
    front_y = 6.6 / (front_x + 0.5) + 0.35
    theme = get_theme()
    configure_plotly(theme=theme)
    figure = go.Figure()
    figure.add_scatter(
        x=front_x,
        y=front_y,
        mode="lines+markers",
        name="Pareto 前沿",
        line={"color": theme.primary_red, "width": 2},
        marker={"size": 9, "color": theme.primary_red},
        hovertemplate="成本：%{x:.2f}<br>风险：%{y:.2f}<extra></extra>",
    )
    figure.update_layout(
        title="交互式 Pareto 前沿",
        xaxis_title="成本",
        yaxis_title="风险",
        template="mathmodel",
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
        (Artifact("01_distribution_ecdf_boxplot.png", "dist-ecdf", "直方图、KDE、ECDF、箱线图与原始观测点。"), lambda: distribution_figure(rng)),
        (Artifact("02_relationship_scatter_hexbin.png", "rel-hexbin-density", "散点图与六边形密度诊断。"), lambda: relationship_figure(rng)),
        (Artifact("03_correlation_heatmap.png", "rel-correlation-heatmap", "带数值标注的多变量相关矩阵。"), lambda: correlation_figure(rng)),
        (Artifact("04_time_series_forecast_acf.png", "ts-acf-pacf", "预测区间与自相关诊断。"), lambda: time_series_figure(rng)),
        (Artifact("05_optimization_convergence_pareto.png", "opt-pareto", "多次运行的收敛分位带与 Pareto 前沿。"), lambda: optimization_figure(rng)),
        (Artifact("06_parameter_sensitivity_heatmap.png", "opt-sensitivity", "参数与情景对应的敏感性矩阵。"), lambda: sensitivity_figure(rng)),
        (Artifact("07_network_communities.png", "net-node-link", "按节点度数缩放的社团网络。"), lambda: network_figure(rng)),
        (Artifact("08_contour_vector_field.png", "math-vector-field", "势能等高线与负梯度向量场。"), field_figure),
        (Artifact("09_schedule_gantt_load.png", "sched-gantt", "甘特计划与资源容量负荷。"), scheduling_figure),
        (Artifact("10_tornado_sensitivity.png", "decision-tornado", "单因素决策敏感性图。"), tornado_figure),
        (Artifact("12_data_quality_missingness_outliers.png", "quality-missingness", "结构性缺失矩阵与稳健异常值筛查。"), lambda: data_quality_figure(rng)),
        (Artifact("13_time_series_decomposition_control.png", "ts-decomposition", "STL 趋势、季节项、残差分解与控制界限。"), lambda: decomposition_control_figure(rng)),
        (Artifact("14_network_adjacency_degree.png", "net-adjacency", "按社团排序的邻接矩阵与度数分布。"), network_adjacency_figure),
        (Artifact("15_dynamics_phase_spacetime.png", "dyn-phase-plane", "相平面状态轨迹与时空传播场。"), dynamics_figure),
        (Artifact("16_optimization_feasible_allocation.png", "opt-feasible-region", "约束可行域、目标等高线与资源分配矩阵。"), feasible_allocation_figure),
        (Artifact("17_queue_length_wait_utilization.png", "sched-queue", "队列长度、等待时间代理指标、利用率与服务目标。"), lambda: queue_metrics_figure(rng)),
        (Artifact("18_scenario_fan_ranking_stability.png", "decision-scenario-fan", "蒙特卡洛情景扇形图与权重变化下的排名稳定性。"), lambda: scenario_fan_ranking_figure(rng)),
        (Artifact("19_machine_learning_diagnostics.png", "ml-prediction-residual", "回归与分类诊断：残差、混淆矩阵、ROC/PR 与校准。"), machine_learning_diagnostics_figure),
        (Artifact("20_statistical_inference_forest_influence.png", "stats-forest", "效应区间以及残差、杠杆值和影响点诊断。"), lambda: statistical_inference_figure(rng)),
        (Artifact("21_route_baseline_optimized.png", "opt-route", "基准与优化路线顺序及总行驶距离。"), route_comparison_figure),
    ]

    artifacts: list[Artifact] = []
    for artifact, factory in tasks:
        DEFAULT_REGISTRY.get(artifact.spec_id)
        _save(factory(), output_dir, artifact.filename)
        artifacts.append(artifact)

    interactive = Artifact("11_interactive_pareto.html", "opt-pareto", "自包含的 Plotly 交互式 Pareto 探索图。")
    DEFAULT_REGISTRY.get(interactive.spec_id)
    interactive_pareto_html(output_dir)
    artifacts.append(interactive)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="生成确定性的数学建模可视化图库。")
    parser.add_argument("--output", type=Path, required=True, help="可视化产物输出目录。")
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
    print(f"已在 {args.output.resolve()} 生成 {len(artifacts)} 个可视化产物")
    print(f"清单文件：{manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
