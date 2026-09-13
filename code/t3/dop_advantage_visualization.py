"""问题三：单位时间信息增益 + 加权 DOP 近优决胜的优势可视化。

图中的候选点与后验区域是用于解释算法机制的确定性示例，
所有指标均按论文中的公式现场计算，不代表模拟器实测结果。

运行：python3 dop_advantage_visualization.py
输出：outputs/t3/dop_advantage_visualization.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from matplotlib.ticker import FuncFormatter

# 导入全局主题配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "visualization" / "src"))
from mathmodel_viz.styles import (
    configure_matplotlib,
    get_theme,
    rgba,
    rgba_css,
)

# 应用全局主题
configure_matplotlib()
THEME = get_theme()

# 主题颜色别名（便于使用）
_PRIMARY_BLUE = THEME.primary_blue
_PRIMARY_RED = THEME.primary_red
_TRANSITION = THEME.transition
_FOREGROUND = THEME.foreground


DOMAIN_RADIUS = 1800.0
SPEED = 5.0
SERVICE_TIME = 6.0
SIGMA_THETA = np.deg2rad(1.0)
TAU = 0.12
EPSILON_W = 0.18
P_W = 2.0
OUTPUT = Path("../../outputs/t3/dop_advantage_visualization.png")


def make_target_samples(mean: np.ndarray, a: float, b: float):
    """在椭圆可行域内生成确定性求积点及区域深度。"""
    gx = np.linspace(mean[0] - a, mean[0] + a, 35)
    gy = np.linspace(mean[1] - b, mean[1] + b, 29)
    xx, yy = np.meshgrid(gx, gy)
    normalized_radius = np.sqrt(((xx - mean[0]) / a) ** 2 + ((yy - mean[1]) / b) ** 2)
    mask = normalized_radius <= 1.0
    samples = np.column_stack((xx[mask], yy[mask]))
    depth = 1.0 - normalized_radius[mask]
    weights = EPSILON_W + (1.0 - EPSILON_W) * depth**P_W
    return samples, depth, weights


def make_candidates(mean: np.ndarray, samples: np.ndarray, first_station: np.ndarray):
    """构造满足示意强接收约束的候选点。"""
    axis = np.linspace(-1050.0, 1350.0, 31)
    xx, yy = np.meshgrid(axis, axis)
    raw = np.column_stack((xx.ravel(), yy.ravel()))

    inside_domain = np.linalg.norm(raw, axis=1) <= DOMAIN_RADIUS
    away_from_target = np.linalg.norm(raw - mean, axis=1) >= 230.0
    raw = raw[inside_domain & away_from_target]

    # 用可行域求积点检验强接收约束：对每个可能目标 G 均可继续接收。
    retained = []
    first_dist = np.linalg.norm(samples - first_station, axis=1)
    receive_limit = np.maximum(1000.0, first_dist)
    for point in raw:
        second_dist = np.linalg.norm(samples - point, axis=1)
        if np.all(second_dist <= receive_limit):
            retained.append(point)
    return np.asarray(retained)


def evaluate_candidates(candidates, samples, weights, first_station, current, mean, covariance):
    """计算时间、信息效用、普通 DOP 与加权 DOP。"""
    travel_time = np.linalg.norm(candidates - current, axis=1) / SPEED + SERVICE_TIME

    delta = mean - candidates
    range_sq = np.sum(delta**2, axis=1)
    h = np.column_stack((-delta[:, 1] / range_sq, delta[:, 0] / range_sq))
    projected_var = np.einsum("ij,jk,ik->i", h, covariance, h)
    information = 0.5 * np.log1p(projected_var / SIGMA_THETA**2)
    utility = information / travel_time

    ordinary_dop = np.empty(len(candidates))
    weighted_dop = np.empty(len(candidates))
    for idx, point in enumerate(candidates):
        v1 = first_station - samples
        v2 = point - samples
        r1 = np.linalg.norm(v1, axis=1)
        r2 = np.linalg.norm(v2, axis=1)
        cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        sin_alpha = cross / np.maximum(r1 * r2, 1e-9)
        dop = np.sqrt(r1**2 + r2**2) / np.maximum(sin_alpha, 0.035)
        ordinary_dop[idx] = np.mean(dop)
        weighted_dop[idx] = np.average(dop, weights=weights)

    return travel_time, information, utility, ordinary_dop, weighted_dop


def normalize(values):
    lo, hi = np.min(values), np.max(values)
    return (values - lo) / (hi - lo + 1e-12)


def main() -> None:
    mean = np.array([300.0, 280.0])
    first_station = np.array([-620.0, 120.0])
    current = np.array([-1180.0, -850.0])
    ellipse_a, ellipse_b = 410.0, 255.0
    covariance = np.array([[210.0**2, 0.30 * 210.0 * 130.0],
                           [0.30 * 210.0 * 130.0, 130.0**2]])

    samples, depth, weights = make_target_samples(mean, ellipse_a, ellipse_b)
    candidates = make_candidates(mean, samples, first_station)
    travel_time, information, utility, ordinary_dop, weighted_dop = evaluate_candidates(
        candidates, samples, weights, first_station, current, mean, covariance
    )

    idx_u = int(np.argmax(utility))
    near_optimal = utility >= (1.0 - TAU) * utility[idx_u]
    near_indices = np.flatnonzero(near_optimal)
    idx_proposed = int(near_indices[np.argmin(weighted_dop[near_optimal])])
    idx_dop = int(np.argmin(weighted_dop))

    # 使用主题颜色：仅最小化DOP(红)、仅最大化信息(橙)、本文方法(蓝)
    selections = {
        "仅最小化 DOP": (idx_dop, _PRIMARY_RED, "X"),
        "仅最大化单位时间信息": (idx_u, _TRANSITION, "s"),
        "本文方法": (idx_proposed, _PRIMARY_BLUE, "*"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0), dpi=220,
                             gridspec_kw={"width_ratios": [1.18, 1.0]},
                             constrained_layout=True)
    fig.patch.set_facecolor(rgba(_PRIMARY_BLUE, 0.02))

    # -------------------- A. 候选点空间位置 --------------------
    ax = axes[0]
    ax.set_facecolor(rgba(_PRIMARY_BLUE, 0.04))
    ax.add_patch(Circle((0, 0), DOMAIN_RADIUS, facecolor=rgba(_PRIMARY_BLUE, 0.06),
                        edgecolor=_FOREGROUND, linewidth=2.0, zorder=0))
    ax.add_patch(Ellipse(mean, 2 * ellipse_a, 2 * ellipse_b,
                         facecolor=rgba(_TRANSITION, 0.35), edgecolor=_TRANSITION,
                         linewidth=1.7, zorder=2))
    ax.scatter(candidates[:, 0], candidates[:, 1], s=8, c=rgba(_PRIMARY_BLUE, 0.45),
               linewidths=0, zorder=1)
    ax.scatter(candidates[near_optimal, 0], candidates[near_optimal, 1],
               s=22, facecolors="none", edgecolors=_PRIMARY_BLUE,
               linewidths=0.9, alpha=0.9, zorder=3)

    ax.scatter(*first_station, s=90, marker="^", c=_PRIMARY_BLUE,
               edgecolors="white", linewidths=1.0, zorder=7)
    ax.scatter(*current, s=100, marker="D", c=_PRIMARY_RED,
               edgecolors="white", linewidths=1.0, zorder=7)
    ax.scatter(*mean, s=70, marker="o", c=_TRANSITION,
               edgecolors="white", linewidths=1.0, zorder=7)

    for label, (idx, color, marker) in selections.items():
        point = candidates[idx]
        ax.plot([current[0], point[0]], [current[1], point[1]],
                color=color, lw=1.55, alpha=0.75, zorder=4)
        ax.scatter(*point, s=130 if marker == "*" else 88, marker=marker,
                   c=color, edgecolors="white", linewidths=1.0, zorder=8)

    ax.set_title("候选点选择：兼顾时间与交会几何", fontsize=13, pad=12, color=_FOREGROUND)
    ax.set_xlim(-1900, 1900)
    ax.set_ylim(-1900, 1900)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m", fontsize=10)
    ax.set_ylabel("y / m", fontsize=10)
    ax.grid(color=rgba(_TRANSITION, 0.20), lw=0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    spatial_legend = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=_PRIMARY_BLUE,
               markeredgecolor="white", markersize=8, label="首次检测点"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=_PRIMARY_RED,
               markeredgecolor="white", markersize=8, label="当前位置"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_TRANSITION,
               markeredgecolor="white", markersize=8, label="后验均值"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=_PRIMARY_BLUE, markersize=7, label="效用近优集"),
    ]
    for label, (_, color, marker) in selections.items():
        spatial_legend.append(Line2D([0], [0], marker=marker, color="none",
                                     markerfacecolor=color, markeredgecolor="white",
                                     markersize=9, label=label))
    ax.legend(handles=spatial_legend, loc="lower center", bbox_to_anchor=(0.5, -0.27),
              ncol=2, frameon=False, fontsize=8.6)

    # -------------------- B. 时间—DOP 权衡 --------------------
    ax = axes[1]
    ax.set_facecolor(rgba(_PRIMARY_BLUE, 0.04))
    sizes = 18 + 70 * normalize(information)
    scatter = ax.scatter(travel_time, weighted_dop, c=utility * 1000.0,
                         s=sizes, cmap=THEME.colormap("blue"), alpha=0.58,
                         edgecolors="none", zorder=2)
    ax.scatter(travel_time[near_optimal], weighted_dop[near_optimal],
               s=45, facecolors="none", edgecolors=_PRIMARY_BLUE,
               linewidths=0.9, zorder=3)

    for _, (idx, color, marker) in selections.items():
        ax.scatter(travel_time[idx], weighted_dop[idx],
                   s=155 if marker == "*" else 100, marker=marker,
                   c=color, edgecolors="white", linewidths=1.0, zorder=6)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.025)
    cbar.ax.set_title("单位时间\n信息效用", fontsize=9, pad=7)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.set_title("优势一：避免为低 DOP 付出过高时间", fontsize=13, pad=12, color=_FOREGROUND)
    ax.set_xlabel("移动与测量时间 / s", fontsize=10, labelpad=7)
    ax.set_ylabel("加权 DOP（越低越好）", fontsize=10, labelpad=7)
    ax.grid(color=rgba(_TRANSITION, 0.20), lw=0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle("加权 DOP 近优决胜策略的优势", fontsize=16, color=_FOREGROUND, y=1.03)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"候选点数量: {len(candidates)}")
    print(f"近优集数量: {near_optimal.sum()} (tau={TAU:.0%})")
    for label, (idx, _, _) in selections.items():
        print(
            f"{label}: P=({candidates[idx, 0]:.0f}, {candidates[idx, 1]:.0f}), "
            f"时间={travel_time[idx]:.1f}s, 加权DOP={weighted_dop[idx]:.1f}, "
            f"U={utility[idx]:.6f}"
        )
    print(f"图像已保存: {OUTPUT}")


if __name__ == "__main__":
    main()
