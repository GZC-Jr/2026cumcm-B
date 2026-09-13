"""时间—加权 DOP 二维热力图（蓝白主题）。

横轴：移动与测量时间 T(P)
纵轴：区域深度加权 DOP，J_DOP(P)
背景：时间和 DOP 共同决定的归一化综合优度，越深蓝越优。
散点：由几何候选点按论文公式计算得到的方案。

运行：python3 time_weighted_dop_heatmap.py
输出：output/time_weighted_dop_heatmap.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


DOMAIN_RADIUS = 1800.0
SPEED = 5.0
SERVICE_TIME = 6.0
SIGMA_THETA = np.deg2rad(1.0)
TAU = 0.03
EPSILON_W = 0.18
P_W = 2.0
OUTPUT = Path("time_weighted_dop_heatmap.png")


def build_target_samples(mean: np.ndarray, a: float, b: float):
    gx = np.linspace(mean[0] - a, mean[0] + a, 35)
    gy = np.linspace(mean[1] - b, mean[1] + b, 29)
    xx, yy = np.meshgrid(gx, gy)
    radial = np.sqrt(((xx - mean[0]) / a) ** 2 + ((yy - mean[1]) / b) ** 2)
    mask = radial <= 1.0
    samples = np.column_stack((xx[mask], yy[mask]))
    depth = 1.0 - radial[mask]
    weights = EPSILON_W + (1.0 - EPSILON_W) * depth**P_W
    return samples, weights


def build_candidates(mean: np.ndarray, samples: np.ndarray, first_station: np.ndarray):
    axis = np.linspace(-1050.0, 1350.0, 101)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    basic = (
        (np.linalg.norm(points, axis=1) <= DOMAIN_RADIUS)
        & (np.linalg.norm(points - mean, axis=1) >= 230.0)
    )
    points = points[basic]

    first_distance = np.linalg.norm(samples - first_station, axis=1)
    receive_limit = np.maximum(1000.0, first_distance)
    feasible = []
    for point in points:
        second_distance = np.linalg.norm(samples - point, axis=1)
        if np.all(second_distance <= receive_limit):
            feasible.append(point)
    return np.asarray(feasible)


def calculate_metrics(candidates, samples, weights, first_station, current, mean, covariance):
    time = np.linalg.norm(candidates - current, axis=1) / SPEED + SERVICE_TIME

    delta = mean - candidates
    range_sq = np.sum(delta**2, axis=1)
    h = np.column_stack((-delta[:, 1] / range_sq, delta[:, 0] / range_sq))
    projected_variance = np.einsum("ij,jk,ik->i", h, covariance, h)
    information = 0.5 * np.log1p(projected_variance / SIGMA_THETA**2)
    utility = information / time

    v1 = first_station - samples
    r1 = np.linalg.norm(v1, axis=1)
    weighted_dop = np.empty(len(candidates))
    for idx, point in enumerate(candidates):
        v2 = point - samples
        r2 = np.linalg.norm(v2, axis=1)
        cross = np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        sin_angle = cross / np.maximum(r1 * r2, 1e-9)
        dop = np.sqrt(r1**2 + r2**2) / np.maximum(sin_angle, 0.035)
        weighted_dop[idx] = np.average(dop, weights=weights)

    return time, weighted_dop, utility


def percentile_normalize(values, low, high):
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def main() -> None:
    mean = np.array([300.0, 280.0])
    first_station = np.array([-620.0, 120.0])
    current = np.array([-1180.0, -850.0])
    covariance = np.array([
        [210.0**2, 0.30 * 210.0 * 130.0],
        [0.30 * 210.0 * 130.0, 130.0**2],
    ])

    samples, weights = build_target_samples(mean, 410.0, 255.0)
    candidates = build_candidates(mean, samples, first_station)
    time, dop, utility = calculate_metrics(
        candidates, samples, weights, first_station, current, mean, covariance
    )

    # 避免少数共线奇异候选点把纵轴压得过宽。
    t_high = np.percentile(time, 99)
    t_low = max(0.0, np.min(time) - 0.03 * (t_high - np.min(time)))
    time_only_idx = int(np.argmin(time))
    d_low = np.percentile(dop, 1)
    d_high = max(np.percentile(dop, 92), dop[time_only_idx] * 1.08)
    d_plot_low = max(0.0, d_low - 0.08 * (d_high - d_low))
    display = (time <= t_high) & (dop <= d_high)

    # 在“时间—DOP”平面构造连续优度背景：左下角兼具低时间、低 DOP。
    tx = np.linspace(t_low, t_high, 280)
    dy = np.linspace(d_plot_low, d_high, 280)
    tt, dd = np.meshgrid(tx, dy)
    t_norm = percentile_normalize(tt, t_low, t_high)
    d_norm = percentile_normalize(dd, d_low, d_high)
    joint_loss = np.sqrt(0.5 * t_norm**2 + 0.5 * d_norm**2)
    desirability = np.clip(1.0 - joint_loss, 0.0, 1.0)

    # 与论文一致的最终选择：先取单位时间信息效用近优集，再最小化加权 DOP。
    utility_star = np.max(utility)
    near_optimal = utility >= (1.0 - TAU) * utility_star
    near_indices = np.flatnonzero(near_optimal)
    idx_joint = int(near_indices[np.argmin(dop[near_optimal])])
    idx_time = int(np.argmin(time))
    idx_dop = int(np.argmin(dop))

    cmap = LinearSegmentedColormap.from_list(
        "blue_white", ["#FFFFFF", "#EAF4FB", "#CBE3F2", "#8EC5E3", "#4A94C2", "#155A8A"]
    )

    fig, ax = plt.subplots(figsize=(10.5, 7.6), dpi=220, constrained_layout=True)
    fig.patch.set_facecolor("#FBFCFE")
    ax.set_facecolor("#FFFFFF")

    heat = ax.contourf(tt, dd, desirability, levels=np.linspace(0, 1, 22),
                       cmap=cmap, vmin=0, vmax=1, zorder=0)
    ax.contour(tt, dd, desirability, levels=[0.25, 0.45, 0.65, 0.82],
               colors="#6EA6C9", linewidths=0.65, alpha=0.52, zorder=1)

    # 全部可视候选点以及信息效用近优点。
    ax.scatter(time[display], dop[display], s=18, c="#6F93AB",
               alpha=0.32, edgecolors="none", zorder=2)
    near_display = near_optimal & display
    ax.scatter(time[near_display], dop[near_display], s=34,
               facecolors="white", edgecolors="#2B7FB8",
               linewidths=1.0, alpha=0.92, zorder=3)

    key_points = [
        (idx_time, "仅考虑时间", "s", "#E48252"),
        (idx_dop, "仅考虑 DOP", "X", "#7356A8"),
        (idx_joint, "本文联合选择", "*", "#0B7A75"),
    ]
    for idx, label, marker, color in key_points:
        ax.scatter(time[idx], dop[idx], s=180 if marker == "*" else 105,
                   marker=marker, c=color, edgecolors="white",
                   linewidths=1.1, zorder=6)

    cbar = fig.colorbar(heat, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("时间—DOP 综合优度", fontsize=10, labelpad=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["低", "中", "高"])

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6F93AB",
               alpha=0.55, markersize=7, label="可行候选点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#2B7FB8", markersize=8, label="信息效用近优集"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#E48252",
               markeredgecolor="white", markersize=8, label="仅考虑时间"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#7356A8",
               markeredgecolor="white", markersize=8, label="仅考虑 DOP"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#0B7A75",
               markeredgecolor="white", markersize=11, label="本文联合选择"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.19),
              ncol=5, frameon=False, fontsize=9.2, columnspacing=1.5)

    ax.set_title("时间—加权 DOP 二维热力图", fontsize=16, color="#243B53", pad=14)
    ax.set_xlabel("移动与测量时间  $T(P)$ / s", fontsize=11, labelpad=9)
    ax.set_ylabel("加权 DOP  $J_{DOP}(P)$（越低越好）", fontsize=11, labelpad=9)
    ax.set_xlim(t_low, t_high)
    ax.set_ylim(d_plot_low, d_high)
    ax.grid(color="#8CB4CC", linewidth=0.45, alpha=0.24)
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"候选点数量: {len(candidates)}")
    print(f"信息效用近优点数量: {near_optimal.sum()}")
    print(f"仅时间点: T={time[idx_time]:.1f}s, DOP={dop[idx_time]:.1f}")
    print(f"仅DOP点: T={time[idx_dop]:.1f}s, DOP={dop[idx_dop]:.1f}")
    print(f"本文选择: T={time[idx_joint]:.1f}s, DOP={dop[idx_joint]:.1f}")
    print(f"图像已保存: {OUTPUT}")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"] + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.unicode_minus"] = False
    main()
