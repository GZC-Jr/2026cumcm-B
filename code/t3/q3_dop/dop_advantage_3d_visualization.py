"""问题三：单位时间信息增益 + 加权 DOP 近优决胜的优势可视化（3D 曲面版）。

图中的候选点与后验区域是用于解释算法机制的确定性示例，
所有指标均按论文中的公式现场计算，不代表模拟器实测结果。

左侧面板：候选点空间位置分布
右侧面板：时间—DOP—效用 3D 曲面

运行：python3 dop_advantage_3d_visualization.py
输出：outputs/t3/dop_advantage_3d_visualization.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d import Axes3D

# 导入全局主题配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "tools" / "visualization" / "src"))
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
OUTPUT = Path("dop_advantage_3d_visualization.png")


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


def percentile_normalize(values, low, high):
    """百分位归一化。"""
    return np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)


def normalize(values):
    """线性归一化。"""
    lo, hi = np.min(values), np.max(values)
    return (values - lo) / (hi - lo + 1e-12)


def main() -> None:
    # 参数设置
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

    # 计算近优集和关键选择点
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

    # -------------------- 创建图形 --------------------
    fig = plt.figure(figsize=(14.0, 6.0), dpi=220)
    fig.patch.set_facecolor(rgba(_PRIMARY_BLUE, 0.02))

    # 左侧面板：候选点空间位置（2D）
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor(rgba(_PRIMARY_BLUE, 0.04))
    ax1.add_patch(Circle((0, 0), DOMAIN_RADIUS, facecolor=rgba(_PRIMARY_BLUE, 0.06),
                        edgecolor=_FOREGROUND, linewidth=2.0, zorder=0))
    ax1.add_patch(Ellipse(mean, 2 * ellipse_a, 2 * ellipse_b,
                         facecolor="none", edgecolor=_TRANSITION,
                         linewidth=1.7, zorder=2.1))

    # 椭圆后验区域：放射渐变填充（中心实、边缘透明），用同一椭圆作为 clip_path
    gradient_size = 256
    lin_x = np.linspace(-1.0, 1.0, gradient_size)
    lin_y = np.linspace(-1.0, 1.0, gradient_size)
    gx, gy = np.meshgrid(lin_x, lin_y)
    ellipse_r = np.sqrt(gx ** 2 + gy ** 2)  # 归一化半径，1.0 = 椭圆边界
    alpha_mask = np.clip(1.0 - ellipse_r, 0.0, 1.0) ** 1.5  # 距中心越远越透明
    red_c, green_c, blue_c, _ = rgba(_TRANSITION, 1.0)
    rgba_image = np.zeros((gradient_size, gradient_size, 4), dtype=float)
    rgba_image[..., 0] = red_c
    rgba_image[..., 1] = green_c
    rgba_image[..., 2] = blue_c
    rgba_image[..., 3] = alpha_mask * 0.7  # 中心最大不透明度 0.7
    extent_ellipse = [mean[0] - ellipse_a, mean[0] + ellipse_a,
                      mean[1] - ellipse_b, mean[1] + ellipse_b]
    ellipse_border = Ellipse(mean, 2 * ellipse_a, 2 * ellipse_b,
                             facecolor="none", edgecolor=_TRANSITION,
                             linewidth=1.7, zorder=2.1)
    ax1.add_patch(ellipse_border)
    ax1.imshow(rgba_image, extent=extent_ellipse, origin="lower",
               interpolation="bilinear", clip_path=ellipse_border,
               zorder=2)

    ax1.scatter(candidates[:, 0], candidates[:, 1], s=8, c=[rgba(_PRIMARY_BLUE, 0.45)],
               linewidths=0, zorder=1)
    ax1.scatter(candidates[near_optimal, 0], candidates[near_optimal, 1],
               s=22, facecolors="none", edgecolors=_PRIMARY_BLUE,
               linewidths=0.9, alpha=0.9, zorder=3)

    ax1.scatter(*first_station, s=90, marker="^", c=_PRIMARY_BLUE,
               edgecolors="white", linewidths=1.0, zorder=7)
    ax1.scatter(*current, s=100, marker="D", c=_PRIMARY_RED,
               edgecolors="white", linewidths=1.0, zorder=7)
    ax1.scatter(*mean, s=70, marker="o", c=_TRANSITION,
               edgecolors="white", linewidths=1.0, zorder=7)

    for label, (idx, color, marker) in selections.items():
        point = candidates[idx]
        ax1.plot([current[0], point[0]], [current[1], point[1]],
                color=color, lw=1.55, alpha=0.75, zorder=4)
        ax1.scatter(*point, s=130 if marker == "*" else 88, marker=marker,
                   c=color, edgecolors="white", linewidths=1.0, zorder=8)

    # 统一的图例（左右两侧共用，放置在左图下方，单行排列）
    unified_legend = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor=_PRIMARY_BLUE,
               markeredgecolor="white", markersize=7, label="首次检测点"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=_PRIMARY_RED,
               markeredgecolor="white", markersize=7, label="当前位置"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_TRANSITION,
               markeredgecolor="white", markersize=7, label="后验均值"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=_PRIMARY_BLUE, markersize=7, label="效用近优集"),
    ]
    for label, (_, color, marker) in selections.items():
        unified_legend.append(Line2D([0], [0], marker=marker, color="none",
                                     markerfacecolor=color, markeredgecolor="white",
                                     markersize=8, label=label))
    # 左图：仅设置轴属性，不再单独绘制图例
    ax1.set_title("候选点选择：兼顾时间与交会几何", fontsize=13, pad=12, color=_FOREGROUND)
    ax1.set_xlim(-1900, 1900)
    ax1.set_ylim(-1900, 1900)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("x / m", fontsize=10)
    ax1.set_ylabel("y / m", fontsize=10)
    ax1.grid(color=rgba(_TRANSITION, 0.20), lw=0.4)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # -------------------- 右侧面板：时间—DOP—效用 3D 曲面 --------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor(rgba(_PRIMARY_BLUE, 0.04))

    # 计算3D曲面范围
    t_high = np.percentile(travel_time, 99)
    t_low = max(0.0, np.min(travel_time) - 0.03 * (np.percentile(travel_time, 99) - np.min(travel_time)))
    # 对数 y 轴：下限取最小正值，上限拉高以使数据点落在中段
    d_low = max(800.0, np.percentile(weighted_dop, 1))
    d_high = max(d_low * 30.0, np.percentile(weighted_dop, 99) * 1.5)

    # 创建网格：x 线性、y 原始线性；对数变换在 plot 时通过 log10 施加
    tx = np.linspace(t_low, t_high, 80)
    dy = np.linspace(d_low, d_high, 80)
    TT, DD = np.meshgrid(tx, dy)

    # 对数变换：在数据层面把 y 投到 log10 空间，z 保持原值，
    # 这样 plot_surface 在线性坐标下绘制，但视觉上呈对数间隔
    DD_LOG = np.log10(DD)
    WD_LOG = np.log10(weighted_dop)

    # 计算综合优度作为Z轴（效用代理）
    t_norm = percentile_normalize(TT, t_low, t_high)
    d_norm = percentile_normalize(DD, d_low, d_high)
    # 效用与时间成正比，与DOP成反比
    joint_utility = (1.0 - t_norm) * (1.0 - d_norm * 0.5)

    # 绘制3D曲面（主题红色 + 沿Z轴的透明度梯度）
    red_cmap = THEME.colormap("red")
    z_min = float(joint_utility.min())
    z_max = float(joint_utility.max())
    quad_z = 0.25 * (
        joint_utility[:-1, :-1]
        + joint_utility[1:, :-1]
        + joint_utility[:-1, 1:]
        + joint_utility[1:, 1:]
    )
    quad_z_norm = (quad_z - z_min) / (z_max - z_min + 1e-12)
    face_colors = red_cmap(quad_z_norm)
    # Z 越高 → 越透明（alpha 由 0.88 渐变到 0.15）
    face_colors[..., 3] = 0.95 - 0.80 * quad_z_norm

    surf = ax2.plot_surface(TT, DD_LOG, joint_utility,
                            facecolors=face_colors,
                            linewidth=0.15,
                            antialiased=True,
                            rstride=2,
                            cstride=2,
                            edgecolor=rgba(_PRIMARY_RED, 0.55),
                            shade=False)

    # 在3D曲面上绘制候选点
    sizes = 18 + 70 * normalize(information)
    scatter3d = ax2.scatter(travel_time, WD_LOG, utility,
                          c=utility * 1000.0,
                          s=sizes,
                          cmap=THEME.colormap("blue"),
                          alpha=0.65,
                          depthshade=True,
                          edgecolors='none',
                          zorder=10)

    # 绘制近优集
    ax2.scatter(travel_time[near_optimal], WD_LOG[near_optimal], utility[near_optimal],
               s=35, facecolors="none", edgecolors=_PRIMARY_BLUE,
               linewidths=1.2, alpha=0.9, label="效用近优集")

    # 标注关键选择点
    for label, (idx, color, marker) in selections.items():
        ax2.scatter(travel_time[idx], WD_LOG[idx], utility[idx],
                   s=155 if marker == "*" else 100, marker=marker,
                   c=color, edgecolors="white", linewidths=1.2,
                   depthshade=False, zorder=11)

    # 设置3D轴标签和标题
    ax2.set_xlabel("移动与测量时间 / s", fontsize=9, labelpad=6)
    ax2.set_ylabel("加权 DOP（对数分布轴）", fontsize=9, labelpad=6)
    ax2.set_zlabel("单位时间信息效用", fontsize=9, labelpad=6)
    ax2.set_title("优势：时间—DOP—效用 3D 视图", fontsize=13, pad=12, color=_FOREGROUND)

    # y 轴在数据层面已做 log10 变换，此处用自定义刻度显示原始 DOP 值；
    # 通过 set_ylim 的反向顺序将原点置于右上角，正方向向下（DOP 越大越靠下）
    d_ticks = [5000, 10000, 15000, 20000, 30000]
    ax2.set_yticks(np.log10(d_ticks))
    ax2.set_yticklabels([f"{v:.0f}" for v in d_ticks])
    ax2.set_ylim(np.log10(d_high), np.log10(d_low))

    # 仅颠倒右图 x 轴（时间）的正方向；加权 DOP 轴保持对数刻度，原点位于右上角
    ax2.invert_xaxis()

    # 设置视角
    ax2.view_init(elev=25, azim=45)

    # 添加颜色条
    cbar = fig.colorbar(scatter3d, ax=ax2, fraction=0.04, pad=0.08, shrink=0.6)
    cbar.set_label("单位时间信息效用", fontsize=9)

    # 左右两侧共用底部统一图例（单行排列，跨越整个 figure 宽度）
    fig.legend(handles=unified_legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), ncol=7, frameon=False, fontsize=8.0,
              columnspacing=1.6, handletextpad=0.6)

    # fig.suptitle("加权 DOP 近优决胜策略的优势（3D 曲面版）", fontsize=16, color=_FOREGROUND, y=1.02)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    # 打印统计信息
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
