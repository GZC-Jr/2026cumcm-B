"""生成问题二的两张敏感性分析图。

输入文件（与本脚本放在同一目录）：
- epsilon_100_trials.csv
- p_100_trials.csv
- tau_100_trials.csv

输出文件：
- weight_parameters_vs_optimal_angle.png
- tau_vs_candidate_area.png
"""
import sys
from pathlib import Path

# 将工作区根目录添加到 Python 路径
# 脚本路径: <workspace>/code/t2/q2_sensitivity_results/generate_two_sensitivity_charts.py
# 需向上回溯 4 层才能到达工作区根目录
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, MaxNLocator

# 导入可视化主题配置（统一字体与配色）
from tools.visualization.src.mathmodel_viz.styles import (
    PRIMARY_RED,
    PRIMARY_BLUE,
    TRANSITION,
    AUXILIARY,
    configure_matplotlib,
    rgba,
    get_theme,
)

# 应用全局主题配置
configure_matplotlib()

OUT = Path(__file__).resolve().parent

# 从主题获取前景色（蓝红混合的深色）
theme = get_theme()
PLUM = theme.foreground  # 蓝红混合的深紫色，用于文字
BG = rgba(PRIMARY_BLUE, 0.02)  # 极淡蓝背景


def load_data():
    """读取三组各 100 次试验的结果。"""
    epsilon_df = pd.read_csv(OUT / "epsilon_100_trials.csv")
    power_df = pd.read_csv(OUT / "p_100_trials.csv")
    tau_df = pd.read_csv(OUT / "tau_100_trials.csv")
    return epsilon_df, power_df, tau_df


def style_axis(ax):
    """设置与参考图一致的网格、边框（背景设为透明）。"""
    ax.set_facecolor("none")
    ax.grid(True, color=TRANSITION, linewidth=0.9, alpha=0.72)
    ax.tick_params(colors=PLUM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(PRIMARY_BLUE)
        spine.set_linewidth(1.0)


def gradient_fill_between(ax, x, y_lower, y_upper, color, alpha_left, alpha_right):
    """在两条曲线之间填充一个水平方向的 alpha 渐变带。

    渐变方向：随 ``x`` 自 ``x.min()`` 增至 ``x.max()``，
    alpha 由 ``alpha_left`` 线性过渡到 ``alpha_right``。
    通过 ``PathPatch`` 裁剪 ``imshow`` 的矩形，使渐变仅出现在两条曲线围成的多边形内。
    """
    x_arr = np.asarray(x, dtype=float)
    y_lo = np.asarray(y_lower, dtype=float)
    y_hi = np.asarray(y_upper, dtype=float)
    rgb = to_rgb(color)

    # 构造闭合多边形路径：下边沿 y_lower 走一遍，再沿 y_upper 反向走一遍
    poly_x = np.concatenate([x_arr, x_arr[::-1]])
    poly_y = np.concatenate([y_lo, y_hi[::-1]])
    band_path = MplPath(np.column_stack([poly_x, poly_y]))
    clip_patch = PathPatch(band_path, facecolor="none", edgecolor="none", zorder=0)
    ax.add_patch(clip_patch)

    # 1xN 的 RGBA 渐变：水平方向上 alpha 由左到右线性变化
    n = 512
    grad = np.zeros((1, n, 4), dtype=float)
    grad[0, :, 0] = rgb[0]
    grad[0, :, 1] = rgb[1]
    grad[0, :, 2] = rgb[2]
    grad[0, :, 3] = np.linspace(alpha_left, alpha_right, n)

    x_min, x_max = float(x_arr.min()), float(x_arr.max())
    y_min = float(min(y_lo.min(), y_hi.min()))
    y_max = float(max(y_lo.max(), y_hi.max()))

    ax.imshow(
        grad,
        aspect="auto",
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        interpolation="bilinear",
        zorder=1,
        clip_path=clip_patch,
        clip_on=True,
    )


def plot_weight_parameters(epsilon_df, power_df):
    """把 εw、pw 对最优偏转角的影响画在同一个坐标区域内。"""
    # 两个参数量纲不同，因此按各自试验区间的相对位置对齐；
    # 下轴显示 εw 原始值，上轴显示 pw 原始值。
    x_normalized = np.linspace(0, 1, len(epsilon_df))

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=200)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.76, bottom=0.22)
    # fig.suptitle(
    #     "权重参数对最优位置偏转角的影响",
    #     fontsize=17,
    #     color=PLUM,
    #     y=0.96,
    # )
    # fig.text(
    #     0.5,
    #     0.885,
    #     "两条曲线在各自试验区间内对齐展示；偏转角变化均不足 1°",
    #     ha="center",
    #     fontsize=10.5,
    #     color=PLUM,
    # )

    style_axis(ax)

    # εw 曲线
    ax.plot(
        x_normalized,
        epsilon_df["beta_star_deg"],
        color=PRIMARY_BLUE,
        linewidth=2.6,
        marker="o",
        markersize=3.4,
        markevery=8,
        label="基础权重 εw",
    )

    # pw 曲线
    ax.plot(
        x_normalized,
        power_df["beta_star_deg"],
        color=PRIMARY_RED,
        linewidth=2.6,
        marker="o",
        markersize=3.4,
        markevery=8,
        label="形状参数 pw",
    )

    # 两条曲线之间使用渐变填充（橙色 TRANSITION），沿 x 向右由浅至深
    gradient_fill_between(
        ax,
        x_normalized,
        epsilon_df["beta_star_deg"].to_numpy(),
        power_df["beta_star_deg"].to_numpy(),
        color=TRANSITION,
        alpha_left=0.2,
        alpha_right=1,
    )

    ax.set_ylim(29.75, 30.60)
    ax.set_ylabel("最优偏转角 β*（度）", fontsize=11, color=PLUM, labelpad=9)

    positions = np.linspace(0, 1, 5)
    epsilon_labels = [
        f"{value:.2f}"
        for value in np.linspace(
            epsilon_df["epsilon_w"].min(),
            epsilon_df["epsilon_w"].max(),
            5,
        )
    ]
    power_labels = [
        f"{value:.1f}"
        for value in np.linspace(
            power_df["p_w"].min(),
            power_df["p_w"].max(),
            5,
        )
    ]

    # 下方横轴：εw
    ax.set_xticks(positions)
    ax.set_xticklabels(epsilon_labels, color=PRIMARY_BLUE)
    ax.set_xlabel("基础权重 εw", fontsize=11, color=PRIMARY_BLUE, labelpad=9)

    # 上方横轴：pw
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(power_labels, color=PRIMARY_RED)
    ax_top.set_xlabel("形状参数 pw", fontsize=11, color=PRIMARY_RED, labelpad=9)
    ax_top.tick_params(axis="x", colors=PRIMARY_RED, labelsize=10)
    ax_top.spines["top"].set_color(PRIMARY_RED)
    ax_top.spines["top"].set_linewidth(1.0)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    epsilon_change = (
        epsilon_df["beta_star_deg"].iloc[-1]
        - epsilon_df["beta_star_deg"].iloc[0]
    )
    power_change = (
        power_df["beta_star_deg"].iloc[-1]
        - power_df["beta_star_deg"].iloc[0]
    )

    ax.annotate(
        f"εw：变化 {epsilon_change:.3f}°",
        xy=(1, epsilon_df["beta_star_deg"].iloc[-1]),
        xytext=(-132, -50),
        textcoords="offset points",
        fontsize=10,
        color=PRIMARY_BLUE,
        arrowprops={"arrowstyle": "->", "color": PRIMARY_BLUE},
    )
    ax.annotate(
        f"pw：变化 {power_change:.3f}°",
        xy=(1, power_df["beta_star_deg"].iloc[-1]),
        xytext=(-130, 5),
        textcoords="offset points",
        fontsize=10,
        color=PRIMARY_RED,
        arrowprops={"arrowstyle": "->", "color": PRIMARY_RED},
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    output_path = OUT / "weight_parameters_vs_optimal_angle.png"
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def plot_tau_area(tau_df):
    """绘制近优容差 τ 对候选区域面积的影响。"""
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=200)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.82, bottom=0.17)
    style_axis(ax)

    # fig.suptitle(
    #     "近优容差对候选区域面积的影响",
    #     fontsize=17,
    #     color=PLUM,
    #     y=0.95,
    # )
    fig.text(
        0.5,
        0.865,
        "τ 增大时，允许的性能损失提高，候选区域快速扩张",
        ha="center",
        fontsize=10.5,
        color=PLUM,
    )

    tau_percent = tau_df["tau"] * 100
    area = tau_df["candidate_area_m2"]

    # 蓝色区域填充，沿 x 向右由深至浅（alpha 由 0.50 线性降至 0.06）
    gradient_fill_between(
        ax,
        tau_percent.to_numpy(),
        np.zeros_like(tau_percent, dtype=float),
        area.to_numpy(),
        color=PRIMARY_BLUE,
        alpha_left=0.75,
        alpha_right=0.06,
    )
    ax.plot(
        tau_percent,
        area,
        color=PRIMARY_RED,
        linewidth=2.8,
        marker="o",
        markersize=3.4,
        markevery=7,
    )

    ax.set_xlabel("近优容差 τ（%）", fontsize=11, color=PLUM, labelpad=8)
    ax.set_ylabel("候选区域面积（平方米）", fontsize=11, color=PLUM, labelpad=9)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, area.max() * 1.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f"{value:,.0f}")
    )

    multiplier = area.iloc[-1] / area.iloc[0]
    ax.annotate(
        f"从 {area.iloc[0]:,.0f} 增至 {area.iloc[-1]:,.0f} 平方米\n"
        f"扩大约 {multiplier:.1f} 倍",
        xy=(tau_percent.iloc[-1], area.iloc[-1]),
        xytext=(-275, -58),
        textcoords="offset points",
        fontsize=10,
        color=PRIMARY_RED,
        linespacing=1.45,
        arrowprops={"arrowstyle": "->", "color": PRIMARY_RED},
    )

    output_path = OUT / "tau_vs_candidate_area.png"
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def main():
    epsilon_df, power_df, tau_df = load_data()
    first_chart = plot_weight_parameters(epsilon_df, power_df)
    second_chart = plot_tau_area(tau_df)
    print(f"已生成：{first_chart}")
    print(f"已生成：{second_chart}")


if __name__ == "__main__":
    main()
