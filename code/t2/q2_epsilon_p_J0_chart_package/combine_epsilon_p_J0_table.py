"""合并 epsilon_w、p_w 对统一评价指标 J0 的测试结果，并生成对比图。"""
import sys
from pathlib import Path

# 将工作区根目录添加到 Python 路径
# 脚本路径: <workspace>/code/t2/q2_epsilon_p_J0_chart_package/combine_epsilon_p_J0_table.py
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
from matplotlib.ticker import FuncFormatter

# 导入可视化主题配置（统一字体与配色）
from tools.visualization.src.mathmodel_viz.styles import (
    PRIMARY_RED,
    PRIMARY_BLUE,
    TRANSITION,
    AUXILIARY,
    configure_matplotlib,
    get_theme,
)

# 应用全局主题配置（字体、颜色循环、网格、保存参数等）
configure_matplotlib()

# 从主题获取前景色（蓝红混合的深色），与现有 PLUM 一致用于文字/注释
theme = get_theme()
PLUM = theme.foreground

DATA_DIR = Path(__file__).resolve().parent
BASELINE_J0 = 1580.6583557041383

BLUE = PRIMARY_BLUE
RED = PRIMARY_RED
PEACH = TRANSITION


def build_table(epsilon_data, power_data):
    """将两组各 100 次试验按试验序号横向合并。"""
    return pd.DataFrame(
        {
            "试验序号": epsilon_data["trial"].astype(int),
            "基础权重_epsilon_w": epsilon_data["epsilon_w"],
            "epsilon对应_J0_米": epsilon_data["J0_m"],
            "epsilon对应_J0相对基准变化_pct": (
                epsilon_data["J0_m"] / BASELINE_J0 - 1
            ) * 100,
            "形状参数_p_w": power_data["p_w"],
            "p对应_J0_米": power_data["J0_m"],
            "p对应_J0相对基准变化_pct": (
                power_data["J0_m"] / BASELINE_J0 - 1
            ) * 100,
        }
    )


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


def plot_combined(combined):
    """在同一个坐标区域中比较两个参数对 J0 的影响。"""
    x_normalized = np.linspace(0, 1, len(combined))
    epsilon_change = combined["epsilon对应_J0相对基准变化_pct"]
    power_change = combined["p对应_J0相对基准变化_pct"]

    fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=200)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.13, right=0.96, top=0.76, bottom=0.22)
    # fig.suptitle("权重参数对统一评价指标 J0 的影响", fontsize=17, color=PLUM, y=0.96)
    fig.text(
        0.5,
        0.885,
        # "两条曲线按各自试验区间对齐展示；纵轴放大以观察微小变化",
        " ",
        ha="center",
        fontsize=10.5,
        color=PLUM,
    )

    ax.set_facecolor("none")
    ax.grid(True, color=PEACH, linewidth=0.9, alpha=0.72)
    ax.tick_params(colors=PLUM, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(BLUE)
        spine.set_linewidth(1.0)

    ax.plot(
        x_normalized,
        epsilon_change,
        color=BLUE,
        linewidth=2.6,
        marker="o",
        markersize=3.4,
        markevery=8,
        label="基础权重 εw",
    )
    ax.plot(
        x_normalized,
        power_change,
        color=RED,
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
        epsilon_change.to_numpy(),
        power_change.to_numpy(),
        color=PEACH,
        alpha_left=0.2,
        alpha_right=1.0,
    )

    ax.axhline(0, color=PLUM, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(-0.0005, 0.0105)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.3f}%"))
    ax.set_ylabel("J0 相对基准值的变化（%）", fontsize=11, color=PLUM, labelpad=9)

    positions = np.linspace(0, 1, 5)
    epsilon_labels = [f"{v:.2f}" for v in np.linspace(0.10, 0.50, 5)]
    power_labels = [f"{v:.1f}" for v in np.linspace(1.0, 3.0, 5)]

    ax.set_xticks(positions)
    ax.set_xticklabels(epsilon_labels, color=BLUE)
    ax.set_xlabel("基础权重 εw", fontsize=11, color=BLUE, labelpad=9)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(positions)
    ax_top.set_xticklabels(power_labels, color=RED)
    ax_top.set_xlabel("形状参数 pw", fontsize=11, color=RED, labelpad=9)
    ax_top.tick_params(axis="x", colors=RED, labelsize=10)
    ax_top.spines["top"].set_color(RED)
    ax_top.spines["top"].set_linewidth(1.0)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax.annotate(
        f"εw 最大变化 {epsilon_change.max():.3f}%",
        xy=(x_normalized[epsilon_change.idxmax()], epsilon_change.max()),
        xytext=(50,-65),
        textcoords="offset points",
        fontsize=10,
        color=BLUE,
        arrowprops={"arrowstyle": "->", "color": BLUE},
    )
    ax.annotate(
        f"pw 最大变化 {power_change.max():.3f}%",
        xy=(x_normalized[power_change.idxmax()], power_change.max()),
        xytext=(8, 15),
        textcoords="offset points",
        fontsize=10,
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED},
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        frameon=False,
        fontsize=10,
    )

    output_path = DATA_DIR / "epsilon_p_effect_on_J0.png"
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return output_path


def main():
    epsilon_data = pd.read_csv(DATA_DIR / "epsilon_100_trials.csv")
    power_data = pd.read_csv(DATA_DIR / "p_100_trials.csv")

    combined = build_table(epsilon_data, power_data)
    table_path = DATA_DIR / "epsilon_p_effect_on_J0_combined.csv"
    combined.to_csv(table_path, index=False, encoding="utf-8-sig")
    chart_path = plot_combined(combined)

    print(f"基准 J0：{BASELINE_J0:.6f} 米")
    print(f"已生成表格：{table_path}")
    print(f"已生成图片：{chart_path}")


if __name__ == "__main__":
    main()
