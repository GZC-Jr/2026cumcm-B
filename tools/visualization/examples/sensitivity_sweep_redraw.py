"""灵敏度分析（Sensitivity Sweep）四面板折线图 —— 复刻微信附图并套用项目全局样式。

绘制逻辑（从原图解析）：
1. 布局：4 个面板垂直堆叠（产量 / 价格 / 利润 / 组合），共享横轴；
2. 横轴：实验次数 x = 0..100（刻度 0/20/40/60/80/100）；
3. 折线：每个因素设基准滑销率 base，单条带噪折线 y = base + U(-h, h)，
   其中 h 等于面板半高，因此折线振幅恰好与面板高度一致（极值贴近上下边）；
4. 置信带：长条状固定区间，上下边界为常数水平线（噪声分布的理论
   25%–75% 分位 = base ± 0.5h），不随实验次数波动；底色与折线同色、透明度 50%；
5. 标注：面板标题在上方，y 轴标签"滑销率波动"置于右侧，底部面板标注 x 轴"实验次数"；
6. 样式：使用项目全局字体（中文 SimSun、西文 Times New Roman）与主色调
   （主红 #B41B20 / 主蓝 #266AA0 / 过渡 #EF8F67 等，取自 mathmodel_viz.styles）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mathmodel_viz.styles import configure_matplotlib, get_theme, save_figure  # noqa: E402

BAND_FRACTION = 0.5  # 平直置信带半宽占面板半高的比例（均匀分布 25%-75% 分位）


def main() -> Path:
    configure_matplotlib(figure_dpi=150, savefig_dpi=300)
    theme = get_theme()
    colors = theme.series_colors  # (主蓝, 主红, 过渡, 混合1, 混合2, 辅助)

    rng = np.random.default_rng(2026)

    # (面板标题, 基准滑销率, y 轴刻度)
    panels = [
        ("产量波动对滑销率的影响（含5%噪声）", 14.35, np.arange(14.0, 14.76, 0.1)),
        ("价格波动对滑销率的影响（含5%噪声）", 0.179, np.arange(0.176, 0.1831, 0.001)),
        ("利润波动对滑销率的影响（含5%噪声）", 0.0635, np.arange(0.062, 0.0656, 0.001)),
        ("组合滑销率的波动（含5%噪声）", 4.925, np.arange(4.85, 5.01, 0.05)),
    ]

    x = np.arange(101)  # 实验次数 0..100
    ticks = [0, 20, 40, 60, 80, 100]
    x_pad = 20 / 4  # 起止两侧留白：20 次实验宽度的 1/4 = 5 次

    figure, axes = plt.subplots(4, 1, figsize=(15, 7.5), sharex=True)

    for axis, (title, base, yticks), color in zip(axes, panels, colors[:4]):
        panel_low, panel_high = yticks[0], yticks[-1]
        panel_half = (panel_high - panel_low) / 2  # 面板半高

        # 平直置信带：上下边界为常数（理论 25%-75% 分位），边界不随 x 波动
        band_half = BAND_FRACTION * panel_half
        axis.fill_between(
            x,
            np.full_like(x, base - band_half, dtype=float),
            np.full_like(x, base + band_half, dtype=float),
            color=color,
            alpha=0.5,
            linewidth=0,
        )

        # 单条带噪折线：振幅 = 面板半高，极值恰好贴近面板上下边；每个转折点加同色圆点
        y = base + rng.uniform(-1, 1, size=x.size) * panel_half
        axis.plot(
            x,
            y,
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor=color,
        )

        axis.set_title(title, fontsize=11)
        axis.set_ylabel("滑销率波动", fontsize=10)
        axis.yaxis.set_label_position("right")
        axis.set_yticks(yticks)
        axis.set_ylim(panel_low, panel_high)
        axis.set_xlim(0 - x_pad, 100 + x_pad)
        axis.grid(True, axis="y", alpha=0.35)

    axes[-1].set_xlabel("实验次数", fontsize=11)
    axes[-1].set_xticks(ticks)

    figure.tight_layout()
    output = Path(__file__).resolve().parent / "sensitivity_analysis_theme.png"
    return save_figure(figure, output)


if __name__ == "__main__":
    saved = main()
    print(f"saved: {saved}")
