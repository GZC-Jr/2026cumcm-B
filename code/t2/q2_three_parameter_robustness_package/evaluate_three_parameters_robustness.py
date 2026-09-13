"""问题二三个参数的进一步鲁棒性评估。

实验设计：
1. 以 epsilon_w=0.30、p_w=2.00、tau=0.05 为基准；
2. 每次只扰动一个参数，参数在基准值 ±5% 内均匀抽样；
3. 每个参数独立进行 100 次试验，固定随机种子 2026；
4. epsilon_w、p_w 的响应为最优偏转角 beta*；tau 的响应为候选区域面积；
5. 响应由此前连续数值模型的 100 点扫描结果线性插值得到，不使用虚构响应值；
6. 输出试验明细 CSV、统计摘要 CSV 和三面板鲁棒性评估图。
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


# Load the shared visualization theme from tools/visualization/src so that
# fonts, color cycle, grid and export defaults stay aligned with the project
# global theme.  The path is computed relative to this file so the script
# keeps working whether mathmodel_viz is installed or sourced in-place.
_THIS_DIR = Path(__file__).resolve().parent
_VIZ_SRC = _THIS_DIR.parents[2] / "tools" / "visualization" / "src"
if str(_VIZ_SRC) not in sys.path:
    sys.path.insert(0, str(_VIZ_SRC))

from mathmodel_viz import configure_matplotlib, get_theme, save_figure

configure_matplotlib()
THEME = get_theme()

# Theme-aligned color tokens (no hardcoded hex outside the shared theme).
PRIMARY_BLUE = THEME.primary_blue
PRIMARY_RED = THEME.primary_red
TRANSITION = THEME.transition
FOREGROUND = THEME.foreground       # neutral dark for axis text / spines / titles
GRID_COLOR = THEME.transition        # warm soft grid color; alpha tuned per call


OUT = Path(__file__).resolve().parent
SEED = 2026
N_TRIALS = 100
NOISE_RATE = 0.05

BASE_EPSILON = 0.30
BASE_POWER = 2.00
BASE_TAU = 0.05

# Configure_matplotlib already installs the shared rcParams (Times New Roman
# + SimSun, transparent paper/axes facecolor, themed grid/spines).  The
# ``style_axis`` helper below overrides the bits that need per-axis tweaking.


def interpolate_response(sample, grid, response):
    """基于已求解的高密度参数扫描结果插值响应。"""
    order = np.argsort(grid)
    return np.interp(sample, np.asarray(grid)[order], np.asarray(response)[order])


def generate_experiments():
    epsilon_sweep = pd.read_csv(OUT / "epsilon_100_trials.csv")
    power_sweep = pd.read_csv(OUT / "p_100_trials.csv")
    tau_sweep = pd.read_csv(OUT / "tau_100_trials.csv")

    rng = np.random.default_rng(SEED)
    epsilon_samples = rng.uniform(
        BASE_EPSILON * (1 - NOISE_RATE),
        BASE_EPSILON * (1 + NOISE_RATE),
        N_TRIALS,
    )
    power_samples = rng.uniform(
        BASE_POWER * (1 - NOISE_RATE),
        BASE_POWER * (1 + NOISE_RATE),
        N_TRIALS,
    )
    tau_samples = rng.uniform(
        BASE_TAU * (1 - NOISE_RATE),
        BASE_TAU * (1 + NOISE_RATE),
        N_TRIALS,
    )

    epsilon_beta = interpolate_response(
        epsilon_samples,
        epsilon_sweep["epsilon_w"],
        epsilon_sweep["beta_star_deg"],
    )
    power_beta = interpolate_response(
        power_samples,
        power_sweep["p_w"],
        power_sweep["beta_star_deg"],
    )
    tau_area = interpolate_response(
        tau_samples,
        tau_sweep["tau"],
        tau_sweep["candidate_area_m2"],
    )

    experiments = pd.DataFrame(
        {
            "trial": np.arange(1, N_TRIALS + 1),
            "epsilon_w": epsilon_samples,
            "beta_from_epsilon_deg": epsilon_beta,
            "p_w": power_samples,
            "beta_from_p_deg": power_beta,
            "tau": tau_samples,
            "candidate_area_m2": tau_area,
        }
    )
    return experiments


def summarize(experiments):
    configs = [
        ("epsilon_w", "beta_from_epsilon_deg", "基础权重 εw", "最优偏转角（度）"),
        ("p_w", "beta_from_p_deg", "形状参数 pw", "最优偏转角（度）"),
        ("tau", "candidate_area_m2", "近优容差 τ", "候选区域面积（平方米）"),
    ]
    rows = []
    for parameter_col, response_col, parameter_name, response_name in configs:
        parameter = experiments[parameter_col]
        response = experiments[response_col]
        rows.append(
            {
                "parameter": parameter_name,
                "response": response_name,
                "parameter_min": parameter.min(),
                "parameter_max": parameter.max(),
                "response_mean": response.mean(),
                "response_std": response.std(ddof=1),
                "response_cv_pct": response.std(ddof=1) / response.mean() * 100,
                "response_q25": response.quantile(0.25),
                "response_q75": response.quantile(0.75),
                "response_min": response.min(),
                "response_max": response.max(),
                "relative_range_pct": (response.max() - response.min()) / response.mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis):
    axis.set_facecolor("none")
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.55)
    axis.grid(True, axis="x", color=GRID_COLOR, linewidth=0.65, alpha=0.30)
    axis.tick_params(colors=FOREGROUND, labelsize=9.5)
    for spine in axis.spines.values():
        spine.set_color(PRIMARY_BLUE)
        spine.set_linewidth(0.9)
    axis.yaxis.set_label_position("right")


def padded_limits(values, padding_ratio=0.18):
    low = float(np.min(values))
    high = float(np.max(values))
    span = max(high - low, abs(high) * 1e-5, 1e-8)
    return low - span * padding_ratio, high + span * padding_ratio


def plot_results(experiments, summary):
    x = experiments["trial"].to_numpy()
    panels = [
        {
            "title": "基础权重 εw 扰动对最优偏转角的影响（±5%）",
            "values": experiments["beta_from_epsilon_deg"].to_numpy(),
            "ylabel": "最优偏转角（度）",
            "color": PRIMARY_BLUE,
            "summary_row": summary.iloc[0],
        },
        {
            "title": "形状参数 pw 扰动对最优偏转角的影响（±5%）",
            "values": experiments["beta_from_p_deg"].to_numpy(),
            "ylabel": "最优偏转角（度）",
            "color": PRIMARY_RED,
            "summary_row": summary.iloc[1],
        },
        {
            "title": "近优容差 τ 扰动对候选区域面积的影响（±5%）",
            "values": experiments["candidate_area_m2"].to_numpy(),
            "ylabel": "候选区域面积（平方米）",
            "color": TRANSITION,
            "summary_row": summary.iloc[2],
        },
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.2), dpi=200, sharex=True)
    fig.subplots_adjust(left=0.075, right=0.90, top=0.91, bottom=0.09, hspace=0.22)
    # fig.suptitle("三个参数的 100 次局部扰动鲁棒性评估", fontsize=17, color=FOREGROUND, y=0.975)

    for axis, panel in zip(axes, panels):
        style_axis(axis)
        values = panel["values"]
        row = panel["summary_row"]
        q25 = float(row["response_q25"])
        q75 = float(row["response_q75"])
        mean = float(row["response_mean"])

        # 参考图中的固定四分位带
        axis.fill_between(
            x,
            np.full_like(x, q25, dtype=float),
            np.full_like(x, q75, dtype=float),
            color=panel["color"],
            alpha=0.42,
            linewidth=0,
            label="25%–75% 分位区间",
        )
        axis.plot(
            x,
            values,
            color=panel["color"],
            linewidth=1.45,
            marker="o",
            markersize=3.0,
            markerfacecolor=panel["color"],
            markeredgecolor=panel["color"],
            label="单次试验结果",
        )
        axis.axhline(mean, color=FOREGROUND, linewidth=1.15, linestyle="--", alpha=0.85)
        axis.set_title(panel["title"], fontsize=11.5, color=FOREGROUND, pad=9)
        axis.set_ylabel(panel["ylabel"], fontsize=10, color=FOREGROUND, labelpad=13)
        axis.set_ylim(*padded_limits(values))
        axis.set_xlim(-4, 105)
        axis.text(
            0.012,
            0.91,
            f"CV={row['response_cv_pct']:.3f}%｜相对极差={row['relative_range_pct']:.3f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color=FOREGROUND,
        )

    axes[-1].set_xlabel("实验次数", fontsize=11, color=FOREGROUND, labelpad=8)
    axes[-1].set_xticks([0, 20, 40, 60, 80, 100])
    axes[-1].yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}"))
    for axis in axes[:2]:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.4f}"))

    output = OUT / "three_parameter_robustness_evaluation.png"
    save_figure(fig, output, dpi=200, transparent=True)
    plt.close(fig)
    return output


def main():
    experiments = generate_experiments()
    summary = summarize(experiments)

    experiments.to_csv(
        OUT / "three_parameter_robustness_100_trials.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        OUT / "three_parameter_robustness_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output = plot_results(experiments, summary)

    print(summary.to_string(index=False))
    print(f"已生成：{output}")


if __name__ == "__main__":
    main()
