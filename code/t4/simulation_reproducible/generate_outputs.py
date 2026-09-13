#!/usr/bin/env python3
"""读取本项目仿真数据生成每题一表三图，支持先重新运行实验。"""
from __future__ import annotations

import csv
import json
import argparse
import sys
import warnings
from pathlib import Path

# 让 ``mathmodel_viz`` 可被本脚本直接导入（无需预先 ``pip install -e``）。
_VIZ_SRC = Path(__file__).resolve().parents[3] / "tools" / "visualization" / "src"
if str(_VIZ_SRC) not in sys.path:
    sys.path.insert(0, str(_VIZ_SRC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from mathmodel_viz import (
    AUXILIARY,
    FONT_CJK,
    FONT_LATIN,
    PRIMARY_BLUE,
    PRIMARY_RED,
    TRANSITION,
    blend_colors,
    configure_matplotlib,
    configure_theme,
    get_theme,
    rgba,
    rgba_css,
)

PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "output"

# 主题：与 ``tools/visualization`` 全局主题保持一致。
configure_theme(
    primary_red="#B41B2",
    primary_blue="#266AA",
    transition="#EF8F67",
    auxiliary="#FFE181",
)
configure_matplotlib()
THEME = get_theme()

# 衍生色：用于构造与旧脚本同等语义的蓝色渐变条 / 行底色等。
BLUE_DARK = blend_colors(THEME.primary_blue, "#000000", 0.35)
BLUE_MID = blend_colors(THEME.primary_blue, THEME.auxiliary, 0.45)
BLUE_LIGHT = blend_colors(THEME.primary_blue, "#FFFFFF", 0.40)
BLUE_PALE = blend_colors(THEME.primary_blue, "#FFFFFF", 0.65)
FOREGROUND = THEME.foreground
# rcParams 仅接受六位十六进制颜色；从主题主蓝向白色大比例融合，得到近白底色。
BACKGROUND = blend_colors(THEME.primary_blue, "#FFFFFF", 0.97)
ROW_ALT = blend_colors(THEME.primary_blue, "#FFFFFF", 0.93)
SOFT_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "soft_blue_transition",
    [BLUE_DARK, blend_colors(THEME.primary_blue, "#FFFFFF", 0.20),
     BLUE_MID, blend_colors(THEME.primary_blue, "#FFFFFF", 0.30),
     BLUE_LIGHT, BLUE_PALE],
)

# 优先使用系统中文字体，兼容Linux、Windows、macOS。
for font_path in font_manager.findSystemFonts():
    if any(token in Path(font_path).name.lower() for token in ("notosanscjk", "simhei", "msyh", "pingfang", "wqy")):
        try:
            font_manager.fontManager.addfont(font_path)
        except (OSError, RuntimeError):
            continue
FONT_FAMILIES = [FONT_CJK, "Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Zen Hei"]
FONT_FAMILIES = [FONT_LATIN] + [name for name in FONT_FAMILIES if name in {font.name for font in font_manager.fontManager.ttflist}]
if not FONT_FAMILIES or FONT_FAMILIES == [FONT_LATIN]:
    warnings.warn(f"未检测到{FONT_CJK}等中文字体，请安装以正常显示中文。")

plt.rcParams["font.family"] = FONT_FAMILIES
plt.rcParams["font.sans-serif"] = FONT_FAMILIES
plt.rcParams["font.serif"] = FONT_FAMILIES
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = BACKGROUND
plt.rcParams["axes.facecolor"] = BACKGROUND
plt.rcParams["text.color"] = FOREGROUND
plt.rcParams["axes.labelcolor"] = FOREGROUND
plt.rcParams["axes.titlecolor"] = FOREGROUND
plt.rcParams["xtick.color"] = FOREGROUND
plt.rcParams["ytick.color"] = FOREGROUND

TITLE_FONT = FontProperties(family=FONT_FAMILIES, size=15, weight="bold")
HEADER_FONT = FontProperties(family=FONT_FAMILIES, size=10, weight="bold")
BODY_FONT = FontProperties(family=FONT_FAMILIES, size=10)


TIME_FIELDS = ("move_time_s", "switch_time_s", "measure_time_s", "clear_action_time_s")


def load_results(problem):
    """统一从逐案例CSV汇总，避免旧summary与新cases混用。"""
    from runner import summarize
    path = PROJECT / problem / "results" / "cases.csv"
    if not path.exists():
        raise ValueError(f"缺少{path}；请先运行 python generate_outputs.py --rerun")
    with path.open(encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        raise ValueError(f"{path}没有案例数据")
    rows = []
    for record in raw:
        row = dict(record)
        for key in ("source_count", "cleared_count", "measure_count"):
            row[key] = int(record[key])
        for key in ("single_source_average_clear_time_s", "per_case_clear_time_s", "program_runtime_s"):
            row[key] = float(record[key]) if record[key] else None
        if record["all_cleared"].lower() not in ("true", "false", "1", "0"):
            raise ValueError(f"非法成功标记：{record['all_cleared']}")
        row["all_cleared"] = record["all_cleared"].lower() in ("true", "1")
        rows.append(row)
    base = summarize(rows)
    summary = dict(zip(
        ("case_count", "all_cleared_success_rate", "total_sources", "mean_average_clear_time_s",
         "mean_total_time_s", "mean_program_runtime_s", "mean_measure_count"),
        (base[key] for key in ("随机案例数", "全部清除成功率", "干扰源总数",
         "单源平均定位清除时间_秒每个", "每局平均定位清除时间_秒每局",
         "每局平均程序运行时间_秒每局", "每局平均检测次数_次每局"))))
    if all(all(row.get(key) not in (None, "") for key in TIME_FIELDS) for row in rows):
        for row in rows:
            if abs(sum(float(row[key]) for key in TIME_FIELDS) - row["per_case_clear_time_s"]) > 1e-5:
                raise ValueError(f"{path} case={row['case']} 时间构成不一致")
        summary.update({"mean_" + key: float(np.mean([float(row[key]) for row in rows]))
                        for key in TIME_FIELDS})
    else:
        warnings.warn(f"{problem}旧CSV缺少时间构成字段；图2将注明数据缺失。使用 --rerun 补全。")
    total_times = [float(row["per_case_clear_time_s"]) for row in rows if row.get("per_case_clear_time_s") is not None]
    summary["median_total_time_s"] = float(np.median(total_times)) if total_times else None
    return summary, rows


def format_seconds(value):
    return "无有效数据" if value is None else f"{value:.2f}"


def summary_rows(summary):
    return [
        {"指标": "随机案例数", "结果": f'{int(summary["case_count"]):,}', "单位": "组"},
        {"指标": "全部清除成功率", "结果": f'{summary["all_cleared_success_rate"]:.2%}', "单位": "%"},
        {"指标": "干扰源总数", "结果": f'{int(summary["total_sources"]):,}', "单位": "个"},
        {"指标": "单源平均定位清除时间", "结果": format_seconds(summary["mean_average_clear_time_s"]), "单位": "秒/个"},
        {"指标": "每局平均定位清除时间", "结果": f'{summary["mean_total_time_s"]:.2f}', "单位": "秒/局"},
        {"指标": "每局平均程序运行时间", "结果": f'{summary["mean_program_runtime_s"]:.2f}', "单位": "秒/局"},
        {"指标": "每局平均检测次数", "结果": f'{summary["mean_measure_count"]:.2f}', "单位": "次/局"},
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["指标", "结果", "单位"])
        writer.writeheader()
        writer.writerows(rows)


def render_table(rows, title, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=220, constrained_layout=True)
    ax.axis("off")
    values = [[row["指标"], row["结果"], row["单位"]] for row in rows]
    table = ax.table(
        cellText=values,
        colLabels=["指标", "结果", "单位"],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.03, 0.03, 0.94, 0.83],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)
    widths = [0.57, 0.28, 0.15]
    for col, width in enumerate(widths):
        for row in range(len(rows) + 1):
            table[(row, col)].set_width(width)
    for col in range(3):
        cell = table[(0, col)]
        cell.set_facecolor(PRIMARY_BLUE)
        cell.set_edgecolor(PRIMARY_BLUE)
        cell.get_text().set_color("white")
        cell.get_text().set_fontproperties(HEADER_FONT)
    for row in range(1, len(rows) + 1):
        for col in range(3):
            cell = table[(row, col)]
            cell.set_facecolor(BACKGROUND if row % 2 else ROW_ALT)
            cell.set_edgecolor(to_rgba(PRIMARY_BLUE, 0.35))
            cell.set_linewidth(0.6)
            cell.get_text().set_fontproperties(BODY_FONT)
            if col == 0:
                cell.get_text().set_ha("left")
            elif col == 1:
                cell.get_text().set_color(PRIMARY_BLUE)
                cell.get_text().set_weight("bold")
    ax.set_title(title, fontproperties=TITLE_FONT, pad=16)
    ax.plot([0.03, 0.97], [0.89, 0.89], transform=ax.transAxes,
            color=PRIMARY_BLUE, linewidth=2.2, solid_capstyle="round")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def style_axes(ax):
    ax.grid(True, axis="x", color=BLUE_LIGHT, alpha=0.32, linewidth=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(to_rgba(PRIMARY_BLUE, 0.55))
    ax.spines["bottom"].set_color(to_rgba(PRIMARY_BLUE, 0.55))


def render_time_composition(problem_title, summary, path):
    names = ["移动", "频道切换", "检测", "精确定位与清除"]
    values = [
        summary["mean_move_time_s"], summary["mean_switch_time_s"],
        summary["mean_measure_time_s"], summary["mean_clear_action_time_s"],
    ]
    total = sum(values)
    ratios = [value / total for value in values] if total > 0 else [0.0] * len(values)
    boundaries = np.concatenate(([0.0], np.cumsum(ratios)))
    median_total = summary.get("median_total_time_s")

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    fig.subplots_adjust(bottom=0.35, top=0.84, left=0.10, right=0.97)

    # 整根构成条使用连续蓝色渐变，分段位置不画边线，避免生硬的白色切割。
    gradient = np.linspace(0, 1, 1600).reshape(1, -1)
    ax.imshow(gradient, extent=[0, 1, -0.31, 0.31], aspect="auto",
              cmap=SOFT_BLUE_CMAP, interpolation="bicubic", zorder=2)

    legend_handles = []
    for index, (name, value, ratio) in enumerate(zip(names, values, ratios)):
        left, right = boundaries[index], boundaries[index + 1]
        midpoint = (left + right) / 2
        representative_color = SOFT_BLUE_CMAP(midpoint)
        legend_handles.append(Patch(facecolor=representative_color, edgecolor="none",
                                    label=f"{name} {value:,.1f}秒"))
        if ratio >= 0.06:
            text_color = "white" if midpoint < 0.68 else FOREGROUND

    # 给整根构成条加上总平均 / 总中位时间的文字说明与箭头指向条形右端。
    if total > 0:
        annotation_lines = [f"总平均时间 {total:,.1f}秒/局"]
        if median_total is not None:
            annotation_lines.append(f"总中位时间 {median_total:,.1f}秒/局")
        annotation_text = "\n".join(annotation_lines)
        ax.annotate(
            annotation_text,
            xy=(1.0, 0.0),
            xytext=(0.985, 0.55),
            ha="right", va="bottom",
            color=FOREGROUND,
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                facecolor=BACKGROUND,
                edgecolor=PRIMARY_BLUE,
                boxstyle="round,pad=0.45",
                linewidth=1.0,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=PRIMARY_BLUE,
                lw=1.5,
                shrinkA=0,
                shrinkB=4,
                connectionstyle="arc3,rad=-0.25",
            ),
            zorder=6,
        )

    ax.set_yticks([0], [problem_title])
    ax.set_ylim(-0.62, 0.62)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:.0%}"))
    ax.set_xlabel("平均总任务时间占比", fontsize=11, labelpad=10)
    style_axes(ax)
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.52),
              ncol=4, frameon=False)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_distribution(rows, key, title, xlabel, path):
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    values = np.array(
        [row[key] for row in rows if row[key] is not None],
        dtype=float
    )
    values = values[np.isfinite(values)]

    # 配色：全部从 ``THEME`` 派生，与全局主题色保持一致。
    background = BACKGROUND
    text_color = FOREGROUND
    # matplotlib 3.10 的 ``edgecolor``/``gridcolor`` 等字段只接受十六进制字符串，
    # 因此对需要降低饱和度的颜色，先用 ``blend_colors`` 与背景色融合得到同色系的淡色。
    axis_color = blend_colors(THEME.primary_blue, BACKGROUND, 0.45)
    grid_color = blend_colors(THEME.transition, BACKGROUND, 0.82)
    hist_edge = blend_colors(FOREGROUND, BACKGROUND, 0.55)

    cmap = LinearSegmentedColormap.from_list(
        "theme_blue_to_red",
        [THEME.primary_blue, THEME.transition, THEME.primary_red],
    )

    fig, ax = plt.subplots(
        figsize=(12, 6.5),
        dpi=220,
        constrained_layout=True
    )
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    if values.size:
        # 18个分组；增大此数值会让柱子更细
        counts, edges, patches = ax.hist(
            values,
            bins=18,
            edgecolor=hist_edge,
            linewidth=0.8,
            alpha=0.95,
            zorder=2
        )

        # 按柱子从左到右的位置分配渐变色
        # 按每根柱子在累计频数中的中间位置分配颜色
        # 避免少量极端值占用大部分渐变范围
        total = counts.sum()

        if total > 0:
            positions = (np.cumsum(counts) - 0.5 * counts) / total

            for patch, position in zip(patches, positions):
                patch.set_facecolor(cmap(float(position)))

        # 均值：主红实线
        ax.axvline(
            values.mean(),
            color=THEME.primary_red,
            linewidth=2.2,
            linestyle="-",
            zorder=4
        )

        # 中位数：前景色虚线
        ax.axvline(
            np.median(values),
            color=text_color,
            linewidth=1.8,
            linestyle="--",
            zorder=4
        )

        # 给均值/中位数线加上文字标注与箭头。
        # 标签放在直方图上方，靠 bbox+facecolor 保证不被柱子遮挡；
        # 当均值与中位数接近时，分别放在不同高度，避免重叠。
        peak_count = float(max(counts))
        mean_val = float(values.mean())
        median_val = float(np.median(values))

        ax.annotate(
            f"平均值 {mean_val:.1f}秒",
            xy=(mean_val, peak_count * 0.04),
            xytext=(mean_val, peak_count * 0.92),
            ha="center", va="center",
            color=THEME.primary_red,
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(
                facecolor=background,
                edgecolor=THEME.primary_red,
                boxstyle="round,pad=0.32",
                linewidth=1.0,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=THEME.primary_red,
                lw=1.5,
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=6,
        )

        ax.annotate(
            f"中位数 {median_val:.1f}秒",
            xy=(median_val, peak_count * 0.04),
            xytext=(median_val, peak_count * 0.74),
            ha="center", va="center",
            color=text_color,
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(
                facecolor=background,
                edgecolor=text_color,
                boxstyle="round,pad=0.32",
                linewidth=1.0,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color=text_color,
                lw=1.4,
                shrinkA=0,
                shrinkB=4,
            ),
            zorder=6,
        )

        ax.set_ylim(0, peak_count * 1.10)
        ax.margins(x=0.04)

    else:
        ax.text(
            0.5, 0.5, "无有效数据",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=text_color
        )

    # 不设置标题，只保留坐标轴名称
    ax.set_xlabel(
        xlabel,
        fontsize=14,
        color=text_color,
        labelpad=14
    )
    ax.set_ylabel(
        "案例数",
        fontsize=14,
        color=text_color,
        labelpad=12
    )

    # 横向、纵向浅过渡色网格
    ax.set_axisbelow(True)
    ax.grid(
        True,
        axis="both",
        color=grid_color,
        linewidth=0.7,
        alpha=0.8
    )

    # 仅保留左侧和底部边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_color)
        ax.spines[side].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        colors=text_color,
        labelsize=12,
        length=5
    )

    # 案例数刻度使用整数
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(
        path,
        bbox_inches="tight",
        facecolor=background
    )
    plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(description="读取本项目Q3/Q4结果并生成每题一表三图")
    parser.add_argument("--rerun", action="store_true", help="先重新仿真，再生成图表")
    parser.add_argument("--cases", type=int, default=300, help="重新仿真的案例数，默认300")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=2026091200)
    parser.add_argument("--problem", choices=("Q3", "Q4", "both"), default="both")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    problems = ("Q3", "Q4") if args.problem == "both" else (args.problem,)
    if args.rerun:
        from runner import run_experiment
        for problem in problems:
            run_experiment(PROJECT / problem, problem, args.cases, args.seed_start, args.workers)
    loaded = {problem: load_results(problem) for problem in problems}
    for problem, (summary, cases) in loaded.items():
        title = "问题3" if problem == "Q3" else "问题4"
        directory = args.output / problem
        directory.mkdir(parents=True, exist_ok=True)
        rows = summary_rows(summary)
        write_csv(directory / f"{title}仿真实验汇总表.csv", rows)
        render_table(rows, f"{title}仿真实验汇总（{summary['case_count']}组随机案例）",
                     directory / f"{title}仿真实验汇总表.png")
        render_distribution(cases, "single_source_average_clear_time_s",
                            f"{title}单源平均定位清除时间分布", "单源平均定位清除时间（秒/个）",
                            directory / f"图1_{title}平均定位清除时间分布.png")
        render_distribution(cases, "program_runtime_s", f"{title}程序运行时间分布",
                            "程序运行时间（秒/局）", directory / f"图3_{title}程序运行时间分布.png")
        time_path = directory / f"图2_{title}任务时间构成.png"
        if "mean_move_time_s" in summary:
            render_time_composition(title, summary, time_path)
        else:
            fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
            ax.axis("off")
            ax.text(.5, .5, "旧结果缺少时间构成明细\n请运行 python generate_outputs.py --rerun 后生成本图",
                    ha="center", va="center", transform=ax.transAxes)
            fig.savefig(time_path, bbox_inches="tight")
            plt.close(fig)
        print(f"{problem}: {summary['case_count']}组，图表已生成至 {directory}", flush=True)
    manifest = {
        "交付结构": "每题汇总表CSV及PNG，另有三张统计图",
        "数据来源": {p: str(PROJECT / p / "results" / "cases.csv") for p in problems},
        "案例数": {p: loaded[p][0]["case_count"] for p in problems},
        "时间构成完整": {p: "mean_move_time_s" in loaded[p][0] for p in problems},
        "时间核算": "移动=距离/5；切换=次数；检测=5×次数；清除=5×成功次数+3×失败次数",
        "平均口径": "单源平均时间先按每局已清除源数量计算，再对局取算术平均；无已清除源时记空值并排除。",
    }
    with (args.output / "输出说明.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
