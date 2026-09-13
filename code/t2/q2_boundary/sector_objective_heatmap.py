#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二：在稳健待选区域内模拟加权 DOP 目标函数并绘制热力图。

模型来自 ``docs/问题二求解思路.md`` 第四节、第六节与第七节：

    w(G; ε_w, p_w) = ε_w + (1 − ε_w) · (d(G) / d_★)^{p_w}
    D(P, G)        = sqrt(r₁² + r₂²) / |sin α(P, G)|
    J(P)           = ∬_Ω w(G) D(P, G) dA / ∬_Ω w(G) dA

边界深度 ``d(G)`` 在扇形情形下可解析给出（到两条射线或外圆弧的最短
距离），``d_★`` 取扇形最大内切圆半径 ``R sin δ / (1 + sin δ)``，与文
档中线性规划求得的 ``d_★`` 等价。

当前按用户修正后的强约束绘制待选区域：

    C_1000 = {P : max_{G in Ω} ||P − G|| ≤ 1000}

近优容差区域：

    R_τ = {P ∈ C_1000 : J(P) ≤ (1 + τ) J*}

示例：

    python3 sector_objective_heatmap.py --tau 0.05 --eps-w 0.2 --p-w 2

颜色与字体全部来自 ``tools/visualization`` 的 ``mathmodel_viz.styles``。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

from t2_boundary_common import (
    DELTA_DEG,
    R,
    S1,
    THETA0_DEG,
    apply_theme,
    max_distance_to_sector,
    save_figure,
    wrap_to_pi,
)

# 保守近优接收半径（m）；与候选区域脚本一致。
RECEIVE_RADIUS = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制待选区域内的加权 DOP 目标函数热力图")
    parser.add_argument("--tau", type=float, default=0.05,
                        help="近优相对容差，默认 0.05（即允许比最优值高 5%%）")
    parser.add_argument("--eps-w", type=float, default=0.2,
                        help="边界基础权重 eps_w，默认 0.2（位于 (0,1)）")
    parser.add_argument("--p-w", type=float, default=2.0,
                        help="深度权重指数 p_w，默认 2（≥ 1）")
    parser.add_argument("--nx", type=int, default=641, help="热力图 x 方向网格数")
    parser.add_argument("--ny", type=int, default=641, help="热力图 y 方向网格数")
    parser.add_argument("--nr", type=int, default=90, help="径向高斯积分节点数")
    parser.add_argument("--nt", type=int, default=50, help="角向高斯积分节点数")
    parser.add_argument("--output", type=Path, default=Path("sector_objective_heatmap.png"),
                        help="输出图片路径")
    args = parser.parse_args()
    if args.tau < 0:
        parser.error("--tau 必须大于等于 0")
    if not 0 < args.eps_w < 1:
        parser.error("--eps-w 必须位于 (0,1)")
    if args.p_w < 1:
        parser.error("--p-w 必须大于等于 1")
    return args


def sector_quadrature(nr: int, nt: int, eps_w: float, p_w: float):
    """构造扇形上的二维 Gauss-Legendre 求积点及归一化权重。

    返回:
        gx, gy:                  求积点坐标，shape (nr*nt,)
        normalized_weight:       归一化到总和为 1 的求积权重
        d_star:                  扇形最大内切圆半径（与 LP 解等价）
    """

    xr, wr = np.polynomial.legendre.leggauss(nr)
    xt, wt = np.polynomial.legendre.leggauss(nt)

    theta0 = math.radians(THETA0_DEG)
    delta = math.radians(DELTA_DEG)
    rho = 0.5 * R * (xr + 1.0)
    rho_w = 0.5 * R * wr
    theta = theta0 + delta * xt
    theta_w = delta * wt

    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    wr2, wt2 = np.meshgrid(rho_w, theta_w, indexing="ij")

    gx = rr * np.cos(tt) + S1[0]
    gy = rr * np.sin(tt) + S1[1]

    # 扇形边界由两条射线和外圆弧组成；对内部点，到边界的最短距离取两
    # 个候选的较小值（文档第五节的 LP 形式对扇形等价于此闭式）。
    angular_depth = rr * np.sin(np.maximum(delta - np.abs(tt - theta0), 0.0))
    radial_depth = R - rr
    depth = np.minimum(angular_depth, radial_depth)
    d_star = R * math.sin(delta) / (1.0 + math.sin(delta))
    q = np.clip(depth / d_star, 0.0, 1.0)
    w_depth = eps_w + (1.0 - eps_w) * q ** p_w

    # 极坐标面积元 dA = ρ dρ dθ。
    area_weight = wr2 * wt2 * rr
    weighted_area = w_depth * area_weight
    normalized_weight = weighted_area.ravel() / weighted_area.sum()

    return gx.ravel(), gy.ravel(), normalized_weight, d_star


def objective_on_points(px: np.ndarray, py: np.ndarray,
                        gx: np.ndarray, gy: np.ndarray,
                        normalized_weight: np.ndarray,
                        chunk_size: int = 96) -> np.ndarray:
    """分块计算 ``J(P)``，避免一次构造过大的 ``P × G`` 数组。

    若 ``S1 − P`` 直线穿过扇形内部，则正权重下连续积分发散，直接记为
    ``+inf``，与文档第七节一致。
    """

    result = np.full(px.shape, np.inf, dtype=float)
    theta0 = math.radians(THETA0_DEG)
    delta = math.radians(DELTA_DEG)

    p_angle = np.arctan2(py - S1[1], px - S1[0])
    singular = np.abs(wrap_to_pi(p_angle - theta0)) <= delta
    finite_indices = np.flatnonzero(~singular)

    g1x = S1[0] - gx
    g1y = S1[1] - gy
    r1 = np.hypot(g1x, g1y)

    for start in range(0, finite_indices.size, chunk_size):
        idx = finite_indices[start:start + chunk_size]
        p2x = px[idx, None] - gx[None, :]
        p2y = py[idx, None] - gy[None, :]
        r2 = np.hypot(p2x, p2y)

        cross = np.abs(g1x[None, :] * p2y - g1y[None, :] * p2x)
        denom = np.maximum(r1[None, :] * r2, 1e-14)
        sin_alpha = cross / denom

        # 对非共线 P，求积节点仍可能极接近奇异线；保留其大 DOP 值而不
        # 是裁平，与文档要求"显式检查奇异性"一致。
        dop = np.sqrt(r1[None, :] ** 2 + r2 ** 2) / np.maximum(sin_alpha, 1e-12)
        result[idx] = dop @ normalized_weight

    return result


def main() -> None:
    args = parse_args()
    theme = apply_theme(figure_dpi=180, savefig_dpi=180)
    t0 = time.time()

    # 覆盖当前严格 1000 m 约束下的完整候选区域。
    x = np.linspace(120.0, 1380.0, args.nx)
    y = np.linspace(-780.0, 780.0, args.ny)
    X, Y = np.meshgrid(x, y)
    receive_feasible = max_distance_to_sector(X, Y) <= RECEIVE_RADIUS

    gx, gy, integral_weight, d_star = sector_quadrature(
        args.nr, args.nt, args.eps_w, args.p_w
    )

    px = X[receive_feasible]
    py = Y[receive_feasible]
    objective_values = objective_on_points(px, py, gx, gy, integral_weight)

    J = np.full(X.shape, np.nan, dtype=float)
    J[receive_feasible] = objective_values
    finite_mask = receive_feasible & np.isfinite(J)
    if not finite_mask.any():
        raise RuntimeError("当前参数下没有有限目标函数值，请检查几何范围。")

    finite_values = J[finite_mask]
    j_star = float(np.min(finite_values))
    min_ij = np.unravel_index(np.nanargmin(J), J.shape)
    p_star = np.array([X[min_ij], Y[min_ij]])
    threshold = (1.0 + args.tau) * j_star
    near_optimal = finite_mask & (J <= threshold)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    near_area = near_optimal.sum() * dx * dy

    # 非线性色标：把约 60% 的色带分配给 2000–4000，使主要近优区梯度更清楚。
    # 4000 以上仍保留颜色，但压缩到色带顶部，避免极大 DOP 吞掉低值区分度。
    color_upper = max(float(np.quantile(finite_values, 0.95)), 4001.0)
    scale_x = np.array([j_star, 2000.0, 4000.0, color_upper])
    scale_x = np.maximum.accumulate(scale_x)
    scale_x[1:] = np.maximum(scale_x[1:], scale_x[:-1] + 1e-6)
    scale_y = np.array([0.0, 0.18, 0.78, 1.0])

    def color_forward(values):
        return np.interp(values, scale_x, scale_y)

    def color_inverse(values):
        return np.interp(values, scale_y, scale_x)

    color_norm = FuncNorm(
        (color_forward, color_inverse), vmin=j_star, vmax=color_upper, clip=True
    )
    J_plot = np.ma.masked_where(~finite_mask, J)

    # 自定义 RGBA 色阶：低值端使用主题蓝且 alpha=0（接近不可见），高值端使用
    # 主题红且 alpha=0.75；中间按 RGBA 线性插值，alpha 随色阶同步提升，使得低 J
    # 区域自然融入背景，高 J 区域显著突出。
    objective_cmap = LinearSegmentedColormap.from_list(
        "sector-objective-blue-to-red",
        [
            theme.color("red", 0.3),
            theme.color("blue", 1.0),
        ],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(13.8, 8.4), dpi=180)
    ax.set_aspect("equal", adjustable="box")

    heat = ax.pcolormesh(
        X, Y, J_plot, shading="auto", cmap=objective_cmap,
        norm=color_norm, zorder=1
    )

    # 1. 近优区域填充：使用主题橙填充
    ax.contourf(
        X, Y, near_optimal.astype(float), levels=[0.5, 1.5],
        colors=[theme.transition], alpha=0.75, zorder=4
    )

    # 2. 边界线渐变色：基于 |y| 从 theme.blue(1.0) 渐变到 theme.red(0.3)
    # 使用 LineCollection + colormap 实现平滑渐变
    boundary_color_start = theme.color("blue", 1.0)
    boundary_color_end = theme.color("red", 0.3)

    # 创建边界线色阶
    boundary_cmap = LinearSegmentedColormap.from_list(
        "boundary-blue-to-red",
        [boundary_color_start, boundary_color_end],
        N=256,
    )

    # 提取边界线顶点
    boundary_contour = ax.contour(
        X, Y, receive_feasible.astype(float), levels=[0.5],
        colors=["white"], linewidths=0  # 隐藏白色轮廓
    )

    all_segments = []
    all_t_values = []

    for seg in boundary_contour.allsegs[0]:
        if len(seg) < 2:
            continue
        vertices = np.array(seg)
        # 按 |y| 归一化到 [0, 1]
        y_abs = np.abs(vertices[:, 1])
        y_min, y_max = y_abs.min(), y_abs.max()
        if y_max > y_min:
            t = (y_abs - y_min) / (y_max - y_min)
        else:
            t = np.zeros_like(y_abs)
        # 将线段和对应的 t 值收集
        for i in range(len(vertices) - 1):
            all_segments.append(vertices[i:i+2])
            all_t_values.append((t[i] + t[i + 1]) / 2.0)

    if all_segments:
        # 使用 LineCollection 配合 colormap 绘制平滑渐变边界
        lc = LineCollection(all_segments, cmap=boundary_cmap, linewidths=1.0, zorder=5)
        lc.set_array(np.array(all_t_values))
        lc.set_clim(0.0, 1.0)
        ax.add_collection(lc)

    # 近优区域边界线（橙线）
    ax.contour(
        X, Y, near_optimal.astype(float), levels=[0.5],
        colors=[theme.transition], linewidths=2.0, zorder=6
    )

    # 叠加蓝色目标扇形，仅显示几何关系。
    sector = Wedge(
        tuple(S1), R, THETA0_DEG - DELTA_DEG, THETA0_DEG + DELTA_DEG,
        facecolor=theme.color("blue", 0.18),
        edgecolor=theme.primary_blue,
        linewidth=1.8, alpha=0.55, zorder=3
    )
    ax.add_patch(sector)
    ax.scatter(*S1, s=42, color=theme.foreground, zorder=6)
    ax.text(S1[0] - 30, S1[1] - 70, "S1",
            fontsize=11, weight="bold", ha="center")
    ax.scatter(*p_star, marker="*", s=190, color=theme.primary_red,
               edgecolor="white", linewidth=0.8, zorder=7)
    ax.text(p_star[0] + 28, p_star[1] + 38, "P*", fontsize=11,
            color=theme.primary_red, weight="bold", zorder=7)

    cbar = fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.025)
    detailed_ticks = [j_star, 1800, 2000, 2250, 2500, 2750, 3000, 3250,
                      3500, 3750, 4000, 6000, 10000, color_upper]
    detailed_ticks = sorted({round(v, 6) for v in detailed_ticks
                             if j_star <= v <= color_upper})
    cbar.set_ticks(detailed_ticks)
    cbar.set_label("加权平均 DOP 目标函数 J(P)（2000—4000 加密色阶）",
                   fontsize=10)

    # 参数文字栏（右上角）：ε_w, p_w, τ 为一行；P*, J* 为一行
    param_text = (
        f"ε_w = {args.eps_w:g}    p_w = {args.p_w:g}    τ = {args.tau:g}\n"
        f"P* = ({p_star[0]:.1f}, {p_star[1]:.1f}) m    "
        f"J* = {j_star:.3f}"
    )
    ax.text(
        1340, -735, param_text, fontsize=10,
        va="bottom", ha="right", linespacing=2.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=theme.color("transition", 0.45), alpha=0.75)
    )

    # 附加信息文字栏（右中）：近优阈值与面积
    sub_text = (
        f"近优阈值 (1+τ)J* = {threshold:.3f}\n"
        f"近优区域面积 ≈ {near_area:,.0f} m²"
    )
    ax.text(
        1340, 730, sub_text, fontsize=10,
        va="top", ha="right", linespacing=2.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=theme.color("transition", 0.45), alpha=0.75)
    )

    # 图例文字栏（左下角）：红线、橙线、白色缺口的说明
    legend_text = (
        "━ 红线：严格 1000 m 接收待选区域\n"
        "━ 橙线：容差 τ 对应的近优区域\n"
        "━ 白色缺口：共线导致积分发散"
    )
    ax.text(
        135, -750, legend_text, fontsize=10,
        va="bottom", ha="left", linespacing=2.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=theme.color("transition", 0.45), alpha=0.75)
    )

    ax.set_xlim(120, 1380)
    ax.set_ylim(-780, 780)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(
        "严格待选区域内的加权 DOP 目标函数及近优区域热力图",
        fontsize=15, weight="bold", pad=13
    )
    ax.grid(True, linewidth=0.55, alpha=0.55)
    ax.axhline(0, color=theme.color("transition", 0.55),
               linewidth=0.8, zorder=0)

    save_figure(fig, args.output)
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"eps_w={args.eps_w:g}, p_w={args.p_w:g}, tau={args.tau:g}")
    print(f"d_star={d_star:.6f} m")
    print(f"J_star={j_star:.6f}")
    print(f"P_star=({p_star[0]:.3f}, {p_star[1]:.3f}) m")
    print(f"near_optimal_threshold={threshold:.6f}")
    print(f"near_optimal_area≈{near_area:.3f} m^2")
    print(f"finite_grid_points={finite_mask.sum()}, "
          f"receive_grid_points={receive_feasible.sum()}")
    print(f"elapsed={elapsed:.2f} s")
    print(f"图像已保存: {args.output.resolve()}")


if __name__ == "__main__":
    main()