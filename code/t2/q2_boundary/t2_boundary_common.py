#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二候选区域可视化脚本的共用工具。

集中以下几类共享逻辑：

1. 颜色与字体：统一调用 ``mathmodel_viz.styles.configure_matplotlib``，
   后续脚本不再各自复制 ``rcParams``、中文字体和 ``savefig`` 流程；
2. 几何参数：第一次成功接收半径 ``R``、单侧测角误差 ``DELTA_DEG``、
   干扰源最小接收半径 ``RECEIVE_RADIUS`` 与第一检测点 ``S1``、中心示
   向度 ``THETA0_DEG`` 与设计文档 ``docs/问题二求解思路.md`` 保持一致；
3. 解析判定：``max_distance_to_sector`` 给出 ``max_{G in Omega} ||P-G||``
   的闭式结果，使网格仅承担可视化、不参与"对所有可能目标点"的约束
   近似。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# 颜色与字体接入 ``tools/visualization`` 的全局配置。
_ROOT = Path(__file__).resolve().parents[2]
_VIZ_SRC = _ROOT / "tools" / "visualization" / "src"
if str(_VIZ_SRC) not in sys.path:
    sys.path.insert(0, str(_VIZ_SRC))

from mathmodel_viz.styles import (  # noqa: E402
    VisualizationTheme,
    configure_matplotlib,
    get_theme,
    save_figure,
)

# 复用入口供其他脚本直接 ``from t2_boundary_common import ...`` 使用。
__all__ = [
    "R",
    "DELTA_DEG",
    "RECEIVE_RADIUS",
    "S1",
    "THETA0_DEG",
    "apply_theme",
    "wrap_to_pi",
    "max_distance_to_sector",
]


# ---- 几何参数（参考 docs/问题二求解思路.md 第二节） -------------------------
R: float = 1500.0                  # 第一次成功接收对应的最大半径（m）
DELTA_DEG: float = 1.0             # 单侧示向度误差（°）；总张角 2°
RECEIVE_RADIUS: float = 1000.0     # 干扰源最小有效接收半径（m）
S1: np.ndarray = np.array([0.0, 0.0])  # 平移/旋转后的第一检测点
THETA0_DEG: float = 0.0            # 示向度中心方向；可按题目替换为真实值


def apply_theme(figure_dpi: int = 180, savefig_dpi: int = 180) -> VisualizationTheme:
    """统一接入项目可视化主题，并返回当前主题对象。

    建议在 ``scratch/t2_boundary`` 下任一绘图脚本入口处调用一次。
    180 dpi 与既有扇形图保持一致；论文正式图可在调用处临时改成
    ``savefig_dpi=300``。
    """

    configure_matplotlib(figure_dpi=figure_dpi, savefig_dpi=savefig_dpi)
    return get_theme()


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """把弧度数组归一化到 ``[-pi, pi)``。"""

    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def max_distance_to_sector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """闭式计算 ``max_{G in Omega} ||P - G||``，支持 NumPy 数组输入。

    推导：固定方向 ``theta`` 后，距离关于 ``rho`` 是单调的，最大值要么在
    ``rho=0``、要么在 ``rho=R``。外圆弧 ``rho=R`` 上最大值只可能出现在
    两个端点，或者出现在 ``-P`` 方向落入扇形角区间时对应的反向圆弧点。
    网格只用来可视化这个解析函数，并不替代"对所有目标点"的判定。
    """

    theta0 = math.radians(THETA0_DEG)
    delta = math.radians(DELTA_DEG)
    theta_minus = theta0 - delta
    theta_plus = theta0 + delta

    px = x - S1[0]
    py = y - S1[1]
    p_norm = np.hypot(px, py)

    d_vertex = p_norm

    gx_minus = R * math.cos(theta_minus)
    gy_minus = R * math.sin(theta_minus)
    gx_plus = R * math.cos(theta_plus)
    gy_plus = R * math.sin(theta_plus)
    d_minus = np.hypot(px - gx_minus, py - gy_minus)
    d_plus = np.hypot(px - gx_plus, py - gy_plus)

    opposite_angle = np.arctan2(-py, -px)
    opposite_inside = np.abs(wrap_to_pi(opposite_angle - theta0)) <= delta
    d_opposite = np.where(opposite_inside, p_norm + R, -np.inf)

    return np.maximum.reduce([d_vertex, d_minus, d_plus, d_opposite])


def figure_save_kwargs(dpi: int) -> Tuple[int, bool]:
    """``save_figure`` 的 ``dpi``/``transparent`` 组合，避免各处重复。"""

    return dpi, False