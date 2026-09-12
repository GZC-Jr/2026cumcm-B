# 第二问连续鲁棒选址求解器

`solve_q2.py` 实现 `docs/问题二求解思路.md` 中的第二问模型：已知第一检测点及其示向度后，在连续目标可行域上积分位置误差型 DOP，寻找满足全域接收约束的第二检测点，并输出近优候选区域与敏感性分析结果。

## 运行

从工作区根目录运行演示：

```powershell
python code/t2/solve_q2.py --demo --no-global --output outputs/t2_demo.json
```

也可以使用 JSON 输入：

```json
{
  "s1": {"x": 0, "y": 0},
  "svd_deg": 32,
  "angle_error_deg": 1,
  "circle_sides": 360,
  "epsilon_w": 0.2,
  "p_w": 2,
  "tau": 0.05,
  "quadrature_order": 6,
  "validation_quadrature_order": 8,
  "candidate_grid": 101,
  "use_global_search": true,
  "global_maxiter": 24,
  "global_popsize": 7,
  "local_maxiter": 240,
  "feasibility_tolerance_m": 1e-6,
  "angle_sin_tolerance": 1e-10
}
```

```powershell
python code/t2/solve_q2.py --input data/question2.json --output outputs/t2_result.json
```

输入也接受 `detector` 代替 `s1`，以及 `bearing_deg`、`theta_deg`、`theta` 或 `angle_deg` 代替 `svd_deg`。JSON 中的配置字段与 `Q2Config` 同名；命令行显式给出的参数优先。`--global-search` 与 `--no-global` 是互斥的可选覆盖项，未给出时保留 JSON 配置。

## 数值模型

- 以目标圆和第一检测点接收圆的外接正多边形，加上示向度误差楔形，构造连续半平面区域。外接近似保持目标区域的保守性；求积节点只是连续积分的数值节点，不是离散目标代表点。
- 用三角剖分和 Gauss-Legendre/Duffy 张量求积计算区域积分。`quadrature_order` 控制主计算阶数，`validation_quadrature_order` 用于在最优点复核积分稳定性。
- 权重为 `epsilon_w + (1-epsilon_w) * (d/d_star)**p_w`，其中 `d` 是到全部半平面边界的最小深度，`d_star` 由最大内切圆线性规划得到。
- 接收约束保持半无限形式 `H(P)=sup_G(||P-G||^2-max(1000^2,||G-S1||^2)) <= 0`。代码用凸多边形与半径 1000 米圆的边界交点、顶点和支持点构造几何分离候选，避免把目标域离散成有限代表点。
- 若 `S1-P` 直线与目标多边形相交，或求积节点的交会角低于 `angle_sin_tolerance`，则 `J(P)=+infinity`，显式排除共线奇异位置。
- 优化先可选用差分进化进行全局探索，再用多初值 SLSQP 处理接收约束。公共 `objective()` 在共线位置严格返回 `+inf`；优化器内部使用有限惩罚包装，以避免有限差分遇到无穷值。结果是数值最优参考点，不宣称解析全局最优。
- 候选区域按 `J(P) <= (1+tau) J_star` 定义，并用搜索包围盒内的规则网格单元中心近似。输出中的 `area`、连通分量和距离范围均是该网格近似量；最优点单独报告，不强行塞入不可行或奇异网格单元。

## 主要参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `target_radius` | 1800 | 目标先验圆半径（米） |
| `min_receive_radius` / `max_receive_radius` | 1000 / 1500 | 接收半径上下界（米）；强约束使用前者与第一测量距离的最大值 |
| `angle_error_deg` | 1 | 第一次示向度误差半宽（度） |
| `circle_sides` | 360 | 两个圆的外接正多边形边数 |
| `epsilon_w` / `p_w` | 0.2 / 2 | 边界基础权重与深度权重幂次 |
| `tau` | 0.05 | 近优水平集的相对容差 |
| `quadrature_order` | 6 | 每个三角形的 Gauss-Duffy 一维阶数 |
| `candidate_grid` | 101 | 候选区域每个坐标方向的网格单元数 |
| `use_global_search` | `true` | 是否运行差分进化全局探索 |
| `global_maxiter` / `global_popsize` | 24 / 7 | 全局搜索预算 |
| `local_maxiter` | 240 | 每个 SLSQP 初值的最大迭代数 |
| `feasibility_tolerance_m` | 1e-6 | 接收约束的距离尺度数值容差；实际 `H` 容差为 `feasibility_tolerance_m * search_radius`（`H` 的量纲为平方距离） |
| `angle_sin_tolerance` | 1e-10 | 求积节点的最小正弦保护阈值 |

最终实验建议逐步提高 `circle_sides`、`quadrature_order` 与优化预算，并比较 `quadrature.check_at_optimum`、最优点和候选面积的变化。输出同时给出 `strict_feasible`（`H<=0`）与 `feasible_with_tolerance`（`H` 不超过数值容差），不能把后者当作严格解析可行性。

## 输出字段

结果 JSON 包含 `target_region`、`reception_constraint`、`quadrature`、`optimum` 和 `candidate_region`。如指定 `--sensitivity`，还会对 `epsilon_w`、`p_w` 和 `tau` 执行文档中的参数扫描；如指定 `--include-grid-values`，候选区域中会附带每个网格中心的目标值（不可行或奇异点为 `null`）。

## 测试

```powershell
python -m unittest discover -s code/t2 -p "test_*.py"
```
