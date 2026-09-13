# 第二问连续鲁棒选址求解器

`solve_q2.py` 实现 `docs/问题二求解思路.md` 中的第二问模型：已知第一检测点及其示向度后，在连续目标可行域上积分位置误差型 DOP，寻找满足全域接收约束的第二检测点，并输出近优候选区域与敏感性分析结果。

## 运行

从工作区根目录运行演示：

```powershell
python code/t2/solve_q2.py --demo --no-global --output outputs/t2/t2_demo.json
```

正式实验建议把结果写入问题二专用目录：

```powershell
python code/t2/solve_q2.py --demo --no-global --output outputs/t2/t2_default_noglobal.json
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
python code/t2/solve_q2.py --input data/question2.json --output outputs/t2/t2_result.json
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

## 完整数值实验

`enhanced_experiments.py` 是面向论文的实验入口，逐项重新优化并导出：

- `circle_sides`、积分阶数和候选网格分辨率收敛表；
- 多随机种子的局部搜索与“差分进化 + 局部搜索”对照；
- 由 marching squares 提取的近优水平集边界；
- 目标区域、接收可行域、两个对称候选分支和多个 `tau` 水平集叠加图；
- 带有测向误差、随机接收半径和双线交会的 Monte Carlo 定位误差/命中率实验。

先用低成本配置检查完整流程：

```powershell
python code/t2/enhanced_experiments.py `
  --demo --quick --samples 200 `
  --output-dir outputs/t2/enhanced_quick
```

论文规模的示例配置如下（运行时间取决于本机 CPU）：

```powershell
python code/t2/enhanced_experiments.py `
  --demo `
  --circle-values 32,64,128,180,360 `
  --quadrature-values 3,4,5,6,8 `
  --grid-values 31,51,81,121,161 `
  --seeds 0,1,2 `
  --samples 1000 `
  --max-local-starts 16 `
  --local-maxiter 60 `
  --global-maxiter 8 `
  --global-popsize 5 `
  --boundary-grid 161 `
  --output-dir outputs/t2/enhanced_formal
```

输出目录中的 `q2_circle_convergence.csv`、`q2_quadrature_convergence.csv`、
`q2_grid_convergence.csv` 和 `q2_search_comparison.csv` 是可直接制表的原始结果；
`q2_boundary_contours.csv` 保存 marching-squares 折线，图形位于 `figures/`。
Monte Carlo 还会写出按种子汇总、跨种子聚合和逐样本明细三个 CSV。

用增强实验包重绘论文级高级敏感度图（参数热力图、收敛表、搜索对照、Monte Carlo
面板、目标/接收可行域与两个对称分支叠加图，以及综合 dashboard）：

```powershell
python code/t2/advanced_sensitivity_visualization.py `
  --input outputs/t2/q2_sensitivity.json `
  --enhanced-dir outputs/t2/enhanced_formal `
  --geometry-input outputs/t2/t2_default_noglobal.json `
  --geometry-grid 161 `
  --output-dir outputs/t2/figures/advanced
```

该命令同时导出 `q2_sensitivity_summary.csv`、`q2_convergence_formal_summary.csv` 和
`q2_advanced_visualization_manifest.json`。manifest 记录实际输入文件、增强实验来源、
smoke/正式口径和全部图形清单，便于论文图表追溯。

实验说明、指标分母和论文表述见 [`docs/问题二算法落地与实验.md`](../../docs/问题二算法落地与实验.md)。

候选面积仍需标注为数值近似：单元中心面积是栅格估计，只有闭合等值线才使用 marching-squares 多边形面积；被搜索框截断的开放等值线会回退到栅格估计。`strict_feasible`（`H<=0`）与 `feasible_with_tolerance`（允许浮点残差）必须分开报告，不能把数值容差当作严格解析证明。

## 敏感性分析与可视化

`run_sensitivity.py` 复用本目录中的连续模型，对 `epsilon_w`、`p_w` 和 `tau`
执行参数扫描，同时写出 JSON 和扁平 CSV。默认输出已经指向 `outputs/t2`：

```powershell
python code/t2/run_sensitivity.py --demo --no-global --output outputs/t2/q2_sensitivity.json
```

需要快速检查流程时可使用低成本 smoke 配置：

```powershell
python code/t2/run_sensitivity.py --demo --smoke --no-global --output outputs/t2/sensitivity_smoke.json
```

`plot_sensitivity.py` 从上述 JSON（也支持同名 CSV）读取结果，并通过共享的
`tools/visualization/src/mathmodel_viz/styles.py` 配置字体、主题和保存函数：

```powershell
python code/t2/plot_sensitivity.py --input outputs/t2/q2_sensitivity.json --output-dir outputs/t2/figures
```

问题二的正式 JSON、CSV 和图形统一放在 `outputs/t2`；历史检查结果放在
`outputs/t2/archive`。合并前的旧版问题二脚本仅作为参考保存在
`scratch/archive/q2_legacy`，不再作为运行入口。
