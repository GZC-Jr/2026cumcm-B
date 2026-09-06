# Python 数学建模可视化项目调研

## 调研口径

本页优先列出维护活跃、文档完善、在数学建模中有明确定位的开源 Python 项目。它们不是互斥替代品：实际项目通常用一个静态图主栈，加上按问题安装的空间、三维、模型诊断或仪表盘组件。

官方主页和仓库链接放在各项目名称上，作为继续核对版本、许可证和 API 的一手来源。当前机器无法通过系统 TLS 凭据完成在线复核，因此本文不锁定任何外部项目的“最新版本”；依赖下限统一在本工具包的 `pyproject.toml` 中维护。

## 核心绘图与交互生态

| 项目 | 主要能力 | 最适合的建模场景 | 本工具包定位 |
|---|---|---|---|
| [Matplotlib](https://matplotlib.org/stable/) | 底层二维/三维静态绘图、精细排版、多格式导出 | 论文图、函数曲线、收敛过程、需要完全控制的图 | 核心静态后端 |
| [Seaborn](https://seaborn.pydata.org/) | 基于 DataFrame 的统计图、分面、回归与分布图 | EDA、组间比较、相关性、回归诊断 | 核心统计图层，底层仍为 Matplotlib |
| [Plotly Python](https://plotly.com/python/) | 悬停、缩放、选择、地图、三维、Sankey | 探索性分析、方案对比、可交互展示 | 核心交互后端 |
| [Altair](https://altair-viz.github.io/) | 基于 Vega-Lite 的声明式统计图与交互 | 整洁表格数据、组合图、小多图、筛选联动 | 核心声明式后端 |
| [Bokeh](https://docs.bokeh.org/) | 浏览器端交互绘图与服务端应用 | 自定义交互、流式或大规模可视化 | `interactive` 可选组 |
| [HoloViews](https://holoviews.org/) / [hvPlot](https://hvplot.holoviz.org/) | 高层声明式对象和快速 DataFrame/xarray 绘图 | 快速探索、多后端切换、交互组合 | `interactive` 可选组 |
| [Panel](https://panel.holoviz.org/) | 将多个图和控件编排为交互应用 | 模型参数探索、研究型 dashboard | `interactive` 可选组 |
| [Pandas plotting](https://pandas.pydata.org/docs/user_guide/visualization.html) | 常见图的快捷入口 | 初步检查和小型表格 | 只作快捷入口，不承载公共图形实现 |

## 空间、网络与三维科学可视化

| 项目 | 主要能力 | 适合的建模场景 | 本工具包定位 |
|---|---|---|---|
| [GeoPandas](https://geopandas.org/en/stable/) | GeoDataFrame、矢量空间运算和地图 | 行政区指标、点位、空间连接与缓冲区 | `geo` 可选组 |
| [Cartopy](https://cartopy.readthedocs.io/) | 地图投影、海岸线和地理坐标系统 | 科学地图、气象/环境数据、严格投影 | `geo` 可选组 |
| [Folium](https://python-visualization.github.io/folium/latest/) | 基于 Leaflet 的交互网页地图 | 点位、热力、路线和结果交付 | `geo` 可选组 |
| [Contextily](https://contextily.readthedocs.io/) | 叠加在线底图 | 静态空间结果的语境表达 | `geo` 可选组；不作为数据来源 |
| [NetworkX](https://networkx.org/documentation/stable/) | 图结构、路径、中心性、社区和基础绘制 | 网络分析、路径、流与关系模型 | 核心网络分析后端 |
| [PyVis](https://pyvis.readthedocs.io/) | 浏览器交互式网络图 | 中小规模网络的探索和展示 | 后续网络交互 recipe 的候选依赖 |
| [PyVista](https://docs.pyvista.org/) | VTK 驱动的网格、体数据和 3D 场 | 有限元、几何、体素、三维连续场 | `three-d` 可选组 |
| [Mayavi](https://docs.enthought.com/mayavi/mayavi/) | 科学三维可视化 | 传统科研工作流中的三维场 | 备选；优先 PyVista 以降低新项目复杂度 |

## 统计、机器学习、敏感性与贝叶斯诊断

| 项目 | 主要能力 | 适合的建模场景 | 本工具包定位 |
|---|---|---|---|
| [statsmodels](https://www.statsmodels.org/stable/index.html) | 回归、时间序列、残差和影响点诊断 | 计量、统计推断、ARIMA/STL、回归假设检查 | 核心诊断后端 |
| [scikit-learn visualization](https://scikit-learn.org/stable/visualizations.html) | 混淆矩阵、ROC/PR、校准、学习曲线、PDP | 分类、回归、聚类和模型选择 | 核心 ML 评估后端 |
| [Yellowbrick](https://www.scikit-yb.org/en/latest/) | sklearn 风格的可视化诊断器 | 快速模型比较和诊断 | `model-diagnostics` 可选组 |
| [SHAP](https://shap.readthedocs.io/) | 特征归因、局部与全局解释 | 黑箱模型解释和方案因素分析 | `model-diagnostics` 可选组 |
| [SALib](https://salib.readthedocs.io/) | Morris、Sobol 等全局敏感性分析 | 仿真、参数扫描、稳健性分析 | `model-diagnostics` 可选组 |
| [ArviZ](https://python.arviz.org/) | 后验、trace、rank、HDI 与采样诊断 | 贝叶斯模型和 MCMC | `model-diagnostics` 可选组 |
| [pymoo](https://pymoo.org/) | 多目标优化和 Pareto 分析 | 多目标进化算法与优化实验 | 算法层候选；图形仍应经本目录的选型规范 |

## 结果交付与应用层

| 项目 | 主要能力 | 适合的建模场景 | 本工具包定位 |
|---|---|---|---|
| [Dash](https://dash.plotly.com/) | Plotly 生态的生产型 Web 应用 | 需要多页面、回调和部署的结果系统 | `dashboard` 可选组 |
| [Streamlit](https://docs.streamlit.io/) | 快速把 Python 脚本变成交互应用 | 答辩演示、参数探索、内部工具 | `dashboard` 可选组 |
| [Voilà](https://voila.readthedocs.io/) | 将 Jupyter Notebook 作为 Web 应用展示 | 保留 Notebook 研究流程的轻量交付 | 不作为默认依赖，按交付方式选择 |

## 当前环境盘点

通过本机 Python 环境检查，以下基础库已安装，可直接作为第一阶段能力：

| 库 | 本机版本 | 建议用途 |
|---|---:|---|
| NumPy | 1.26.4 | 数组与数值计算 |
| pandas | 2.2.3 | 表格数据、时间索引与汇总 |
| Matplotlib | 3.10.0 | 静态论文图与精细排版 |
| Seaborn | 0.13.2 | 统计图与分面比较 |
| Plotly | 6.0.0 | 交互图、地图和 Sankey |
| Altair | 5.5.0 | 声明式统计图 |
| NetworkX | 3.4.2 | 网络、路径与中心性 |
| statsmodels | 0.14.4 | 时序和统计诊断 |
| scikit-learn | 1.4.1.post1 | 模型评估、分类/回归诊断 |

后续再次检查环境时运行：

```powershell
mathmodel-viz backends --available-only
```

## 建议的分层组合

| 需求 | 首选组合 | 原因 |
|---|---|---|
| 数学建模论文和 PDF 输出 | Matplotlib + Seaborn + statsmodels | 可控、可复现、适合静态高分辨率输出 |
| 快速探索和答辩展示 | Plotly + pandas + Streamlit | 交互代价低，能够暴露数据细节 |
| 多指标、多目标优化 | Matplotlib/Plotly + pymoo 或自有求解器 | Pareto、收敛和敏感性表达成熟 |
| 空间选址、配送、风险 | GeoPandas + Cartopy/Folium + Plotly | 覆盖空间运算、严格投影与交互交付 |
| 网络关系、路径、流 | NetworkX + Matplotlib/Plotly | 分析图结构和展示分开，避免单纯布局图替代指标 |
| 连续场、网格和有限元 | PyVista + Matplotlib | 三维检查与论文二维切片兼顾 |
| 黑箱模型解释和稳健性 | scikit-learn/statsmodels + SHAP/SALib/ArviZ | 评估、归因、敏感性和不确定性各司其职 |

## 采用原则

1. 不要为了一个图引入完整 dashboard 框架；交付形态需要时再安装 `dashboard` 组。
2. 一个图形可以有多个绘图库，但同一项目的同一类图应固定一个默认后端，保证样式和导出一致。
3. 图形库负责渲染，算法库负责计算。比如 NetworkX、SALib、pymoo 的结果仍需经过统一的样式和输出入口。
4. 增加依赖前先检查其许可、维护状态、Python 版本支持和 Windows 安装成本，尤其是 GIS/VTK 类二进制包。
