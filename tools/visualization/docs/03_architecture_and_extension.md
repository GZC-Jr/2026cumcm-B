# 架构、扩展与避免冗余的规则

## 设计目标

本工具包面向竞赛或研究型数学建模项目。它必须同时满足：

- 图形知识可查询，成员不会因不知道“该画什么”而临时堆脚本。
- 公共行为只维护一处，尤其是中文字体、导出分辨率、目录策略和依赖声明。
- 新题型可以加领域图而不改动不相关的工具或复制已有图形。
- 工具包不拥有具体题目的数据、模型或论文产物。

## 分层与职责

```text
题目数据、求解结果、实验日志
            |
            v
项目中的分析/模型脚本 --------------> outputs/figures/<experiment>/
            |
            v
tools/visualization/src/mathmodel_viz/
  models.py   -> 稳定的数据契约
  catalog.py  -> 图形语义元数据的唯一真源
  registry.py -> 查询、筛选和可解释推荐
  backends.py -> 渲染/分析后端及可选依赖索引
  styles.py   -> Matplotlib 公共样式与安全导出
  cli.py      -> 人和自动化脚本的统一入口
```

`docs/` 解释选择理由，`examples/` 只展示如何调用公共 API，`tests/` 验证契约。三者都不保存可复用的业务实现。

## 全局字体与颜色主题

`styles.py` 是字体和颜色的唯一入口。调用 `configure_matplotlib()` 会同时设置 Matplotlib/Seaborn 的字体、颜色循环、背景和网格；安装 Plotly 时还会注册同名的 `mathmodel` 模板。中文字体固定优先使用宋体（`SimSun`），英文与数字优先使用 `Times New Roman`，缺少字体时才沿用 Matplotlib 的回退机制。

主题由 `VisualizationTheme` 管理，默认主色为主红 `#B41B20`、主蓝 `#266AA0`、过渡色 `#EF8F67`，辅助色为 `#FFE181`。图形 recipe 不应直接写 `tab:*`、灰度字面量或第三方色图；应使用 `get_theme()` 返回的颜色、透明度变体和 `colormap()`。项目入口可以通过 `configure_theme(...)` 一次修改全局配色，后续静态和交互图会读取同一套颜色。

原始需求中的 `#B41B2`、`#266AA` 只有五位十六进制数字，工具在 `normalize_hex_color()` 中统一追加末位 `0`，保证传给 Matplotlib 和 Plotly 的值始终是合法的 `#RRGGBB` 或 `#RRGGBBAA`。

## 为什么不是“一堆绘图脚本”

一个典型的冗余模式是：每个模型脚本各自写 `rcParams`、中文字体、颜色、`savefig`、文件夹创建和散点图函数。短期很快，长期会出现图形样式漂移、路径不一致和难以改动的问题。

这里采用“语义目录 + 按需 recipe”的方式：

1. `catalog.py` 只声明图形的目的、数据形态、工作流阶段与推荐后端。
2. `registry.py` 只做筛选与推荐，不做具体绘图。
3. `styles.py` 只处理跨图通用的样式和输出，不知道任何领域业务。
4. 具体图形需要时才创建 `recipes/<领域>.py`；它依赖目录中的稳定 ID 和公共样式。
5. 某个新后端需要专有适配时才创建 `renderers/<后端>.py`，不能让每个 recipe 自己处理后端差异。

因此，目录、渲染和业务问题分别有唯一归属。

## 当前目录结构

```text
tools/visualization/
├── pyproject.toml
├── docs/
│   ├── 01_visualization_catalog.md
│   ├── 02_python_visualization_projects.md
│   └── 03_architecture_and_extension.md
├── examples/
│   ├── quickstart.py
│   └── chart_gallery.py  # 跨类别合成数据 smoke test
├── src/mathmodel_viz/
│   ├── models.py
│   ├── catalog.py
│   ├── registry.py
│   ├── backends.py
│   ├── styles.py
│   ├── cli.py
│   └── __main__.py
└── tests/
```

空的 `recipes/`、`renderers/` 或 `templates/` 目录没有提前创建。只有存在真实、可测试的实现时才增加目录，避免形式上的“预留结构”变成无人维护的冗余层。

## 扩展一个新图形的步骤

### 1. 先检查是否已有同义原型

```powershell
mathmodel-viz list --stage explain
mathmodel-viz recommend --shape graph --mode static
```

如果已有原型只是缺一种样式或参数，扩展已有 recipe；不要新建一个几乎等价的 ID。

### 2. 只在语义确实不同的时候登记目录

向 `catalog.py` 增加一个 `VisualizationSpec`，并复用已有的：

- `category`；
- `WorkflowStage`；
- `data_shapes` 标签；
- 后端 key。

只有无法表达的新数据结构才新增 tag。这样 `recommend` 的历史查询不会因为任意命名而失效。

### 3. 添加领域 recipe

当一个图形需要复用时，创建例如 `src/mathmodel_viz/recipes/optimization.py`。recipe 应当：

- 接收数据和显式参数，不读取题目目录或全局变量；
- 引用已登记的 `spec_id`；
- 调用 `configure_matplotlib()` 和 `save_figure()`，不复制通用配置；
- 返回 Figure/Axes 或 Plotly Figure，由调用方决定展示还是导出；
- 不在模块导入时写文件或启动 GUI。

### 4. 需要新依赖时添加可选组

依赖只在 `pyproject.toml` 的一个位置声明。例如 GIS 能力进入 `geo` 组，三维能力进入 `three-d` 组。不要新增第二份 `requirements.txt`，也不要把重型可选库放进默认依赖。

### 5. 补充验证

至少新增：

- 一个针对参数或目录条目的单元测试；
- 一个能生成非空文件的 smoke test；
- 一项文档说明，说明它回答的建模问题和不应使用的情形。

## 输出与命名约定

工具包本身不保存题目输出。项目脚本的产物应该落在：

```text
outputs/figures/<experiment-or-model>/<ordinal>_<short-purpose>.<ext>
```

例如：

```text
outputs/figures/arima_validation/01_series_forecast_interval.png
outputs/figures/route_optimization/03_route_baseline_vs_optimal.pdf
```

建议同时保存：

- `png`：答辩、预览和文档插图；
- `pdf` 或 `svg`：矢量论文图；
- 同名 `.json` 或 `.csv`：可复核的绘图数据或参数摘要。

## 质量门槛

提交新图形前应满足：

1. 能说明一个明确问题，而不是“展示数据”。
2. 轴、单位、色标、样本量和不确定性含义完整。
3. 同一比较中的尺度、颜色映射和排序一致。
4. 静态图在灰度打印时仍可区分关键系列。
5. 代码不依赖 notebook 隐式状态或工作目录。
6. 测试能在无 GUI 的环境中运行。

## 验证命令

从 `tools/visualization/` 目录执行：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m mathmodel_viz report
python -m mathmodel_viz recommend --stage optimize --shape multi-objective
python examples/quickstart.py
```

这组命令分别验证目录契约、命令行查询、推荐逻辑与实际图文件导出。

## 跨类别图表回归测试

`examples/chart_gallery.py` 是工具包的端到端 smoke test：它采用固定随机种子和合成数据，生成覆盖不同目录原型的 PNG，以及一个自包含 Plotly HTML。它不应被当作任何真实题目的分析脚本或数据来源。

```powershell
$env:PYTHONPATH = "src"
python examples/chart_gallery.py --output "..\..\scratch\vis-test"
```

输出中的 `manifest.json` 记录每个文件、目录 ID、类别和文件大小，可用于自动检查“目录条目—实际图像”是否仍一一对应。新增一类视觉能力时，应优先扩展本画廊或增加专门 smoke test，而不是把可测试的公共示例复制到多个题目目录。
