# 数学建模可视化工具包

这是本仓库的可视化能力入口。它把两类内容分开维护：

- `docs/` 是调研、图形选型与架构规范，回答“该画什么、为什么画”。
- `src/mathmodel_viz/` 是轻量的可查询目录和公共工具，回答“如何稳定地复用”。

工具包并不试图用一个巨型脚本实现所有图形。图形分类、工作流阶段、数据形态和推荐后端只在 `catalog.py` 中登记一次；具体图形实现按需加入后端或领域 recipe。这样不会在报告脚本、示例和文档中重复维护同一套判断逻辑。

## 快速开始

在本目录执行：

```powershell
python -m pip install -e .
mathmodel-viz report
mathmodel-viz list --stage optimize
mathmodel-viz recommend --stage forecast --shape series --mode static
python examples/quickstart.py

# 图库包含机器学习诊断面板（可选依赖）
python -m pip install -e ".[model-diagnostics]"
# 在指定目录生成跨类别测试图库
python examples/chart_gallery.py --output "..\..\scratch\vis-test"
```

`quickstart.py` 会在 `outputs/figures/visualization-demo/` 创建一个演示图。真实建模项目应把结果写入仓库根目录的 `outputs/figures/<模型或实验名>/`，而不是写入本工具目录。

## 全局主题

所有示例和可复用图形都从 `mathmodel_viz.styles` 读取统一主题。中文字体使用宋体（`SimSun`），英文与数字优先使用 `Times New Roman`；Matplotlib、Seaborn 和 Plotly 的颜色循环、网格、色图与交互模板也由同一处配置。

默认颜色为：主红 `#B41B20`、主蓝 `#266AA0`、过渡色 `#EF8F67`，辅助色 `#FFE181`。需求中的五位写法 `#B41B2` 和 `#266AA` 不是标准十六进制颜色，工具会在末尾补 `0`，分别规范化为 `#B41B20` 和 `#266AA0`。

在项目入口处配置一次即可让后续图形跟随新配色：

```python
from mathmodel_viz import configure_matplotlib, configure_theme, get_theme

configure_theme(
    primary_red="#B41B2",
    primary_blue="#266AA",
    transition="#EF8F67",
    auxiliary="#FFE181",
)
configure_matplotlib()
theme = get_theme()
```

绘图代码应使用 `theme.primary_red`、`theme.primary_blue`、`theme.transition`、`theme.auxiliary`，或使用 `theme.color(name, alpha)` / `theme.css_color(name, alpha)` 获取透明度变体；连续色图使用 `theme.colormap()` 或 `theme.plotly_colorscale()`。这样只修改 `configure_theme()` 的参数就能整体换色，不会引入未登记的默认色。

可选能力按场景安装，避免默认环境变得臃肿：

```powershell
python -m pip install -e ".[geo]"
python -m pip install -e ".[interactive]"
python -m pip install -e ".[three-d]"
python -m pip install -e ".[dashboard,model-diagnostics]"
```

## 目录

```text
tools/visualization/
├── README.md
├── pyproject.toml                 # 唯一依赖定义，避免 requirements 重复
├── docs/
│   ├── 01_visualization_catalog.md
│   ├── 02_python_visualization_projects.md
│   └── 03_architecture_and_extension.md
├── examples/
│   ├── quickstart.py              # 可运行的最小示例，不承载公共逻辑
│   └── chart_gallery.py           # 跨类别可重复的 smoke-test 图库
├── src/mathmodel_viz/
│   ├── models.py                  # 稳定领域契约
│   ├── catalog.py                 # 图形元数据唯一真源
│   ├── registry.py                # 查询与选型推荐
│   ├── backends.py                # 后端和可选依赖清单
│   ├── styles.py                  # 中文字体和输出策略
│   └── cli.py                     # 命令行入口
└── tests/
    ├── test_registry.py
    └── test_styles.py
```

## 常用命令

```powershell
# 查看某阶段可选图形
mathmodel-viz list --stage validate

# 按数据形态和建模阶段推荐图形
mathmodel-viz recommend --stage optimize --shape matrix --shape series

# 输出可被其他脚本消费的 JSON
mathmodel-viz list --category time-series --format json

# 检查当前 Python 环境可用的渲染后端
mathmodel-viz backends --available-only

# 无额外测试框架的最小验证
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 资料导航

- [可视化分类与选型](docs/01_visualization_catalog.md)
- [Python 生态项目调研](docs/02_python_visualization_projects.md)
- [架构、扩展和避免冗余的规则](docs/03_architecture_and_extension.md)

## 约定

- 每幅图回答一个可检验的问题，并标明量纲、时间窗、样本量或不确定性来源。
- 静态论文图优先使用 Matplotlib/Seaborn；需要悬停、缩放或网页交付时优先 Plotly；声明式比较视图可选 Altair。
- 在添加“新图”前先查询目录，防止已有的散点、热力图或收敛曲线被以不同名称重复实现。
- 公共样式、中文字体和文件输出统一由 `styles.py` 处理；领域脚本不得各自复制 `rcParams` 或 `savefig` 逻辑。
- `chart_gallery.py` 使用固定随机种子生成合成数据，覆盖数据质量、时序、网络、动力学、优化、调度、风险、机器学习和统计推断；它是能力回归测试，不承载真实题目结论。
