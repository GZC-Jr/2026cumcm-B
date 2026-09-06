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

# The gallery includes the optional ML diagnostic panel
python -m pip install -e ".[model-diagnostics]"
# Generate a multi-category test gallery in a chosen output directory
python examples/chart_gallery.py --output "..\..\scratch\vis-test"
```

`quickstart.py` 会在 `outputs/figures/visualization-demo/` 创建一个演示图。真实建模项目应把结果写入仓库根目录的 `outputs/figures/<模型或实验名>/`，而不是写入本工具目录。

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
