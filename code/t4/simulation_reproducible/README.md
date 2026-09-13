# 问题三、问题四可复现仿真实验项目

本项目将问题三和问题四分别组织为独立实验目录，支持生成CSV/JSON数据，以及每题一张汇总表和三张统计图。

## 项目结构

```text
simulation_reproducible/
├── generate_output.py       # 单数名称兼容入口
├── generate_outputs.py      # 仿真与图表统一入口
├── output/                  # Q3、Q4汇总表及三张统计图
├── runner.py                 # 公共并行实验与统计逻辑
├── Q3/
│   ├── algorithm.py          # 问题三算法与本地仿真环境
│   ├── run_experiment.py     # 问题三实验入口
│   └── results/
│       ├── cases.csv         # 300组逐案例数据
│       ├── summary.csv       # 7项核心指标
│       └── summary.json      # 汇总数据与复现参数
└── Q4/
    ├── algorithm.py          # 问题四算法与本地仿真环境
    ├── run_experiment.py     # 问题四实验入口
    └── results/
        ├── cases.csv
        ├── summary.csv
        └── summary.json
```

## 默认实验设置

- 每个问题运行300组随机案例
- 共同随机种子范围：`2026091200`—`2026091499`
- 默认最多使用8个并行进程
- 实验使用算法代码自带的 `LocalBackend` 本地合成环境

## 运行方法

在项目根目录执行：

```bash
python Q3/run_experiment.py
python Q4/run_experiment.py
```

快速检查可以减少案例数：

```bash
python Q3/run_experiment.py --cases 2 --workers 1
python Q4/run_experiment.py --cases 2 --workers 1
```

如需完整复现，请不要使用快速检查命令覆盖随项目交付的300组结果；直接执行默认命令即可重新生成完整结果。

## 汇总指标

`summary.csv` 仅包含：

1. 随机案例数
2. 全部清除成功率
3. 干扰源总数
4. 单源平均定位清除时间
5. 每局平均定位清除时间
6. 每局平均程序运行时间
7. 每局平均检测次数

程序运行时间会受机器性能和系统负载影响，其余虚拟仿真指标应在相同代码、参数、随机种子及数值计算环境下复现；跨Python/NumPy/SciPy版本可能产生数值分支差异。


## 一键生成数据和图表（修复版）

需要 Python 3.10 或更高版本。在项目根目录执行：

```bash
python -m pip install -r requirements.txt
# 使用已有逐案例数据重新生成图表（不重新仿真）
python generate_output.py
# 重新运行Q3、Q4各300组，再生成图表（会覆盖对应results）
python generate_output.py --rerun --cases 300 --workers 8
```

`generate_output.py` 和 `generate_outputs.py` 完全等价。
`--problem Q3` 或 `--problem Q4` 可只处理一个问题，默认两题；
`--seed-start` 设置起始种子，`--output` 设置图表目录。
案例数、进程数和种子仅在 `--rerun` 时用于仿真。

输出位置：逐案例数据在 `Q3/results`、`Q4/results`，图表在 `output/Q3`、`output/Q4`。
每题输出汇总表CSV、汇总表PNG，以及单源平均定位清除时间分布、任务时间构成、程序运行时间分布三张PNG。
案例数量自动读取，不固定写成300组。

### 本次修复

1. 移除对外部旧目录 `simulation_experiment` 的依赖，直接读取本项目的 `cases.csv`。
2. 按公共运行器的字段重新汇总，解决旧英文JSON结构与当前中文汇总结构不匹配的问题。
3. 分布图直接由逐案例数据绘制，不再复制不存在的旧图片。
4. 公共运行器在临时日志删除前读取真实时间明细，记录 `move_time_s`、`switch_time_s`、`measure_time_s`、`clear_action_time_s`，并逐案例校验它们之和。
5. 补充 `matplotlib` 依赖及跨平台中文字体选择。中文图表需要系统安装 Noto Sans CJK SC、微软雅黑或黑体等中文字体。
6. 兼容单数入口名称，支持一键重跑。Q3和Q4的 `algorithm.py` 算法逻辑未修改。

旧版CSV缺少时间构成明细时，仍可生成汇总及分布图；图2明确显示数据缺失提示。用 `--rerun` 重新运行即可补全，不根据总耗时猜测各项构成。

### 统计口径

- 单源平均时间：每局总任务时间除以该局已清除源数，再对各局取算术平均；并非所有局总时间除以总源数。没有清除任何源的局记空值，不纳入此均值。
- 虚拟时间核算与算法一致：移动距离/5；频道切换1秒/次；检测5秒/次；清除成功5秒/次、失败3秒/次。
- 程序运行时间是实际计时，受硬件、并行进程及系统负载影响。
- Q3为全向源，Q4为全向/定向混合源；使用各自原始的本地合成环境。

随包保留 `original_results/Q3` 和 `original_results/Q4`，为上传时附带的数据；当前 `Q3/results`、`Q4/results` 和 `output` 使用本次重跑结果。旧结果与本次结果存在少量差异，其原因未作确定归因，请勿混用两版汇总或图表。
