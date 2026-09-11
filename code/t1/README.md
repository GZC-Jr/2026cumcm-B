# 问题一：交会定位区域

`solve_q1.py` 将每个检测点的示向度 `theta` 按 `theta +/- 1°` 转成两个半平面约束

```text
a_lower = (sin(theta-delta), -cos(theta-delta))
a_upper = (-sin(theta+delta), cos(theta+delta))
a_lower · z <= a_lower · s
a_upper · z <= a_upper · s
```

所有约束的交集就是定位区域。程序先用线性可行性和四个坐标方向的有界性检查对区域分类，再对有界区域构造边界顶点。二维多边形使用旋转卡壳计算直径；直径圆覆盖判定检查最远点对中点到全部顶点的距离；最小覆盖圆通过两点直径圆和三点外接圆候选枚举求得。算法不使用人为大矩形截断区域。

## 输入

JSON 示例：

```json
{
  "angle_error_deg": 1.0,
  "measurements": [
    {"x": 1000, "y": 0, "svd_deg": 180},
    {"x": -500, "y": 866.0254038, "svd_deg": 300}
  ]
}
```

也支持三列 CSV（表头为 `x,y,svd_deg`）。`angle_error_deg` 必须位于 `(0, 90)`；角度统一按度输入，程序内部转为弧度。

## 运行

```powershell
python code/t1/solve_q1.py --input scratch/t1_input.csv --output outputs/t1_result.json --vertices-output outputs/t1_vertices.csv
python code/t1/solve_q1.py --demo
python code/t1/solve_q1.py --demo --tolerance 1e-8
python -m unittest discover -s code/t1 -p "test_*.py"
```

结果中的 `status` 取值为 `empty`、`unbounded`、`point`、`segment` 或 `polygon`。空集和无界集不定义有限直径；有界区域会输出 `vertices`、`diameter`、`diameter_circle` 和 `minimum_cover_circle`。

半平面交主流程按方向排序后使用双端队列，低维或数值异常时自动回退到可行边界交点枚举。直径使用旋转卡壳；最小覆盖圆使用固定随机种子的 Welzl 随机增量算法，并保留两点/三点候选枚举作为数值兜底。运行环境需要 `numpy` 和 `scipy`。
