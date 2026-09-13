#!/usr/bin/env python3
"""可复现随机仿真实验公共运行器：仅输出CSV/JSON数据，不生成图像。"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_MODULE_CACHE = {}


def load_algorithm(path: Path):
    key = str(path.resolve())
    if key not in _MODULE_CACHE:
        spec = importlib.util.spec_from_file_location(f"algorithm_{path.parent.name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载算法文件：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[key] = module
    return _MODULE_CACHE[key]


def run_one(task):
    algorithm_path, problem, seed, case_number = task
    module = load_algorithm(Path(algorithm_path))
    layout = module.resolve_layout()
    with tempfile.TemporaryDirectory(prefix=f"{problem.lower()}_{case_number:03d}_") as temp_dir:
        row, _ = module.run_local_case(seed, case_number, layout, Path(temp_dir))
        detail = json.loads((Path(temp_dir) / "summary.json").read_text(encoding="utf-8"))
    if row["status"] == "error":
        raise RuntimeError(f"{problem} case={case_number} seed={seed}: {row['error']}")
    components = {
        "move_time_s": detail["distance_m"] / 5,
        "switch_time_s": float(detail["switch_count"]),
        "measure_time_s": 5.0 * detail["measure_count"],
        "clear_action_time_s": 3.0 * detail["clear_attempts"] + 2.0 * detail["cleared_count"],
    }
    if abs(sum(components.values()) - row["virtual_time_s"]) > 1e-5:
        raise ValueError(f"{problem} case={case_number}: 时间构成与总时间不一致")
    return {
        "problem": problem,
        "case": case_number,
        "seed": seed,
        "source_count": int(row["true_total"]),
        "cleared_count": int(row["cleared_count"]),
        "all_cleared": bool(row["success"]),
        "single_source_average_clear_time_s": float(row["average_clear_time_s"]) if row["average_clear_time_s"] is not None else None,
        "per_case_clear_time_s": float(row["virtual_time_s"]),
        "program_runtime_s": float(row["program_runtime_s"]),
        "measure_count": int(row["measure_count"]),
        "status": row["status"],
        "error": row["error"],
        **components,
    }


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    case_count = len(rows)
    return {
        "随机案例数": case_count,
        "全部清除成功率": sum(row["all_cleared"] for row in rows) / case_count,
        "干扰源总数": sum(row["source_count"] for row in rows),
        "单源平均定位清除时间_秒每个": mean(row["single_source_average_clear_time_s"] for row in rows),
        "每局平均定位清除时间_秒每局": mean(row["per_case_clear_time_s"] for row in rows),
        "每局平均程序运行时间_秒每局": mean(row["program_runtime_s"] for row in rows),
        "每局平均检测次数_次每局": mean(row["measure_count"] for row in rows),
    }


def summary_rows(summary):
    return [
        {"指标": "随机案例数", "结果": f'{summary["随机案例数"]}', "单位": "组"},
        {"指标": "全部清除成功率", "结果": f'{summary["全部清除成功率"]:.2%}', "单位": "%"},
        {"指标": "干扰源总数", "结果": f'{summary["干扰源总数"]}', "单位": "个"},
        {"指标": "单源平均定位清除时间", "结果": ("无有效数据" if summary["单源平均定位清除时间_秒每个"] is None else f'{summary["单源平均定位清除时间_秒每个"]:.2f}'), "单位": "秒/个"},
        {"指标": "每局平均定位清除时间", "结果": f'{summary["每局平均定位清除时间_秒每局"]:.2f}', "单位": "秒/局"},
        {"指标": "每局平均程序运行时间", "结果": f'{summary["每局平均程序运行时间_秒每局"]:.2f}', "单位": "秒/局"},
        {"指标": "每局平均检测次数", "结果": f'{summary["每局平均检测次数_次每局"]:.2f}', "单位": "次/局"},
    ]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(problem_dir: Path, problem: str, cases: int = 300,
                   seed_start: int = 2026091200, workers: int | None = None,
                   output_dir: Path | None = None):
    if cases < 1:
        raise ValueError("cases必须大于0")
    problem_dir = problem_dir.resolve()
    algorithm_path = problem_dir / "algorithm.py"
    output_dir = Path(output_dir) if output_dir is not None else problem_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers is not None and workers < 1:
        raise ValueError("workers必须大于0")
    workers = workers if workers is not None else max(1, min(8, os.cpu_count() or 1))
    tasks = [(str(algorithm_path), problem, seed_start + i, i + 1) for i in range(cases)]

    started = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0 or index == cases:
                print(f"{problem} 已完成 {index}/{cases}", flush=True)
    rows.sort(key=lambda row: row["case"])
    summary = summarize(rows)

    write_csv(output_dir / "cases.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows(summary))
    payload = {
        "problem": problem,
        "seed_start": seed_start,
        "seed_end": seed_start + cases - 1,
        "workers": workers,
        "wall_clock_s": time.monotonic() - started,
        "summary": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload
