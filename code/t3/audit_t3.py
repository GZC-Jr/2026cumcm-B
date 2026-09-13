#!/usr/bin/env python3
"""问题三技术审计与可复现汇总。

该脚本只做只读核对和本地验证，不连接官方模拟器，也不会把 LocalBackend
结果标记为正式成绩。它把 ``docs/问题三求解思路.md``、``code/t3/t3.py``
和验证报告关联起来，输出机器可读 JSON 及便于论文复核的 Markdown。

默认用一个新的输出子目录运行 ``verify_t3.py``。如需只汇总已有报告，可
使用 ``--skip-verification``。所有输出均应放在 ``outputs/t3`` 下。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "t3"
VERIFY_SCRIPT = CODE_DIR / "verify_t3.py"
THINKING_DOC = PROJECT_ROOT / "docs" / "问题三求解思路.md"
TECHNICAL_DOC = PROJECT_ROOT / "docs" / "问题三技术核对.md"
SOURCE_SCRIPT = CODE_DIR / "t3.py"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import t3  # noqa: E402  (sibling script intentionally loaded after path setup)


def _json_safe(value: Any) -> Any:
    """Convert common Python/numpy values to strict JSON values."""

    # Avoid importing numpy just for serialization; numpy scalars expose these
    # conversion methods and are handled before the generic iterable cases.
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _source_location(symbol: str) -> dict[str, Any]:
    """Resolve a dotted symbol in t3 and return a stable source location."""

    obj: Any = t3
    try:
        for part in symbol.split("."):
            obj = getattr(obj, part)
        filename = inspect.getsourcefile(obj)
        line = inspect.getsourcelines(obj)[1]
        return {
            "symbol": symbol,
            "exists": True,
            "file": str(Path(filename).resolve()) if filename else None,
            "line": int(line),
        }
    except (AttributeError, OSError, TypeError):
        return {"symbol": symbol, "exists": False, "file": None, "line": None}


# This table is deliberately explicit: it is the review checklist that maps
# the equations/constraints in the modelling note to executable entry points.
# ``authority`` prevents a probabilistic convenience from being mistaken for
# a correctness guarantee in the paper.
TECHNICAL_ITEMS: list[dict[str, Any]] = [
    {
        "id": "domain_and_clock",
        "requirement": "目标圆域、接收半径、速度及动作计时",
        "formula_or_contract": "Omega=B(0,1800), rho in [1000,1500]; move/5, measure=5, switch=1, clear=3/5",
        "symbols": ["LocalBackend", "Robot.call", "Robot.run"],
        "authority": "hard_protocol_and_accounting",
        "verification": ["local_time_accounting", "protocol_boundaries"],
    },
    {
        "id": "coverage",
        "requirement": "中心点加均匀环点的全域接收覆盖",
        "formula_or_contract": "max endpoint distance <= 1000 m; convex quadratic endpoint certificate",
        "symbols": ["coverage_certificate", "scan_points", "resolve_layout"],
        "authority": "hard_geometric_guarantee",
        "verification": ["layout_certificates", "scan_boundary_reception"],
    },
    {
        "id": "negative_channel_elimination",
        "requirement": "未知频道无信号排除与九点不存在认证",
        "formula_or_contract": "all scan points must be observed before absent is certified",
        "symbols": ["CoverageMap", "CoverageMap.update_negative", "update_unknown_belief", "Robot.schedule_run"],
        "authority": "hard_state_transition_plus_planning_map",
        "verification": ["local_time_accounting", "discovered_negative_conditioning"],
    },
    {
        "id": "rolling_route",
        "requirement": "搜索点与已发现目标的滚动开放路径",
        "formula_or_contract": "Held-Karp for <=10 nodes; multi-start nearest-neighbour + 2-opt otherwise",
        "symbols": ["rolling_open_route", "Robot.schedule_run"],
        "authority": "optimization_heuristic",
        "verification": ["local_time_accounting"],
    },
    {
        "id": "intersection_region",
        "requirement": "测向误差楔形与接收距离约束的保守外包",
        "formula_or_contract": "intersection of wedges, 1500 m disks and target domain; 96-direction outer disks",
        "symbols": ["update_poly", "clip", "disk_clip", "initial_poly"],
        "authority": "hard_geometric_guarantee",
        "verification": ["polygon_contains_truth", "degenerate_geometry_regressions"],
    },
    {
        "id": "mec_clear_certificate",
        "requirement": "最小覆盖圆与可靠清除点",
        "formula_or_contract": "r* <= 19.8 m numerical-margin certificate; fallback on optimizer failure",
        "symbols": ["mec", "polygon_diameter", "problem1_geometry_certificate", "Robot.clear_point"],
        "authority": "hard_geometric_guarantee",
        "verification": ["mec_jung_certificates", "optimizer_failure_fallbacks"],
    },
    {
        "id": "reception_safe_measurement",
        "requirement": "补测点对整个定位区域的接收充分证书",
        "formula_or_contract": "triangle-vertex convex-quadratic upper bound with strict 1000 m margin",
        "symbols": ["reception_upper_bound_m2", "reception_certified", "bayes_select", "mandatory_stop_measurement_score", "route_aligned_measurement"],
        "authority": "hard_geometric_guarantee",
        "verification": ["strict_reception_boundary", "empty_station_narrow_polygon"],
    },
    {
        "id": "posterior",
        "requirement": "固定接收半径先验下的位置/半径规划后验",
        "formula_or_contract": "triangular quadrature + Gaussian moment projection; U[1000,1500] planning prior",
        "symbols": ["radius_interval", "update_gaussian", "update_gaussian_negative", "update_unknown_belief", "gaussian_moments"],
        "authority": "planning_approximation_only",
        "verification": ["discovered_negative_conditioning", "posterior_weight_degeneracy"],
    },
    {
        "id": "information_gain",
        "requirement": "未知频道批量扫描顺序与补测候选排序",
        "formula_or_contract": "binary detection entropy + search_gain*receive probability; linearized angle information",
        "symbols": ["binary_entropy", "detection_information", "bayes_select"],
        "authority": "planning_approximation_only",
        "verification": ["posterior_weight_degeneracy"],
    },
    {
        "id": "dop_tiebreak",
        "requirement": "第二问DOP作为近优候选决胜项",
        "formula_or_contract": "depth-weighted, sin-floor-regularized quadrature; only near-primary ties",
        "symbols": ["depth_weighted_dop", "bayes_select", "Q12_LINK_CONFIG"],
        "authority": "planning_approximation_only",
        "verification": ["direct_planner_input_guards", "empty_station_narrow_polygon"],
    },
    {
        "id": "probe_chain",
        "requirement": "有限前瞻试探清除与绕行约束",
        "formula_or_contract": "horizon <=3, attempts <=5, candidate limit and full-failure detour bound",
        "symbols": ["probe_distribution", "joint_probe_plan", "PROBE_CONFIG", "JOINT_CONFIG"],
        "authority": "planning_approximation_only",
        "verification": ["direct_planner_input_guards", "failed_clear_cache_invalidation"],
    },
    {
        "id": "failure_feedback",
        "requirement": "失败清除后的排除圆、计划撤销和缓存失效",
        "formula_or_contract": "condition on distance >20 m; invalidate planned/probe cache",
        "symbols": ["condition_failed_clear", "Robot.record_failed_clear"],
        "authority": "hard_state_update_plus_planning_conditioning",
        "verification": ["failed_clear_cache_invalidation", "fallback_failed_clear_feedback"],
    },
    {
        "id": "completion",
        "requirement": "未知总数下的安全完成判断",
        "formula_or_contract": "done >=16 OR done+absent ==20; final done >=10",
        "symbols": ["Robot.schedule_run", "Robot.run"],
        "authority": "hard_state_transition",
        "verification": ["local_time_accounting"],
    },
    {
        "id": "http_protocol",
        "requirement": "串行HTTP+JSON、幂等重试、时间/响应防护",
        "formula_or_contract": "new request_id per action; same ID/content on retry; accepted and timestamp validation",
        "symbols": ["HttpBackend", "Robot.call"],
        "authority": "hard_protocol_guard",
        "verification": ["protocol_boundaries", "enter_budget_guards", "measure_response_guards"],
    },
]


OPEN_DECISIONS: list[dict[str, Any]] = [
    {
        "id": "default_layout",
        "question": "正式测试沿用 ring8_939，还是先扩大共同随机数演练后切换 ring9_904？",
        "current": "ring8_939",
        "evidence": "当前默认布局有解析覆盖证书；3 个共同随机数案例中 ring9_904 均值虚拟时间较低，但样本很小。",
        "impact": "影响扫描点数量、移动距离和检测服务次数；不改变覆盖证书。",
    },
    {
        "id": "search_gain",
        "question": "未知频道批量扫描的 search_gain 是否保持 0.2，或依据演练日志调参？",
        "current": 0.2,
        "evidence": "排序分数为 H(检测结果)+search_gain×接收概率；只影响顺序，不改变硬完成条件。",
        "impact": "可能改变发现顺序与频道切换次数。",
    },
    {
        "id": "lookahead",
        "question": "试探/顺路规划的 lookahead 是否保持 1.0？",
        "current": 1.0,
        "evidence": "它是后续尾程距离的权重，代码限制在 [0,1]，不是概率或秒数；信息增益时长公式另有固定 beta=0.5。",
        "impact": "改变试探对后续绕行的偏好。",
    },
    {
        "id": "posterior_scope",
        "question": "论文是否明确把高斯后验/信息增益写成规划近似，而非统计保证？",
        "current": "仅排序，不参与几何完成证明",
        "evidence": "硬定位外包、接收证书和模拟器 clear_result 拥有最终权限。",
        "impact": "影响论文表述的可复现性与结论边界。",
    },
    {
        "id": "dop_logging",
        "question": "正式演练是否单独记录 DOP 决胜触发频率及相对收益？",
        "current": "仅在主效用近优且预计改善至少2%时替换",
        "evidence": "Q12_LINK_CONFIG 同时使用主目标相对容差、DOP近优带和最小改善阈值。",
        "impact": "决定论文是否报告 DOP 的实际贡献，而非只报告公式。",
    },
    {
        "id": "fallback_cost",
        "question": "是否接受极窄区域/优化失败时 25 m 网格兜底带来的额外清除尝试？",
        "current": "启用兜底；失败反馈不会被记为成功",
        "evidence": "兜底优先保证继续探索，但网格中心未必在多边形内部，可能浪费少量尝试。",
        "impact": "极端案例的虚拟时间与失败次数可能增加。",
    },
    {
        "id": "official_tests",
        "question": "请在模拟器登录后提供三次正式测试的案例编码和原始加密日志，以填充表1。",
        "current": "尚未连接官方模拟器",
        "evidence": "当前所有可见数值来自 LocalBackend 或确定性几何检查。",
        "impact": "没有官方证据就不能声称正式成绩或计算正式统计值。",
    },
]


def run_verification(output_dir: Path, seed: int) -> dict[str, Any]:
    """Run the deterministic verifier in a subprocess and load its report."""

    verification_dir = output_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(VERIFY_SCRIPT),
        "--output",
        str(verification_dir),
        "--seed",
        str(seed),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(CODE_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        returncode = int(completed.returncode)
        stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "script": str(VERIFY_SCRIPT.resolve()),
            "returncode": None,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "report_path": str((verification_dir / "verification_results.json").resolve()),
        }

    report_path = verification_dir / "verification_results.json"
    report = _read_json(report_path)
    if report is None:
        return {
            "script": str(VERIFY_SCRIPT.resolve()),
            "returncode": returncode,
            "passed": False,
            "error": "verify_t3.py 未产生可解析的 verification_results.json",
            "stderr_tail": stderr_tail,
            "report_path": str(report_path.resolve()),
        }
    checks = report.get("checks", [])
    passed_count = sum(bool(item.get("passed")) for item in checks if isinstance(item, dict))
    return {
        "script": str(VERIFY_SCRIPT.resolve()),
        "returncode": returncode,
        "passed": bool(report.get("passed") is True and returncode == 0),
        "official_simulator_test": bool(report.get("official_simulator_test", False)),
        "check_count": len(checks) if isinstance(checks, list) else 0,
        "passed_count": passed_count,
        "failed_checks": [
            item.get("name")
            for item in checks
            if isinstance(item, dict) and not item.get("passed")
        ] if isinstance(checks, list) else [],
        "report_path": str(report_path.resolve()),
        "stderr_tail": stderr_tail,
    }


def _verification_names(verification: dict[str, Any] | None) -> set[str]:
    """Return names of checks that passed in a generated/existing report."""

    if not verification:
        return set()
    report = _read_json(Path(verification.get("report_path", "")))
    if not report or not isinstance(report.get("checks"), list):
        return set()
    return {
        str(item.get("name"))
        for item in report["checks"]
        if isinstance(item, dict) and item.get("passed") is True
    }


def build_technical_inventory(verification: dict[str, Any] | None) -> list[dict[str, Any]]:
    passed_checks = _verification_names(verification)
    inventory = []
    for item in TECHNICAL_ITEMS:
        locations = [_source_location(symbol) for symbol in item["symbols"]]
        symbols_ok = all(location["exists"] for location in locations)
        required_checks = set(item.get("verification", []))
        checked = sorted(required_checks & passed_checks)
        if symbols_ok and required_checks and required_checks <= passed_checks:
            status = "implemented_and_verified"
        elif symbols_ok:
            status = "implemented_symbolically"
        else:
            status = "missing_code_entry_point"
        inventory.append(
            {
                "id": item["id"],
                "requirement": item["requirement"],
                "formula_or_contract": item["formula_or_contract"],
                "authority": item["authority"],
                "symbols": locations,
                "verification": item.get("verification", []),
                "passed_verification": checked,
                "status": status,
            }
        )
    return inventory


def build_parameter_audit() -> dict[str, Any]:
    layouts = {}
    for name in t3.LAYOUT_PRESETS:
        try:
            resolved = t3.resolve_layout(name)
            layouts[name] = {
                "ring_points": resolved["ring_points"],
                "ring_radius_m": resolved["ring_radius"],
                "certificate": resolved["certificate"],
                "scan_point_count": len(
                    t3.scan_points(resolved["ring_points"], resolved["ring_radius"])
                ),
            }
        except (KeyError, ValueError, TypeError) as exc:
            layouts[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "target_domain_radius_m": 1800.0,
        "guaranteed_receive_radius_m": t3.RECEIVE_RADIUS_M,
        "angle_error_margin_deg": math.degrees(t3.DELTA),
        "outer_disk_halfplanes": t3.N_CIRCLE,
        "strict_receive_margin_m": t3.RECEIVE_MARGIN_M,
        "default_layout": t3.DEFAULT_LAYOUT,
        "layouts": layouts,
        "schedule_config": dict(t3.SCHEDULE_CONFIG),
        "bayes_config": dict(t3.BAYES_CONFIG),
        "q12_link_config": dict(t3.Q12_LINK_CONFIG),
        "probe_config": dict(t3.PROBE_CONFIG),
        "joint_config": dict(t3.JOINT_CONFIG),
        "completion_rule": {
            "stop_if_cleared": 16,
            "stop_if_cleared_plus_absent": 20,
            "minimum_cleared_at_final_check": 10,
        },
        "protocol_limits_from_task": {
            "channels": "1..20",
            "remaining_real_duration_s": "integer in [0,1200]",
            "max_virtual_duration_s": 360000.0,
            "coordinate_abs_limit_m": 2000000.0,
        },
    }


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_layout_comparison(path: Path) -> dict[str, Any]:
    """Summarize an existing common-random-number comparison, if present."""

    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "official_simulator_test": False,
        "source_kind": "LocalBackend synthetic comparison; not official",
        "rows": [],
        "best_by_mean_virtual_time": None,
    }
    if not path.is_file():
        return result
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    cleaned = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in {
                "ring_points",
                "case_count",
                "success_count",
                "mean_measure_count",
            }:
                try:
                    item[key] = int(float(value))
                except (TypeError, ValueError):
                    item[key] = value
            elif key in {
                "ring_radius",
                "mean_clear_ratio",
                "mean_virtual_time_s",
                "std_virtual_time_s",
                "pooled_time_per_source_s",
                "mean_distance_m",
                "mean_program_runtime_s",
                "failed_clear_attempts",
            }:
                item[key] = _float_or_none(value)
            else:
                item[key] = value
        cleaned.append(item)
    result["rows"] = cleaned
    candidates = [
        row for row in cleaned if _float_or_none(row.get("mean_virtual_time_s")) is not None
    ]
    if candidates:
        best = min(candidates, key=lambda row: float(row["mean_virtual_time_s"]))
        result["best_by_mean_virtual_time"] = {
            "layout_name": best.get("layout_name"),
            "mean_virtual_time_s": best.get("mean_virtual_time_s"),
            "case_count": best.get("case_count"),
            "success_count": best.get("success_count"),
        }
    manifest = path.parent / "experiment_manifest.json"
    result["manifest_path"] = str(manifest.resolve())
    manifest_data = _read_json(manifest)
    if manifest_data:
        result["common_random_numbers"] = True
        result["manifest_case_count"] = manifest_data.get("case_count")
        result["seeds"] = manifest_data.get("seeds")
    else:
        result["common_random_numbers"] = False
    return result


def build_markdown(audit: dict[str, Any]) -> str:
    verification = audit.get("verification") or {}
    inventory = audit.get("technical_inventory", [])
    parameters = audit.get("parameters", {})
    comparison = audit.get("layout_comparison", {})
    lines: list[str] = []
    lines.extend(
        [
            "# 问题三技术审计报告",
            "",
            f"> 生成时间：{audit.get('generated_at', '')}（本地审计脚本，不连接官方模拟器）",
            "",
            "## 结论摘要",
            "",
        ]
    )
    if verification.get("passed"):
        lines.append(
            f"- 确定性验证：{verification.get('passed_count', 0)}/{verification.get('check_count', 0)} 项通过。"
        )
    else:
        lines.append(
            f"- 确定性验证未通过或未运行；详情见 `{verification.get('report_path', '')}`。"
        )
    lines.extend(
        [
            "- 几何覆盖、保守定位外包、全域接收证书、MEC 清除条件和协议状态机属于硬条件。",
            "- 高斯后验、信息增益、DOP 和有限前瞻试探只用于行动排序/成本代理，不产生完成证明。",
            "- 当前所有数值实验均是 LocalBackend 合成案例或确定性证书检查，`official_simulator_test=false`。",
            "",
            "## 公式—代码—验证对应",
            "",
            "| 环节 | 代码入口（含源位置） | 权限层级 | 验证状态 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in inventory:
        locations = ", ".join(
            f"`{loc['symbol']}`:{loc['line']}" if loc.get("exists") else f"`{loc['symbol']}`(缺失)"
            for loc in item.get("symbols", [])
        )
        checks = ", ".join(item.get("passed_verification", [])) or "—"
        lines.append(
            f"| {item['requirement']} | {locations} | {item['authority']} | "
            f"{item['status']}；通过：{checks} |"
        )
    lines.extend(
        [
            "",
            "## 当前参数与布局证书",
            "",
            f"- 默认布局：`{parameters.get('default_layout')}`；接收保证半径："
            f"{parameters.get('guaranteed_receive_radius_m')} m；角度裕量："
            f"{parameters.get('angle_error_margin_deg', 0):.3f}°；外圆盘方向数："
            f"{parameters.get('outer_disk_halfplanes')}。",
            "",
            "| 布局 | 环点数 | 半径(m) | 外环最坏距离(m) | 证书 |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for name, item in parameters.get("layouts", {}).items():
        cert = item.get("certificate", {}) if isinstance(item, dict) else {}
        lines.append(
            f"| {name} | {item.get('ring_points', '—')} | {item.get('ring_radius_m', '—')} | "
            f"{cert.get('outer_annulus_distance_bound_m', '—')} | "
            f"{'通过' if cert.get('certified') else '未通过'} |"
        )
    lines.extend(["", "## 共同随机数布局比较（非官方）", ""])
    if comparison.get("exists"):
        best = comparison.get("best_by_mean_virtual_time") or {}
        lines.append(
            f"- 输入：`{comparison.get('path')}`；共同随机数："
            f"`{comparison.get('common_random_numbers')}`；案例数："
            f"`{comparison.get('manifest_case_count', '未知')}`。"
        )
        if best:
            lines.append(
                f"- 这批小样本中均值虚拟时间最低的是 `{best.get('layout_name')}` "
                f"（{best.get('mean_virtual_time_s'):.3f} s，{best.get('success_count')}/"
                f"{best.get('case_count')} 成功）；不能据此替换默认布局。"
            )
        lines.append("")
        lines.append("| 布局 | 案例数 | 成功数 | 均值虚拟时间(s) | 均值程序运行时间(s) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in comparison.get("rows", []):
            lines.append(
                f"| {row.get('layout_name', '—')} | {row.get('case_count', '—')} | "
                f"{row.get('success_count', '—')} | {row.get('mean_virtual_time_s', '—')} | "
                f"{row.get('mean_program_runtime_s', '—')} |"
            )
    else:
        lines.append("- 未找到布局比较 CSV；不会据此推断布局优劣。")
    lines.extend(["", "## 待与队员确认的决策", ""])
    lines.append("| 决策 | 当前值 | 需要确认 | 影响 |")
    lines.append("| --- | --- | --- | --- |")
    for decision in audit.get("open_decisions", []):
        lines.append(
            f"| {decision['id']} | `{decision['current']}` | {decision['question']} | "
            f"{decision['impact']} |"
        )
    lines.extend(
        [
            "",
            "## 限制与正式测试边界",
            "",
            "- `LocalBackend` 只用于算法、几何和虚拟计时回归；它不生成模拟器案例编码、正式统计值或加密行为日志。",
            "- 当前实现只覆盖问题三全向源；定向源的 `no_signal` 不能按同一覆盖证书解释。",
            "- `plan_cpp` 是历史候选布局辅助函数，当前调度使用 `resolve_layout` 的固定环形扫描点；论文应避免把未调用函数写成主流程。",
            "- 兜底网格的中心点可能落在窄多边形外；这不会伪造成功，但可能增加失败尝试和时间。",
            "- 正式表1须在模拟器完成三次正式测试后，由原始日志原样补填；在此之前应保留“待填”，不要把本报告的本地数值写入表1。",
            "",
            "## 复现命令",
            "",
            "```text",
            "python code/t3/audit_t3.py --output outputs/t3/technical_audit_YYYYMMDD",
            "python -m py_compile code/t3/t3.py code/t3/verify_t3.py code/t3/audit_t3.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_name = f"technical_audit_{dt.date.today():%Y%m%d}"
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / default_name,
        help="审计产物目录（必须位于 outputs/t3 下，默认按日期命名）",
    )
    parser.add_argument(
        "--layout-comparison",
        type=Path,
        default=OUTPUT_ROOT / "layout_comparison_final" / "comparison_summary.csv",
        help="可选的共同随机数布局比较 CSV",
    )
    parser.add_argument("--seed", type=int, default=2026091399, help="verify_t3.py 的本地回归种子")
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="不重复运行 verify_t3.py；仅构建代码/参数/已有比较结果审计",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    output_dir = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_verification:
        verification = {
            "passed": False,
            "skipped": True,
            "official_simulator_test": False,
            "report_path": str((output_dir / "verification" / "verification_results.json").resolve()),
        }
    else:
        verification = run_verification(output_dir, args.seed)

    inventory = build_technical_inventory(verification)
    parameters = build_parameter_audit()
    comparison_path = args.layout_comparison
    if not comparison_path.is_absolute():
        comparison_path = PROJECT_ROOT / comparison_path
    comparison = load_layout_comparison(comparison_path.resolve())
    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    audit = {
        "audit_version": 1,
        "generated_at": generated_at,
        "official_simulator_test": False,
        "scope": "question_3_omnidirectional_sources",
        "source_files": {
            "thinking_doc": {"path": str(THINKING_DOC.resolve()), "sha256": _sha256(THINKING_DOC)},
            "technical_doc": {"path": str(TECHNICAL_DOC.resolve()), "sha256": _sha256(TECHNICAL_DOC)},
            "implementation": {"path": str(SOURCE_SCRIPT.resolve()), "sha256": _sha256(SOURCE_SCRIPT)},
            "verifier": {"path": str(VERIFY_SCRIPT.resolve()), "sha256": _sha256(VERIFY_SCRIPT)},
        },
        "verification": verification,
        "technical_inventory": inventory,
        "parameters": parameters,
        "layout_comparison": comparison,
        "open_decisions": OPEN_DECISIONS,
        "limitations": [
            "LocalBackend is synthetic and is not an official simulator result.",
            "Posterior/information/DOP/probe probability are planning approximations only.",
            "The implementation is for omnidirectional sources in question 3.",
            "Official case codes and encrypted logs are not available in this workspace.",
        ],
    }
    _write_json(output_dir / "technical_audit.json", audit)
    (output_dir / "technical_audit.md").write_text(build_markdown(audit), encoding="utf-8")
    print(json.dumps(_json_safe({
        "output_dir": str(output_dir),
        "verification": verification,
        "default_layout": t3.DEFAULT_LAYOUT,
        "layout_comparison_best": comparison.get("best_by_mean_virtual_time"),
        "official_simulator_test": False,
    }), ensure_ascii=False, indent=2, allow_nan=False))
    # A skipped run is intentionally non-failing: it is useful for refreshing
    # the inventory when numpy/scipy are unavailable. A real verifier failure
    # remains visible through a non-zero exit code for CI/manual review.
    return 0 if verification.get("passed") or verification.get("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
