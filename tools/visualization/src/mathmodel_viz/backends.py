"""Backend inventory and optional dependency groups for this tool package."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class BackendInfo:
    key: str
    package: str
    import_name: str
    use_case: str
    optional_group: str | None

    @property
    def installed(self) -> bool:
        return find_spec(self.import_name) is not None


BACKENDS: tuple[BackendInfo, ...] = (
    BackendInfo("matplotlib", "matplotlib", "matplotlib", "论文级静态图、函数图、定制布局", None),
    BackendInfo("seaborn", "seaborn", "seaborn", "统计分布、分组比较、回归诊断", None),
    BackendInfo("plotly", "plotly", "plotly", "交互式二维/三维图、地图、桑基图", None),
    BackendInfo("altair", "altair", "altair", "声明式统计图与交互筛选", None),
    BackendInfo("networkx", "networkx", "networkx", "网络分析与基础网络绘图", None),
    BackendInfo("statsmodels", "statsmodels", "statsmodels", "时序、回归和统计诊断图", None),
    BackendInfo("sklearn", "scikit-learn", "sklearn", "模型评估、校准和学习曲线", None),
    BackendInfo("geopandas", "geopandas", "geopandas", "矢量地理数据和静态地图", "geo"),
    BackendInfo("folium", "folium", "folium", "基于 Leaflet 的交互地图", "geo"),
    BackendInfo("cartopy", "cartopy", "cartopy", "投影、地理边界和科学地图", "geo"),
    BackendInfo("bokeh", "bokeh", "bokeh", "浏览器端交互图", "interactive"),
    BackendInfo("holoviews", "holoviews", "holoviews", "高层声明式组合可视化", "interactive"),
    BackendInfo("pyvista", "pyvista", "pyvista", "网格、体数据和三维科学可视化", "three-d"),
    BackendInfo("dash", "dash", "dash", "Plotly 驱动的数据应用", "dashboard"),
    BackendInfo("streamlit", "streamlit", "streamlit", "快速建模结果交互展示", "dashboard"),
    BackendInfo("yellowbrick", "yellowbrick", "yellowbrick", "机器学习可视化诊断", "model-diagnostics"),
    BackendInfo("shap", "shap", "shap", "特征归因和局部解释", "model-diagnostics"),
    BackendInfo("salib", "SALib", "SALib", "全局敏感性分析", "model-diagnostics"),
    BackendInfo("arviz", "arviz", "arviz", "贝叶斯后验与采样诊断", "model-diagnostics"),
)


def list_backends(*, available_only: bool = False) -> tuple[BackendInfo, ...]:
    """Return known backends, optionally excluding packages not installed locally."""

    if not available_only:
        return BACKENDS
    return tuple(backend for backend in BACKENDS if backend.installed)
