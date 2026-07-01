"""pyCauchyKesai — Horizon Robotics BPU Python bindings.

Usage:
    from pyCauchyKesai import CauchyKesai
    model = CauchyKesai("model.hbm")
    outputs = model([input1, input2])
"""

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import Any

# ============================================================================
# 环境变量默认值（仅缺省时填入，已设置不覆盖）
# ============================================================================
# L2 Cache 大小（BPU 推理默认开，性能优化） | 日志级别：6=never（0=trace..6=never）
# UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1：跳过填默认 L2 Cache——让用户能不配 L2 Cache，
#   从而启用抢占优先级 254/255（L2 Cache 与抢占互斥；抢占另需模型编译时设 max_time_per_fc）。
#   已显式设置的 HB_DNN_USER_DEFINED_L2M_SIZES 一律不动。
if os.environ.get("UNSET_HB_DNN_USER_DEFINED_L2M_SIZES") != "1":
    os.environ.setdefault("HB_DNN_USER_DEFINED_L2M_SIZES", "6:6:6:6")
os.environ.setdefault("HB_UCP_LOG_LEVEL", "6")
os.environ.setdefault("HB_NN_LOG_LEVEL", "6")
os.environ.setdefault("HB_VP_LOG_LEVEL", "6")
os.environ.setdefault("HB_HPL_LOG_LEVEL", "6")


# ============================================================================
# CauchyKesai (所有平台) — Platform / from_numpy 已迁 C++，Python 仅留展示壳
# ============================================================================
from .pycauchykesai import CauchyKesai as _CauchyKesaiNative
from .pycauchykesai import (
    IONArray,
    IONArrayDesc,
    IONMemory,
    Platform,
    platform,
    read_bpu_fw_version,
)

# 版本号唯一源头 = pyproject.toml；运行时从已装包元数据取（项目只 whl 不 editable）.
# 源码树未安装（无 .dist-info）时兜底为 0.0.0+unknown.
try:
    __version__ = _dist_version("pycauchykesai")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
__all__ = [
    "CauchyKesai",
    "IONArray",
    "IONMemory",
    "IONArrayDesc",
    "from_numpy",
    "Platform",
    "platform",
    "read_bpu_fw_version",
]


# ============================================================================
# 模块级工厂: from_numpy（C++ IONArray.from_numpy_array 静态方法的 re-export 别名）
# ============================================================================
# C++ 侧 IONArray.from_numpy_array(arr, cached) = ascontiguousarray + IONArray(dtype,shape)
# + from_numpy。此处仅导出别名，满足 `from pyCauchyKesai import from_numpy`（文档/测试契约）。
from_numpy = IONArray.from_numpy_array


# ============================================================================
# 渲染层：summary()/benchmark() 由 C++ 返回原生 dict（scale/zero_point 已是 list，
# 可直接 json.dumps / 供 AI 消费）；s()/t() 内部 print 美观彩色字符串.
# ============================================================================
from ._render import render_benchmark, render_summary  # noqa: E402


class CauchyKesai(_CauchyKesaiNative):
    """BPU 推理引擎 (Nash-e / Nash-m / Nash-p).

    summary()/benchmark() 由 C++ 返回原生 dict（含 platform / bpu_fw_version /
    compile_config；scale/zero_point 已是 list，可直接 json.dumps / 供 AI 结构化消费）；
    s()/t() 内部 print 美观彩色字符串（直接调用即可，无需再用 print(...) 包裹）.

    持有一个 Platform 实例 (self.platform)，整合机型 / 版本 / 核数 / 占用率 /
    频率 / 温度 / CPU / 内存 等平台原子能力（机型无关，不依赖本模型）.
    """

    def __init__(self, model_path: str, n_task: int = 1, model_cnt_select: int = 0) -> None:
        """加载 BPU 模型（平台 / 固件版本 / 编译配置由 C++ summary 按需读取）."""
        super().__init__(model_path, n_task, model_cnt_select)
        # platform 由 C++ 成员提供（m.platform），无需 Python 构造。
        # compile_config / platform 由 C++ summary() 注入，无需 Python 缓存。

    # ── 摘要 / 基准测试：summary()/benchmark() 走 C++；s()/t() 内部 print 美观输出 ──

    def s(self) -> None:
        """美观打印模型摘要（内部 print，返回 None）."""
        print(render_summary(self.summary()))

    def t(self, timeout_ms: int = 0) -> None:
        """美观打印单次推理计时（内部 print，返回 None）."""
        print(render_benchmark(self.benchmark(timeout_ms)))

    def __repr__(self) -> str:
        try:
            info = super().summary()
            selected = next(m for m in info["model_names"] if m["selected"])
            n_task = info["n_task"]
            busy = [i for i in range(n_task) if super().is_busy(i)]
            busy_str = f", {len(busy)} busy" if busy else ""
            return (
                f"CauchyKesai(model='{selected['name']}', "
                f"inputs={len(info['inputs'])}, outputs={len(info['outputs'])}, "
                f"n_task={n_task}{busy_str})"
            )
        except Exception:
            return "CauchyKesai(<error>)"

    # ── Context Manager ──

    def __enter__(self) -> "CauchyKesai":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # C++ ~CauchyKesai 由 pybind11 自动管理 (无显式 close/release)
        # GC 会在引用归零后调用 C++ 析构释放 BPU 资源.
        return None

