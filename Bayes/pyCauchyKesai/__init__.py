import sys
import os
import ctypes

# 在导入 C 扩展前，用 RTLD_GLOBAL 预加载包内的 DNN 动态库，
# 确保运行时优先使用 wheel 包内的版本，而非系统路径中的同名库。
#
# 加载顺序：
#   1. libcnn_intf.so  — 提供 bpu_mem_alloc 等符号，是 libhbrt_bayes_aarch64.so 的传递依赖，
#                        由系统（/usr/hobot/lib）提供，不打包进 wheel。
#   2. libhbrt_bayes_aarch64.so — 包内版本，依赖 libcnn_intf.so
#   3. libdnn.so               — 包内版本，依赖 libhbrt_bayes_aarch64.so
#
# 若系统中找不到 libcnn_intf.so，则跳过预加载（回退到系统库）。
_pkg_lib = os.path.join(os.path.dirname(__file__), "lib")
try:
    ctypes.CDLL("libcnn_intf.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(_pkg_lib, "libhbrt_bayes_aarch64.so"), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(os.path.join(_pkg_lib, "libdnn.so"), mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass

from .pyCauchyKesai import CauchyKesai as _CauchyKesai
from .pyCauchyKesai import *
from .pyCauchyKesai import __version__, __author__, __date__, __doc__


def _color(text: str) -> str:
    """仅在终端环境中输出 ANSI 红色加粗，非终端环境（Jupyter、日志、管道）返回纯文本。"""
    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        return f"\033[1;31m{text}\033[0m"
    return text


class ModelSummary(dict):
    """s() 的返回值：既是 dict（Agent 可结构化访问），又能 print 出格式化文本。"""

    def __repr__(self):
        lines = []
        lines.append("================== Model Summarys ==================")
        lines.append(_color("Model File: ") + self["model_path"])
        lines.append(_color("Model Names: "))
        for entry in self["model_names"]:
            suffix = " [*Select]" if entry["selected"] else ""
            lines.append(f"{entry['index']}: {entry['name']}{suffix}")
        lines.append(_color("Task N: ") + str(self["n_task"]))
        lines.append(_color("Inputs/Outputs AlignedByteSize: ") + str(self["memory_mb"]) + "MB.")
        lines.append(_color("Inputs Info: "))
        for inp in self["inputs"]:
            shape_str = ", ".join(str(d) for d in inp["shape"]) + ", "
            lines.append(f"[{inp['index']}][{inp['name']}]: {inp['dtype']}, ({shape_str})")
        lines.append(_color("Outputs Info: "))
        for out in self["outputs"]:
            shape_str = ", ".join(str(d) for d in out["shape"]) + ", "
            lines.append(f"[{out['index']}][{out['name']}]: {out['dtype']}, ({shape_str})")
        lines.append("====================================================")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


class BenchmarkResult(dict):
    """t() 的返回值：既是 dict（Agent 可结构化访问），又能 print 出格式化文本。"""

    def __repr__(self):
        lines = []
        lines.append(_color("Inference Info: "))
        lines.append(f"Time in microseconds: {self['time_us']:.6g} \u03bcs")
        lines.append(f"Time in milliseconds: {self['time_ms']:.6g} ms")
        lines.append(f"Time in seconds:      {self['time_s']:.6g} s")
        lines.append(f"Time in minutes:      {self['time_min']:.6g} min")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


class CauchyKesai(_CauchyKesai):
    def s(self):
        return ModelSummary(super().s())

    def t(self):
        return BenchmarkResult(super().t())

    def __repr__(self):
        """Agent 友好的模型对象表示，显示关键信息。"""
        info = super().s()
        selected_model = next(m for m in info["model_names"] if m["selected"])
        n_inputs = len(info["inputs"])
        n_outputs = len(info["outputs"])
        n_task = info["n_task"]

        _base = super()
        busy_tasks = [i for i in range(n_task) if _base.is_busy(i)]
        busy_str = f", {len(busy_tasks)} busy" if busy_tasks else ""

        return (f"CauchyKesai(model='{selected_model['name']}', "
                f"inputs={n_inputs}, outputs={n_outputs}, "
                f"n_task={n_task}{busy_str})")
