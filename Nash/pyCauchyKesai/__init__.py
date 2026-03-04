from .pyCauchyKesai import CauchyKesai as _CauchyKesai
from .pyCauchyKesai import *


class ModelSummary(dict):
    """s() 的返回值：既是 dict（Agent 可结构化访问），又能 print 出和原来完全一致的格式化文本。"""

    def __repr__(self):
        lines = []
        lines.append("================== Model Summarys ==================")
        lines.append("\033[1;31mModel File: \033[0m" + self["model_path"])
        lines.append("\033[1;31mModel Names: \033[0m")
        for entry in self["model_names"]:
            suffix = " [*Select]" if entry["selected"] else ""
            lines.append(f"{entry['index']}: {entry['name']}{suffix}")
        lines.append("\033[1;31mTask N: \033[0m" + str(self["n_task"]))
        lines.append("\033[1;31mInputs/Outputs AlignedByteSize: \033[0m" + str(self["memory_mb"]) + "MB.")
        lines.append("\033[1;31mInputs Info: \033[0m")
        for inp in self["inputs"]:
            shape_str = ", ".join(str(d) for d in inp["shape"]) + ", "
            lines.append(f"[{inp['index']}][{inp['name']}]: {inp['dtype']}, ({shape_str})")
        lines.append("\033[1;31mOutputs Info: \033[0m")
        for out in self["outputs"]:
            shape_str = ", ".join(str(d) for d in out["shape"]) + ", "
            lines.append(f"[{out['index']}][{out['name']}]: {out['dtype']}, ({shape_str})")
        lines.append("====================================================")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


class BenchmarkResult(dict):
    """t() 的返回值：既是 dict（Agent 可结构化访问），又能 print 出和原来完全一致的格式化文本。"""

    def __repr__(self):
        lines = []
        lines.append("\033[1;31mInference Info: \033[0m")
        lines.append(f"Time in microseconds: {self['time_us']:.6g} μs")
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

        # 统计忙碌的 task
        busy_tasks = [i for i in range(n_task) if super().is_busy(i)]
        busy_str = f", {len(busy_tasks)} busy" if busy_tasks else ""

        return (f"CauchyKesai(model='{selected_model['name']}', "
                f"inputs={n_inputs}, outputs={n_outputs}, "
                f"n_task={n_task}{busy_str})")
