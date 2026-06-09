"""pyCauchyKesai — Horizon Robotics BPU Python bindings.

Usage:
    from pyCauchyKesai import CauchyKesai
    model = CauchyKesai("model.hbm")
    outputs = model([input1, input2])
"""

import sys


def _color(text: str) -> str:
    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        return f"\033[1m{text}\033[0m"
    return text


# ============================================================================
# CauchyKesai (所有平台)
# ============================================================================
from . import pycauchykesai as _native
from .pycauchykesai import CauchyKesai as _CauchyKesaiNative
from .pycauchykesai import IONArray

__version__ = _native.__version__
__all__ = ["CauchyKesai", "IONArray"]


class ModelSummary(dict):
    def __repr__(self):
        try:
            lines = [
                "================== Model Summary ==================",
                _color("Model File: ") + self.get("model_path", "<unknown>"),
                _color("Model Names: "),
            ]
            for entry in self.get("model_names", []):
                suffix = " [*Select]" if entry.get("selected") else ""
                lines.append(f"{entry.get('index', '?')}: {entry.get('name', '?')}{suffix}")
            lines.append(_color("Task N: ") + str(self.get("n_task", "?")))
            lines.append(_color("Memory: ") + str(self.get("memory_mb", "?")) + " MB")
            lines.append(_color("System: ") +
                         f"DNN={self.get('dnn_version', '?')}, "
                         f"BPU={self.get('bpu_version', '?')}, "
                         f"SoC={self.get('soc_name', '?')}")
            lines.append(_color("BPU Cores: ") + str(self.get('bpu_core_num', '?')))
            model_desc = self.get('model_desc', 'N/A')
            if model_desc and model_desc != 'N/A':
                lines.append(_color("Model Desc: ") + model_desc)
            lines.append(_color("Inputs Info: "))
            for inp in self.get("inputs", []):
                shape_str = ", ".join(str(d) for d in inp.get("shape", []))
                quanti = "SCALE" if inp.get('quantiType', 0) == 1 else "NONE"
                line = (f"  [{inp.get('index', '?')}][{inp.get('name', '?')}]: "
                        f"{inp.get('dtype', '?')}, ({shape_str}), "
                        f"size={inp.get('alignedByteSize', '?')}, quanti={quanti}")
                if quanti == "SCALE":
                    scale_arr = inp.get('scale')
                    zp_arr = inp.get('zero_point')
                    axis = inp.get('quantizeAxis', 0)
                    line += f", axis={axis}"
                    if scale_arr is not None and scale_arr.size > 0:
                        line += f", scale.shape={scale_arr.shape}"
                        if scale_arr.size <= 4:
                            line += f", scale={scale_arr.flatten().tolist()}"
                        else:
                            flat = scale_arr.flatten()
                            line += f", scale=[{flat[0]:.6f}, ..., {flat[-1]:.6f}]"
                    if zp_arr is not None and zp_arr.size > 0:
                        line += f", zp.shape={zp_arr.shape}"
                        if zp_arr.size <= 4:
                            line += f", zp={zp_arr.flatten().tolist()}"
                        else:
                            flat = zp_arr.flatten()
                            line += f", zp=[{flat[0]}, ..., {flat[-1]}]"
                if inp.get('desc', 'N/A') != 'N/A':
                    line += f", desc={inp.get('desc')}"
                lines.append(line)
            lines.append(_color("Outputs Info: "))
            for out in self.get("outputs", []):
                shape_str = ", ".join(str(d) for d in out.get("shape", []))
                quanti = "SCALE" if out.get('quantiType', 0) == 1 else "NONE"
                line = (f"  [{out.get('index', '?')}][{out.get('name', '?')}]: "
                        f"{out.get('dtype', '?')}, ({shape_str}), "
                        f"size={out.get('alignedByteSize', '?')}, quanti={quanti}")
                if quanti == "SCALE":
                    scale_arr = out.get('scale')
                    zp_arr = out.get('zero_point')
                    axis = out.get('quantizeAxis', 0)
                    line += f", axis={axis}"
                    if scale_arr is not None and scale_arr.size > 0:
                        line += f", scale.shape={scale_arr.shape}"
                        if scale_arr.size <= 4:
                            line += f", scale={scale_arr.flatten().tolist()}"
                        else:
                            flat = scale_arr.flatten()
                            line += f", scale=[{flat[0]:.6f}, ..., {flat[-1]:.6f}]"
                    if zp_arr is not None and zp_arr.size > 0:
                        line += f", zp.shape={zp_arr.shape}"
                        if zp_arr.size <= 4:
                            line += f", zp={zp_arr.flatten().tolist()}"
                        else:
                            flat = zp_arr.flatten()
                            line += f", zp=[{flat[0]}, ..., {flat[-1]}]"
                if out.get('desc', 'N/A') != 'N/A':
                    line += f", desc={out.get('desc')}"
                lines.append(line)
            lines.append("====================================================")
            return "\n".join(lines)
        except Exception:
            return f"ModelSummary(<error: {dict.__repr__(self)}>)"

    def __str__(self):
        return self.__repr__()


class BenchmarkResult(dict):
    def __repr__(self):
        lines = [
            _color("Inference Info: "),
            f"  Time: {self['time_us']:.6g} us",
            f"  Time: {self['time_ms']:.6g} ms",
            f"  Time: {self['time_s']:.6g} s",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


class CauchyKesai(_CauchyKesaiNative):
    """BPU 推理引擎 (Nash-e / Nash-m / Nash-p)."""

    def s(self):
        return ModelSummary(super().s())

    def t(self):
        return BenchmarkResult(super().t())

    def __repr__(self):
        try:
            info = super().s()
            selected = next(m for m in info["model_names"] if m["selected"])
            n_task = info["n_task"]
            busy = [i for i in range(n_task) if super().is_busy(i)]
            busy_str = f", {len(busy)} busy" if busy else ""
            return (f"CauchyKesai(model='{selected['name']}', "
                    f"inputs={len(info['inputs'])}, outputs={len(info['outputs'])}, "
                    f"n_task={n_task}{busy_str})")
        except Exception:
            return "CauchyKesai(<error>)"
