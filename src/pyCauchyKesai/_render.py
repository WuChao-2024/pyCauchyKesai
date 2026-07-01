"""_render — summary/benchmark 字典的渲染层.

职责两件事：
1. _jsonable(): 递归把 numpy 转成 python 原生类型，让 summary()/benchmark() 返回的
   dict 可直接 json.dumps / 供 AI 结构化消费（数据契约）.
2. render_*(): native dict → 美观带颜色的字符串.由 CauchyKesai.s()/t() 与
   Platform.s() 内部 print 调用（展示契约）.

颜色策略：返回的字符串在生成时按 sys.stdout.isatty() 决定是否嵌 ANSI
（与 tools/ez_onnx.py 的 _color 同思路，仅 stderr→stdout）.非终端（管道/重定向/CI）
自动去色，保证纯文本可读.

排版思路借鉴（不复用）tools/ez_onnx.py 的 ModelSummary/BenchmarkResult.__repr__，
适配 native summary 的字段形状（platform / bpu_fw_version / scheduled_cores / memory_mb）.
"""

import sys
from typing import Any

# ANSI 样式码（前景色 + 加粗/暗淡）
# .s() 统一约定：加粗绿 = 普通标签 / 分段；加粗红 = ==== 段标题横幅.
_STYLES = {
    "bold": "1",
    "dim": "2",
    "cyan": "36",
    "green": "1;32",  # 加粗绿（默认：标签 / 分段标题）
    "yellow": "33",
    "magenta": "35",
    "red": "1;31",  # 加粗红（==== 段标题横幅：Model Summary / Compile Config）
}


def _color(text: str, style: str = "green") -> str:
    """TTY 检测：终端加 ANSI 色（默认加粗绿），非终端原样返回.返回值供调用方后续 print."""
    isatty = getattr(sys.stdout, "isatty", None)
    if callable(isatty) and isatty():
        code = _STYLES.get(style, "1")
        return f"\033[{code}m{text}\033[0m"
    return text


# ----------------------------------------------------------------------------
# numpy → python 原生（AI-Native dict）
# ----------------------------------------------------------------------------


def _jsonable(obj: Any) -> Any:
    """递归 numpy→python 原生（dict / list / 标量）.

    保证 json.dumps(obj) 不抛异常、值对 LLM / 结构化消费友好.
    native summary 里仅 scale / zero_point 是 numpy array，其余已是原生类型.
    """
    try:
        import numpy as np
    except ImportError:  # 无 numpy 时退化为恒等（不应发生在本包）
        return obj

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # numpy 标量
        return obj.item()
    return obj


# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------


def _fmt_np_array(arr: Any) -> str:
    """把 numpy array 压成紧凑字符串：空 [] / ≤4 个全列 / 否则首尾省略."""
    try:
        import numpy as np
    except ImportError:
        return str(arr)
    a = np.asarray(arr).flatten()
    if a.size == 0:
        return "[]"
    if a.size <= 4:
        if a.dtype.kind == "f":
            return "[" + ", ".join(f"{x:.6g}" for x in a) + "]"
        return "[" + ", ".join(str(int(x)) for x in a) + "]"
    if a.dtype.kind == "f":
        return f"[{a[0]:.6g}, ..., {a[-1]:.6g}]"
    return f"[{int(a[0])}, ..., {int(a[-1])}]"


def _fmt_shape(shape: Any) -> str:
    """shape(list/np) → '(d0, d1, ...)'."""
    try:
        return "(" + ", ".join(str(int(d)) for d in shape) + ")"
    except Exception:
        return f"({shape})"


def _fmt_io_line(io: dict[str, Any], kind: str) -> str:
    """单个 input/output dict → 一行文本（含量化 scale/zp 简写）.

    kind: "Input" / "Output"（仅影响展示，无逻辑分支）.
    """
    name = io.get("name", "?")
    idx = io.get("index", "?")
    dtype = io.get("dtype", "?")
    shape_str = _fmt_shape(io.get("shape", []))
    size = io.get("alignedByteSize", "?")
    quanti = "SCALE" if io.get("quantiType", 0) == 1 else "NONE"

    line = f"  [{idx}][{name}]: {dtype}, {shape_str}, size={size}, quanti={quanti}"
    if quanti == "SCALE":
        axis = io.get("quantizeAxis", 0)
        line += f", axis={axis}"
        scale = io.get("scale")
        zp = io.get("zero_point")
        if scale is not None:
            try:
                import numpy as _np

                if _np.asarray(scale).size > 0:
                    line += f", scale={_fmt_np_array(scale)}"
            except Exception:
                pass
        if zp is not None:
            try:
                import numpy as _np

                if _np.asarray(zp).size > 0:
                    line += f", zp={_fmt_np_array(zp)}"
            except Exception:
                pass
    if io.get("desc", "N/A") not in ("N/A", "", None):
        line += f", desc={io.get('desc')}"
    return line


# ----------------------------------------------------------------------------
# 顶层渲染
# ----------------------------------------------------------------------------


def render_summary(d: dict[str, Any]) -> str:
    """把 native summary dict 渲染成多行彩色字符串."""
    try:
        lines = [
            _color("================== Model Summary ==================", "red"),
            _color("Model File: ") + str(d.get("model_path", "<unknown>")),
            _color("Model Names: "),
        ]
        for entry in d.get("model_names", []):
            suffix = " [*Select]" if entry.get("selected") else ""
            lines.append(f"  {entry.get('index', '?')}: {entry.get('name', '?')}{suffix}")

        lines.append(_color("Task N: ") + str(d.get("n_task", "?")))
        lines.append(_color("Memory: ") + str(d.get("memory_mb", "?")) + " MB")

        # 系统 / 平台段（platform 由 Python wrapper 注入）
        plat = d.get("platform", {}) or {}
        sys_parts = [
            f"SoC={plat.get('soc_name', '?')}",
            f"BPU={plat.get('bpu_type', '?')}",
            f"DNN={plat.get('dnn_version', '?')}",
            f"BPU FW={d.get('bpu_fw_version', '?')}",
            f"Cores={d.get('bpu_core_num', '?')}",
        ]
        lines.append(_color("System: ") + ", ".join(sys_parts))

        sched = d.get("scheduled_cores")
        if sched:
            lines.append(_color("Scheduled Cores: ") + str(list(sched)))

        lines.append(_color("Inputs Info: "))
        for inp in d.get("inputs", []):
            lines.append(_fmt_io_line(inp, "Input"))

        lines.append(_color("Outputs Info: "))
        for out in d.get("outputs", []):
            lines.append(_fmt_io_line(out, "Output"))

        # 编译配置（compile_config，由 Python wrapper 从 model_desc 的 JSON 解析）放最后一段.
        cfg = d.get("compile_config")
        if cfg:
            lines.append(render_compile_config(cfg))

        lines.append("====================================================")
        return "\n".join(lines)
    except Exception as e:  # 渲染失败也不抛，退回 repr
        return f"ModelSummary(<render error: {e}; data={d!r}>)"


def render_benchmark(d: dict[str, Any]) -> str:
    """把 benchmark dict 渲染成 'Inference Info' + Time 多单位行."""
    try:
        lines = [_color("Inference Info: ")]
        for key, unit in (
            ("time_us", "us"),
            ("time_ms", "ms"),
            ("time_s", "s"),
            ("time_min", "min"),
        ):
            v = d.get(key)
            if v is None:
                continue
            try:
                lines.append(f"  Time: {float(v):.6g} {unit}")
            except (TypeError, ValueError):
                lines.append(f"  Time: {v} {unit}")
        return "\n".join(lines)
    except Exception as e:
        return f"BenchmarkResult(<render error: {e}; data={d!r}>)"


_PLATFORM_TEMP_SENSOR = "pvt_bpu_pvtc1_t1"  # p.s() 温度只展示这个 BPU 主传感器
_LABEL_W = 17  # p.s() label 字段宽（最长 'ION cma_reserved:'=17）：静态/Realtime 段同宽对齐


def _mem_used_total(used_bytes: int, total_bytes: int) -> str:
    """used/total 字节 → 'used/total MB (pct %)'，1 位小数.

    定宽对齐：used / total 右对齐到 7（覆盖 S 系列 ≤43503.0 MB），pct 右对齐到
    '100.0' 宽（5），使 3 行 ION + Mem 的 % 括号左右都齐.读不到用 '?'（仍占定宽）.
    """
    um = f"{used_bytes / 1048576:.1f}" if used_bytes >= 0 else "?"
    tm = f"{total_bytes / 1048576:.1f}" if total_bytes >= 0 else "?"
    pct = f"{used_bytes / total_bytes * 100:.1f}" if (used_bytes >= 0 and total_bytes > 0) else "?"
    return f"{um:>7}/{tm:>7} MB ({pct:>5} %)"


def render_platform(d: dict[str, Any]) -> str:
    """把 platform dict 渲染成紧凑彩色块（静态 + Realtime 段）.

    p.s() 展示是 curated 子集：温度只取 BPU 主传感器 _PLATFORM_TEMP_SENSOR，
    不显示 BPU/CPU 频率、电压、静态 Memory 总量；内存统一 used/total MB (pct).
    summary() dict 仍含全量动态字段（温度 19 / 电压 28 全量等），数据契约不变.
    """
    try:
        lines = [_color("Platform: ")]
        rows = [
            ("SoC", d.get("soc_name", "?")),
            ("BPU Type", d.get("bpu_type", "?")),
            ("DNN Version", d.get("dnn_version", "?")),
            ("BPU Version", d.get("bpu_version", "?")),
            ("Physical Cores", d.get("physical_core_num", "?")),
            ("CPU", f"{d.get('cpu_model', '?')} x{d.get('cpu_count', '?')}"),
            ("UCP Library", d.get("ucp_library_path") or "<unknown>"),
        ]
        for label, val in rows:
            lines.append(f"  {label + ':':{_LABEL_W}s} {val}")

        # ── Realtime（动态，summary() 调用时现读 sysfs 放进 dict）──
        rt = []
        br = d.get("bpu_rate") or []
        if br:
            rt.append(("BPU Rate", f"{br} %"))
        # 温度：仅 BPU 主传感器（19 个全量在 dict）
        temp = d.get("temperature") or {}
        tval = temp.get(_PLATFORM_TEMP_SENSOR)
        if isinstance(tval, (int, float)) and tval >= 0:
            rt.append(("Temp", f"{tval:.1f} °C ({_PLATFORM_TEMP_SENSOR})"))
        # 内存统一 used/total MB (pct)，1 位小数（ION 三块 + 系统内存）
        ion = d.get("ion_memory") or {}
        for h, vt in ion.items():
            if isinstance(vt, dict):
                rt.append((f"ION {h}", _mem_used_total(vt.get("used", -1), vt.get("total", -1))))
        mtot = d.get("mem_total_mb")
        mavail = d.get("mem_available_mb")
        if isinstance(mtot, int) and mtot >= 0 and isinstance(mavail, int) and mavail >= 0:
            rt.append(("Mem Used", _mem_used_total((mtot - mavail) * 1048576, mtot * 1048576)))

        if rt:
            lines.append(_color("── Realtime ──"))
            for label, val in rt:
                lines.append(f"  {label + ':':{_LABEL_W}s} {val}")

        return "\n".join(lines)
    except Exception as e:
        return f"Platform(<render error: {e}; data={d!r}>)"


# ----------------------------------------------------------------------------
# 编译配置（compile_config）：从 model_desc 的 JSON 解析出的 mapper config dict
# ----------------------------------------------------------------------------


def _skip_config_value(v: Any) -> bool:
    """是否跳过该配置项：None / 空串 / 字符串 "[]""{}" / 空 list / 空 dict."""
    if v is None or v == "":
        return True
    if isinstance(v, str) and v in ("[]", "{}"):
        return True
    if isinstance(v, (list, dict)) and not v:
        return True
    return False


# 分组与字段顺序（只挑有意义的展示项；null/空值由 _skip_config_value 过滤）.
_COMPILE_CONFIG_GROUPS = [
    ("Toolchain", ["BUILDER_VERSION", "HBDK_VERSION", "HBDK_RUNTIME_VERSION", "HMCT_VERSION"]),
    (
        "Model",
        [
            "MARCH",
            "MODEL_PREFIX",
            "ONNX_MODEL",
            "INPUT_NAMES",
            "INPUT_SHAPE",
            "INPUT_SOURCE",
            "WORKING_DIR",
        ],
    ),
    (
        "Preprocess",
        [
            "INPUT_TYPE_RT",
            "INPUT_TYPE_TRAIN",
            "INPUT_LAYOUT_TRAIN",
            "INPUT_LAYOUT_RT",
            "INPUT_SPACE_AND_RANGE",
            "NORM_TYPE",
            "MEAN_VALUE",
            "SCALE_VALUE",
            "STD_VALUE",
        ],
    ),
    (
        "Quantization",
        ["CALI_TYPE", "CAL_DATA_DIR", "PER_CHANNEL", "MAX_PERCENTILE", "QUANT_CONFIG", "ADVICE"],
    ),
    (
        "Compile",
        [
            "OPTIMIZE_LEVEL",
            "COMPILE_MODE",
            "CORE_NUM",
            "MAX_TIME_PER_FC",
            "BALANCE_FACTOR",
            "max_l2m_size",
            "cache_mode",
            "DEBUG",
        ],
    ),
]


def render_compile_config(d: dict[str, Any]) -> str:
    """compile_config dict → 分组彩色块（Toolchain/Model/Preprocess/Quantization/Compile）.

    跳过 None / 空 / "[]" / "{}".返回 header + 分组行（无尾部分隔线，由 render_summary 收尾）.
    """
    lines = [_color("================== Compile Config ==================", "red")]
    for title, keys in _COMPILE_CONFIG_GROUPS:
        rows = [(k, d.get(k)) for k in keys if not _skip_config_value(d.get(k))]
        if not rows:
            continue
        lines.append(_color(f"{title}: "))
        for k, v in rows:
            lines.append(f"  {k + ':':22s} {v}")
    return "\n".join(lines)
