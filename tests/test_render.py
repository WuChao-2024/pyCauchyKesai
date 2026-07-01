"""tests/test_render.py — _render 渲染层 CPU 单测（无需 BPU / native .so）。

_render.py 无包内依赖（仅 sys + 懒加载 numpy），故用 importlib 按路径直接加载，
绕过 pyCauchyKesai/__init__.py（其 import native .so，本机未编译）。
本机任意带 numpy 的 env 即可跑：python -m pytest tests/test_render.py -q
"""
import io
import json
import os
import sys
import importlib.util

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RENDER_PATH = os.path.join(_HERE, "..", "src", "pyCauchyKesai", "_render.py")
_spec = importlib.util.spec_from_file_location("pyck_render", _RENDER_PATH)
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)


def _fake_summary():
    """仿 native CauchyKesai.summary() 的 dict（含 numpy scale/zero_point + 嵌套 platform）。"""
    return {
        "model_path": "/opt/models/demo.hbm",
        "model_names": [{"index": 0, "name": "demo", "selected": True}],
        "n_task": 2,
        "memory_mb": 12.5,
        "bpu_core_num": 2,
        "scheduled_cores": [0, 1],
        "bpu_fw_version": "1.1.26",
        "platform": {"soc_name": "S600", "bpu_type": "Nash-p", "dnn_version": "1.x",
                     "bpu_version": "2.x", "physical_core_num": 6, "cpu_model": "a76",
                     "cpu_count": 8, "mem_total_mb": 8192},
        "inputs": [{
            "index": 0, "name": "images", "dtype": "uint8", "tensorType": 2,
            "alignedByteSize": 409600, "quantiType": 1, "quantizeAxis": 0,
            "desc": "N/A", "shape": [1, 640, 640, 1], "stride": [640],
            "scale": np.array([[[[2.0]]]], dtype=np.float32),
            "zero_point": np.array([[[[128]]]], dtype=np.int32),
        }],
        "outputs": [{
            "index": 0, "name": "logits", "dtype": "float32", "tensorType": 1,
            "alignedByteSize": 8000, "quantiType": 0, "quantizeAxis": 0,
            "desc": "N/A", "shape": [1, 1000], "stride": [1000],
            "scale": np.array([], dtype=np.float32),
            "zero_point": np.array([], dtype=np.int32),
        }],
    }


# ── _jsonable（AI-Native dict）─────────────────────────────────────────────

def test_jsonable_converts_numpy_and_is_json_serializable():
    out = _render._jsonable(_fake_summary())
    assert isinstance(out, dict)
    inp = out["inputs"][0]
    assert isinstance(inp["scale"], list)        # np.ndarray → list
    assert isinstance(inp["zero_point"], list)
    assert isinstance(inp["shape"], list)
    # 整 dict 可 json.dumps（这是 AI-Native 的核心保证）
    s = json.dumps(out, ensure_ascii=False)
    assert "demo.hbm" in s and "images" in s


def test_jsonable_numpy_scalar():
    assert _render._jsonable(np.float32(1.5)) == 1.5
    assert isinstance(_render._jsonable(np.int64(3)), int)


# ── render_summary / render_benchmark / render_platform ─────────────────────

def test_render_summary_returns_str_with_labels():
    s = _render.render_summary(_fake_summary())
    assert isinstance(s, str)
    for label in ("Model Summary", "Model File", "images", "logits",
                  "Task N", "Memory", "System", "Inputs Info", "Outputs Info"):
        assert label in s, f"render_summary 缺标签: {label}"


def test_render_summary_survives_incomplete_dict():
    s = _render.render_summary({"model_path": "x"})
    assert isinstance(s, str) and "Model Summary" in s


def test_render_benchmark_returns_str():
    s = _render.render_benchmark({"time_us": 12345.0, "time_ms": 12.345,
                                  "time_s": 0.012345, "time_min": 2.0e-4})
    assert isinstance(s, str)
    assert "Time" in s and "us" in s and "ms" in s and "s" in s


def test_render_platform_returns_str():
    s = _render.render_platform({"soc_name": "S600", "bpu_type": "Nash-p",
                                 "dnn_version": "1", "bpu_version": "2",
                                 "physical_core_num": 6, "cpu_model": "a76",
                                 "cpu_count": 8, "mem_total_mb": 8192})
    assert isinstance(s, str) and "Platform" in s and "S600" in s


# ── p.s() 对齐：内存 % 括号齐 + label 字段跨段齐 ──────────────────────────────

def test_mem_used_total_aligns_parens():
    # 不同量级 used/total，'(pct %)' 左右括号都对齐：'(' 列、')' 列各自相同
    a = _render._mem_used_total(0, 1024 * 1048576)                              # (  0.0 %)
    b = _render._mem_used_total(int(4915.0 * 1048576), int(43503.0 * 1048576))  # ( 11.3 %)
    c = _render._mem_used_total(int(11.9 * 1048576), 1024 * 1048576)            # (  1.2 %)
    assert a.endswith(" %)") and b.endswith(" %)")
    assert a.index("(") == b.index("(") == c.index("(")
    assert a.rindex(")") == b.rindex(")") == c.rindex(")")


def test_render_platform_memory_parens_align():
    d = {
        "soc_name": "S600", "bpu_type": "nash-p", "dnn_version": "1",
        "bpu_version": "2", "physical_core_num": 4, "cpu_model": "a76",
        "cpu_count": 8, "mem_total_mb": 43503, "mem_available_mb": 43503 - 4915,
        "ion_memory": {
            "ion_cma": {"used": 0, "total": 1024 * 1048576},
            "cma_reserved": {"used": int(11.9 * 1048576), "total": 1024 * 1048576},
            "carveout": {"used": int(12.9 * 1048576), "total": 16384 * 1048576},
        },
    }
    s = _render.render_platform(d)
    mem_lines = [l for l in s.splitlines() if "ION " in l or "Mem Used:" in l]
    assert len(mem_lines) == 4
    # 4 行内存的 '(' 与 ')' 各自在同一列
    assert len({l.index("(") for l in mem_lines}) == 1
    assert len({l.rindex(")") for l in mem_lines}) == 1


def test_render_platform_value_column_aligned():
    # 静态段（SoC / UCP Library）与 Realtime 段（Temp）值起始列一致
    d = {
        "soc_name": "S600", "bpu_type": "nash-p", "dnn_version": "1",
        "bpu_version": "2", "physical_core_num": 4, "cpu_model": "a76",
        "cpu_count": 8, "mem_total_mb": 43503, "mem_available_mb": 40000,
        "ucp_library_path": "/root/libhbucp.so",
        "temperature": {"pvt_bpu_pvtc1_t1": 44.6},
    }
    s = _render.render_platform(d)
    lines = s.splitlines()

    def val_col(line):  # ':' 后首个非空格字符的列（值起始列）
        i = line.index(":") + 1
        while i < len(line) and line[i] == " ":
            i += 1
        return i

    soc = next(l for l in lines if l.strip().startswith("SoC:"))
    ucp = next(l for l in lines if l.strip().startswith("UCP Library:"))
    temp = next(l for l in lines if l.strip().startswith("Temp:"))
    assert val_col(soc) == val_col(ucp) == val_col(temp)


# ── render_compile_config（编译配置块，从 model_desc JSON 解析）──────────────────

def _fake_compile_config():
    """仿 summary()['compile_config']：从 model_desc JSON 解析出的 mapper config dict。"""
    return {
        "BUILDER_VERSION": "3.5.3", "HBDK_VERSION": "4.7.5",
        "HBDK_RUNTIME_VERSION": None, "HMCT_VERSION": "2.6.5",
        "MARCH": "nash-p", "MODEL_PREFIX": "resnet18_224x224_nv12",
        "ONNX_MODEL": "/x/resnet18.onnx", "INPUT_NAMES": "input",
        "INPUT_SHAPE": "1x3x224x224", "INPUT_SOURCE": {"input": "pyramid"},
        "INPUT_TYPE_RT": "nv12", "INPUT_LAYOUT_RT": "",   # 空串应被跳过
        "STD_VALUE": "[]",                                 # "[]" 应被跳过
        "CALI_TYPE": "default", "PER_CHANNEL": "False",    # Quantization 组
        "CORE_NUM": 1, "MAX_TIME_PER_FC": 0, "max_l2m_size": 0,
        "COMPILE_MODE": "latency", "OPTIMIZE_LEVEL": "O2",
    }


def test_render_compile_config_groups_and_skips():
    s = _render.render_compile_config(_fake_compile_config())
    assert isinstance(s, str)
    for title in ("Toolchain", "Model", "Preprocess", "Quantization", "Compile"):
        assert title in s, f"render_compile_config 缺分组: {title}"
    # 关键字段出现
    assert "MAX_TIME_PER_FC" in s and "CORE_NUM" in s and "nash-p" in s
    # 空值 / None 被跳过
    assert "STD_VALUE" not in s and "INPUT_LAYOUT_RT" not in s
    assert "HBDK_RUNTIME_VERSION" not in s


def test_render_summary_compile_config_at_end():
    d = _fake_summary()
    d["compile_config"] = _fake_compile_config()
    s = _render.render_summary(d)
    # Compile Config 块出现，且在 Outputs Info 之后（最后一段）
    assert "Compile Config" in s
    assert s.index("Compile Config") > s.index("Outputs Info")


# ── _color TTY 行为（返回字符串按 sys.stdout.isatty() 决定是否嵌 ANSI）─────────

class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_color_plain_when_not_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert _render._color("hi") == "hi"
    assert "\033[" not in _render.render_summary({"model_path": "x"})


def test_color_ansi_when_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdout", _FakeTTY())
    assert "\033[" in _render._color("hi", "cyan")
    assert "\033[" in _render.render_summary({"model_path": "x"})
