"""Phase 4 + Phase 5 集成测试 — BPU required."""

import os
import sys

# tests/ 自身在 path（供 import _pipeline 等测试内部模块）。
# pyCauchyKesai 用已安装包（不在 path 注入 src/——避免 src/ 无编译 .so 时遮蔽已装包）。
_here = os.path.dirname(__file__)
if _here not in sys.path:
    sys.path.insert(0, _here)

import pytest
import numpy as np
from pyCauchyKesai import CauchyKesai, IONArray, IONMemory, from_numpy
from _pipeline import Pipeline   # 测试内部模块（已从主包移除）

MODEL_PATH = "/root/ssd/OELLM_Runtime/robotrea_model/v0.0.1/pi05_siglip_robotera_alpha_v0.0.1.hbm"


def _make_inputs(model):
    """根据 model summary 构造随机输入 (float16 pixel + int64 token)."""
    info = model.summary()
    inputs = []
    for inp in info["inputs"]:
        shape = tuple(inp["shape"])
        if "int" in inp.get("dtype", ""):
            inputs.append(np.zeros(shape, dtype=np.int64))
        else:
            inputs.append(np.random.randn(*shape).astype(np.float16))
    return inputs


# ============================================================================
# Phase 5: Context Manager
# ============================================================================
@pytest.mark.bpu
class TestContextManager:

    def test_context_manager_basic(self):
        """with CauchyKesai(model) as ck: 正常进出 + 推理."""
        with CauchyKesai(MODEL_PATH, n_task=1) as ck:
            inputs = _make_inputs(ck)
            outputs = ck(inputs)
            assert len(outputs) == len(ck.summary()["outputs"])
            assert outputs[0].shape == tuple(ck.summary()["outputs"][0]["shape"])
        # 退出后 ck 引用释放, GC 清理 C++ 资源 — 不 crash 即通过


# ============================================================================
# Phase 4: Pipeline 单模型
# ============================================================================
@pytest.mark.bpu
class TestPipelineSingle:

    def test_pipeline_single_model(self):
        """Pipeline([ck]) 等价单模型推理 (不触发 IONMemory 共享)."""
        model = CauchyKesai(MODEL_PATH, n_task=1)
        info = model.summary()
        inputs = _make_inputs(model)

        # 直接推理
        direct_out = model(inputs)

        # Pipeline 单模型
        pipe = Pipeline([model])
        pipe_out = pipe.pipe(inputs)

        assert len(direct_out) == len(pipe_out)
        for d, p in zip(direct_out, pipe_out):
            assert d.shape == p.shape
            if d.dtype == p.dtype:
                np.testing.assert_allclose(d, p, rtol=1e-6)


# ============================================================================
# Phase 4: Pipeline 双模型 — 受限标注
# ============================================================================
@pytest.mark.bpu
class TestPipelineDual:

    def test_pipeline_dual_limited(self):
        """双模型串联: siglip output (1,256,2048) 无匹配下游模型."""
        # 标注受限: 没有输入形状匹配 siglip output 的模型
        print("\n  [受限] 双模型串联: 无匹配模型对 "
              "(siglip output shape 需下游匹配, 当前只有单模型可用). "
              "Pipeline 逻辑已在单模型测试验证通过; "
              "from_memory 零拷贝路径在纯内存测试中验证 (test_ion_array.py).")
        pytest.skip("无匹配的双模型对, Pipeline 逻辑在单模型 + 纯内存测中覆盖")


# ============================================================================
# 纯 Python (不需 BPU)
# ============================================================================
class TestFromNumpy:

    def test_from_numpy_basic(self):
        """from_numpy 工厂: 构造 IONArray + 数据一致."""
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        ion = from_numpy(arr)
        assert tuple(ion.desc.shape) == (3, 4)
        assert ion.is_allocated
        assert np.array_equal(ion.numpy(), arr)

    def test_from_numpy_int64(self):
        """from_numpy int64 类型."""
        arr = np.arange(6, dtype=np.int64).reshape(2, 3)
        ion = from_numpy(arr)
        assert tuple(ion.desc.shape) == (2, 3)
        assert np.array_equal(ion.numpy(), arr)

    def test_from_numpy_non_contiguous(self):
        """from_numpy 非连续数组自动 ascontiguousarray."""
        arr = np.arange(24, dtype=np.float32).reshape(4, 6)
        sliced = arr[::2, ::3]  # 非连续
        ion = from_numpy(sliced)
        expected = np.ascontiguousarray(sliced)
        assert np.array_equal(ion.numpy(), expected)
