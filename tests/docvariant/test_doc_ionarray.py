"""文档 IONArray / IONArrayDesc / IONMemory 原子能力(结合模型模板 desc)节测试。

覆盖:
  - IONMemory 构造/size/is_allocated/numpy(uint8 raw)/flush(uncached no-op)
  - IONArrayDesc 构造/成员/计算方法(ndim/size/nbytes/has_stride/is_padded_layout/
    has_tensor_properties/is_quantized);shape<=0 ValueError
  - 模型模板 desc 性质齐全(aligned_byte_size/has_tensor_properties)
  - IONArray 重载1(裸)/重载2(从 desc)/defer+allocate/重复 allocate RuntimeError
  - from_numpy 往返 + dtype 不匹配 ValueError
  - __array__ 协议零拷贝(np.asarray/np.shares_memory)
  - from_memory 偏移隔离 + 越界 IndexError
  - dequantize/quantize(NONE passthrough/memcpy;SCALE 若平台可达单独验证)
  - BPU dtype 对照(uint8/int8/float32/...)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
import conftest as C
from pyCauchyKesai import IONArray, IONArrayDesc, IONMemory


@pytest.mark.docvar
class TestIONMemory:
    def test_alloc_size(self, bpu_free):
        m = IONMemory(1024)
        assert m.size == 1024
        assert m.is_allocated is True

    def test_numpy_uint8_raw(self, bpu_free):
        m = IONMemory(256)
        arr = m.numpy()
        assert arr.dtype == np.uint8
        assert arr.shape == (256,)
        arr[:] = np.arange(256, dtype=np.uint8)
        assert np.array_equal(m.numpy(), np.arange(256, dtype=np.uint8))

    def test_flush_uncached_noop(self, bpu_free):
        m = IONMemory(64, cached=False)
        m.numpy()[:] = 1
        m.flush_clean()
        m.flush_invalidate()           # 不抛(uncached 静默 no-op)


@pytest.mark.docvar
class TestIONArrayDesc:
    def test_construct_and_methods(self, bpu_free):
        d = IONArrayDesc(np.dtype("float32"), [1, 3, 8, 8])
        assert d.ndim() == 4
        assert d.size() == 1 * 3 * 8 * 8
        assert d.nbytes() == 1 * 3 * 8 * 8 * 4
        assert d.has_stride() is False            # 裸构造 natural
        assert d.is_padded_layout() is False
        assert d.has_tensor_properties() is False  # 裸构造 tensor_type=-1
        assert d.is_quantized() is False

    def test_shape_le_zero_raises(self, bpu_free):
        with pytest.raises(ValueError):
            IONArrayDesc(np.dtype("float32"), [1, 0, 8])

    def test_model_desc_has_properties(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        d = m.input_descs[0]
        assert d.has_tensor_properties() is True
        assert d.aligned_byte_size > 0
        assert tuple(d.shape) == tuple(m.summary()["inputs"][0]["shape"])

    def test_member_equality(self, bpu_free):
        # 文档:成员级 ==(无整体 __eq__)
        d1 = IONArrayDesc(np.dtype("float32"), [1, 4])
        d2 = IONArrayDesc(np.dtype("float32"), [1, 4])
        assert d1.dtype == d2.dtype
        assert d1.shape == d2.shape


@pytest.mark.docvar
class TestIONArray:
    def test_overload1_bare(self, bpu_free):
        ion = IONArray(np.dtype("float32"), [1, 4])
        assert ion.is_allocated
        assert tuple(ion.desc.shape) == (1, 4)

    def test_overload2_from_desc(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        ion = IONArray(m.input_descs[0])
        assert ion.is_allocated
        assert ion.desc.aligned_byte_size > 0

    def test_defer_allocate(self, bpu_free):
        ion = IONArray(np.dtype("float32"), [1, 4], defer=True)
        assert ion.is_allocated is False
        ion.allocate()
        assert ion.is_allocated is True
        with pytest.raises(RuntimeError):
            ion.allocate()                          # 重复分配

    def test_from_numpy_roundtrip(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        ion = IONArray(m.input_descs[0])
        data = np.random.randn(*tuple(ion.desc.shape)).astype(np.float32)
        ion.from_numpy(data)
        assert np.allclose(np.asarray(ion), data, atol=1e-5)

    def test_from_numpy_dtype_mismatch_raises(self, bpu_free):
        ion = IONArray(np.dtype("float32"), [1, 4])
        with pytest.raises(ValueError):
            ion.from_numpy(np.zeros((1, 4), np.int32))

    def test_from_numpy_shape_mismatch_raises(self, bpu_free):
        ion = IONArray(np.dtype("float32"), [1, 4])
        with pytest.raises((ValueError, IndexError)):
            ion.from_numpy(np.zeros((1, 5), np.float32))

    def test_array_protocol_zerocopy(self, bpu_free):
        ion = IONArray(np.dtype("float32"), [1, 4])
        ion.numpy()[:] = 1.0
        a = np.asarray(ion)
        assert np.shares_memory(a, ion.numpy())     # 零拷贝铁证
        a.flat[0] = 9.0
        assert ion.numpy().flat[0] == 9.0            # 改 a 同步到 ion

    def test_from_memory_offset_isolation(self, bpu_free):
        mem = IONMemory(8192)
        tpl = IONArrayDesc(np.dtype("float32"), [1024])   # 4096B
        a = IONArray.from_memory(mem, 0, tpl)
        b = IONArray.from_memory(mem, 4096, tpl)
        assert a.memory is b.memory is mem           # 共享同一 IONMemory
        a.from_numpy(np.ones(1024, np.float32))
        b.from_numpy(np.zeros(1024, np.float32))
        assert a.numpy().flat[0] == 1.0
        assert b.numpy().flat[0] == 0.0              # 偏移隔离

    def test_from_memory_oob_raises(self, bpu_free):
        mem = IONMemory(128)
        tpl = IONArrayDesc(np.dtype("float32"), [4])
        with pytest.raises(IndexError):
            IONArray.from_memory(mem, 200, tpl)      # offset > size

    def test_from_memory_none_memory_raises(self, bpu_free):
        tpl = IONArrayDesc(np.dtype("float32"), [4])
        with pytest.raises(ValueError):
            IONArray.from_memory(None, 0, tpl)


@pytest.mark.docvar
class TestQuantize:
    def test_dequantize_none_passthrough(self, factory, bpu_free):
        # S600 IO 恒 float32(quantiType=NONE):dequantize passthrough 原 dtype
        m = factory(C.ENTRIES[0])
        out = m(C.make_inputs(m))
        ion = IONArray(m.output_descs[0])
        ion.from_numpy(out[0].astype(np.float32))
        dq = ion.dequantize()
        assert dq.dtype == np.float32               # NONE → passthrough

    def test_quantize_none_memcpy(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        ion = IONArray(m.input_descs[0])
        data = np.random.randn(*tuple(ion.desc.shape)).astype(np.float32)
        ion.quantize(data)                            # NONE → memcpy
        assert np.allclose(np.asarray(ion), data, atol=1e-5)

    def test_scale_path_if_reachable(self, factory, bpu_free):
        # 若 scaleout entry 编译成功且输出 quantiType=SCALE,验证 SCALE 反量化路径
        e = C.by_name("matmul_scaleout_c1")
        if e is None:
            pytest.skip("无 scaleout entry")
        m = factory(e)
        outs = m(C.make_inputs(m))
        s = m.summary()
        qts = [o["quantiType"] for o in s["outputs"]]
        if 1 not in qts:
            pytest.skip("S600 工具链仍插 Dequantize,SCALE 输出不可达(平台事实)")
        # SCALE 可达:dequantize 应返回 float32 新数组
        ion = IONArray(m.output_descs[0])
        ion.from_numpy(outs[0])
        dq = ion.dequantize()
        assert dq.dtype == np.float32
