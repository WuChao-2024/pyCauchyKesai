"""D5 内存/对齐(最大接口盲区: 零拷贝端到端 + padded/strided 布局)。

活体探针确认: IONArray(input_descs)/inputs[=]/outputs[=]/start()/wait/m.outputs 七个 API
此前从未被测试调用; 现有 mixed_io_small 是 padded 模型(输入 [1,8] 占 256 字节)。
"""
import os
import sys
import numpy as np
import pytest

from pyCauchyKesai import IONArray

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H
import _paths as P


def _models():
    """单IO 3 族 + 多IO mixed_io, 全 small/int16/core1。"""
    out = []
    for fam in ("conv_stack", "matmul_stack"):
        out.extend(H.select_entries(family=fam, variant="small", precision="int16",
                                    core_num=1, single_io=True))
    out.extend(H.select_entries(family="mixed_io", variant="small", precision="int16",
                                core_num=1, multi_io=True))
    return out


@pytest.mark.dim5
@pytest.mark.parametrize("entry", _models(), ids=[e["cid"] for e in _models()])
def test_d5_zerocopy_matches_sync(bpu_free, acc, entry):
    """零拷贝 7 步端到端输出 == 同步路径(bit-identical)。"""
    inputs = H.sample_inputs_raw(entry, 0)
    ref = P.run_sync(H.load_model(entry), inputs)
    zc = P.run_zerocopy(H.load_model(entry), inputs)
    assert len(zc) == len(ref)
    for i, (r, o) in enumerate(zip(ref, zc)):
        d = P.assert_identical(r, o, strict=True, label=f"{entry['cid']} zerocopy out{i}")
        H.collect(acc, "d5_zerocopy_diff", {
            "cid": entry["cid"], "out_idx": i, "max_abs_diff": d, "identical": 1})


@pytest.mark.dim5
@pytest.mark.parametrize("entry", _models(), ids=[e["cid"] for e in _models()])
def test_d5_from_numpy_numpy_roundtrip(bpu_free, acc, entry):
    """IONArray(input_descs) → from_numpy(data) → numpy() 往返 bit-exact(natural 与 padded 都验)。"""
    m = H.load_model(entry)
    inputs = H.sample_inputs_raw(entry, 0)
    sumry = m.summary()
    for i, arr in enumerate(inputs):
        io_meta = {
            "shape": sumry["inputs"][i]["shape"],
            "dtype": sumry["inputs"][i]["dtype"],
            "alignedByteSize": sumry["inputs"][i]["alignedByteSize"],
        }
        padded = H.is_padded(io_meta)
        ion = IONArray(m.input_descs[i])
        ion.from_numpy(np.ascontiguousarray(arr))
        back = np.array(ion.numpy(), copy=True)
        assert np.array_equal(arr, back), f"input{i} from_numpy→numpy 往返不一致(padded={padded})"
        H.collect(acc, "d5_layout_summary", {
            "cid": entry["cid"], "tensor": sumry["inputs"][i]["name"],
            "shape": str(io_meta["shape"]), "padded": int(padded),
            "natural_bytes": int(np.prod(io_meta["shape"])) * np.dtype(io_meta["dtype"]).itemsize,
            "aligned_bytes": io_meta["alignedByteSize"],
            "roundtrip_identical": 1,
        })


@pytest.mark.dim5
@pytest.mark.parametrize("entry", _models(), ids=[e["cid"] for e in _models()])
def test_d5_manual_flush_path(bpu_free, entry):
    """numpy()[:]=arr + 显式 flush_clean → 零拷贝结果正确(验证 flush 是用户契约且生效)。"""
    m = H.load_model(entry)
    inputs = H.sample_inputs_raw(entry, 0)
    n_in = len(m.input_names)
    n_out = len(m.output_names)
    ion_in = [IONArray(m.input_descs[i]) for i in range(n_in)]
    ion_out = [IONArray(m.output_descs[i]) for i in range(n_out)]
    for ion, arr in zip(ion_in, inputs):
        view = ion.numpy()             # 直写 buffer(不走 from_numpy)
        view[:] = np.ascontiguousarray(arr)
        ion.flush_clean()                 # 用户契约: 必须显式刷
    for i, ion in enumerate(ion_in):
        m.inputs[0][i] = ion
    for i, ion in enumerate(ion_out):
        m.outputs[0][i] = ion
    m.start(0); m.wait(0)
    ref = P.run_sync(H.load_model(entry), inputs)
    for ion, r in zip(ion_out, ref):
        ion.flush_invalidate()
        o = np.array(ion.numpy(), copy=True)
        P.assert_identical(r, o, strict=True, label=f"{entry['cid']} manual-flush")


@pytest.mark.dim5
@pytest.mark.parametrize("entry", _models(), ids=[e["cid"] for e in _models()])
def test_d5_bound_inputs_outputs_identity(bpu_free, entry):
    """inputs[0][i]=ion / outputs[0][i]=ion 后, m.inputs[0] / m.outputs[0] 返回同一对象(is)。"""
    m = H.load_model(entry)
    n_in = len(m.input_names)
    n_out = len(m.output_names)
    ion_in = [IONArray(m.input_descs[i]) for i in range(n_in)]
    ion_out = [IONArray(m.output_descs[i]) for i in range(n_out)]
    for i, ion in enumerate(ion_in):
        m.inputs[0][i] = ion
    for i, ion in enumerate(ion_out):
        m.outputs[0][i] = ion
    got_in = m.inputs[0]; got_out = m.outputs[0]
    assert all(a is b for a, b in zip(ion_in, got_in)), "m.inputs[0] 非同一对象"
    assert all(a is b for a, b in zip(ion_out, got_out)), "m.outputs[0] 非同一对象"
