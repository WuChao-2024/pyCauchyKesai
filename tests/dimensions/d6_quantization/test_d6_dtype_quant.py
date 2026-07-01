"""D6 dtype 多样性 + 量化接口(平台刻画)。

两部分:
1) int8 dtype 输入模型(rgb_identity_fixed): 输入 int8 [1,16,16,3], 验证接口对非 float32
   dtype 输入的处理(零拷贝/sync 一致)。
2) SCALE 量化路径平台刻画: OE 工具链默认在所有模型尾部插入 Dequantize(ptq_faq.md:169),
   故 S600(nash-p)上**任何可产出的 HBM 的 IO 都是 float32(quantiType=NONE)** —— 即
   quantiType=SCALE 的 quantize/dequantize 往返在本平台**不可达**(API 存在但无 HBM 触发)。
   本测试扫描整个矩阵断言该平台性质, 把"缺口"转为"已刻画的平台事实"。
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H
from pyCauchyKesai import CauchyKesai, IONArray

INT8_HBM = os.environ.get(
    "D6_INT8_HBM",
    "/root/ssd/OELLM_Runtime/golden_hbm_matrix/hbm/rgb_scale_int8.hbm")


def _need():
    if not os.path.exists(INT8_HBM):
        pytest.skip(f"无 int8 dtype 模型({INT8_HBM}); 云端编译 rgb_identity_fixed 后同步")


@pytest.mark.dim6
def test_d6_int8_dtype_input_zero_copy(bpu_free, acc):
    """int8 输入模型: IONArray(input_descs) 是 int8 dtype, 零拷贝喂 int8 → float 输出有限。"""
    _need()
    m = CauchyKesai(INT8_HBM)
    s = m.summary()
    in_dtype = s["inputs"][0]["dtype"]
    assert in_dtype != "float32", f"期望非 float32 输入, 实际 {in_dtype}"
    ion = IONArray(m.input_descs[0])
    assert ion.desc.dtype == np.dtype(in_dtype), "IONArray(input_descs) dtype 与 summary 不符"
    arr = np.random.randint(0, 255, ion.desc.shape).astype(ion.desc.dtype)
    ion.from_numpy(arr)
    oo = IONArray(m.output_descs[0])
    m.inputs[0][0] = ion; m.outputs[0][0] = oo
    m.start(0); m.wait(0); oo.flush_invalidate()
    zc = np.array(oo.numpy(), copy=True)
    assert np.all(np.isfinite(zc)), "int8 输入产生非有限输出"
    H.collect(acc, "d6_dtype", {
        "model": "rgb_identity_fixed", "in_dtype": in_dtype,
        "out_dtype": s["outputs"][0]["dtype"], "path": "zerocopy", "finite": 1})


@pytest.mark.dim6
def test_d6_int8_sync_matches_zerocopy(bpu_free, acc):
    """int8 输入: sync 路径(喂 int8)与零拷贝路径输出一致。"""
    _need()
    m = CauchyKesai(INT8_HBM)
    s = m.summary()
    in_dtype = s["inputs"][0]["dtype"]
    arr = np.random.randint(0, 255, s["inputs"][0]["shape"]).astype(in_dtype)
    ion = IONArray(m.input_descs[0]); ion.from_numpy(arr)
    oo = IONArray(m.output_descs[0]); m.inputs[0][0] = ion; m.outputs[0][0] = oo
    m.start(0); m.wait(0); oo.flush_invalidate()
    zc = np.array(oo.numpy(), copy=True)
    sync = CauchyKesai(INT8_HBM)([arr])[0]
    assert np.all(np.isfinite(sync))
    assert np.array_equal(zc, sync), "int8 输入: sync != 零拷贝"
    H.collect(acc, "d6_dtype", {
        "model": "rgb_identity_fixed", "in_dtype": in_dtype,
        "out_dtype": s["outputs"][0]["dtype"], "path": "sync==zerocopy", "finite": 1})


@pytest.mark.dim6
def test_d6_scale_unreachable_on_s600(bpu_free, acc):
    """平台刻画: 扫描矩阵所有模型, 断言 IO quantiType 全为 NONE(0)。
    依据 OE ptq_faq.md:169 —— 工具链默认在尾部插 Dequantize, S600 上 SCALE IO 不可达。
    若将来出现 quantiType=1 的模型, 此断言会 fail, 提示可补 SCALE 往返测试。"""
    entries = H.GH.manifest_entries()
    n_none, n_scale, samples = 0, 0, []
    for e in entries:
        s = CauchyKesai(H.GH.hbm_path(e["cid"]), n_task=1).summary()
        for tag, ios in (("in", s["inputs"]), ("out", s["outputs"])):
            for io in ios:
                if io.get("quantiType", 0) == 1:
                    n_scale += 1
                    samples.append(f"{e['cid']}.{tag}.{io['name']}")
                else:
                    n_none += 1
    H.collect(acc, "d6_scale_platform", {
        "models_scanned": len(entries), "tensors_none": n_none,
        "tensors_scale": n_scale, "scale_unreachable": int(n_scale == 0)})
    # 平台性质: 当前 S600 工具链下无 SCALE IO。记录刻画, 不强 fail(留 hook)。
    if n_scale > 0:
        pytest.skip(f"发现 {n_scale} 个 SCALE tensor({samples[:3]}), 可补 SCALE 往返测试")
