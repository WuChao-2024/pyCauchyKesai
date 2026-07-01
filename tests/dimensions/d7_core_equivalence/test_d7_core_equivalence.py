"""D7 核数等价: core1 vs core2 同模型同输入输出一致 + 调度核数约束。"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H
import _paths as P


def _pairs():
    """(core1, core2) 同 model_id/precision/variant 配对(单IO 3 族 small/int16)。"""
    pairs = []
    for fam in ("conv_stack", "matmul_stack", "transformer_block"):
        c1 = H.select_entries(family=fam, variant="small", precision="int16",
                              core_num=1, single_io=True)
        c2 = H.select_entries(family=fam, variant="small", precision="int16",
                              core_num=2, single_io=True)
        for a in c1:
            for b in c2:
                if a["model_id"] == b["model_id"]:
                    pairs.append((a, b))
    return pairs


@pytest.mark.dim7
@pytest.mark.parametrize("e1,e2", _pairs(), ids=[f"{a['model_id']}" for a, b in _pairs()])
def test_d7_core1_equals_core2(bpu_free, acc, e1, e2):
    """同模型 core1 vs core2, 同输入, 输出逐元素相等(BPU 确定性 → 跨核应 bit-identical)。"""
    inputs = H.sample_inputs_raw(e1, 0)
    o1 = P.run_sync(H.load_model(e1), inputs)
    o2 = P.run_sync(H.load_model(e2), inputs)
    bit = True
    for i, (a, b) in enumerate(zip(o1, o2)):
        if not np.array_equal(a, b):
            bit = False
        d = P.assert_identical(a, b, strict=True, label=f"{e1['model_id']} core1 vs core2 out{i}")
        H.collect(acc, "d7_core_diff", {
            "model_id": e1["model_id"], "out_idx": i,
            "max_abs_diff": d, "bit_identical": int(bit)})


@pytest.mark.dim7
@pytest.mark.parametrize("entry",
    H.select_entries(family="matmul_stack", variant="small", precision="int16", core_num=2),
    ids=lambda e: e["cid"])
def test_d7_wrong_corecount_rejected(bpu_free, entry):
    """core2 模型若只指定 1 核 → set_scheduling_params 调用时即拒(核数 != 编译核数, 前置校验)。"""
    m = H.load_model(entry)
    with pytest.raises(Exception):
        m.set_scheduling_params([0])    # 调用时即抛 ValueError, 不再 deferred 到 SubmitTask


@pytest.mark.dim7
@pytest.mark.parametrize("entry",
    H.select_entries(family="matmul_stack", variant="small", precision="int16", core_num=1),
    ids=lambda e: e["cid"])
def test_d7_empty_resets_coreany(bpu_free, entry):
    """set_scheduling_params([]) 重置为 CORE_ANY, 单核模型仍可推理。"""
    m = H.load_model(entry)
    m.set_scheduling_params([])
    inputs = H.sample_inputs_raw(entry, 0)
    out = m(inputs)
    assert np.all(np.isfinite(out[0]))
