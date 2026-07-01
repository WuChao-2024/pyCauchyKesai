"""D3 IO 元数: 多输入/多输出的绑定顺序与数量契约(用 mixed_io 2in/2out)。"""
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


def _mixed():
    return H.select_entries(family="mixed_io", variant="large", precision="int16",
                            core_num=1, multi_io=True)


@pytest.mark.dim3
@pytest.mark.parametrize("entry", _mixed(), ids=[e["cid"] for e in _mixed()])
def test_d3_multi_output_count_and_names(bpu_free, entry):
    """多输出模型: 输出数 == meta, 输出名与 summary 一致。"""
    m = H.load_model(entry)
    sumry = m.summary()
    assert len(m.output_names) == len(entry["outputs"]) == 2
    assert m.output_names == [o["name"] for o in sumry["outputs"]]
    assert m.input_names == [i["name"] for i in sumry["inputs"]]


@pytest.mark.dim3
@pytest.mark.parametrize("entry", _mixed(), ids=[e["cid"] for e in _mixed()])
def test_d3_input_order_matters(bpu_free, entry):
    """多输入绑定顺序敏感: 调换两个输入顺序后输出应不同(证明不是无序绑定)。"""
    m = H.load_model(entry)
    a, b = H.sample_inputs_raw(entry, 0)            # 正确顺序 input_a, input_b
    out_correct = P.run_sync(m, [a, b])
    # 调换顺序(两个输入 shape 不同时几乎必然报错或得不同结果)
    swapped_ok = False
    try:
        m2 = H.load_model(entry)
        out_swapped = P.run_sync(m2, [b, a])
        swapped_ok = not np.array_equal(out_correct[0], out_swapped[0])
    except Exception:
        swapped_ok = True   # shape 不同直接报错也算"顺序有影响"
    assert swapped_ok, "调换输入顺序后输出仍相同, 绑定可能无序"


@pytest.mark.dim3
@pytest.mark.parametrize("entry", _mixed(), ids=[e["cid"] for e in _mixed()])
def test_d3_multi_io_zerocopy(bpu_free, entry):
    """多 IO 零拷贝: IONArray(input_descs)×2/inputs[=]×2 路由正确, 结果 == 同步。"""
    inputs = H.sample_inputs_raw(entry, 0)
    ref = P.run_sync(H.load_model(entry), inputs)
    zc = P.run_zerocopy(H.load_model(entry), inputs)
    assert len(zc) == 2
    for i, (r, o) in enumerate(zip(ref, zc)):
        P.assert_identical(r, o, strict=True, label=f"{entry['cid']} multi-io zerocopy out{i}")
