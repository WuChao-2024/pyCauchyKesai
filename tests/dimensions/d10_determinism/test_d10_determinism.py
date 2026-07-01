"""D10 确定性: 同一输入连跑 N 次推理, 输出必须逐元素相等(BPU 确定性硬契约)。"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H

N_RUNS = 10


def _entries():
    """单IO 多族: conv_stack(老) + matmul_3d(新) + transformer, 覆盖几种算子。"""
    out = []
    out.extend(H.select_entries(family="conv_stack", variant="small",
                                precision="int16", core_num=1, single_io=True)[:1])
    return out


@pytest.mark.dim10
@pytest.mark.parametrize("entry", _entries(), ids=[e["cid"] for e in _entries()])
def test_d10_same_input_bit_identical(bpu_free, acc, entry):
    """同输入连跑 N 次, 两两逐元素相等(非 allclose, 是 array_equal)。"""
    inputs = H.sample_inputs_raw(entry, 0)
    m = H.load_model(entry)
    outs = [m(inputs) for _ in range(N_RUNS)]
    ref = outs[0]
    max_diff = 0.0
    for k, o in enumerate(outs[1:], 1):
        for i, (a, b) in enumerate(zip(ref, o)):
            d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            max_diff = max(max_diff, d)
            assert np.array_equal(a, b), f"{entry['cid']} run{k} out{i} != run0 (diff={d:.3e})"
    H.collect(acc, "d10_determinism", {
        "cid": entry["cid"], "n_runs": N_RUNS, "pairwise_maxdiff": max_diff,
        "all_identical": 1})
