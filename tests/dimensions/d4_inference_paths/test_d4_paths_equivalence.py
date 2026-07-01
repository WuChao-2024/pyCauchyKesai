"""D4 推理路径 cross-path 一致性(核心接口契约)。

同一组输入, 4 条推理路径必须产出一致:
  - 同内存路径对(sync/async) → bit-identical(BPU 确定性)
  - 不同内存布局对(zerocopy/pipeline) → allclose(实测本批模型也 bit-identical)
另测 from_numpy 工厂往返 bit-exact。

所有路径对的 maxdiff 落 report/d4_paths_maxdiff.csv(刻画, 即使 bit-identical 也记录)。
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
import _paths as P


def _repr_single_io():
    """单输入单输出代表模型(3 族各 1 个 small/int16/core1)。"""
    out = []
    for fam in ("conv_stack", "matmul_stack", "transformer_block"):
        es = H.select_entries(family=fam, variant="small", precision="int16",
                              core_num=1, single_io=True)
        out.extend(es)
    return out


@pytest.mark.dim4
@pytest.mark.parametrize("entry", _repr_single_io(),
                         ids=[e["cid"] for e in _repr_single_io()])
def test_d4_all_paths_match_sync(bpu_free, acc, entry):
    """4 条路径对同一输入产出一致; 记录每路径对的 maxdiff。"""
    inputs = H.sample_inputs_raw(entry, 0)
    ref = P.run_sync(H.load_model(entry), inputs)   # 同步为基准
    for name, fn in P.INFERENCE_PATHS[1:]:          # 跳过 sync(它是 ref)
        outs = fn(H.load_model(entry), inputs)      # 每路 fresh model, 隔离状态
        assert len(outs) == len(ref), f"{name}: 输出数 {len(outs)}!={len(ref)}"
        strict = (name == "async")    # 同内存路径 bit-identical
        for i, (r, o) in enumerate(zip(ref, outs)):
            d = P.assert_identical(r, o, strict=strict, atol=1e-5,
                                   label=f"{entry['cid']} sync vs {name} out{i}")
            H.collect(acc, "d4_paths_maxdiff", {
                "cid": entry["cid"], "path_a": "sync", "path_b": name,
                "out_idx": i, "max_abs_diff": d,
                "identical": int(np.array_equal(np.asarray(r), np.asarray(o))),
            })


@pytest.mark.dim4
@pytest.mark.parametrize("entry", _repr_single_io(),
                         ids=[e["cid"] for e in _repr_single_io()])
def test_d4_from_numpy_roundtrip(bpu_free, entry):
    """from_numpy(arr).numpy() == arr(IONArray 工厂零拷贝往返 bit-exact)。"""
    inputs = H.sample_inputs_raw(entry, 0)
    for arr in inputs:
        back = P.from_numpy_roundtrip(arr)
        assert np.array_equal(arr, back), "from_numpy 往返不一致"
        assert arr.dtype == back.dtype
