"""D2 张量形态: rank 1D~5D + 8 种 shape 变体, 接口(加载/三路一致/零拷贝)全部正确。

数据来自云端 build_all 矩阵(concat_split_1d / matmul_2d / *_3d / conv2d_4d / conv3d_5d,
变体 aligned/odd/nonpow2_channel/bigc/smallc/tiny/large_batch/boundary_reduce)。
"""
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H


def _curated():
    """每个 rank 一个(aligned) + 每个 shape 变体一个, 去重。"""
    rank_reps = H.one_per(H.select_matrix(variant="aligned"), "rank")
    var_reps = H.one_per(H.select_matrix(), "variant")
    seen, out = set(), []
    for e in rank_reps + var_reps:
        if e["cid"] in seen:
            continue
        seen.add(e["cid"]); out.append(e)
    return out


_CURATED = _curated()


@pytest.mark.dim2
@pytest.mark.parametrize("entry", _CURATED, ids=[e["cid"] for e in _CURATED])
def test_d2_rank_shape_consistent(bpu_free, acc, entry):
    """各 rank/shape 模型: sync 有限+shape 正确, 零拷贝/async == sync(bit-identical)。"""
    p = H.parse_model_id(entry["model_id"])
    r = H.path_consistency(entry)
    assert r["sync_finite"], f"{entry['cid']}: sync 输出含非有限值"
    assert r["sync_shape_ok"], f"{entry['cid']}: sync 输出 shape != meta"
    assert r["zerocopy_maxdiff"] == 0.0, f"{entry['cid']}: 零拷贝 != sync"
    assert r["async_maxdiff"] == 0.0, f"{entry['cid']}: async != sync"
    H.collect(acc, "d2_matrix", {
        "cid": entry["cid"], "backbone": p["backbone"], "rank": p["rank"],
        "variant": p["variant"], "arity": p["arity"],
        "in_shapes": r["in_shapes"], "out_shapes": r["out_shapes"],
        "zerocopy_maxdiff": r["zerocopy_maxdiff"], "async_maxdiff": r["async_maxdiff"],
    })
