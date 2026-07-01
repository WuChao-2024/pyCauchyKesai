"""D1 算子/结构: 10 个 backbone(conv2d/conv3d/depthwise/matmul/transformer/elementwise/
softmax/layernorm/reshape_transpose/concat_split)各取一个 aligned 模型,
接口(加载/三路一致/零拷贝)全部正确。"""
import os
import sys
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H


def _curated():
    """每个 backbone 一个 aligned core1 模型(覆盖全部 10 个算子族)。"""
    return H.one_per(H.select_matrix(variant="aligned"), "backbone")


_CURATED = _curated()


@pytest.mark.dim1
@pytest.mark.parametrize("entry", _CURATED, ids=[e["cid"] for e in _CURATED])
def test_d1_each_backbone_consistent(bpu_free, acc, entry):
    """每个算子族: sync 有限+shape 正确, 零拷贝/async == sync(bit-identical)。"""
    p = H.parse_model_id(entry["model_id"])
    r = H.path_consistency(entry)
    assert r["sync_finite"], f"{entry['cid']}: sync 输出含非有限值"
    assert r["sync_shape_ok"], f"{entry['cid']}: sync 输出 shape != meta"
    assert r["zerocopy_maxdiff"] == 0.0, f"{entry['cid']}: 零拷贝 != sync"
    assert r["async_maxdiff"] == 0.0, f"{entry['cid']}: async != sync"
    H.collect(acc, "d1_ops", {
        "cid": entry["cid"], "backbone": p["backbone"], "rank": p["rank"],
        "arity": p["arity"], "in_shapes": r["in_shapes"], "out_shapes": r["out_shapes"],
        "zerocopy_maxdiff": r["zerocopy_maxdiff"], "all_identical": int(r["all_identical"]),
    })
