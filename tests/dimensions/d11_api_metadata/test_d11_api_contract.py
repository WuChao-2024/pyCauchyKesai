"""D11 API 元数据契约: summary() schema 稳定 + 只读属性与 summary 一致 + repr/with 不崩。"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H
from pyCauchyKesai import CauchyKesai


def _entries():
    es = []
    for fam in ("conv_stack", "mixed_io"):
        es.extend(H.select_entries(family=fam, variant="small", precision="int16", core_num=1))
    return es


# soc_name 不在顶层，而在 platform 子 dict 里（s["platform"]["soc_name"]）
_SUMMARY_KEYS = {"model_path", "model_names", "n_task", "inputs", "outputs",
                 "bpu_core_num", "platform"}
_IO_KEYS = {"name", "dtype", "shape", "alignedByteSize", "quantiType"}


@pytest.mark.dim11
@pytest.mark.parametrize("entry", _entries(), ids=[e["cid"] for e in _entries()])
def test_d11_summary_schema(bpu_free, entry):
    s = H.load_model(entry).summary()
    assert _SUMMARY_KEYS.issubset(s.keys()), f"缺 key: {_SUMMARY_KEYS - set(s.keys())}"
    assert s["n_task"] >= 1
    assert len(s["model_names"]) >= 1
    for io in s["inputs"] + s["outputs"]:
        assert _IO_KEYS.issubset(io.keys()), f"IO 缺 key: {_IO_KEYS - set(io.keys())}"
        assert io["quantiType"] in (0, 1)        # NONE / SCALE


@pytest.mark.dim11
@pytest.mark.parametrize("entry", _entries(), ids=[e["cid"] for e in _entries()])
def test_d11_readonly_attrs_match_summary(bpu_free, entry):
    m = H.load_model(entry)
    s = m.summary()
    assert m.input_names == [i["name"] for i in s["inputs"]]
    assert m.output_names == [o["name"] for o in s["outputs"]]
    assert m.bpu_core_num == s["bpu_core_num"]


@pytest.mark.dim11
@pytest.mark.parametrize("entry", _entries(), ids=[e["cid"] for e in _entries()])
def test_d11_repr_and_context_manager(bpu_free, entry):
    e = entry
    m = H.load_model(e)
    assert "CauchyKesai" in repr(m)
    with CauchyKesai(H.GH.hbm_path(e["cid"]), n_task=1) as cm:
        out = cm(H.sample_inputs_raw(e, 0))
        assert np.all(np.isfinite(out[0]))
