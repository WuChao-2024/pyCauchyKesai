"""文档「构造函数」「Context Manager」「__repr__」节细节测试。

覆盖:
  - 构造成功 + summary
  - model_path 不存在 / 是目录 → RuntimeError
  - n_task ≤0 钳到 1(不抛)
  - n_task>1 多 slot
  - model_cnt_select 超范围钳位(不抛)
  - context manager
  - __repr__ 格式
  - del 后 GC 释放不 crash
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import conftest as C
from pyCauchyKesai import CauchyKesai


@pytest.mark.docvar
class TestConstruct:
    def test_basic_load_and_summary(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        s = m.summary()
        assert s["model_path"].endswith(e["name"] + ".hbm")
        assert s["n_task"] >= 1

    def test_bad_path_raises_runtime(self, bpu_free):
        # 文档:模型文件不存在 → RuntimeError
        with pytest.raises(RuntimeError):
            CauchyKesai("/nonexistent/path/xyz.hbm")

    def test_dir_path_raises_runtime(self, bpu_free):
        # 文档:不是文件(目录) → RuntimeError
        with pytest.raises(RuntimeError):
            CauchyKesai("/tmp")

    def test_n_task_clamp_to_one(self, bpu_free):
        # 文档:n_task<=0 钳到 1 并警告(不抛)
        e = C.ENTRIES[0]
        for bad in (0, -1, -5):
            m = CauchyKesai(C.hbm_path(e), n_task=bad)
            assert m.n_task == 1

    def test_n_task_multi_slots(self, bpu_free):
        # 文档:n_task 用户自定义无上限;每 slot 预分配 input+output ION
        e = C.ENTRIES[0]
        m = CauchyKesai(C.hbm_path(e), n_task=4)
        assert m.n_task == 4
        assert len(m.inputs) == 4
        assert len(m.outputs) == 4

    def test_model_cnt_select_out_of_range_clamps(self, bpu_free):
        # 文档:model_cnt_select 超范围自动钳位并警告(不抛)
        e = C.ENTRIES[0]
        m = CauchyKesai(C.hbm_path(e), model_cnt_select=99)
        assert m.summary()["n_task"] >= 1  # 加载成功即可

    def test_context_manager(self, factory, bpu_free):
        # 文档:with CauchyKesai(...) as ck: ck(inputs)
        e = C.ENTRIES[0]
        with factory(e) as ck:
            outs = ck(C.make_inputs(ck))
            assert isinstance(outs, list) and len(outs) == len(ck.summary()["outputs"])
        # 退出后 ck 引用释放,GC 清理 C++ 资源 — 不 crash 即通过

    def test_repr_format(self, factory, bpu_free):
        # 文档:CauchyKesai(model='...', inputs=N, outputs=N, n_task=N)
        e = C.ENTRIES[0]
        m = factory(e)
        r = repr(m)
        assert "CauchyKesai" in r
        s = m.summary()
        assert str(len(s["inputs"])) in r
        assert str(len(s["outputs"])) in r
        assert str(s["n_task"]) in r

    def test_del_release_then_reload_no_crash(self, factory, bpu_free):
        # GC 析构由 pybind11 管理,del 后再加载不 crash
        e = C.ENTRIES[0]
        m = factory(e)
        del m
        m2 = factory(e)
        assert m2.summary()["n_task"] >= 1

    def test_input_output_descs_readonly(self, factory, bpu_free):
        # 文档:input_descs/output_descs 模板 IONArrayDesc(只读,性质齐全)
        e = C.ENTRIES[0]
        m = factory(e)
        ides = m.input_descs
        odes = m.output_descs
        assert len(ides) == len(m.input_names)
        assert len(odes) == len(m.output_names)
        for d in ides + odes:
            assert d.has_tensor_properties()  # 模板 desc 带模型性质
            assert d.aligned_byte_size > 0
