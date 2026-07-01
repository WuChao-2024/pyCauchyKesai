"""文档「CauchyKesai 零拷贝接入 / check_input / check_output / inputs-outputs 绑定 /
多模型零拷贝链式 cookbook」节细节测试。

覆盖:
  - 零拷贝同步 m(task_id) == Auto m(inputs)
  - check_input/check_output True(匹配)/False(不匹配)
  - inputs[slot][idx]=ion 纯赋值,读返回同一对象(is 成立)
  - len(inputs)==n_task,len(inputs[slot])==input_count
  - flush 契约(零拷贝需手动 flush_clean)
  - 多模型链式 A.outputs[0] → B.inputs[0](from_memory 共享同物理内存)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
import conftest as C
from pyCauchyKesai import IONArray


@pytest.mark.docvar
class TestZeroCopy:
    def test_zerocopy_sync_matches_auto(self, factory, bpu_free):
        # 文档零拷贝同步 m(task_id) == Auto m(inputs)
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        auto_out = m([x.copy() for x in inp])
        # 绑 IONArray + flush_clean + start(task_id) + wait
        for i, arr in enumerate(inp):
            ion = IONArray(m.input_descs[i])
            ion.from_numpy(np.ascontiguousarray(arr))
            ion.flush_clean()
            m.inputs[0][i] = ion
        zc_out = m(task_id=0)
        for a, b in zip(auto_out, zc_out):
            assert np.array_equal(a, b)

    def test_zerocopy_needs_flush(self, factory, bpu_free):
        # 文档 flush 契约:零拷贝路径用户写 buffer 后须 flush_clean 再 start
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        for i, arr in enumerate(inp):
            ion = IONArray(m.input_descs[i])
            ion.from_numpy(np.ascontiguousarray(arr))  # from_numpy 自动 flush_clean
            m.inputs[0][i] = ion
        out = m(task_id=0)
        assert len(out) == len(m.output_names)

    def test_check_input_true_matching(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        ion = IONArray(m.input_descs[0])
        assert m.check_input(ion, 0) is True

    def test_check_input_false_wrong_dtype(self, factory, bpu_free):
        es = C.select(image=False)
        if not es:
            pytest.skip("无 featuremap entry")
        m = factory(es[0])
        wrong = IONArray(np.dtype("int32"), list(m.input_descs[0].shape))
        assert m.check_input(wrong, 0) is False

    def test_check_output_true_matching(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        ion = IONArray(m.output_descs[0])
        assert m.check_output(ion, 0) is True

    def test_check_input_returns_bool_not_raises(self, factory, bpu_free):
        # 文档:check 返回 bool 不抛(即使 None/不匹配)
        e = C.ENTRIES[0]
        m = factory(e)
        assert m.check_input(None, 0) is False  # ion 为 None

    def test_inputs_assignment_identity(self, factory, bpu_free):
        # 文档:m.inputs[slot][idx]=ion 纯赋值;读返回同一 C++ 对象(is 成立)
        e = C.ENTRIES[0]
        m = factory(e)
        ion = IONArray(m.input_descs[0])
        m.inputs[0][0] = ion
        assert m.inputs[0][0] is ion

    def test_inputs_len(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=3)
        assert len(m.inputs) == 3
        assert len(m.inputs[0]) == len(m.input_names)
        assert len(m.outputs[0]) == len(m.output_names)

    def test_bound_inputs_preallocated(self, factory, bpu_free):
        # 文档:构造时所有 slot 的 input/output IONArray 已建好并分配
        e = C.ENTRIES[0]
        m = factory(e, n_task=2)
        for slot in range(2):
            for ion in m.inputs[slot] + m.outputs[slot]:
                assert ion.is_allocated


@pytest.mark.docvar
class TestChain:
    def test_chain_a_to_b_zerocopy(self, factory, bpu_free):
        # 文档多模型链式:A.outputs[0] 共享给 B.inputs[0](同物理内存,省中转)
        a = C.by_name("chain_a")
        b = C.by_name("chain_b")
        if not a or not b:
            pytest.skip("无 chain_a/chain_b")
        ma = factory(a)
        mb = factory(b)
        # A 推理
        a_in = IONArray(ma.input_descs[0])
        a_in.numpy()[:] = np.random.randn(*tuple(a_in.desc.shape)).astype(np.float32)
        a_in.flush_clean()
        ma.inputs[0][0] = a_in
        ma.start(task_id=0)
        ma.wait_done(task_id=0)
        a_out = ma.outputs[0][0]            # 取 A 输出 IONArray(含物理地址)
        # 预检能否零拷贝喂 B
        if not mb.check_input(a_out, 0):
            pytest.skip("A/B 性质不匹配,需 dequant/转换(非零拷贝)")
        mb.inputs[0][0] = a_out             # 零拷贝:同物理内存
        mb.start(task_id=0)
        out = mb.wait(task_id=0)
        assert len(out) == len(mb.output_names)
        assert np.all(np.isfinite(out[0]))
