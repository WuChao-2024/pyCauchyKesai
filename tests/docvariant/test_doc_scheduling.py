"""文档「BPU 核调度 / set_scheduling_params / scheduled_cores / scheduled_priority」节测试。

覆盖:
  - 单核模型宽松:掩码含>=1有效核即可([]/[0]/[0,1,2,3] 都能跑)
  - 多核模型严格:恰好 N 个有效核(少/多都 ValueError)
  - 任意 N 核子集均可(不必前 N 核)
  - 无效物理核/越界 → ValueError
  - priority 不在 [-1,255] → ValueError
  - priority -1 = 不改现有
  - task_id 越界 → IndexError
  - [] 重置 CORE_ANY;多核 []推理被拒
  - scheduled_cores/scheduled_priority 是方法(带括号)
  - 构造默认掩码(多核前 N 核)
  - per-slot 独立
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import conftest as C
from pyCauchyKesai import CauchyKesai


@pytest.mark.docvar
class TestScheduling:
    def test_single_core_loose(self, factory, bpu_free):
        # 文档:单核模型宽松,掩码含>=1有效核即可
        es = C.select(core=1)
        if not es:
            pytest.skip("无 core1 entry")
        e = es[0]
        for cores in ([0], [], [0, 1, 2, 3]):
            m = factory(e)
            m.set_scheduling_params(cores)       # set 不抛
            m(C.make_inputs(m))                  # 推理也不抛

    def test_multi_core_strict_count(self, factory, bpu_free):
        # 文档:多核模型掩码必须恰好 N 个有效核
        es = C.select(core=2)
        if not es:
            pytest.skip("无 core2 entry")
        e = es[0]
        m = factory(e)
        m.set_scheduling_params([0, 1])          # 恰好2核 ok
        with pytest.raises(ValueError):
            m.set_scheduling_params([0])         # 不足2核
        with pytest.raises(ValueError):
            m.set_scheduling_params([0, 1, 2])   # 超过2核

    def test_multi_core_arbitrary_subset(self, factory, bpu_free):
        # 文档:任意 N 核子集均可,不必是前 N 核
        es = C.select(core=2)
        if not es:
            pytest.skip("无 core2 entry")
        e = es[0]
        for ok in ([0, 1], [0, 3], [1, 2], [2, 3]):
            m = factory(e)
            m.set_scheduling_params(ok)
            m(C.make_inputs(m))

    def test_invalid_core_index_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        with pytest.raises(ValueError):
            m.set_scheduling_params([99])        # 越界物理核

    def test_priority_out_of_range_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        for bad in (256, -2, 1000):
            with pytest.raises(ValueError):
                m.set_scheduling_params([0], priority=bad)

    def test_priority_valid_set_get(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        for good in (0, 1, 100, 253):
            m.set_scheduling_params([0], priority=good)
            assert m.scheduled_priority() == good

    def test_priority_neg1_no_change(self, factory, bpu_free):
        # 文档:priority=-1 = 不改现有优先级
        e = C.ENTRIES[0]
        m = factory(e)
        m.set_scheduling_params([0], priority=50)
        m.set_scheduling_params([0], priority=-1)
        assert m.scheduled_priority() == 50

    def test_task_id_oob_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        with pytest.raises(IndexError):
            m.set_scheduling_params([0], task_id=99)

    def test_empty_resets_coreany(self, factory, bpu_free):
        # 文档:空列表重置 CORE_ANY(set 放行);scheduled_cores 返回空
        e = C.ENTRIES[0]
        m = factory(e)
        m.set_scheduling_params([])
        assert m.scheduled_cores() == []

    def test_empty_coreany_multi_core_rejected_at_infer(self, factory, bpu_free):
        # 文档:多核模型 [](CORE_ANY) set 放行,但推理时被 hbUCPSubmitTask 拒
        es = C.select(core=2)
        if not es:
            pytest.skip("无 core2 entry")
        e = es[0]
        m = factory(e)
        m.set_scheduling_params([])              # 放行
        with pytest.raises(RuntimeError):
            m(C.make_inputs(m))                  # 推理被拒

    def test_scheduled_methods_are_methods(self, factory, bpu_free):
        # 文档:scheduled_cores/scheduled_priority 是方法(带括号),不是属性
        e = C.ENTRIES[0]
        m = factory(e)
        assert isinstance(m.scheduled_cores(), list)
        assert isinstance(m.scheduled_priority(), int)

    def test_construct_default_mask_multi_core(self, factory, bpu_free):
        # 文档:多核模型构造时自动给所有 slot 设前 N 核,构造完可直接推理
        es = C.select(core=2)
        if not es:
            pytest.skip("无 core2 entry")
        e = es[0]
        m = CauchyKesai(C.hbm_path(e), n_task=1)   # 不手动 set
        assert m.scheduled_cores() == [0, 1]
        m(C.make_inputs(m))                         # 直接推理不抛

    def test_per_slot_independent(self, factory, bpu_free):
        # 文档:不同 slot 配不同核/优先级,跨 slot 零竞态
        e = C.ENTRIES[0]
        m = factory(e, n_task=2)
        m.set_scheduling_params([0], priority=10, task_id=0)
        m.set_scheduling_params([1], priority=20, task_id=1)
        assert m.scheduled_priority(0) == 10
        assert m.scheduled_priority(1) == 20
        assert m.scheduled_cores(0) == [0]
        assert m.scheduled_cores(1) == [1]

    def test_broadcast_task_id_neg1(self, factory, bpu_free):
        # 文档:task_id=-1 广播所有 slot
        e = C.ENTRIES[0]
        m = factory(e, n_task=3)
        m.set_scheduling_params([0], priority=7, task_id=-1)
        for t in range(3):
            assert m.scheduled_priority(t) == 7
            assert m.scheduled_cores(t) == [0]
