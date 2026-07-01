"""文档「推理方法 __call__」「任务槽」「异步推理 start/wait/wait_done/is_busy」节测试。

覆盖:
  - Auto 同步 m(inputs) → list[ndarray], shape/dtype 对
  - Auto 输入数量/dtype/shape 不匹配 → ValueError
  - task_id 越界 → IndexError(__call__/start/wait/is_busy)
  - start+wait 异步 == 同步
  - slot busy 重入 start → RuntimeError
  - wait 空闲 slot → RuntimeError
  - wait 重复 → RuntimeError
  - wait_done → None
  - is_busy 推理后 False
  - 同输入确定性
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pytest
import conftest as C
from pyCauchyKesai import CauchyKesai


def _feat_entry():
    """取一个 featuremap(非图像)单输入 entry 做错误路径测试,可控。"""
    es = C.select(image=False)
    return es[0] if es else None


@pytest.mark.docvar
class TestCallAuto:
    def test_auto_basic_shapes(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        info = m.summary()
        outs = m(C.make_inputs(m))
        assert isinstance(outs, list)
        assert len(outs) == len(info["outputs"])
        for o, ref in zip(outs, info["outputs"]):
            assert o.shape == tuple(ref["shape"])

    def test_auto_count_mismatch_raises(self, factory, bpu_free):
        e = _feat_entry()
        if e is None:
            pytest.skip("无 featuremap entry")
        m = factory(e)
        n_in = len(m.input_names)
        inputs = C.make_inputs(m)
        # 少传
        with pytest.raises(ValueError):
            m(inputs[:max(0, n_in - 1)])
        # 多传
        with pytest.raises(ValueError):
            m(inputs + [np.zeros((1,), np.float32)])

    def test_auto_dtype_mismatch_raises(self, factory, bpu_free):
        e = _feat_entry()
        if e is None:
            pytest.skip("无 featuremap entry")
        m = factory(e)
        info = m.summary()
        # 构造全错 dtype(float64 代替模板 dtype)
        bad = [np.zeros(tuple(i["shape"]), np.float64) for i in info["inputs"]]
        with pytest.raises((ValueError, RuntimeError)):
            m(bad)

    def test_auto_shape_mismatch_raises(self, factory, bpu_free):
        e = _feat_entry()
        if e is None:
            pytest.skip("无 featuremap entry")
        m = factory(e)
        info = m.summary()
        # 最后一维 +1 → shape 不匹配
        bad = []
        for i in info["inputs"]:
            sh = list(i["shape"])
            sh[-1] = sh[-1] + 1
            dt = np.float32 if i["dtype"] not in ("uint8", "int8") else np.uint8
            bad.append(np.zeros(tuple(sh), dt))
        with pytest.raises((ValueError, RuntimeError)):
            m(bad)

    def test_call_task_id_oob_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        with pytest.raises(IndexError):
            m(C.make_inputs(m), task_id=99)

    def test_determinism_same_input(self, factory, bpu_free):
        # 文档 D10:同输入两次推理结果一致(BPU 确定性)
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        r1 = m([x.copy() for x in inp])
        r2 = m([x.copy() for x in inp])
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert np.array_equal(a, b)


@pytest.mark.docvar
class TestAsync:
    def test_start_wait_equals_sync(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        sync_out = m([x.copy() for x in inp])
        m.start([x.copy() for x in inp], task_id=0)
        async_out = m.wait(task_id=0)
        assert len(sync_out) == len(async_out)
        for a, b in zip(sync_out, async_out):
            assert np.array_equal(a, b)

    def test_start_task_id_oob_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        with pytest.raises(IndexError):
            m.start(C.make_inputs(m), task_id=99)

    def test_busy_slot_resubmit_raises(self, factory, bpu_free):
        # 文档:同 slot 不能重入 → task_id N is already in use
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        inp = C.make_inputs(m)
        m.start(inp, task_id=0)
        with pytest.raises(RuntimeError):
            m.start(inp, task_id=0)
        m.wait(task_id=0)  # 清理

    def test_wait_idle_slot_raises(self, factory, bpu_free):
        # 文档:wait 空闲 slot → task_id N is not in use
        e = C.ENTRIES[0]
        m = factory(e, n_task=2)
        with pytest.raises(RuntimeError):
            m.wait(task_id=1)

    def test_wait_double_raises(self, factory, bpu_free):
        # 文档:wait 重复(slot 已释放回 idle)→ already being waited / not in use
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        m.start(inp, task_id=0)
        m.wait(task_id=0)
        with pytest.raises(RuntimeError):
            m.wait(task_id=0)

    def test_wait_done_returns_none(self, factory, bpu_free):
        # 文档:wait_done 纯等完成,不返回
        e = C.ENTRIES[0]
        m = factory(e)
        inp = C.make_inputs(m)
        m.start(inp, task_id=0)
        assert m.wait_done(task_id=0) is None

    def test_wait_task_id_oob_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        with pytest.raises(IndexError):
            m.wait(task_id=99)

    def test_is_busy_after_infer_false(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e)
        m(C.make_inputs(m))
        assert m.is_busy(0) is False

    def test_is_busy_task_id_oob_raises(self, factory, bpu_free):
        e = C.ENTRIES[0]
        m = factory(e, n_task=1)
        with pytest.raises(IndexError):
            m.is_busy(99)

    def test_interleaved_start_wait_no_crosstalk(self, factory, bpu_free):
        # 文档 D8:4 slot 交错 start/wait 不串结果
        e = C.ENTRIES[0]
        if e["yaml"].get("image"):
            pytest.skip("图像模型跳过 scale 交错")
        m = factory(e, n_task=4)
        info = m.summary()
        base_shape = tuple(info["inputs"][0]["shape"])
        dt = info["inputs"][0]["dtype"]
        ins = [np.ones(base_shape, np.float32) * (k + 1) for k in range(4)]
        # refs 用独立 n_task=1 model 顺序算参考(避免与并发 m 共用 slot 的干扰)
        refs = [factory(e)([x.copy()])[0] for x in ins]
        for k in range(4):
            m.start([ins[k]], task_id=k)
        got = {k: m.wait(task_id=k)[0] for k in range(4)}
        for k in range(4):
            assert np.array_equal(got[k], refs[k]), f"task {k} 串了"
