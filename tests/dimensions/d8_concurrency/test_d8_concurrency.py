"""D8 并发语义: n_task 边界、交错 start/wait(结果不串)、busy 二次提交、wait 空闲/重复。"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DIMS = os.path.dirname(_HERE)
if _DIMS not in sys.path:
    sys.path.insert(0, _DIMS)

import _harness as H


def _one():
    return H.select_entries(family="matmul_stack", variant="small", precision="int16",
                            core_num=1, single_io=True)[0]


@pytest.mark.dim8
def test_d8_task_id_out_of_range(bpu_free):
    e = _one()
    m = H.load_model(e, n_task=4)
    inp = H.sample_inputs_raw(e, 0)
    with pytest.raises(IndexError):
        m.start(inp, task_id=4)          # n_task=4, 合法 task_id ∈ [0,3]
    with pytest.raises(IndexError):
        m.is_busy(9)


@pytest.mark.dim8
def test_d8_interleaved_start_wait_no_crosstalk(bpu_free):
    """4 slot 交错 start(乱序)→ wait(乱序), 每个 wait 必须拿到对应 task 的正确结果。"""
    e = _one()
    m = H.load_model(e, n_task=4)
    base = H.sample_inputs_raw(e, 0)[0]
    # 4 组不同输入(scale 不同), 预先用同步算各自参考输出
    ins = [base * (k + 1) for k in range(4)]
    refs = [H.load_model(e)([x])[0] for x in ins]
    # 交错: 按顺序 start, 但乱序 wait
    for k in range(4):
        m.start([ins[k]], task_id=k)
    got = {}
    for k in [2, 0, 3, 1]:                # 乱序 wait
        got[k] = m.wait(task_id=k)[0]
    for k in range(4):
        assert np.array_equal(got[k], refs[k]), f"task {k} 结果串了"


@pytest.mark.dim8
def test_d8_busy_resubmit(bpu_free):
    e = _one()
    m = H.load_model(e, n_task=2)
    inp = H.sample_inputs_raw(e, 0)
    m.start(inp, task_id=0)
    with pytest.raises(RuntimeError):
        m.start(inp, task_id=0)           # slot busy 二次提交
    m.wait(task_id=0)                     # 清理


@pytest.mark.dim8
def test_d8_wait_idle_and_double_wait(bpu_free):
    e = _one()
    m = H.load_model(e, n_task=2)
    inp = H.sample_inputs_raw(e, 0)
    with pytest.raises(RuntimeError):
        m.wait(task_id=1)                 # 空闲 slot wait
    m.start(inp, task_id=0)
    m.wait(task_id=0)
    with pytest.raises(RuntimeError):
        m.wait(task_id=0)                 # wait 重复(slot 已释放回 idle)
