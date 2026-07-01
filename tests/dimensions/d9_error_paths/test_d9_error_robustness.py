"""D9 错误路径鲁棒: 各类非法输入应抛预期异常(契约一致性)。

异常类型由板端活体探针确认:
  task_id/priority 越界      → ValueError/IndexError
  输入数量不匹配             → ValueError
  wait 空闲 / busy 二次提交  → RuntimeError
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
from pyCauchyKesai import IONArray


def _one():
    return H.select_entries(family="matmul_stack", variant="small", precision="int16",
                            core_num=1, single_io=True)[0]


def _m():
    return H.load_model(_one(), n_task=2)


@pytest.mark.dim9
@pytest.mark.parametrize("prio", [256, -2, 1000])
def test_d9_priority_out_of_range(bpu_free, prio):
    """per-slot priority 越界(非 [-1,255])应被 set_scheduling_params 拒(ValueError)。"""
    m = _m()
    with pytest.raises(ValueError):
        m.set_scheduling_params([0], priority=prio, task_id=0)


@pytest.mark.dim9
def test_d9_input_count_mismatch(bpu_free):
    """输入数量与模型不符应被拒(非空但数量不对)。
    注: start(task_id) 零拷贝不传 inputs, 不算 mismatch。"""
    m = _m()
    inp = H.sample_inputs_raw(_one(), 0)[0]
    with pytest.raises(ValueError):
        m.start([inp, inp], task_id=0)        # 给 2 个, 模型要 1 个(ValueError 在提交前抛, slot 仍 idle)


@pytest.mark.dim9
@pytest.mark.parametrize("bad_dtype", [np.int32, np.float64, np.int8])
def test_d9_dtype_mismatch(bpu_free, bad_dtype):
    """Auto memcpy 路径 from_numpy 校验 dtype, 不匹配抛(ValueError)。"""
    m = _m()
    inp = H.sample_inputs_raw(_one(), 0)[0].astype(bad_dtype)
    with pytest.raises((ValueError, RuntimeError)):
        m.start([inp], task_id=0)


@pytest.mark.dim9
def test_d9_zerocopy_sync_bound_mismatch_rejected(bpu_free):
    """零拷贝同步 m(task_id) 校验 bound, bound IONArray 性质不符模板应抛。

    用错误 dtype 的 IONArray 绑到 slot, check_input 返回 False → 同步路径校验抛。
    （注：bare IONArray(对的 dtype+shape) 对 natural 布局模型是 check 通过的——
    那是合法绑定，不是 mismatch。这里用错 dtype 制造真 mismatch。）
    """
    m = _m()
    tpl = m.input_descs[0]
    # 错 dtype（模型要 int16/float32，给 int32），check_input 必 False
    bad_dtype = np.int32 if tpl.dtype != np.dtype(np.int32) else np.float64
    bad = IONArray(np.dtype(bad_dtype), list(tpl.shape))
    assert not m.check_input(bad, 0), "错 dtype IONArray 应 check_input=False"
    m.inputs[0][0] = bad                  # 纯赋值无校验(异步路径不拦)
    with pytest.raises((ValueError, RuntimeError)):
        m(task_id=0)                      # 零拷贝同步: 校验 bound 抛


@pytest.mark.dim9
def test_d9_busy_slot_start_rejected(bpu_free):
    """busy slot 二次 start 应抛(同 slot 不能重入, RuntimeError)。"""
    m = _m()
    inp = H.sample_inputs_raw(_one(), 0)
    m.start(inp, task_id=0)
    with pytest.raises(RuntimeError):
        m.start(inp, task_id=0)           # slot 0 busy, 重入被拒
    m.wait(task_id=0)
