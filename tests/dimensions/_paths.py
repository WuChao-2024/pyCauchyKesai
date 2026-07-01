"""D4/D5/D7/D8 共享: pyCauchyKesai 的多条推理路径执行器 + 跨路径比对。

4 条"产生推理结果"的路径(返回 list[np.ndarray]):
  sync       model(inputs)                  —— Auto 同步 (校验+from_numpy→BPU→output), 自动 flush
  async      start(inputs)+wait             —— 异步 memcpy
  zerocopy   IONArray(input_descs)/from_numpy/inputs[=]/start()/wait/flush_invalidate/numpy
  pipeline   Pipeline([model]).pipe(inputs)

from_numpy 是 IONArray 工厂(非推理), 单独 roundtrip 验证。

行为依据(板端活体探针确认):
  - from_numpy 内部已 flush_clean; start() 零拷贝不 flush(flush 是用户契约)
  - 输出 quanti_type=NONE(本批 featuremap 模型), numpy/dequantize 都返回 float32
"""
import numpy as np
from pyCauchyKesai import IONArray


def run_sync(model, inputs, task_id=0):
    return list(model(inputs, task_id=task_id))


def run_async(model, inputs, task_id=0):
    model.start(inputs, task_id=task_id)
    return list(model.wait(task_id=task_id))


def run_zerocopy(model, inputs, task_id=0):
    """零拷贝端到端: 从模板 desc 建 IONArray→from_numpy→绑定→start()→wait→flush_invalidate→numpy。"""
    n_in = len(model.input_names)
    n_out = len(model.output_names)
    ion_in = [IONArray(model.input_descs[i]) for i in range(n_in)]
    ion_out = [IONArray(model.output_descs[i]) for i in range(n_out)]
    for ion, arr in zip(ion_in, inputs):
        ion.from_numpy(np.ascontiguousarray(arr))   # from_numpy 自动 flush_clean
    for i, ion in enumerate(ion_in):
        model.inputs[task_id][i] = ion              # 纯赋值,无校验(浅/快)
    for i, ion in enumerate(ion_out):
        model.outputs[task_id][i] = ion
    model.start(task_id=task_id)                    # 零拷贝提交(用 slot 已绑定的 IONArray)
    model.wait(task_id=task_id)
    outs = []
    for ion in ion_out:
        ion.flush_invalidate()                    # DDR→CPU cache(用户契约)
        outs.append(np.array(ion.numpy(), copy=True))  # copy 防 capsule 生命周期
    return outs


def run_pipeline(model, inputs, task_id=0):
    import os, sys
    _tests = os.path.join(os.path.dirname(__file__), '..')
    if _tests not in sys.path:
        sys.path.insert(0, _tests)
    from _pipeline import Pipeline
    return list(Pipeline([model]).pipe(inputs, task_id=task_id))


# 推理路径注册表(顺序固定, 用于 D4 cross-path 比对)
INFERENCE_PATHS = [
    ("sync", run_sync),
    ("async", run_async),
    ("zerocopy", run_zerocopy),
    ("pipeline", run_pipeline),
]


def from_numpy_roundtrip(arr):
    """IONArray 工厂往返: from_numpy(arr).numpy() == arr。"""
    from pyCauchyKesai import from_numpy
    return np.array(from_numpy(arr).numpy(), copy=True)


def assert_identical(ref, cand, strict=True, atol=1e-5, rtol=1e-5, label=""):
    """跨路径比对。
    strict=True: np.array_equal(bit-identical, 同内存路径对, BPU 确定性)
    strict=False: assert_allclose(atol, 不同内存布局对)
    返回 max_abs_diff(供刻画记录)。
    """
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    assert ref.shape == cand.shape, f"{label}: shape {ref.shape} vs {cand.shape}"
    diff = float(np.max(np.abs(ref.astype(np.float64) - cand.astype(np.float64))))
    if strict:
        assert np.array_equal(ref, cand), f"{label}: not bit-identical (maxdiff={diff:.3e})"
    else:
        np.testing.assert_allclose(ref, cand, atol=atol, rtol=rtol,
                                   err_msg=f"{label}: not allclose (maxdiff={diff:.3e})")
    return diff
