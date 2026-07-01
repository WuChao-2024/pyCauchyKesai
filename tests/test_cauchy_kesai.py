"""CauchyKesai 推理功能测试 (所有平台)"""

import pytest
import numpy as np
from pyCauchyKesai import CauchyKesai, IONArray


MODEL_PATH = None
INPUT_SHAPE = None
INPUT_DTYPE = None


def pytest_generate_tests(metafunc):
    """动态发现模型，跳过无模型环境"""
    if "model" in metafunc.fixturenames:
        global MODEL_PATH
        if MODEL_PATH is None:
            MODEL_PATH = _find_model()
        if MODEL_PATH is None:
            pytest.skip("未设置 TEST_MODEL_PATH 且未找到 .hbm 模型")


def _find_model():
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from conftest import find_test_model
    return find_test_model()


@pytest.mark.bpu
class TestCauchyKesaiBasic:
    """基本功能测试"""

    @pytest.fixture(scope="class")
    def model(self):
        m = CauchyKesai(MODEL_PATH, n_task=2)
        yield m
        del m

    def test_construct_and_summary(self):
        """构造和 summary"""
        m = CauchyKesai(MODEL_PATH, n_task=1)
        s = m.summary()
        assert "model_path" in s
        assert "inputs" in s
        assert "outputs" in s
        assert s["n_task"] == 1
        assert len(s["model_names"]) > 0

    def test_summary_structured(self, model):
        """summary dict 可结构化访问"""
        s = model.summary()
        for inp in s["inputs"]:
            assert "name" in inp
            assert "dtype" in inp
            assert "shape" in inp

    def test_benchmark_dict(self, model):
        """t() benchmark 返回 dict"""
        b = model.benchmark()
        assert b["time_us"] > 0
        assert b["time_ms"] > 0
        assert b["time_s"] > 0

    def test_s_t_pretty_print_return_none(self, model):
        """s()/t()/platform.s() 内部 print 美观输出，返回 None；summary() 返 dict"""
        assert model.s() is None                  # 模型摘要（内部 print）
        assert model.t() is None                  # 计时（内部 print）
        assert model.platform.s() is None         # 平台信息（内部 print）
        assert isinstance(model.summary(), dict)  # 数据契约：summary 永远返 dict

    def test_inference_sync(self, model):
        """同步推理正常返回"""
        info = model.summary()
        inputs = []
        for inp in info["inputs"]:
            shape = tuple(inp["shape"])
            if inp["dtype"] == "float32":
                arr = np.random.randn(*shape).astype(np.float32)
            elif inp["dtype"] == "uint8":
                arr = np.random.randint(0, 255, shape, dtype=np.uint8)
            else:
                arr = np.random.randn(*shape).astype(np.float32)
            inputs.append(arr)
        outputs = model(inputs)
        assert len(outputs) == len(info["outputs"])
        for i, out in enumerate(outputs):
            assert out.shape == tuple(info["outputs"][i]["shape"])

    def test_inference_call(self, model):
        """__call__ 同步推理（确定性：同输入两次结果一致）"""
        info = model.summary()
        inputs = [np.random.randn(*tuple(inp["shape"])).astype(np.float32)
                  for inp in info["inputs"]]
        r1 = model(inputs)
        r2 = model(inputs)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert np.allclose(a, b, atol=1e-3), "同输入两次推理结果应一致"

    def test_is_busy(self, model):
        """推理后不 busy"""
        info = model.summary()
        inputs = [np.random.randn(*tuple(inp["shape"])).astype(np.float32)
                  for inp in info["inputs"]]
        model(inputs)
        assert not model.is_busy(0)

    def test_invalid_task_id(self, model):
        """无效 task_id 抛异常"""
        with pytest.raises((IndexError, Exception)):
            model.is_busy(99)

    def test_repr(self, model):
        """__repr__ 有信息"""
        r = repr(model)
        assert "CauchyKesai" in r


@pytest.mark.bpu
class TestCauchyKesaiConcurrent:
    """并发测试"""

    @pytest.fixture(scope="class")
    def model(self):
        return CauchyKesai(MODEL_PATH, n_task=4)

    def test_concurrent_start_wait(self, model):
        """多任务并发 start/wait"""
        info = model.summary()
        data = []
        for t in range(4):
            inputs = [np.random.randn(*tuple(inp["shape"])).astype(np.float32)
                      for inp in info["inputs"]]
            data.append(inputs)
        for t in range(4):
            model.start(data[t], task_id=t)
        outputs = [model.wait(t) for t in range(4)]
        assert all(len(o) == len(info["outputs"]) for o in outputs)


@pytest.mark.nash
class TestIONArray:
    """IONArray 功能测试"""

    def test_basic_construct(self):
        """基本构造"""
        arr = IONArray(np.dtype('float32'), [1, 3, 224, 224])
        assert arr.numpy().shape == (1, 3, 224, 224)
        assert arr.numpy().dtype == np.float32
        assert arr.is_allocated
        assert arr.desc.nbytes() > 0

    def test_allocated_accessible(self):
        """分配状态可获取（phy_addr 已下沉内部）"""
        arr = IONArray(np.dtype('uint8'), [1, 640, 640, 1])
        assert arr.is_allocated
        assert arr.desc.nbytes() == 640 * 640

    def test_read_write(self):
        """读写如 numpy"""
        arr = IONArray(np.dtype('float32'), [10])
        arr.numpy()[:] = np.arange(10, dtype=np.float32)
        assert arr.numpy()[0] == 0
        assert arr.numpy()[9] == 9

    def test_flush_methods(self):
        """flush/invalidate 方法存在"""
        arr = IONArray(np.dtype('float32'), [4])
        arr.flush_clean()
        arr.flush_invalidate()
