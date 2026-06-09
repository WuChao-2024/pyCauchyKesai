#!/usr/bin/env python3
"""EZ_ONNX — CauchyKesai 接口兼容的 ONNXRuntime 推理引擎。

纯 Python 实现，使用 ONNXRuntime 替代 BPU 后端，
调用方式与 pyCauchyKesai.CauchyKesai 尽可能一致（加载 .onnx 而非 .hbm）。

Usage:
    from pyCauchyKesai.tools.ez_onnx import EZ_ONNX

    model = EZ_ONNX("model.onnx")
    outputs = model([input1, input2])

    # 或异步
    model.start([input1, input2])
    outputs = model.wait()

    # 或零内存模式
    model = EZ_ONNX("model.onnx", _no_alloc=True)
    model.set_inputs([numpy_arr1])
    model.set_outputs([numpy_arr2])
    outputs = model([input1])
"""

import os
import sys
import time
import platform
import warnings
import threading
import concurrent.futures
from typing import List, Optional, Union

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is required for EZ_ONNX. Install with: pip install onnxruntime"
    )


# ============================================================================
# 全局工具函数
# ============================================================================

def _color(text: str) -> str:
    """TTY 检测：终端加粗，非终端原样返回。"""
    if hasattr(sys.stderr, "isatty") and sys.stderr.isatty():
        return f"\033[1m{text}\033[0m"
    return text


# ONNX element type → numpy dtype 字符串
_ONNX_TYPE_TO_NUMPY_STR = {
    1: "float32",     # FLOAT
    2: "uint8",       # UINT8
    3: "int8",        # INT8
    4: "uint16",      # UINT16
    5: "int16",       # INT16
    6: "int32",       # INT32
    7: "int64",       # INT64
    9: "bool",        # BOOL
    10: "float16",    # FLOAT16
    11: "float64",    # DOUBLE
    12: "uint32",     # UINT32
    13: "uint64",     # UINT64
}

_NUMPY_STR_TO_ONNX_TYPE = {v: k for k, v in _ONNX_TYPE_TO_NUMPY_STR.items()}


def _onnx_dtype_to_numpy_str(onnx_type: int) -> str:
    """ONNX TensorProto.DataType → numpy dtype 字符串。"""
    return _ONNX_TYPE_TO_NUMPY_STR.get(onnx_type, f"unknown_type_{onnx_type}")


def _numpy_dtype_to_str(dtype: np.dtype) -> str:
    """numpy dtype → 字符串（与 pyCauchyKesai 的 dtype_np2str 一致）。"""
    return str(dtype)


# ============================================================================
# NumPyArray — IONArray 的纯 Python 模拟层
# ============================================================================

class NumPyArray:
    """IONArray 的 numpy 模拟层。

    在 CPU/ONNXRuntime 环境下替代 IONArray，用 numpy ndarray 管理"共享内存"。
    保留与 IONArray 一致的属性接口，BPU 特有属性返回占位值或抛 NotImplementedError。

    Constructors
    ------------
    NumPyArray(dtype, shape, cached=True, defer=False)
        分配 numpy 数组。defer=True 时延迟分配。
    NumPyArray(dtype, shape, byte_size, cached=True)
        指定 byte_size 分配（padding 记录在 mem_size 中）。
    """

    def __init__(
        self,
        dtype,
        shape,
        cached: bool = True,
        defer: bool = False,
        byte_size: Optional[int] = None,
    ):
        # dtype 统一为 numpy dtype
        if isinstance(dtype, str):
            self._dtype = np.dtype(dtype)
        else:
            self._dtype = np.dtype(dtype)

        # shape 统一为 tuple
        if isinstance(shape, list):
            self._shape = tuple(shape)
        else:
            self._shape = tuple(shape)

        # 校验 shape 全部 > 0
        for d in self._shape:
            if d <= 0:
                raise ValueError(f"shape 中的维度必须 > 0, got {self._shape}")

        self._cached = cached
        self._byte_size = byte_size  # 用户指定的 byte_size (含 padding)
        self._array: Optional[np.ndarray] = None  # defer 模式下为 None
        self._owns_mem = True
        self._parent: Optional[NumPyArray] = None  # sub_view 时引用父对象

        if not defer:
            self.allocate(cached=cached)

    # ---- 类方法: 从已有 numpy 数组创建 (非持有) ----

    @classmethod
    def _from_array(cls, arr: np.ndarray, parent: Optional["NumPyArray"] = None) -> "NumPyArray":
        """从已有 numpy 数组创建非持有包装 (类似 IONArray 的非持有构造函数)。"""
        obj = cls.__new__(cls)
        obj._dtype = arr.dtype
        obj._shape = arr.shape
        obj._cached = True
        obj._byte_size = None
        obj._array = arr
        obj._owns_mem = False
        obj._parent = parent  # sub_view 时引用父 NumPyArray
        return obj

    # ---- 分配 ----

    def allocate(self, cached: bool = True) -> None:
        """延迟分配 numpy 内存 (类似 IONArray.allocate)。"""
        if self._array is not None and self._owns_mem:
            raise RuntimeError("NumPyArray 已经分配过了")

        self._cached = cached
        self._array = np.empty(self._shape, dtype=self._dtype)
        self._owns_mem = True

    # ---- numpy 视图导出 ----

    def as_array(self) -> np.ndarray:
        """返回 numpy 数组视图 (类似 IONArray.as_array)。

        与原版不同：无需 capsule/shared_ptr 机制，直接返回内部数组。
        返回的是引用而非拷贝——修改会影响原始数据。
        """
        if self._array is None:
            raise RuntimeError("NumPyArray 未分配 (defer=True 时需先调用 allocate)")
        return self._array

    # ---- 子视图 ----

    def sub_view(self, dtype, shape, element_offset: int) -> "NumPyArray":  # noqa: F821
        """创建元素偏移子视图 (类似 IONArray.sub_view)。

        element_offset 以父 dtype 的元素为单位计算字节偏移。
        新视图可使用不同的 dtype 和 shape (reinterpret_cast 语义)。
        """
        if self._array is None:
            raise RuntimeError("NumPyArray 未分配，无法创建 sub_view")
        if element_offset < 0:
            raise ValueError(f"element_offset 必须 >= 0, got {element_offset}")

        # 统一参数类型
        if isinstance(dtype, str):
            new_dtype = np.dtype(dtype)
        else:
            new_dtype = np.dtype(dtype)
        if isinstance(shape, list):
            new_shape = tuple(shape)
        else:
            new_shape = tuple(shape)

        # 计算字节偏移
        byte_offset = element_offset * self._dtype.itemsize
        # 计算视图字节大小
        view_bytes = new_dtype.itemsize
        for d in new_shape:
            view_bytes *= d

        # 边界检查
        available = self._array.nbytes
        if self._byte_size is not None:
            available = self._byte_size  # 用用户指定的 byte_size 做边界检查
        if byte_offset + view_bytes > available:
            raise ValueError(
                f"sub_view 超出范围: offset={byte_offset} + size={view_bytes} "
                f"> available={available}"
            )

        # 从父数组切片创建子视图
        flat = self._array.ravel()
        # 先用 byte offset 做字节级切片
        parent_bytes = flat.view(np.uint8)
        sub_bytes = parent_bytes[byte_offset:byte_offset + view_bytes]
        sub_flat = sub_bytes.view(new_dtype)
        sub_arr = sub_flat.reshape(new_shape)

        # 创建非持有 NumPyArray，引用父对象防止过早释放
        return NumPyArray._from_array(sub_arr, parent=self)

    # ---- 缓存操作 (no-op) ----

    def flush(self) -> None:
        """CPU 缓存 flush (no-op)。CPU 内存天然缓存一致，无需操作。"""
        pass

    def invalidate(self) -> None:
        """CPU 缓存 invalidate (no-op)。同 flush。"""
        pass

    # ---- 只读属性 ----

    @property
    def dtype(self) -> np.dtype:
        """numpy dtype。"""
        return self._dtype

    @property
    def shape(self) -> tuple:
        """数组形状。"""
        return self._shape

    @property
    def ndim(self) -> int:
        """维度数。"""
        return len(self._shape)

    @property
    def size(self) -> int:
        """总元素数。"""
        s = 1
        for d in self._shape:
            s *= d
        return s

    @property
    def nbytes(self) -> int:
        """理论字节大小 (dtype.itemsize * size)，与 IONArray.nbytes 一致。"""
        return self._dtype.itemsize * self.size

    @property
    def mem_size(self) -> int:
        """实际分配字节大小。

        若构造时指定了 byte_size (含 BPU 对齐 padding)，返回 byte_size。
        否则返回 nbytes (numpy 无额外 padding)。
        """
        if self._byte_size is not None:
            return self._byte_size
        return self.nbytes

    @property
    def is_cached(self) -> bool:
        """是否缓存分配。CPU 上永远为 True。"""
        return self._cached

    @property
    def is_allocated(self) -> bool:
        """是否已分配。defer 模式下为 False。"""
        return self._array is not None

    # ---- BPU 特有属性 (占位 / NotImplementedError) ----

    @property
    def phy_addr(self) -> int:
        """ION 物理地址。CPU 无物理地址概念，返回 0。"""
        return 0

    @property
    def vir_addr(self):
        """CPU 侧虚拟地址。Python 不暴露裸指针，返回 None。"""
        return None

    @property
    def sys_mem(self):
        """hbUCPSysMem 结构体。BPU 特有，不可模拟。"""
        raise NotImplementedError("sys_mem is BPU-specific, not available on CPU")

    def __repr__(self) -> str:
        alloc = "allocated" if self.is_allocated else "deferred"
        cached = "cached" if self._cached else "uncached"
        return (f"NumPyArray(dtype={self._dtype}, shape={self._shape}, "
                f"{alloc}, {cached}, nbytes={self.nbytes})")


# ============================================================================
# ModelSummary / BenchmarkResult — 与 pyCauchyKesai.__init__.py 一致
# ============================================================================

class ModelSummary(dict):
    """模型信息摘要 (dict 子类 + 格式化输出)。

    与 pyCauchyKesai.ModelSummary 格式完全一致，
    仅 dnn_version/bpu_version/soc_name 等字段含义不同 (ORT 版本而非 BPU 版本)。
    """

    def __repr__(self):
        try:
            lines = [
                "================== Model Summary ==================",
                _color("Model File: ") + self.get("model_path", "<unknown>"),
                _color("Model Names: "),
            ]
            for entry in self.get("model_names", []):
                suffix = " [*Select]" if entry.get("selected") else ""
                lines.append(f"{entry.get('index', '?')}: {entry.get('name', '?')}{suffix}")
            lines.append(_color("Task N: ") + str(self.get("n_task", "?")))
            lines.append(_color("Memory: ") + str(self.get("memory_mb", "?")) + " MB")
            lines.append(_color("System: ") +
                         f"ORT={self.get('dnn_version', '?')}, "
                         f"Provider={self.get('bpu_version', '?')}, "
                         f"SoC={self.get('soc_name', '?')}")
            lines.append(_color("BPU Cores: ") + str(self.get('bpu_core_num', '?')))
            model_desc = self.get('model_desc', 'N/A')
            if model_desc and model_desc != 'N/A':
                lines.append(_color("Model Desc: ") + model_desc)
            lines.append(_color("Inputs Info: "))
            for inp in self.get("inputs", []):
                shape_str = ", ".join(str(d) for d in inp.get("shape", []))
                quanti = "SCALE" if inp.get('quantiType', 0) == 1 else "NONE"
                line = (f"  [{inp.get('index', '?')}][{inp.get('name', '?')}]: "
                        f"{inp.get('dtype', '?')}, ({shape_str}), "
                        f"size={inp.get('alignedByteSize', '?')}, quanti={quanti}")
                if quanti == "SCALE":
                    scale_arr = inp.get('scale')
                    zp_arr = inp.get('zero_point')
                    axis = inp.get('quantizeAxis', 0)
                    line += f", axis={axis}"
                    if scale_arr is not None and scale_arr.size > 0:
                        line += f", scale.shape={scale_arr.shape}"
                        if scale_arr.size <= 4:
                            line += f", scale={scale_arr.flatten().tolist()}"
                        else:
                            flat = scale_arr.flatten()
                            line += f", scale=[{flat[0]:.6f}, ..., {flat[-1]:.6f}]"
                    if zp_arr is not None and zp_arr.size > 0:
                        line += f", zp.shape={zp_arr.shape}"
                        if zp_arr.size <= 4:
                            line += f", zp={zp_arr.flatten().tolist()}"
                        else:
                            flat = zp_arr.flatten()
                            line += f", zp=[{flat[0]}, ..., {flat[-1]}]"
                if inp.get('desc', 'N/A') != 'N/A':
                    line += f", desc={inp.get('desc')}"
                lines.append(line)
            lines.append(_color("Outputs Info: "))
            for out in self.get("outputs", []):
                shape_str = ", ".join(str(d) for d in out.get("shape", []))
                quanti = "SCALE" if out.get('quantiType', 0) == 1 else "NONE"
                line = (f"  [{out.get('index', '?')}][{out.get('name', '?')}]: "
                        f"{out.get('dtype', '?')}, ({shape_str}), "
                        f"size={out.get('alignedByteSize', '?')}, quanti={quanti}")
                if quanti == "SCALE":
                    scale_arr = out.get('scale')
                    zp_arr = out.get('zero_point')
                    axis = out.get('quantizeAxis', 0)
                    line += f", axis={axis}"
                    if scale_arr is not None and scale_arr.size > 0:
                        line += f", scale.shape={scale_arr.shape}"
                        if scale_arr.size <= 4:
                            line += f", scale={scale_arr.flatten().tolist()}"
                        else:
                            flat = scale_arr.flatten()
                            line += f", scale=[{flat[0]:.6f}, ..., {flat[-1]:.6f}]"
                    if zp_arr is not None and zp_arr.size > 0:
                        line += f", zp.shape={zp_arr.shape}"
                        if zp_arr.size <= 4:
                            line += f", zp={zp_arr.flatten().tolist()}"
                        else:
                            flat = zp_arr.flatten()
                            line += f", zp=[{flat[0]}, ..., {flat[-1]}]"
                if out.get('desc', 'N/A') != 'N/A':
                    line += f", desc={out.get('desc')}"
                lines.append(line)
            lines.append("====================================================")
            return "\n".join(lines)
        except Exception:
            return f"ModelSummary(<error: {dict.__repr__(self)}>)"

    def __str__(self):
        return self.__repr__()


class BenchmarkResult(dict):
    """推理耗时结果 (dict 子类 + 格式化输出)。

    与 pyCauchyKesai.BenchmarkResult 格式完全一致。
    """

    def __repr__(self):
        lines = [
            _color("Inference Info: "),
            f"  Time: {self['time_us']:.6g} us",
            f"  Time: {self['time_ms']:.6g} ms",
            f"  Time: {self['time_s']:.6g} s",
        ]
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


# ============================================================================
# is_infer 状态机 (Python 级模拟)
# ============================================================================

# 状态常量 — 与 C++ 原版语义一致
_IS_INFER_IDLE = 0      # 空闲，可提交
_IS_INFER_RUNNING = 1   # 已提交，推理中
_IS_INFER_WAITING = 2   # 等待中


class _InferStateMachine:
    """Python 级 is_infer 状态机模拟。

    每个 task_id 对应一个状态:
        0 (IDLE) → 1 (RUNNING) → 2 (WAITING) → 0 (IDLE)

    使用 threading.Lock 保证线程安全 (替代 C++ std::atomic + CAS)。
    """

    def __init__(self, n_task: int):
        self._states = [_IS_INFER_IDLE] * n_task
        self._lock = threading.Lock()

    def try_start(self, task_id: int) -> bool:
        """尝试 0→1 转换 (类似 CAS 0→1)。成功返回 True，失败返回 False。"""
        with self._lock:
            if self._states[task_id] != _IS_INFER_IDLE:
                return False
            self._states[task_id] = _IS_INFER_RUNNING
            return True

    def try_wait(self, task_id: int) -> bool:
        """尝试 1→2 转换 (类似 CAS 1→2)。

        返回 (success, reason):
            (True, "") — 成功转换
            (False, "not_in_use") — 状态为 0
            (False, "already_waiting") — 状态为 2
        """
        with self._lock:
            state = self._states[task_id]
            if state == _IS_INFER_IDLE:
                return False, "not_in_use"
            if state == _IS_INFER_WAITING:
                return False, "already_waiting"
            self._states[task_id] = _IS_INFER_WAITING
            return True, ""

    def finish(self, task_id: int) -> None:
        """设置状态为 IDLE (2→0)。"""
        with self._lock:
            self._states[task_id] = _IS_INFER_IDLE

    def rollback(self, task_id: int) -> None:
        """异常回滚到 IDLE。"""
        with self._lock:
            self._states[task_id] = _IS_INFER_IDLE

    def is_busy(self, task_id: int) -> bool:
        """检查是否处于非 IDLE 状态。"""
        with self._lock:
            return self._states[task_id] != _IS_INFER_IDLE

    def force_idle_all(self) -> None:
        """析构时强制所有 slot 回到 IDLE (类似 C++ 析构中的清理)。"""
        with self._lock:
            for i in range(len(self._states)):
                self._states[i] = _IS_INFER_IDLE


# ============================================================================
# EZ_ONNX — 主类
# ============================================================================

class EZ_ONNX:
    """CauchyKesai 接口兼容的 ONNXRuntime 推理引擎。

    使用 ONNXRuntime 替代 BPU 后端，调用方式与 pyCauchyKesai.CauchyKesai 一致
    (仅加载 .onnx 而非 .hbm，无 model_cnt_select 参数)。

    Parameters
    ----------
    model_path : str
        ONNX 模型文件路径 (.onnx)。
    n_task : int
        并发任务槽数 (默认 1, 最大 32)。与 BPU 版 n_task 语义一致——
        多 task_id 允许 start/wait 交替使用。
    _no_alloc : bool
        零内存模式 (默认 False)。True 时不预分配 numpy 缓冲，
        用户需通过 set_inputs/set_outputs 绑定外部数组。

    Examples
    --------
    标准模式:
        model = EZ_ONNX("resnet.onnx")
        outputs = model([img_array])

    异步模式:
        model.start([img_array], task_id=0)
        ...做其他事情...
        outputs = model.wait(task_id=0)

    零内存模式:
        model = EZ_ONNX("resnet.onnx", _no_alloc=True)
        model.set_inputs([input_np], n_task=0)
        model.set_outputs([output_np], n_task=0)
        outputs = model([img_array])
    """

    def __init__(
        self,
        model_path: str,
        n_task: int = 1,
        _no_alloc: bool = False,
    ):
        # ---- 文件检查 ----
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # ---- n_task 校验与 clamp (与原版一致) ----
        self._n_task_orig = n_task
        if n_task < 1 or n_task > 32:
            warnings.warn(
                f"n_task={n_task} 超出 [1, 32] 范围, clamp 到 "
                f"{max(1, min(32, n_task))}",
                UserWarning,
                stacklevel=2,
            )
        self._n_task = max(1, min(32, n_task))
        self._no_alloc_mode = _no_alloc

        # ---- 创建 ORT Session ----
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # 只输出 error
        self._session = ort.InferenceSession(model_path, sess_opts)

        # ---- 初始化模型元信息 ----
        self._model_path = model_path
        self._init_model()

        # ---- 分配缓冲 / 初始化 slot ----
        # 每个 task_id 的状态
        self._state_machine = _InferStateMachine(self._n_task)
        # 每个 task_id 的 Future (异步推理)
        self._futures: List[Optional[concurrent.futures.Future]] = [None] * self._n_task
        # 每个 task_id 的输入 numpy 数组 (用于 input_tensors 属性)
        self._input_tensors: List[List[np.ndarray]] = [[] for _ in range(self._n_task)]
        # 每个 task_id 的输出 numpy 数组 (用于 output_tensors 属性)
        self._output_tensors: List[List[np.ndarray]] = [[] for _ in range(self._n_task)]

        if not _no_alloc:
            # 标准模式: 预分配 NumPyArray 输入/输出
            self._owned_inputs: List[List[NumPyArray]] = [
                [] for _ in range(self._n_task)
            ]
            self._owned_outputs: List[List[NumPyArray]] = [
                [] for _ in range(self._n_task)
            ]
            self._allocate_buffers()
        else:
            # 零内存模式: 用 defer NumPyArray 占位 (virAddr=None → 未分配)
            self._owned_inputs: List[List[NumPyArray]] = [
                [NumPyArray(inp["dtype"], inp["shape"], defer=True)
                 for inp in self._inputs_info]
                for _ in range(self._n_task)
            ]
            self._owned_outputs: List[List[NumPyArray]] = [
                [NumPyArray(out["dtype"], out["shape"], defer=True)
                 for out in self._outputs_info]
                for _ in range(self._n_task)
            ]

        # 线程池 (用于异步 start/wait 模拟)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._n_task)

    # ---- 模型元信息初始化 ----

    def _init_model(self) -> None:
        """从 ORT session 读取输入输出元信息 (类似 _init_model)。"""
        session = self._session

        # 输入信息
        self._inputs_info: List[dict] = []
        self._input_names: List[str] = []
        inputs_meta = session.get_inputs()
        for i, meta in enumerate(inputs_meta):
            name = meta.name
            # 解析输入元信息 → shape + dtype
            shape, dtype_str, onnx_type = self._parse_onnx_type(meta)
            self._input_names.append(name)
            self._inputs_info.append({
                "index": i,
                "name": name,
                "dtype": dtype_str,
                "shape": shape,
                "onnx_type": onnx_type,
                "alignedByteSize": np.dtype(dtype_str).itemsize * (
                    1 if any(d <= 0 for d in shape) else int(np.prod(shape))
                ),
                "quantiType": 0,    # ONNX 默认无量化，量化信息从 Q/DQ 节点提取
                "quantizeAxis": 0,
                "scale": np.array([], dtype=np.float32),
                "zero_point": np.array([], dtype=np.int32),
                "desc": "N/A",
            })

        # 输出信息
        self._outputs_info: List[dict] = []
        self._output_names: List[str] = []
        outputs_meta = session.get_outputs()
        for i, meta in enumerate(outputs_meta):
            name = meta.name
            shape, dtype_str, onnx_type = self._parse_onnx_type(meta)
            self._output_names.append(name)
            self._outputs_info.append({
                "index": i,
                "name": name,
                "dtype": dtype_str,
                "shape": shape,
                "onnx_type": onnx_type,
                "alignedByteSize": np.dtype(dtype_str).itemsize * (
                    1 if any(d <= 0 for d in shape) else int(np.prod(shape))
                ),
                "quantiType": 0,
                "quantizeAxis": 0,
                "scale": np.array([], dtype=np.float32),
                "zero_point": np.array([], dtype=np.int32),
                "desc": "N/A",
            })

        # 尝试提取量化元数据 (从 Q/DQ 节点)
        self._extract_quantization_metadata()

        # 运行环境信息
        self._dnn_version = ort.__version__
        providers = session.get_providers()
        self._bpu_version = f"ONNXRuntime-{','.join(providers)}"
        self._soc_name = platform.processor() or "CPU"
        self._bpu_core_num = os.cpu_count() or 1

        # 模型描述
        try:
            meta = session.get_modelmeta()
            self._model_desc = meta.description or "N/A"
        except Exception:
            self._model_desc = "N/A"

        # 总内存占用估算
        if not self._no_alloc_mode:
            total_bytes = 0
            for inp in self._inputs_info:
                total_bytes += inp["alignedByteSize"] * self._n_task
            for out in self._outputs_info:
                total_bytes += out["alignedByteSize"] * self._n_task
            self._memory_mb = round(total_bytes / (1024 * 1024), 2)
        else:
            self._memory_mb = 0.0

    def _parse_onnx_type(self, meta) -> tuple:
        """从 ORT session 输入/输出元信息解析 shape + dtype。

        Parameters
        ----------
        meta : onnxruntime NodeArg
            session.get_inputs()[i] 或 session.get_outputs()[i] 返回的对象。

        Returns
        -------
        (shape, dtype_str, onnx_type)
            shape: list[int], 动态维度标记为 -1
            dtype_str: numpy dtype 字符串 (如 "float32")
            onnx_type: ONNX element type int
        """
        name = meta.name
        type_str = str(meta.type)  # 如 "tensor(float)"

        # shape: 直接从 meta.shape 获取 (ORT 已处理好)
        shape = list(meta.shape) if meta.shape else [-1]

        # dtype: 从 type string 解析
        # "tensor(float)" → "float32", "tensor(int64)" → "int64", etc.
        dtype_str = self._type_str_to_numpy(type_str)
        onnx_type = _NUMPY_STR_TO_ONNX_TYPE.get(dtype_str, 1)

        return shape, dtype_str, onnx_type

    def _type_str_to_numpy(self, type_str: str) -> str:
        """ORT type string → numpy dtype 字符串。

        Examples: "tensor(float)" → "float32", "tensor(uint8)" → "uint8"
        """
        # 提取括号内的类型名
        if "tensor(" in type_str:
            onnx_name = type_str.split("tensor(")[1].rstrip(")")
        else:
            onnx_name = type_str

        # ONNX 类型名 → numpy dtype 字符串映射
        mapping = {
            "float": "float32",
            "double": "float64",
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "bool": "bool",
            "float16": "float16",
            "bfloat16": "float32",  # bfloat16 无 numpy 支持，降级为 float32
            "string": "object",     # string 类型特殊处理
        }
        return mapping.get(onnx_name, "float32")

    def _extract_quantization_metadata(self) -> None:
        """尝试从 ONNX 模型图中提取量化元数据 (scale/zero_point)。

        扫描 QuantizeLinear / DequantizeLinear 节点，
        将 scale/zero_point 信息关联到对应的输入/输出张量。
        如果模型不含量化节点，则 quantiType 保持 0 (NONE)。
        """
        try:
            # 获取模型图 (通过 ONNX 库直接解析)
            import onnx
            onnx_model = onnx.load(self._model_path)

            # 收集所有 QuantizeLinear / DequantizeLinear 节点的 scale/zp
            quant_info = {}  # tensor_name → {scale, zero_point, axis}
            for node in onnx_model.graph.node:
                if node.op_type == "QuantizeLinear":
                    # 输入: x, scale, zero_point (可选)
                    output_name = node.output[0]
                    scale_init_name = node.input[1] if len(node.input) > 1 else ""
                    zp_init_name = node.input[2] if len(node.input) > 2 else ""
                    # 从 initializers 中找 scale/zp
                    scale_val, zp_val = self._find_initializer_values(
                        onnx_model, scale_init_name, zp_init_name
                    )
                    quant_info[output_name] = {
                        "scale": scale_val,
                        "zero_point": zp_val,
                        "axis": 0,  # 默认 per-tensor
                    }
                elif node.op_type == "DequantizeLinear":
                    input_name = node.input[0]
                    scale_init_name = node.input[1] if len(node.input) > 1 else ""
                    zp_init_name = node.input[2] if len(node.input) > 2 else ""
                    scale_val, zp_val = self._find_initializer_values(
                        onnx_model, scale_init_name, zp_init_name
                    )
                    quant_info[input_name] = {
                        "scale": scale_val,
                        "zero_point": zp_val,
                        "axis": 0,
                    }

            # 将量化信息关联到 inputs_info / outputs_info
            for inp in self._inputs_info:
                if inp["name"] in quant_info:
                    q = quant_info[inp["name"]]
                    inp["quantiType"] = 1  # SCALE
                    inp["scale"] = q["scale"]
                    inp["zero_point"] = q["zero_point"]
                    inp["quantizeAxis"] = q["axis"]

            for out in self._outputs_info:
                if out["name"] in quant_info:
                    q = quant_info[out["name"]]
                    out["quantiType"] = 1
                    out["scale"] = q["scale"]
                    out["zero_point"] = q["zero_point"]
                    out["quantizeAxis"] = q["axis"]

        except ImportError:
            # onnx 库不可用时跳过量化提取
            pass
        except Exception:
            # 解析失败时跳过
            pass

    def _find_initializer_values(self, onnx_model, scale_name: str, zp_name: str) -> tuple:
        """从 ONNX 模型 initializers 中找 scale 和 zero_point 的 numpy 值。"""
        init_map = {}
        for init in onnx_model.graph.initializer:
            try:
                arr = onnx.numpy_helper.to_array(init)
                init_map[init.name] = arr
            except Exception:
                pass

        scale_val = init_map.get(scale_name, np.array([], dtype=np.float32))
        zp_val = init_map.get(zp_name, np.array([], dtype=np.int32))
        return scale_val, zp_val

    # ---- 缓冲分配 (标准模式) ----

    def _allocate_buffers(self) -> None:
        """为所有 task_id 预分配输入输出 NumPyArray (类似标准构造函数)。"""
        for t in range(self._n_task):
            # 输入
            for i, inp in enumerate(self._inputs_info):
                shape = inp["shape"]
                # 动态维度 (-1) 替换为 1 (最小有效尺寸)
                alloc_shape = tuple(max(1, d) if d <= 0 else d for d in shape)
                np_arr = NumPyArray(inp["dtype"], alloc_shape, cached=True)
                self._owned_inputs[t].append(np_arr)
                self._input_tensors[t].append(np_arr.as_array())

            # 输出
            for i, out in enumerate(self._outputs_info):
                shape = out["shape"]
                alloc_shape = tuple(max(1, d) if d <= 0 else d for d in shape)
                np_arr = NumPyArray(out["dtype"], alloc_shape, cached=True)
                self._owned_outputs[t].append(np_arr)
                self._output_tensors[t].append(np_arr.as_array())

    # ---- 核心推理 ----

    def inference(
        self,
        inputs: List[np.ndarray],
        task_id: int = 0,
        priority: int = 0,
    ) -> List[np.ndarray]:
        """同步推理 (类似 CauchyKesai.inference)。

        Parameters
        ----------
        inputs : list[np.ndarray]
            输入数组列表，顺序与 input_names 对应。
        task_id : int
            任务槽 ID (默认 0)。
        priority : int
            优先级 (CPU 无优先级概念，参数保留但不生效)。

        Returns
        -------
        list[np.ndarray]
            输出数组列表，顺序与 output_names 对应。
        """
        self._validate_task_id(task_id)
        self._validate_inputs(inputs)

        # 构建 input_feed
        input_feed = {}
        for i, name in enumerate(self._input_names):
            input_feed[name] = inputs[i]

        # 运行推理
        results = self._session.run(self._output_names, input_feed)

        # 更新 input_tensors / output_tensors
        self._input_tensors[task_id] = [np.array(inp) for inp in inputs]
        self._output_tensors[task_id] = results

        return results

    def __call__(
        self,
        inputs: List[np.ndarray],
        task_id: int = 0,
        priority: int = 0,
    ) -> List[np.ndarray]:
        """便捷推理入口 (同 inference)。"""
        return self.inference(inputs, task_id=task_id, priority=priority)

    # ---- 异步推理 ----

    def start(
        self,
        inputs: List[np.ndarray],
        task_id: int = 0,
        priority: int = 0,
    ) -> None:
        """异步提交推理任务 (类似 CauchyKesai.start)。

        使用线程池模拟 BPU 的异步提交。
        调用后需通过 wait() 获取结果。

        Parameters
        ----------
        inputs : list[np.ndarray]
            输入数组列表。传入空列表 [] 时跳过输入写入 (零拷贝路径)。
        task_id : int
            任务槽 ID。
        priority : int
            优先级 (CPU 无优先级，保留参数)。
        """
        self._validate_task_id(task_id)

        # 状态机: 0→1
        if not self._state_machine.try_start(task_id):
            raise RuntimeError(f"task_id {task_id} 正在使用中 (is_busy)")

        # 零拷贝路径: 空列表跳过输入写入
        if len(inputs) > 0:
            self._validate_inputs(inputs)

            # 写入到预分配缓冲 (类似 C++ 的 memcpy to ION)
            if not self._no_alloc_mode:
                owned = self._owned_inputs[task_id]
                for i, inp_arr in enumerate(inputs):
                    dst = owned[i].as_array()
                    src = np.ascontiguousarray(inp_arr)
                    if src.nbytes > dst.nbytes:
                        # 输入大于缓冲 → 直接传原始数组给 session.run
                        pass
                    else:
                        dst[:] = src.flat[:dst.size]

        # 提交到线程池
        actual_inputs = inputs if len(inputs) > 0 else [
            arr.copy() for arr in self._input_tensors[task_id]
        ]
        input_feed = {}
        for i, name in enumerate(self._input_names):
            input_feed[name] = np.ascontiguousarray(actual_inputs[i])

        def _run():
            return self._session.run(self._output_names, input_feed)

        future = self._executor.submit(_run)
        self._futures[task_id] = future

    def safe_start(
        self,
        inputs: List[np.ndarray],
        task_id: int = 0,
        priority: int = 0,
    ) -> None:
        """带校验的异步提交 (类似 CauchyKesai.safe_start)。

        相比 start() 增加了:
        1. 输入数量校验 (空列表跳过)
        2. task_id / priority 范围校验
        3. 零内存模式下检查输入输出是否已绑定
        """
        # 空列表跳过输入数量校验 (零拷贝路径)
        if len(inputs) > 0:
            self._validate_inputs(inputs)

        self._validate_task_id(task_id)

        if priority < 0 or priority > 255:
            raise ValueError(f"priority 必须 >= 0 且 <= 255, got {priority}")

        # 零内存模式: 检查缓冲是否已绑定
        if self._no_alloc_mode:
            for np_arr in self._owned_inputs[task_id]:
                if not np_arr.is_allocated:
                    raise RuntimeError(
                        "零内存模式下输入未绑定, 请先调用 set_inputs()"
                    )
            for np_arr in self._owned_outputs[task_id]:
                if not np_arr.is_allocated:
                    raise RuntimeError(
                        "零内存模式下输出未绑定, 请先调用 set_outputs()"
                    )

        self.start(inputs, task_id=task_id, priority=priority)

    def wait(self, task_id: int = 0) -> List[np.ndarray]:
        """等待异步推理完成并返回输出 (类似 CauchyKesai.wait)。

        Parameters
        ----------
        task_id : int
            任务槽 ID。

        Returns
        -------
        list[np.ndarray]
            输出数组列表。
        """
        self._validate_task_id(task_id)

        # 状态机: 1→2
        success, reason = self._state_machine.try_wait(task_id)
        if not success:
            if reason == "not_in_use":
                raise RuntimeError(f"task_id {task_id} 未在运行 (not in use)")
            elif reason == "already_waiting":
                raise RuntimeError(f"task_id {task_id} 已在等待中 (already being waited)")

        try:
            future = self._futures[task_id]
            if future is None:
                raise RuntimeError(f"task_id {task_id} 没有 pending Future")

            results = future.result()  # 阻塞等待
            self._futures[task_id] = None

            # 更新 output_tensors
            self._output_tensors[task_id] = results

            # 状态机: 2→0
            self._state_machine.finish(task_id)

            return results
        except Exception as e:
            # 异常回滚
            self._state_machine.rollback(task_id)
            self._futures[task_id] = None
            raise

    def is_busy(self, task_id: int = 0) -> bool:
        """检查任务槽是否被占用 (类似 CauchyKesai.is_busy)。"""
        self._validate_task_id(task_id)
        return self._state_machine.is_busy(task_id)

    # ---- 零内存模式绑定 ----

    def set_inputs(self, ion_inputs: List[Union[NumPyArray, np.ndarray]], n_task: int = 0) -> None:
        """绑定外部输入缓冲 (类似 CauchyKesai.set_inputs)。

        仅在零内存模式 (_no_alloc=True) 下可用。
        接收 NumPyArray 或 numpy ndarray 列表。

        Parameters
        ----------
        ion_inputs : list[NumPyArray | np.ndarray]
            输入缓冲列表，顺序与 input_names 对应。
        n_task : int
            绑定到哪个任务槽 (默认 0)。
        """
        if not self._no_alloc_mode:
            raise RuntimeError("set_inputs 仅在零内存模式 (_no_alloc=True) 下可用")

        self._validate_task_id(n_task)
        self._bind_inputs(ion_inputs, n_task)

    def set_outputs(self, ion_outputs: List[Union[NumPyArray, np.ndarray]], n_task: int = 0) -> None:
        """绑定外部输出缓冲 (类似 CauchyKesai.set_outputs)。

        仅在零内存模式 (_no_alloc=True) 下可用。

        Parameters
        ----------
        ion_outputs : list[NumPyArray | np.ndarray]
            输出缓冲列表，顺序与 output_names 对应。
        n_task : int
            绑定到哪个任务槽 (默认 0)。
        """
        if not self._no_alloc_mode:
            raise RuntimeError("set_outputs 仅在零内存模式 (_no_alloc=True) 下可用")

        self._validate_task_id(n_task)
        self._bind_outputs(ion_outputs, n_task)

    def _bind_inputs(self, ion_inputs: list, n_task: int) -> None:
        """内部输入绑定逻辑 (类似 _bind_ion_inputs)。"""
        # 数量校验
        if len(ion_inputs) != len(self._inputs_info):
            raise ValueError(
                f"输入数量不匹配: 期望 {len(self._inputs_info)}, "
                f"实际 {len(ion_inputs)}"
            )

        # 逐个校验 + 绑定
        bound_arrays: List[NumPyArray] = []
        for i, item in enumerate(ion_inputs):
            inp_info = self._inputs_info[i]

            if isinstance(item, NumPyArray):
                # dtype 校验
                if _numpy_dtype_to_str(item.dtype) != inp_info["dtype"]:
                    raise ValueError(
                        f"输入 [{i}] dtype 不匹配: 期望 {inp_info['dtype']}, "
                        f"实际 {_numpy_dtype_to_str(item.dtype)}"
                    )
                # ndim 校验
                if item.ndim != len(inp_info["shape"]):
                    raise ValueError(
                        f"输入 [{i}] ndim 不匹配: 期望 {len(inp_info['shape'])}, "
                        f"实际 {item.ndim}"
                    )
                # shape 校验 (动态维度 -1 跳过)
                for j, (expected, actual) in enumerate(zip(inp_info["shape"], item.shape)):
                    if expected > 0 and expected != actual:
                        raise ValueError(
                            f"输入 [{i}] shape 维度 {j} 不匹配: "
                            f"期望 {expected}, 实际 {actual}"
                        )
                bound_arrays.append(item)
            elif isinstance(item, np.ndarray):
                # dtype 校验
                if _numpy_dtype_to_str(item.dtype) != inp_info["dtype"]:
                    raise ValueError(
                        f"输入 [{i}] dtype 不匹配: 期望 {inp_info['dtype']}, "
                        f"实际 {_numpy_dtype_to_str(item.dtype)}"
                    )
                # shape 校验
                for j, (expected, actual) in enumerate(zip(inp_info["shape"], item.shape)):
                    if expected > 0 and expected != actual:
                        raise ValueError(
                            f"输入 [{i}] shape 维度 {j} 不匹配: "
                            f"期望 {expected}, 实际 {actual}"
                        )
                # 包装为 NumPyArray
                bound_arrays.append(NumPyArray._from_array(item))
            else:
                raise TypeError(
                    f"输入 [{i}] 类型错误: 期望 NumPyArray 或 np.ndarray, "
                    f"实际 {type(item)}"
                )

        # 更新 owned_inputs 和 input_tensors
        self._owned_inputs[n_task] = bound_arrays
        self._input_tensors[n_task] = [
            arr.as_array() if isinstance(arr, NumPyArray) else np.array(arr)
            for arr in bound_arrays
        ]

    def _bind_outputs(self, ion_outputs: list, n_task: int) -> None:
        """内部输出绑定逻辑 (类似 _bind_ion_outputs)。"""
        if len(ion_outputs) != len(self._outputs_info):
            raise ValueError(
                f"输出数量不匹配: 期望 {len(self._outputs_info)}, "
                f"实际 {len(ion_outputs)}"
            )

        bound_arrays: List[NumPyArray] = []
        for i, item in enumerate(ion_outputs):
            out_info = self._outputs_info[i]

            if isinstance(item, NumPyArray):
                if _numpy_dtype_to_str(item.dtype) != out_info["dtype"]:
                    raise ValueError(
                        f"输出 [{i}] dtype 不匹配: 期望 {out_info['dtype']}, "
                        f"实际 {_numpy_dtype_to_str(item.dtype)}"
                    )
                if item.ndim != len(out_info["shape"]):
                    raise ValueError(
                        f"输出 [{i}] ndim 不匹配: 期望 {len(out_info['shape'])}, "
                        f"实际 {item.ndim}"
                    )
                for j, (expected, actual) in enumerate(zip(out_info["shape"], item.shape)):
                    if expected > 0 and expected != actual:
                        raise ValueError(
                            f"输出 [{i}] shape 维度 {j} 不匹配: "
                            f"期望 {expected}, 实际 {actual}"
                        )
                bound_arrays.append(item)
            elif isinstance(item, np.ndarray):
                if _numpy_dtype_to_str(item.dtype) != out_info["dtype"]:
                    raise ValueError(
                        f"输出 [{i}] dtype 不匹配: 期望 {out_info['dtype']}, "
                        f"实际 {_numpy_dtype_to_str(item.dtype)}"
                    )
                for j, (expected, actual) in enumerate(zip(out_info["shape"], item.shape)):
                    if expected > 0 and expected != actual:
                        raise ValueError(
                            f"输出 [{i}] shape 维度 {j} 不匹配: "
                            f"期望 {expected}, 实际 {actual}"
                        )
                bound_arrays.append(NumPyArray._from_array(item))
            else:
                raise TypeError(
                    f"输出 [{i}] 类型错误: 期望 NumPyArray 或 np.ndarray, "
                    f"实际 {type(item)}"
                )

        self._owned_outputs[n_task] = bound_arrays
        self._output_tensors[n_task] = [
            arr.as_array() if isinstance(arr, NumPyArray) else np.array(arr)
            for arr in bound_arrays
        ]

    def ion_inputs(self, task_id: int = 0) -> List[NumPyArray]:
        """返回 auto-allocated 输入 NumPyArray (类似 CauchyKesai.ion_inputs)。

        标准模式返回预分配的 NumPyArray 列表。
        零内存模式下返回用户绑定的 NumPyArray 列表。
        """
        self._validate_task_id(task_id)
        return self._owned_inputs[task_id]

    def ion_outputs(self, task_id: int = 0) -> List[NumPyArray]:
        """返回 auto-allocated 输出 NumPyArray (类似 CauchyKesai.ion_outputs)。"""
        self._validate_task_id(task_id)
        return self._owned_outputs[task_id]

    # ---- 信息查询 ----

    def s(self) -> ModelSummary:
        """返回模型摘要 (类似 CauchyKesai.s)。

        Returns
        -------
        ModelSummary
            包含模型路径、输入输出信息、运行环境等字段。
        """
        info = {
            "model_path": self._model_path,
            "model_names": [
                {"index": 0, "name": self._input_names[0] if self._input_names else "onnx_model",
                 "selected": True}
            ],
            "n_task": self._n_task,
            "memory_mb": self._memory_mb,
            "dnn_version": self._dnn_version,
            "bpu_version": self._bpu_version,
            "soc_name": self._soc_name,
            "model_desc": self._model_desc,
            "bpu_core_num": self._bpu_core_num,
            "inputs": self._inputs_info,
            "outputs": self._outputs_info,
        }
        return ModelSummary(info)

    def t(self) -> BenchmarkResult:
        """单次推理计时 (类似 CauchyKesai.t)。

        在 task_id=0 上执行一次同步推理，测量耗时。

        Returns
        -------
        BenchmarkResult
            包含 time_us, time_ms, time_s 字段。
        """
        # 构造 dummy 输入
        dummy_inputs = []
        for inp in self._inputs_info:
            shape = tuple(max(1, d) if d <= 0 else d for d in inp["shape"])
            dummy_inputs.append(np.zeros(shape, dtype=inp["dtype"]))

        start_time = time.perf_counter()
        self._session.run(self._output_names, {
            name: np.ascontiguousarray(dummy_inputs[i])
            for i, name in enumerate(self._input_names)
        })
        elapsed_us = (time.perf_counter() - start_time) * 1_000_000

        return BenchmarkResult({
            "time_us": elapsed_us,
            "time_ms": elapsed_us / 1_000,
            "time_s": elapsed_us / 1_000_000,
        })

    # ---- 调度参数 ----

    def set_scheduling_params(self, bpu_cores: List[int]) -> None:
        """设置调度参数 (类似 CauchyKesai.set_scheduling_params)。

        CPU 无 BPU 核概念。空列表恢复默认。
        非空列表映射到 session 的 intra_op_num_threads。

        Parameters
        ----------
        bpu_cores : list[int]
            BPU 核列表。在 CPU 上映射为线程数。
        """
        if not bpu_cores:
            # 空列表 → 恢复默认 (不限制线程数)
            # ONNXRuntime 不支持运行时更改线程数，仅记录意图
            pass
        else:
            # 非空列表 → 核数映射为线程数
            n_threads = len(bpu_cores)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = n_threads
            # 注意: session 创建后无法更改线程数，
            # 此处仅记录意图，实际需在构造时通过 SessionOptions 设置
            warnings.warn(
                f"set_scheduling_params: ONNXRuntime 不支持运行时更改线程数, "
                f"请在构造 EZ_ONNX 时通过 SessionOptions 设置 "
                f"intra_op_num_threads={n_threads}",
                UserWarning,
                stacklevel=2,
            )

    # ---- 只读属性 ----

    @property
    def input_tensors(self) -> List[List[np.ndarray]]:
        """所有 task_id 的输入 numpy 数组 (类似 CauchyKesai.input_tensors)。"""
        return self._input_tensors

    @property
    def output_tensors(self) -> List[List[np.ndarray]]:
        """所有 task_id 的输出 numpy 数组 (类似 CauchyKesai.output_tensors)。"""
        return self._output_tensors

    @property
    def input_names(self) -> List[str]:
        """模型输入名称列表。"""
        return self._input_names

    @property
    def output_names(self) -> List[str]:
        """模型输出名称列表。"""
        return self._output_names

    @property
    def dnn_version(self) -> str:
        """ONNXRuntime 版本 (类似 CauchyKesai.dnn_version)。"""
        return self._dnn_version

    @property
    def bpu_version(self) -> str:
        """执行提供者版本 (类似 CauchyKesai.bpu_version)。"""
        return self._bpu_version

    @property
    def soc_name(self) -> str:
        """CPU/平台标识 (类似 CauchyKesai.soc_name)。"""
        return self._soc_name

    @property
    def model_desc(self) -> str:
        """模型描述 (类似 CauchyKesai.model_desc)。"""
        return self._model_desc

    @property
    def bpu_core_num(self) -> int:
        """CPU 核数 (类似 CauchyKesai.bpu_core_num)。"""
        return self._bpu_core_num

    # ---- 内部校验 ----

    def _validate_task_id(self, task_id: int) -> None:
        """校验 task_id 范围。"""
        if task_id < 0 or task_id >= self._n_task:
            raise IndexError(
                f"task_id={task_id} 超出范围 [0, {self._n_task - 1}]"
            )

    def _validate_inputs(self, inputs: List[np.ndarray]) -> None:
        """校验输入数量、dtype、shape (类似 _bind_ion_inputs 的校验)。"""
        if len(inputs) != len(self._inputs_info):
            raise ValueError(
                f"输入数量不匹配: 期望 {len(self._inputs_info)}, "
                f"实际 {len(inputs)}"
            )

        for i, inp_arr in enumerate(inputs):
            inp_info = self._inputs_info[i]
            # dtype 校验
            if _numpy_dtype_to_str(inp_arr.dtype) != inp_info["dtype"]:
                raise ValueError(
                    f"输入 [{i}] dtype 不匹配: 期望 {inp_info['dtype']}, "
                    f"实际 {_numpy_dtype_to_str(inp_arr.dtype)}"
                )
            # ndim 校验
            if inp_arr.ndim != len(inp_info["shape"]):
                raise ValueError(
                    f"输入 [{i}] ndim 不匹配: 期望 {len(inp_info['shape'])}, "
                    f"实际 {inp_arr.ndim}"
                )
            # shape 校验 (动态维度 -1 跳过)
            for j, (expected, actual) in enumerate(zip(inp_info["shape"], inp_arr.shape)):
                if expected > 0 and expected != actual:
                    raise ValueError(
                        f"输入 [{i}] shape 维度 {j} 不匹配: "
                        f"期望 {expected}, 实际 {actual}"
                    )

    # ---- repr ----

    def __repr__(self) -> str:
        try:
            busy_count = sum(1 for i in range(self._n_task) if self.is_busy(i))
            busy_str = f", {busy_count} busy" if busy_count else ""
            n_inputs = len(self._inputs_info)
            n_outputs = len(self._outputs_info)
            return (f"EZ_ONNX(model='{self._model_path}', "
                    f"inputs={n_inputs}, outputs={n_outputs}, "
                    f"n_task={self._n_task}{busy_str})")
        except Exception:
            return "EZ_ONNX(<error>)"

    # ---- 析构 ----

    def __del__(self) -> None:
        """析构: 清理 pending tasks + 释放线程池。"""
        try:
            # 强制所有 slot 回到 IDLE
            self._state_machine.force_idle_all()
            # 取消所有 pending futures
            for i in range(self._n_task):
                if self._futures[i] is not None:
                    self._futures[i].cancel()
                    self._futures[i] = None
            # 关闭线程池
            self._executor.shutdown(wait=False)
        except Exception:
            pass


# ============================================================================
# 模块级导出
# ============================================================================

__all__ = ["EZ_ONNX", "NumPyArray", "ModelSummary", "BenchmarkResult"]