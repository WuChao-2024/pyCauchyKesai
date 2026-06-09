![](sources/imgs/CauchyKesai.jpeg)

# pyCauchyKesai

pyCauchyKesai 是地平线 BPU 平台的 Python AI Native 推理接口。基于 C++ 实现 + pybind11 绑定，将 BPU 推理管线封装为 `numpy in → numpy out` 的 Python 模块，支持地平线全系 BPU 架构（Nash-e / Nash-m / Nash-p）。

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/Numpy-1.x/2.x-00A0E8?logo=numpy&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/Platform-Nash--e|Nash--m|Nash--p-FF6F00" height="25" />
  <img src="https://img.shields.io/badge/ABI-stable_abi3-blue?logo=python&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?logo=gnu&logoColor=white" height="25" />
</p>

- **版本**: `0.2.0`
- **平台**: Linux aarch64（地平线 RDK 系列开发板）
- **Python**: >= 3.10（abi3 兼容，一次编译跨 3.10/3.11/3.12）
- **License**: GNU AGPL v3
- **作者**: Cauchy - WuChao in D-Robotics

---

## 目录结构

```
pyCauchyKesai/
├── pyproject.toml                          # 统一构建配置 (scikit-build-core)
├── README.md / README_cn.md               # 中英文文档
├── LICENSE                                 # GNU AGPL v3
├── CONTRIBUTING.md                         # 贡献指南
├── IMPROVEMENTS.md                         # 改进记录
│
├── src/pyCauchyKesai/                      # Python 包 (src-layout)
│   └── __init__.py                         # CauchyKesai, IONArray, ModelSummary, BenchmarkResult
│
├── csrc/                                   # 全部 C/C++ 源码
│   ├── CMakeLists.txt                      # 顶层: HB_PLATFORM 平台切换
│   ├── common/
│   │   └── bpu_align.h                     # BPU 对齐宏 (32/64 字节自适应)
│   └── nash/                               # Nash-e / Nash-m / Nash-p 平台
│       ├── CMakeLists.txt                  # 编译 + Horizon 库打包逻辑
│       ├── cauchy_kesai.cpp                # 推理引擎实现 (1025 行)
│       ├── bindings.cpp                    # pybind11 绑定
│       ├── ion_array.h / ion_array.cpp     # IONArray: ION 内存管理器
│       ├── include/
│       │   ├── cauchykesai/pycauchykesai.h # CauchyKesai 类定义 (217 行)
│       │   └── hobot/                      # 地平线 SDK 头文件 (hb_dnn.h, hb_ucp.h 等)
│       └── lib/                            # 预编译 SDK 动态库 (git tracked，编译链接用)
│
├── tests/                                  # 测试套件
│   ├── conftest.py                         # 平台检测 + pytest markers
│   ├── test_cauchy_kesai.py                # CauchyKesai 功能测试
│   └── nash/test_ion_array.py              # IONArray 综合测试 (8 种排列)
│
├── examples/
│   └── ion_preprocess_demo.py              # ION 前处理零拷贝 Demo (YOLO / ResNet)
│
├── scripts/
│   └── build_wheels.sh                     # 多平台 wheel 构建脚本
│
└── dist/                                   # 构建产物 (.whl)
```

---

## 平台矩阵

| 产品 | BPU 架构 | CMake 参数 | 对齐字节 | 链接库 |
|------|---------|-----------|---------|--------|
| RDK S100 | nash-e | `Nash-e` | 32 | libhbucp, libdnn, libhbrt4, libhbtl |
| RDK S100P | nash-m | `Nash-m` | 32 | libhbucp, libdnn, libhbrt4, libhbtl |
| RDK S600 | nash-p | `Nash-p` | 64 | libhbucp, libdnn, libhbrt4, libhbtl |

---

## AI Native 设计理念

pyCauchyKesai 从设计之初就考虑了 AI Agent 的使用场景，让 LLM 能直接理解并调用边缘 NPU 模型。

### 1. 结构化 + 人类友好的双模式输出

`.s()` 和 `.t()` 返回 **既能 print 又能结构化访问** 的 `dict` 子类：

```python
# 人类: terminal 中直接看格式化输出
model.s()
# ================== Model Summary ==================
# Model File: /opt/.../model.hbm
# Inputs Info:
#   [0][images_y]: uint8, (1, 640, 640, 1), size=409600, quanti=NONE
# ...

# Agent: 结构化访问
info = model.s()
input_shape = info['inputs'][0]['shape']  # [1, 640, 640, 1]
input_dtype = info['inputs'][0]['dtype']  # 'uint8'
```

### 2. 精确的异常处理

校验失败抛出带详细描述的异常，Agent 可捕获并自动修正：

```python
try:
    outputs = model([wrong_input])
except ValueError as e:
    # ValueError: shape mismatch at input[0] 'images_y': expected (1, 640, 640, 1), got (1, 480, 640, 1)
    pass
```

异常类型: `IndexError` (范围越界) / `ValueError` (dtype/shape/ndim 不匹配) / `RuntimeError` (槽位占用或 BPU 调用失败)

### 3. 自动内存连续性处理

```python
x = data.transpose(0, 2, 3, 1)   # 非 C 连续
outputs = model([x])               # 自动处理，无需手动 ascontiguousarray
```

内部使用 `py::array::ensure()` 检测：已 C 连续的零开销直传，非连续的自动创建临时拷贝。

### 4. 零拷贝 ION 内存管理

`IONArray` 是纯 ION 内存管理器，分配 CPU/BPU 共享物理内存，通过 `as_array()` 导出标准 `np.ndarray` 视图：

```python
from pyCauchyKesai import IONArray
import numpy as np

# 分配 ION 内存
ion = IONArray(np.dtype('float32'), [1, 3, 224, 224])

# 导出为标准 numpy 数组（零拷贝视图）
arr = ion.as_array()         # → np.ndarray, shape (1, 3, 224, 224)
arr[:] = data                # 标准 numpy 操作

# numpy 兼容属性
print(ion.shape)             # (1, 3, 224, 224) — tuple, 与 numpy 一致
print(ion.dtype)             # dtype('float32')
print(ion.ndim)              # 4
print(ion.size)              # 150528

# BPU 侧
print(hex(ion.phy_addr))     # 物理地址
ion.flush()                  # CPU cache → BPU
ion.invalidate()             # BPU → CPU cache

# 元素级偏移子视图
sub = ion.sub_view(np.dtype('float32'), [1, 1, 224, 224], offset=224*224)

# 懒分配
ion_deferred = IONArray(np.dtype('float32'), [1, 3, 224, 224], defer=True)
ion_deferred.allocate(cached=True)   # 按需分配
```

### 5. 任务状态查询 `is_busy()`

```python
free_slot = next(i for i in range(model.s()['n_task']) if not model.is_busy(i))
model.start(inputs, task_id=free_slot)
```

### 6. 多线程真并发

`wait()` 和 `start()` 在 BPU 等待期间通过 `py::gil_scoped_release` 释放 GIL，多线程可真正并发执行。`hbUCPWaitTaskDone` 内部使用 `pthread_cond_clockwait` 阻塞等待——是纯 C 库调用，完全不感知 Python GIL。

### 7. 自描述模型对象

```python
print(model)
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=4)
# 推理中: CauchyKesai(model='...', inputs=2, outputs=6, n_task=4, 2 busy)
```

### 8. 可捕获的构造警告

参数自动修正时通过标准 `warnings` 模块发出警告，Agent 可捕获：

```python
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = CauchyKesai(path, n_task=99)
    # UserWarning: n_task > 32, clamped to 32
```

---

## 快速开始

```python
from pyCauchyKesai import CauchyKesai
import numpy as np

# 加载模型
model = CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm")

# 查看模型信息
print(model)
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=1)

# 查看详细摘要
model.s()

# 查看输入/输出名称
print("Inputs:", model.input_names)    # ['images_y', 'images_uv']
print("Outputs:", model.output_names)  # ['output0', '318', '342', ...]

# 性能测试 (脏跑一次)
model.t()

# 构造 NV12 输入
y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 推理
outputs = model([y, uv])
print(outputs[0].shape)  # (1, 80, 80, 80)
```

---

## 安装

### 前置条件

| 依赖 | 说明 |
|------|------|
| Python >= 3.10 | 目标运行环境 |
| CMake >= 3.18 | 构建系统 |
| g++ (C++17) | 编译器 |
| pybind11 >= 2.12 | pip 自动拉取 |
| scikit-build-core >= 0.9 | pip 自动拉取 |
| 地平线 BPU 系统库 | `/usr/hobot/lib/` (板端预装) |

### 编译安装

```bash
# Nash-p / RDK S600
HB_PLATFORM=Nash-p pip install .

# Nash-e / RDK S100
HB_PLATFORM=Nash-e pip install .

# Nash-m / RDK S100P
HB_PLATFORM=Nash-m pip install .
```

构建时 CMake 自动将 `/usr/hobot/lib/` 中的依赖 .so 打包进 wheel 的 `pyCauchyKesai/lib/` 目录。运行时通过 `RPATH=$ORIGIN/lib` 自动定位，无需额外设置 `LD_LIBRARY_PATH`。

### 构建 wheel（分发用）

```bash
# 单版本
HB_PLATFORM=Nash-p pip wheel . -w dist/

# 多 Python 版本
./scripts/build_wheels.sh Nash-p
```

生成文件格式: `pycauchykesai-0.2.0-cp312-cp312-linux_aarch64.whl`

### 验证安装

```bash
python3 -c "from pyCauchyKesai import CauchyKesai; print('OK')"
```

---

## 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `HB_PLATFORM` | 目标 BPU 平台 | `Nash-p` |
| `HB_SYSROOT` | 地平线 SDK 根路径 | `/usr/hobot` |

---

## 架构设计

### is_infer 三态机

推理任务的生命周期由原子状态机管理：

```
     ┌──────────────────────────────────┐
     │                                  │
     ▼                                  │
  ┌─────┐  start()/t()   ┌─────┐  wait()  ┌─────┐
  │  0  │ ────CAS──────→ │  1  │ ──CAS──→ │  2  │
  │空闲 │                │已提交│          │等待中│
  └─────┘                └─────┘          └─────┘
     ▲                      │                │
     │       store(0)       │   store(0)     │
     └──────────────────────┴────────────────┘
```

- `start()`: CAS 0→1，提交 BPU 任务
- `wait()`: CAS 1→2，等待 BPU 完成，完成后 store(0)
- `t()`: CAS 0→1，提交后立即 CAS 1→0 (单步完成)
- `is_busy()`: load(acquire) != 0 → 槽位被占用

所有状态转换使用 `std::atomic<int>` + `memory_order_acquire/release`，保证无锁线程安全。

### 双模式统一架构

两条路径内部走完全相同的 `_bind_ion_inputs` / `_bind_ion_outputs` 代码：

```
标准模式 (默认):
  构造 → 自动创建 IONArray → _bind_ion_inputs/outputs → 绑定到 BPU Tensor

零内存模式 (_no_alloc=True):
  构造 → (不分配内存) → set_inputs/set_outputs → _bind_ion_inputs/outputs → 绑定到 BPU Tensor
```

标准模式创建的 IONArray 通过 `shared_ptr` 管理，可通过 `ion_inputs()` / `ion_outputs()` 暴露给 Python。

### IONArray 内存管理

`IONArray` 是纯 C++ 内存管理器，**不继承** `np.ndarray`：

```
IONArray (ION Memory Manager)
│
├── 持有: hbUCPSysMem (phyAddr, virAddr, memSize)
├── 存储: dtype + shape (逻辑类型和形状)
├── 生命周期: shared_ptr 管理
│
├── as_array()           → np.ndarray (CPU 视图, capsule 保持 ION 存活)
├── sub_view(dtype, shape, offset) → IONArray 子视图 (元素级偏移, 零拷贝)
├── allocate(cached=True)→ 懒分配: 按 dtype×shape 分配 ION 内存
├── flush() / invalidate() → Cache 操作
├── phy_addr / mem_size / is_cached / is_allocated → 内存属性
└── dtype / shape / ndim / size / nbytes → numpy 兼容属性
```

关键设计决策:
- **as_array() 而非继承 numpy**: 返回标准 `np.ndarray`，100% 兼容 numpy 生态。通过 `py::capsule` 持有 `shared_ptr<IONArray>`，确保 ION 内存在数组存活期间不被释放。
- **元素级偏移**: `sub_view(dtype, shape, offset)` 中 offset 是元素个数（非字节），内部自动乘 `dtype.itemsize()`。
- **子视图生命周期**: 子视图通过内部 `parent_` 持有父对象 `shared_ptr`，父内存不会被提前释放。
- **懒分配**: `IONArray(dtype, shape, defer=True)` 只记录类型形状，`.allocate()` 按需分配。

---

## API 参考

### CauchyKesai

#### 构造函数

**标准构造**: `CauchyKesai(model_path, n_task=1, model_cnt_select=0)`

加载 BPU 模型，自动为每个任务槽分配 ION 输入/输出内存。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | `.hbm` 模型文件路径 |
| `n_task` | `int` | `1` | 最大并发任务数 (1~32)，每个槽独立分配 ION 内存 |
| `model_cnt_select` | `int` | `0` | Packed 模型中子模型的索引 |

超范围参数自动 clamp 并通过 `warnings` 模块告警。

**零内存构造**: `CauchyKesai(model_path, n_task=1, model_cnt_select=0, _no_alloc=True)`

预分配任务槽但不分配任何 ION 内存。推理前必须通过 `set_inputs()` / `set_outputs()` 绑定外部 IONArray，实现完全零拷贝。

```python
from pyCauchyKesai import CauchyKesai, IONArray
import numpy as np

model = CauchyKesai("model.hbm", n_task=1, _no_alloc=True)

ion_in  = IONArray(np.dtype('uint8'),   [1, 640, 640, 1])
ion_out = IONArray(np.dtype('float32'), [1, 80, 80, 80])

model.set_inputs([ion_in], n_task=0)
model.set_outputs([ion_out], n_task=0)

ion_in.as_array()[:] = my_data
ion_in.flush()
model.start([], task_id=0)
model.wait(task_id=0)
ion_out.invalidate()
result = ion_out.as_array().copy()
```

#### `.s()` → ModelSummary

返回 `dict` 子类，包含模型完整元信息。print 时输出格式化文本，同时支持结构化访问。

```python
info = model.s()
info['model_path']              # str
info['n_task']                  # int
info['memory_mb']               # float
info['dnn_version']             # str  — hbDNNGetVersion()
info['bpu_version']             # str  — hbUCPGetVersion()
info['soc_name']                # str  — 芯片型号
info['bpu_core_num']            # int  — 编译核数
info['model_names']             # list[dict] — packed 模型列表
info['inputs'][i]['name']       # str
info['inputs'][i]['shape']      # list[int]
info['inputs'][i]['dtype']      # str
info['inputs'][i]['quantiType'] # int  — 0=NONE, 1=SCALE
info['outputs'][i]['name']      # str
info['outputs'][i]['shape']     # list[int]
info['outputs'][i]['dtype']     # str
```

#### `.t()` → BenchmarkResult

使用 task_id=0 脏跑一次完整推理，返回 `dict` 子类，包含 `time_us`、`time_ms`、`time_s`。

```python
bench = model.t()
print(f"{bench['time_ms']:.2f} ms")
```

#### `.inference(inputs, task_id=0, priority=0)` / `.__call__`

同步推理: 校验 → 提交 → 等待 → 返回结果。`model(inputs)` 等价于 `model.inference(inputs)`。

wait 阶段释放 GIL，多线程调用不互相阻塞。

```python
outputs: list[np.ndarray] = model([input1, input2])
```

#### `.start(inputs, task_id=0, priority=0)`

异步提交，立即返回。传空列表 `[]` 跳过 memcpy，使用预写入 ION 内存的数据。提交阶段释放 GIL。

```python
model.start([input1, input2], task_id=0, priority=128)   # 普通模式
model.start([], task_id=0)                                 # 零拷贝模式
```

#### `.safe_start(inputs, task_id=0, priority=0)`

带预飞行校验的异步提交——校验 task_id/priority 范围、槽位空闲、ION 绑定状态，通过后调用 `start()`。推荐用户使用此方法代替直接调用 `start()`。

#### `.wait(task_id=0)`

阻塞等待指定槽位结果，返回零拷贝输出张量列表。等待阶段释放 GIL。

```python
outputs: list[np.ndarray] = model.wait(task_id=0)
```

#### `.is_busy(task_id=0)` → bool

查询指定任务槽是否正在推理。

#### `.input_tensors` / `.output_tensors`

ION 内存 numpy 视图: `list[list[np.ndarray]]`, shape `[n_task][count]`。

```python
np.copyto(model.input_tensors[0][0], my_data)   # 零拷贝写入
model.start([], task_id=0)                       # 跳过拷贝
model.wait(task_id=0)
result = model.output_tensors[0][0]              # 零拷贝读取
```

> 使用 `input_tensors` 零拷贝写入时，源数组必须 C 连续。非连续数组先调 `np.ascontiguousarray()`。

#### `.input_names` / `.output_names` → list[str]

模型输入/输出张量名称，顺序对应 `input_tensors` / `output_tensors` 的第二维。

#### `.ion_inputs(task_id=0)` / `.ion_outputs(task_id=0)` → list[IONArray]

获取标准模式自动创建的 IONArray 列表。每个元素可访问 `phy_addr`、`mem_size`、`as_array()`、`sub_view()` 等。零内存模式返回空列表。

#### `.set_inputs(ion_inputs, n_task=0)` / `.set_outputs(ion_outputs, n_task=0)`

零内存模式专用: 为指定槽位绑定外部 IONArray。

#### `.set_scheduling_params(bpu_cores)`

设置 BPU 核心调度掩码:

```python
model.set_scheduling_params(bpu_cores=[0, 1, 2, 3])   # 绑定 4 核
model.set_scheduling_params(bpu_cores=[0, 1])          # 绑定 2 核
model.set_scheduling_params(bpu_cores=[])              # 重置为 ANY
```

#### 运行时环境属性（只读）

| 属性 | 类型 | 来源 |
|------|------|------|
| `model.dnn_version` | `str` | `hbDNNGetVersion()` |
| `model.bpu_version` | `str` | `hbUCPGetVersion()` |
| `model.soc_name` | `str` | `hbUCPGetSocName()` |
| `model.model_desc` | `str` | `hbDNNGetModelDesc()` |
| `model.bpu_core_num` | `int` | `hbDNNGetCompileBpuCoreNum()` |

### IONArray

#### 构造函数

**标准分配**: `IONArray(dtype, shape, cached=True)`

立即分配 `dtype × shape` 大小的 ION 物理内存。`cached=True` 启用 CPU cache 一致性（默认），`cached=False` 获得更高 BPU 带宽但无 CPU cache。

**指定字节大小**: `IONArray(dtype, shape, byte_size, cached=True)`

用于 BPU 对齐场景，`byte_size` 可大于 `dtype × shape` 的理论大小。

**懒分配**: `IONArray(dtype, shape, cached=True, defer=True)`

只记录 dtype + shape，不分配内存。稍后调用 `.allocate(cached=True)` 实际分配。

#### 只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `phy_addr` | `int` | BPU 可访问的物理地址 |
| `mem_size` | `int` | ION 分配的字节数 |
| `is_cached` | `bool` | 是否使用 CPU cache |
| `is_allocated` | `bool` | 是否已分配 ION 内存 |
| `dtype` | `np.dtype` | 元素类型 |
| `shape` | `tuple` | 逻辑形状 (与 numpy 一致) |
| `ndim` | `int` | 维度数 |
| `size` | `int` | 总元素数 |
| `nbytes` | `int` | 总字节数 |

#### 方法

| 方法 | 说明 |
|------|------|
| `as_array()` | 返回标准 `np.ndarray` 零拷贝视图。数组通过 capsule 持有 IONArray 引用，ION 内存不会提前释放 |
| `flush()` | 刷新 CPU cache → BPU 可见 (uncached 时 no-op) |
| `invalidate()` | 使 CPU cache 失效 → CPU 可见 BPU 最新数据 (uncached 时 no-op) |
| `sub_view(dtype, shape, offset)` | 元素级偏移子视图。offset 为**元素个数**（非字节）。子视图通过 parent_ 引用保持父对象存活 |
| `allocate(cached=True)` | 懒分配: 按构造时记录的 dtype×shape 分配 ION 内存 |

#### as_array() 生命周期

```python
ion = IONArray(np.dtype('float32'), [1000])
arr = ion.as_array()

# arr 和 ion 指向同一块 ION 物理内存
# arr 通过 capsule 持有 ion 的 shared_ptr
del ion
arr[0] = 42.0   # 仍然安全 — ION 内存未被释放
```

---

## 使用示例

### 标准推理

```python
from pyCauchyKesai import CauchyKesai
import numpy as np

model = CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm")

y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

outputs = model([y, uv])
print(f"output[0] shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
```

### 零拷贝推理 (input_tensors)

```python
model = CauchyKesai("/path/to/model.hbm", n_task=1)
task_id = 0

y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

np.copyto(model.input_tensors[task_id][0], y)
np.copyto(model.input_tensors[task_id][1], uv)

model.start([], task_id=task_id)
model.wait(task_id=task_id)

result = model.output_tensors[task_id][0]
```

### 零内存模式 + 外部 IONArray

```python
from pyCauchyKesai import CauchyKesai, IONArray

model = CauchyKesai("model.hbm", n_task=1, _no_alloc=True)

ion_in  = IONArray(np.dtype('uint8'),   [1, 640, 640, 1])
ion_out = IONArray(np.dtype('float32'), [1, 80, 80, 80])

model.set_inputs([ion_in], n_task=0)
model.set_outputs([ion_out], n_task=0)

ion_in.as_array()[:] = data
ion_in.flush()
model.start([], task_id=0)
model.wait(task_id=0)
ion_out.invalidate()
result = ion_out.as_array().copy()
```

### ION 内存分区 (sub_view)

```python
from pyCauchyKesai import IONArray
import numpy as np

# 分配大块 ION: 3000 个 float32
big_ion = IONArray(np.dtype('float32'), [3000])

# 按元素偏移分区 (offset 是元素个数，非字节)
slot_0 = big_ion.sub_view(np.dtype('float32'), [1000], offset=0)
slot_1 = big_ion.sub_view(np.dtype('float32'), [1000], offset=1000)
slot_2 = big_ion.sub_view(np.dtype('float32'), [1000], offset=2000)

# 三个子视图共享同一块 ION 物理内存，物理地址连续
print(hex(slot_0.phy_addr), hex(slot_1.phy_addr), hex(slot_2.phy_addr))

# 子视图保持父对象存活
del big_ion
arr = slot_0.as_array()   # 安全
```

### 多路并发推理

```python
import threading
from pyCauchyKesai import CauchyKesai
import numpy as np

model = CauchyKesai("/path/to/model.hbm", n_task=4)

def run(task_id):
    y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
    uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)
    result = model.inference([y, uv], task_id=task_id)

threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
```

### 异步管线

```python
model = CauchyKesai("/path/to/model.hbm", n_task=2)

model.start([y_a, uv_a], task_id=0)
model.start([y_b, uv_b], task_id=1)   # 不等 task 0

result_a = model.wait(task_id=0)
result_b = model.wait(task_id=1)
```

### IONArray 前处理 (零拷贝优化)

```python
from pyCauchyKesai import IONArray
import numpy as np
import cv2

img = cv2.imread("test.jpg")   # BGR uint8 HWC

# resize 直写 ION
ion_u8 = IONArray(np.dtype('uint8'), [640, 640, 3])
cv2.resize(img, (640, 640), dst=ion_u8.as_array())

# BGR→RGB + HWC→CHW (全是视图操作，零拷贝)
rgb  = ion_u8.as_array()[..., ::-1]
chw  = rgb.transpose(2, 0, 1)
nchw = chw[np.newaxis]

# float32 转换 + 写入 ION + 原地 normalize
ion_f32 = IONArray(np.dtype('float32'), [1, 3, 640, 640])
arr = ion_f32.as_array()
arr[:] = nchw.astype(np.float32)
arr /= 255.0
ion_f32.flush()
```

---

## 输入校验

调用 `inference()` / `safe_start()` 时自动执行以下校验，失败抛出带详细描述的异常：

| 校验项 | 异常类型 | 示例错误信息 |
|--------|----------|-------------|
| task_id 范围 | `IndexError` | `task_id out of range: got 5, valid range [0, 3]` |
| priority 范围 | `IndexError` | `priority out of range: got 300, valid range [0, 255]` |
| 输入数量 | `ValueError` | `input count mismatch: expected 2, got 1` |
| dtype | `ValueError` | `dtype mismatch at input[0] 'images_y': expected uint8, got float32` |
| ndim | `ValueError` | `ndim mismatch at input[0] 'images_y': expected 4, got 2` |
| shape | `ValueError` | `shape mismatch at input[0] 'images_y': expected (1,640,640,1), got ...` |
| 槽位占用 | `RuntimeError` | `task_id 0 is already in use` |
| no_alloc 未绑定 | `RuntimeError` | `no_alloc mode: ION inputs for n_task=0 not set` |
| sub_view 越界 | `IndexError` | `sub_view: offset + view_bytes exceeds mem_size` |

---

## 支持的数据类型

| NumPy dtype | BPU 类型 | 位宽 |
|-------------|---------|------|
| `uint8` | U8 | 8-bit |
| `int8` | S8 | 8-bit |
| `float16` | F16 | 16-bit |
| `int16` | S16 | 16-bit |
| `uint16` | U16 | 16-bit |
| `float32` | F32 | 32-bit |
| `int32` | S32 | 32-bit |
| `uint32` | U32 | 32-bit |
| `float64` | F64 | 64-bit |
| `int64` | S64 | 64-bit |
| `uint64` | U64 | 64-bit |
| `bool` | BOOL8 | 8-bit |

---

## 测试

```bash
# 全量 (需 BPU 硬件)
pytest tests/ -v -m "bpu"

# 跳过耗时测试
pytest tests/ -v -m "bpu and not slow"
```

---

## 常见问题

**Q: 传入 `torch.Tensor` 报 TypeError**

pybind11 只接受 `numpy.ndarray`:
```python
outputs = model([tensor.cpu().detach().numpy()])
```

**Q: `ImportError: GLIBCXX_3.4.30 not found`**

conda 环境的 libstdc++ 版本过低:
```bash
conda install libstdcxx-ng -c conda-forge
```

**Q: 非 C 连续数组会怎样**

`inference()` / `start()` 内部自动处理，只有一次 CPU→ION 搬运。零拷贝模式 (`input_tensors` + `np.copyto`) 仍需手动确保 C 连续。

**Q: `ion_inputs()` 和 `input_tensors` 有什么区别**

- `input_tensors`: `list[list[np.ndarray]]` — 纯 numpy 视图，直接读写数据
- `ion_inputs()`: 返回 `IONArray` 列表 — 可访问物理地址、cache 操作、创建子视图

两者指向同一块 ION 物理内存，只是暴露的接口粒度不同。

**Q: `as_array()` 和旧版 `._array` 有什么区别**

v0.2.0 起使用 `as_array()` 替代旧版 `._array`:
- `as_array()`: 返回标准 `np.ndarray`，通过 capsule 管理 ION 内存生命周期
- `._array`: 已移除（旧版 IONArray 直接继承 numpy 时的遗留属性）

**Q: `sub_view()` 的 offset 参数是字节还是元素个数**

**元素个数**，不是字节数。内部自动计算: `字节偏移 = offset × this.dtype.itemsize()`。

例如 float32 IONArray，`offset=100` 表示跳过 100 个 float32 (400 字节)。

---

## 声明

所有源代码均开源，使用前请确保对程序有足够了解。本接口供社区开发者使用，不完全保证功能正确性。如发现问题，欢迎提 issue 或 PR。

---

## 附录 1: 更新 OE SDK

重新编译时如需对齐更新版本的 OpenExplore SDK:

**更新头文件** (`csrc/nash/include/hobot/`):
```bash
rm -rf csrc/nash/include/hobot
cp -r /path/to/oe/samples/ucp_tutorial/deps_aarch64/ucp/include/hobot csrc/nash/include/
```

**系统库**由 `/usr/hobot/lib/` 提供。项目 `csrc/nash/lib/` 中的预编译 .so 用于离线编译和 git 版本锁定。构建时 CMake 从 `/usr/hobot/lib/` 拷贝依赖 .so 打包进 wheel。

**查看库版本**:
```bash
strings /usr/hobot/lib/libhbucp.so | grep SO_VERSION
# SO_VERSION = (3U).(7U).(4U)
```

---

## 附录 2: pybind11 GIL 与多线程并发分析

### 问题现象

Python 多线程中，多个线程各自持有 `task_id` 并发调用 `inference()` 或 `wait()` 时，执行实际串行化。

### 根因

pybind11 绑定的 C++ 函数默认持有 GIL。原 `wait()` 未释放 GIL:

```
线程 A: 调用 wait() → 持有 GIL → hbUCPWaitTaskDone → 内核态阻塞等待
                                                    ↑
                                   GIL 被 A 持有，线程 B 永远无法被调度
```

### 证据

1. `libhbucp.so` 不含任何 Python 符号 (`nm -D | grep PyEval` 无输出) — 纯 C 库，无法自行释放 GIL
2. `hbUCPWaitTaskDone` 内部使用 `pthread_cond_clockwait` 阻塞等待
3. 反汇编确认调用路径: `pthread_mutex_lock → condition_wait → pthread_mutex_unlock`

### 修复

在 `start()` 的 `hbUCPSubmitTask` 和 `wait()` 的 `hbUCPWaitTaskDone` 调用处包裹 `py::gil_scoped_release`:

```cpp
// start()
{
    py::gil_scoped_release release;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "...");
}

// wait()
{
    py::gil_scoped_release release;
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0), "...");
}
```

memcpy 阶段（`inputs[i].data()` 访问 Python 对象）仍需持有 GIL。

### 修复后并发行为

```
线程 A: start() → [释放GIL] → BPU运行... → wait()[释放GIL] → 等待完成 → 结果
线程 B:              start() → [释放GIL] → BPU运行... → wait()[释放GIL] → 等待完成 → 结果
线程 C:                             start() → [释放GIL] → BPU运行...
```

BPU 推理在多线程间真正重叠执行，GIL 仅在 memcpy 和返回值构造时短暂持有。

---

## 附录 3: IONArray v0.2.0 迁移指南

从 v0.1.0 升级到 v0.2.0 时的 API 变更:

| v0.1.0 (旧) | v0.2.0 (新) | 说明 |
|-------------|-------------|------|
| `ion._array` | `ion.as_array()` | 返回标准 np.ndarray |
| `ion.sub_view(dtype, shape, byte_offset=N)` | `ion.sub_view(dtype, shape, offset=N)` | offset 改为元素个数 |
| `ion._array[:] = data` | `ion.as_array()[:] = data` | |
| `cv2.resize(img, dst=ion._array)` | `cv2.resize(img, dst=ion.as_array())` | |

核心变化: IONArray 从 numpy 子类变为纯内存管理器，通过 `as_array()` 按需导出标准 numpy 视图。
