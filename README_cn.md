![](sources/imgs/CauchyKesai.jpeg)

# pyCauchyKesai

pyCauchyKesai 是地平线 BPU 平台的 Python AI Native 推理接口。基于 C++ 实现 + pybind11 绑定，将 BPU 推理管线封装为 `numpy in → numpy out` 的 Python 模块，支持地平线全系 BPU 架构（Nash-e / Nash-m / Nash-p）。

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/Numpy-1.x/2.x-00A0E8?logo=numpy&logoColor=white" height="25" />
  <img src="https://img.shields.io/badge/Platform-Nash--e|Nash--m|Nash--p-FF6F00" height="25" />
  <img src="https://img.shields.io/badge/License-AGPL--3.0-blue?logo=gnu&logoColor=white" height="25" />
</p>

- **平台**: Linux aarch64（地平线 RDK 系列开发板）
- **Python**: >= 3.10（构建与部署须同一 Python 版本；3.10–3.14 已实测）
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
│   └── __init__.py                         # CauchyKesai, IONArray, IONMemory, from_numpy, Platform
│
├── csrc/                                   # 全部 C/C++ 源码
│   ├── CMakeLists.txt                      # 顶层构建配置
│   └── nash/                               # Nash-e / Nash-m / Nash-p 平台
│       ├── CMakeLists.txt                  # 编译 + Horizon 库打包逻辑
│       ├── cauchy_kesai.cpp                # 推理引擎实现
│       ├── bindings.cpp                    # pybind11 绑定
│       ├── ion_array.h / ion_array.cpp     # IONArray: ION 内存管理器
│       ├── include/
│       │   ├── cauchykesai/pycauchykesai.h # CauchyKesai 类定义
│       │   └── hobot/                      # 地平线 SDK 头文件 (hb_dnn.h, hb_ucp.h 等)
│       └── lib/                            # 预编译 SDK 动态库 (git tracked，编译链接用)
│
├── tests/                                  # 测试套件
│   ├── conftest.py                         # 平台检测 + pytest markers
│   ├── test_cauchy_kesai.py                # CauchyKesai 功能测试
│   ├── test_ai_native.py                   # AI Native 接口契约测试
│   ├── nash/                               # 纯内存层测试 (IONMemory / IONArray)
│   ├── golden/                             # 黄金模型三路一致性 + 量化刻画
│   ├── dimensions/                         # 11 维度接口一致性套件
│   └── generate/                           # 模型生成流水线
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

| 产品 | BPU 架构 | 链接库 |
|------|---------|--------|
| RDK S100 | nash-e | libhbucp, libdnn, libhbrt4, libhbtl |
| RDK S100P | nash-m | libhbucp, libdnn, libhbrt4, libhbtl |
| RDK S600 | nash-p | libhbucp, libdnn, libhbrt4, libhbtl |

> ⚠️ 产出的 wheel 能跑在哪些板上取决于**构建机的 glibc / 工具链**：S100/S100P 构建可三板通用，S600 构建仅 S600 可用。详见 [安装指南 · 移植性说明](docs/cn_install.md#移植性说明wheel-能跑在哪些板上)。

---

## AI Native 设计理念

pyCauchyKesai 从设计之初就考虑了 AI Agent 的使用场景，让 LLM 能直接理解并调用边缘 NPU 模型。

### 1. 结构化数据 + 美观打印分离

`summary()` / `benchmark()` **始终返回原生 `dict`**（numpy 已转原生，可直接 json.dumps，供 AI / 程序结构化消费）；`s()` / `t()` / `platform.s()` 内部 print 美观带颜色的字符串（直接调用即出图，返回 None）：

```python
# 人类: terminal 中看彩色格式化输出（内部 print，返回 None）
model.s()                        # 美观打印模型摘要
# ================== Model Summary ==================
# Model File: /opt/.../model.hbm
# Inputs Info:
#   [0][images_y]: uint8, (1, 640, 640, 1), size=409600, quanti=NONE
# ...
model.t()                        # 美观打印计时
model.platform.s()               # 美观打印平台信息

# Agent: 结构化访问（始终返回原生 dict，可 json.dumps）
info = model.summary()
input_shape = info['inputs'][0]['shape']  # [1, 640, 640, 1]
input_dtype = info['inputs'][0]['dtype']  # 'uint8'
```

### 2. 精确的异常处理

校验失败抛出带详细描述的异常，Agent 可捕获并自动修正：

```python
try:
    outputs = model([wrong_input])
except ValueError as e:
    # ValueError: shape mismatch at input[0]: expected (1, 640, 640, 1), got (1, 480, 640, 1)
    # (通过 set_inputs/set_outputs 绑定路径会提供完整的 expected/got 信息)
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

`IONArray` 是自描述 Tensor + ION 内存管理器，分配 CPU/BPU 共享物理内存，通过 `numpy()` 导出标准 `np.ndarray` 视图。从模型创建时携带完整 tensor 性质：

```python
from pyCauchyKesai import IONArray
import numpy as np

# 分配 ION 内存
ion = IONArray(np.dtype('float32'), [1, 3, 224, 224])

# 导出为标准 numpy 数组（零拷贝视图）
arr = ion.numpy()         # → np.ndarray, shape (1, 3, 224, 224)
arr[:] = data                # 标准 numpy 操作

# numpy 兼容属性
print(ion.shape)             # (1, 3, 224, 224) — tuple, 与 numpy 一致
print(ion.dtype)             # dtype('float32')
print(ion.ndim)              # 4
print(ion.size)              # 150528

# Cache 维护（CPU ↔ BPU 一致性）
ion.flush_clean()                  # CPU cache → BPU
ion.flush_invalidate()             # BPU → CPU cache

# 从模型创建的 IONArray 有完整 tensor 性质（完整诊断见 model.summary()）
ion_model = model.make_input(0)
print(ion_model.is_quantized)      # False

# 量化操作（透明，NONE 时 no-op）
float_result = ion_model.dequantize()
ion_model.quantize(float_arr)

# 懒分配
ion_deferred = IONArray(np.dtype('float32'), [1, 3, 224, 224], defer=True)
ion_deferred.allocate(cached=True)   # 按需分配
```

### 5. 任务状态查询 `is_busy()`

```python
free_slot = next(i for i in range(model.summary()['n_task']) if not model.is_busy(i))
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
model.summary()

# 查看输入/输出名称
print("Inputs:", model.input_names)    # ['images_y', 'images_uv']
print("Outputs:", model.output_names)  # ['output0', '318', '342', ...]

# 性能测试 (脏跑一次)
model.benchmark()

# 构造 NV12 输入
y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 推理
outputs = model([y, uv])
print(outputs[0].shape)  # (1, 80, 80, 80)
```

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [安装指南](docs/cn_install.md) | conda 环境、构建 wheel、安装、动态库管理、OE SDK 替换 |
| [API 参考](docs/API_cn.md) | 全部类完整 API（CauchyKesai / IONArray / IONMemory / EZ_ONNX / tools）+ 数据类型 + 常见用法 + 异常汇总 |
| [CauchyKesai](docs/cn_CauchyKesai.md) | 推理编排器：快速开始、任务槽状态机、内存管理、BPU 核调度 |
| [IONArray](docs/cn_IONArray.md) | ION 内存管理器 + 自描述 Tensor、布局、量化、Cache |
| [EZ_ONNX](docs/cn_Easy_ONNX.md) | ONNX Runtime CPU 模拟层（无 BPU 也可推理） |
| [DataAnalyzer](docs/cn_DataAnalyzer.md) | 推理数据分析：量化误差、tensor 对比、精度回归 |
| [测试方法](docs/cn_TestCases.md) | 测试套件结构与运行方法 |

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

`inference()` / `start()` 内部自动处理，只有一次 CPU→ION 搬运。手动零拷贝路径（`ion_inputs()` + `from_numpy()`）同样要求源数组 C 连续，否则先 `np.ascontiguousarray()`。

**Q: `numpy()` 和旧版 `._array` 有什么区别**

v0.2.0 起使用 `numpy()` 替代旧版 `._array`:
- `numpy()`: 返回标准 `np.ndarray`，通过 capsule 管理 ION 内存生命周期
- `._array`: 已移除（旧版 IONArray 直接继承 numpy 时的遗留属性）

---

## 声明

所有源代码均开源，使用前请确保对程序有足够了解。本接口供社区开发者使用，不完全保证功能正确性。如发现问题，欢迎提 issue 或 PR。

---

## 附录 1: pybind11 GIL 与多线程并发分析

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

## 附录 2: IONArray v0.2.0 迁移指南

从 v0.1.0 升级到 v0.2.0 时的 API 变更:

| v0.1.0 (旧) | v0.2.0 (新) | 说明 |
|-------------|-------------|------|
| `ion._array` | `ion.numpy()` | 返回标准 np.ndarray |
| `ion.sub_view(...)` | `IONArray.from_memory(mem, byte_offset, template)` | sub_view 已移除，改用 from_memory（字节偏移） |
| `ion._array[:] = data` | `ion.numpy()[:] = data` | |
| `cv2.resize(img, dst=ion._array)` | `cv2.resize(img, dst=ion.numpy())` | |

核心变化: IONArray 从 numpy 子类变为纯内存管理器，通过 `numpy()` 按需导出标准 numpy 视图。
