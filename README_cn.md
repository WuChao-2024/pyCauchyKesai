![](sources/imgs/CauchyKesai.jpeg)

# pyCauchyKesai

pyCauchyKesai 是面向地平线（Horizon Robotics）BPU 平台的 AI Native 推理层，基于 pybind11 封装 C++ 实现，让 Python 开发者能够在 Nash-e / Nash-m 等边缘 AI 芯片上高效加载和运行 `.hbm` 神经网络模型。

- 版本：`0.0.9`
- 平台：Linux aarch64（地平线 RDK 系列开发板）
- Python：>= 3.10
- 许可证：GNU AGPL v3

BPU相关内容请关注地平线开发者社区释放的[开放材料](https://oe.horizon.auto/)
 
---

## AI Native 设计理念

pyCauchyKesai 是一个 **AI Native** 的推理接口，从设计之初就考虑了 AI Agent 的使用场景，让 LLM 能够直接理解和调用边缘 NPU 模型。

### 核心特性

**1. 结构化 + 人类友好的双模式输出**

传统接口的 `print()` 输出对 Agent 不可见，pyCauchyKesai 的 `.s()` 和 `.t()` 返回的是 **既能 print 又能结构化访问的对象**：

```python
# 人类在 terminal 里直接看格式化输出
model.s()
# ================== Model Summarys ==================
# Model File: /opt/.../yolov8.hbm
# Inputs Info:
# [0][images_y]: uint8, (1, 640, 640, 1, )
# ...

# Agent 可以结构化访问
info = model.s()
input_shape = info['inputs'][0]['shape']  # [1, 640, 640, 1]
input_dtype = info['inputs'][0]['dtype']  # 'uint8'
```

**2. 精确的异常处理**

传统接口出错时 print 错误信息并返回空列表，Agent 无法感知。pyCauchyKesai 抛出带详细描述的异常，Agent 可以捕获并自动修正：

```python
try:
    outputs = model([wrong_input])
except ValueError as e:
    # ValueError: shape mismatch at input[0] 'images_y': expected (1, 640, 640, 1), got (1, 480, 640, 1)
    # Agent 可以解析错误信息，自动调整输入 shape 后重试
    pass
```

异常类型：
- `IndexError`：`task_id` 或 `priority` 超出范围
- `ValueError`：输入数量、dtype、ndim、shape 不匹配
- `RuntimeError`：task 已被占用，或底层 BPU 调用失败

**3. 自动内存连续性处理**

Agent 生成的数组可能经过 `transpose()`、切片等操作，不是 C 连续的。pyCauchyKesai 在 C++ 层自动检测并转换，Agent 无需关心底层细节：

```python
# Agent 生成的非连续数组可以直接传入
x = data.transpose(0, 2, 3, 1)  # 非 C 连续
outputs = model([x])  # 自动处理，无需手动 np.ascontiguousarray()
```

**4. 零拷贝高性能路径**

对于性能敏感的场景，暴露 ION 内存视图（`input_tensors` / `output_tensors`），Agent 可以直接操作硬件内存，避免 CPU-NPU 数据搬运：

```python
# Agent 可以直接写入 NPU 可访问的物理内存
np.copyto(model.input_tensors[0][0], my_data)
model.start([], task_id=0)
result = model.output_tensors[0][0]  # 零拷贝读取
```

**5. 可捕获的构造警告**

传统接口构造时的参数修正（如 `n_task` 被截断）只打印到 stdout，Agent 无法感知。pyCauchyKesai 使用 Python 标准 `warnings` 模块，Agent 可以用 `warnings.catch_warnings()` 捕获并做出响应：

```python
import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = pyCauchyKesai.CauchyKesai(path, n_task=99)
    if w:
        # UserWarning: n_task > 32, clamped to 32
        # Agent 可以记录日志或调整后续逻辑
        print(w[0].message)
```

**6. 任务状态查询 `is_busy()`**

Agent 在多任务调度时需要知道哪些 task slot 空闲，无需靠 try/except 探测：

```python
model = pyCauchyKesai.CauchyKesai(path, n_task=4)

# Agent 可以主动查询，选择空闲的 slot
free_slot = next(i for i in range(4) if not model.is_busy(i))
model.start(inputs, task_id=free_slot)
```

**7. 自描述的模型对象 `__repr__`**

`print(model)` 或在 REPL / Agent 上下文中直接引用模型对象，立即得到关键信息，无需调用 `.s()`：

```python
print(model)
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=4)

# 推理进行中时，busy 状态也会体现
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=4, 2 busy)
```

**8. 多线程真并发**

`wait()` 和 `start()` 在等待 NPU 期间释放 GIL，多个 Agent 线程可以真正并发调用，充分利用 NPU 的多任务能力。

**9. 全程 AI Agent 开发**

本项目从第一行代码到文档撰写，全部由 AI Agent（Claude）完成，是 AI Native 开发模式的实践案例。

---

## 快速开始

```python
import numpy as np
import pyCauchyKesai

# 加载模型
model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm")

# 快速查看模型信息（Agent 友好）
print(model)
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=1)

# 查看详细摘要
model.s()

# 查看输入输出名称
print("输入:", model.input_names)   # ['images_y', 'images_uv']
print("输出:", model.output_names)  # ['output0', '318', '342', ...]

# 性能测试（脏运行）
model.t()

# 构造 NV12 输入（Y 平面 + UV 平面）
y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 推理
outputs = model([y, uv])
print(f"output[0] shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
# output[0] shape: (1, 80, 80, 80), dtype: float32
```

---

## 安装

pyCauchyKesai 不发布到 PyPI，需要从源码编译生成 whl 包后本地安装。

whl 包是**自包含**的：CMake 构建时会将所有 HOBOT C 运行时动态库（`libhbucp.so`、`libdnn.so` 等）一并打包进 `pyCauchyKesai/lib/` 目录，扩展模块通过 `RPATH=$ORIGIN/lib` 在运行时自动找到这些依赖，安装后无需在系统路径中单独部署任何 `.so`。

whl 包与 **Python 版本强绑定**（体现在文件名中，如 `cp312` 表示 CPython 3.12），用哪个环境的 Python 编译，生成的包就只能安装到该版本的 Python 中。

### 前置条件

- Linux aarch64（地平线 RDK 系列开发板）
- Python >= 3.10
- CMake >= 3.18
- C++17 编译器（gcc / g++）
- 网络可访问 PyPI（构建时会自动下载 `scikit-build-core` 和 `pybind11`，仅用于构建，不会安装到用户环境）

### 编译生成 whl

**第一步：进入源码目录**

```bash
cd pyCauchyKesai/Nash
```

**第二步：激活目标 Python 环境**

```bash
# 示例：使用 conda 环境
conda activate your_env

# 确认解释器路径和版本
which python3
python3 --version
```

**第三步：构建 whl**

```bash
pip wheel .
```

构建过程中 pip 会自动在隔离的临时环境中安装 `scikit-build-core` 和 `pybind11`，调用 CMake 编译 C++ 扩展，并将 `Nash/lib/` 下的所有 HOBOT 动态库打包进 whl，完成后清理临时环境，不影响用户环境。

构建完成后在当前目录生成 whl 文件，文件名格式为：

```
pycauchykesai-0.0.9-cp312-cp312-linux_aarch64.whl
#                   ^^^^  ^^^^  ^^^^^^^^^^^^^
#                   Python版本  平台
```

**第四步：安装**

```bash
pip install pycauchykesai-*.whl
```

### 验证安装

```bash
python3 -c "import pyCauchyKesai; print(pyCauchyKesai.__version__)"
# 0.0.9
```

### 在其他环境中使用同一个 whl

只要目标环境的 Python 版本与 whl 文件名中的版本标签一致，可以直接拷贝 whl 文件安装，无需重新编译：

```bash
pip install pycauchykesai-0.0.9-cp312-cp312-linux_aarch64.whl
```

如果目标环境 Python 版本不同（如 3.10），需要切换到对应环境重新执行第二至四步，生成对应版本的 whl。

---

## API 参考

### 导入

```python
import pyCauchyKesai
from pyCauchyKesai import CauchyKesai

# 查看模块信息
print(pyCauchyKesai.__version__, pyCauchyKesai.__date__, pyCauchyKesai.__author__)
print(pyCauchyKesai.__doc__)
```

---

### `CauchyKesai(model_path, n_task=1, model_cnt_select=0)`

加载 BPU 模型，初始化推理环境。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | `.hbm` 模型文件路径 |
| `n_task` | `int` | `1` | 最大并发任务数（1~32），每个任务独立分配输入/输出缓冲区 |
| `model_cnt_select` | `int` | `0` | packed 模型中的模型索引，默认选第 0 个 |

**关于 `n_task`：** 构造时会预分配 `n_task` 套完整的输入/输出 Tensor 内存，多线程或异步场景下不同线程使用不同的 `task_id`，避免内存竞争，也避免运行时动态 malloc 的开销。

**参数自动修正：** 如果 `n_task` 或 `model_cnt_select` 超出有效范围，会自动截断并通过 Python `warnings` 模块发出警告（Agent 可用 `warnings.catch_warnings()` 捕获）。

```python
model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm")  # 单任务
model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm", n_task=4)  # 4 路并发
model = pyCauchyKesai.CauchyKesai("packed.hbm", model_cnt_select=1)  # 选 packed 中第 1 个模型

# Agent 捕获警告
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    model = pyCauchyKesai.CauchyKesai(path, n_task=99)
    if w:
        print(w[0].message)  # n_task > 32, clamped to 32
```

---

### `.s()` — 模型摘要

返回 `ModelSummary` 对象（`dict` 子类）：terminal 中直接输出格式化文本，Agent 可结构化访问。

```python
# 人类使用：直接 print 格式化输出
model.s()

# Agent 使用：结构化访问
info = model.s()
info['model_path']            # str
info['n_task']                # int
info['memory_mb']             # float
info['inputs'][0]['name']     # str，如 'images_y'
info['inputs'][0]['shape']    # list，如 [1, 640, 640, 1]
info['inputs'][0]['dtype']    # str，如 'uint8'
info['outputs'][0]['name']    # str
info['outputs'][0]['shape']   # list
info['outputs'][0]['dtype']   # str
```

---

### `.t()` — 性能测试

使用随机数据脏运行一次推理，返回 `BenchmarkResult` 对象（`dict` 子类）：terminal 中直接输出格式化文本，Agent 可读取数值做决策。

```python
# 人类使用：直接 print 格式化输出
model.t()

# Agent 使用：读取数值
bench = model.t()
bench['time_us']   # float，微秒
bench['time_ms']   # float，毫秒
bench['time_s']    # float，秒
bench['time_min']  # float，分钟
```

---

### `.inference(inputs, task_id=0, priority=0)` — 同步推理

完整的一次推理：参数校验 → 提交任务 → 等待结果。适合单次调用场景。

底层在等待 BPU 完成时会释放 GIL，因此多线程下调用 `.inference()` 不会互相阻塞。

校验不通过时抛出异常（`IndexError` / `ValueError` / `RuntimeError`），不会静默返回空列表。

```python
outputs: List[np.ndarray] = model.inference([input1, input2], task_id=0, priority=0)

# Agent 友好的用法：捕获异常后可解析错误信息自动修正
try:
    outputs = model([input1, input2])
except ValueError as e:
    print(e)  # shape mismatch at input[0] 'images_y': expected (1, 640, 640, 1), got (1, 480, 640, 1)
```

### `model(inputs, task_id=0, priority=0)` — 直接调用（等价于 `.inference()`）

```python
outputs = model([input1, input2])
```

---

### `.is_busy(task_id=0)` — 查询任务状态

返回指定 task slot 是否正在运行推理。Agent 可用于多任务调度，选择空闲的 slot。

```python
# 查询单个 task
if not model.is_busy(0):
    model.start(inputs, task_id=0)

# Agent 多任务调度：找到第一个空闲 slot
free_slot = next((i for i in range(model.s()['n_task']) if not model.is_busy(i)), None)
if free_slot is not None:
    model.start(inputs, task_id=free_slot)
```

**异常：** `task_id` 超出范围时抛出 `IndexError`。

---

### `.start(inputs, task_id=0, priority=0)` — 异步提交

将推理任务提交到 BPU 调度器，立即返回，不等待结果。提交过程中会释放 GIL。

**零拷贝模式：** 如果 `inputs` 为空列表 `[]`，则跳过数据拷贝步骤，直接使用 `input_tensors` 中已写入的数据。适合用户已通过 `np.copyto()` 预先写入 ION 内存的场景。

```python
# 常规模式：传入数据
model.start([input1, input2], task_id=0, priority=128)

# 零拷贝模式：传入空列表（需先通过 input_tensors 写入数据）
np.copyto(model.input_tensors[0][0], my_data)
model.start([], task_id=0)
```

---

### `.wait(task_id=0)` — 等待结果

阻塞等待指定任务完成，返回输出张量列表（零拷贝，直接指向 BPU 输出缓冲区）。

等待期间（`hbUCPWaitTaskDone`）会释放 GIL，其他 Python 线程可以正常被调度。

```python
outputs: List[np.ndarray] = model.wait(task_id=0)
```

---

### `.input_tensors` — 输入 ION 内存视图

类型：`list[list[np.ndarray]]`，形状：`[n_task][input_count]`

构造时预分配的输入 ION 内存的 numpy 视图，**零拷贝**，直接指向 BPU 可访问的物理内存。可用于绕过 `inference()` 内部的 `memcpy`，由用户自行将数据写入 ION 内存后再调用推理。

```python
# 查看输入缓冲区信息
buf = model.input_tensors[0][0]
print(buf.shape, buf.dtype)   # e.g. (1, 640, 640, 1) uint8

# 直接写入 ION 内存（零拷贝）
# 注意：np.copyto() 是 numpy 层操作，my_data 必须是 C 连续数组
# 如有 transpose/permute 请先 np.ascontiguousarray()，或改用 inference()/start() 自动处理
np.copyto(model.input_tensors[task_id][i], my_data)
```

---

### `.output_tensors` — 输出 ION 内存视图

类型：`list[list[np.ndarray]]`，形状：`[n_task][output_count]`

构造时预分配的输出 ION 内存的 numpy 视图，**零拷贝**，直接指向 BPU 写入结果的物理内存。`wait()` 返回的数组与此视图指向同一块内存。

```python
# wait() 完成后直接读取，无需额外拷贝
result = model.output_tensors[task_id][0]
print(result.shape, result.dtype)
```

---

### `.input_names` / `.output_names` — 张量名称

类型：`list[str]`

模型输入/输出张量的名称列表，顺序与 `input_tensors` / `output_tensors` 的第二维一致。

```python
print(model.input_names)   # ['images_y', 'images_uv']
print(model.output_names)  # ['output0', '318', ...]
```

---

### 参数说明

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `inputs` | `List[np.ndarray]` | — | 输入张量列表，顺序和类型须与模型一致 |
| `task_id` | `int` | `[0, n_task-1]` | 任务槽 ID，多并发时用不同 ID 区分 |
| `priority` | `int` | `[0, 255]` | BPU 调度优先级，255 最高，0 最低 |

---

## 使用示例

### 单次推理

```python
import pyCauchyKesai
import numpy as np

model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm")

# 查看模型信息
print("输入:", model.input_names)   # ['images_y', 'images_uv']
print("输出:", model.output_names)  # ['output0', '318', '342', ...]

# 构造 NV12 输入
y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 推理
outputs = model([y, uv])
print(outputs[0].shape)  # (1, 80, 80, 80)
```

### 零拷贝推理（高级用法）

使用 `input_tensors` 和 `output_tensors` 直接操作 ION 内存，避免 `inference()` 内部的 `memcpy`，适合对性能要求极高的场景。

```python
import pyCauchyKesai
import numpy as np

model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm", n_task=1)

task_id = 0

# 准备数据
y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 零拷贝写入 ION 内存
# 注意：np.copyto() 是 numpy 层操作，要求源数组必须是 C 连续的
# 如果数据经过 transpose/permute 等操作，需先调用 np.ascontiguousarray()
np.copyto(model.input_tensors[task_id][0], y)
np.copyto(model.input_tensors[task_id][1], uv)

# 提交推理（传空列表，因为数据已在 ION 内存中）
model.start([], task_id=task_id)

# 等待完成
model.wait(task_id=task_id)

# 零拷贝读取结果（直接访问 ION 内存）
result = model.output_tensors[task_id][0]
print(result.shape, result.dtype)  # (1, 80, 80, 80) float32
```

**注意事项：**
- 使用 `np.copyto()` 直接写入 ION 内存时，源数组必须是 **C 连续的**，否则会导致数据错乱
- 如果数组经过 `transpose()`、`permute()` 等操作，需先调用 `np.ascontiguousarray()` 转换
- 如果不想手动处理连续性，可以改用 `inference()` 或 `start()` 方法，它们会自动处理非连续数组（内部使用 `py::array::ensure()` 自动转换，仅在必要时创建副本）

### 多路并发推理（多线程）

`wait()` 在等待 BPU 期间会释放 GIL，因此多个线程可以真正并发运行，互不阻塞。每个线程使用独立的 `task_id`，对应构造时预分配的独立缓冲区，不存在内存竞争。

```python
import threading
import pyCauchyKesai
import numpy as np

model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm", n_task=4)

def run(task_id):
    y  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
    uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)
    # inference() 内部等价于 start() + wait()，等待期间 GIL 已释放
    result = model.inference([y, uv], task_id=task_id)
    print(f"task {task_id}: {result[0].shape}")

threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join()
```

### 异步流水线

```python
import pyCauchyKesai
import numpy as np

model = pyCauchyKesai.CauchyKesai("/opt/hobot/model/s100/basic/yolov8_640x640_nv12.hbm", n_task=2)

y_a  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv_a = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)
y_b  = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
uv_b = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)

# 提交任务 0
model.start([y_a, uv_a], task_id=0)

# 提交任务 1（不等任务 0 完成）
model.start([y_b, uv_b], task_id=1)

# 按需获取结果
result_a = model.wait(task_id=0)
result_b = model.wait(task_id=1)
```

---

## 输入校验

调用 `.inference()` 时会自动进行以下校验，不符合时抛出带详细描述的异常（不再静默返回空列表）：

| 校验项 | 异常类型 | 错误信息示例 |
|--------|----------|-------------|
| `task_id` 范围 | `IndexError` | `task_id out of range: got 5, valid range [0, 3]` |
| `priority` 范围 | `IndexError` | `priority out of range: got 300, valid range [0, 255]` |
| 输入数量 | `ValueError` | `input count mismatch: expected 2, got 1` |
| 数据类型 | `ValueError` | `dtype mismatch at input[0] 'images_y': expected uint8, got float32` |
| 维度数 | `ValueError` | `ndim mismatch at input[0] 'images_y': expected 4, got 2` |
| 形状 | `ValueError` | `shape mismatch at input[0] 'images_y': expected (1, 640, 640, 1), got (1, 480, 640, 1)` |
| 任务状态 | `RuntimeError` | `task_id 0 is already in use` |

> **关于内存连续性：** `inference()` / `start()` 内部会自动处理非 C 连续数组。如果传入的数组已经是 C 连续的（如直接创建的数组），不会产生额外拷贝；如果是非连续数组（如 `transpose()`、`[::2]` 切片、Fortran order），会自动创建一个 C 连续的副本再拷入 ION 内存。整个过程始终只有一次从 CPU 内存到 ION 内存的搬运，用户无需手动调用 `np.ascontiguousarray()`。

---

## 支持的数据类型

| NumPy dtype | BPU 类型 | 位宽 |
|-------------|----------|------|
| `uint8` | U8 | 8-bit |
| `int8` | S8 | 8-bit |
| `uint16` | U16 | 16-bit |
| `int16` | S16 | 16-bit |
| `float16` | F16 | 16-bit |
| `uint32` | U32 | 32-bit |
| `int32` | S32 | 32-bit |
| `float32` | F32 | 32-bit |
| `uint64` | U64 | 64-bit |
| `int64` | S64 | 64-bit |
| `float64` | F64 | 64-bit |
| `bool` | BOOL8 | 8-bit |

---

## Combine 编译配置参考

### FeatureMap 输入（推荐，行为与 ONNX 完全一致）

全部使用 `featuremap` 类型，模型前后处理行为与 ONNX 保持一致，无隐式预处理。

```yaml
model_parameters:
  onnx_model: 'your_model.onnx'
  march: "nash-e"
  working_dir: 'bpu_model_output'
  output_model_file_prefix: 'BPU_model'

input_parameters:
  input_name: "input1;input2;input3;"
  input_type_rt: 'featuremap;featuremap;featuremap;'
  input_layout_rt: 'NCHW;NCHW;NCHW;'
  input_type_train: 'featuremap;featuremap;featuremap;'
  input_layout_train: 'NCHW;NCHW;NCHW;'
  norm_type: 'no_preprocess;no_preprocess;no_preprocess;'

calibration_parameters:
  cal_data_dir: 'cal_data1;cal_data2;cal_data3;'
  cal_data_type: 'float32;float32;float32;'
  calibration_type: 'default'
  optimization: set_all_nodes_int16

compiler_parameters:
  extra_params: {'input_no_padding': True, 'output_no_padding': True}
  jobs: 16
  compile_mode: 'latency'
  optimize_level: 'O2'
```

---

## 常见问题

**Q: 传入 `torch.Tensor` 报 `TypeError`**

```
TypeError: __call__(): incompatible function arguments.
```

pybind11 在进入 C++ 前做类型检查，只接受 `numpy.ndarray`。请先转换：

```python
outputs = model([tensor.numpy()])
# 或
outputs = model([tensor.cpu().detach().numpy()])
```

---

**Q: `ImportError: version 'GLIBCXX_3.4.30' not found`**

conda 环境的 `libstdc++` 版本低于系统 HOBOT 库的要求，升级即可：

```bash
conda install libstdcxx-ng -c conda-forge
```

---

**Q: 如何在非默认 Python 环境中使用**

找到目标 Python 解释器的库路径，将 `.so` 拷贝过去：

```bash
python3 -c "import os; print(os.path.dirname(os.__file__))"
# /root/miniconda3/envs/myenv/lib/python3.10
cp pyCauchyKesai*.so /root/miniconda3/envs/myenv/lib/python3.10/
```

---

**Q: 传入非 C 连续数组（如 transpose、切片后的数组）会怎样？**

`inference()` / `start()` 内部会自动处理非连续数组，无需手动调用 `np.ascontiguousarray()`。C++ 层使用 `py::array::ensure()` 检测连续性：若已是 C 连续则直接使用原数组（零额外开销），否则自动创建一个 C 连续副本。整个过程始终只有一次从 CPU 内存到 ION 内存的搬运。

```python
# 以下写法均正确，无需手动转换
x = data.transpose(0, 2, 3, 1)
outputs = model([x])  # 自动处理，数据正确

x = data[::2]         # 切片（非连续）
outputs = model([x])  # 同样自动处理
```

**注意：** 使用零拷贝方式（`np.copyto` 直接写入 `input_tensors`）时，`np.copyto` 是纯 numpy 操作，不经过 C++ 层，此时仍需确保源数组是 C 连续的：

```python
# 零拷贝写入前需手动确保 C 连续
x = np.ascontiguousarray(data.transpose(0, 2, 3, 1))
np.copyto(model.input_tensors[task_id][0], x)
```

---

## 声明

所有源代码均开源，使用前请确保您对程序有足够的了解。本接口供社区开发者个人调试使用，不完全保证功能正确性。如发现问题，欢迎提 issue 或 PR。

---


## 附录1：更新 OpenExplore 头文件和动态库（Nash / S100）

重新编译 pyCauchyKesai 时，如果需要对齐到更新版本的 OpenExplore SDK，需要先替换 `Nash/include/` 和 `Nash/lib/` 下的头文件和动态库，再重新执行 `pip wheel .`。

从 OpenExplore 包中获取以下文件：

```
samples/ucp_tutorial/deps_aarch64/ucp/
├── include/hobot/   ← 头文件
└── lib/             ← 动态库
```

**更新 Nash/include/：**

```bash
rm -rf pyCauchyKesai/Nash/include/hobot
cp -r /path/to/ucp/include/hobot pyCauchyKesai/Nash/include/
```

**更新 Nash/lib/：**

```bash
rm pyCauchyKesai/Nash/lib/*.so
cp /path/to/ucp/lib/*.so pyCauchyKesai/Nash/lib/
```

**查看动态库版本：**

```bash
strings pyCauchyKesai/Nash/lib/libhbucp.so | grep SO_VERSION
# SO_VERSION = (3U).(7U).(4U)
```

**查看头文件版本：**

```bash
file="pyCauchyKesai/Nash/include/hobot/hb_ucp.h"
eval $(grep -e 'HB_UCP_VERSION_MAJOR' -e 'HB_UCP_VERSION_MINOR' -e 'HB_UCP_VERSION_PATCH' $file | \
      sed -E 's/#define HB_UCP_VERSION_(MAJOR|MINOR|PATCH)[^0-9]*([0-9]+).*/VERSION_\1=\2/')
echo "$file Version: $VERSION_MAJOR.$VERSION_MINOR.$VERSION_PATCH"
```


## 附录2：pybind11 GIL 与 hbUCPWaitTaskDone 阻塞问题分析

> 研究对象：`libhbucp.so`，版本 `SO_VERSION = (3U).(12U).(3U)`

### 问题现象

在 Python 多线程中，多个线程各持一个 `task_id` 并发调用 `inference()` 或 `wait()`，实际表现为串行执行，无法真正并发。

### 根本原因

**pybind11 绑定的 C++ 函数在执行期间默认持有 GIL（全局解释器锁）。** 只有显式使用 `py::gil_scoped_release` 才会释放。原始代码的 `wait()` 中没有释放 GIL，导致：

```
线程A: 调用 wait() → 持有 GIL → 进入 hbUCPWaitTaskDone → 阻塞在内核等待
                                                              ↑
                                         GIL 被线程A持有且不释放，线程B永远无法被调度
```

这不是 Python 调度器的问题，而是 GIL 被持有后线程 A 自己阻塞，既不计算也不交出锁。

### 证据一：libhbucp.so 不含任何 Python 符号

```bash
nm -D libhbucp.so | grep -E "PyEval|Py_BEGIN|PyGILState|python"
# 无输出
```

`libhbucp.so` 是纯 C/C++ 库，对 Python GIL 一无所知，不可能自行释放。

### 证据二：hbUCPWaitTaskDone 内部使用了阻塞原语

通过 `nm -D libhbucp.so` 可以看到该库依赖以下系统符号：

```
U pthread_cond_clockwait@GLIBC_2.34   ← 条件变量等待，阻塞线程直到信号到来
U pthread_mutex_lock@GLIBC_2.17       ← 互斥锁
U nanosleep@GLIBC_2.17                ← 睡眠等待
U poll@GLIBC_2.17                     ← I/O 轮询等待
```

### 证据三：反汇编确认调用路径

对 `hbUCPWaitTaskDone`（位于 `.so` 偏移 `0x3aae0`）反汇编，可以直接看到函数体内的调用序列：

```asm
3ab20:  bl  pthread_mutex_lock      ← 进入等待前加锁
...
3ab30:  bl  pthread_mutex_unlock    ← 检查状态后解锁
...
3ab64:  blr x2                      ← 调用虚函数 Wait（内部走 pthread_cond_clockwait）
```

这是标准的"加锁 → 条件变量等待 → 解锁"模式，整个过程线程阻塞在内核态，持续时间等于 BPU 推理耗时。

### 修复方案

在 `wait()` 的 `hbUCPWaitTaskDone` 调用处，以及 `start()` 的 `hbUCPSubmitTask` 调用处，用 `py::gil_scoped_release` 包裹：

```cpp
// wait() 中
{
    py::gil_scoped_release release;
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0),
                      "hbUCPWaitTaskDone failed");
}

// start() 中
{
    py::gil_scoped_release release;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                      "hbUCPSubmitTask failed");
}
```

注意：`memcpy`（将 numpy 数据拷入 BPU 缓冲区）阶段仍须持有 GIL， 因为此时在访问 Python 对象 `inputs[i].data()`。

### 修复后的并发行为

```
线程A: start() → [释放GIL] → BPU运行中...          → wait()[释放GIL] → 等待完成 → 返回结果
线程B:               start() → [释放GIL] → BPU运行中... → wait()[释放GIL] → 等待完成 → 返回结果
线程C:                             start() → [释放GIL] → BPU运行中...
```

多个线程的 BPU 推理真正重叠执行，GIL 只在 memcpy 和构造返回值时短暂持有。

---

## 附录3：更新 OpenExplore 头文件和动态库（Bayes / X5）

重新编译 Bayes（RDK X5）版本时，需先替换 `Bayes/include/dnn/` 和 `Bayes/lib/` 下的头文件和动态库，再重新执行 `pip wheel .`。

从 OpenExplore 包中获取以下文件：

```
package/host/host_package/x5_aarch64/dnn/
├── include/dnn/   ← 头文件（hb_dnn.h、hb_sys.h、hb_dnn_status.h 等）
└── lib/           ← 动态库（libdnn.so、libhbrt_bayes_aarch64.so）
```

**更新 Bayes/include/dnn/：**

```bash
rm -rf pyCauchyKesai/Bayes/include/dnn
cp -r /path/to/oe/package/host/host_package/x5_aarch64/dnn/include/dnn pyCauchyKesai/Bayes/include/
```

**更新 Bayes/lib/：**

```bash
rm pyCauchyKesai/Bayes/lib/*.so
cp /path/to/oe/package/host/host_package/x5_aarch64/dnn/lib/libdnn.so pyCauchyKesai/Bayes/lib/
cp /path/to/oe/package/host/host_package/x5_aarch64/dnn/lib/libhbrt_bayes_aarch64.so pyCauchyKesai/Bayes/lib/
```

**查看动态库版本：**

```bash
strings pyCauchyKesai/Bayes/lib/libdnn.so | grep "Runtime version"
# Runtime version = 1.24.5_(3.15.55 HBRT)
```

**说明：** wheel 包中只打包 `libdnn.so` 和 `libhbrt_bayes_aarch64.so`。传递依赖（`libcnn_intf.so`、`libhbmem.so`、`libalog.so`）由 RDK X5 系统在 `/usr/hobot/lib/` 提供，不打包进 wheel。

---
