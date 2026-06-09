# pyCauchyKesai 接口文档

**版本**: 0.2.0
**平台**: Linux aarch64 (Nash-e / Nash-m / Nash-p)
**Python**: >= 3.10

---

## 模块导入

```python
from pyCauchyKesai import CauchyKesai   # 所有平台
from pyCauchyKesai import IONArray      # Nash 平台
```

## 平台 API 差异概览

Nash 系列 (Nash-e/Nash-m/Nash-p) 全功能支持：

| API | 可用 |
|-----|------|
| CauchyKesai (标准模式) | ✅ |
| CauchyKesai (零内存模式) | ✅ |
| IONArray | ✅ |
| set_inputs / set_outputs | ✅ |
| ion_inputs / ion_outputs | ✅ |
| input_tensors / output_tensors | ✅ |

---

## IONArray

ION 物理内存管理器。分配 CPU 和 BPU 共享的物理内存，通过 `as_array()` 按需导出标准 `np.ndarray` 视图。

> 仅 Nash 平台可用（Nash-e / Nash-m / Nash-p）。

### 构造函数

#### `IONArray(dtype, shape, cached=True, defer=False)`

分配 `dtype.itemsize() × prod(shape)` 字节的 ION 物理内存。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dtype` | `np.dtype` | — | 元素类型，如 `np.dtype('float32')` |
| `shape` | `list[int]` | — | 逻辑形状，如 `[1, 3, 224, 224]` |
| `cached` | `bool` | `True` | `True` = `hbUCPMallocCached`，CPU cache 一致；`False` = `hbUCPMalloc`，无 cache，BPU 带宽更高 |
| `defer` | `bool` | `False` | `True` = 只记录 dtype+shape，**不分配**内存，稍后调用 `.allocate()` |

```python
# 立即分配（默认）
ion = IONArray(np.dtype('float32'), [1, 3, 640, 640])

# 懒分配：先记录类型/形状，稍后分配
ion = IONArray(np.dtype('float32'), [1, 3, 640, 640], defer=True)
# ... 此时 dtype/shape 可访问，但 is_allocated=False
ion.allocate(cached=True)  # 实际分配内存
```

#### `IONArray(dtype, shape, byte_size, cached=True)`

显式指定分配字节数。`byte_size` 必须 >= `dtype.itemsize() × prod(shape)`。用于 BPU 对齐场景。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dtype` | `np.dtype` | — | 元素类型 |
| `shape` | `list[int]` | — | 逻辑形状 |
| `byte_size` | `int` | — | 实际分配字节数 (>= dtype×shape) |
| `cached` | `bool` | `True` | 同上 |

```python
# BPU 对齐场景：逻辑形状小，但分配更多空间
ion = IONArray(np.dtype('float32'), [1], byte_size=1024*1024)
```

### 只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `phy_addr` | `int` | BPU 可访问的物理地址 (uint64)；未分配时返回 `0` |
| `mem_size` | `int` | ION 分配的字节数；未分配时返回 `0` |
| `is_cached` | `bool` | 是否使用 CPU cache |
| `is_allocated` | `bool` | 是否已分配 ION 内存（不含 `defer=True` 中的虚拟构造） |
| `dtype` | `np.dtype` | 元素类型，与 `numpy.ndarray.dtype` 一致 |
| `shape` | `tuple[int, ...]` | 逻辑形状，与 `numpy.ndarray.shape` 一致 |
| `ndim` | `int` | 维度数，与 `numpy.ndarray.ndim` 一致 |
| `size` | `int` | 总元素数 (`prod(shape)`)，与 `numpy.ndarray.size` 一致 |
| `nbytes` | `int` | 已分配字节数（numpy 兼容别名，等同于 `mem_size`） |

### 方法

#### `allocate(cached=True)` → `None`

按已存储的 `dtype` × `shape` 分配 ION 物理内存。通常配合 `defer=True` 构造使用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cached` | `bool` | `True` | `True` = hbUCPMallocCached；`False` = hbUCPMalloc |

- 构造时已在 `IONArray` 内部复用此方法
- 重复调用抛出 `RuntimeError("IONArray::allocate: already allocated")`
- 分配后 `is_allocated` 变为 `True`，`phy_addr` / `mem_size` 变为有效值

```python
ion = IONArray(np.dtype('float32'), [1, 3, 640, 640], defer=True)
print(ion.is_allocated)  # False
print(ion.phy_addr)      # 0

ion.allocate(cached=True)
print(ion.is_allocated)  # True
print(ion.phy_addr > 0)  # True

# 可直接用于推理
model.set_inputs([ion])
```

#### `as_array()` → `numpy.ndarray`

返回标准 `np.ndarray` 零拷贝视图，数据直接指向 ION 虚拟地址。

- 返回的数组是纯 `numpy.ndarray`，100% 兼容 numpy 生态
- 内部通过 `py::capsule` 持有 `IONArray` 的 `shared_ptr`，确保 ION 内存不会被提前释放
- 多次调用返回独立的 ndarray 对象，但指向同一块 ION 内存
- **未分配时抛 `RuntimeError("IONArray: not allocated, call allocate() first")`**

```python
ion = IONArray(np.dtype('float32'), [2, 3])
arr = ion.as_array()
arr[:] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

# 生命周期：即使 ion 被 del，只要 arr 还在，ION 内存就不会释放
del ion
print(arr[0, 0])  # 1.0 — 仍然安全
```

#### `sub_view(dtype, shape, offset)` → `IONArray`

在父 ION 内存上创建零拷贝子视图。

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | `np.dtype` | 子视图的元素类型 |
| `shape` | `list[int]` | 子视图的逻辑形状 |
| `offset` | `int` | **元素偏移量**（不是字节），跳过 N 个**本 IONArray 的 dtype 元素** |

- 字节偏移计算：`byte_offset = offset × self.dtype.itemsize()`
- 子视图是 non-owning 的 IONArray，通过内部引用保持父对象存活
- 越界抛出 `IndexError`
- **未分配时抛 `RuntimeError("IONArray: not allocated, call allocate() first")`**

```python
ion = IONArray(np.dtype('float32'), [3000])

# offset=1000: 跳过 1000 个 float32 (4000 字节)
slot_0 = ion.sub_view(np.dtype('float32'), [1000], offset=0)
slot_1 = ion.sub_view(np.dtype('float32'), [1000], offset=1000)
slot_2 = ion.sub_view(np.dtype('float32'), [1000], offset=2000)

# 子视图可独立绑定给不同 BPU 模型
print(hex(slot_0.phy_addr))  # 父物理地址 + 0
print(hex(slot_1.phy_addr))  # 父物理地址 + 4000

# 父对象删除后子视图仍安全
del ion
arr = slot_0.as_array()  # 安全
```

#### `flush()` → `None`

刷新 CPU cache，确保 ION 内存对 BPU 可见。

- 仅 `cached=True` 且已分配时有效（调用 `hbUCPMemFlush(CLEAN)`）
- `cached=False` 或未分配时为 no-op

```python
ion.as_array()[:] = data
ion.flush()   # CPU 写完后刷新
model.start([], task_id=0)
```

#### `invalidate()` → `None`

使 CPU cache 失效，确保 CPU 看到 ION 内存中 BPU 写入的数据。

- 仅 `cached=True` 且已分配时有效（调用 `hbUCPMemFlush(INVALIDATE)`）
- `cached=False` 或未分配时为 no-op

```python
model.wait(task_id=0)
ion.invalidate()   # BPU 写完后刷新
result = ion.as_array().copy()
```

### 异常

| 触发条件 | 异常类型 | 信息 |
|----------|---------|------|
| ION 内存分配失败 | `RuntimeError` | `IONArray: hbUCPMallocCached failed` |
| 重复调用 `allocate()` | `RuntimeError` | `IONArray::allocate: already allocated` |
| 未分配时调 `as_array()` | `RuntimeError` | `IONArray: not allocated, call allocate() first` |
| 未分配时调 `sub_view()` | `RuntimeError` | `IONArray: not allocated, call allocate() first` |
| sub_view 越界 | `IndexError` | `sub_view: offset N elements (X bytes) + view Y bytes > mem_size Z` |

---

## CauchyKesai (Nash 平台)

BPU 推理引擎。支持 Nash-e / Nash-m / Nash-p 平台。

### 构造函数

#### `CauchyKesai(model_path, n_task=1, model_cnt_select=0)` — 标准模式

加载 BPU 模型，自动分配 `n_task` 组 ION 输入/输出内存。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | `.hbm` 模型文件路径 |
| `n_task` | `int` | `1` | 并发任务槽数量 (1~32) |
| `model_cnt_select` | `int` | `0` | packed 模型中的模型索引 |

- 自动为每个任务槽创建 `IONArray` 输入/输出
- `n_task` 超出范围自动钳位并发出警告
- 构造完成后即可通过 `ion_inputs()` / `ion_outputs()` 访问底层 IONArray

```python
model = CauchyKesai("/path/to/model.hbm", n_task=4)
```

#### `CauchyKesai(model_path, n_task, model_cnt_select, _no_alloc=True)` — 零内存模式

预分配 `n_task` 个任务槽，但**不分配**任何 ION 内存。需在推理前通过 `set_inputs()` / `set_outputs()` 绑定外部 IONArray。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | `str` | — | `.hbm` 模型文件路径 |
| `n_task` | `int` | `1` | 并发任务槽数量 (1~32) |
| `model_cnt_select` | `int` | `0` | packed 模型索引 |
| `_no_alloc` | `bool` | `True` | 固定传 `True` |

```python
model = CauchyKesai("model.hbm", n_task=1, _no_alloc=True)
ion_in = IONArray(np.dtype('uint8'), [1, 640, 640, 1])
ion_out = IONArray(np.dtype('float32'), [1, 80, 80, 80])
model.set_inputs([ion_in], n_task=0)
model.set_outputs([ion_out], n_task=0)
```

### 推理方法

#### `inference(inputs, task_id=0, priority=0)` → `list[numpy.ndarray]`

同步推理：校验 → 提交 → 等待 → 返回结果。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `list[np.ndarray]` | — | 输入数组列表，长度须等于模型输入数量 |
| `task_id` | `int` | `0` | 任务槽索引 (0 ~ n_task-1) |
| `priority` | `int` | `0` | BPU 调度优先级 (0~255) |

- 内部等价于 `start()` + `wait()`
- 自动处理非 C 连续输入（`py::array::ensure`）

```python
outputs = model([y, uv])
# 等价于:
outputs = model.inference([y, uv])
```

#### `__call__(inputs, task_id=0, priority=0)` → `list[numpy.ndarray]`

等价于 `inference()`，支持 `model(inputs)` 调用语法。

#### `start(inputs, task_id=0, priority=0)` → `None`

异步提交推理任务。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inputs` | `list[np.ndarray]` | — | 输入数组列表。传 `[]` 跳过 memcpy |
| `task_id` | `int` | `0` | 任务槽索引 |
| `priority` | `int` | `0` | BPU 调度优先级 |

- 传空列表 `[]` 时跳过 CPU→ION 拷贝，使用预写入的 ION 内存
- GIL 在 `hbUCPSubmitTask` 期间释放，支持多线程并发
- 调用前会检查 `is_busy`，若任务槽被占用抛出 `RuntimeError`

```python
model.start([y, uv], task_id=0)
# 或零拷贝模式:
model.start([], task_id=0)
```

#### `wait(task_id=0)` → `list[numpy.ndarray]`

阻塞等待推理完成，返回零拷贝输出数组。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_id` | `int` | `0` | 任务槽索引 |

- GIL 在 `hbUCPWaitTaskDone` 期间释放
- 返回后自动释放 BPU task handle
- 输出数组指向 ION 内存，是零拷贝视图

```python
results = model.wait(task_id=0)
print(results[0].shape)
```

### 信息查询

#### `s()` → `ModelSummary`

返回模型摘要信息。`ModelSummary` 是 `dict` 子类，支持结构化访问和格式化打印。

**返回 dict 结构：**

| Key | 类型 | 说明 |
|-----|------|------|
| `model_path` | `str` | 模型文件路径 |
| `model_names` | `list[dict]` | 模型名称列表，每项含 `index`, `name`, `selected` |
| `n_task` | `int` | 并发任务数 |
| `memory_mb` | `float` | ION 内存总大小 (MB) |
| `inputs` | `list[dict]` | 输入信息，每项含 `index`, `name`, `dtype`, `shape` |
| `outputs` | `list[dict]` | 输出信息，每项含 `index`, `name`, `dtype`, `shape` |

```python
info = model.s()
print(info)  # 终端格式化输出
shape = info['inputs'][0]['shape']  # 结构化访问
```

#### `t()` → `BenchmarkResult`

使用任务槽 0 执行一次完整推理，返回耗时信息。`BenchmarkResult` 是 `dict` 子类。

**返回 dict 结构：**

| Key | 类型 | 说明 |
|-----|------|------|
| `time_us` | `float` | 微秒 |
| `time_ms` | `float` | 毫秒 |
| `time_s` | `float` | 秒 |
| `time_min` | `float` | 分钟 |

```python
timing = model.t()
print(f"{timing['time_ms']:.2f} ms")
```

#### `is_busy(task_id=0)` → `bool`

查询指定任务槽是否正在推理。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_id` | `int` | `0` | 任务槽索引 |

```python
if not model.is_busy(0):
    model.start(inputs, task_id=0)
```

### ION 内存访问

#### `ion_inputs(task_id=0)` → `list[IONArray]`

获取标准模式自动创建的 IONArray 输入列表。零内存模式下返回空列表。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_id` | `int` | `0` | 任务槽索引 |

```python
ion_in = model.ion_inputs(0)
print(hex(ion_in[0].phy_addr))
print(ion_in[0].dtype)        # np.dtype('uint8')
print(ion_in[0].shape)        # [1, 640, 640, 1]
arr = ion_in[0].as_array()    # 导出为标准 numpy
```

#### `ion_outputs(task_id=0)` → `list[IONArray]`

获取标准模式自动创建的 IONArray 输出列表。零内存模式下返回空列表。参数同 `ion_inputs`。

#### `input_tensors` — 只读属性

类型：`list[list[numpy.ndarray]]`，shape `[n_task][input_count]`。

ION 内存的 numpy 视图，可直接读写。每次 `set_inputs` 或标准构造后更新。

```python
np.copyto(model.input_tensors[0][0], my_data)
```

#### `output_tensors` — 只读属性

类型：`list[list[numpy.ndarray]]`，shape `[n_task][output_count]`。

ION 内存的 numpy 视图，`wait()` 返回后可直接读取。

```python
result = model.output_tensors[0][0]
```

#### `input_names` — 只读属性

类型：`list[str]`。模型输入张量名称列表。

#### `output_names` — 只读属性

类型：`list[str]`。模型输出张量名称列表。

### 零内存模式专用

#### `set_inputs(ion_inputs, n_task=0)` → `None`

为指定任务槽绑定外部 IONArray 输入。仅零内存模式可用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ion_inputs` | `list[IONArray]` | — | IONArray 列表，长度须等于模型输入数量 |
| `n_task` | `int` | `0` | 任务槽索引 |

- 标准模式调用抛出 `RuntimeError`
- 校验 dtype、ndim、shape 是否与模型匹配
- 校验失败抛出 `ValueError`

#### `set_outputs(ion_outputs, n_task=0)` → `None`

为指定任务槽绑定外部 IONArray 输出。仅零内存模式可用。参数和校验逻辑同 `set_inputs`。

### 调度

#### `set_scheduling_params(bpu_cores)` → `None`

设置 BPU 核心调度掩码。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bpu_cores` | `list[int]` | — | 核心索引列表，如 `[0, 1, 2, 3]`。空列表重置为 ANY |

- 核心索引范围 0~3，超出抛出 `ValueError`

```python
model.set_scheduling_params([0, 1, 2, 3])  # 使用全部 4 核
model.set_scheduling_params([])              # 重置为自动选择
```

### 异常汇总

| 触发条件 | 异常类型 | 典型信息 |
|----------|---------|---------|
| task_id 超出范围 | `IndexError` | `task_id out of range: got 5` |
| priority 超出范围 | `IndexError` | `priority out of range: got 300` |
| 输入数量不匹配 | `ValueError` | `input count mismatch: expected 2, got 1` |
| dtype 不匹配 | `ValueError` | `dtype mismatch at input[0]` |
| ndim 不匹配 | `ValueError` | `ndim mismatch at input[0]: expected 4, got 3` |
| shape 不匹配 | `ValueError` | `shape mismatch at input[0]: expected (1,640,640,1), got ...` |
| 任务槽被占用 | `RuntimeError` | `task_id 0 is already in use` |
| 零内存模式下缺少绑定 | `RuntimeError` | `input[0] has no ION memory bound. Call set_inputs() first.` |
| 标准模式调用 set_inputs | `RuntimeError` | `set_inputs() is only available in no-alloc mode` |
| 模型文件不存在 | `RuntimeError` | `Model Path does not exist or is not a file` |
| 模型加载失败 | `RuntimeError` | `hbDNNInitializeFromFiles failed` |

### `__repr__` 自描述

```python
print(model)
# CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=4)
```

---

## Python 辅助类

### ModelSummary

`dict` 子类，由 `CauchyKesai.s()` 返回。支持：

- **结构化访问**: `result['inputs'][0]['shape']`
- **格式化打印**: `print(result)` 在终端输出带颜色的格式化表格

### BenchmarkResult

`dict` 子类，由 `CauchyKesai.t()` 返回。支持：

- **结构化访问**: `result['time_ms']`
- **格式化打印**: `print(result)` 输出耗时信息

---

## 支持的数据类型

NumPy dtype 与 BPU tensor type 对照表：

| NumPy dtype | BPU 类型 | 位宽 | 构造方式 |
|-------------|---------|------|---------|
| `uint8` | U8 | 8-bit | `np.dtype('uint8')` |
| `int8` | S8 | 8-bit | `np.dtype('int8')` |
| `float16` | F16 | 16-bit | `np.dtype('float16')` |
| `int16` | S16 | 16-bit | `np.dtype('int16')` |
| `uint16` | U16 | 16-bit | `np.dtype('uint16')` |
| `float32` | F32 | 32-bit | `np.dtype('float32')` |
| `int32` | S32 | 32-bit | `np.dtype('int32')` |
| `uint32` | U32 | 32-bit | `np.dtype('uint32')` |
| `float64` | F64 | 64-bit | `np.dtype('float64')` |
| `int64` | S64 | 64-bit | `np.dtype('int64')` |
| `uint64` | U64 | 64-bit | `np.dtype('uint64')` |
| `bool` | BOOL8 | 8-bit | `np.dtype('bool')` |
