# IONArray 档：零拷贝 / 内存视图

- [IONArray 档：零拷贝 / 内存视图](#ionarray-档零拷贝--内存视图)
  - [概述](#概述)
  - [快速开始](#快速开始)
  - [IONMemory](#ionmemory)
    - [构造](#构造)
    - [属性](#属性)
    - [flush\_clean](#flush_clean)
    - [flush\_invalidate](#flush_invalidate)
    - [numpy](#numpy)
    - [对应的 UCP 接口](#对应的-ucp-接口)
  - [IONArrayDesc](#ionarraydesc)
    - [构造](#构造-1)
    - [成员](#成员)
    - [计算方法](#计算方法)
    - [完整性质的来源](#完整性质的来源)
  - [IONArray](#ionarray)
    - [构造（双重载）](#构造双重载)
    - [成员与属性](#成员与属性)
    - [allocate](#allocate)
    - [numpy](#numpy-1)
    - [from\_numpy](#from_numpy)
    - [\_\_array\_\_ 协议](#__array__-协议)
    - [flush\_clean / flush\_invalidate](#flush_clean--flush_invalidate)
    - [dequantize](#dequantize)
    - [quantize](#quantize)
    - [from\_memory](#from_memory)
    - [布局详解](#布局详解)
      - [natural layout](#natural-layout)
      - [padded / aligned layout](#padded--aligned-layout)
    - [BPU 数据类型对照](#bpu-数据类型对照)
  - [CauchyKesai 零拷贝接入](#cauchykesai-零拷贝接入)
    - [设计：ION 原子 + Auto 组合](#设计ion-原子--auto-组合)
    - [任务槽状态机](#任务槽状态机)
    - [构造时已分配](#构造时已分配)
    - [input\_descs / output\_descs](#input_descs--output_descs)
    - [inputs / outputs（py::list 绑定，下标纯赋值）](#inputs--outputspylist-绑定下标纯赋值)
    - [check\_input / check\_output](#check_input--check_output)
    - [异步推理 start](#异步推理-start)
    - [wait / wait\_done](#wait--wait_done)
    - [is\_busy](#is_busy)
    - [零拷贝同步 m(task\_id)](#零拷贝同步-mtask_id)
    - [零拷贝推理流程](#零拷贝推理流程)
    - [flush 契约](#flush-契约)
  - [from\_numpy 工厂函数](#from_numpy-工厂函数)
  - [多模型零拷贝链式 cookbook](#多模型零拷贝链式-cookbook)
  - [best-practice：何时用零拷贝](#best-practice何时用零拷贝)
  - [注意事项](#注意事项)

## 概述

本档是 pyCauchyKesai 的**零拷贝 / 内存视图**高级用法。与 `cn_CauchyKesai_Auto.md`（Auto 档，`m([y, uv])` 允许拷贝、numpy in/out）互补：

- **Auto 档**（自动）：直接传 numpy，内部 from_numpy 写入 slot 的 bound IONArray（布局感知 + 自动 flush）。简单，多数场景够用。
- **IONArray 档**（本档）：用户自己管理 ION 物理内存——多模型 pipeline 省中转、大输入跨帧复用 buffer、跨 slot 共享、偏移切段。

两档共用同一组入口，按实参类型派发：传 list 走 Auto，不传走零拷贝。按同步/异步划分：**同步 `__call__`** 的完整介绍见 cn_CauchyKesai_Auto.md；**异步 `start` / `wait` / `wait_done` / `is_busy`**（含其重载）的完整介绍在本档。ION 层是唯一真相，Auto 档只是 ION 原子的组合（详见「设计」）。

三层架构管理 BPU 与 CPU 共享的 ION 物理内存：

```
IONMemory   (一块 ION 物理内存，纯字节)
   ↑  shared_ptr + byte_offset
IONArray   (在这块内存上叠加 tensor 性质：dtype/shape/stride/量化)
   ↑
CauchyKesai   (推理编排，把 IONArray 喂给 hbDNNInferV2)
```

- **IONMemory**：最底层。一块连续的 ION 物理字节 + RAII + cache 维护。不知道 dtype/shape。
- **IONArrayDesc**：纯张量描述（无内存）。持有 dtype/shape/stride/量化参数/alignedByteSize 等。
- **IONArray**：持有 IONArrayDesc + IONMemory（组合，has-a），是 BPU 推理的内存载体，导出零拷贝 numpy 视图。多个 IONArray 可共享同一块物理内存（from_memory 切偏移段）。

源码：`csrc/nash/ion_memory.cpp`、`csrc/nash/ion_array.cpp`、`csrc/nash/ion_array_desc.cpp`，绑定 `csrc/nash/bindings.cpp`。

---

## 快速开始

```python
import numpy as np
from pyCauchyKesai import CauchyKesai, IONArray

m = CauchyKesai("/app/model/yolov8n.hbm")

# 零拷贝输入：按模板 desc 建 IONArray，就地写 buffer，下标绑定（纯赋值，无校验）
ion_in = IONArray(m.input_descs[0])
ion_in.numpy()[:] = my_data        # 就地写 ION buffer（零拷贝视图）
ion_in.flush_clean()               # CPU 写后回写，BPU 读前
m.inputs[0][0] = ion_in            # 绑到 slot 0 的 input[0]（纯赋值，无校验）

m.start(task_id=0)                 # 零拷贝提交（不传 inputs，用 slot 0 已绑定的 IONArray）
m.wait_done(task_id=0)             # 纯等完成（不返回；输出已 flush_invalidate）

# 取输出 IONArray（含物理地址，可零拷喂下游模型）
a_out = m.outputs[0][0]
```

---

## IONMemory

hbUCPSysMem 的超薄 RAII 包装。多数用户用不到——直接用 `from_numpy` / `IONArray` 即可。

### 构造

```python
def __init__(self, size: int, cached: bool = True) -> None:
```

分配指定字节数的 ION 物理内存。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| size | int | 是 | - | 分配字节数 |
| cached | bool | 否 | True | True 分配 cacheable 内存 hbUCPMallocCached，CPU 读写经过 cache；False 分配 uncacheable 内存 hbUCPMalloc，CPU 读写不经过 cache |

| 异常 | 说明 |
|------|------|
| RuntimeError | 分配失败 |


释放靠 RAII：引用计数归零时自动 hbUCPFree，无显式 free。物理地址、fd 等下沉内部，Python 拿不到。

### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| size | int | ION 分配字节数 |
| is_allocated | bool | 是否已分配 |

### flush_clean

```python
def flush_clean(self) -> None:
```

无参数。回写脏行（CLEAN），让 BPU 读到 CPU 写的最新值。仅 cached 且已分配时生效，否则静默 no-op。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 原地刷 cache；完整语义见「对应的 UCP 接口」 |

### flush_invalidate

```python
def flush_invalidate(self) -> None:
```

无参数。回写脏行并使缓存失效（INVALIDATE），强制 CPU 从 DDR 重读 BPU 写入的数据。仅 cached 且已分配时生效，否则静默 no-op。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 原地刷 cache；完整语义见「对应的 UCP 接口」 |

### numpy

```python
def numpy(self) -> np.ndarray:
```

无参数。返回固定 uint8 一维 raw bytes 视图（零拷贝），不接受 dtype——tensor 语义归 IONArray。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| arr | np.ndarray | shape=(size,) dtype=uint8，数据直指 ION 虚拟地址；通过 capsule 持有 shared_ptr 防内存提前释放 |

| 异常 | 说明 |
|------|------|
| RuntimeError | 未分配 |

```python
mem = IONMemory(1024)
raw = mem.numpy()   # shape=(1024,), dtype=uint8
raw[:] = 0
```

### 对应的 UCP 接口

IONMemory 只是 hbUCP\* 系列函数的直通，真正的 ION/dma-heap ioctl + mmap 全在闭源 libhbucp.so 里，pyCauchyKesai 只触达这一跳：

| Python | UCP 接口 | 时机 |
|--------|----------|------|
| IONMemory(size, cached=True/False) | hbUCPMallocCached / hbUCPMalloc | 分配 |
| 析构（RAII） | hbUCPFree | 引用计数归零 |
| flush_clean() | hbUCPMemFlush(..., HB_SYS_MEM_CACHE_CLEAN) | CPU 写后、BPU 读前 |
| flush_invalidate() | hbUCPMemFlush(..., HB_SYS_MEM_CACHE_INVALIDATE) | BPU 写后、CPU 读前 |

两个 flush 底层是同一个 hbUCPMemFlush(mem, flag)，靠 flag 区分。两个 flag 语义与直觉相反：

| flag（值） | 动作 | 语义 |
|-----------|------|------|
| HB_SYS_MEM_CACHE_CLEAN (=2) | Clean | 回写脏行，保留缓存副本。CPU 写完后调，让 BPU 读到最新值 |
| HB_SYS_MEM_CACHE_INVALIDATE (=1) | Clean + Invalidate | 回写脏行并使缓存失效。BPU 写完后、CPU 读前调，强制 CPU 从 DDR 重读 |

典型时序：

```
CPU 写 → flush_clean（回写，BPU 读前）→ BPU 读
BPU 写 → flush_invalidate（回写+失效，CPU 读前）→ CPU 读
```

> [!NOTE]
> 多数场景不用手动调：`from_numpy` / `quantize` 内部已自动 flush_clean；`CauchyKesai.wait()` 对输出自动 flush_invalidate。只有直接操作底层 IONMemory（如裸 `numpy()` 读写）时才需手动维护。

cacheable 内存（默认）需配合 flush 维护 CPU 与 BPU 间的数据一致性——BPU 等后端硬件与主存间无 cache，仅 CPU 侧有 cache。

---

## IONArrayDesc

纯张量描述（无内存）。持有 tensor 的全部性质：dtype / shape / stride / 量化参数 / alignedByteSize 等，但**不持有任何 ION 物理内存**。IONArray 持有它作为 desc 成员（组合关系，has-a，不继承）。

把"张量描述"从"内存载体"里独立出来，是为了让性质可以脱离内存单独传递、复用、比较——构造多个 IONArray 共享同一份描述（多模型串联、内存池复用），或直接喂给 `IONArray(desc)` / `from_memory(mem, off, desc)` 规划偏移。

### 构造

```python
def __init__(self, dtype: np.dtype, shape: list[int]) -> None:
```

基本构造：只给 dtype + shape，其余性质安全默认（stride 空、无量化、aligned_byte_size=-1、tensor_type=-1）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| dtype | np.dtype | 是 | - | 元素类型，如 np.dtype('float32') |
| shape | list[int] | 是 | - | 逻辑形状，所有维度必须 > 0 |

| 异常 | 说明 |
|------|------|
| ValueError | 任一 shape 维度 ≤ 0 |

```python
d = IONArrayDesc(np.dtype('float32'), [1, 3, 224, 224])
```

### 成员

成员 public 直接访问（无 getter）。在 IONArray 上通过 `ion.desc.X` 访问（`isinstance(ion, IONArrayDesc)` 为 False）：

| 成员 | 类型 | 说明 |
|------|------|------|
| dtype | np.dtype | 元素类型 |
| shape | list[int] | 逻辑形状 |
| stride | list[int] | 各维字节步长（含对齐 padding；空=natural 连续布局） |
| name | str | 张量名（hbDNNGetInputName/OutputName；裸构造为空） |
| desc | str | hbDNN 描述文本（hbDNNGetInputDesc/OutputDesc；裸构造为 N/A） |
| tensor_type | int | hbDNNDataType 枚举值（裸构造 -1） |
| quanti_type | int | 量化类型（NONE=0 / SCALE=1） |
| quantize_axis | int | 量化轴 |
| scale | list[float] | 反量化 scale |
| zero_point | list[int] | 零点 |
| aligned_byte_size | int | 含 BPU padding 的对齐字节大小（裸构造 -1） |

> [!NOTE]
> 成员都是标准类型（str/int/list/np.dtype），`==` 直接复用标准类型的相等：`ion.desc.dtype == other.dtype`、`ion.desc.shape == other.shape`、`ion.desc.name == other.name` 等。无整体 `__eq__`。

### 计算方法

| 方法 | 返回 | 说明 |
|------|------|------|
| ndim() | int | 维度数 len(shape) |
| size() | int | 总元素数 prod(shape) |
| nbytes() | int | 理论字节数 dtype.itemsize × prod(shape) |
| has_stride() | bool | stride 是否非空 |
| is_padded_layout() | bool | 是否 padded（对齐）布局：stride[0] != ∏内维 × itemsize |
| has_tensor_properties() | bool | tensor_type >= 0（是否带模型性质） |
| is_quantized() | bool | quanti_type == SCALE |

### 完整性质的来源

完整性质（含 stride / 量化 / aligned_byte_size）来自模型：`model.input_descs[i]` / `output_descs[i]` 是带完整性质的模板 IONArrayDesc。取它喂给 `IONArray(desc)` 或 `from_memory(mem, off, desc)` 做偏移规划。

---

## IONArray

ION 物理内存载体。**持有 IONArrayDesc（desc，张量描述）+ IONMemory（物理内存）**，组合关系（has-a，不继承）。desc 是直接成员：`ion.desc` 直接访问，dtype/shape 等走 `ion.desc.dtype`（不转发到 IONArray）。cache 操作（flush_clean / flush_invalidate）转发到底层 IONMemory。

### 构造（双重载）

```python
def __init__(self, dtype: np.dtype, shape: list[int], cached: bool = True, defer: bool = False) -> None:  # 重载1：裸构造
def __init__(self, desc: IONArrayDesc, cached: bool = True, defer: bool = False) -> None:                # 重载2：从描述构造
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| dtype / desc | np.dtype / IONArrayDesc | 是 | - | 重载1 给 dtype+shape（性质残缺）；重载2 给 IONArrayDesc（性质齐全，如来自 model.input_descs[i]） |
| shape | list[int] | 重载1 | - | 逻辑形状，所有维度必须 > 0 |
| cached | bool | 否 | True | True → hbUCPMallocCached；False → hbUCPMalloc |
| defer | bool | 否 | False | True 只记录性质不分配内存，稍后 allocate()；False 立即分配 |

| 异常 | 说明 |
|------|------|
| ValueError | 任一 shape 维度 ≤ 0 |
| RuntimeError | defer=False 时分配失败 |

两种最简写法：

```python
ion = IONArray(np.dtype('float32'), [1, 3, 224, 224])   # 重载1：立即分配 cacheable 内存
src = m.input_descs[0]                                  # IONArrayDesc（完整性质）
ion2 = IONArray(src)                                    # 重载2：从 src 的 desc 构造
```

如需 BPU 对齐大小的 IONArray，请用 `IONArray(model.input_descs[i])` / `IONArray(model.output_descs[i])` 从模型模板 desc 创建（再可选 `from_memory` 规划偏移）。

### 成员与属性

public 成员（直接访问，无 getter）：

| 成员 | 类型 | 说明 |
|------|------|------|
| desc | IONArrayDesc | 张量描述（dtype/shape/stride/量化等）。dtype/shape 等走 ion.desc.dtype，IONArray 不转发 |
| memory | IONMemory | 底层 ION 物理内存（shared_ptr 共享所有权）。多个 IONArray 可共享同一 IONMemory（from_memory 切偏移段），最后一份析构时 hbUCPFree |
| byte_offset | int | 在 IONMemory 内的字节偏移 |

| 属性 | 类型 | 说明 |
|------|------|------|
| is_allocated | bool | 是否已分配 ION 内存（memory 非空且已分配） |


### allocate

```python
def allocate(self, cached: bool = True) -> None:
```

懒分配：为 defer=True 构造的 IONArray 分配 ION 物理内存。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| cached | bool | 否 | True | True → hbUCPMallocCached；False → hbUCPMalloc |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值；分配后 is_allocated=True |

| 异常 | 说明 |
|------|------|
| RuntimeError | 已分配（重复调用）；底层分配失败 |

分配大小优先用 `desc.aligned_byte_size`（来自模型 properties，含 BPU padding）；未设置则用 `dtype.itemsize × prod(shape)`。

```python
ion = IONArray(np.dtype('float32'), [1, 3, 640, 640], defer=True)
print(ion.is_allocated)  # False
ion.allocate()
print(ion.is_allocated)  # True
```

### numpy

```python
def numpy(self) -> np.ndarray:
```

无参数。返回标准 numpy 零拷贝视图，数据直指 ION 虚拟地址。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| arr | np.ndarray | 布局感知：有 stride 返回带 strides 的 strided 视图，否则返回 natural（连续）视图；通过 capsule 持有 shared_ptr 保活，多次调用返回独立对象但指向同一块 ION 内存 |

| 异常 | 说明 |
|------|------|
| RuntimeError | 未分配 |
| IndexError | 视图越界（byte_offset + 占用 > memory.size）——offset 规划错时抛，防读到邻居段 |

```python
arr = ion.numpy()
arr[:] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

# 生命周期：即使 ion 被 del，只要 arr 还在，ION 内存就不会释放
del ion
print(arr[0, 0])  # 1.0 — 仍然安全
```

### from_numpy

```python
def from_numpy(self, arr: np.ndarray) -> None:
```

布局感知写入：把连续 numpy 数组写入 ION buffer。这是写 buffer 的统一入口。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| arr | np.ndarray | 是 | - | 待写入的连续 numpy 数组 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值。natural layout → 连续 memcpy；padded/aligned → 按 stride 散写；写后自动 flush_clean |

| 异常 | 说明 |
|------|------|
| RuntimeError | 未分配；输入过大（input too large for buffer） |
| ValueError | dtype/shape 不匹配 |
| IndexError | strided 写越界（strided write out of buffer bounds） |

```python
ion = IONArray(model.input_descs[0])
ion.from_numpy(my_numpy)  # 自动按正确布局写入，无需关心 stride；自动 flush
```

### \_\_array\_\_ 协议

```python
def __array__(self, dtype=None) -> np.ndarray:
```

IONArray 实现了 numpy 的 `__array__` 协议，无需显式 `.numpy()` 即可被 numpy 直接接受。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| arr | np.ndarray | dtype=None 或同 dtype → 零拷贝视图（委托 numpy，capsule 保活）；异 dtype → numpy 侧拷贝转换（脱离原物理内存） |

```python
np.asarray(ion)                # 等价于 ion.numpy()，零拷贝视图
np.asarray(ion, dtype=...)     # 异 dtype → numpy 侧拷贝转换
np.sum(ion)                    # 任何接受 array-like 的 API 直通
```

### flush_clean / flush_invalidate

```python
def flush_clean(self) -> None:           # 转发底层 IONMemory
def flush_invalidate(self) -> None:
```

转发到底层 IONMemory（**整块刷**，cache line 可能跨子视图边界）。语义、flag 对照、典型时序见 IONMemory「对应的 UCP 接口」。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值；原地刷 cache（仅 cached 且已分配时生效，否则静默 no-op） |

### dequantize

```python
def dequantize(self) -> np.ndarray:
```

无参数。读侧反量化：`float = scale × (int − zero_point)`，沿 quantize_axis 广播。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| arr | np.ndarray | 随 quanti_type 变（见下表）：NONE→passthrough（原 dtype 视图）；SCALE→float32 新数组 |

| quanti_type | zero_point | 公式 | 行为 |
|-------------|-----------|------|------|
| NONE | - | - | passthrough，直接返回 numpy()（保留原 dtype） |
| SCALE | 空 (len=0) | f = x × scale[i] | 返回 float32 新数组 |
| SCALE | 有 (len>0) | f = (x - zero_point[i]) × scale[i] | 返回 float32 新数组（copy） |

| 异常 | 说明 |
|------|------|
| RuntimeError | 未分配 |

```python
result_float = ion_out.dequantize()  # int8/uint8 → float32
```

### quantize

```python
def quantize(self, float_arr: np.ndarray) -> None:
```

写侧量化：`int = round(float / scale) + zero_point`，clip 到 dtype 范围，直接写入 ION buffer。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| float_arr | np.ndarray | 是 | - | 待量化的 float 数组 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值；量化结果直接写入 ION buffer（随 quanti_type，见下表），写后自动 flush_clean |

| quanti_type | zero_point | 公式 | 行为 |
|-------------|-----------|------|------|
| NONE | - | - | memcpy 逐字节拷贝 |
| SCALE | 空 | int = round(x / scale[i]) | 无零点 |
| SCALE | 有 | int = round(x / scale[i]) + zero_point[i] | 有零点 |

舍入 FE_TONEAREST（最近整数），clip 到 dtype 范围（U8→[0,255]，S8→[-128,127]）。写后自动 flush_clean。

| 异常 | 说明 |
|------|------|
| RuntimeError | 未分配；NONE 路径输入超 buffer |
| ValueError | SCALE 路径 ndim/shape 不匹配 |

```python
ion_in.quantize(float_arr)  # float32 → int8/uint8
```

### from_memory

```python
@staticmethod
def from_memory(memory: IONMemory, byte_offset: int, template: IONArrayDesc) -> IONArray:
```

静态方法。从描述继承 tensor 性质 + 共享指定 IONMemory 的偏移处。用于多模型零拷贝串联 / 大块内存偏移规划。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| memory | IONMemory | 是 | - | 要共享的 IONMemory（shared_ptr 共享所有权） |
| byte_offset | int | 是 | - | 在 IONMemory 中的字节偏移 |
| template | IONArrayDesc | 是 | - | 描述，从中继承 tensor 性质（dtype/shape/stride/量化参数等） |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| ion_array | IONArray | 共享 memory 的物理内存（shared_ptr 共享所有权），tensor 性质从 template 复制 |

| 异常 | 说明 |
|------|------|
| ValueError | memory 为 None |
| IndexError | byte_offset > memory.size（越界） |

```
┌─────────── 一块 IONMemory (8192 B) ───────────┐
│  IONArray A (offset 0)   │  IONArray B (offset 4096) │
│  → 喂模型1               │  → 喂模型2               │
└─────────────────────────────────────────────────┘
```

```python
up_output = model_a.outputs[0][0]                  # 上游输出 IONArray
tpl = model_b.input_descs[0]                       # 下游输入模板 desc
down_input = IONArray.from_memory(up_output.memory, 0, tpl)
model_b.inputs[0][0] = down_input
```

> [!IMPORTANT]
> byte_offset 必须**对齐到 BPU 对齐步长**（S600=64 字节，S100/S100P=32）——喂 BPU 的 buffer 起址不对齐会出错。`m.inputs[slot][idx]=ion` 是纯赋值不校验；预检用 `check_input`（含对齐），不对齐返回 False、同步 `m(task_id)` 抛。

### 布局详解

#### natural layout

- stride 为空（`has_stride()` 为 False）
- 数据在内存中连续排列
- 可直接 memcpy，from_numpy 路径走 memcpy
- 裸构造默认 natural layout

#### padded / aligned layout

- stride 非空，且 stride[0] != ∏内维 × itemsize
- buffer 中有 BPU 对齐 padding（S600 对齐 64 字节，S100/S100P 为 32）
- 必须按 stride 读写，直接 memcpy 会错位
- from_numpy 路径走 stride 散写

| 操作 | natural | padded |
|------|---------|--------|
| numpy() | 连续 numpy 视图 | 带 strides 的 numpy 视图 |
| from_numpy() | memcpy | 按 stride 散写 |
| summary() 的 alignedByteSize | nbytes | aligned_byte_size（含 padding） |

### BPU 数据类型对照

NumPy dtype 与 BPU tensor type 对照：

| NumPy dtype | BPU 类型 | 位宽 |
|-------------|---------|------|
| uint8 | U8 | 8-bit |
| int8 | S8 | 8-bit |
| float16 | F16 | 16-bit |
| int16 | S16 | 16-bit |
| uint16 | U16 | 16-bit |
| float32 | F32 | 32-bit |
| int32 | S32 | 32-bit |
| uint32 | U32 | 32-bit |
| float64 | F64 | 64-bit |
| int64 | S64 | 64-bit |
| uint64 | U64 | 64-bit |
| bool | BOOL8 | 8-bit |

---

## CauchyKesai 零拷贝接入

CauchyKesai（Auto 档见 `cn_CauchyKesai_Auto.md`）的零拷贝接入点：用 IONArray 替代内部默认 buffer，经 `m.inputs[slot][idx]=ion` 替换 slot 的绑定。

### 设计：ION 原子 + Auto 组合

**ION 层 = 原子能力**（零拷贝为线索，向外展示）。**Auto 档 = ION 原子的组合**（不重写路径，调 ION 原子）。Auto 不是独立 memcpy 路径，而是用 ION 原子拼出来的语法糖——`m([y,uv])` 内部就是 `from_numpy` 写 slot 的 bound IONArray → 走零拷贝 start。ION 层是唯一真相。

| ION 原子（零拷贝能力） | 说明 |
|---|---|
| `input_descs` / `output_descs` | 模板 IONArrayDesc 列表（模型固有性质，只读） |
| `inputs` / `outputs` | slot 当前绑定的 IONArray（py::list，下标纯赋值，无校验） |
| `check_input` / `check_output` | ion 能否绑到 input/output[idx]（properties_match + BPU 对齐），返回 bool |
| `start(task_id=0)` | 零拷贝提交（不传 inputs，用 bound） |
| `wait_done(task_id, timeout_ms)` | 纯等完成，不返回 |

| Auto 组合（调 ION 原子） | 说明 |
|---|---|
| `start(inputs, task_id=0)` | `bound[i]->from_numpy(inputs[i])` 后走零拷贝 start |
| `wait(task_id, timeout_ms)` | `wait_done` + 导出 list[ndarray] |
| `m(inputs, task_id)` / `m(task_id=0)` | 同步：check → start → wait → 导出 |

**校验策略**：异步不校验（`start` / `wait_done` / `wait` 不校验 bound，要快/浅，用户改错自己负责）；同步才校验（`__call__` / `inference` 调 check，不匹配抛）。

**bound 是 py::list（非校验代理）**：`m.inputs` / `m.outputs` 是 C++ 的 `py::list` 成员（`list[list[IONArray]]`，shape `[n_task][input_count]`），经 `def_readwrite` 暴露。Python list 原生 `__setitem__`，`m.inputs[slot][idx] = ion` 是**纯赋值，无校验，浅/快**——直接落回 C++ 成员。start 时从 py::list cast 出 shared_ptr 拷到局部 vector 保活到 wait_done（防用户 wait 前改 bound 换掉 ion）。

### 任务槽状态机

每个 task slot 有三态，`m.inputs[slot][idx]=ion` 改绑建议在 slot idle 时做（异步路径不校验，改 running slot 的 bound 不立即生效——start 已把 ion 拷到局部 vector），start 不能对同一 slot 重入：

| 状态 | 值 | 含义 | 可执行 | 不可执行 |
|------|-----|------|--------|----------|
| idle | 0 | 空闲 | start, inputs[=]/outputs[=] 改绑 | wait |
| infering | 1 | BPU 执行中 | is_busy 检查 | start（同 slot 不能重入） |
| waiting | 2 | 推理完成待取结果 | wait / wait_done | start |

转换：idle → start → infering → wait → waiting → finish → idle。

### 构造时已分配

构造 `CauchyKesai("model.hbm")` 后，所有 slot 的 input/output IONArray 已按 desc 建好并分配（is_allocated=True），可直接推理，无需手动分配。每个 slot 的 bound IONArray 性质完善（含 stride/aligned_byte_size）。

### input\_descs / output\_descs

```python
@property
def input_descs(self) -> list[IONArrayDesc]: ...
@property
def output_descs(self) -> list[IONArrayDesc]: ...
```

input/output 模板 IONArrayDesc 列表（模型固有性质，只读）。用于查模板性质、构造匹配的 IONArray、from_memory 切段。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| descs | list[IONArrayDesc] | 每个输入/输出一个 IONArrayDesc（dtype/shape/stride/量化/aligned_byte_size/name 等） |

```python
tpl = m.input_descs[0]              # input[0] 模板 desc
ion = IONArray(tpl)                 # 按模板建 IONArray（立即分配，性质齐全，可直接喂 BPU）
```

> [!NOTE]
> 旧版的 `make_input(i)` / `make_output(i)` 工厂已删——`IONArray(m.input_descs[i])` 完全覆盖（desc 自带完整性质，构造即分配）。要延迟分配用 `IONArray(tpl, defer=True)` + `allocate()`。

### inputs / outputs（py::list 绑定，下标纯赋值）

```python
inputs: list[list[IONArray]]     # m.inputs[slot][idx]，def_readwrite，纯赋值无校验
outputs: list[list[IONArray]]    # m.outputs[slot][idx]，同上
```

slot 当前绑定的 IONArray（`py::list` 成员，def_readwrite 暴露）。input/output 完全对称。`m.inputs[slot][idx] = ion` 是 Python list 原生 `__setitem__`——**纯赋值，无校验，浅/快**，直接落回 C++ 成员（bound 唯一真相）。

| 用法 | 说明 |
|------|------|
| `m.inputs[slot][idx] = ion` | 写：纯赋值绑定 ion 到 slot 的 input[idx]（无校验） |
| `m.inputs[slot][idx]` | 读：返回当前绑定的 IONArray（同一 C++ 对象，`is` 成立） |
| `m.outputs[slot][idx] = ion` | 输出侧写 |
| `m.outputs[slot][idx]` | 输出侧读 |
| `len(m.inputs)` | n_task（slot 数） |
| `len(m.inputs[slot])` | input_count（该 slot 的输入数） |

```python
m.inputs[0][0] = ion_in        # 绑定（纯赋值，无校验）
ion = m.inputs[0][0]           # 读取（返回同一 C++ IONArray 对象）
```

> [!IMPORTANT]
> **无校验 = 自己负责**。`m.inputs[slot][idx]=ion` 不查 dtype/shape/对齐。绑错（如 dtype 不符、byte_offset 不对齐 BPU）异步 start 不拦，到 BPU 才出错。要预检用 `check_input`/`check_output`；要同步带校验用 `m(task_id=0)`。也可塞非 IONArray（如 `m.inputs[0][0]=123`），start cast 时抛 TypeError——符合「异步不校验、改错自负」。

> [!NOTE]
> bound 是 `py::list` 而非 `std::vector`：pybind11 对嵌套 `vector<vector<shared_ptr<IONArray>>>` 的 def_readwrite，元素级赋值会写副本不落成员；py::list 引用语义让下标赋值原生生效。

### check\_input / check\_output

```python
def check_input(self, ion: IONArray, idx: int) -> bool:
def check_output(self, ion: IONArray, idx: int) -> bool:
```

预检 ion 能否绑到 input[idx]/output[idx]。判定 = properties_match + BPU byte_offset 对齐，返回 bool 不抛——绑定前的安全检查。check 返回 True ⟺ 同步路径 `m(task_id)` 校验通过。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| ion | IONArray | 是 | - | 待检查的 IONArray |
| idx | int | 是 | - | 输入/输出索引 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| ok | bool | True = 可绑定（同步校验必过）；False = 不可（性质不匹配 / 内存不够 / 不对齐 / 越界 / ion 为 None） |

> [!NOTE]
> check 的语义是「能否绑定」（properties_match + 对齐），与同步 `m(task_id)` 的校验同源，所以 check 返回 True ⟺ 同步路径不抛。它**不是**描述结构等价（那是 IONArrayDesc 成员级 `==` 的事，判 dtype/shape/stride 是否描述同一类张量，不含内存大小）。

### 异步推理 start

```python
def start(self, inputs: list[np.ndarray], task_id: int = 0) -> None:   # Auto memcpy
def start(self, task_id: int = 0) -> None:                             # 零拷贝
```

异步提交推理任务。对应 `hbUCPSubmitTask`（非阻塞提交到优先级队列）。按实参类型派发：传 list 走 Auto（from_numpy 写 bound IONArray），不传走零拷贝（用 bound）。**异步不校验 bound**（仅校验 task_id 范围与 Auto 档的输入数量）；priority 为 per-slot，由 set_scheduling_params 设置（见 cn_CauchyKesai_Auto.md「BPU 核调度」）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| inputs | list[np.ndarray] | Auto 档必填 | - | Auto 档：输入数组列表，dtype/shape 须与模型一致；零拷贝档不传 |
| task_id | int | 否 | 0 | 任务槽索引 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值。异步提交，任务进优先级队列即返回；结果用 wait(task_id) 或 wait_done(task_id) 取 |

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界 |
| ValueError | Auto 档输入数量不匹配；from_numpy 的 dtype/shape 不匹配 |
| RuntimeError | 任务槽被占用（task_id N is already in use）；输入数据超 ION 缓冲区 |

```python
m.start([y, uv], task_id=0)   # Auto 异步：from_numpy 写 slot 0 的 bound IONArray 后提交
m.start(task_id=0)            # 零拷贝异步：用 m.inputs[0] 已绑定的 IONArray 提交
```

### wait / wait\_done

```python
def wait(self, task_id: int = 0, timeout_ms: int = 0) -> list[np.ndarray]:   # 等完成 + 导出
def wait_done(self, task_id: int = 0, timeout_ms: int = 0) -> None:         # 纯等完成，不返回
```

`wait`：阻塞等待推理完成，返回输出数组（wait_done + 导出 list[ndarray]）。`wait_done`：纯等完成（flush_invalidate 输出 + release task），不返回——零拷贝档用户想自己经 `m.outputs[task_id]` 取 IONArray 时用。两者都对应 `hbUCPWaitTaskDone`（阻塞等结果）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| task_id | int | 否 | 0 | 任务槽索引 |
| timeout_ms | int | 否 | 0 | 时间预算（毫秒），语义详见下方说明 |

> [!IMPORTANT]
> **timeout_ms 不是"最大等待时间"**。无论取何值，wait/wait_done 都阻塞到任务真正完成、**从不提前返回**。它只作为时间预算——任务跑完后，BPU 拿实际执行耗时与 timeout_ms 比较（基准为 BPU 内部任务耗时，略小于 Python 侧墙钟）：超出则返回 `HB_UCP_TASK_TIMEOUT`（-200002，本包抹成 RuntimeError），未超出则正常返回结果。`timeout_ms ≤ 0` = 不设预算、无限等待。

> [!NOTE]
> **默认值 0 对齐 UCP 习惯**。`hbUCPWaitTaskDone` 的 timeout 是必填形参、SDK 头文件无默认值，但官方样例（quick_start / ai_benchmark 等）清一色传 `0`，意为"阻塞到完成、永不超时"，本包沿用同一默认。若想给慢推理兜底（实际耗时超 timeout_ms 时抛 RuntimeError），显式传一个正数即可——但注意它**不会提前 bailout**，也不防任务卡死（卡死任务无论 timeout 多少都死等）。

| 返回值 | 类型 | 说明 |
|--------|------|------|
| wait → outputs | list[np.ndarray] | 该槽推理输出数组列表，顺序与模型输出张量一致 |
| wait_done → - | None | 无返回；输出已 flush_invalidate，可经 m.outputs[task_id] 取 IONArray |

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界 |
| RuntimeError | 槽位空闲（task_id N is not in use）；已被等待（task_id N is already being waited）；任务实际耗时超过 timeout_ms（HB_UCP_TASK_TIMEOUT）或底层失败 |

### is\_busy

```python
def is_busy(self, task_id: int = 0) -> bool:
```

查询指定任务槽是否正在推理（infering 或 waiting 都算 busy）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| task_id | int | 否 | 0 | 任务槽索引 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| busy | bool | True=该槽正在推理（infering 或 waiting）；False=空闲 |

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界 |

### 零拷贝同步 m(task_id)

```python
def __call__(self, inputs: list[np.ndarray], task_id: int = 0) -> list[np.ndarray]:   # Auto 同步
def __call__(self, *, task_id: int = 0) -> list[np.ndarray]:                          # 零拷贝同步
```

零拷贝同步 `m(task_id=0)`：校验 bound（check_input/output，不匹配抛）→ 零拷贝 start → wait → 导出 list[ndarray]。同步 `__call__` 的完整签名与两档派发语义见 cn_CauchyKesai_Auto.md，本节聚焦零拷贝档用法。用于用户提前 `m.inputs[slot][idx]=ion` 绑好 IONArray（并写好 buffer + flush_clean）后一行同步推理。两档都导出 list[ndarray]。

```python
ion = IONArray(m.input_descs[0]); ion.numpy()[:] = data; ion.flush_clean()
m.inputs[0][0] = ion
out = m(task_id=0)            # 零拷贝同步：校验 bound + start + wait + 导出
```

### 零拷贝推理流程

CauchyKesai 推理不做透明量化——量化/反量化由 `IONArray.quantize()` / `dequantize()` 负责。输入数据写入 ION buffer 有两条路径：

**Auto 路径**（简单）：直接把 numpy 数组传给 start，内部走 from_numpy 布局感知写入并自动 flush。

```python
m.start([y, uv], task_id=0)
outputs = m.wait(task_id=0)
```

**零拷贝路径**（自定义 buffer，本档）：按模板 desc 建 IONArray，写 buffer 后显式 flush_clean，再绑定、`start(task_id)` 提交。

```python
ion_in = IONArray(m.input_descs[0])
ion_in.numpy()[:] = my_data
ion_in.flush_clean()
m.inputs[0][0] = ion_in          # 纯赋值绑定（无校验）
m.start(task_id=0)               # 零拷贝提交（不传 inputs）
m.wait_done(task_id=0)           # 纯等完成；或 outs = m.wait(task_id=0)
```

### flush 契约

start() 本身不做输入 flush，由两条路径各自负责：

- Auto 路径（传 inputs）：内部走 from_numpy，自动 flush_clean。
- 零拷贝路径（`start(task_id)` 配合 `m.inputs[slot][idx]=ion`）：用户写 buffer 后，必须显式 `ion.flush_clean()` 再 start。

输出侧由 wait/wait_done 对输出自动 flush_invalidate，无需用户处理。零拷贝档用 `m.outputs[task_id]` 取 IONArray 喂下游时，wait_done 已 invalidate，可直接共享物理内存。

---

## from\_numpy 工厂函数

```python
def from_numpy(arr: np.ndarray, cached: bool = True) -> IONArray:
```

模块级工厂函数（`from pyCauchyKesai import from_numpy`），从 numpy 数组创建 IONArray（分配 + 写入 + flush 一步到位）。与实例方法 `ion.from_numpy(arr)` 同名不冲突——前者造**新** IONArray，后者把数据写入**已有** IONArray；模块级工厂内部即 `IONArray(dtype, shape)` + `ion.from_numpy(arr)`。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| arr | np.ndarray | 是 | - | 输入数组（非连续时自动转为连续） |
| cached | bool | 否 | True | True → hbUCPMallocCached；False → hbUCPMalloc |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| ion | IONArray | dtype/shape 与输入一致，数据已写入并 flush；异常透传 IONArray 构造 + from_numpy |

```python
ion = from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
```

流程：

1. np.ascontiguousarray(arr) → 确保内存连续
2. IONArray(dtype, shape, cached) → 构造 + 立即分配
3. ion.from_numpy(arr) → 拷入数据 + 自动 flush
4. 返回已填充的 IONArray

---

## 多模型零拷贝链式 cookbook

模型 A 的输出直接喂模型 B 的输入，省一次 CPU 中转。核心：把 A 的输出 IONArray 共享给 B 的输入（同一物理内存）。

```python
# A、B 是两个 CauchyKesai 实例（可不同模型）
a_in = IONArray(A.input_descs[0])
a_in.numpy()[:] = data
A.inputs[0][0] = a_in            # 绑定（纯赋值，无校验）
A.start(task_id=0); A.wait_done(task_id=0)   # 推理，输出写入 A 的 output IONArray

# 取 A 的输出 IONArray（用 m.outputs，不是 wait 返回的 ndarray！）
a_out = A.outputs[0][0]          # A 的 output[0] IONArray（BPU 写入的，含物理地址）

# 预检能否零拷贝喂 B
if not B.check_input(a_out, 0):
    # 性质不匹配（如 A 输出 int8、B 要 float）→ 需 dequant + from_numpy，破零拷贝
    b_in = IONArray(B.input_descs[0])
    b_in.from_numpy(a_out.dequantize())     # 转 float 再写
    B.inputs[0][0] = b_in
else:
    B.inputs[0][0] = a_out        # 零拷贝：A 输出 IONArray 直接当 B 输入（同物理内存）
B.start(task_id=0); out = B.wait(task_id=0)
```

要点：

- **取上游输出用 `m.outputs[slot][idx]`** 拿 IONArray。**不要用 `wait()` 返回的 ndarray**——它丢掉了 IONArray 包装（物理地址），喂不下去。
- **链式衔接无需额外 flush**：wait/wait_done 对输出已自动 flush_invalidate；下游 `inputs[=]` 用同一物理内存。
- **性质不匹配才破零拷贝**：dtype/shape/量化不同需 dequant/转换（物理限制）。
- 多输出/多输入：逐个 `B.inputs[0][i] = a_outs[i]`（纯赋值）。

---

## best-practice：何时用零拷贝

- **用零拷贝**（IONArray 档）：多模型 pipeline（省每次中转）、大输入跨帧复用 buffer、跨 slot/n_task 共享 ION。
- **用 Auto 档**（允许拷贝）：单模型、输入来自 numpy、无需内存复用——`m([y, uv])` 一行，内部 from_numpy 自动 flush。
- 零拷贝的复杂度在 cache 维护 + 内存生命周期；用 `from_numpy` / `IONArray(m.input_descs[i])` / `inputs[=]` 这些封装入口，cache 已自动（from_numpy 写后 flush_clean，wait/wait_done 输出 flush_invalidate），只有裸操作底层 IONMemory 才需手动 flush。
- 判「两个 desc 描述的张量能否互换」用成员级 `==`（如 `ion.desc.dtype == other.dtype`）；判「这个 ion 能否绑到 input[idx]」用 `check_input`（含内存够不够 + 对齐）。串联判可用 check_input。

---

## 注意事项

1. 零拷贝路径（`start(task_id)`）要先用 `m.inputs[slot][idx]=ion` / `m.outputs[slot][idx]=ion` 绑定 IONArray，且 buffer 已写入并 flush_clean。
2. `from_memory` 的 byte_offset 必须**对齐 BPU 对齐步长**（S600=64 字节，S100/S100P=32）；`m.inputs[slot][idx]=ion` 是纯赋值不校验，预检用 `check_input`（含对齐），不对齐 check 返回 False、同步 `m(task_id)` 抛。
3. `from_memory` / `numpy()` 有越界校验（offset + 占用 <= memory.size），切错段抛 IndexError，不静默读邻居。
4. 多个 IONArray 共享同一 IONMemory（from_memory 切段）时，引用计数管理：最后一份 IONArray 析构才 hbUCPFree。
5. 取输出 IONArray 喂下游用 `m.outputs[slot][idx]`；`wait()` 返回的 ndarray 丢物理地址，不能再零拷喂下游。零拷贝档用 `wait_done`（不返回）+ `m.outputs[task_id]`。
6. 量化/反量化由 IONArray 负责（`quantize`/`dequantize`），CauchyKesai 不做透明量化。
7. `m.inputs`/`m.outputs` 是 py::list（无校验纯赋值）：绑错 dtype/不对齐异步不拦，到 BPU 才出错；要校验用 `check_input` 或同步 `m(task_id=0)`。
