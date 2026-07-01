# CauchyKesai — BPU 推理编排器

- [CauchyKesai — BPU 推理编排器](#cauchykesai--bpu-推理编排器)
  - [概述](#概述)
  - [快速开始](#快速开始)
  - [构造函数](#构造函数)
  - [推理方法](#推理方法)
    - [同步推理 __call__](#同步推理-call)
  - [任务槽](#任务槽)
  - [BPU 执行模型](#bpu-执行模型)
    - [三个粒度](#三个粒度)
    - [单核串行可排队](#单核串行可排队)
    - [任务优先级 priority](#任务优先级-priority)
  - [BPU 核调度](#bpu-核调度)
    - [set\_scheduling\_params](#set_scheduling_params)
    - [scheduled\_cores](#scheduled_cores)
    - [scheduled\_priority](#scheduled_priority)
  - [信息查询](#信息查询)
    - [summary](#summary)
    - [benchmark](#benchmark)
    - [只读属性](#只读属性)
    - [__repr__](#repr)
  - [Context Manager](#context-manager)
  - [环境变量](#环境变量)
    - [L2 Cache：HB\_DNN\_USER\_DEFINED\_L2M\_SIZES](#l2-cachehb_dnn_user_defined_l2m_sizes)
    - [日志级别：HB\_UCP\_LOG\_LEVEL / HB\_NN\_LOG\_LEVEL / HB\_VP\_LOG\_LEVEL / HB\_HPL\_LOG\_LEVEL](#日志级别hb_ucp_log_level--hb_nn_log_level--hb_vp_log_level--hb_hpl_log_level)
    - [自定义](#自定义)
  - [注意事项](#注意事项)

## 概述

CauchyKesai 是 BPU 平台的 Python 推理编排器。一句话概括：numpy in → numpy out——加载 .hbm 模型，传入 list[np.ndarray]，返回 list[np.ndarray]。

- 架构：C++ 实现 + pybind11 绑定 + Python 包装层
- 平台：Nash-e (S100), Nash-m (S100P), Nash-p (S600)


## 快速开始

```python
import numpy as np
from pyCauchyKesai import CauchyKesai, __version__
print(f"Cauchy Kesai: {__version__}")

m = CauchyKesai("/app/model/basic/resnet18_224x224_nv12.hbm")

m.s()   # Model Summary（美观打印）

m.t()   # Model Benchmark（美观打印计时）

y = np.random.rand(1,224,224,1).astype(np.uint8)
uv = np.random.rand(1,112,112,2).astype(np.uint8)

o = m([y, uv])[0]
print(f"{o.shape = }, {o.dtype = }")
```



## 构造函数

```python
def __init__(self, model_path: str, n_task: int = 1, model_cnt_select: int = 0) -> None:
```

加载 BPU 模型，读取输入输出张量元信息。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| model_path | str | 是 | - | .hbm 模型文件路径 |
| n_task | int | 否 | 1 | 并发任务槽数量，用户自定义、无上限（≤0 钳到 1 并警告）；每个 slot 构造时预分配输入+输出 ION，n_task×单槽内存需落在 ION 堆预算内；实际同时存活的 BPU task 数另受任务配额约束 |
| model_cnt_select | int | 否 | 0 | packed 模型中的子模型索引，超范围自动钳位并警告 |


构造时行为：加载模型、读取输入输出张量的 dtype/shape 等元信息。packed 模型：一个 .hbm 可包含多个模型，model_cnt_select 选择第几个，默认选第一个（索引 0）。

| 异常 | 说明 |
|------|------|
| RuntimeError | 模型文件不存在或不是文件；模型加载失败（hbDNNInitializeFromFiles failed） |

> [!NOTE]
> **packed 模型：权重只载入一份**。一个 *.hbm 可 pack 多个子模型（LLM 常把 decode/prefill 打成一个文件）。`hbDNNInitializeFromFiles` 一次载入整个文件 = 所有子模型权重，`model_cnt_select` 只是从中挑一个子模型的句柄，不另加载权重。
> 多个实例加载同一个 .hbm 文件共享同一份权重，不会翻倍。

---

## 推理方法

CauchyKesai 的推理入口分两类：**同步**（`__call__`，阻塞到出结果）与**异步**（`start` / `wait` / `wait_done` / `is_busy`，提交后返回）。本档介绍同步接口 `__call__`。

- **同步 `__call__`**（本档）：按实参类型派发两档——传 list[np.ndarray] 走 Auto（内部 from_numpy 写 slot 的 bound IONArray + 自动 flush），不传走零拷贝（用 slot 已绑定的 IONArray）。简单，多数场景够用。
- **同步才校验，异步不校验**：`__call__`（同步）校验 bound（check_input/output，不匹配抛）；`start`/`wait_done`/`wait`（异步）不校验 bound（要快/浅，用户改错自己负责）。

### 同步推理 __call__

```python
def __call__(self, inputs: list[np.ndarray], task_id: int = 0) -> list[np.ndarray]:   # Auto 同步
def __call__(self, *, task_id: int = 0) -> list[np.ndarray]:                          # 零拷贝同步
```

同步推理 = start + wait。按实参类型派发：传 list 走 Auto（校验 dtype/shape + from_numpy），传 int（或仅 task_id）走零拷贝（校验 bound）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| inputs | list[np.ndarray] | Auto 档必填 | - | Auto 档：输入数组列表，dtype/shape 必须与模型模板一致；零拷贝档不传 |
| task_id | int | 否 | 0 | 任务槽索引 0 ~ n_task-1 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| outputs | list[np.ndarray] | 输出数组列表，顺序与模型输出张量一致，元素 dtype/shape 同模型模板（两档一致都导出） |

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界 |
| ValueError | Auto 档输入数量/dtype/shape 不匹配；零拷贝档 bound IONArray 性质不符模板（check 失败） |
| RuntimeError | 输入数据超 ION 缓冲区；slot busy |

```python
o = m([y, uv])[0]        # Auto 同步：numpy in → numpy out
o = m(task_id=0)[0]      # 零拷贝同步：用 m.inputs[0] 已绑定的 IONArray（用户提前写好 buffer + flush）
```

---

## 任务槽

task_id 是并发任务槽索引，取值 0 ~ n_task-1（n_task 构造时指定，默认 1），即第几个 OE 任务槽。不同 task_id 互不干扰，可并发提交；同步推理（__call__）和 start/wait 都带 task_id 参数。

每个任务槽有三态：idle（空闲，可提交 / 改绑）、infering（BPU 执行中，同 slot 不可重入）、waiting（推理完成待取结果）。

---

## BPU 执行模型

理解推理在 BPU 上的真实执行方式，对用好 task_id / priority / set_scheduling_params 至关重要。一句话总纲：**核是串行的、提交是异步的、排序靠优先级。**

### 三个粒度

| 层 | 是什么 | 关键属性 |
|---|---|---|
| 核 core | 物理 BPU 计算核 | 每个核有独立的硬件队列，核间彼此并行 |
| function-call | BPU 的执行粒度（编译期产物） | 一个模型 = 1 个或多个 function-call，在核的硬件队列上按序执行，全部跑完即一次推理完成 |
| task | 用户提交的一次推理任务 | 一个 task 驱动整个模型的 function-call 序列 |

### 单核串行可排队

- **单核同一时刻只跑 1 个 function-call**：BPU 硬件本身无任务抢占——一个任务进入核计算后会一直占用该核，直到其 function-call 序列跑完，其他任务只能排队等待。
- **提交是异步的**：`start()` 调用 `hbUCPSubmitTask`，把 task 丢进调度器优先级队列即返回，不等核空闲。故可连续 `start()` 多个 task（跨 task_id，或复用已空闲的槽），它们排队，核空闲时按优先级挑一个派上去。
- **slot 与 task 是两回事**：`n_task`（slot 数）由用户自定义、无上限——但每个 slot 在构造时预分配自己的输入+输出 ION 缓冲区，故 n_task×单槽内存需落在 ION 堆预算内（大模型 + n_task 过大会 OOM）；slot 本身不占 BPU 任务配额。真正提交到 BPU 的 task 只有在 `start()` 时才产生（`hbUCPSubmitTask`），`wait`/结束后释放。同一时刻存活的 task 数受两道独立约束：① SDK 任务池 `HB_UCP_MAX_TASK_NUM`（默认 32，环境变量可调）；② BPU 描述符配额 ≈ 2048/（输入+输出张量数）（闭源驱动硬编码，不可调），与内存/核数无关。即你可以开很多 slot 随时有空闲槽可提交。

> 跨核并行 = 把不同 task 的核掩码指向不同核，或在 S600 上编译 2 核模型（core_num=2 张量并行）。单核模型在单个核上只能排队串行。

### 任务优先级 priority

`priority`（0-255，默认 0）是 **per-slot 调度参数**，与核掩码同经 `set_scheduling_params(priority=...)` 设置（不再随 `__call__` / `start` 传），写入 `hbUCPSchedParam.priority`，决定 task 在优先级队列里的执行顺序，分两档：

**普通排队（0-253，非抢占，默认）**：核一空闲，调度器从队列挑 priority 最高的 task 上核；同级再比 customId（数值小优先；规则 priority > customId）。不打断正在跑的任务。

**抢占（254 = high / 255 = urgent）**：高优可抢占低优（urgent > high > 普通）。抢占由 runtime 软件实现，粒度是 function-call 边界。生效需同时满足两个前提，缺一不可：
1. 模型编译时设置了 `max_time_per_fc`（否则所有 function-call 被 merge 成一个大块，无法被抢占）；
2. submit 时 `priority` 设为 254/255。

> [!IMPORTANT]
> **抢占与 L2 Cache 互斥**：启用 L2 Cache时不支持抢占优先级。即在本包默认配置下 254/255 不生效，实际只有 0-253 的排队语义。要启用抢占需先关闭本包默认 L2 Cache：import 前设 `UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1`，本包即不再填入 `HB_DNN_USER_DEFINED_L2M_SIZES`。


## BPU 核调度

每个 task slot 独享一份核掩码（backends_[task_id]）与优先级（priorities_[task_id]），可给不同 slot 配不同核子集 / 不同优先级、并发提交互不干扰（跨 slot 零竞态）。两者同经 set_scheduling_params 设置，在 SubmitTask 时写入 hbUCPSchedParam。

```python
model.set_scheduling_params([0, 1, 2, 3])                     # task_id=-1：广播所有 slot（核+优先级）
model.set_scheduling_params([0, 1], priority=10, task_id=0)   # slot 0 → 核 0,1 + 优先级 10
model.set_scheduling_params([2, 3], priority=20, task_id=1)   # slot 1 → 核 2,3 + 优先级 20
model.set_scheduling_params([], priority=253)                 # 广播：核重置 CORE_ANY，优先级提至 253
model.set_scheduling_params([0, 1], task_id=0)                # priority 默认 -1：只改核，优先级不变
```

### set_scheduling_params

```python
def set_scheduling_params(self, bpu_cores: list[int], priority: int = -1, task_id: int = -1) -> None:
```

设置 BPU 核调度（per-slot：核掩码 + 优先级）。`bpu_cores` 映射为 `hbUCPSchedParam.backend` 的核 bitmask、`priority` 映射为 `hbUCPSchedParam.priority`，写入对应 slot 的 backends_[task_id] / priorities_[task_id]，在 SubmitTask 时生效。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| bpu_cores | list[int] | 是 | - | 核心索引列表，映射为 HB_UCP_BPU_CORE_n 的 bitmask。空列表重置为 CORE_ANY |
| priority | int | 否 | -1 | -1=不改现有优先级；0-255=设该 slot 优先级（254/255 抢占） |
| task_id | int | 否 | -1 | -1=广播所有 slot；[0,n_task)=仅该 slot |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| - | None | 无返回值；新掩码写入 backends_[task_id]，下一次 start() 生效 |

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界 |
| ValueError | 核掩码错配（核数与编译核数不符、含无效物理核或索引越界）；priority 不在 [-1,255] |

**成本**：只是写 backends_[task_id] / priorities_[task_id]（单元素，或广播 fill），无内存申请、无 SDK 调用、无重编译，实测约 0.5 μs/次，近乎免费。新值在下一次推理才生效（start() 现读这两个向量填进 hbUCPSchedParam.backend / .priority），所以可以逐帧切，没有切换开销。真正贵的是构造句柄——加载 .hbm 模型是一次性开销，建好后反复复用。

**并发**：不同 slot 读写 backends_ / priorities_ 的不同元素，跨 slot 零竞态——可安全地给 slot 0 配 [0,1]+p10、slot 1 配 [2,3]+p20 同时 start，BPU 上分核并发。同一 slot 的 set 与 start 若并发则有竞态（非原子读写），契约：slot 空闲时配、submit 时读；aarch64 对齐 uint64/int32 读写单指令无撕裂，最坏只是某次 submit 拿到略旧值。

**边界**：能廉价切换的是核掩码（把编译好的并行度映射到哪几个物理核），不是并行核数本身——并行核数编译时烙进 .hbm，运行时改不了，要改得重新编译。

**运行时核约束**（实测 S600 物理 4 核 0–3）：

| 编译核数 | bpu_cores 要求 | 默认 [](CORE_ANY) | 惯用写法 |
|---|---|---|---|
| 1 | 含 ≥1 个有效核即可 | 可跑 | [0] 或 [] |
| 2 | 恰好 2 个有效核 | 被拒 | [0, 1] |
| 3 | 恰好 3 个有效核 | 被拒 | [0, 1, 2] |
| 4 | 恰好 4 个有效核 | 被拒 | [0, 1, 2, 3] |

- 单核模型宽松：掩码只要含至少 1 个有效核，[]/[0]/[2]/[0,1,2,3] 都能跑（SDK 从掩码里挑一个有效核）。
- 多核模型严格：掩码必须恰好 N 个有效核——多一个少一个都拒、含不存在核也拒。但任意 N 核子集均可，不必是前 N 核（core2 同样接受 [2,3]/[1,2]/[0,3]）。
- 构造已默认：构造 CauchyKesai 时自动给**所有 slot**设同一个默认掩码——多核模型给前 N 个有效物理核（[0..N-1]），单核留 CORE_ANY；优先级默认 0。所以多核模型构造完可直接推理，无需手动 set_scheduling_params。查某 slot 当前生效掩码用 scheduled_cores(task_id)、优先级用 scheduled_priority(task_id)；summary()['scheduled_cores'] / ['scheduled_priority'] 展示的是 slot 0。

注意：[]（显式重置 CORE_ANY）set 时放行，但多核模型推理时仍被 hbUCPSubmitTask 拒。物理核数由构造时一次性读 sysfs 得到（m.platform.physical_core_num，S600=4；无 sysfs 时为 -1，跳过物理核校验）。

### scheduled_cores

```python
def scheduled_cores(self, task_id: int = 0) -> list[int]:
```

返回指定 slot 当前生效的核索引列表，即该槽当前 `hbUCPSchedParam.backend`（backends_[task_id]）。默认 slot 0。注意它是方法（带括号），不是属性。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| task_id | int | 否 | 0 | 任务槽索引 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| cores | list[int] | 该槽当前生效核索引；空=CORE_ANY（SDK 自动挑核） |

### scheduled_priority

```python
def scheduled_priority(self, task_id: int = 0) -> int:
```

返回指定 slot 当前生效的优先级，即该槽当前 `hbUCPSchedParam.priority`（priorities_[task_id]）。值由 set_scheduling_params 的 priority 参数设置，构造默认 0。范围 0-255（254/255 抢占）。注意它是方法（带括号），不是属性。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| task_id | int | 否 | 0 | 任务槽索引 |

| 返回值 | 类型 | 说明 |
|--------|------|------|
| priority | int | 该槽当前生效优先级（0-255）；构造默认 0 |

---

## 信息查询

本节四个方法分两对：`summary()` / `benchmark()` **始终返回原生 dict**（numpy 已转原生，可 json.dumps / 供 AI 结构化消费，数据契约）；`s()` / `t()` **内部 print 美观彩色字符串**（直接调用即出图，返回 None，展示契约）。所有内部 SDK 字段（地址、枚举、对齐步长等）的诊断信息都收敛到 `summary()`。

### summary

```python
def summary(self) -> dict
def s(self) -> None          # 美观打印模型摘要（内部 print）
```

当模型含编译配置（hbDNNGetModelDesc 返回的配置 JSON，已解析为 `compile_config` 字段）时，`s()` 末尾会按 Toolchain/Model/Preprocess/Quantization/Compile 分组打印 Compile Config 段。

`summary()` 返回的原生 dict 字段如下（numpy 已转原生，可 json.dumps）；`s()` 返回 None（内部 print 美观输出）。

| 字段 | 类型 | 说明 |
|------|------|------|
| model_path | str | 模型文件路径 |
| model_names | list[dict] | 模型名列表，每个元素含 index/name/selected |
| n_task | int | 并发任务槽数量 |
| memory_mb | float | 输入+输出 alignedByteSize 总和 (MB) |
| compile_config | dict \| None | hbDNNGetModelDesc 配置 JSON 解析出的编译配置 dict（Toolchain/Model/Preprocess/Quantization/Compile 等字段）；非配置 JSON 时为 None |
| bpu_core_num | int | 模型编译时的 BPU 核数 |
| bpu_fw_version | str | BPU 固件版本（bpu0/fw_version，模型加载后读，如 1.1.26） |
| scheduled_cores | list[int] | slot 0 当前生效核掩码（空=CORE_ANY） |
| scheduled_priority | int | slot 0 当前生效优先级（构造默认 0） |
| inputs | list[dict] | 输入张量信息列表 |
| outputs | list[dict] | 输出张量信息列表 |
| platform | dict | 平台信息子 dict（Platform.summary() 快照） |

每个 input/output dict 含：index, name, dtype, tensorType, shape, alignedByteSize（含 padding 的对齐大小）, quantiType (0=NONE, 1=SCALE), quantizeAxis, scale, zero_point（scale/zero_point 经 _jsonable 已转原生 list，可 json.dumps）, stride（对齐行步长）, desc（hbDNNGetInputDesc/OutputDesc 元信息串，无则 N/A）。

```python
info = model.summary()
shape = info['inputs'][0]['shape']
```

### benchmark

```python
def benchmark(self, timeout_ms: int = 0) -> dict
def t(self, timeout_ms: int = 0) -> None   # 美观打印计时（内部 print）
```

使用任务槽 0 执行一次完整推理（零拷贝 `start(0)` + `wait(0)`，用 slot 0 当前 bound 数据计时，不传 inputs）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| timeout_ms | int | 否 | 0 | 时间预算（毫秒），0=无限等待；任务实际耗时超过该值则抛 RuntimeError |

`benchmark()` 返回的原生 dict 字段如下；`t()` 返回 None（内部 print 美观计时）。

| 字段 | 类型 | 说明 |
|------|------|------|
| time_us | float | 推理耗时（微秒） |
| time_ms | float | 推理耗时（毫秒） |
| time_s | float | 推理耗时（秒） |
| time_min | float | 推理耗时（分钟） |

```python
timing = model.benchmark()
print(f"{timing['time_ms']:.2f} ms")
```

### 只读属性

| 属性 | 类型 | 说明 |
|------|------|------|
| input_names | list[str] | 模型输入张量名称列表 |
| output_names | list[str] | 模型输出张量名称列表 |
| bpu_core_num | int | 模型编译时的 BPU 核数 hbDNNGetCompileBpuCoreNum |
| platform | Platform | 平台原子能力（全局单例；m.platform 与模块级 platform 同源） |


### __repr__

格式化输出，如 `CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=1)`；有 task 槽正忙时尾部追加 `, N busy`。

---

## Context Manager

CauchyKesai 支持 context manager 语法（Python wrapper 独有）。C++ 析构由 pybind11 自动管理（GC 在引用归零后释放 BPU 资源，__exit__ 不主动 close）。

```python
with CauchyKesai("model.hbm", n_task=1) as ck:
    outputs = ck(inputs)
# 自动释放 BPU 资源
```

---

## 环境变量

pyCauchyKesai 在 import 时（src/pyCauchyKesai/__init__.py）用 os.environ.setdefault() 为以下环境变量设置默认值——仅当变量未设置时填入，import 前显式设置即可覆盖。

### L2 Cache：HB_DNN_USER_DEFINED_L2M_SIZES

默认 6:6:6:6。格式 N:N:N:N，四段对应 4 个 BPU 核（核0/1/2/3），单位 MB，仅整数，0 表示该核不分配 L2 Cache。官方示例：6:6:6:6（4 核各 6 MB）、0:6:0:0（仅核1）、12:0:0:12（核0 与核3）。

UCP 对 L2 Cache 采用静态映射，在推理准备阶段按配置一次性分配。约束：多进程下各进程 L2 Cache 总和不能超过硬件最大 L2 Cache；启用 L2 Cache 优化时不支持推理优先级设为抢占。硬件最大值查询：

```bash
cat /sys/kernel/debug/ion/heaps/custom
```

**关闭本包默认 L2 Cache（启用抢占）**：import 前设 `UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1`，本包 import 时即不再为 `HB_DNN_USER_DEFINED_L2M_SIZES` 填默认值（已显式设置的值不动）。代价是放弃 L2 Cache 优化（推理变慢）；用途是换取抢占优先级——L2 Cache 与抢占互斥，关掉 L2 Cache 后抢占 254/255 才可能生效，抢占还需模型编译时设 `max_time_per_fc`（否则 function-call 被 merge 成一大块，无法被抢占）。

### 日志级别：HB_UCP_LOG_LEVEL / HB_NN_LOG_LEVEL / HB_VP_LOG_LEVEL / HB_HPL_LOG_LEVEL

默认全部 6（never，不输出任何日志）。四个变量同用 0-6 七级体系：

| 数值 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|---|
| 名称 | trace | debug | info | warn | error | critical | never |

规则：发生的 log 等级大于等于设置等级才会输出，设置越小输出越多。SDK 原始默认 warn(3)，pyCauchyKesai 改为 never(6)。

### 自定义

import 前设置即可覆盖默认值：

```python
import os
os.environ["HB_DNN_USER_DEFINED_L2M_SIZES"] = "8:8:4:4"  # 自定义 L2 Cache
os.environ["HB_UCP_LOG_LEVEL"] = "2"                      # 开 info 日志

from pyCauchyKesai import CauchyKesai
```

---

## 注意事项

1. 环境变量（L2 Cache、日志级别）已在 import 时自动设默认值。
2. 输入 dtype/shape 必须与模型模板一致，否则抛 ValueError/RuntimeError。
3. 构造后即可直接推理。
4. 多核模型构造时已自动给所有 slot 设默认核（前 N 核），通常无需手动 set_scheduling_params；显式调用可改。set_scheduling_params 支持 per-slot（task_id 参数），可给不同 slot 配不同核子集 / 不同优先级（priority 参数）并发。
