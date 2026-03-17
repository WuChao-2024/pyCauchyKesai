# pyCauchyKesai Bayes (RDK X5) 与 Nash (RDK S100) 差异说明

## 概述

Bayes 实现基于 RDK X5 的 **libDNN** 库，Nash 实现基于 RDK S100 的 **libUCP** 库。
Python 接口完全相同，差异仅在 C++ 底层。

---

## API 差异对照表

| 功能 | Nash (S100 / libUCP) | Bayes (X5 / libDNN) |
|------|----------------------|---------------------|
| Packed handle 类型 | `hbDNNPackedHandle_t` | `hbPackedDNNHandle_t` |
| Task handle 类型 | `hbUCPTaskHandle_t` | `hbDNNTaskHandle_t` |
| 内存分配 | `hbUCPMallocCached(mem, size, deviceId)` | `hbSysAllocCachedMem(mem, size)` |
| 内存释放 | `hbUCPFree(mem)` | `hbSysFreeMem(mem)` |
| 内存刷新 | `hbUCPMemFlush(mem, flag)` | `hbSysFlushMem(mem, flag)` |
| 内存结构体 | `hbUCPSysMem { phyAddr, virAddr, memSize(u64) }` | `hbSysMem { phyAddr, virAddr, memSize(u32) }` |
| Tensor 内存字段 | `hbDNNTensor.sysMem`（单体） | `hbDNNTensor.sysMem[4]`（数组，使用 `[0]`） |
| `alignedByteSize` 类型 | `int64_t` | `int32_t` |
| `stride` 类型 | `int64_t[10]` | `int32_t[8]` |
| 最大 Tensor 维度 | 10 | 8 |
| 推理提交 | `hbDNNInferV2` + `hbUCPSubmitTask` | `hbDNNInfer`（含 ctrl param，一步完成） |
| 调度参数结构体 | `hbUCPSchedParam { priority, customId, backend, deviceId }` | `hbDNNInferCtrlParam { bpuCoreId, dspCoreId, priority, more, customId, ... }` |
| 调度参数初始化宏 | `HB_UCP_INITIALIZE_SCHED_PARAM` | `HB_DNN_INITIALIZE_INFER_CTRL_PARAM` |
| 等待推理完成 | `hbUCPWaitTaskDone(handle, timeout)` | `hbDNNWaitTaskDone(handle, timeout)` |
| 释放推理任务 | `hbUCPReleaseTask(handle)` | `hbDNNReleaseTask(handle)` |
| BPU Core 常量 | `HB_UCP_BPU_CORE_ANY` | `HB_BPU_CORE_ANY` |
| 优先级常量 | `HB_UCP_PRIORITY_LOWEST` | `HB_DNN_PRIORITY_LOWEST` |

---

## 数据类型枚举差异

X5 的 `hbDNNDataType` 枚举前 6 项是图像类型，Nash 无此前缀：

| 枚举值 | Nash (S100) 数值 | Bayes (X5) 数值 |
|--------|-----------------|-----------------|
| `HB_DNN_IMG_TYPE_Y` | 不存在 | 0 |
| `HB_DNN_IMG_TYPE_NV12` | 不存在 | 1 |
| `HB_DNN_IMG_TYPE_NV12_SEPARATE` | 不存在 | 2 |
| `HB_DNN_IMG_TYPE_YUV444` | 不存在 | 3 |
| `HB_DNN_IMG_TYPE_RGB` | 不存在 | 4 |
| `HB_DNN_IMG_TYPE_BGR` | 不存在 | 5 |
| `HB_DNN_TENSOR_TYPE_S4` | 0 | 6 |
| `HB_DNN_TENSOR_TYPE_S8` | 2 | 8 |
| `HB_DNN_TENSOR_TYPE_F32` | 7 | 13 |
| `HB_DNN_TENSOR_TYPE_BOOL8` | 13 | 不存在 |

Bayes 实现中，图像类型（Y/NV12/NV12_SEPARATE/YUV444/RGB/BGR）统一映射为 `"uint8"`。

---

## 内存布局差异（重要）

X5 上 `validShape` 是**逻辑形状**，实际内存大小由 `alignedByteSize` 决定，两者可能不一致。

典型案例：NV12 输入模型
- `validShape = (1, 3, 224, 224)` → 逻辑上 150528 字节
- `alignedByteSize = 75264` → 实际 NV12 内存（224×224×1.5）

因此 Bayes 实现中：
- `memcpy` 使用 `alignedByteSize`，而非 `numpy.nbytes`
- `input_tensors` 零拷贝视图展平为 `(alignedByteSize,)` 的一维 `uint8` 数组
- `s()` 中图像类型输入的 `shape` 显示为 `(alignedByteSize,)`，而非 `validShape`
- `inference()` 对图像类型输入的校验也使用 `(alignedByteSize,)` 一维形状

这意味着对于 NV12 输入模型，用户应传入 `(alignedByteSize,)` 的一维 `uint8` 数组，而非按 `validShape` 构造的多维数组：

```python
# 正确用法：用 input_tensors 的 shape 构造输入
inp = np.zeros(model.input_tensors[0][0].shape, dtype=np.uint8)  # (75264,)
out = model([inp])

# 或直接查询 s() 获取 shape
info = model.s()
inp = np.zeros(info['inputs'][0]['shape'], dtype=np.uint8)  # (75264,)
```

Nash 实现中 `validShape` 与实际内存大小一致，无此问题。

---

## pybind11 零拷贝差异

pybind11 3.x 中 `py::array(dtype, shape, ptr)` 当 `base=handle()` 时会触发 `PyArray_NewCopy`（数据拷贝）。

Bayes 实现中所有零拷贝视图构造均传入 `py::none()` 作为 base：
```cpp
py::array(dtype, shape, ptr, py::none())  // Bayes: 显式指定 base=None，零拷贝
py::array(dtype, shape, ptr)              // Nash: 旧版 pybind11 默认零拷贝
```

---

## Tensor 数组存储方式差异

| | Nash (S100) | Bayes (X5) |
|--|-------------|------------|
| 存储方式 | `std::vector<std::vector<hbDNNTensor>>` | `std::vector<hbDNNTensor *>`（裸指针，`new[]` 分配） |
| 原因 | `hbDNNInferV2` 不修改输出指针 | `hbDNNInfer` 的 `hbDNNTensor **output` 参数会修改指针本身，`std::vector::data()` 是临时值，必须用稳定的成员指针 |

---

## 头文件路径差异

| | Nash (S100) | Bayes (X5) |
|--|-------------|------------|
| DNN 头文件 | `hobot/dnn/hb_dnn.h` | `dnn/hb_dnn.h` |
| 内存头文件 | `hobot/hb_ucp_sys.h` | `dnn/hb_sys.h` |
| UCP 头文件 | `hobot/hb_ucp.h` | 不需要（X5 无独立 UCP 层） |

---

## 构建差异

| | Nash (S100) | Bayes (X5) |
|--|-------------|------------|
| 链接库 | `hbucp dnn hbrt4 hbtl hb_arm_rpc hlog_wrapper perfetto_sdk` | `dnn hbrt_bayes_aarch64`（打包进 wheel） |
| 库路径 | 项目内 `lib/` 目录，需打包进 wheel | 项目内 `lib/` 目录，需打包进 wheel |
| RPATH | `$ORIGIN/lib` | `$ORIGIN/lib` |
| 运行时库加载 | RPATH 自动解析 | `__init__.py` 预加载（需先加载系统 `libcnn_intf.so`） |

---

## Python 接口差异

| 功能 | Nash | Bayes | 说明 |
|------|------|-------|------|
| `input_tensors` 形状 | `validShape` 对应的多维数组 | `(alignedByteSize,)` 一维 uint8 | X5 图像类型内存布局与逻辑形状不一致 |
| `s()` 输入 shape 显示 | `validShape` | 图像类型显示 `(alignedByteSize,)`，tensor 类型显示 `validShape` | 与 `input_tensors` 保持一致 |
| `inference()` 输入校验 | 按 `validShape` 校验 | 图像类型按 `(alignedByteSize,)` 校验，tensor 类型按 `validShape` 校验 | 同上 |
| `__call__` / `inference` | 完全相同 | 完全相同 | — |
| `s()` / `t()` / `start()` / `wait()` / `is_busy()` | 完全相同 | 完全相同 | — |
