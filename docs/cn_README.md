# pyCauchyKesai 文档导航

**平台**: Linux aarch64 (Nash-e / Nash-m / Nash-p)
**Python**: >= 3.10
**许可证**: GNU AGPL v3

本文档是 pyCauchyKesai 的顶层导航：架构概览 + 包导出契约 + 跨文档索引。每个类/函数的完整 API 细节（签名、参数表、返回值、异常、示例）见对应专题文档。

## 架构概览

pyCauchyKesai 采用三层架构管理 BPU 与 CPU 共享的 ION 物理内存：

```
IONMemory (纯物理内存块) → IONArray (tensor 性质 + shared_ptr<IONMemory>) → CauchyKesai (推理编排)
```

- IONMemory：纯物理内存块，无 tensor 语义，是 IONArray 的底层（组合关系，IONArray 持 shared_ptr<IONMemory>）
- IONArray：在 IONMemory 之上叠加 tensor 性质（dtype、shape、stride、量化参数等），提供布局感知的读写
- CauchyKesai：BPU 推理引擎，负责模型加载、内存分配、任务调度

cache 操作（flush / invalidate）统一归 IONMemory；多模型零拷贝靠 IONMemory 共享（IONArray.memory + IONArray.from_memory()）。

## 模块导入

```python
from pyCauchyKesai import CauchyKesai, IONArray, IONMemory, from_numpy, Platform
from pyCauchyKesai.tools.ez_onnx import EZ_ONNX, NumPyArray
```

## 包导出契约（__all__）

顶层包导出：

| 名称 | 类型 | 说明 |
|------|------|------|
| CauchyKesai | 类 | BPU 推理引擎，详见 cn_CauchyKesai.md |
| IONArray | 类 | 自描述 Tensor，详见 cn_IONArray.md |
| IONMemory | 类 | ION 物理内存块，详见 cn_IONArray.md |
| from_numpy | 函数 | numpy→IONArray 工厂，详见 cn_IONArray.md |
| Platform | 类 | 平台原子能力，详见 cn_Platform.md |

tools 子包：

| 名称 | 类型 | 说明 |
|------|------|------|
| EZ_ONNX | 类 | ONNXRuntime CPU 模拟层，详见 cn_Easy_ONNX.md |
| NumPyArray | 类 | IONArray 的 CPU 模拟，详见 cn_Easy_ONNX.md |

## 跨文档索引

| 主题 | 文档 | 内容 |
|------|------|------|
| BPU 推理引擎 | cn_CauchyKesai.md | CauchyKesai 全部 API：构造、推理、内存、核调度、信息查询 |
| ION 内存与 Tensor | cn_IONArray.md | IONMemory / IONArray / from_numpy + UCP 接口 + 布局 + 量化 + BPU 数据类型 |
| 平台信息 | cn_Platform.md | Platform：soc/版本/核数/占用率/频率/温度/ION 内存 |
| ONNX CPU 模拟 | cn_Easy_ONNX.md | EZ_ONNX / NumPyArray + 与 CauchyKesai 对比 |
| 数据分析 | cn_DataAnalyzer.md | DataAnalyzer（未实现，设计提案） |
| 安装与构建 | cn_install.md | wheel 构建、移植性、UCP 库依赖 |
| 测试体系 | cn_TestCases.md | 11 维接口一致性套件 + 量化刻画 + 模型生成 |
| HBDK 编译配方 | cn_HBDK_Proposal.md | ONNX→HBM 编译场景与高带宽优化 |

## 快速选择

- 想跑模型：cn_CauchyKesai.md
- 想管 ION 内存 / 零拷贝：cn_IONArray.md
- 想查板子状态：cn_Platform.md
- 无 BPU 想在 CPU 验证：cn_Easy_ONNX.md
- 想装/编译：cn_install.md + cn_HBDK_Proposal.md
