# pyCauchyKesai 贡献指南

感谢参与。本文是本项目的**唯一行为规范**——代码怎么写、文档怎么写、PR 怎么提。目标：让任何人类贡献者或 AI 贡献者产出的代码与文档，都能与 PyTorch / NumPy / OpenCV 这类国际大厂项目看齐。

搭建环境、构建、移植性、测试体系见专题文档：

- 安装与构建：docs/cn_install.md
- 测试体系：docs/cn_TestCases.md

## 对标的成熟项目

本包是 C++（pybind11 绑定）+ Python wrapper 的 BPU 推理库，文档与代码规范对标同类成熟项目：

| 项目 | 借鉴点 |
|------|--------|
| PyTorch | Google style docstring、PEP-257、ruff/pydocstyle 强制、Examples 用 doctest、Shape 段、.. deprecated:: 指令、native function 必带 docstring |
| NumPy / numpydoc | API 文档段顺序（摘要/参数/返回/异常/示例/备注）、参数行 name : type 写法、versionadded/versionchanged/deprecated 版本标记 |
| pandas | docstring 段的强一致性、示例必带可复现 |
| pybind11 | C++ 绑定层 docstring 作为字符串传 .def()、签名由 pybind 自动追加、Python 侧 docstring 保持 Pythonic |
| OpenCV | 按 module 分组的文档组织 |

核心共识（五项一致遵循）：① 文档与代码同等重要，每个公开 API 都有 docstring；② 签名有类型注解时 docstring 不重复类型；③ 每个公开 API 至少一个可运行示例；④ 一个语义只在一处讲；⑤ 改了功能同步改文档。

---

## 项目结构

```
pyCauchyKesai/
├── src/pyCauchyKesai/        # Python 包（薄 wrapper + 平台层 + 工具）
│   ├── __init__.py           #   CauchyKesai 子类、from_numpy、版本、环境变量默认值
│   ├── platform.py           #   Platform：机型/核数/占用率/温度等平台原子能力
│   ├── _render.py            #   summary/benchmark/platform 的 dict→彩色串渲染
│   └── tools/                #   ez_onnx（CPU 模拟）、data_analyzer
├── csrc/nash/                # C++ 实现 + pybind11 绑定（编译产物 = native 模块）
│   ├── cauchy_kesai.cpp      #   推理编排、任务槽、核调度、summary/benchmark
│   ├── ion_array.cpp         #   IONArray：自描述 Tensor + 量化
│   ├── ion_memory.cpp        #   IONMemory：ION 物理内存 RAII
│   ├── platform.cpp          #   C 桥：soc_name/dnn_version/bpu_version/核数/对齐
│   ├── bindings.cpp          #   pybind11 绑定层（Python 看到的 API 真相）
│   └── include/              #   头文件
├── tests/                    # 测试（11 维接口一致性 + 各模块单测）
├── docs/                     # 文档（cn_ 前缀中文，无前缀英文）
├── examples/                 # 可运行示例
├── pyproject.toml            # 构建（scikit-build-core + pybind11）+ lint 配置
└── .clang-format             # C++ 格式（基于 Google）
```

分层铁律：**Python 层薄、C++ 层重**。硬件交互、性能路径全在 C++；Python 只做编排、类型包装、信息聚合。跨层调用单向：Python → C++（经绑定层），C++ 不回调 Python 业务逻辑（构造期的 warnings.attr 除外）。

---

## 代码风格

### Python

- 遵循 PEP 8，单行不超过 100 字符
- 遵循 PEP 257（docstring 约定），详见下文「文档规范」
- 公开 API 全部加类型注解；内部辅助函数按需
- import 顺序：标准库 → 第三方 → 本包，组间一空行；同组按字母序

命名总表：

| 类别 | 风格 | 示例 |
|------|------|------|
| 函数 / 变量 / 方法 | snake_case | from_numpy、bpu_align |
| 类 | PascalCase | CauchyKesai、IONArray、Platform |
| 常量 | UPPER_SNAKE | HB_DNN_DESC_TYPE_STRING |
| 私有 / 内部 | 前缀下划线 | _render.py、_read_text、_jsonable |
| 模块文件 | snake_case | platform.py、ion_array.cpp |

#### Lint 与类型检查

用 ruff（含 pydocstyle 规则）+ mypy 强制，配置建议加进 pyproject.toml：

```toml
[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src/pyCauchyKesai"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "D"]   # D = pydocstyle，强制 docstring 规范
ignore = ["D100", "D104", "D105"]            # 模块/包/魔术方法 docstring 按需

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D"]                            # 测试不强制 docstring

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true                # numpy / pycauchykesai native
```

日常命令：`ruff check src/pyCauchyKesai/`、`ruff format src/pyCauchyKesai/`、`mypy src/pyCauchyKesai/`。

### C++

- 基于 Google C++ 风格，配置见 .clang-format（Allman 大括号、缩进 4、列宽 120）
- 单行不超过 120 字符，缩进 4 空格，不用 Tab
- 命名：变量/函数 snake_case，类 PascalCase，**成员变量带尾下划线**（backends_、memory_、byte_offset_），宏 UPPER_SNAKE，命名空间全小写（cauchykesai）
- 头文件守卫用 `#ifndef / #define / #endif`（见 ion_array.h），或 `#pragma once`
- include 顺序：对应头 → C 标准库 → C++ 标准库 → 第三方（pybind11/hobot）→ 本项目，组间空行
- 资源用 **RAII** 管理：shared_ptr 持有、构造即分配、析构即释放，禁止裸 new/delete 配手工 free
- **阻塞操作释放 GIL**：调 hbUCPWaitTaskDone / hbUCPSubmitTask 等可能阻塞的 SDK 前，用 `py::gil_scoped_release release;`（见 cauchy_kesai.cpp 的 wait/start/析构），防止 BPU 阻塞时 Python 死锁
- 共享状态用 `std::atomic`（见 is_infer），避免数据竞争
- 异常映射约定（pybind11 自动）：`std::runtime_error`→RuntimeError、`std::invalid_argument`→ValueError、`std::out_of_range`→IndexError——按想要的 Python 异常类型选 C++ 异常类

格式化：`clang-format -i csrc/nash/*.cpp csrc/nash/include/**/*.h`。

### C++ 绑定层（pybind11）

绑定层（csrc/nash/bindings.cpp）是 C++ 与 Python 的**契约边界**，docstring 直接决定 Python 侧 help() 与文档产出：

- 每个 `.def()` / `.def_property_readonly()` / `.def_static()` / `__call__` / `__array__` 都带 docstring 字符串（多行用相邻字符串字面量拼接）
- **docstring 正文不重复签名**：pybind 会自动把参数名 + 默认值拼到 docstring 前面。正文只写行为、约束、异常、副作用——对标 PyTorch native function
- `py::arg()` 必须给全参数名，默认值写进 `py::arg("name") = default`，让自动签名完整可读
- 语义只讲一处：实现细节写 csrc 注释，Python 用户看的写 docstring，二者不复制

### Commit 信息

用 conventional commit，正文换行写动机与影响：

| 前缀 | 用途 | 示例 |
|------|------|------|
| feat | 新功能 | feat: 新增 Platform.ucp_library_path() |
| fix | bug 修复 | fix: from_numpy strided 写越界未抛 IndexError |
| docs | 文档变更 | docs: 补 compile_config 字段说明 |
| refactor | 重构（不改行为） | refactor: 对齐职责迁回 IONArray |
| test | 测试增改 | test: 补 from_memory 零拷贝用例 |
| chore | 杂项维护 | chore: 升级 pybind11 至 2.13 |

---

## 文档规范

文档分两类，都对标 PyTorch / NumPy，用 markdown / 纯文本落地，**不依赖 Sphinx**：

- **外部文档**（docs/cn_*.md、docs/*.md）：给人读的教程与 API 参考
- **源码 docstring**：给 help()、IDE 提示、AI 结构化消费的契约

### 外部文档通用约定

1. 简单 Markdown，不炫技
2. 表格内除了换行，不准用任何 Markdown 语法 —— 不准反引号、不准加粗、不准斜体

| 正确 | 错误 |
|------|------|
| 直接写文字 | 用反引号包起来 |
| 直接写文字 | 加粗或斜体 |

3. 中文文档用 cn_ 前缀放 docs/（如 cn_Platform.md），英文文档无前缀（如 API_cn.md 为导航例外）
4. 改了功能同步更新文档，保持中英文档一致
5. **一个语义只写一处**：重复内容收敛到权威节，其余处用「详见 X 节」指引，避免多处描述漂移

### API 文档结构（外部文档）

每个公开 API 按以下节段顺序写（缺的省略），对标 numpydoc 段顺序：

1. 签名 —— 代码块，写全默认值
2. 一句话摘要 —— 不超过 80 字符，紧跟标题，句号结尾
3. 参数表 —— 五列：参数、类型、必填、默认值、说明
4. 返回值 —— 类型加说明（无返回则省略）
5. 异常 —— 列出会抛的异常名和触发条件
6. 示例 —— 代码块，省略默认参数，关键结果用注释标注预期输出
7. 备注 —— 提示框、版本标记（按需）

完整范例见 docs/cn_CauchyKesai_with_IONArray.md 的 IONMemory 构造。

### 参数表

统一五列，顺序固定：参数、类型、必填、默认值、说明。示例代码省略默认参数，函数签名写全默认值，二者必须区分（对标 numpydoc：默认值归签名，示例不重复）。

### 提示与警告

用 GitHub 原生 Alerts，不要裸写「注意：」：

> [!NOTE]
> 补充说明

> [!WARNING]
> 容易踩坑的警告

> [!IMPORTANT]
> 必须遵守的强约束

接口抛出的异常用**异常表格**（两列：异常、说明），放在 API 小节末尾，不要裸写「抛出异常：」也不要用 Alert：

| 异常 | 说明 |
|------|------|
| IndexError | task_id 越界；priority 越界 |
| ValueError | 输入数量不匹配；dtype/shape 不匹配 |

### 版本与变更标记

新增、变更、废弃的 API 在备注里标注版本（对标 NumPy / Sphinx 的 versionadded / versionchanged / deprecated）：

- 新增于 1.2.0
- 自 1.2.0 起行为变更
- 已废弃，改用 xxx（标注预计移除版本）

### Python docstring（源码内）

对标 PyTorch（Google style + PEP-257）。分级要求：

| API 复杂度 | docstring 要求 |
|-----------|---------------|
| 简单 getter / one-liner | 单行摘要（三引号一句话），如 platform.py 的各只读属性 |
| 普通函数 / 方法 | 摘要 + 按需的 Args/Returns/Raises/Example 段 |
| 复杂公开 API | 全段：摘要 → Args → Returns → Raises → Example → Note |

规则：

1. 三引号，首行摘要不超过 80 字符、句号结尾；函数用祈使句动词开头（返回…/分配…/读取…）
2. 段名用 Google style，段序固定：Args / Returns / Raises / Yields / Example(s) / Note
3. **签名已有类型注解时，docstring 不重复类型**（PyTorch / JAX 约定）；Args 段只在注解不能自明（约束、单位、取值范围）时补一句
4. 每个公开 API 都要有 docstring；私有辅助函数至少一行摘要
5. Example 用 doctest 格式（`>>>` 开头），确保可复制运行；公开 API 至少一个，至多 5 个
6. 类 docstring 加 Attributes 段列公开属性；张量类按需加 Shape 段（对标 PyTorch nn.Linear）

完整模板（函数）：

```python
def from_numpy(arr, cached=True):
    """从 numpy 数组创建 IONArray（分配 + 写入 + flush 一步到位）。

    非连续输入会先 np.ascontiguousarray 转连续。

    Args:
        arr: numpy 数组，非连续时自动转连续。
        cached: True 走 hbUCPMallocCached（默认）；False 走 hbUCPMalloc。

    Returns:
        IONArray，dtype 和 shape 与输入一致，数据已写入并 flush。

    Raises:
        RuntimeError: 底层 ION 分配失败（透传 IONArray 构造）。
        ValueError: from_numpy 时 dtype/shape 与 buffer 不匹配。

    Example:
        >>> ion = from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4))
        >>> ion.shape
        (3, 4)
    """
```

类 docstring 模板（含 Attributes / Shape）：

```python
class IONArray:
    """ION 物理内存管理器 + 自描述 Tensor。

    通过 shared_ptr<IONMemory> 管理底层物理内存（组合关系），在其上叠加
    tensor 性质（dtype/shape/stride/量化），导出零拷贝 numpy 视图。

    Attributes:
        dtype: 元素类型（np.dtype）。
        shape: 逻辑形状（tuple[int]）。
        is_allocated: 是否已分配 ION 内存。

    Shape:
        与 numpy.ndarray.shape / .ndim / .size 语义一致。
    """
```

---

## 测试

测试体系、目录结构、运行方法见 docs/cn_TestCases.md。

- 补测试放对应 tests/ 子目录，命名 `test_功能_场景`
- 硬件依赖用 conftest.py 的 fixture mock 掉，保证 CI（无 BPU）可跑
- 修 bug 先写能复现的失败用例，再修代码（回归防护）
- 运行：`pytest`（全量）、`pytest tests/path/to/test_xxx.py -k 场景`（单测）

## Pull Request 流程

1. Fork 并建分支：`git checkout -b feat/your-feature-name`
2. 写干净、带注释的代码，为新功能补测试，按需更新文档
3. 本地自测：`pytest`、`ruff check`、`ruff format --check`、`mypy src/pyCauchyKesai/`
4. 用 conventional commit 提交并推到 fork
5. 向 main 提 PR，填模板，关联相关 issue
6. 处理 review 意见，保持 PR 聚焦，需要则 squash 提交

### PR 自检清单

提 PR 前逐条核对（AI 贡献者同样适用）：

- [ ] ruff check / format / mypy 全绿
- [ ] 新公开 API 有 docstring + 至少一个 doctest 示例
- [ ] 改了功能同步改了文档（外部文档 + docstring 一致）
- [ ] 新增 / 变更 API 标注了版本（新增于/自…起变更/已废弃）
- [ ] 新功能补了测试，修 bug 补了回归用例
- [ ] C++ 改动跑过 clang-format；阻塞 SDK 调用释放了 GIL
- [ ] 没有「一个语义多处描述」的新增重复

## 架构约束

- C++ 层只做硬件交互，用 RAII 管资源，阻塞操作释放 GIL，共享状态用 std::atomic
- Python 层保持薄，信息类接口返回原生 dict 配库渲染，**不要手写 dict 子类**
- 公开 API 必须线程安全，并在文档里写明线程安全保证
- 绑定层 docstring 是 Python 用户看到的真相，实现细节下沉到 csrc 注释，不外泄
- 一个语义只在一处讲：对齐权威归 IONArray、渲染归 _render、平台查询归 Platform

## 贡献方向

- 高优先级：性能优化、文档、测试覆盖、错误处理
- 中优先级：类型存根、示例、benchmark、自动化
- 低优先级：代码清理、结构化日志、profiling 工具

## 协议

提交贡献即表示同意以 GNU AGPL v3 授权，见 LICENSE。

## 联系

提 issue 或在现有 issue 下讨论。维护者：WuChao (D-Robotics)
