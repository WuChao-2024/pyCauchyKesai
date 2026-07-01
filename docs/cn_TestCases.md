# pyCauchyKesai 测试方法文档

本文档说明 pyCauchyKesai（BPU Nash-m runtime 的 Python 绑定）的测试体系：测什么、怎么组织、怎么跑、模型从哪来。

---

## 一、测试目标

pyCauchyKesai 是 BPU 推理引擎的 pybind11 绑定（numpy in → numpy out）。测试回答四个层次的问题：

1. 单个 API 能不能用 —— 构造、summary、推理、并发、ION 内存（nash/ 与基础用例）。
2. 量化后精度掉多少 —— torch 浮点真值 vs 板端定点输出（golden/）。
3. 接口在多个正交维度上一致吗 —— 跨算子、跨 rank、跨推理路径、跨核（dimensions/）。
4. 文档介绍的每个细节都成立吗 —— 针对 Auto / IONArray 两份文档逐条断言（docvariant/）。

---

## 二、目录结构

```
tests/
├── conftest.py                 根 pytest 配置：平台检测 + marker 注册(bpu/nash/dim1..dim11/docvar)
├── test_cauchy_kesai.py        基础功能(单模型:构造/summary/推理/并发/IONArray)
├── test_ai_native.py           AI Native 糖(from_numpy;Pipeline 为测试内部辅助)
├── nash/                       纯内存层(IONMemory/IONArray),不依赖 BPU
│   ├── test_ion_memory.py
│   └── test_ion_array.py       shape×操作×cached/uncached 全排列
├── golden/                     量化刻画(torch 真值 vs 板端)
│   ├── harness.py              数据驱动:manifest/meta/npz 加载 + 误差指标
│   ├── run_characterization.py 闭环刻画 → report/{per_case,by_axis,summary}
│   └── test_golden_models.py   烟雾测试(加载+sample0+有限性断言)
├── dimensions/                 接口一致性多维度套件(11 维度)
│   ├── _harness.py _paths.py _make_total_report.py conftest.py
│   └── d1_ops_structure/ … d11_api_metadata/   每维度一个目录
├── docvariant/                 文档细节测试(针对 Auto/IONArray 两份文档,变异 hbm 驱动)
│   ├── conftest.py             manifest 驱动 + bpu_free 探活 + make_inputs
│   ├── test_doc_construct.py / test_doc_inference.py / test_doc_scheduling.py
│   ├── test_doc_metadata.py / test_doc_ionarray.py / test_doc_zerocopy.py / test_doc_envvar.py
│   └── REPORT.md               文档细节测试报告
└── generate/                   模型生成代码(云端 docker 跑 torch→onnx→hbm)+ 闭环脚本
    ├── build_matrix.py         矩阵流水线(10 backbone × rank × 变体 × core)
    ├── gen_doc_variants.py     文档变异流水线(15 编译规范,驱动 docvariant/)
    ├── run_cloud.sh / sync_to_board.sh
    └── README.md
```

五层是历史叠加，每层回答不同问题，不是重复：基础功能 → 纯内存 → 量化刻画 → 多维接口一致性 → 文档细节。

---

## 三、文档细节测试(docvariant/,针对两份 API 文档)

这是最贴近文档的一层。逐条核对 `docs/cn_CauchyKesai_Auto.md` 与 `docs/cn_CauchyKesai_with_IONArray.md` 介绍的接口/参数/异常/路径，用一组**变异编译规范的 HBM**驱动(每个 hbm 对应文档一组细节)。

### 3.1 变异 HBM(gen_doc_variants.py,云端 docker 产)

`gen_doc_variants.py` 在云端 docker 内 torch 导出 onnx + 生成 calib/golden + 变异 YAML，逐个 `hb_compile` 产出 15 个不同编译规范的 HBM。**15/15 全部闭环可产**(实测)：

| 变异 spec | 编译规范 | 覆盖文档细节 |
|---|---|---|
| conv_core1 / core2 / core4 | conv2d int16,核数 1/2/4 | set_scheduling_params 核掩码严格性/构造默认掩码/core 等价/核数边界 |
| conv_int8_c1 | int8 | int8 量化 |
| conv_nv12_c1 | input_type_rt=nv12 | 文档「快速开始」m([y,uv]) nv12 双输入 |
| conv_rgb_c1 | input_type_rt=rgb | rgb uint8 输入/quantize 写侧 |
| concat_2i1o / 1i2o / 2i2o | 多输入/多输出/mixed_io | inputs/outputs 多 idx 绑定/check 多 idx/绑定顺序 |
| matmul_1d / 3d | rank 1D/3D | IONArrayDesc.ndim/shape 变体 |
| conv_preempt_c1 | max_time_per_fc=1000 | 抢占优先级 254/255(运行时配 UNSET L2 Cache) |
| matmul_scaleout_c1 | model_output_type=int8 | SCALE 输出可达性(实测:S600 仍 NONE,平台事实) |
| chain_a / chain_b | matmul [1,32] | 多模型零拷贝链式 cookbook |

编译规范要点(踩过的坑，已固化在脚本)：
- nv12/rgb 的**校准数据用 float32 featuremap**(不是 uint8 图像)——input_type_train=rgb 但 calib 是 float32,runtime 才接 uint8。云端 rgb_identity_fixed 先例证实。
- max_time_per_fc 只能为 0 或 [1000, 4294967295](us)，设 1000 才能编译。
- onnx_model / cal_data_dir 路径含 `docvariants/` 层(gen_doc_variants 的 WORK=/work/docvariants)。

### 3.2 用例与跑法

7 个测试模块按文档章节组织，共 87 用例：

| 模块 | 文档章节 |
|---|---|
| test_doc_construct.py | 构造函数/Context Manager/__repr__ |
| test_doc_inference.py | __call__ Auto+零拷贝/异步 start·wait·wait_done·is_busy/确定性 |
| test_doc_scheduling.py | set_scheduling_params/核掩码严格性/priority/per-slot |
| test_doc_metadata.py | summary/benchmark/只读属性/compile_config |
| test_doc_ionarray.py | IONMemory/IONArrayDesc/IONArray/量化/布局/from_memory |
| test_doc_zerocopy.py | 零拷贝接入/check_input·output/链式 |
| test_doc_envvar.py | L2 Cache/日志级别/抢占 |

```bash
# 数据由 gen_doc_variants 产 + sync_to_board 拉回(见第五节)
DOC_DATA_DIR=/root/ssd/OELLM_Runtime/docvariant_data \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest tests/docvariant -q
```

实测：86 passed, 1 skipped(SCALE 输出在 S600 平台不可达，见第八节)。

---

## 四、接口一致性多维度套件(dimensions/)

把接口拆成 11 个正交维度，每个维度独立可跑（pytest -m dim5），堵住接口盲区。

| 维度 | 名称 | 验证什么 | 数据 |
|---|---|---|---|
| D1 | 算子/结构 | 10 个 backbone 推理三路一致 | 矩阵 |
| D2 | 张量形态 | rank 1D~5D + 8 shape 变体 | 矩阵 |
| D3 | IO 元数 | 多输入/多输出绑定顺序与数量 | 4族 |
| D4 | 推理路径 | 4 条路径对同输入 bit-identical | 4族 |
| D5 | 内存/对齐 | 零拷贝端到端 + padded/strided 往返 | 4族 |
| D6 | dtype/量化 | int8 输入 + SCALE 平台刻画 | 矩阵+rgb |
| D7 | 核数等价 | core1 vs core2 输出逐元素相等 | 4族 |
| D8 | 并发语义 | n_task 边界/交错/busy/超时 | 4族 |
| D9 | 错误路径 | 各类非法输入抛预期异常 | 4族 |
| D10 | 确定性 | 同输入 N 次逐元素相等 | 4族 |
| D11 | API 元数据 | summary schema + 只读属性 + repr | 4族 |

### 4 条推理路径(D4 核心)

同一个模型同一组输入，用 4 种方式推理，要求结果一致：

| 路径 | 入口 | 说明 |
|---|---|---|
| sync | model(inputs) | Auto 同步,from_numpy→BPU→output,自动 flush |
| async | start(inputs)+wait() | 异步 memcpy |
| zerocopy | IONArray(input_descs)+from_numpy+inputs[slot][idx]=ion+start(task_id)+wait+flush_invalidate+numpy | 零拷贝 ION,flush 是用户契约 |
| pipeline | Pipeline([model]).pipe(inputs) | 多模型零拷贝串联(Pipeline 为测试内部辅助类 tests/_pipeline.py,已从主包移除,底层用 IONArray.memory/from_memory,非公共 API) |

分层断言：同内存路径对(sync/async)要求 np.array_equal(bit-identical)；不同内存布局对(zerocopy/pipeline)要求 allclose。

### 共享基础设施

- _harness.py：复用 golden/harness.py，补模型加载(自动给多核模型设调度核)、保留原 dtype 取数、BPU 空闲探测、跨路径比对。
- _paths.py：4 条推理路径的统一执行器。
- _make_total_report.py：聚合各维度 CSV → consistency_total_report.json。
- conftest.py：bpu_free(开跑前探 BPU 没被占)、acc(session 级报告累加器)。

### D6 平台刻画结论

扫描全部矩阵模型 + 变异 hbm，确认 S600(Nash-m)工具链**总是在模型尾部插入 Dequantize**，故任何 HBM 的 IO 都是 float32(quantiType=NONE)——quantize()/dequantize() 的 SCALE 路径在本平台不可达(API 存在但无 HBM 触发)。这是平台事实，非测试缺口。

---

## 五、模型生成与闭环(generate/)

测试用的 HBM/golden 由**云端 docker**生成(板端无 torch/hb_compile)。docker 镜像：`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`，云主机 `chao.wu@120.48.157.2`，工作目录 `~/openexplore_test_cases`(只允许访问此目录)。

### 闭环命令(板端执行)

```bash
cd tests/generate
./run_cloud.sh            # 同步代码到云端 + docker 跑 build_matrix.py(--gpus all)
./sync_to_board.sh        # 拉 manifest+hbm+golden 回板端 golden_hbm_matrix/
```

### 两套生成流水线

| 流水线 | 脚本 | 产出 | 喂给 |
|---|---|---|---|
| 矩阵 | build_matrix.py | 10 backbone × rank 1D~5D × 8 变体 × 元数 × core,全 int16 | D1/D2/D6 |
| 文档变异 | gen_doc_variants.py | 15 编译规范(core/int8/多IO/rank/nv12/rgb/preempt/scaleout/chain) | docvariant/(两份文档细节) |

文档变异流水线单独跑(产 docvariants/ 目录)：

```bash
# 云端产
rsync -az tests/generate/gen_doc_variants.py chao.wu@120.48.157.2:~/openexplore_test_cases/generate/
ssh chao.wu@120.48.157.2 "docker run --rm --gpus all --entrypoint '' \
  -v ~/openexplore_test_cases:/work -e OETC_WORK=/work \
  registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  python3 /work/generate/gen_doc_variants.py"
# 拉回板端(docvariants/{hbm,golden,manifest_doc.json} → docvariant_data/)
DST=/root/ssd/OELLM_Runtime/docvariant_data
rsync -azm --include='*/' --include='*.hbm' --exclude='*' \
  chao.wu@120.48.157.2:~/openexplore_test_cases/docvariants/hbm/ "$DST/hbm/"
rsync -azm --include='*/' --include='meta.json' --include='data.npz' --exclude='*' \
  chao.wu@120.48.157.2:~/openexplore_test_cases/docvariants/golden/ "$DST/golden/"
scp chao.wu@120.48.157.2:~/openexplore_test_cases/docvariants/manifest_doc.json "$DST/"
```

> 旧的 4 族流水线(legacy_4family/)已废弃删除：其 manifest schema(family)与矩阵(backbone/ndim)不同，且功能被 docvariant(针对文档细节)+ build_matrix(矩阵)覆盖。

---

## 六、运行方法

### 前置条件

- 环境：pyCauchyKesai 装在 conda env(板端用 `pyCauchyKesai` env,v1.4.0,Python 3.12；或 `robotrea_python_runtime`,v1.2.0,Python 3.10)。**不要 PYTHONPATH=src**——src/ 下无编译好的 native .so，注入 src/ 会遮蔽已装 wheel 导致 import 失败。
- BPU 空闲：被 robotrea 服务占用时，需硬件的套件会明确 skip/退出，不伪造结果。
- 数据：板端对应数据目录已同步；缺数据时对应套件 skip。

### 跑文档细节(docvariant/)

```bash
DOC_DATA_DIR=/root/ssd/OELLM_Runtime/docvariant_data \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest tests/docvariant -q
```

### 跑接口一致性(dimensions/)

```bash
# 板端维度(D3~D11,默认 golden_hbm)
GOLDEN_DATA_DIR=/root/ssd/OELLM_Runtime/golden_hbm \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest tests/dimensions/ \
  -m "dim3 or dim4 or dim5 or dim7 or dim8 or dim9 or dim10 or dim11" -v
# 矩阵维度(D1/D2/D6,golden_hbm_matrix)
GOLDEN_DATA_DIR=/root/ssd/OELLM_Runtime/golden_hbm_matrix \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest \
  tests/dimensions/d1_ops_structure tests/dimensions/d2_tensor_shape tests/dimensions/d6_quantization -v
# 汇总
/root/ssd/miniconda3/envs/pyCauchyKesai/bin/python tests/dimensions/_make_total_report.py
```

### 跑量化刻画(golden/)

```bash
GOLDEN_DATA_DIR=/root/ssd/OELLM_Runtime/golden_hbm \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python tests/golden/run_characterization.py
```

### 跑纯内存(nash/,不需 BPU)

```bash
/root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest tests/nash/ -q
```

### 跑基础功能(test_cauchy_kesai.py)

需先指定模型(默认搜 /root/OELLM_Runtime/assets,该目录常为空)：

```bash
TEST_MODEL_PATH=/root/ssd/OELLM_Runtime/golden_hbm/hbm/<cid>.hbm \
  /root/ssd/miniconda3/envs/pyCauchyKesai/bin/python -m pytest tests/test_cauchy_kesai.py -q
```

---

## 七、报告产物

| 路径 | 内容 |
|---|---|
| tests/docvariant/REPORT.md | 文档细节测试报告(覆盖度/结果/平台限制) |
| tests/dimensions/report/consistency_total_report.json | 11 维度总报告 |
| tests/dimensions/report/d{1,2,4,5,6,7,10}_*.csv | 各维度明细 |
| tests/golden/report/{per_case,by_axis,summary}.{csv,json} | 量化刻画(cos/maxabs/latency) |

dimensions/report/ 与 golden/report/ 是测试自动生成的产物，删后重跑会重建。

---

## 八、平台限制与接口契约备忘

1. **S600(Nash-m)IO 恒 float32(quantiType=NONE)**：工具链总在尾部插 Dequantize。即便 model_output_type=int8 编译，输出仍 quantiType=0/float32。文档 IONArray 的 quantize/dequantize 的 SCALE 路径(quanti_type=1)在本平台无 HBM 可触发——API 存在且 NONE 路径已测(docvariant 86 passed),SCALE 为平台事实不可达。
2. **多核(core2)HBM 构造时已默认设 [0,1]**(前 N 核),可直接推理；错配核数在 set_scheduling_params 时抛 ValueError。单核宽松(掩码含≥1有效核即可),多核严格(恰好 N 个有效核)。
3. **S600 是 64 字节对齐**,padded 布局由 alignedByteSize > natural nbytes 判断；from_memory 的 byte_offset 必须对齐 BPU 步长(64)。
4. **抢占 254/255 需三前提**：编译时 max_time_per_fc(≥1000us) + 运行时 priority=254/255 + UNSET L2 Cache(import 前设 UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1,因 L2 Cache 与抢占互斥)。conv_preempt_c1 hbm 已含 max_time_per_fc,可测抢占运行时语义。
5. nv12/rgb 输入的**校准数据用 float32**(input_type_train=rgb 但 calib float32),runtime 才接 uint8 图像。gen_doc_variants 已固化此约定。
