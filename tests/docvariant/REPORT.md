# pyCauchyKesai 文档细节测试报告

针对 `docs/cn_CauchyKesai_Auto.md` 与 `docs/cn_CauchyKesai_with_IONArray.md` 两份 API 文档,
设计变异编译规范的 HBM 模型,驱动一组覆盖文档全部细节的接口测试。**只测不改**(被测代码 src/ 与文档 docs/ 未改动)。

## 1. 闭环流程

1. 云端 step1:`tests/generate/gen_doc_variants.py` 在云端 docker 内,torch 导出 onnx + 生成 calib/golden + 变异 YAML,产出 15 个变异工作目录。
2. 云端 step2:docker 内逐个 `hb_compile` 产出不同编译规范的 HBM。
3. scp:扁平化 hbm + golden + manifest 拉回本地 `/root/ssd/OELLM_Runtime/docvariant_data/`。
4. 本地测试:`tests/docvariant/` 套件(87 用例)以 pyCauchyKesai v1.4.0 跑。

## 2. 云端变异 HBM(12/15 成功)

| 变异 spec | 编译规范 | 覆盖文档细节 | 状态 |
|---|---|---|---|
| conv_core1 | conv2d int16 core1 | 基线:构造/summary/__call__/异步/IONArray 通用 | ok |
| conv_core2 | core2 | set_scheduling_params 严格性/构造默认[0,1]/core 等价 | ok |
| conv_core4 | core4 | 核数=4 边界 | ok |
| conv_int8_c1 | int8 | int8 量化 | ok |
| concat_2i1o | 2 输入 | inputs 多 idx 绑定/顺序 | ok |
| concat_1i2o | 2 输出 | outputs 多 idx/wait 多 ndarray | ok |
| concat_2i2o | 2 输入 2 输出 | mixed_io 交叉绑定 | ok |
| matmul_1d | 1D | rank/IONArrayDesc.ndim | ok |
| matmul_3d | 3D | rank 变体 | ok |
| matmul_scaleout_c1 | model_output_type=int8 | SCALE 输出可达性验证 | ok |
| chain_a / chain_b | matmul [1,32] | 多模型零拷贝链式 | ok |
| conv_nv12_c1 | nv12(y+uv) | 文档「快速开始」nv12 双输入 | 云端不可达 |
| conv_rgb_c1 | rgb uint8 | quantize 写侧量化 | 云端不可达 |
| conv_preempt_c1 | max_time_per_fc | 抢占 254/255 | 云端不可达 |

## 3. 测试结果

`pytest tests/docvariant -q` → **85 passed, 2 skipped, 0 failed**(1.39s)。

| 测试模块 | 文档章节 | 用例数 |
|---|---|---|
| test_doc_construct.py | 构造函数/Context Manager/__repr__ | 10 |
| test_doc_inference.py | __call__ Auto/异步 start·wait·wait_done·is_busy/确定性 | 14 |
| test_doc_scheduling.py | set_scheduling_params/核掩码严格性/priority/per-slot | 14 |
| test_doc_metadata.py | summary/benchmark/只读属性/compile_config | 12 |
| test_doc_ionarray.py | IONMemory/IONArrayDesc/IONArray/量化/布局/from_memory | 18 |
| test_doc_zerocopy.py | 零拷贝接入/check_input·output/链式 cookbook | 10 |
| test_doc_envvar.py | L2 Cache/日志级别/抢占 | 7 |

2 skipped:scale_out 的 SCALE 路径(S600 IO 恒 NONE)、抢占(preempt hbm 云端不可达)。

## 4. 平台/工具链限制发现(测试揭示,非被测代码 bug)

1. **S600/nash-p 工具链总在尾部插 Dequantize** → 任何 HBM 的 IO 都是 float32(quantiType=NONE)。文档 IONArray 介绍的 `quantize`/`dequantize` 的 **SCALE 路径(quantiType=1)在本平台无 HBM 可触发**(model_output_type=int8 编译出的 hbm 仍 NONE)。API 存在且 NONE 路径已测,SCALE 为平台事实不可达,与 cn_TestCases.md D6 结论一致。
2. **nv12/rgb hbm 在纯云端 docker 不可产**:hb_compile 校准阶段卡在 `Reset batch_size=1 and execut`(OE 对图像校准数据格式要求严格,纯 torch 随机 uint8 不满足)。文档「快速开始」的 nv12 m([y,uv]) 双输入细节需真实图像校准数据或现网 hbm 才能端到端测。
3. **max_time_per_fc 抢占 hbm 不可产**:`hmct.utility.tempdir` import 失败(v3.7.0 docker 内 max_time_per_fc 触发的序列化路径有 bug)。文档 priority 254/255 抢占的编译前提在本工具链不可达;抢占运行时语义(priority 设 254/255 不抛)用普通 hbm 已部分覆盖。

> 以上 3 点是文档「高可用」需面对的真实约束:文档介绍了 nv12/SCALE/抢占,但本平台/工具链部分不可端到端触发。建议文档标注平台可达性(本次不改文档)。

## 5. 文档一致性结论

85 个用例验证的接口契约与两份文档**完全一致**,包括:
- 构造异常(RuntimeError)/n_task 钳位/model_cnt_select 钳位/Context Manager/__repr__ 格式
- __call__ Auto 校验(ValueError 数量/dtype/shape)/零拷贝档/check 一致性
- 异步状态机(idle/infering/waiting):busy 重入/空 wait/重复 wait/越界 IndexError 全部符合
- set_scheduling_params:单核宽松/多核严格恰好 N/任意子集/[]CORE_ANY 多核推理被拒/priority 边界
- summary/benchmark 字段齐全 + 可 json.dumps;s()/t() 返 None
- ION 三层原子(IONMemory/IONArrayDesc/IONArray)构造/计算方法/from_memory 偏移隔离/越界
- 零拷贝 == Auto;check_input/output 返 bool;链式 A→B 同物理内存
- envvar:setdefault 默认 6:6:6:6/UNSET 关闭/日志默认 6/可覆盖

未发现文档与实现的契约偏差。

## 6. 复现命令

云端产 hbm(在板端执行):
```bash
cd tests/generate && rsync -az gen_doc_variants.py chao.wu@120.48.157.2:~/openexplore_test_cases/generate/
ssh chao.wu@120.48.157.2 "docker run --rm --gpus all --entrypoint '' -v ~/openexplore_test_cases:/work \
  -e OETC_WORK=/work registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0 \
  python3 /work/generate/gen_doc_variants.py"
```
拉回本地 + 测试:
```bash
DOC_DATA_DIR=/root/ssd/OELLM_Runtime/docvariant_data  # rsync 云端 docvariants/{hbm,golden,manifest_doc.json} 到此
conda activate pyCauchyKesai
DOC_DATA_DIR=/root/ssd/OELLM_Runtime/docvariant_data pytest tests/docvariant -q
```
