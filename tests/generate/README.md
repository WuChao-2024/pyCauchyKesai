# tests/generate — torch → onnx → hbm 模型生成（云端）+ 测试闭环

本目录是**生成侧代码**：在云端 x86 服务器的 docker（torch/onnx/hb_compile 全在）里，从随机权重批量产出测试用的 ONNX、golden 真值、calib 数据、HBM、manifest。**不含任何权重/模型产物，只有代码**——产物由代码在云端现生成。

板端（arm64）只读产物跑测试，不参与生成。

---

## 一、闭环三条命令（板端执行）

```bash
cd /root/ssd/OELLM_Runtime/CauchyKesai_ProMax/pyCauchyKesai/tests/generate

# 1) 云端 docker 生成（同步本目录代码到云端 → 跑 build_matrix.py）
./run_cloud.sh                       # 全量；--dry-run 只看矩阵不编译；--limit N 调试

# 2) 把云端产物拉回板端
./sync_to_board.sh                   # → /root/ssd/OELLM_Runtime/golden_hbm_matrix/

# 3) 板端跑接口一致性测试
cd /root/ssd/OELLM_Runtime/CauchyKesai_ProMax/pyCauchyKesai
conda activate robotrea_python_runtime
GOLDEN_DATA_DIR=/root/ssd/OELLM_Runtime/golden_hbm_matrix PYTHONPATH=src pytest \
  tests/dimensions/d1_ops_structure tests/dimensions/d2_tensor_shape tests/dimensions/d6_quantization -v
```

---

## 二、两套生成流水线

| 流水线 | 主脚本 | 产出 | 给谁用 |
|---|---|---|---|
| **矩阵（主）** | `build_matrix.py` | 10 backbone × rank 1D~5D × 8 shape 变体 × IO 元数 × {core1,core2}，全 int16（≈317 HBM） | D1 算子 / D2 张量形态 / D6 dtype |
| **4 族（旧）** | `legacy_4family/run_4family.sh` | conv_stack/matmul_stack/transformer_block/mixed_io × 3 size × {int8,int16} × {core1,core2} = 48 HBM | D3/D4/D5/D7/D8/D9/D10/D11（P0） |

> 矩阵是自包含的一键脚本（`build_matrix.py` 单文件做完 torch→onnx→golden→calib→hb_compile→manifest）。
> 4 族是旧的 HDF5 多步流水线（gen_golden→convert_npy→gen_configs→compile→collect_manifest），板端 `/root/ssd/OELLM_Runtime/golden_hbm` 已有产物，通常无需重跑。

### 矩阵维度（`build_matrix.py` 顶部可改）

```
BACKBONES     = conv2d, conv3d, depthwise_conv2d, matmul, transformer,
                elementwise, softmax, layernorm, reshape_transpose, concat_split
rank(ndim)    = 1d, 2d, 3d, 4d, 5d   （每个 backbone 适用的 rank 不同）
SHAPE_VARIANTS= aligned, odd, bigc_smallspatial, smallc_bigspatial,
                tiny, large_batch, boundary_reduce, nonpow2_channel
ARITY         = 1i1o（多数）; elementwise=2i1o/3i1o; concat_split=2i1o/3i1o/1i2o/2i2o
CORES         = 1, 2（S600 双核）
PRECISION     = int16
```

`python3 build_matrix.py --dry-run` 本地可跑（不 import torch），枚举交叉积并计数。

---

## 三、云端产物布局（docker 内 `/work` = 云端 `~/openexplore_test_cases`）

```
onnx/<model_id>.onnx                 torch 导出, opset 14, 静态 shape
golden/<model_id>/data.npz           前 10 组 in+out 对（key: in_XX_<name> / out_XX_<name>）
golden/<model_id>/meta.json          shape/dtype/n_golden/n_calib/seed/backbone/...
calib/<model_id>/[<input>/]sample_XX.npy   50 份校准数据
configs/<cid>.yaml                   hb_compile 配置（自动生成）
hbm/<cid>/<cid>.hbm                  板端推理模型 + _advice.csv + _quant_info.json
manifest.json                        全部成功条目的索引（测试数据驱动入口）
compile_errors.json                  失败条目（不掩盖）
```

`cid = <model_id>__<precision>__core<N>`，例如 `conv3d_5d_aligned_1i1o__int16__core1`。

---

## 四、前置条件

- **SSH**：板端能免密 `ssh chao.wu@120.48.157.2`（云端 `~/openexplore_test_cases` 可写）。
- **docker 镜像**：`registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0`（云端已拉取；ENTRYPOINT 是 `/bin/bash`，跑 python 不需额外包装）。
- **GPU**：`hb_compile` 校准阶段**必须 `--gpus all`**，否则 SIGSEGV（脚本已带）。
- **平台**：`march=nash-p`（S600），64 字节对齐。换 S100/S100P 改 `build_matrix.py` 里 yaml 的 `march`。

---

## 五、测试侧如何消费

板端测试套件 `tests/dimensions/` 是**数据驱动**的——只读 `GOLDEN_DATA_DIR` 下的 `manifest.json` + `hbm/<cid>.hbm` + `golden/<model_id>/{data.npz,meta.json}`：

- D1/D2/D6（矩阵维度）→ `GOLDEN_DATA_DIR=golden_hbm_matrix`
- D3/D4/D5/D7/D8/D9/D10/D11（P0）→ 默认 `golden_hbm`（4 族，manifest 有 `family` 字段）

> 两份 manifest schema 不同（矩阵用 `backbone/ndim/shape_variant`，4 族用 `family`），**不能混用同一目录**，所以分两个数据目录、两次 pytest。
