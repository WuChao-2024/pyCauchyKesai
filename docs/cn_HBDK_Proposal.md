# HBDK 编译方案 (cn_HBDK_Proposal)

本文给出一组用 HBDK 工具链把浮点模型 (ONNX) 编译成上板 `.hbm` 的推荐 yaml 文件配置，覆盖 pyCauchyKesai 实际会遇到的四类场景：FeatureMap 通用 IO、图像色彩/排布、NV12 摄像头直输入、以及hbdk高带宽优化。

### 编译阶段产物

`hb_compile` 内部：`parse → optimize → calibrate → quantize(.bc) → compile(.hbm)`，典型产物：

```
*_original_float_model.onnx     # 原始浮点
*_optimized_float_model.onnx    # 算子优化后浮点
*_calibrated_model.onnx         # 校准后（带 scale）
*_quantized_model.bc            # HBIR 定点 (hbdk4 load/convert 的中间态)
*.hbm                           # 上板模型 ← 喂给 CauchyKesai
hb_compile.log                  # 编译日志
```

---

## 一、全 FeatureMap 的输入输出 + No Padding


### hb_compile yaml

```yaml
model_parameters:
  onnx_model: 'model.onnx'
  march: 'nash-p'                       # S600: nash-p, S100: nash-e, S100P: nash-m
  output_model_file_prefix: 'model_nashp_featuremaps'
  working_dir: './model_output'

input_parameters:
  input_type_train: 'featuremap;featuremap;'        # 浮点 featuremap，不做色彩转换
  input_type_rt:    'featuremap;featuremap;'        # 上板也是 featuremap

calibration_parameters:
  cal_data_dir: './calib'            
  quant_config:
    model_config:
      all_node_type: 'int16'          

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
  extra_params:                          # ← no_padding 在这里
    input_no_padding: True
    output_no_padding: True
```

**约束**（来自官方 `convert.md`）：
- `input_no_padding` **只对非图像输入**生效；图像输入（pyramid/resizer）不能去 padding。
- `output_no_padding` 对**所有输出**生效。
- 两者可同时配置：`extra_params: {input_no_padding: True, output_no_padding: True}`。

### hbdk4 等价写法

```python
import onnx
from hbdk4.compiler.onnx import export
from hbdk4.compiler import convert, compile, link, March

m = export(onnx.load("model.onnx"), name="llm")
m = convert(m, "nash-p")                                   # 转 backend IR
hbm = compile(
    m, "llm_fp_nopad.hbm", "nash-p",
    opt=2, jobs=8,
    balance=100,                                           # latency
    input_no_padding=True,                                 # ← hbdk4 直接暴露
    output_no_padding=True,
)
```

> pyCauchyKesai 消费：`CauchyKesai("llm_fp_nopad.hbm")` → `summary()` 中 `inputs[].quantiType == NONE`、`dtype == float32`、`alignedByteSize == prod(shape)*4`。

---

## 二、NCHW / NHWC + RGB / BGR 输入

**场景**：图像分类/检测/分割。训练用 NCHW·RGB（PyTorch 默认），上板想用 NHWC 或不同色彩序。

### 关键事实（`convert.md` + `hbdk4` 实测）

- 预处理节点（mean/scale/色彩转换）**只支持 NHWC 输入**。原始模型若是 NCHW，工具会自动转成 NHWC（不影响精度/性能）。
- 当 `input_type_rt ∈ {rgb, bgr}`（NHWC/NCHW）时，上板模型的**输入 dtype 被固化为 `int8`（si8，需 -128 起算）**，这一步在 API 内自动完成。
- `input_type_train` 与 `input_type_rt` 不必一致；工具按组合自动插入转换节点。合法组合表（Y=支持）：

  | input_type_train ＼ input_type_rt | nv12 | yuv444 | rgb | bgr | gray | featuremap |
  |---|---|---|---|---|---|---|
  | rgb  | Y | Y | Y | Y | N | N |
  | bgr  | Y | Y | Y | Y | N | N |
  | yuv444 | Y | Y | N | N | N | N |
  | gray | N | N | N | N | Y | N |
  | featuremap | N | N | N | N | N | Y |

### hb_compile yaml（RGB/NCHW 训练 → 上板 RGB/NHWC）

```yaml
model_parameters:
  onnx_model: 'resnet50.onnx'
  march: 'nash-p'
  output_model_file_prefix: 'resnet50_rgb_nhwc'

input_parameters:
  input_type_train:   'rgb'              # 训练时的色彩序与排布
  input_layout_train: 'NCHW'
  input_type_rt:      'rgb'              # 上板色彩序（可与 train 不同）
  input_shape: '1x3x224x224'
  mean_value:  '123.675 116.28 103.53'   # 通道均值，空格分隔
  scale_value: '0.01712475 0.017507 0.01742919'   # scale 与 std 二选一，scale=1/std

calibration_parameters:
  cal_data_dir: './calib_rgb'

compiler_parameters:
  compile_mode: 'latency'
  optimize_level: 'O2'
```

### hbdk4 等价写法（手动插 transpose / image_preprocess）

`Argument` overlay 提供两个相关方法（实测签名）：

```python
# insert_transpose(permutes) —— 改输入/输出维序（NCHW↔NHWC 等）
func.inputs[0].insert_transpose([0, 2, 3, 1])     # NCHW → NHWC

# insert_image_preprocess(mode, divisor, mean, std, is_signed, bit_width, image_layout)
#   整数图(必须 NHWC) → float：Output = ((ColorConvert(ToUnsigned(in), mode)/divisor) - mean)/std
func.inputs[0].insert_image_preprocess(
    mode="yuvbt601full2rgb",            # skip/yuvbt601full2rgb/...2bgr/video2rgb/video2bgr
    divisor=255,                        # None=按 bit_width 推断：int8→256, int16→65536
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    is_signed=True,
    bit_width=8,
    # image_layout=...                  # 可选布局提示
)
```

> 注意：`insert_*` 必须在 **`convert` 之前**调用，否则可能不被某些转换 pass 执行。`mode != "skip"` 时要求输入 C 维 == 3。

---

## 三、YUV420SP (nv12) 图像输入

**场景**：摄像头/ISP 直出 NV12，想让模型直接吃 NV12，省掉 ARM 侧的色彩转换。

### hb_compile yaml

```yaml
model_parameters:
  onnx_model: 'yolov8.onnx'
  march: 'nash-p'
  output_model_file_prefix: 'yolov8_640x640_nv12'

input_parameters:
  input_type_train:   'rgb'
  input_layout_train: 'NCHW'
  input_type_rt:      'nv12'             # ← 上板直接吃 NV12
  input_shape: '1x3x640x640'
  input_space_and_range: 'regular'       # nv12 专属：regular=[0,255] / bt601_video=[16,235]
  # mean/scale 仍可配，工具会把 nv12→rgb 转换 + 归一化一起插进图里

calibration_parameters:
  cal_data_dir: './calib_nv12'

compiler_parameters:
  compile_mode: 'latency'
  input_source:                          # 可选：ddr / pyramid / resizer
    data: pyramid                        # nv12/gray 默认即 pyramid；想用硬件缩放改 resizer
```

**NV12 约束**：
- `input_type_rt: nv12` 时，输入 shape 的 H/W **不能出现奇数**。
- `bt601_video` 仅在 `input_type_train ∈ {rgb, bgr}` 时可配。

### `input_source`：pyramid vs resizer vs ddr（实测语义）

| 来源 | 含义 | 输入 shape |
|---|---|---|
| `ddr` | 数据来自普通内存（默认非 nv12/gray） | 静态 |
| `pyramid` | 处理器固定硬件 pyramid（nv12/gray 默认） | **静态**，编译期固定 |
| `resizer` | 硬件 resizer，支持 ROI 缩放（仅 nv12/gray） | **动态**，运行时由用户提供 shape |

### hbdk4 等价写法

`insert_image_convert` 把单个图像输入拆成 NV12 的 Y + UV（实测 mode：`nv12` / `gray` / `nv12_yh12` / `nv12_yh10`，后两者处理 16-bit Y 取高 12/10 位）：

```python
func = m[0]
y, uv = func.inputs[0].insert_image_convert("nv12")
# y  : tensor<1xHxWx1xui8>     ← runtime: images_y
# uv : tensor<1xH/2xW/2x2xui8> ← runtime: images_uv

# 想用硬件 ROI 缩放（对应 resizer 源）：
y, uv, roi = func.inputs[0].insert_roi_resize(
    mode="nv12", interp_mode="bilinear", pad_mode="constant", pad_value=(0, -128),
)
```

> pyCauchyKesai 消费：`model([images_y, images_uv])`——两个 `uint8` 输入，顺序固定（Y 在前）。`input_names` 形如 `['images_y', 'images_uv']`。

---

## 四、解锁 HBDK 高带宽选项

**目标**：在 S600 上精确控制 DDR 带宽预算、用满 L2M、张量并行，把延迟/吞吐调到最优。

### 头号旋钮：`_ddr_bandwidth_gb`（DDR 带宽预算，**仅 hbdk4 可设**）

控制 DDR 带宽的真正开关不是 yaml 里的 `compile_mode`/`balance`，而是 **Module 层属性 `_ddr_bandwidth_gb`**（底层 mlir 属性 `hbdk.ddr_bw_gb`，单位 GB/s）。它告诉编译器「这块板子实际可用多少 GB/s 的 DDR 带宽」，编译器据此决定把多少算子换成更省 DDR 的指令排布。

**关键事实（容器内反射 + 云端实验脚本核对）**：
- 它是 `hbdk4` Module 对象的 property（`module_modifier.py`：`_ddr_bandwidth_gb` 读写 mlir `FloatAttr` `hbdk.ddr_bw_gb`），**必须在 `compile()` 之前**赋值才生效。
- `hb_compile` / `horizon_tc_ui` **不暴露这个旋钮**——参考脚本注释原文："Module 层属性 (compile 前设置, horizon_tc_ui 不设这些)"。也就是说 yaml 流程设不了，**必须走 hbdk4**。
- 单位 GB/s；地平线参考脚本默认与上限取 **80.0**（S600 BPU 标称 DDR 带宽量级）。

### 因此必须走「两段式」编译

因为 `_ddr_bandwidth_gb` 只在 hbdk4 可设，标准做法是：先 `hb_compile` 产 `.bc`，再用 hbdk4 设属性后重编 `.hbm`：

```python
# Stage A: hb_compile -c config.yaml  → 产出 *_quantized_model.bc（量化好的 HBIR）
# Stage B: 在 .bc 上设 DDR 带宽，再编译成 .hbm
from hbdk4.compiler import load, compile, hbm_perf

m = load("yolo26x_..._quantized_model.bc")
m._ddr_bandwidth_gb = 80.0                    # ← 真正的 DDR 带宽旋钮，compile 前设
compile(
    m, 
    "out.hbm", 
    "nash-p",
    opt=2, 
    jobs=16, 
    balance=100,              # balance 保持 100(latency)；DDR 由上面那行管
    core_num=4,               # core_num 作为 kwarg 透传给 compile (见下)
    max_l2m_size=0,           # 想用 L2M 削 DDR 改这里
    input_no_padding=True, 
    output_no_padding=True,
)
hbm_perf("out.hbm")                           # 产 perf HTML，看实际 DDR/延迟曲线
```

### 其余旋钮（compile 参数 / yaml 通用）

- **`max_l2m_size`（仅 S600）**：L2M 片上缓存削 DDR 流量。`0`=不用；`N`=最多用 N 字节；`None`=编译器自动分配（最大化「单位 L2M 的 DDR 削减」）。yaml：`max_l2m_size: None` 或 `[0, 24x1024x1024]`（≤24 MiB）。⚠️ 板端评测前 `export HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6`（4 核各 6 MiB），否则收益测不准。
- **`core_num`（S600 张量并行）**：hbdk4 里作为 kwarg 透传给 `compile()`（签名里没列但 `**kwargs` 接收）；yaml 里是 `core_num: N`——**S600 (nash-p) 可取 `[1, 2, 3, 4]` 任意值**（权威来源 `horizon_tc_ui` 的 `core_num_range`，OE 文档 `convert.md` 写的 [1,2] 已过时）；S100/S100P 固定 1。单模型张量并行切 N 核，运行时 `set_scheduling_params([0..N-1])` 且核数须等于编译核数。S600 物理 4 核（`num_cores=4`），`core_num=N` 占 N 核张量并行，余 4−N 核留多任务并发（`n_task` 多 slot）。
- **辅助**：`opt`(O0/O1/O2,默认 O2)、`jobs`(编译并发进程数)、`advice`(打印实际耗时超理论值 N us 的算子 + padding 比例)、`cache_mode`/`cache_path`(disable/enable/force_overwrite,二次编译加速)。

### 实测效果（云端 YOLO S600 benchmark 摘录）

`yolov8x` 640 输入，多核 + L2M 对延迟的影响：

| 配置 | 延迟 | 相对 1core |
|---|---|---|
| 1core nocache | 7.33ms | — |
| 4core nocache | 5.82ms | -20.6% |
| 4core cache(L2M) | 4.89ms | **-33.3%** |

> 结论：多核 + L2M 显著降延迟；1280 大模型在 1core/2core + cache 时 L2 OOM（>10MB/core 需求）。DDR 带宽优化看 `hbm_perf` 的 DDR 曲线，板端用 `hrt_model_exec perf`（多核 `--core_id 0-3`）复核，pyCauchyKesai 侧 `model.benchmark()`。

---

## 附：pyCauchyKesai 看到的字段 ↔ 编译配置

| `summary()` 字段 | 由编译配置决定 |
|---|---|
| `dtype` (输入) | `input_type_rt`：rgb/bgr/nv12/yuv444→`int8`(si8)；gray→`uint8`；featuremap→`float32` |
| `quantiType` / `scale` | S600 工具链总在尾部插 Dequantize → IO 全 `NONE`（见下） |
| `alignedByteSize` | 对齐(32/64) + `extra_params.input/output_no_padding` |
| `stride` | 对齐后的行步长（no_padding 时 == 紧凑步长） |
| 输入个数 / 名称 | nv12→2 个 (`*_y`, `*_uv`)；`insert_split/transpose` 会改变 IO 数与序 |

**平台事实（D6 结论）**：S600 (nash-p) 工具链**总是在模型尾部插入 Dequantize**，所以任何 HBM 的 IO 都是 `float32`（`quantiType=NONE`）——`IONArray.quantize()/dequantize()` 的 SCALE 路径在本平台**不可达**（API 存在但无 HBM 触发）。这是平台事实，非缺陷。

---

## 附：hbdk4 最小编译示例（ONNX → HBM）

```python
import onnx
from hbdk4.compiler.onnx import export
from hbdk4.compiler import convert, compile, March

# 1. ONNX → HBIR (mlir Module)
m = export(onnx.load("model.onnx"), name="model")

# 2. （可选）编译期改 IO：必须先于 convert
# func = m[0]; func.inputs[0].insert_image_convert("nv12")

# 3. HBIR → backend IR（定点/校准由 hb_compile 完成；纯 hbdk4 路径需自行量化）
m = convert(m, "nash-p")

# 4. 编译为 HBM（方法一：直接 hbm）
compile(m, "model.hbm", "nash-p", opt=2, jobs=8, balance=2,
        max_l2m_size=None, output_no_padding=True)

#   方法二：先 hbo 再 link（便于多模型打包）
# hbo = compile(m, "model.hbo", "nash-p")
# link([hbo], "model.hbm")
```

> 日常建模用 `hb_compile -c config.yaml` 即可；上面的 hbdk4 路径适合需要 `insert_*` 改图或脚本批量编译（如 pyCauchyKesai 的矩阵测试 `build_all.py`）的场景。
