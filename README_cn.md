![](sources/imgs/CauchyKesai.jpeg)

## Quick Start

```python
import numpy as np
from libpyCauchyKesai import CauchyKesai, __version__
print(f"Cauchy Kesai: {__version__}")

m = CauchyKesai("/app/model/basic/resnet18_224x224_nv12.hbm")

m.s()   # Model Summarys

m.t()   # Model Dirty Test

y = np.random.rand(1,224,224,1).astype(np.uint8)
uv = np.random.rand(1,112,112,2).astype(np.uint8)

o = m([y, uv])[0]
print(f"{o.shape = }, {o.dtype = }")
```

## Constructor

### import
```python
import libpyCauchyKesai as ξ
```

### Informations
```python
print(f"__version__: {ξ.__version__}, __date__: {ξ.__date__}, __author__: {ξ.__author__}", )
print("__doc__", ξ.__doc__) 
```

### Model Init

```python
CauchyKesai(
    model_path: str, 
    n_task: int = 1, 
    model_cnt_select: int = 0
    )
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | — | `*.hbm` Model File Path. |
|   `n_task`   | `int` | `1` | 并行推理任务数 |
| `model_cnt_select` | `int` | `0` | 模型选择索引（用于多模型场景） |

1. 用于初始化模型.
2. n_task 参数用于在对象的内部初始化多组完整的推理环境, 包括开辟对应的输入输出Tensor的内存, 对应的任务初始化句柄, 可以在多线程任务或者异步任务中使用不同的n_task来推理, 避免内存踩踏等异常现象, 也避免边运行边推理时malloc内存的低效行为.
3. model_cnt_select 参数设计用于pack模型的推理, 默认选择第0个模型. C/C++接口是支持pack的多个模型一次加载反复推理的, 这里我就不写了, 太累了。


### Print Model Summarys

```python
s()
```

```bash

```

打印模型摘要信息

### Fast Test
```python
t()
```
---

### 2. `.t()`
- **功能**: 脏运行一次模型, 并打印性能数据（如耗时、吞吐量等）
- **返回值**: 无
- **示例**:
  ```python
  model.t()
  ```

---

### 3. `.start(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0)`
- **功能**: 启动一次异步推理任务
- **参数**:
  | Parameter | Type | Default | Description |
  |-----------|------|---------|-------------|
  | `inputs` | `List[np.ndarray]` | — | 输入张量列表（每个元素是一个 numpy 数组） |
  | `task_id` | `int` | `0` | 任务 ID（用于区分多个并发任务） |
  | `priority` | `int` | `0` | 任务优先级（数值越大优先级越高） |
- **返回值**: 无（需调用 `.wait()` 获取结果）

---

### 4. `.wait(task_id: int = 0) -> List[np.ndarray]`
- **功能**: 等待指定任务完成, 并返回推理结果（零拷贝优化）
- **参数**:
  | Parameter | Type | Default | Description |
  |-----------|------|---------|-------------|
  | `task_id` | `int` | `0` | 要等待的任务 ID |
- **返回值**: `List[np.ndarray]` - 输出张量列表

---

### 5. `.inference(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0) -> List[np.ndarray]`

ibpyCauchyKesaiS100FeaturemapsTools.CauchyKesai, inputs: List[numpy.ndarray], task_id: int = 0, priority: int = 0) -> List[numpy.ndarray]

- **功能**: 安全调用：依次执行 `check + start + wait`, 适合一次性推理
- **参数**:
  | Parameter | Type | Default | Description |
  |-----------|------|---------|-------------|
  | `inputs` | `List[np.ndarray]` | — | 输入张量列表 |
  | `task_id` | `int` | `0` | 任务 ID |
  | `priority` | `int` | `0` | 任务优先级 |
- **返回值**: `List[np.ndarray]` - 输出张量列表

---

### 6. `__call__(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0) -> List[np.ndarray]`
- **功能**: 支持对象直接调用语法, 等价于 `.inference(...)`
- **示例**:
  ```python
  outputs = model([input1, input2])
  ```

---

## 🧠 内部成员变量（仅供了解）

| 变量名 | 类型 | 描述 |
|--------|------|------|
| `model_path_` | `std::string` | 模型路径 |
| `n_task_` | `int32_t` | 最大并发任务数 |
| `is_infer` | `std::vector<int>` | 每个任务是否正在推理 |
| `task_handles` | `std::vector<hbUCPTaskHandle_t>` | 任务句柄数组 |
| `packed_dnn_handle`, `dnn_handle` | `hbDNN*` | DNN 相关句柄 |
| `input_properties`, `output_properties` | `hbDNNTensorProperties` | 输入/输出张量属性 |
| `inputs_shape`, `outputs_shape` | `std::vector<std::vector<size_t>>` | 输入/输出形状 |
| `inputs_dtype`, `outputs_dtype` | `std::vector<std::string>` | 数据类型（如 float32, uint8） |






```

### Model init
```python
m1 = ξ.CauchyKesai("model1.hbm")
m2 = ξ.CauchyKesai("model2.hbm")
```

### Model INference
```python
output = m1([input1, input2, input3, ...])
```

## 接口说明



## Combine与Runtime指南

### FeatureMaps without PreProcess

全部使用featuremap, 不要使用其他, 这样hbm模型的前后处理行为和ONNX是完全一致的.

```yaml
model_parameters:
  onnx_model: 'onnx_name_BPU_ACTPolicy_TransformerLayers'
  march: "nash-e"
  layer_out_dump: False
  working_dir: 'bpu_model_output'
  output_model_file_prefix: 'BPU_TransformerLayers'
input_parameters:
  input_name: "name1;name2;name3;"
  input_type_rt: 'featuremap;featuremap;featuremap;'
  input_layout_rt: 'NCHW;NCHW;NCHW;'
  input_type_train: 'featuremap;featuremap;featuremap;'
  input_layout_train: 'NCHW;NCHW;NCHW;'
  norm_type: 'no_preprocess;no_preprocess;no_preprocess;'
calibration_parameters:
  cal_data_dir: 'cal_data1;cal_data2;cal_data3;'
  cal_data_type: 'float32;float32;float32;'
  calibration_type: 'default'
  optimization: set_all_nodes_int16
compiler_parameters:
  extra_params: {'input_no_padding': True, 'output_no_padding': True}
  jobs: 16
  compile_mode: 'latency'
  debug: False
  optimize_level: 'O2'
```  


## 替换OpenExplore的头文件和动态库

### Nash-e / Nash-m

从最新版本的OpenExplore包的文件中获得以下文件

```bash
./samples/ucp_tutorial/deps_aarch64/ucp
├── bin
├── include
├── lib
└── plugin
```

动态库

```bash
./samples/ucp_tutorial/deps_aarch64/ucp/lib
├── libdnn.so
├── libhb_arm_rpc.so
├── libhbdsp_plugin.so
├── libhbhpl.so
├── libhbrt4.so
├── libhbtl_ext_dnn.so
├── libhbtl.so
├── libhbucp.so
├── libhbvp.so
├── libhlog_wrapper.so
└── libperfetto_sdk.so
```

头文件

```bash
./samples/ucp_tutorial/deps_aarch64/ucp/include/
└── hobot
    ├── dnn
    ├── hb_ucp.h
    ├── hb_ucp_status.h
    ├── hb_ucp_sys.h
    ├── hpl
    ├── plugin
    └── vp
```

删除板子上自带的动态库并更新

```bash
cd /usr/hobot/lib/

rm libdnn.so libhbdsp_plugin.so libhbrt4.so libhbtl.so libhbvp.so libperfetto_sdk.so libhb_arm_rpc.so  libhbhpl.so libhbtl_ext_dnn.so libhbucp.so libhlog_wrapper.so

cp ./lib/* /usr/hobot/lib/
```


删除板子上自带的头文件并更新
```bash
rm -rf /usr/include/hobot
cp -r ./include/hobot /usr/include/
```

查看动态库中的版本号
```bash
strings /usr/hobot/lib/libhbucp.so | grep SO_VERSION
```
```bash
SO_VERSION = (3U).(7U).(4U)
```


查看头文件中的版本号
```bash
export file="/usr/include/hobot/hb_ucp.h"
eval $(grep -e 'HB_UCP_VERSION_MAJOR' -e 'HB_UCP_VERSION_MINOR' -e 'HB_UCP_VERSION_PATCH' $file | \
      sed -E 's/#define HB_UCP_VERSION_(MAJOR|MINOR|PATCH)[^0-9]*([0-9]+).*/VERSION_\1=\2/')
echo -e "\n$file Version: $VERSION_MAJOR.$VERSION_MINOR.$VERSION_PATCH\n"
```
```bash
/usr/include/hobot/hb_ucp.h Version: 3.7.4
```

## pyCauchyKesai编译方法

下载项目

```bash
git clone https://github.com/WuChao-2024/pyCauchyKesai.git
cd pyCauchyKesai/Nash  # Optional
cd pyCauchyKesai/Bayes # Optional
```

编译和安装

```bash
mkdir -p build && cd build
cmake .. 
make -j3
cp libpyCauchyKesai.so /usr/lib/python3.10/
```

测试导入是否正常

```bash
python3 -c "from libpyCauchyKesai import __version__ ;print(__version__)"
```

如果你想使用其他Python解释器来使用这个模块, 可以在您的Python解释器中找到这样的路径, 将`*.so`文件拷贝到这样的路径中即可.

```bash
python3 -c "import os;print(os.__file__)"
```

```bash
/root/ssd/miniconda3/envs/torch/lib/python3.10/os.py
```

numpy库版本依赖
```bash
pip install numpy==1.26.4
```

## 常见问题

1. 以下报错是传入了torch.tensor导致的, pybind11 在调用函数前自动进行的类型检查失败，它在进入 C++ 函数之前就抛出了异常.
```bash
TypeError: __call__(): incompatible function arguments. The following argument types are supported:
    1. (self: libpyCauchyKesai.CauchyKesai, inputs: List[numpy.ndarray], task_id: int = 0, priority: int = 0) -> List[numpy.ndarray]
```

2. ImportError: /root/miniconda3/envs/rdt/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/hobot/lib/libhbucp.so)
这个错误是一个典型的 C++ 标准库版本不兼容 问题，升级conda环境的依赖即可
```bash
conda install libstdcxx-ng -c conda-forge
```


# 声明
## 所有源代码均开源, 使用前请务必保证您对程序有足够的认识. 本接口仅供社区开发者个人临时调试使用, 不保证其功能正确性, 作者不对任何错误和后果负责!
## 作者不对任何错误和后果负责!
## 作者不对任何错误和后果负责!
## 作者不对任何错误和后果负责!
