# pyCauchyKesai
RDK Tools



以下是根据你提供的 C++ 类 `CauchyKesai` 和其 pybind11 接口生成的 **英文版接口文档（API Documentation）**，适用于技术文档、SDK 说明或开发者手册。

---

# 📘 `CauchyKesai` API Reference (English)

The `CauchyKesai` class is a C++ implementation for loading and running BPU models, exposed to Python via `pybind11`. It supports multi-task inference, asynchronous execution, performance profiling, and safe model interaction.

---

## 🔧 Constructor

```python
CauchyKesai(model_path: str, n_task: int = 1, model_cnt_select: int = 0)
```

### Parameters:

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | `str` | — | Path to the BPU model file (e.g., `.hbmodel`) |
| `n_task` | `int` | `1` | Number of concurrent tasks allowed |
| `model_cnt_select` | `int` | `0` | Model index selector (used in multi-model scenarios) |

---

## 📚 Public Methods

### 1. `.summ()`
- **Description**: Prints a summary of the model's input/output tensor properties.
- **Returns**: None
- **Example**:
  ```python
  model.summ()
  ```

---

### 2. `.t()`
- **Description**: Runs one dirty inference pass and prints performance metrics (e.g., latency, throughput).
- **Returns**: None
- **Example**:
  ```python
  model.t()
  ```

---

### 3. `.start(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0)`
- **Description**: Starts an asynchronous inference task.
- **Parameters**:
  | Parameter | Type | Default | Description |
  |----------|------|---------|-------------|
  | `inputs` | `List[np.ndarray]` | — | List of input tensors as NumPy arrays |
  | `task_id` | `int` | `0` | Task ID for identifying this inference request |
  | `priority` | `int` | `0` | Priority level (higher value means higher priority) |
- **Returns**: None (use `.wait()` to retrieve results)

---

### 4. `.wait(task_id: int = 0) -> List[np.ndarray]`
- **Description**: Waits for the specified task to complete and returns output tensors with zero-copy optimization.
- **Parameters**:
  | Parameter | Type | Default | Description |
  |----------|------|---------|-------------|
  | `task_id` | `int` | `0` | The task ID to wait for |
- **Returns**: `List[np.ndarray]` - Output tensors as NumPy arrays

---

### 5. `.inference(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0) -> List[np.ndarray]`
- **Description**: Safe call that performs: check + start + wait. Suitable for one-time inference.
- **Parameters**:
  | Parameter | Type | Default | Description |
  |----------|------|---------|-------------|
  | `inputs` | `List[np.ndarray]` | — | Input tensors |
  | `task_id` | `int` | `0` | Task ID |
  | `priority` | `int` | `0` | Task priority |
- **Returns**: `List[np.ndarray]` - Output tensors

---

### 6. `__call__(inputs: List[np.ndarray], task_id: int = 0, priority: int = 0) -> List[np.ndarray]`
- **Description**: Callable interface, equivalent to `.inference(...)`.
- **Example**:
  ```python
  outputs = model([input1, input2])
  ```

---

## 🧠 Internal Member Variables (for reference only)

| Variable Name | Type | Description |
|---------------|------|-------------|
| `model_path_` | `std::string` | Model path string |
| `n_task_` | `int32_t` | Max number of concurrent tasks |
| `is_infer` | `std::vector<int>` | Inference status per task |
| `task_handles` | `std::vector<hbUCPTaskHandle_t>` | Task handle list |
| `packed_dnn_handle`, `dnn_handle` | `hbDNN*` | DNN runtime handles |
| `input_properties`, `output_properties` | `hbDNNTensorProperties` | Tensor metadata |
| `inputs_shape`, `outputs_shape` | `std::vector<std::vector<size_t>>` | Input/output shapes |
| `inputs_dtype`, `outputs_dtype` | `std::vector<std::string>` | Data types (e.g., float32, uint8) |

---

## 🧪 Example Usage (Python)

```python
import cauchy_kesai as ck
import numpy as np

# Load model
model = ck.CauchyKesai("/path/to/model", n_task=2)

# Print model summary
model.summ()

# Prepare inputs
input_data = [np.random.rand(1, 3, 224, 224).astype(np.float32)]

# Run inference
output = model(input_data)

# Async usage
model.start(input_data, task_id=0)
result = model.wait(task_id=0)
```

---

## ✅ Summary of Recommended Usage

| Method | Recommendation | Use Case |
|--------|----------------|----------|
| `.inference()` / `__call__()` | ✅ Recommended | One-time safe inference |
| `.start()` + `.wait()` | ✅ Recommended | Asynchronous control |
| `.summ()` | ✅ Recommended | View model structure |
| `.t()` | ⚠️ For testing | Quick performance test |

---

如果你需要将这份文档导出为 Markdown 文件、HTML 页面或集成到 Sphinx 文档系统中，我也可以帮你一键生成 😊