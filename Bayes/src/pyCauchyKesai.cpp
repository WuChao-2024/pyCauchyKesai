#include "pyCauchyKesai.h"

bool checkFileExists(const std::string &path)
{
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

// X5 hbDNNDataType 枚举前 6 项是图像类型（Y/NV12/NV12_SEPARATE/YUV444/RGB/BGR），
// 均以 uint8 存储。Nash/S100 的枚举无图像类型前缀，起点直接是 S4。
std::string dtype_dnn2str(hbDNNTensorProperties properties)
{
    switch (properties.tensorType)
    {
    case HB_DNN_IMG_TYPE_Y:
    case HB_DNN_IMG_TYPE_NV12:
    case HB_DNN_IMG_TYPE_NV12_SEPARATE:
    case HB_DNN_IMG_TYPE_YUV444:
    case HB_DNN_IMG_TYPE_RGB:
    case HB_DNN_IMG_TYPE_BGR:
        return "uint8";
    case HB_DNN_TENSOR_TYPE_S4:
    case HB_DNN_TENSOR_TYPE_S8:
        return "int8";
    case HB_DNN_TENSOR_TYPE_U4:
    case HB_DNN_TENSOR_TYPE_U8:
        return "uint8";
    case HB_DNN_TENSOR_TYPE_F16:
        return "float16";
    case HB_DNN_TENSOR_TYPE_S16:
        return "int16";
    case HB_DNN_TENSOR_TYPE_U16:
        return "uint16";
    case HB_DNN_TENSOR_TYPE_F32:
        return "float32";
    case HB_DNN_TENSOR_TYPE_S32:
        return "int32";
    case HB_DNN_TENSOR_TYPE_U32:
        return "uint32";
    case HB_DNN_TENSOR_TYPE_F64:
        return "float64";
    case HB_DNN_TENSOR_TYPE_S64:
        return "int64";
    case HB_DNN_TENSOR_TYPE_U64:
        return "uint64";
    default:
        return "unknown";
    }
}

std::string dtype_np2str(const py::dtype &dt)
{
    if (dt.is(py::dtype::of<float>()))    return "float32";
    if (dt.is(py::dtype::of<double>()))   return "float64";
    if (dt.is(py::dtype::of<int8_t>()))   return "int8";
    if (dt.is(py::dtype::of<uint8_t>()))  return "uint8";
    if (dt.is(py::dtype::of<int16_t>()))  return "int16";
    if (dt.is(py::dtype::of<uint16_t>())) return "uint16";
    if (dt.is(py::dtype::of<int32_t>()))  return "int32";
    if (dt.is(py::dtype::of<uint32_t>())) return "uint32";
    if (dt.is(py::dtype::of<int64_t>()))  return "int64";
    if (dt.is(py::dtype::of<uint64_t>())) return "uint64";
    if (dt.is(py::dtype::of<bool>()))     return "bool";
    return "unknown";
}

py::dtype dtype_str2np(const std::string &dtype_str)
{
    if (dtype_str == "float32") return py::dtype::of<float>();
    if (dtype_str == "float64") return py::dtype::of<double>();
    if (dtype_str == "int8")    return py::dtype::of<int8_t>();
    if (dtype_str == "uint8")   return py::dtype::of<uint8_t>();
    if (dtype_str == "int16")   return py::dtype::of<int16_t>();
    if (dtype_str == "uint16")  return py::dtype::of<uint16_t>();
    if (dtype_str == "int32")   return py::dtype::of<int32_t>();
    if (dtype_str == "uint32")  return py::dtype::of<uint32_t>();
    if (dtype_str == "int64")   return py::dtype::of<int64_t>();
    if (dtype_str == "uint64")  return py::dtype::of<uint64_t>();
    if (dtype_str == "bool")    return py::dtype::of<bool>();
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

CauchyKesai::CauchyKesai(const std::string &model_path, int32_t n_task = 1, int32_t model_cnt_select = 0)
{
    auto warnings = py::module_::import("warnings");

    if (n_task <= 0)
    {
        warnings.attr("warn")("n_task <= 0, clamped to 1", py::module_::import("builtins").attr("UserWarning"));
        n_task = 1;
    }
    else if (n_task > 32)
    {
        warnings.attr("warn")("n_task > 32, clamped to 32", py::module_::import("builtins").attr("UserWarning"));
        n_task = 32;
    }
    n_task_ = n_task;
    task_handles.resize(n_task_);
    is_infer = std::vector<std::atomic<int>>(n_task_);

    if (!checkFileExists(model_path))
        throw std::runtime_error("Error: Model Path does not exist or is not a file: " + model_path);

    model_path_ = model_path;
    modelFileName = model_path_.c_str();

    // X5: hbPackedDNNHandle_t（Nash 为 hbDNNPackedHandle_t）
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1),
        "hbDNNInitializeFromFiles failed");

    model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    model_cnt_select_ = model_cnt_select;
    if (model_count > 1)
        warnings.attr("warn")(
            "Packed model contains " + std::to_string(model_count) + " models, will select only 1",
            py::module_::import("builtins").attr("UserWarning"));
    if (model_cnt_select_ >= model_count)
    {
        warnings.attr("warn")(
            "model_cnt_select (" + std::to_string(model_cnt_select_) +
            ") >= model_count (" + std::to_string(model_count) + "), clamped to " + std::to_string(model_count - 1),
            py::module_::import("builtins").attr("UserWarning"));
        model_cnt_select_ = model_count - 1;
    }
    else if (model_cnt_select_ < 0)
    {
        warnings.attr("warn")("model_cnt_select < 0, clamped to 0",
            py::module_::import("builtins").attr("UserWarning"));
        model_cnt_select_ = 0;
    }

    model_name = name_list[model_cnt_select_];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // ── 输入 ──────────────────────────────────────────────────────────────
    mbs = 0;
    input_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "hbDNNGetInputCount failed");

    inputs_shape.resize(input_count);
    inputs_byteSize.resize(input_count);

    // 用裸指针数组，与参考实现（pycauchyX5tools）保持一致，
    // 避免 std::vector 重分配后 hbDNNInfer 的 hbDNNTensor** 指针失效导致 segfault
    inputs_hbTensor.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        inputs_hbTensor[t] = new hbDNNTensor[input_count];

    for (int32_t i = 0; i < input_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");

        inputs_dtype.push_back(dtype_dnn2str(input_properties));

        char const *input_name;
        RDK_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
        inputs_name.push_back(std::string(input_name));

        // 判断是否是图像类型（Y/NV12/NV12_SEPARATE/YUV444/RGB/BGR）
        bool is_image = (input_properties.tensorType >= HB_DNN_IMG_TYPE_Y &&
                         input_properties.tensorType <= HB_DNN_IMG_TYPE_BGR);
        inputs_is_image.push_back(is_image);

        // 图像类型：用 alignedByteSize 展平为一维 (alignedByteSize,) uint8
        // Tensor 类型：用 validShape
        if (is_image)
        {
            inputs_numDimension.push_back(1);
            inputs_shape.push_back(std::vector<size_t>{});  // 占位，后面填充
        }
        else
        {
            inputs_numDimension.push_back(input_properties.validShape.numDimensions);
            inputs_shape.push_back(std::vector<size_t>{});
            for (int32_t j = 0; j < inputs_numDimension[i]; j++)
                inputs_shape[i].push_back(input_properties.validShape.dimensionSize[j]);
        }

        for (int32_t t = 0; t < n_task_; t++)
        {
            hbDNNGetInputTensorProperties(&inputs_hbTensor[t][i].properties, dnn_handle, i);
            // X5: hbSysAllocCachedMem + sysMem[0]（Nash 为 hbUCPMallocCached + sysMem 单体）
            RDK_CHECK_SUCCESS(
                hbSysAllocCachedMem(&inputs_hbTensor[t][i].sysMem[0],
                                    inputs_hbTensor[t][i].properties.alignedByteSize),
                "hbSysAllocCachedMem failed");
            mbs += double(inputs_hbTensor[t][i].properties.alignedByteSize) / 1024.0 / 1024.0;
        }
        inputs_byteSize[i] = inputs_hbTensor[0][i].properties.alignedByteSize;

        // 图像类型：填充展平的一维 shape
        if (is_image)
            inputs_shape[i].push_back(inputs_byteSize[i]);
    }

    // ── 输出 ──────────────────────────────────────────────────────────────
    output_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "hbDNNGetOutputCount failed");

    outputs_shape.resize(output_count);

    outputs_hbTensor.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        outputs_hbTensor[t] = new hbDNNTensor[output_count];

    for (int32_t i = 0; i < output_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");

        outputs_dtype.push_back(dtype_dnn2str(output_properties));

        char const *output_name;
        RDK_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i), "hbDNNGetOutputName failed");
        outputs_name.push_back(std::string(output_name));

        outputs_numDimension.push_back(output_properties.validShape.numDimensions);
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            outputs_shape[i].push_back(output_properties.validShape.dimensionSize[j]);

        for (int32_t t = 0; t < n_task_; t++)
        {
            hbDNNGetOutputTensorProperties(&outputs_hbTensor[t][i].properties, dnn_handle, i);
            RDK_CHECK_SUCCESS(
                hbSysAllocCachedMem(&outputs_hbTensor[t][i].sysMem[0],
                                    outputs_hbTensor[t][i].properties.alignedByteSize),
                "hbSysAllocCachedMem failed");
            mbs += double(outputs_hbTensor[t][i].properties.alignedByteSize) / 1024.0 / 1024.0;
        }
    }

    for (int32_t t = 0; t < n_task_; t++)
        is_infer[t].store(0, std::memory_order_relaxed);

    // ── 零拷贝 numpy 视图 ─────────────────────────────────────────────────
    // 零拷贝视图用 alignedByteSize 展平为一维 uint8，
    // 因为 X5 上 validShape 是逻辑形状，实际内存大小由 alignedByteSize 决定
    // （例如 NV12 输入 validShape=(1,3,224,224) 但 alignedByteSize=75264）
    input_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        for (int32_t i = 0; i < input_count; i++)
            input_tensors[t].push_back(
                py::array(py::dtype::of<uint8_t>(),
                          std::vector<size_t>{(size_t)inputs_hbTensor[t][i].properties.alignedByteSize},
                          inputs_hbTensor[t][i].sysMem[0].virAddr, py::none()));

    output_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        for (int32_t i = 0; i < output_count; i++)
            output_tensors[t].push_back(
                py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i],
                          outputs_hbTensor[t][i].sysMem[0].virAddr, py::none()));

    input_names  = inputs_name;
    output_names = outputs_name;
}

CauchyKesai::~CauchyKesai()
{
    std::cout << "[INFO] release model " << "\033[1;31m" << name_list[model_cnt_select_] << "\033[0m";
    for (int32_t t = 0; t < n_task_; t++)
    {
        for (int32_t i = 0; i < input_count; i++)
            hbSysFreeMem(&inputs_hbTensor[t][i].sysMem[0]);
        delete[] inputs_hbTensor[t];

        for (int32_t i = 0; i < output_count; i++)
            hbSysFreeMem(&outputs_hbTensor[t][i].sysMem[0]);
        delete[] outputs_hbTensor[t];
    }
    hbDNNRelease(packed_dnn_handle);
    std::cout << " success." << std::endl;
}

bool CauchyKesai::is_busy(int32_t task_id) const
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range(
            "task_id out of range: got " + std::to_string(task_id) +
            ", valid range [0, " + std::to_string(n_task_ - 1) + "]");
    return is_infer[task_id].load(std::memory_order_acquire) != 0;
}

py::dict CauchyKesai::s()
{
    py::list model_names_list;
    for (int32_t i = 0; i < model_count; i++)
    {
        py::dict entry;
        entry["index"]    = i;
        entry["name"]     = std::string(name_list[i]);
        entry["selected"] = (i == model_cnt_select_);
        model_names_list.append(entry);
    }

    py::list inputs_list;
    for (int32_t i = 0; i < input_count; i++)
    {
        py::dict inp;
        inp["index"] = i;
        inp["name"]  = inputs_name[i];
        inp["dtype"] = inputs_dtype[i];
        py::list shape;
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
            shape.append(inputs_shape[i][j]);
        inp["shape"] = shape;
        inputs_list.append(inp);
    }

    py::list outputs_list;
    for (int32_t i = 0; i < output_count; i++)
    {
        py::dict out;
        out["index"] = i;
        out["name"]  = outputs_name[i];
        out["dtype"] = outputs_dtype[i];
        py::list shape;
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            shape.append(outputs_shape[i][j]);
        out["shape"] = shape;
        outputs_list.append(out);
    }

    py::dict result;
    result["model_path"]  = model_path_;
    result["model_names"] = model_names_list;
    result["n_task"]      = n_task_;
    result["memory_mb"]   = mbs;
    result["inputs"]      = inputs_list;
    result["outputs"]     = outputs_list;
    return result;
}

py::dict CauchyKesai::t()
{
    int32_t task_id = 0;
    auto tp_start = std::chrono::system_clock::now();

    is_infer[task_id].store(1, std::memory_order_release);

    // X5: hbDNNInfer + hbDNNInferCtrlParam
    // Nash: hbDNNInferV2 + hbUCPSchedParam + hbUCPSubmitTask
    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param);
    ctrl_param.bpuCoreId = HB_BPU_CORE_ANY;
    ctrl_param.priority  = HB_DNN_PRIORITY_LOWEST;

    RDK_CHECK_SUCCESS(
        hbDNNInfer(&task_handle, &outputs_hbTensor[task_id],
                   inputs_hbTensor[task_id], dnn_handle, &ctrl_param),
        "hbDNNInfer failed");

    // X5: hbDNNWaitTaskDone（Nash: hbUCPWaitTaskDone）
    RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone failed");

    for (int i = 0; i < output_count; i++)
        hbSysFlushMem(&outputs_hbTensor[task_id][i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

    // X5: hbDNNReleaseTask（Nash: hbUCPReleaseTask）
    RDK_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask failed");

    is_infer[task_id].store(0, std::memory_order_release);

    auto tp_end = std::chrono::system_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count();

    py::dict result;
    result["time_us"]  = total_time;
    result["time_ms"]  = total_time / 1000.0;
    result["time_s"]   = total_time / 1000000.0;
    result["time_min"] = total_time / (1000000.0 * 60);
    return result;
}

void CauchyKesai::start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    is_infer[task_id].store(1, std::memory_order_release);

    if (!inputs.empty())
    {
        for (int32_t i = 0; i < input_count; i++)
        {
            py::array contiguous = py::array::ensure(inputs[i], py::array::c_style);
            // 用 alignedByteSize 而非 contiguous.nbytes()：
            // X5 上 validShape 是逻辑形状，实际内存由 alignedByteSize 决定
            // （如 NV12 输入 validShape=(1,3,224,224) 但 alignedByteSize=75264）
            std::memcpy(inputs_hbTensor[task_id][i].sysMem[0].virAddr,
                        contiguous.data(), inputs_byteSize[i]);
        }
    }
    for (int32_t i = 0; i < input_count; i++)
        hbSysFlushMem(&inputs_hbTensor[task_id][i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    hbDNNTaskHandle_t task_handle = nullptr;
    hbDNNInferCtrlParam ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&ctrl_param);
    ctrl_param.bpuCoreId = HB_BPU_CORE_ANY;
    ctrl_param.priority  = priority;

    // 释放 GIL，让其他 Python 线程在 BPU 运行期间可以被调度
    {
        py::gil_scoped_release release;
        RDK_CHECK_SUCCESS(
            hbDNNInfer(&task_handle, &outputs_hbTensor[task_id],
                       inputs_hbTensor[task_id], dnn_handle, &ctrl_param),
            "hbDNNInfer failed");
    }

    task_handles[task_id] = task_handle;
}

std::vector<py::array> CauchyKesai::wait(int32_t task_id)
{
    // 等待推理结束，释放 GIL
    {
        py::gil_scoped_release release;
        RDK_CHECK_SUCCESS(hbDNNWaitTaskDone(task_handles[task_id], 0),
                          "hbDNNWaitTaskDone failed");
    }

    for (int i = 0; i < output_count; i++)
        hbSysFlushMem(&outputs_hbTensor[task_id][i].sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);

    RDK_CHECK_SUCCESS(hbDNNReleaseTask(task_handles[task_id]), "hbDNNReleaseTask failed");

    is_infer[task_id].store(0, std::memory_order_release);

    std::vector<py::array> rs;
    for (int32_t i = 0; i < output_count; i++)
        rs.push_back(py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i],
                               outputs_hbTensor[task_id][i].sysMem[0].virAddr, py::none()));
    return rs;
}

std::vector<py::array> CauchyKesai::inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range(
            "task_id out of range: got " + std::to_string(task_id) +
            ", valid range [0, " + std::to_string(n_task_ - 1) + "]");

    if (priority < 0 || priority > 255)
        throw std::out_of_range(
            "priority out of range: got " + std::to_string(priority) +
            ", valid range [0, 255]");

    if ((int32_t)inputs.size() != input_count)
        throw std::invalid_argument(
            "input count mismatch: expected " + std::to_string(input_count) +
            ", got " + std::to_string(inputs.size()));

    for (int32_t cnt = 0; cnt < input_count; cnt++)
    {
        std::string got_dtype = dtype_np2str(inputs[cnt].dtype());
        if (inputs_dtype[cnt] != got_dtype)
            throw std::invalid_argument(
                "dtype mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                "': expected " + inputs_dtype[cnt] + ", got " + got_dtype);

        if (inputs[cnt].ndim() != inputs_numDimension[cnt])
            throw std::invalid_argument(
                "ndim mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                "': expected " + std::to_string(inputs_numDimension[cnt]) +
                ", got " + std::to_string(inputs[cnt].ndim()));

        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            if (inputs_shape[cnt][i] != (size_t)inputs[cnt].shape()[i])
            {
                std::string expected = "(";
                for (int j = 0; j < inputs_numDimension[cnt]; ++j)
                    expected += std::to_string(inputs_shape[cnt][j]) + (j + 1 < inputs_numDimension[cnt] ? ", " : ")");
                std::string got = "(";
                for (int j = 0; j < inputs[cnt].ndim(); ++j)
                    got += std::to_string(inputs[cnt].shape()[j]) + (j + 1 < inputs[cnt].ndim() ? ", " : ")");
                throw std::invalid_argument(
                    "shape mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                    "': expected " + expected + ", got " + got);
            }
        }
    }

    if (is_infer[task_id].load(std::memory_order_acquire))
        throw std::runtime_error("task_id " + std::to_string(task_id) + " is already in use");

    start(inputs, task_id, priority);
    return wait(task_id);
}
