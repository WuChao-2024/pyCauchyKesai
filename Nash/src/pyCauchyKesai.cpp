#include "pyCauchyKesai.h"

bool checkFileExists(const std::string &path)
{
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

std::string dtype_ucp2str(hbDNNTensorProperties properties)
{
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_S8)
        return "int8";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8)
        return "uint8";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F16)
        return "float16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
        return "int16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U16)
        return "uint16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F32)
        return "float32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S32)
        return "int32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U32)
        return "uint32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F64)
        return "float64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S64)
        return "int64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U64)
        return "uint64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_BOOL8)
        return "bool";
    else
        return "unknown";
}
std::string dtype_np2str(const py::dtype &dt)
{
    if (dt.is(py::dtype::of<float>()))
        return "float32";
    if (dt.is(py::dtype::of<double>()))
        return "float64";
    if (dt.is(py::dtype::of<int8_t>()))
        return "int8";
    if (dt.is(py::dtype::of<uint8_t>()))
        return "uint8";
    if (dt.is(py::dtype::of<int16_t>()))
        return "int16";
    if (dt.is(py::dtype::of<uint16_t>()))
        return "uint16";
    if (dt.is(py::dtype::of<int32_t>()))
        return "int32";
    if (dt.is(py::dtype::of<uint32_t>()))
        return "uint32";
    if (dt.is(py::dtype::of<int64_t>()))
        return "int64";
    if (dt.is(py::dtype::of<uint64_t>()))
        return "uint64";
    if (dt.is(py::dtype::of<bool>()))
        return "bool";
    return "unknown";
}
py::dtype dtype_str2np(const std::string &dtype_str)
{
    if (dtype_str == "float32")
        return py::dtype::of<float>();
    if (dtype_str == "float64")
        return py::dtype::of<double>();
    if (dtype_str == "int8")
        return py::dtype::of<int8_t>();
    if (dtype_str == "uint8")
        return py::dtype::of<uint8_t>();
    if (dtype_str == "int16")
        return py::dtype::of<int16_t>();
    if (dtype_str == "uint16")
        return py::dtype::of<uint16_t>();
    if (dtype_str == "int32")
        return py::dtype::of<int32_t>();
    if (dtype_str == "uint32")
        return py::dtype::of<uint32_t>();
    if (dtype_str == "int64")
        return py::dtype::of<int64_t>();
    if (dtype_str == "uint64")
        return py::dtype::of<uint64_t>();
    if (dtype_str == "bool")
        return py::dtype::of<bool>();

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
    inputs_hbTensor.resize(n_task_);
    outputs_hbTensor.resize(n_task_);
    task_handles.resize(n_task_);
    is_infer = std::vector<std::atomic<int>>(n_task_);

    // 检查 model_path 是否存在且是文件
    if (!checkFileExists(model_path))
    {
        throw std::runtime_error("Error: Model Path does not exist or is not a file: " + model_path);
    }

    // 加载 BPU 模型
    model_path_ = model_path;
    modelFileName = model_path_.c_str();
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1),
        "hbDNNInitializeFromFiles failed");

    model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");
    model_cnt_select_ = model_cnt_select;
    if (model_count > 1)
    {
        warnings.attr("warn")(
            "Packed model contains " + std::to_string(model_count) + " models, will select only 1",
            py::module_::import("builtins").attr("UserWarning"));
    }
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
        warnings.attr("warn")(
            "model_cnt_select < 0, clamped to 0",
            py::module_::import("builtins").attr("UserWarning"));
        model_cnt_select_ = 0;
    }

    model_name = name_list[model_cnt_select_];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 输入信息
    mbs = 0;
    input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");
    inputs_shape.resize(input_count);
    inputs_byteSize.resize(input_count);
    for (int32_t t = 0; t < n_task_; t++)
    {
        inputs_hbTensor[t].resize(input_count);
    }
    for (int32_t i = 0; i < input_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        // 输入头数据类型
        inputs_dtype.push_back(dtype_ucp2str(input_properties));

        // 输入头名称
        char const *input_name;
        RDK_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
        std::string input_name_(input_name);
        inputs_name.push_back(input_name_);

        // 输入头形状
        inputs_numDimension.push_back(input_properties.validShape.numDimensions);
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
        {
            inputs_shape[i].push_back(input_properties.validShape.dimensionSize[j]);
        }

        // 为输入Tensor开辟内存
        for (int32_t t = 0; t < n_task_; t++)
        {
            hbDNNGetInputTensorProperties(&inputs_hbTensor[t][i].properties, dnn_handle, i);
            if (inputs_hbTensor[t][i].properties.alignedByteSize < 0)
            {
                for (int32_t dim_i = inputs_hbTensor[t][i].properties.validShape.numDimensions - 1; dim_i >= 0; --dim_i)
                {
                    if (inputs_hbTensor[t][i].properties.stride[dim_i] == -1)
                    {
                        auto cur_stride =
                            inputs_hbTensor[t][i].properties.stride[dim_i + 1] *
                            inputs_hbTensor[t][i].properties.validShape.dimensionSize[dim_i + 1];
                        inputs_hbTensor[t][i].properties.stride[dim_i] = ALIGN_32(cur_stride);
                    }
                }
                inputs_hbTensor[t][i].properties.alignedByteSize = inputs_hbTensor[t][i].properties.stride[0] * inputs_hbTensor[t][i].properties.validShape.dimensionSize[0];
            }
            RDK_CHECK_SUCCESS(
                hbUCPMallocCached(&inputs_hbTensor[t][i].sysMem, inputs_hbTensor[t][i].properties.alignedByteSize, 0),
                "hbUCPMallocCached failed");

            mbs += double(inputs_hbTensor[t][i].properties.alignedByteSize) / 1024.0 / 1024.0;
        }
        inputs_byteSize[i] = inputs_hbTensor[0][i].properties.alignedByteSize;
    }
    // 输出信息
    output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");
    outputs_shape.resize(output_count);
    for (int32_t t = 0; t < n_task_; t++)
    {
        outputs_hbTensor[t].resize(output_count);
    }

    for (int32_t i = 0; i < output_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        // 输出头数据类型
        outputs_dtype.push_back(dtype_ucp2str(output_properties));

        // 输出头名称
        char const *output_name;
        RDK_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i),
                          "hbDNNGetOutputName failed");
        std::string output_name_(output_name);
        outputs_name.push_back(output_name_);

        // 输出头形状
        outputs_numDimension.push_back(output_properties.validShape.numDimensions);
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
        {
            outputs_shape[i].push_back(output_properties.validShape.dimensionSize[j]);
        }

        // 为输出Tensor开辟内存
        for (int32_t t = 0; t < n_task_; t++)
        {
            hbDNNGetOutputTensorProperties(&outputs_hbTensor[t][i].properties, dnn_handle, i);
            RDK_CHECK_SUCCESS(
                hbUCPMallocCached(&outputs_hbTensor[t][i].sysMem, outputs_hbTensor[t][i].properties.alignedByteSize, 0),
                "hbUCPMallocCached failed");
            mbs += double(outputs_hbTensor[t][i].properties.alignedByteSize) / 1024.0 / 1024.0;
        }
    }

    // 异步推理标志初始化（std::atomic 已在构造时初始化为 0）
    for (int32_t t = 0; t < n_task_; t++)
    {
        is_infer[t].store(0, std::memory_order_relaxed);
    }

    // 构建 ION 内存的 numpy 视图（零拷贝）
    input_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
    {
        for (int32_t i = 0; i < input_count; i++)
        {
            input_tensors[t].push_back(
                py::array(dtype_str2np(inputs_dtype[i]), inputs_shape[i],
                          inputs_hbTensor[t][i].sysMem.virAddr));
        }
    }

    output_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
    {
        for (int32_t i = 0; i < output_count; i++)
        {
            output_tensors[t].push_back(
                py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i],
                          outputs_hbTensor[t][i].sysMem.virAddr));
        }
    }

    // 名称列表
    input_names  = inputs_name;
    output_names = outputs_name;
}

CauchyKesai::~CauchyKesai()
{
    std::cout << "[INFO] release model " << "\033[1;31m" << name_list[model_cnt_select_] << "\033[0m";
    for (int32_t t = 0; t < n_task_; t++)
    {
        for (int32_t i = 0; i < input_count; i++)
            hbUCPFree(&(inputs_hbTensor[t][i].sysMem));
        for (int32_t i = 0; i < output_count; i++)
            hbUCPFree(&(outputs_hbTensor[t][i].sysMem));
    }
    hbDNNRelease(packed_dnn_handle);

    std::cout << " success." << std::endl;
}

bool CauchyKesai::is_busy(int32_t task_id) const
{
    if (task_id < 0 || task_id >= n_task_)
    {
        throw std::out_of_range(
            "task_id out of range: got " + std::to_string(task_id) +
            ", valid range [0, " + std::to_string(n_task_ - 1) + "]");
    }
    return is_infer[task_id].load(std::memory_order_acquire) != 0;
}

py::dict CauchyKesai::s()
{
    // 构建结构化数据
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
    result["model_path"]   = model_path_;
    result["model_names"]  = model_names_list;
    result["n_task"]       = n_task_;
    result["memory_mb"]    = mbs;
    result["inputs"]       = inputs_list;
    result["outputs"]      = outputs_list;

    return result;
}

py::dict CauchyKesai::t()
{
    int32_t task_id = 0;

    // 开始计时
    auto tp_start = std::chrono::system_clock::now();

    is_infer[task_id].store(1, std::memory_order_release);

    // BPU推理任务
    hbUCPTaskHandle_t task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
        "hbDNNInferV2 failed");
    hbUCPSchedParam ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.priority = HB_UCP_PRIORITY_LOWEST;
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                      "hbUCPSubmitTask failed");

    task_handles[task_id] = task_handle;

    // 等待推理结束
    RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0),
                      "hbUCPWaitTaskDone failed");

    // 刷新带Cache的内存
    for (int i = 0; i < output_count; i++)
    {
        hbUCPMemFlush(&outputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // 释放推理句柄
    RDK_CHECK_SUCCESS(hbUCPReleaseTask(task_handles[task_id]),
                      "hbUCPReleaseTask failed");

    is_infer[task_id].store(0, std::memory_order_release);

    // 停止计时
    auto tp_end = std::chrono::system_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count();
    double time_us = total_time;                     // 微秒
    double time_ms = total_time / 1000.0;            // 毫秒
    double time_s = total_time / 1000000.0;          // 秒
    double time_min = total_time / (1000000.0 * 60); // 分钟

    py::dict result;
    result["time_us"]  = time_us;
    result["time_ms"]  = time_ms;
    result["time_s"]   = time_s;
    result["time_min"] = time_min;

    return result;
}

void CauchyKesai::start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    is_infer[task_id].store(1, std::memory_order_release);

    // 将array的数据拷贝进输入Tensor, 并刷新带Cache的内存
    // 如果 inputs 为空，说明用户已通过 input_tensors 零拷贝写入，跳过拷贝只刷 cache
    if (!inputs.empty())
    {
        for (int32_t i = 0; i < input_count; i++)
        {
            // 确保数组是 C 连续的
            // 如果已经是 C 连续，ensure() 不会拷贝，直接返回原数组引用
            // 如果不连续，ensure() 会创建一个 C 连续的副本
            py::array contiguous_input = py::array::ensure(inputs[i], py::array::c_style);
            std::memcpy(inputs_hbTensor[task_id][i].sysMem.virAddr, contiguous_input.data(), contiguous_input.nbytes());
        }
    }
    for (int32_t i = 0; i < input_count; i++)
    {
        hbUCPMemFlush(&inputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_CLEAN);
    }

    // BPU推理任务（提交后释放 GIL，让其他 Python 线程可以运行）
    hbUCPTaskHandle_t task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
        "hbDNNInferV2 failed");
    hbUCPSchedParam ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.priority = priority;
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;

    {
        py::gil_scoped_release release;
        RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                          "hbUCPSubmitTask failed");
    }

    task_handles[task_id] = task_handle;
}

std::vector<py::array> CauchyKesai::wait(int32_t task_id)
{
    // 等待推理结束：释放 GIL，让其他 Python 线程在 BPU 运行期间可以被调度
    {
        py::gil_scoped_release release;
        RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0),
                          "hbUCPWaitTaskDone failed");
    }

    // 刷新带Cache的内存
    for (int i = 0; i < output_count; i++)
    {
        hbUCPMemFlush(&outputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // 释放推理句柄
    RDK_CHECK_SUCCESS(hbUCPReleaseTask(task_handles[task_id]),
                      "hbUCPReleaseTask failed");

    // 推理标志
    is_infer[task_id].store(0, std::memory_order_release);

    // 返回推理结果
    std::vector<py::array> rs;

    for (int32_t i = 0; i < output_count; i++)
    {
        rs.push_back(py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i], outputs_hbTensor[task_id][i].sysMem.virAddr));
    }

    return rs;
}

std::vector<py::array> CauchyKesai::inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // task id 检查
    if (task_id < 0 || task_id >= n_task_)
    {
        throw std::out_of_range(
            "task_id out of range: got " + std::to_string(task_id) +
            ", valid range [0, " + std::to_string(n_task_ - 1) + "]");
    }

    // priority 检查
    if (priority < 0 || priority > 255)
    {
        throw std::out_of_range(
            "priority out of range: got " + std::to_string(priority) +
            ", valid range [0, 255]");
    }

    // 输入数量检查
    if ((int32_t)inputs.size() != input_count)
    {
        throw std::invalid_argument(
            "input count mismatch: expected " + std::to_string(input_count) +
            ", got " + std::to_string(inputs.size()));
    }

    // dtype / ndim / shape 检查
    for (int32_t cnt = 0; cnt < input_count; cnt++)
    {
        // dtype
        std::string got_dtype = dtype_np2str(inputs[cnt].dtype());
        if (inputs_dtype[cnt] != got_dtype)
        {
            throw std::invalid_argument(
                "dtype mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                "': expected " + inputs_dtype[cnt] + ", got " + got_dtype);
        }

        // ndim
        if (inputs[cnt].ndim() != inputs_numDimension[cnt])
        {
            throw std::invalid_argument(
                "ndim mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                "': expected " + std::to_string(inputs_numDimension[cnt]) +
                ", got " + std::to_string(inputs[cnt].ndim()));
        }

        // shape
        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            if (inputs_shape[cnt][i] != (size_t)inputs[cnt].shape()[i])
            {
                // 构造 expected shape 字符串
                std::string expected = "(";
                for (int j = 0; j < inputs_numDimension[cnt]; ++j)
                    expected += std::to_string(inputs_shape[cnt][j]) + (j + 1 < inputs_numDimension[cnt] ? ", " : ")");
                // 构造 got shape 字符串
                std::string got = "(";
                for (int j = 0; j < inputs[cnt].ndim(); ++j)
                    got += std::to_string(inputs[cnt].shape()[j]) + (j + 1 < inputs[cnt].ndim() ? ", " : ")");
                throw std::invalid_argument(
                    "shape mismatch at input[" + std::to_string(cnt) + "] '" + inputs_name[cnt] +
                    "': expected " + expected + ", got " + got);
            }
        }
    }

    // task 占用检查
    if (is_infer[task_id].load(std::memory_order_acquire))
    {
        throw std::runtime_error(
            "task_id " + std::to_string(task_id) + " is already in use");
    }

    start(inputs, task_id, priority);
    return wait(task_id);
}
