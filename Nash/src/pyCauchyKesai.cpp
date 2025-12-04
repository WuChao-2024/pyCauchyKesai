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
    if (n_task <= 0)
    {
        std::cout << "[CauchyKesai][W] n_task <= 0, let: n_task = 1;" << std::endl;
        n_task = 1;
    }
    else if (n_task > 32)
    {
        std::cout << "[CauchyKesai][W] n_task > 32, let: n_task = 32;" << std::endl;
        n_task = 32;
    }
    n_task_ = n_task;
    inputs_hbTensor.resize(n_task_);
    outputs_hbTensor.resize(n_task_);
    // results.resize(n_task_);
    task_handles.resize(n_task_);
    is_infer.resize(n_task_);

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
        std::cout << "[CauchyKesai][W] model_count: " << model_count << ", will select only 1." << std::endl;
    }
    else if (model_cnt_select_ >= model_count)
    {
        std::cout << "[CauchyKesai][W] model_cnt_select > model_count, let: model_cnt_select_ = model_count-1" << std::endl;
        model_cnt_select_ = model_count - 1;
    }
    else if (model_cnt_select_ < 0)
    {
        std::cout << "[CauchyKesai][W] model_cnt_select < 0, let: model_cnt_select_ = 0" << std::endl;
        model_cnt_select_ = 0;
    }

    model_name = name_list[model_cnt_select_];
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 输入信息
    // std::cout << " 输入信息 ..." << std::endl;
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
            // std::cout << "[CauchyKesai][I][input] t=" << t << ", i=" <<i << ", size=" << inputs_hbTensor[t][i].properties.alignedByteSize <<std::endl;
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
    // std::cout << " 输出信息 ..." << std::endl;
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
            // std::cout << "[CauchyKesai][I][output] t=" << t << ", i=" <<i << ", size=" << outputs_hbTensor[t][i].properties.alignedByteSize <<std::endl;
            RDK_CHECK_SUCCESS(
                hbUCPMallocCached(&outputs_hbTensor[t][i].sysMem, outputs_hbTensor[t][i].properties.alignedByteSize, 0),
                "hbUCPMallocCached failed");
            mbs += double(outputs_hbTensor[t][i].properties.alignedByteSize) / 1024.0 / 1024.0;
        }
    }

    // 异步推理标志
    // std::cout << "异步推理标志" << std::endl;
    for (int32_t t = 0; t < n_task_; t++)
    {
        // std::cout << "t: " <<t << std::endl;
        is_infer[t] = 0;
    }
}

CauchyKesai::~CauchyKesai()
{
    std::cout << "[INFO] release model " << "\033[1;31m" << name_list[model_cnt_select_] << "\033[0m";
    // summ();
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

void CauchyKesai::s()
{
    std::cout << "================== Model Summarys ==================" << std::endl;
    // 模型路径
    std::cout << "\033[1;31m" << "Model File: " << "\033[0m" << model_path_ << std::endl;

    // 总的模型名称
    std::cout << "\033[1;31m" << "Model Names: " << "\033[0m" << std::endl;
    for (int32_t i = 0; i < model_count; i++)
    {
        if (i == model_cnt_select_)
            std::cout << i << ": " << name_list[i] << " [*Select]" << std::endl;
        else
            std::cout << i << ": " << name_list[i] << std::endl;
    }

    // task n
    std::cout << "\033[1;31m" << "Task N: " << "\033[0m" << n_task_ << std::endl;

    // 输入输出占用tensor的aligned byte size (MB)
    std::cout << "\033[1;31m" << "Inputs/Outputs AlignedByteSize: " << "\033[0m" << mbs << "MB." << std::endl;

    // 模型输入信息
    std::cout << "\033[1;31m" << "Inputs Info: " << "\033[0m" << std::endl;
    for (int32_t i = 0; i < input_count; i++)
    {
        std::cout << "[" << i << "]";
        std::cout << "[" << inputs_name[i] << "]: " << inputs_dtype[i] << ", (";
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
            std::cout << inputs_shape[i][j] << ", ";
        std::cout << ")" << std::endl;
    }

    // 模型输出信息
    std::cout << "\033[1;31m" << "Outputs Info: " << "\033[0m" << std::endl;
    for (int32_t i = 0; i < output_count; i++)
    {
        std::cout << "[" << i << "]";
        std::cout << "[" << outputs_name[i] << "]: " << outputs_dtype[i] << ", (";
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            std::cout << outputs_shape[i][j] << ", ";
        std::cout << ")" << std::endl;
    }
    std::cout << "====================================================" << std::endl;
}

void CauchyKesai::t()
{
    int32_t task_id = 0;

    // 开始计时
    auto tp_start = std::chrono::system_clock::now();

    // 推理标志
    is_infer[task_id] = 1;

    // BPU推理任务
    hbUCPTaskHandle_t task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
        "hbDNNInferV2 failed");
    hbUCPSchedParam ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.priority = HB_UCP_PRIORITY_LOWEST;
    // ctrl_param.deviceId = 0;
    // ctrl_param.customId = 0;
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

    // 推理标志
    is_infer[task_id] = 0;

    // 停止计时
    auto tp_end = std::chrono::system_clock::now();

    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count();
    double time_us = total_time;                     // 微秒
    double time_ms = total_time / 1000.0;            // 毫秒
    double time_s = total_time / 1000000.0;          // 秒
    double time_min = total_time / (1000000.0 * 60); // 分钟

    std::cout.precision(6); // 设置浮点数输出精度
    std::cout << "\033[1;31m" << "Inference Info: " << "\033[0m" << std::endl;
    std::cout << "Time in microseconds: " << time_us << " μs" << std::endl;
    std::cout << "Time in milliseconds: " << time_ms << " ms" << std::endl;
    std::cout << "Time in seconds:      " << time_s << " s" << std::endl;
    std::cout << "Time in minutes:      " << time_min << " min" << std::endl;
}

void CauchyKesai::start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // 推理标志
    is_infer[task_id] = 1;

    // 将array的数据拷贝进输入Tensor, 并刷新带Cache的内存
    for (int32_t i = 0; i < input_count; i++)
    {
        std::memcpy(inputs_hbTensor[task_id][i].sysMem.virAddr, inputs[i].data(), inputs[i].nbytes());
        // std::cout << "[CauchyKesai][D] inputs[" << i << "].nbytes() = " << inputs[i].nbytes() << std::endl;
    }
    for (int32_t i = 0; i < input_count; i++)
    {
        hbUCPMemFlush(&inputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // BPU推理任务
    hbUCPTaskHandle_t task_handle{nullptr};

    RDK_CHECK_SUCCESS(
        hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
        "hbDNNInferV2 failed");
    hbUCPSchedParam ctrl_param;

    HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
    ctrl_param.priority = priority; // HB_UCP_PRIORITY_LOWEST;
    // ctrl_param.deviceId = 0;
    // ctrl_param.customId = 0;
    ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
    RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param),
                      "hbUCPSubmitTask failed");

    task_handles[task_id] = task_handle;
}

std::vector<py::array> CauchyKesai::wait(int32_t task_id)
{
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

    // 推理标志
    is_infer[task_id] = 0;

    // 返回推理结果
    std::vector<py::array> rs;

    for (int32_t i = 0; i < output_count; i++)
    {
        rs.push_back(py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i], outputs_hbTensor[task_id][i].sysMem.virAddr));
    }

    return rs;
    // return results[task_id];
}

std::vector<py::array> CauchyKesai::inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // task id 检查
    if (task_id < 0 || task_id >= n_task_)
    {
        std::cout << "[CauchyKesai][E] Task ID out of range, task_id: " << task_id << ", n_task: " << n_task_ << std::endl;
        return std::vector<py::array>();
    }

    // priority检查
    if (priority < 0 || priority > 255)
    {
        std::cout << "[CauchyKesai][E] Priority out of range, Expected: [0, 255], Got: " << priority << "." << std::endl;
        return std::vector<py::array>();
    }

    // 传入list的len()检查
    if (inputs.size() != input_count)
    {
        std::cout << "[CauchyKesai][E] Input Conut Not Match, input_count: " << input_count << ", inputs.size(): " << inputs.size() << std::endl;
        return std::vector<py::array>();
    }

    // 传入list[np.array]的dtype检查
    for (size_t cnt = 0; cnt < input_count; cnt++)
    {
        if (inputs_dtype[cnt] != dtype_np2str(inputs[cnt].dtype()))
        {
            std::cout << "[CauchyKesai][E] ERROR Data Type, input cnt: " << cnt << std::endl;
            return std::vector<py::array>();
        }
        // std::cout << "[CauchyKesai][I] Data Type Check Success, input cnt: " << cnt << std::endl;
    }

    // 传入list[np.array]的ndim检查
    for (size_t cnt = 0; cnt < input_count; cnt++)
    {
        if (inputs[cnt].ndim() != inputs_numDimension[cnt])
        {
            std::cout << "[CauchyKesai][E] ERROR Data Dimension, input cnt: " << cnt;
            std::cout << ", Expected ndim:" << inputs_numDimension[cnt];
            std::cout << ", Got ndim: " << inputs[cnt].ndim() << ". " << std::endl;
            return std::vector<py::array>();
        }
        // std::cout << "[CauchyKesai][I] Data Dimension Check Success, input cnt: " << cnt << std::endl;
    }

    // 传入list[np.array]的stride检查
    for (size_t cnt = 0; cnt < input_count; cnt++)
    {
        ssize_t prev_stride = 0;
        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            ssize_t current_stride = inputs[cnt].strides()[i];
            if (i > 0 && current_stride > prev_stride)
            {
                std::cout << "[CauchyKesai][E] ERROR Stride, input cnt: " << cnt << ", Strides: (";
                for (int i = 0; i < inputs[cnt].ndim(); ++i)
                {
                    std::cout << inputs[cnt].strides()[i] << ", ";
                }
                std::cout << "), please use numpy.ndarray.copy() first." << std::endl;
                return std::vector<py::array>();
            }
            prev_stride = current_stride;
        }
        // std::cout << "[CauchyKesai][I] Stride Check Success, input cnt: " << cnt << std::endl;
    }

    // 传入list[np.array]的shape检查
    for (size_t cnt = 0; cnt < input_count; cnt++)
    {
        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            if (inputs_shape[cnt][i] != inputs[cnt].shape()[i])
            {
                std::cout << "[CauchyKesai][E] ERROR array shape, input cnt: " << cnt << ", Expected: (";
                for (int i = 0; i < inputs[cnt].ndim(); ++i)
                {
                    std::cout << inputs_shape[cnt][i] << ", ";
                }
                std::cout << "), Got: (";
                for (int i = 0; i < inputs[cnt].ndim(); ++i)
                {
                    std::cout << inputs[cnt].shape()[i] << ", ";
                }
                std::cout << ")" << std::endl;
                return std::vector<py::array>();
            }
        }
        // std::cout << "[CauchyKesai][I] Array shape Check Success, input cnt: " << cnt << std::endl;
    }

    if (is_infer[task_id])
    {
        std::cout << "[CauchyKesai][E] ERROR task_id: " << task_id << ", used." << std::endl;
        return std::vector<py::array>();
    }

    start(inputs, task_id, priority);
    return wait(task_id);
}

// void CauchyKesai::start(const std::vector<py::array_t<float>> &inputs, int task_id)
// {
//     return;
// }
// std::vector<py::array_t<float>> CauchyKesai::wait(int task_id)
// {
//     return std::vector<py::array_t<float>>();
// }

// py::array_t<float> CauchyKesai::start(py::array_t<float> state, py::array_t<float> laptop, py::array_t<float> phone)
// {
//     // np check
//     py::buffer_info state_buf_info = state.request();
//     if (state_buf_info.ndim != transformerLayers_input[0].properties.validShape.numDimensions ||
//         state_buf_info.shape[0] != transformerLayers_input[0].properties.validShape.dimensionSize[0] ||
//         state_buf_info.shape[1] != transformerLayers_input[0].properties.validShape.dimensionSize[1])
//     {
//         std::stringstream ss;
//         ss << "wrong input numpy array state. need: (";
//         for (int32_t i = 0; i < transformerLayers_input[0].properties.validShape.numDimensions; i++)
//         {
//             ss << transformerLayers_input[0].properties.validShape.dimensionSize[i] << ", ";
//         }
//         ss << "), got: (";
//         for (int32_t i = 0; i < state_buf_info.ndim; i++)
//         {
//             ss << state_buf_info.shape[i] << ", ";
//         }
//         ss << ")";
//         throw std::runtime_error(ss.str());
//     }

//     if (state_buf_info.format != py::format_descriptor<float>::format())
//         throw std::runtime_error("Input numpy array state must have dtype float32.");

//     py::buffer_info laptop_buf_info = laptop.request();
//     if (laptop_buf_info.ndim != laptop_input[0].properties.validShape.numDimensions ||
//         laptop_buf_info.shape[0] != laptop_input[0].properties.validShape.dimensionSize[0] ||
//         laptop_buf_info.shape[1] != laptop_input[0].properties.validShape.dimensionSize[1] ||
//         laptop_buf_info.shape[2] != laptop_input[0].properties.validShape.dimensionSize[2] ||
//         laptop_buf_info.shape[3] != laptop_input[0].properties.validShape.dimensionSize[3])
//     {
//         std::stringstream ss;
//         ss << "wrong input numpy array laptop. need: (";
//         for (int32_t i = 0; i < laptop_input[0].properties.validShape.numDimensions; i++)
//         {
//             ss << laptop_input[0].properties.validShape.dimensionSize[i] << ", ";
//         }
//         ss << "), got: (";
//         for (int32_t i = 0; i < laptop_buf_info.ndim; i++)
//         {
//             ss << laptop_buf_info.shape[i] << ", ";
//         }
//         ss << ")";
//         throw std::runtime_error(ss.str());
//     }

//     if (laptop_buf_info.format != py::format_descriptor<float>::format())
//         throw std::runtime_error("Input numpy array laptop must have dtype float32.");

//     py::buffer_info phone_buf_info = phone.request();
//     if (phone_buf_info.ndim != phone_input[0].properties.validShape.numDimensions ||
//         phone_buf_info.shape[0] != phone_input[0].properties.validShape.dimensionSize[0] ||
//         phone_buf_info.shape[1] != phone_input[0].properties.validShape.dimensionSize[1] ||
//         phone_buf_info.shape[2] != phone_input[0].properties.validShape.dimensionSize[2] ||
//         phone_buf_info.shape[3] != phone_input[0].properties.validShape.dimensionSize[3])
//     {
//         std::stringstream ss;
//         ss << "wrong input numpy array phone. need: (";
//         for (int32_t i = 0; i < phone_input[0].properties.validShape.numDimensions; i++)
//         {
//             ss << phone_input[0].properties.validShape.dimensionSize[i] << ", ";
//         }
//         ss << "), got: (";
//         for (int32_t i = 0; i < laptop_buf_info.ndim; i++)
//         {
//             ss << phone_buf_info.shape[i] << ", ";
//         }
//         ss << ")";
//         throw std::runtime_error(ss.str());
//     }

//     if (phone_buf_info.format != py::format_descriptor<float>::format())
//         throw std::runtime_error("Input numpy array phone must have dtype float32.");

//     // 启动 laptop 的 VisonEncoder 特征推理
//     float *laptop_np_ptr = reinterpret_cast<float *>(laptop_buf_info.ptr);
//     float *laptop_hbTensor_ptr = reinterpret_cast<float *>(laptop_input[0].sysMem.virAddr);
//     std::memcpy(laptop_hbTensor_ptr, laptop_np_ptr, laptop_buf_info.size * sizeof(float));

//     for (int i = 0; i < output_count; i++)
//     {
//         hbUCPMemFlush(&laptop_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     hbUCPTaskHandle_t laptop_task_handle{nullptr};

//     RDK_CHECK_SUCCESS(
//         hbDNNInferV2(&laptop_task_handle, laptop_output.data(), laptop_input.data(), dnn_handle),
//         "BPU ACT Policy Vision Encoder laptop_task hbDNNInferV2 failed");
//     hbUCPSchedParam laptop_ctrl_param;

//     HB_UCP_INITIALIZE_SCHED_PARAM(&laptop_ctrl_param);
//     laptop_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
//     RDK_CHECK_SUCCESS(hbUCPSubmitTask(laptop_task_handle, &laptop_ctrl_param),
//                       "BPU ACT Policy Vision Encoder laptop_task hbUCPSubmitTask failed");

//     // 启动 phone 的 VisonEncoder 特征推理
//     float *phone_np_ptr = reinterpret_cast<float *>(phone_buf_info.ptr);
//     float *phone_hbTensor_ptr = reinterpret_cast<float *>(phone_input[0].sysMem.virAddr);
//     std::memcpy(phone_hbTensor_ptr, phone_np_ptr, phone_buf_info.size * sizeof(float));

//     for (int i = 0; i < output_count; i++)
//     {
//         hbUCPMemFlush(&phone_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     hbUCPTaskHandle_t phone_task_handle{nullptr};

//     RDK_CHECK_SUCCESS(
//         hbDNNInferV2(&phone_task_handle, phone_output.data(), phone_input.data(), dnn_handle),
//         "BPU ACT Policy Vision Encoder phone_task hbDNNInferV2 failed");
//     hbUCPSchedParam phone_ctrl_param;

//     HB_UCP_INITIALIZE_SCHED_PARAM(&phone_ctrl_param);
//     phone_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
//     RDK_CHECK_SUCCESS(hbUCPSubmitTask(phone_task_handle, &phone_ctrl_param),
//                       "BPU ACT Policy Vision Encoder phone_task hbUCPSubmitTask failed");

//     // 为 TransformerLayers 准备输入数据
//     float *state_np_ptr = reinterpret_cast<float *>(state_buf_info.ptr);
//     float *state_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_input[0].sysMem.virAddr);
//     std::memcpy(state_hbTensor_ptr, state_np_ptr, state_buf_info.size * sizeof(float));

//     // 等待 laptop 的推理结束
//     RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(laptop_task_handle, 0),
//                       "BPU ACT Policy Vision Encoder laptop_task hbUCPWaitTaskDone failed");

//     for (int i = 0; i < output_count; i++)
//     {
//         hbUCPMemFlush(&laptop_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     // laptop_hbTensor_ptr = reinterpret_cast<float *>(laptop_output[0].sysMem.virAddr);
//     RDK_CHECK_SUCCESS(hbUCPReleaseTask(laptop_task_handle), "BPU ACT Policy Vision Encoder laptop_task hbUCPReleaseTask failed");

//     // 等待 phone 的推理结束
//     RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(phone_task_handle, 0),
//                       "BPU ACT Policy Vision Encoder phone_task hbUCPWaitTaskDone failed");

//     for (int i = 0; i < output_count; i++)
//     {
//         hbUCPMemFlush(&phone_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     // phone_hbTensor_ptr = reinterpret_cast<float *>(phone_output[0].sysMem.virAddr);
//     RDK_CHECK_SUCCESS(hbUCPReleaseTask(phone_task_handle), "BPU ACT Policy Vision Encoder phone_task hbUCPReleaseTask failed");

//     // 启动 TransformerLayers 的推理
//     for (int i = 0; i < transformerLayers_input_count; i++)
//     {
//         hbUCPMemFlush(&transformerLayers_input[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     hbUCPTaskHandle_t transformerLayers_task_handle{nullptr};
//     RDK_CHECK_SUCCESS(
//         hbDNNInferV2(&transformerLayers_task_handle, transformerLayers_output.data(), transformerLayers_input.data(), transformerLayers_dnn_handle),
//         "BPU ACT Policy TransformerLayers phone_task hbDNNInferV2 failed");

//     hbUCPSchedParam transformerLayers_ctrl_param;
//     HB_UCP_INITIALIZE_SCHED_PARAM(&transformerLayers_ctrl_param);
//     phone_ctrl_param.backend = HB_UCP_BPU_CORE_ANY;
//     RDK_CHECK_SUCCESS(hbUCPSubmitTask(transformerLayers_task_handle, &transformerLayers_ctrl_param),
//                       "BPU ACT Policy TransformerLayers hbUCPSubmitTask failed");

//     // 等待 TransformerLayers 推理结束
//     RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(transformerLayers_task_handle, 0),
//                       "BPU ACT Policy TransformerLayers hbUCPWaitTaskDone failed");

//     for (int i = 0; i < transformerLayers_output_count; i++)
//     {
//         hbUCPMemFlush(&transformerLayers_output[i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);
//     }

//     auto actions_hbTensor_ptr = reinterpret_cast<float *>(transformerLayers_output[0].sysMem.virAddr);
//     RDK_CHECK_SUCCESS(hbUCPReleaseTask(transformerLayers_task_handle), "BPU ACT Policy TransformerLayers hbUCPReleaseTask failed");

//     // 返回多维 NumPy 数组

//     return py::array(shape, actions_hbTensor_ptr);
// }
