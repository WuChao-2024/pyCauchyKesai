#include "cauchykesai/pycauchykesai.h"
#include <cassert>
#include <iostream>
#include <algorithm>

bool checkFileExists(const std::string &path)
{
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
}

std::string dtype_ucp2str(hbDNNTensorProperties properties)
{
    if (properties.tensorType == HB_DNN_TENSOR_TYPE_S8) return "int8";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U8) return "uint8";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F16) return "float16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S16) return "int16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U16) return "uint16";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F32) return "float32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S32) return "int32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U32) return "uint32";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_F64) return "float64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_S64) return "int64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_U64) return "uint64";
    else if (properties.tensorType == HB_DNN_TENSOR_TYPE_BOOL8) return "bool";
    else return "unknown";
}

// ══════════════════════════════════════════════════════════════════════════════
// dtype 缓存：避免每次调用时从字符串构造 py::dtype
// ══════════════════════════════════════════════════════════════════════════════
static const py::dtype DTYPE_FLOAT16 = py::dtype("float16");
static const py::dtype DTYPE_FLOAT32 = py::dtype::of<float>();
static const py::dtype DTYPE_INT32   = py::dtype::of<int32_t>();

std::string dtype_np2str(const py::dtype &dt)
{
    if (dt.is(py::dtype::of<float>())) return "float32";
    if (dt.is(py::dtype::of<double>())) return "float64";
    if (dt.is(py::dtype::of<int8_t>())) return "int8";
    if (dt.is(py::dtype::of<uint8_t>())) return "uint8";
    if (dt.is(py::dtype::of<int16_t>())) return "int16";
    if (dt.is(py::dtype::of<uint16_t>())) return "uint16";
    if (dt.is(py::dtype::of<int32_t>())) return "int32";
    if (dt.is(py::dtype::of<uint32_t>())) return "uint32";
    if (dt.is(py::dtype::of<int64_t>())) return "int64";
    if (dt.is(py::dtype::of<uint64_t>())) return "uint64";
    if (dt.equal(DTYPE_FLOAT16)) return "float16";
    if (dt.is(py::dtype::of<bool>())) return "bool";
    return "unknown";
}
py::dtype dtype_str2np(const std::string &dtype_str)
{
    if (dtype_str == "float32") return py::dtype::of<float>();
    if (dtype_str == "float64") return py::dtype::of<double>();
    if (dtype_str == "int8") return py::dtype::of<int8_t>();
    if (dtype_str == "uint8") return py::dtype::of<uint8_t>();
    if (dtype_str == "int16") return py::dtype::of<int16_t>();
    if (dtype_str == "uint16") return py::dtype::of<uint16_t>();
    if (dtype_str == "int32") return py::dtype::of<int32_t>();
    if (dtype_str == "uint32") return py::dtype::of<uint32_t>();
    if (dtype_str == "int64") return py::dtype::of<int64_t>();
    if (dtype_str == "uint64") return py::dtype::of<uint64_t>();
    if (dtype_str == "float16") return DTYPE_FLOAT16;
    if (dtype_str == "bool") return py::dtype::of<bool>();
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

// ─────────────────────────────────────────────────────────────────────────────
// 私有辅助：加载模型并读取输入/输出元信息（不分配 ION 内存）
// ─────────────────────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
// 公共辅助：补齐 aligned stride 并计算 alignedByteSize
// 消除 6 处重复代码，带边界安全检查。替换了原有的 assert + 裸循环。
// ══════════════════════════════════════════════════════════════════════════════
void CauchyKesai::fixup_aligned_stride(hbDNNTensorProperties &props)
{
    if (props.alignedByteSize >= 0) return;
    for (int32_t dim_i = props.validShape.numDimensions - 1; dim_i >= 0; --dim_i)
    {
        if (props.stride[dim_i] == -1)
        {
            // 最后一维 stride = 元素大小，不会是 -1
            if (dim_i + 1 >= props.validShape.numDimensions) continue;
            auto cur_stride = props.stride[dim_i + 1] * props.validShape.dimensionSize[dim_i + 1];
            props.stride[dim_i] = ALIGN_BPU(cur_stride);
        }
    }
    props.alignedByteSize = props.stride[0] * props.validShape.dimensionSize[0];
}

void CauchyKesai::_init_model(const std::string &model_path, int32_t model_cnt_select)
{
    auto warnings = py::module_::import("warnings");

    if (!checkFileExists(model_path))
        throw std::runtime_error("Error: Model Path does not exist or is not a file: " + model_path);

    model_path_ = model_path;
    const char *modelFileName = model_path_.c_str();
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &modelFileName, 1),
        "hbDNNInitializeFromFiles failed");

    // ── packed_dnn_handle 有效；后续任何 API 失败需释放 ──
    try {
    model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    if (model_count == 0)
        throw std::runtime_error("Packed model contains 0 models");

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

    // ── 运行时环境信息 ──
    dnn_version_ = hbDNNGetVersion() ? hbDNNGetVersion() : "unknown";
    bpu_version_ = hbUCPGetVersion() ? hbUCPGetVersion() : "unknown";
    soc_name_    = hbUCPGetSocName() ? hbUCPGetSocName() : "unknown";

    // ── 模型级信息 ──
    bpu_core_num_ = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetCompileBpuCoreNum(&bpu_core_num_, dnn_handle),
        "hbDNNGetCompileBpuCoreNum failed");

    {
        const char *desc = nullptr;
        uint32_t size = 0;
        int32_t type = 0;
        if (hbDNNGetModelDesc(&desc, &size, &type, dnn_handle) == 0 &&
            type == HB_DNN_DESC_TYPE_STRING && desc != nullptr && size > 0)
            model_desc_ = std::string(desc, size);
        else
            model_desc_ = "N/A";
    }

    // ── 输入元信息 ──
    mbs = 0;
    input_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "hbDNNGetInputCount failed");
    inputs_shape.resize(input_count);

    for (int32_t i = 0; i < input_count; i++)
    {
        hbDNNTensorProperties input_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&input_properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        inputs_dtype.push_back(dtype_ucp2str(input_properties));

        char const *input_name;
        RDK_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
        inputs_name.push_back(std::string(input_name));

        inputs_numDimension.push_back(input_properties.validShape.numDimensions);
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
            inputs_shape[i].push_back(input_properties.validShape.dimensionSize[j]);

        // ── 额外 TensorProperties 字段 ──
        inputs_alignedByteSize.push_back(input_properties.alignedByteSize);
        inputs_tensorType.push_back(input_properties.tensorType);
        inputs_quantiType.push_back(input_properties.quantiType);
        inputs_quantizeAxis.push_back(input_properties.quantizeAxis);
        inputs_scaleLen.push_back(input_properties.scale.scaleLen);
        {
            std::vector<float> sv;
            for (int32_t j = 0; j < input_properties.scale.scaleLen; j++)
                sv.push_back(input_properties.scale.scaleData[j]);
            inputs_scaleData.push_back(sv);
        }
        inputs_zeroPointLen.push_back(input_properties.scale.zeroPointLen);
        {
            std::vector<int32_t> zv;
            for (int32_t j = 0; j < input_properties.scale.zeroPointLen; j++)
                zv.push_back(input_properties.scale.zeroPointData[j]);
            inputs_zeroPointData.push_back(zv);
        }
        {
            std::vector<int64_t> stride_vec;
            for (int32_t j = 0; j < inputs_numDimension[i]; j++)
                stride_vec.push_back(input_properties.stride[j]);
            inputs_stride.push_back(stride_vec);
        }

        // ── InputDesc ──
        {
            const char *desc = nullptr;
            uint32_t size = 0;
            int32_t type = 0;
            if (hbDNNGetInputDesc(&desc, &size, &type, dnn_handle, i) == 0 &&
                type == HB_DNN_DESC_TYPE_STRING && desc != nullptr && size > 0)
                inputs_desc.push_back(std::string(desc, size));
            else
                inputs_desc.push_back("N/A");
        }
    }

    // ── 输出元信息 ──
    output_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "hbDNNGetOutputCount failed");
    outputs_shape.resize(output_count);

    for (int32_t i = 0; i < output_count; i++)
    {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        outputs_dtype.push_back(dtype_ucp2str(output_properties));

        char const *output_name;
        RDK_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i), "hbDNNGetOutputName failed");
        outputs_name.push_back(std::string(output_name));

        outputs_numDimension.push_back(output_properties.validShape.numDimensions);
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            outputs_shape[i].push_back(output_properties.validShape.dimensionSize[j]);

        // ── 额外 TensorProperties 字段 ──
        outputs_alignedByteSize.push_back(output_properties.alignedByteSize);
        outputs_tensorType.push_back(output_properties.tensorType);
        outputs_quantiType.push_back(output_properties.quantiType);
        outputs_quantizeAxis.push_back(output_properties.quantizeAxis);
        outputs_scaleLen.push_back(output_properties.scale.scaleLen);
        {
            std::vector<float> sv;
            for (int32_t j = 0; j < output_properties.scale.scaleLen; j++)
                sv.push_back(output_properties.scale.scaleData[j]);
            outputs_scaleData.push_back(sv);
        }
        outputs_zeroPointLen.push_back(output_properties.scale.zeroPointLen);
        {
            std::vector<int32_t> zv;
            for (int32_t j = 0; j < output_properties.scale.zeroPointLen; j++)
                zv.push_back(output_properties.scale.zeroPointData[j]);
            outputs_zeroPointData.push_back(zv);
        }
        {
            std::vector<int64_t> stride_vec;
            for (int32_t j = 0; j < outputs_numDimension[i]; j++)
                stride_vec.push_back(output_properties.stride[j]);
            outputs_stride.push_back(stride_vec);
        }

        // ── OutputDesc ──
        {
            const char *desc = nullptr;
            uint32_t size = 0;
            int32_t type = 0;
            if (hbDNNGetOutputDesc(&desc, &size, &type, dnn_handle, i) == 0 &&
                type == HB_DNN_DESC_TYPE_STRING && desc != nullptr && size > 0)
                outputs_desc.push_back(std::string(desc, size));
            else
                outputs_desc.push_back("N/A");
        }
    }

    input_names  = inputs_name;
    output_names = outputs_name;
    } catch (...) {
        hbDNNRelease(packed_dnn_handle);
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 标准构造：自动分配 n_task 组 ION 输入/输出内存（通过 IONArray 统一管理）
// ─────────────────────────────────────────────────────────────────────────────
CauchyKesai::CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select)
{
    auto warnings = py::module_::import("warnings");

    if (n_task <= 0) { n_task = 1; warnings.attr("warn")("n_task <= 0, clamped to 1", py::module_::import("builtins").attr("UserWarning")); }
    else if (n_task > 32) { n_task = 32; warnings.attr("warn")("n_task > 32, clamped to 32", py::module_::import("builtins").attr("UserWarning")); }
    n_task_ = n_task;
    no_alloc_mode_ = false;

    inputs_hbTensor.resize(n_task_);
    outputs_hbTensor.resize(n_task_);
    task_handles.resize(n_task_);
    is_infer = std::vector<std::atomic<int>>(n_task_);

    _init_model(model_path, model_cnt_select);

    try {
    // ── 为每个任务槽创建 IONArray 输入内存 ──
    owned_ion_inputs_.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        inputs_hbTensor[t].resize(input_count);

    for (int32_t i = 0; i < input_count; i++)
    {
        // 先获取 properties 和计算 alignedByteSize（需要知道分配大小）
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        fixup_aligned_stride(props);
        for (int32_t t = 0; t < n_task_; t++)
        {
            // 用 IONArray 替代 hbUCPMallocCached
            std::vector<ssize_t> shape_vec(inputs_shape[i].begin(), inputs_shape[i].end());
            auto ion = std::make_shared<IONArray>(
                dtype_str2np(inputs_dtype[i]), shape_vec,
                static_cast<uint64_t>(props.alignedByteSize), true);
            owned_ion_inputs_[t].push_back(ion);
            mbs += double(props.alignedByteSize) / 1024.0 / 1024.0;
        }
    }

    // ── 为每个任务槽创建 IONArray 输出内存 ──
    owned_ion_outputs_.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        outputs_hbTensor[t].resize(output_count);

    for (int32_t i = 0; i < output_count; i++)
    {
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        fixup_aligned_stride(props);

        for (int32_t t = 0; t < n_task_; t++)
        {
            std::vector<ssize_t> shape_vec(outputs_shape[i].begin(), outputs_shape[i].end());
            auto ion = std::make_shared<IONArray>(
                dtype_str2np(outputs_dtype[i]), shape_vec,
                static_cast<uint64_t>(props.alignedByteSize), true);
            owned_ion_outputs_[t].push_back(ion);
            mbs += double(props.alignedByteSize) / 1024.0 / 1024.0;
        }
    }

    // ── 异步推理标志初始化 ──
    for (int32_t t = 0; t < n_task_; t++)
        is_infer[t].store(0, std::memory_order_relaxed);

    // ── 通过公共绑定方法统一绑定（与 set_inputs/set_outputs 走同一代码路径）──
    input_tensors.resize(n_task_);
    output_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
    {
        std::vector<IONArray *> in_ptrs, out_ptrs;
        for (auto &sp : owned_ion_inputs_[t])  in_ptrs.push_back(sp.get());
        for (auto &sp : owned_ion_outputs_[t]) out_ptrs.push_back(sp.get());
        _bind_ion_inputs(in_ptrs, t);
        _bind_ion_outputs(out_ptrs, t);
    }
    } catch (...) {
        hbDNNRelease(packed_dnn_handle);
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 零内存构造：用 defer IONArray 占位每个槽位，不实际分配 ION 内存。
//
// 占位设计解决了空内层 vector 导致的 UB：start() 守卫代码通过检查
// virAddr==nullptr 来判断是否已 set_inputs，但若内层 vector 未 resize，
// operator[] 本身就是 UB。用 defer IONArray 占位后，内层 vector 被 resize，
// operator[] 安全，virAddr==nullptr 的检查正常工作。
//
// set_inputs()/set_outputs() 再用用户的 IONArray 替换占位。
// ─────────────────────────────────────────────────────────────────────────────
CauchyKesai::CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select, bool _no_alloc)
{
    auto warnings = py::module_::import("warnings");

    if (n_task <= 0) { n_task = 1; warnings.attr("warn")("n_task <= 0, clamped to 1", py::module_::import("builtins").attr("UserWarning")); }
    else if (n_task > 32) { n_task = 32; warnings.attr("warn")("n_task > 32, clamped to 32", py::module_::import("builtins").attr("UserWarning")); }
    n_task_ = n_task;
    no_alloc_mode_ = true;

    inputs_hbTensor.resize(n_task_);
    outputs_hbTensor.resize(n_task_);
    task_handles.resize(n_task_);
    is_infer = std::vector<std::atomic<int>>(n_task_);

    _init_model(model_path, model_cnt_select);

    try {
    // ── 用 defer IONArray 占位输入槽 ──
    owned_ion_inputs_.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        inputs_hbTensor[t].resize(input_count);  // ← 消除空内层 vector UB

    for (int32_t i = 0; i < input_count; i++)
    {
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        fixup_aligned_stride(props);
        for (int32_t t = 0; t < n_task_; t++)
        {
            std::vector<ssize_t> shape_vec(inputs_shape[i].begin(), inputs_shape[i].end());
            auto ion = std::make_shared<IONArray>(
                dtype_str2np(inputs_dtype[i]), shape_vec,
                /*cached=*/true, /*defer=*/true);  // 只记录 dtype+shape，不分配
            owned_ion_inputs_[t].push_back(ion);
        }
    }

    // ── 用 defer IONArray 占位输出槽 ──
    owned_ion_outputs_.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
        outputs_hbTensor[t].resize(output_count);

    for (int32_t i = 0; i < output_count; i++)
    {
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        fixup_aligned_stride(props);

        for (int32_t t = 0; t < n_task_; t++)
        {
            std::vector<ssize_t> shape_vec(outputs_shape[i].begin(), outputs_shape[i].end());
            auto ion = std::make_shared<IONArray>(
                dtype_str2np(outputs_dtype[i]), shape_vec,
                /*cached=*/true, /*defer=*/true);
            owned_ion_outputs_[t].push_back(ion);
        }
    }

    for (int32_t t = 0; t < n_task_; t++)
        is_infer[t].store(0, std::memory_order_relaxed);

    // ── 与标准构造走同一绑定路径 ──
    input_tensors.resize(n_task_);
    output_tensors.resize(n_task_);
    for (int32_t t = 0; t < n_task_; t++)
    {
        std::vector<IONArray *> in_ptrs, out_ptrs;
        for (auto &sp : owned_ion_inputs_[t])  in_ptrs.push_back(sp.get());
        for (auto &sp : owned_ion_outputs_[t]) out_ptrs.push_back(sp.get());
        _bind_ion_inputs(in_ptrs, t);
        _bind_ion_outputs(out_ptrs, t);
    }
    } catch (...) {
        hbDNNRelease(packed_dnn_handle);
        throw;
    }
}

CauchyKesai::~CauchyKesai()
{
    // 清理所有挂起的 BPU 任务（用户调 start() 后未调 wait() 的场景）
    for (int32_t t = 0; t < n_task_; t++)
    {
        if (is_infer[t].load(std::memory_order_acquire))
        {
            try { hbUCPWaitTaskDone(task_handles[t], 0); } catch (...) {}
            try { hbUCPReleaseTask(task_handles[t]); } catch (...) {}
            is_infer[t].store(0, std::memory_order_release);
        }
    }
    std::cout << "[INFO] release model " << name_list[model_cnt_select_]
              << " success." << std::endl;
    owned_ion_inputs_.clear();
    owned_ion_outputs_.clear();
    hbDNNRelease(packed_dnn_handle);
}

bool CauchyKesai::is_busy(int32_t task_id) const
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));
    return is_infer[task_id].load(std::memory_order_acquire) != 0;
}

py::dict CauchyKesai::s()
{
    py::list model_names_list;
    for (int32_t i = 0; i < model_count; i++)
    {
        py::dict entry;
        entry["index"] = i;
        entry["name"]  = std::string(name_list[i]);
        entry["selected"] = (i == model_cnt_select_);
        model_names_list.append(entry);
    }

    py::list inputs_list;
    for (int32_t i = 0; i < input_count; i++)
    {
        py::dict inp;
        inp["index"]           = i;
        inp["name"]            = inputs_name[i];
        inp["dtype"]           = inputs_dtype[i];
        inp["tensorType"]      = inputs_tensorType[i];
        inp["alignedByteSize"] = inputs_alignedByteSize[i];
        inp["quantiType"]      = inputs_quantiType[i];
        inp["quantizeAxis"]    = inputs_quantizeAxis[i];
        inp["desc"]            = inputs_desc[i];
        py::list shape;
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
            shape.append(inputs_shape[i][j]);
        inp["shape"] = shape;
        py::list stride;
        for (int32_t j = 0; j < inputs_numDimension[i]; j++)
            stride.append(inputs_stride[i][j]);
        inp["stride"] = stride;

        // ── 量化参数: 按 quantizeAxis 广播成 numpy array ──
        int32_t ndim = inputs_numDimension[i];
        int32_t axis = inputs_quantizeAxis[i];
        int32_t slen = inputs_scaleLen[i];
        int32_t zlen = inputs_zeroPointLen[i];

        // 构建 broadcast shape: 所有维度为 1，quantizeAxis 位置为 scaleLen/zeroPointLen
        // quantiType=NONE 时 scaleLen=0 → 返回空 array
        if (slen > 0)
        {
            std::vector<ssize_t> bshape(ndim, 1);
            if (axis >= 0 && axis < ndim) bshape[axis] = slen;
            else bshape[0] = slen;  // fallback: axis 无效时放第 0 维
            inp["scale"] = py::array(DTYPE_FLOAT32, bshape, inputs_scaleData[i].data());
        }
        else
        {
            inp["scale"] = py::array(DTYPE_FLOAT32, 0);  // 空 float32 array
        }

        if (zlen > 0)
        {
            std::vector<ssize_t> bshape(ndim, 1);
            if (axis >= 0 && axis < ndim) bshape[axis] = zlen;
            else bshape[0] = zlen;
            inp["zero_point"] = py::array(DTYPE_INT32, bshape, inputs_zeroPointData[i].data());
        }
        else
        {
            inp["zero_point"] = py::array(DTYPE_INT32, 0);  // 空 int32 array
        }

        inputs_list.append(inp);
    }

    py::list outputs_list;
    for (int32_t i = 0; i < output_count; i++)
    {
        py::dict out;
        out["index"]           = i;
        out["name"]            = outputs_name[i];
        out["dtype"]           = outputs_dtype[i];
        out["tensorType"]      = outputs_tensorType[i];
        out["alignedByteSize"] = outputs_alignedByteSize[i];
        out["quantiType"]      = outputs_quantiType[i];
        out["quantizeAxis"]    = outputs_quantizeAxis[i];
        out["desc"]            = outputs_desc[i];
        py::list shape;
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            shape.append(outputs_shape[i][j]);
        out["shape"] = shape;
        py::list stride;
        for (int32_t j = 0; j < outputs_numDimension[i]; j++)
            stride.append(outputs_stride[i][j]);
        out["stride"] = stride;

        // ── 量化参数: 按 quantizeAxis 广播成 numpy array ──
        int32_t ndim = outputs_numDimension[i];
        int32_t axis = outputs_quantizeAxis[i];
        int32_t slen = outputs_scaleLen[i];
        int32_t zlen = outputs_zeroPointLen[i];

        if (slen > 0)
        {
            std::vector<ssize_t> bshape(ndim, 1);
            if (axis >= 0 && axis < ndim) bshape[axis] = slen;
            else bshape[0] = slen;
            out["scale"] = py::array(DTYPE_FLOAT32, bshape, outputs_scaleData[i].data());
        }
        else
        {
            out["scale"] = py::array(DTYPE_FLOAT32, 0);
        }

        if (zlen > 0)
        {
            std::vector<ssize_t> bshape(ndim, 1);
            if (axis >= 0 && axis < ndim) bshape[axis] = zlen;
            else bshape[0] = zlen;
            out["zero_point"] = py::array(DTYPE_INT32, bshape, outputs_zeroPointData[i].data());
        }
        else
        {
            out["zero_point"] = py::array(DTYPE_INT32, 0);
        }

        outputs_list.append(out);
    }

    py::dict result;
    result["model_path"]   = model_path_;
    result["model_names"]  = model_names_list;
    result["n_task"]       = n_task_;
    result["memory_mb"]    = mbs;
    result["dnn_version"]  = dnn_version_;
    result["bpu_version"]  = bpu_version_;
    result["soc_name"]     = soc_name_;
    result["model_desc"]   = model_desc_;
    result["bpu_core_num"] = bpu_core_num_;
    result["inputs"]       = inputs_list;
    result["outputs"]      = outputs_list;
    return result;
}

void CauchyKesai::set_scheduling_params(const std::vector<int32_t> &bpu_cores)
{
    if (bpu_cores.empty()) { backend_ = HB_UCP_BPU_CORE_ANY; return; }
    uint64_t mask = 0;
    for (auto core : bpu_cores)
    {
        if (core < 0 || core > 3)
            throw std::invalid_argument("bpu_cores must contain values in [0, 3]");
        mask |= (1ULL << core);
    }
    backend_ = mask;
}

py::dict CauchyKesai::t()
{
    int32_t task_id = 0;
    int expected = 0;
    if (!is_infer[task_id].compare_exchange_strong(expected, 1,
            std::memory_order_acquire, std::memory_order_relaxed))
        throw std::runtime_error("task 0 is already in use, cannot run benchmark t()");
    auto tp_start = std::chrono::system_clock::now();

    hbUCPTaskHandle_t task_handle{nullptr};
    bool handle_submitted = false;
    bool handle_released = false;
    try {
        RDK_CHECK_SUCCESS(
            hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
            "hbDNNInferV2 failed");
        hbUCPSchedParam ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.priority = HB_UCP_PRIORITY_LOWEST;
        ctrl_param.backend = backend_;
        task_handles[task_id] = task_handle;
        handle_submitted = true;
        {
            py::gil_scoped_release release;
            RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "hbUCPSubmitTask failed");
        }

        {
            py::gil_scoped_release release;
            RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0), "hbUCPWaitTaskDone failed");
        }

        for (int i = 0; i < output_count; i++)
            hbUCPMemFlush(&outputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);

        RDK_CHECK_SUCCESS(hbUCPReleaseTask(task_handles[task_id]), "hbUCPReleaseTask failed");
        handle_released = true;
    } catch (...) {
        if (!handle_released && handle_submitted && task_handles[task_id] != nullptr) {
            try { hbUCPReleaseTask(task_handles[task_id]); } catch (...) {}
            task_handles[task_id] = nullptr;
        } else if (!handle_released && task_handle != nullptr) {
            try { hbUCPReleaseTask(task_handle); } catch (...) {}
        }
        is_infer[task_id].store(0, std::memory_order_release);
        throw;
    }
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

void CauchyKesai::safe_start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // ── 0. 输入数量校验 (零拷贝路径: 空 inputs 跳过校验) ──
    if (!inputs.empty() && (int32_t)inputs.size() != input_count)
        throw std::invalid_argument("input count mismatch: expected " +
            std::to_string(input_count) + ", got " + std::to_string(inputs.size()));

    // ── 1. task_id / priority 边界检查 ──
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));
    if (priority < 0 || priority > 255)
        throw std::out_of_range("priority out of range: got " + std::to_string(priority));

    // ── 2. no_alloc 模式 ION 内存绑定检查 ──
    if (no_alloc_mode_)
    {
        if (task_id >= (int32_t)inputs_hbTensor.size())
            throw std::runtime_error("task_id has no ION memory bound. Call set_inputs() first.");
        for (int32_t i = 0; i < input_count; i++)
        {
            if (inputs_hbTensor[task_id][i].sysMem.virAddr == nullptr)
                throw std::runtime_error("input[" + std::to_string(i) + "] has no ION memory bound. Call set_inputs() first.");
        }
        if (task_id >= (int32_t)outputs_hbTensor.size())
            throw std::runtime_error("task_id has no ION output memory bound. Call set_outputs() first.");
        for (int32_t i = 0; i < output_count; i++)
        {
            if (outputs_hbTensor[task_id][i].sysMem.virAddr == nullptr)
                throw std::runtime_error("output[" + std::to_string(i) + "] has no ION memory bound. Call set_outputs() first.");
        }
    }

    // ── 3. 委托给 start()（start() 负责原子标志设置 + 推理提交）──
    start(inputs, task_id, priority);
}

void CauchyKesai::start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // ── 1. 原子获取任务槽 ──
    int expected = 0;
    if (!is_infer[task_id].compare_exchange_strong(expected, 1,
            std::memory_order_acquire, std::memory_order_relaxed))
        throw std::runtime_error("task_id " + std::to_string(task_id) + " is already in use");

    hbUCPTaskHandle_t task_handle{nullptr};
    bool handle_stored = false;
    try {
        if (!inputs.empty())
        {
            for (int32_t i = 0; i < input_count; i++)
            {
                py::array contiguous_input = py::array::ensure(inputs[i], py::array::c_style);
                size_t input_bytes = (size_t)contiguous_input.nbytes();
                size_t mem_bytes   = (size_t)inputs_hbTensor[task_id][i].sysMem.memSize;
                if (input_bytes > mem_bytes)
                    throw std::runtime_error(
                        "input[" + std::to_string(i) + "] too large for ION buffer: " +
                        std::to_string(input_bytes) + " bytes > " +
                        std::to_string(mem_bytes) + " bytes");
                std::memcpy(inputs_hbTensor[task_id][i].sysMem.virAddr, contiguous_input.data(), input_bytes);
            }
        }
        for (int32_t i = 0; i < input_count; i++)
            hbUCPMemFlush(&inputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_CLEAN);

        RDK_CHECK_SUCCESS(
            hbDNNInferV2(&task_handle, outputs_hbTensor[task_id].data(), inputs_hbTensor[task_id].data(), dnn_handle),
            "hbDNNInferV2 failed");
        hbUCPSchedParam ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.priority = priority;
        ctrl_param.backend = backend_;

        task_handles[task_id] = task_handle;
        handle_stored = true;
        {
            py::gil_scoped_release release;
            RDK_CHECK_SUCCESS(hbUCPSubmitTask(task_handle, &ctrl_param), "hbUCPSubmitTask failed");
        }
    } catch (...) {
        if (handle_stored && task_handles[task_id] != nullptr) {
            try { hbUCPReleaseTask(task_handles[task_id]); } catch (...) {}
            task_handles[task_id] = nullptr;
        } else if (!handle_stored && task_handle != nullptr) {
            try { hbUCPReleaseTask(task_handle); } catch (...) {}
        }
        is_infer[task_id].store(0, std::memory_order_release);
        throw;
    }
}

std::vector<py::array> CauchyKesai::wait(int32_t task_id)
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));
    int expected = 1;
    if (!is_infer[task_id].compare_exchange_strong(expected, 2,
            std::memory_order_acquire, std::memory_order_relaxed))
    {
        if (expected == 0)
            throw std::runtime_error("task_id " + std::to_string(task_id) + " is not in use");
        else
            throw std::runtime_error("task_id " + std::to_string(task_id) + " is already being waited");
    }
    bool handle_released = false;
    try {
        {
            py::gil_scoped_release release;
            RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], 0), "hbUCPWaitTaskDone failed");
        }
        for (int i = 0; i < output_count; i++)
            hbUCPMemFlush(&outputs_hbTensor[task_id][i].sysMem, HB_SYS_MEM_CACHE_INVALIDATE);

        RDK_CHECK_SUCCESS(hbUCPReleaseTask(task_handles[task_id]), "hbUCPReleaseTask failed");
        handle_released = true;
    } catch (...) {
        if (!handle_released && task_handles[task_id] != nullptr) {
            try { hbUCPReleaseTask(task_handles[task_id]); } catch (...) {}
        }
        is_infer[task_id].store(0, std::memory_order_release);
        throw;
    }
    is_infer[task_id].store(0, std::memory_order_release);

    std::vector<py::array> rs;
    for (int32_t i = 0; i < output_count; i++)
    {
        // 标准模式: 通过 capsule 持有 IONArray shared_ptr，防止 CauchyKesai 析构后悬空指针
        // no-alloc 模式: 用户自行管理 IONArray 生命周期
        py::object base = py::none();
        if (!no_alloc_mode_ && task_id < (int32_t)owned_ion_outputs_.size()
            && i < (int32_t)owned_ion_outputs_[task_id].size())
        {
            auto sp = owned_ion_outputs_[task_id][i];
            base = py::capsule(
                new std::shared_ptr<IONArray>(sp),
                [](void *p) { delete static_cast<std::shared_ptr<IONArray>*>(p); });
        }
        rs.push_back(py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i],
                               outputs_hbTensor[task_id][i].sysMem.virAddr, base));
    }
    return rs;
}

std::vector<py::array> CauchyKesai::inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority)
{
    // ── input 数据校验: 数量 / dtype / ndim / shape ──
    if ((int32_t)inputs.size() != input_count)
        throw std::invalid_argument("input count mismatch: expected " + std::to_string(input_count) + ", got " + std::to_string(inputs.size()));

    for (int32_t cnt = 0; cnt < input_count; cnt++)
    {
        std::string got_dtype = dtype_np2str(inputs[cnt].dtype());
        if (inputs_dtype[cnt] != got_dtype)
            throw std::invalid_argument("dtype mismatch at input[" + std::to_string(cnt) + "]");
        if (inputs[cnt].ndim() != inputs_numDimension[cnt])
            throw std::invalid_argument("ndim mismatch at input[" + std::to_string(cnt) + "]");
        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            if (inputs_shape[cnt][i] != (size_t)inputs[cnt].shape()[i])
                throw std::invalid_argument("shape mismatch at input[" + std::to_string(cnt) + "]");
        }
    }

    // ── 任务槽校验 + 异步提交 (safe_start) → 等待完成 (wait) ──
    safe_start(inputs, task_id, priority);
    return wait(task_id);
}

// ─────────────────────────────────────────────────────────────────────────────
// set_inputs：为指定任务槽绑定外部 IONArray 输入（零拷贝，仅零内存模式）
// 含 aligned stride 计算 — 与标准构造函数逻辑一致
// ─────────────────────────────────────────────────────────────────────────────
void CauchyKesai::set_inputs(const std::vector<IONArray *> &ion_inputs, int32_t n_task)
{
    if (!no_alloc_mode_)
        throw std::runtime_error("set_inputs() is only available in no-alloc mode");
    _bind_ion_inputs(ion_inputs, n_task);
}

// ─────────────────────────────────────────────────────────────────────────────
// set_outputs：为指定任务槽绑定外部 IONArray 输出（零拷贝，仅零内存模式）
// 含 aligned stride 计算
// ─────────────────────────────────────────────────────────────────────────────
void CauchyKesai::set_outputs(const std::vector<IONArray *> &ion_outputs, int32_t n_task)
{
    if (!no_alloc_mode_)
        throw std::runtime_error("set_outputs() is only available in no-alloc mode");
    _bind_ion_outputs(ion_outputs, n_task);
}

// ─────────────────────────────────────────────────────────────────────────────
// _bind_ion_inputs：公共绑定逻辑 — 校验 + 属性设置 + sysMem 赋值 + numpy 视图
// 供标准构造函数和 set_inputs 共用
// ─────────────────────────────────────────────────────────────────────────────
void CauchyKesai::_bind_ion_inputs(const std::vector<IONArray *> &ion_inputs, int32_t n_task)
{
    if ((int32_t)ion_inputs.size() != input_count)
        throw std::invalid_argument("input count mismatch: expected " +
            std::to_string(input_count) + ", got " + std::to_string(ion_inputs.size()));

    for (int32_t i = 0; i < input_count; i++)
    {
        if (ion_inputs[i] == nullptr)
            throw std::invalid_argument("ion_inputs[" + std::to_string(i) + "] is nullptr");
        std::string got_dtype = dtype_np2str(ion_inputs[i]->dtype());
        if (inputs_dtype[i] != got_dtype)
            throw std::invalid_argument("dtype mismatch at input[" + std::to_string(i) + "]");
        if (ion_inputs[i]->ndim() != inputs_numDimension[i])
            throw std::invalid_argument("ndim mismatch at input[" + std::to_string(i) + "]");
        for (int j = 0; j < inputs_numDimension[i]; j++)
        {
            if ((size_t)ion_inputs[i]->shape()[j] != inputs_shape[i][j])
            {
                std::string expected = "(";
                for (int k = 0; k < inputs_numDimension[i]; k++)
                    expected += std::to_string(inputs_shape[i][k]) + (k + 1 < inputs_numDimension[i] ? ", " : ")");
                std::string got = "(";
                for (int k = 0; k < ion_inputs[i]->ndim(); k++)
                    got += std::to_string(ion_inputs[i]->shape()[k]) + (k + 1 < ion_inputs[i]->ndim() ? ", " : ")");
                throw std::invalid_argument("shape mismatch at input[" + std::to_string(i) + "]: expected " + expected + ", got " + got);
            }
        }
    }

    if (n_task < 0 || n_task >= n_task_)
        throw std::out_of_range("n_task out of range");

    inputs_hbTensor[n_task].resize(input_count);
    input_tensors[n_task].clear();
    for (int32_t i = 0; i < input_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&inputs_hbTensor[n_task][i].properties, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");
        fixup_aligned_stride(inputs_hbTensor[n_task][i].properties);
        inputs_hbTensor[n_task][i].sysMem = ion_inputs[i]->sys_mem();
        {
            py::object base = py::none();
            if (!no_alloc_mode_
                && n_task < (int32_t)owned_ion_inputs_.size()
                && i < (int32_t)owned_ion_inputs_[n_task].size())
            {
                auto sp = owned_ion_inputs_[n_task][i];
                base = py::capsule(
                    new std::shared_ptr<IONArray>(sp),
                    [](void *p) { delete static_cast<std::shared_ptr<IONArray>*>(p); });
            }
            input_tensors[n_task].push_back(
                py::array(dtype_str2np(inputs_dtype[i]), inputs_shape[i],
                          inputs_hbTensor[n_task][i].sysMem.virAddr, base));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// _bind_ion_outputs：公共绑定逻辑 — 校验 + 属性设置 + sysMem 赋值 + numpy 视图
// 供标准构造函数和 set_outputs 共用
// ─────────────────────────────────────────────────────────────────────────────
void CauchyKesai::_bind_ion_outputs(const std::vector<IONArray *> &ion_outputs, int32_t n_task)
{
    if ((int32_t)ion_outputs.size() != output_count)
        throw std::invalid_argument("output count mismatch: expected " +
            std::to_string(output_count) + ", got " + std::to_string(ion_outputs.size()));

    for (int32_t i = 0; i < output_count; i++)
    {
        if (ion_outputs[i] == nullptr)
            throw std::invalid_argument("ion_outputs[" + std::to_string(i) + "] is nullptr");
        std::string got_dtype = dtype_np2str(ion_outputs[i]->dtype());
        if (outputs_dtype[i] != got_dtype)
            throw std::invalid_argument("dtype mismatch at output[" + std::to_string(i) + "]");
        if (ion_outputs[i]->ndim() != outputs_numDimension[i])
            throw std::invalid_argument("ndim mismatch at output[" + std::to_string(i) + "]");
        for (int j = 0; j < outputs_numDimension[i]; j++)
        {
            if ((size_t)ion_outputs[i]->shape()[j] != outputs_shape[i][j])
            {
                std::string expected = "(";
                for (int k = 0; k < outputs_numDimension[i]; k++)
                    expected += std::to_string(outputs_shape[i][k]) + (k + 1 < outputs_numDimension[i] ? ", " : ")");
                std::string got = "(";
                for (int k = 0; k < ion_outputs[i]->ndim(); k++)
                    got += std::to_string(ion_outputs[i]->shape()[k]) + (k + 1 < ion_outputs[i]->ndim() ? ", " : ")");
                throw std::invalid_argument("shape mismatch at output[" + std::to_string(i) + "]: expected " + expected + ", got " + got);
            }
        }
    }

    if (n_task < 0 || n_task >= n_task_)
        throw std::out_of_range("n_task out of range");

    outputs_hbTensor[n_task].resize(output_count);
    output_tensors[n_task].clear();
    for (int32_t i = 0; i < output_count; i++)
    {
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&outputs_hbTensor[n_task][i].properties, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");
        fixup_aligned_stride(outputs_hbTensor[n_task][i].properties);
        outputs_hbTensor[n_task][i].sysMem = ion_outputs[i]->sys_mem();
        {
            py::object base = py::none();
            if (!no_alloc_mode_
                && n_task < (int32_t)owned_ion_outputs_.size()
                && i < (int32_t)owned_ion_outputs_[n_task].size())
            {
                auto sp = owned_ion_outputs_[n_task][i];
                base = py::capsule(
                    new std::shared_ptr<IONArray>(sp),
                    [](void *p) { delete static_cast<std::shared_ptr<IONArray>*>(p); });
            }
            output_tensors[n_task].push_back(
                py::array(dtype_str2np(outputs_dtype[i]), outputs_shape[i],
                          outputs_hbTensor[n_task][i].sysMem.virAddr, base));
        }
    }
}
