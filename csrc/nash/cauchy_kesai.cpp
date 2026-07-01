#include "cauchykesai/pycauchykesai.h"
#include "cauchykesai/platform.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>  // model_desc JSON → compile_config（MIT, vendored）

bool checkFileExists(const std::string &path)
{
    return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);
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
// nlohmann::json → py::object 递归转换
// pybind11 未注册 nlohmann::json 的 type_caster，py::cast(json) 会落到
// type_caster_generic 找不到已注册类而抛异常（被 catch 成 None，即旧 bug）。
// 这里手写递归：object→dict / array→list / number→int|float / bool / str / null.
// ─────────────────────────────────────────────────────────────────────────────
static py::object json_to_py(const nlohmann::json &j)
{
    switch (j.type()) {
        case nlohmann::json::value_t::object: {
            py::dict d;
            for (auto it = j.begin(); it != j.end(); ++it)
                d[py::str(it.key())] = json_to_py(it.value());
            return d;
        }
        case nlohmann::json::value_t::array: {
            py::list l;
            for (const auto &v : j) l.append(json_to_py(v));
            return l;
        }
        case nlohmann::json::value_t::string:
            return py::str(j.get<std::string>());
        case nlohmann::json::value_t::boolean:
            return py::bool_(j.get<bool>());
        case nlohmann::json::value_t::number_integer:
            return py::int_(j.get<int64_t>());
        case nlohmann::json::value_t::number_unsigned:
            return py::int_(j.get<uint64_t>());
        case nlohmann::json::value_t::number_float:
            return py::float_(j.get<double>());
        case nlohmann::json::value_t::null:
        case nlohmann::json::value_t::binary:     // model_desc 不含二进制；兜底 None
        case nlohmann::json::value_t::discarded:
        default:
            return py::none();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 私有辅助：加载模型并读取输入/输出元信息（不分配 ION 内存）
// ─────────────────────────────────────────────────────────────────────────────
// NOTE: 对齐职责（resolve stride -1 / 算 alignedByteSize）已迁回 IONArray
// （IONArray::_resolve_stride_and_aligned_size）。CauchyKesai 不再算对齐。
// ─────────────────────────────────────────────────────────────────────────────

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

    // ── 模型级信息 ──
    bpu_core_num_ = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetCompileBpuCoreNum(&bpu_core_num_, dnn_handle),
        "hbDNNGetCompileBpuCoreNum failed");

    // 物理核数：构造时一次性读 sysfs（平台原子能力 platform.cpp），供 set_scheduling_params 校验。
    physical_core_num_ = cauchykesai::physical_core_count();

    // BPU 对齐值：运行时由 hbUCPGetSocName 检测（S100/S100P=32, S600=64），供 IONArray 模板 resolve stride。
    bpu_align_ = cauchykesai::bpu_align();

    // 默认核调度（用户未显式调 set_scheduling_params 也能直接推理），广播到所有 slot：
    //   多核模型(N≥2) → 前 N 个有效物理核；已知物理核数时上限 N=min(编译核数, 物理核数)。
    //   单核模型     → 留 CORE_ANY，SDK 自动挑核。
    // n_task_ 已在外层构造（见 _init_model 调用前）赋值，此处可安全 resize。
    uint64_t default_mask = HB_UCP_BPU_CORE_ANY;
    if (bpu_core_num_ >= 2) {
        int32_t take = bpu_core_num_;
        if (physical_core_num_ > 0)
            take = std::min(take, physical_core_num_);
        uint64_t mask = 0;
        for (int32_t i = 0; i < take; ++i) mask |= (1ULL << i);
        default_mask = mask;
    }
    backends_.assign(n_task_, default_mask);
    priorities_.assign(n_task_, 0);   // per-slot 优先级默认 0（hbUCPSchedParam.priority = LOWEST）

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

    // ═══════════════════════════════════════════════════════════════════════════
    // 输入/输出元信息 + 建模板（一次遍历：性质全部进 IONArray template，不再并行存 vector）
    // ═══════════════════════════════════════════════════════════════════════════
    mbs = 0;

    input_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetInputCount(&input_count, dnn_handle), "hbDNNGetInputCount failed");
    input_templates_.resize(input_count);
    input_names.resize(input_count);

    for (int32_t i = 0; i < input_count; i++)
    {
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetInputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetInputTensorProperties failed");

        char const *input_name = nullptr;
        RDK_CHECK_SUCCESS(hbDNNGetInputName(&input_name, dnn_handle, i), "hbDNNGetInputName failed");
        std::string name_str(input_name ? input_name : "");

        std::string desc_str = "N/A";
        {
            const char *desc = nullptr;
            uint32_t size = 0;
            int32_t type = 0;
            if (hbDNNGetInputDesc(&desc, &size, &type, dnn_handle, i) == 0 &&
                type == HB_DNN_DESC_TYPE_STRING && desc != nullptr && size > 0)
                desc_str = std::string(desc, size);
        }

        // 建模板：IONArray 构造内 _resolve 自动补 stride -1 / alignedByteSize；
        // name/desc 不在 properties 里，单独 set（CauchyKesai 不算对齐，对齐权威归 IONArray）
        auto tpl = std::make_shared<IONArray>(props, bpu_align_, /*cached=*/true, /*defer=*/true);
        tpl->desc.name = name_str;
        tpl->desc.desc = desc_str;
        input_templates_[i] = tpl;
        input_names[i] = name_str;
        mbs += double(props.alignedByteSize) / 1024.0 / 1024.0;
    }

    output_count = 0;
    RDK_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle), "hbDNNGetOutputCount failed");
    output_templates_.resize(output_count);
    output_names.resize(output_count);

    for (int32_t i = 0; i < output_count; i++)
    {
        hbDNNTensorProperties props;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&props, dnn_handle, i),
            "hbDNNGetOutputTensorProperties failed");

        char const *output_name = nullptr;
        RDK_CHECK_SUCCESS(hbDNNGetOutputName(&output_name, dnn_handle, i), "hbDNNGetOutputName failed");
        std::string name_str(output_name ? output_name : "");

        std::string desc_str = "N/A";
        {
            const char *desc = nullptr;
            uint32_t size = 0;
            int32_t type = 0;
            if (hbDNNGetOutputDesc(&desc, &size, &type, dnn_handle, i) == 0 &&
                type == HB_DNN_DESC_TYPE_STRING && desc != nullptr && size > 0)
                desc_str = std::string(desc, size);
        }

        auto tpl = std::make_shared<IONArray>(props, bpu_align_, /*cached=*/true, /*defer=*/true);
        tpl->desc.name = name_str;
        tpl->desc.desc = desc_str;
        output_templates_[i] = tpl;
        output_names[i] = name_str;
        mbs += double(props.alignedByteSize) / 1024.0 / 1024.0;
    }

    } catch (...) {
        hbDNNRelease(packed_dnn_handle);
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 统一构造：lazy-per-slot，每个 slot 按 desc 建好 input/output IONArray（完善 + allocated）
// ─────────────────────────────────────────────────────────────────────────────
CauchyKesai::CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select)
    : platform_(cauchykesai::global_platform())
{
    auto warnings = py::module_::import("warnings");

    // n_task（slot 数）由用户自定义、无上限：slot 只是占位 + 预分配 ION，
    // 真正占用 BPU 任务配额的 live task 仅在 start() 时才提交。
    // 仅对退化输入（≤0）兜底到 1（n_task=0 无意义）。
    if (n_task <= 0) { n_task = 1; warnings.attr("warn")("n_task <= 0, clamped to 1", py::module_::import("builtins").attr("UserWarning")); }
    n_task_ = n_task;

    task_handles.resize(n_task_);
    is_infer = std::vector<std::atomic<int>>(n_task_);

    _init_model(model_path, model_cnt_select);

    try {
    // ── lazy-per-slot: 每个 slot 按 desc clone 模板并立即分配（defer=false），完善 + allocated ──
    // bound_inputs/outputs 是 py::list 成员（def_readwrite，Python list 引用语义）：
    // m.inputs[slot][idx]=ion 走 Python 原生 __setitem__ 直接落回成员（无校验，浅/快）。
    // Auto 档 start(list) 复用 slot 的 IONArray（from_numpy 覆盖数据）；零拷贝档用户替换 bound。
    // hbDNNTensor 不再作成员缓存——start 时从 py::list cast shared_ptr 拷局部 vector，再现建 hbDNNTensor。
    for (int32_t t = 0; t < n_task_; t++)
    {
        py::list in_slot, out_slot;
        for (int32_t i = 0; i < input_count; i++)
            in_slot.append(py::cast(IONArray::clone(input_templates_[i], /*cached=*/true, /*defer=*/false)));
        for (int32_t i = 0; i < output_count; i++)
            out_slot.append(py::cast(IONArray::clone(output_templates_[i], /*cached=*/true, /*defer=*/false)));
        bound_inputs.append(in_slot);
        bound_outputs.append(out_slot);
    }

    // ── 异步推理标志初始化 ──
    for (int32_t t = 0; t < n_task_; t++)
        is_infer[t].store(0, std::memory_order_relaxed);
    } catch (...) {
        hbDNNRelease(packed_dnn_handle);
        throw;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 析构：释放 GIL 调 hbUCPWaitTaskDone（S2 修复）
// ─────────────────────────────────────────────────────────────────────────────
CauchyKesai::~CauchyKesai()
{
    // S2: 析构在调 hbUCPWaitTaskDone/hbUCPReleaseTask 前 gil_scoped_release
    // 防止 BPU 阻塞时 GIL 死锁
    {
        py::gil_scoped_release release;
        for (int32_t t = 0; t < n_task_; t++)
        {
            if (is_infer[t].load(std::memory_order_acquire))
            {
                try { hbUCPWaitTaskDone(task_handles[t], 3000); } catch (...) {}
                try { hbUCPReleaseTask(task_handles[t]); } catch (...) {}
                is_infer[t].store(0, std::memory_order_release);
            }
        }
    }
    std::cout << "[INFO] release model " << name_list[model_cnt_select_]
              << " success." << std::endl;
    hbDNNRelease(packed_dnn_handle);
}

bool CauchyKesai::is_busy(int32_t task_id) const
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));
    return is_infer[task_id].load(std::memory_order_acquire) != 0;
}

// ══════════════════════════════════════════════════════════════════════════════
// summary(): 模型摘要（读 templates，单一真相）
// ══════════════════════════════════════════════════════════════════════════════

py::dict CauchyKesai::summary()
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
        auto &tpl = input_templates_[i];
        py::dict inp;
        inp["index"]           = i;
        inp["name"]            = tpl->desc.name;
        inp["dtype"]           = dtype_np2str(tpl->desc.dtype);
        inp["tensorType"]      = tpl->desc.tensor_type;
        inp["alignedByteSize"] = tpl->desc.aligned_byte_size;
        inp["quantiType"]      = static_cast<int>(tpl->desc.quanti_type);
        inp["quantizeAxis"]    = tpl->desc.quantize_axis;
        inp["desc"]            = tpl->desc.desc;
        py::list shape;
        for (auto d : tpl->desc.shape) shape.append(d);
        inp["shape"] = shape;
        py::list stride;
        for (int64_t s : tpl->desc.stride) stride.append(s);
        inp["stride"] = stride;

        // ── 量化参数: 原生 list（jsonable，不再依赖 Python _jsonable）──
        // 原返回按 axis 广播的 numpy array（shape [1,..,slen,..,1]）；迁移后返一维 list
        // （长度 = slen）。无测试/渲染依赖 scale 的广播 shape（d11 _IO_KEYS 不含 scale，
        // _render 仅判断非空），一维 list 更符合「量化参数向量」语义且直接可 json.dumps。
        const auto &scale = tpl->desc.scale;
        const auto &zp = tpl->desc.zero_point;
        inp["scale"] = py::cast(scale);           // std::vector<float> → py::list
        inp["zero_point"] = py::cast(zp);         // std::vector<int32_t> → py::list

        inputs_list.append(inp);
    }

    py::list outputs_list;
    for (int32_t i = 0; i < output_count; i++)
    {
        auto &tpl = output_templates_[i];
        py::dict out;
        out["index"]           = i;
        out["name"]            = tpl->desc.name;
        out["dtype"]           = dtype_np2str(tpl->desc.dtype);
        out["tensorType"]      = tpl->desc.tensor_type;
        out["alignedByteSize"] = tpl->desc.aligned_byte_size;
        out["quantiType"]      = static_cast<int>(tpl->desc.quanti_type);
        out["quantizeAxis"]    = tpl->desc.quantize_axis;
        out["desc"]            = tpl->desc.desc;
        py::list shape;
        for (auto d : tpl->desc.shape) shape.append(d);
        out["shape"] = shape;
        py::list stride;
        for (int64_t s : tpl->desc.stride) stride.append(s);
        out["stride"] = stride;

        // ── 量化参数: 原生 list（jsonable；与 input 段一致，详见 input 段注释）──
        const auto &scale = tpl->desc.scale;
        const auto &zp = tpl->desc.zero_point;
        out["scale"] = py::cast(scale);
        out["zero_point"] = py::cast(zp);

        outputs_list.append(out);
    }

    py::dict result;
    result["model_path"]   = model_path_;
    result["model_names"]  = model_names_list;
    result["n_task"]       = n_task_;
    result["memory_mb"]    = mbs;
    result["bpu_core_num"] = bpu_core_num_;
    py::list sched;
    for (auto c : scheduled_cores(0)) sched.append(c);   // summary 展示 slot 0；per-slot 查询走 scheduled_cores(task_id)
    result["scheduled_cores"] = sched;
    result["scheduled_priority"] = priorities_[0];   // summary 展示 slot 0；per-slot 查询走 scheduled_priority(task_id)
    result["inputs"]       = inputs_list;
    result["outputs"]      = outputs_list;

    // ── 平台 / 固件 / 编译配置（原 Python wrapper 注入，现迁 C++）──
    // platform：机型无关原子能力（16-key dict，已 jsonable）。
    // bpu_fw_version：依赖模型加载后才初始化的固件，模型已加载故在此读。
    // compile_config：model_desc 是编译配置 JSON 时解析为 dict，非 JSON / 非 dict 时为 None
    //   （等价原 Python _parse_compile_config 语义）。
    result["platform"] = platform_.summary();
    result["bpu_fw_version"] = cauchykesai::read_bpu_fw_version();
    try {
        result["compile_config"] = json_to_py(nlohmann::json::parse(model_desc_));
    } catch (...) {
        result["compile_config"] = py::none();
    }
    return result;
}

void CauchyKesai::set_scheduling_params(const std::vector<int32_t> &bpu_cores, int32_t priority, int32_t task_id)
{
    // task_id 范围：-1 = 广播所有 slot；否则须在 [0, n_task_)。
    if (task_id != -1 && (task_id < 0 || task_id >= n_task_))
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id)
            + ", n_task=" + std::to_string(n_task_) + " (use -1 to broadcast all slots)");

    // priority: -1（默认）= 不改现有优先级；[0,255] = 设该 slot 优先级（254/255 抢占）。
    if (priority != -1 && (priority < 0 || priority > 255))
        throw std::invalid_argument("priority out of range [-1,255]: got " + std::to_string(priority));

    // 写入辅助：-1 广播到所有 slot，否则仅写指定 slot。各 slot 独享掩码 + 优先级。
    auto apply = [&](uint64_t mask) {
        if (task_id == -1) {
            std::fill(backends_.begin(), backends_.end(), mask);
            if (priority != -1) std::fill(priorities_.begin(), priorities_.end(), priority);
        } else {
            backends_[task_id] = mask;
            if (priority != -1) priorities_[task_id] = priority;
        }
    };

    if (bpu_cores.empty()) { apply(HB_UCP_BPU_CORE_ANY); return; }

    const int32_t n = static_cast<int32_t>(bpu_cores.size());

    // 1) 索引范围 [0,63]（bitmask 有效位）
    for (auto core : bpu_cores)
        if (core < 0 || core >= 64)
            throw std::invalid_argument("bpu_cores must contain values in [0, 63]");

    // 2) 物理核有效性统计（构造时读到的 physical_core_num_，-1=未知则跳过物理核校验）。
    //    单核模型宽松（≥1 有效核即可，SDK 从掩码挑一颗）；多核模型严格（全部核必须有效）。
    int32_t valid = 0;
    if (physical_core_num_ > 0)
        for (auto core : bpu_cores)
            if (core < physical_core_num_) ++valid;

    // 3) 核数 == 编译核数（仅多核模型；单核模型对核数宽松）。
    if (bpu_core_num_ >= 2 && n != bpu_core_num_)
        throw std::invalid_argument(
            "bpu_cores count (" + std::to_string(n) + ") != compile core_num ("
            + std::to_string(bpu_core_num_) + "); multi-core model requires exactly "
            + std::to_string(bpu_core_num_) + " valid cores (see bpu_core_num)");

    // 4) 物理核越界（已知物理核数时才校验，提前拦掉会触发 hbUCPSubmitTask 失败的掩码）。
    if (physical_core_num_ > 0)
    {
        if (bpu_core_num_ >= 2)
        {
            if (valid != n)
                throw std::invalid_argument(
                    "bpu_cores contains an invalid physical core index (>= "
                    + std::to_string(physical_core_num_) + "); this chip has "
                    + std::to_string(physical_core_num_) + " BPU cores (0.."
                    + std::to_string(physical_core_num_ - 1) + ")");
        }
        else if (valid < 1)
        {
            throw std::invalid_argument(
                "bpu_cores has no valid physical core; this chip has "
                + std::to_string(physical_core_num_) + " BPU cores (0.."
                + std::to_string(physical_core_num_ - 1) + ")");
        }
    }

    uint64_t mask = 0;
    for (auto core : bpu_cores) mask |= (1ULL << core);
    apply(mask);
}

std::vector<int32_t> CauchyKesai::scheduled_cores(int32_t task_id) const {
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id)
            + ", n_task=" + std::to_string(n_task_));
    std::vector<int32_t> v;
    uint64_t mask = backends_[task_id];
    if (mask != HB_UCP_BPU_CORE_ANY) {
        for (int32_t i = 0; i < 64; ++i)
            if (mask & (1ULL << i)) v.push_back(i);
    }
    return v;
}

int32_t CauchyKesai::scheduled_priority(int32_t task_id) const {
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id)
            + ", n_task=" + std::to_string(n_task_));
    return priorities_[task_id];
}

// ══════════════════════════════════════════════════════════════════════════════
// benchmark(): 计时单次推理（委托 start+wait，不重复 SDK 编排）
// ══════════════════════════════════════════════════════════════════════════════

py::dict CauchyKesai::benchmark(int32_t timeout_ms)
{
    auto tp_start = std::chrono::system_clock::now();
    // 委托零拷贝 start()/wait()：用 slot 0 当前 bound 数据计时（不再传空 list 哨兵）。
    start(0);
    wait(0, timeout_ms);
    auto tp_end = std::chrono::system_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::microseconds>(tp_end - tp_start).count();

    py::dict result;
    result["time_us"]  = total_time;
    result["time_ms"]  = total_time / 1000.0;
    result["time_s"]   = total_time / 1000000.0;
    result["time_min"] = total_time / (1000000.0 * 60);
    return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// 校验能力：ion 能否绑到 input/output[idx]（properties_match + BPU 对齐），返回 bool 不抛
// 与 bind 同源（旧 bind_input 的判定），check 返回 True ⟺ 绑定必成功
// ══════════════════════════════════════════════════════════════════════════════

bool CauchyKesai::check_input(std::shared_ptr<IONArray> ion, int32_t idx) const
{
    if (idx < 0 || idx >= input_count || !ion) return false;
    if (bpu_align_ > 0 && (ion->byte_offset % bpu_align_) != 0) return false;
    return ion->properties_match(*input_templates_[idx]);
}

bool CauchyKesai::check_output(std::shared_ptr<IONArray> ion, int32_t idx) const
{
    if (idx < 0 || idx >= output_count || !ion) return false;
    if (bpu_align_ > 0 && (ion->byte_offset % bpu_align_) != 0) return false;
    return ion->properties_match(*output_templates_[idx]);
}

// ══════════════════════════════════════════════════════════════════════════════
// 模板 desc（模型固有性质，只读）
// ══════════════════════════════════════════════════════════════════════════════

std::vector<IONArrayDesc> CauchyKesai::input_descs() const
{
    std::vector<IONArrayDesc> v;
    v.reserve(input_templates_.size());
    for (const auto &t : input_templates_) v.push_back(t->desc);
    return v;
}

std::vector<IONArrayDesc> CauchyKesai::output_descs() const
{
    std::vector<IONArrayDesc> v;
    v.reserve(output_templates_.size());
    for (const auto &t : output_templates_) v.push_back(t->desc);
    return v;
}

// ══════════════════════════════════════════════════════════════════════════════
// start：异步推理提交
//   start(task_id)         零拷贝：用 slot 已绑定的 bound IONArray，不传 inputs
//   start(inputs, task_id) Auto memcpy：from_numpy 写 slot 的 bound IONArray，再走零拷贝 start
// 异步入口不校验 bound（要快/浅，用户改错自己负责）；公共体 _start_impl。
// ══════════════════════════════════════════════════════════════════════════════

void CauchyKesai::start(int32_t task_id)
{
    _start_impl(task_id, /*inputs=*/nullptr);
}

void CauchyKesai::start(const std::vector<py::array> &inputs, int32_t task_id)
{
    if ((int32_t)inputs.size() != input_count)
        throw std::invalid_argument("input count mismatch: expected " +
            std::to_string(input_count) + ", got " + std::to_string(inputs.size()));
    _start_impl(task_id, &inputs);
}

void CauchyKesai::_start_impl(int32_t task_id, const std::vector<py::array> *inputs)
{
    // ── task_id 边界校验（priority 为 per-slot，由 set_scheduling_params 设）──
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));

    // ── 原子获取任务槽 ──
    int expected = 0;
    if (!is_infer[task_id].compare_exchange_strong(expected, 1,
            std::memory_order_acquire, std::memory_order_relaxed))
        throw std::runtime_error("task_id " + std::to_string(task_id) + " is already in use");

    hbUCPTaskHandle_t task_handle{nullptr};
    bool handle_stored = false;
    try {
        // ── GIL 下：从 py::list bound 拷出 shared_ptr 到局部 vector 保活（防用户 wait 前改 bound 换掉 ion）──
        // 同时让后续 from_numpy / dnn_tensor 不再触碰 Python 容器（SubmitTask 释放 GIL 后不可访问 py::list）。
        py::list in_slot_py  = bound_inputs[task_id];
        py::list out_slot_py = bound_outputs[task_id];
        if ((int32_t)in_slot_py.size()  != input_count)
            throw std::runtime_error("slot input count mismatch: expected " +
                std::to_string(input_count) + ", bound has " + std::to_string(in_slot_py.size()) +
                " (用户改坏了 bound)");
        if ((int32_t)out_slot_py.size() != output_count)
            throw std::runtime_error("slot output count mismatch: expected " +
                std::to_string(output_count) + ", bound has " + std::to_string(out_slot_py.size()));

        std::vector<std::shared_ptr<IONArray>> in_ions(input_count);
        std::vector<std::shared_ptr<IONArray>> out_ions(output_count);
        for (int32_t i = 0; i < input_count; i++)
            in_ions[i] = in_slot_py[i].cast<std::shared_ptr<IONArray>>();   // 非 IONArray 抛 TypeError
        for (int32_t i = 0; i < output_count; i++)
            out_ions[i] = out_slot_py[i].cast<std::shared_ptr<IONArray>>();

        if (inputs)
        {
            // 委托 IONArray::from_numpy：布局感知写入 + 校验 + flush。
            // 写入目标是 bound（slot 当前绑定；零拷贝档用户 m.inputs[slot][idx]=ion 替换后即用户 IONArray）。
            for (int32_t i = 0; i < input_count; i++)
                in_ions[i]->from_numpy((*inputs)[i]);
        }
        // 注：不再统一 flush input。memcpy 路径 from_numpy 已 flush；
        //     零拷贝路径（m.inputs[slot][idx]=ion + 用户写 buffer）由用户契约负责 flush_clean。

        // ── 现建局部 vector<hbDNNTensor> from in/out_ions->dnn_tensor()（每帧投影，bound 唯一真相无腐化）──
        // 不维护 hbDNNTensor 副本成员：bound 改了立即生效。
        // 注意：dnn_tensor() 的 properties.scale/zeroPoint 指针指向 IONArray 的 desc 内部 vector，
        // in/out_ions 局部 vector 持 shared_ptr 保活，覆盖 hbDNNInferV2 + SubmitTask 全程。
        std::vector<hbDNNTensor> in_tensors(input_count);
        std::vector<hbDNNTensor> out_tensors(output_count);
        for (int32_t i = 0; i < input_count; i++)
            in_tensors[i] = in_ions[i]->dnn_tensor();
        for (int32_t i = 0; i < output_count; i++)
            out_tensors[i] = out_ions[i]->dnn_tensor();

        RDK_CHECK_SUCCESS(
            hbDNNInferV2(&task_handle, out_tensors.data(), in_tensors.data(), dnn_handle),
            "hbDNNInferV2 failed");
        hbUCPSchedParam ctrl_param;
        HB_UCP_INITIALIZE_SCHED_PARAM(&ctrl_param);
        ctrl_param.priority = priorities_[task_id];
        ctrl_param.backend = backends_[task_id];

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

// ══════════════════════════════════════════════════════════════════════════════
// wait_done：纯等完成（flush_invalidate 输出 + release task），不返回
// wait：wait_done + 导出 list[ndarray]（Auto 用）
// ══════════════════════════════════════════════════════════════════════════════

void CauchyKesai::wait_done(int32_t task_id, int32_t timeout_ms)
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
            RDK_CHECK_SUCCESS(hbUCPWaitTaskDone(task_handles[task_id], timeout_ms), "hbUCPWaitTaskDone failed");
        }
        // GIL 下：从 py::list bound 拷出输出 shared_ptr，再 flush_invalidate（output 是 BPU 写的，CPU 读前必须 invalidate）。
        py::list out_slot_py = bound_outputs[task_id];
        if ((int32_t)out_slot_py.size() != output_count)
            throw std::runtime_error("wait: slot output count mismatch: expected " +
                std::to_string(output_count) + ", bound has " + std::to_string(out_slot_py.size()));
        std::vector<std::shared_ptr<IONArray>> out_ions(output_count);
        for (int i = 0; i < output_count; i++)
            out_ions[i] = out_slot_py[i].cast<std::shared_ptr<IONArray>>();
        for (int i = 0; i < output_count; i++)
            out_ions[i]->flush_invalidate();

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
}

std::vector<py::array> CauchyKesai::wait(int32_t task_id, int32_t timeout_ms)
{
    wait_done(task_id, timeout_ms);

    // GIL 下：从 py::list bound 拷出输出 shared_ptr，导出 numpy（布局感知 + capsule 保活）。
    py::list out_slot_py = bound_outputs[task_id];
    if ((int32_t)out_slot_py.size() != output_count)
        throw std::runtime_error("wait: slot output count mismatch: expected " +
            std::to_string(output_count) + ", bound has " + std::to_string(out_slot_py.size()));
    std::vector<std::shared_ptr<IONArray>> out_ions(output_count);
    for (int32_t i = 0; i < output_count; i++)
        out_ions[i] = out_slot_py[i].cast<std::shared_ptr<IONArray>>();

    std::vector<py::array> rs;
    for (int32_t i = 0; i < output_count; i++)
    {
        if (!out_ions[i])
            throw std::runtime_error("wait: output[" + std::to_string(i) + "] has no bound IONArray (internal error)");
        rs.push_back(out_ions[i]->numpy());
    }
    return rs;
}

// ══════════════════════════════════════════════════════════════════════════════
// inference：同步推理（校验 + start + wait + 导出）
//   inference(inputs, task_id) Auto 同步 memcpy：校验 dtype/ndim/shape + from_numpy + start + wait + 导出
//   inference(task_id)        零拷贝同步：校验 bound（check_input/output）+ start + wait + 导出
// 同步才校验（异步不校验）；不做透明量化，量化/反量化由 IONArray 负责
// ══════════════════════════════════════════════════════════════════════════════

std::vector<py::array> CauchyKesai::inference(const std::vector<py::array> &inputs, int32_t task_id)
{
    // ── input 数据校验: 数量 / dtype / ndim / shape ──
    if ((int32_t)inputs.size() != input_count)
        throw std::invalid_argument("input count mismatch: expected " + std::to_string(input_count) + ", got " + std::to_string(inputs.size()));

    for (int32_t cnt = 0; cnt < input_count; cnt++)
    {
        auto &tpl = input_templates_[cnt];
        if (!inputs[cnt].dtype().is(tpl->desc.dtype))
            throw std::invalid_argument("dtype mismatch at input[" + std::to_string(cnt) + "]");
        if (inputs[cnt].ndim() != tpl->desc.ndim())
            throw std::invalid_argument("ndim mismatch at input[" + std::to_string(cnt) + "]");
        for (int i = 0; i < inputs[cnt].ndim(); ++i)
        {
            if (tpl->desc.shape[i] != (ssize_t)inputs[cnt].shape()[i])
                throw std::invalid_argument("shape mismatch at input[" + std::to_string(cnt) + "]");
        }
    }

    start(inputs, task_id);
    return wait(task_id);
}

std::vector<py::array> CauchyKesai::inference(int32_t task_id)
{
    if (task_id < 0 || task_id >= n_task_)
        throw std::out_of_range("task_id out of range: got " + std::to_string(task_id));
    if (is_infer[task_id].load(std::memory_order_acquire) != 0)
        throw std::runtime_error("task_id " + std::to_string(task_id) + " is busy, cannot inference");

    // ── 零拷贝同步：校验 bound（check_input/output），不匹配抛 ──
    py::list in_slot_py  = bound_inputs[task_id];
    py::list out_slot_py = bound_outputs[task_id];
    if ((int32_t)in_slot_py.size()  != input_count)
        throw std::invalid_argument("slot input count mismatch: expected " +
            std::to_string(input_count) + ", bound has " + std::to_string(in_slot_py.size()));
    if ((int32_t)out_slot_py.size() != output_count)
        throw std::invalid_argument("slot output count mismatch: expected " +
            std::to_string(output_count) + ", bound has " + std::to_string(out_slot_py.size()));

    for (int32_t i = 0; i < input_count; i++)
    {
        auto ion = in_slot_py[i].cast<std::shared_ptr<IONArray>>();
        if (!ion)
            throw std::runtime_error("input[" + std::to_string(i) + "] has no bound IONArray");
        if (!check_input(ion, i))
            throw std::invalid_argument("input[" + std::to_string(i) + "] bound IONArray mismatch with template (dtype/shape/memSize/align)");
    }
    for (int32_t i = 0; i < output_count; i++)
    {
        auto ion = out_slot_py[i].cast<std::shared_ptr<IONArray>>();
        if (!ion)
            throw std::runtime_error("output[" + std::to_string(i) + "] has no bound IONArray");
        if (!check_output(ion, i))
            throw std::invalid_argument("output[" + std::to_string(i) + "] bound IONArray mismatch with template (dtype/shape/memSize/align)");
    }

    start(task_id);
    return wait(task_id);
}
