#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cauchykesai/pycauchykesai.h"
#include "cauchykesai/ion_memory.h"
#include "cauchykesai/platform.h"

namespace py = pybind11;

PYBIND11_MODULE(pycauchykesai, m)
{
    m.attr("__author__")  = "Cauchy - WuChao in D-Robotics";
    m.attr("__date__")    = "2025-05-30";
    m.attr("__doc__")     = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3";

    // ── IONArrayDesc: 纯张量描述（无内存），成员 public 直访 ──────────────
    //
    // 持有 tensor 全部性质（dtype/shape/stride/量化 等），无不变量，成员 public。
    // IONArray 持有它（has-a）。性质直接访问：ion.desc.dtype。
    //
    py::class_<IONArrayDesc, std::shared_ptr<IONArrayDesc>>(m, "IONArrayDesc",
        "Pure tensor descriptor (no memory), public members.\n"
        "IONArray holds it (has-a). Access dtype/shape/etc directly: ion.desc.dtype.")
        .def(py::init<py::dtype, std::vector<ssize_t>>(),
             py::arg("dtype"), py::arg("shape"),
             "Create a bare IONArrayDesc (dtype + shape; other properties default).")
        // ── 成员 public 直访（def_readonly）──
        .def_readonly("dtype", &IONArrayDesc::dtype, "Element dtype (np.dtype)")
        .def_readonly("shape", &IONArrayDesc::shape, "Logical shape (list)")
        .def_readonly("name", &IONArrayDesc::name, "Tensor name")
        .def_readonly("desc", &IONArrayDesc::desc, "hbDNN tensor description text")
        .def_readonly("aligned_byte_size", &IONArrayDesc::aligned_byte_size, "Aligned byte size")
        .def_readonly("tensor_type", &IONArrayDesc::tensor_type, "hbDNNDataType enum value")
        .def_property_readonly("quanti_type",
             [](IONArrayDesc &s) { return static_cast<int>(s.quanti_type); },
             "Quantization type (NONE=0/SCALE=1)")
        .def_readonly("quantize_axis", &IONArrayDesc::quantize_axis, "Quantization axis")
        .def_readonly("stride", &IONArrayDesc::stride, "Stride array (bytes)")
        .def_readonly("scale", &IONArrayDesc::scale, "Dequantization scale")
        .def_readonly("zero_point", &IONArrayDesc::zero_point, "Zero point")
        // ── 计算方法（有逻辑，保留）──
        .def("ndim", &IONArrayDesc::ndim, "Number of dimensions")
        .def("size", &IONArrayDesc::size, "Total number of elements")
        .def("nbytes", &IONArrayDesc::nbytes, "dtype.itemsize × prod(shape)")
        .def("has_stride", &IONArrayDesc::has_stride, "True if stride is set")
        .def("is_padded_layout", &IONArrayDesc::is_padded_layout, "True if padded (aligned) layout")
        .def("has_tensor_properties", &IONArrayDesc::has_tensor_properties, "True if has tensor properties")
        .def("is_quantized", &IONArrayDesc::is_quantized, "True if quanti_type == SCALE")
        ;

    // ── IONArray: 持有 desc + memory（class，成员 public）──────────────────
    //
    // 成员 public：ion.desc / ion.memory / ion.byte_offset 直接访问。
    // 内存操作方法是 IONArray 本职：numpy / from_numpy / flush / allocate 等。
    //
    py::class_<IONArray, std::shared_ptr<IONArray>>(m, "IONArray",
        "ION memory carrier: holds IONArrayDesc (desc) + IONMemory (memory).\n"
        "Members are public: ion.desc / ion.memory / ion.byte_offset.\n"
        "Use .numpy() for np.ndarray; .from_memory(mem, off, desc) for zero-copy;\n"
        "defer=True + .allocate() for lazy allocation.")
        // ── 成员 public 直访 ──
        .def_readonly("desc", &IONArray::desc, "Tensor descriptor (IONArrayDesc)")
        .def_readonly("memory", &IONArray::memory, "Underlying IONMemory (shared_ptr)")
        .def_readonly("byte_offset", &IONArray::byte_offset, "Byte offset within IONMemory")
        // ── 构造 ──
        .def(py::init<py::dtype, std::vector<ssize_t>, bool, bool>(),
             py::arg("dtype"), py::arg("shape"),
             py::arg("cached") = true, py::arg("defer") = false,
             "Create an IONArray (bare: dtype + shape).")
        .def(py::init<IONArrayDesc, bool, bool>(),
             py::arg("desc"), py::arg("cached") = true, py::arg("defer") = false,
             "Create an IONArray from an IONArrayDesc (full properties).")
        // ── 内存操作 ──
        .def("allocate", &IONArray::allocate, py::arg("cached") = true,
             "Allocate ION memory (uses aligned_byte_size if set, else dtype×shape).")
        .def_property_readonly("is_allocated", &IONArray::is_allocated,
             "True if ION memory has been allocated")
        .def("flush_clean", &IONArray::flush_clean,
             "Flush CPU cache (CLEAN): write dirty lines. No-op on uncached/unallocated.")
        .def("flush_invalidate", &IONArray::flush_invalidate,
             "Invalidate CPU cache (INVALIDATE): force re-read. No-op on uncached/unallocated.")
        .def("numpy", &IONArray::numpy,
             "Return a plain np.ndarray view of this ION memory (zero-copy, layout-aware).\n"
             "The returned array holds a reference to this IONArray (capsule keeps ION memory alive).")
        .def("__array__",
             [](IONArray &self, py::object dtype_obj) -> py::array {
                 py::array arr = self.numpy();
                 if (dtype_obj.is_none()) return arr;
                 py::module np = py::module::import("numpy");
                 return np.attr("asarray")(arr, py::arg("dtype") = dtype_obj).cast<py::array>();
             },
             py::arg("dtype") = py::none(),
             "NumPy __array__ protocol: np.asarray(ion)/np.sum(ion) work directly.")
        .def_static("from_memory", &IONArray::from_memory,
             py::arg("memory"), py::arg("byte_offset"), py::arg("template"),
             "Create an IONArray sharing an IONMemory at a byte offset,\n"
             "with tensor properties inherited from the template (IONArrayDesc).\n"
             "  memory: IONMemory to share\n"
             "  byte_offset: byte offset within the IONMemory\n"
             "  template: IONArrayDesc to inherit properties from (e.g. ion.desc)")
        .def("dequantize", &IONArray::dequantize,
             "Dequantize buffer (int) → float numpy array. NONE→passthrough; SCALE→(x-zp)*scale.")
        .def("quantize", &IONArray::quantize, py::arg("float_arr"),
             "Quantize float numpy → write into ION buffer (int). NONE→memcpy; SCALE→round/scale+zp, clip.")
        .def("from_numpy", &IONArray::from_numpy, py::arg("arr"),
             "Write a contiguous numpy array into the ION buffer (layout-aware, auto flush).")
        // ── 模块级工厂的 C++ 落地（from_numpy 同名双义：工厂分配+写入 vs 方法只写入）──
        // 做成静态方法 from_numpy_array；Python __init__ 加一行 `from_numpy = IONArray.from_numpy_array`
        // re-export 满足 `from pyCauchyKesai import from_numpy`（文档/测试契约）。
        // 行为 = np.ascontiguousarray + IONArray(dtype,shape,cached) + from_numpy（精确复现原 Python 工厂）。
        .def_static("from_numpy_array",
            [](py::array arr, bool cached) {
                py::module np = py::module::import("numpy");
                py::array c = np.attr("ascontiguousarray")(arr).cast<py::array>();
                std::vector<ssize_t> shape(c.ndim());
                for (ssize_t i = 0; i < c.ndim(); i++) shape[i] = c.shape()[i];
                auto ion = std::make_shared<IONArray>(c.dtype(), shape, cached, /*defer=*/false);
                ion->from_numpy(c);
                return ion;
            },
            py::arg("arr"), py::arg("cached") = true,
            "Create an IONArray from a numpy array (allocate + from_numpy + flush).\n"
            "  arr: numpy array (made contiguous if not already)\n"
            "  cached: True -> hbUCPMallocCached (default); False -> hbUCPMalloc\n"
            "Returns IONArray with the same dtype/shape, data copied in.")
        ;

    // ── IONMemory: 裸 ION 内存块（无 Tensor 语义）──────────────────────────
    py::class_<IONMemory, std::shared_ptr<IONMemory>>(m, "IONMemory",
        "Raw ION memory block without tensor semantics.\n"
        "Manages physical memory accessible by both CPU and BPU.\n"
        "Use .numpy() to get a uint8 numpy view; use .phy_addr for BPU DMA.\n"
        "Supports cached/uncached allocation, cache flush/invalidate, and move semantics.")
        .def(py::init<uint64_t, bool>(),
             py::arg("size"), py::arg("cached") = true,
             "Allocate ION memory of given byte size.\n"
             "  size: byte size to allocate\n"
             "  cached=True -> hbUCPMallocCached (default)\n"
             "  cached=False -> hbUCPMalloc (no CPU cache)")
        // ── 内存属性 ──（phy_addr/is_cached 下沉内部）
        .def_property_readonly("size", &IONMemory::size,
             "ION allocation size in bytes (uint64)")
        .def_property_readonly("is_allocated", &IONMemory::is_allocated,
             "True if ION memory has been allocated")
        // ── Cache 操作 ──
        .def("flush_clean", &IONMemory::flush_clean,
             "Flush CPU cache: write dirty lines to physical memory")
        .def("flush_invalidate", &IONMemory::flush_invalidate,
             "Invalidate CPU cache: discard stale lines, force re-read from physical memory")
        // ── numpy 视图 ──
        .def("numpy", &IONMemory::numpy,
             "Return a uint8 numpy array view of this ION memory (zero-copy).\n"
             "The returned array holds a reference to this IONMemory via capsule,\n"
             "keeping the ION memory alive as long as the numpy array exists.\n"
             "Throws RuntimeError if not allocated.")
        ;

    // ── Platform: 板卡 / 芯片平台原子能力（机型无关，不依赖加载模型）─────
    // 8 静态 property（构造时缓存）+ 7 实时方法（每次现读 sysfs）+ summary()。
    // s() 展示留 Python 子类（调 _render.render_platform），C++ 不绑 s()。
    py::class_<cauchykesai::Platform>(m, "Platform",
        "Board / chip platform atomic capabilities (model-independent).\n"
        "Static properties (cached at construction) + realtime methods (re-read sysfs each call).\n"
        "Held by CauchyKesai (m.platform); also usable standalone: Platform().\n"
        "summary() returns a native dict (jsonable); use .s() (Python override) for pretty print.")
        .def(py::init<>(), "Construct; read static properties once (SDK chip query + sysfs/proc).")
        // ── 8 静态属性（只读 property）──
        .def_property_readonly("soc_name", &cauchykesai::Platform::soc_name,
             "SoC name (hbUCPGetSocName, e.g. 'D-Robotics RDK S600 MCB V0p2'); '' if unreadable.")
        .def_property_readonly("bpu_type", &cauchykesai::Platform::bpu_type,
             "BPU arch: 'nash-e' / 'nash-m' / 'nash-p', inferred from soc_name; '' if unknown.")
        .def_property_readonly("dnn_version", &cauchykesai::Platform::dnn_version,
             "DNN library version (hbDNNGetVersion).")
        .def_property_readonly("bpu_version", &cauchykesai::Platform::bpu_version,
             "BPU runtime version (hbUCPGetVersion).")
        .def_property_readonly("physical_core_num", &cauchykesai::Platform::physical_core_num,
             "Physical BPU core count (sysfs core_num, fallback count bpu dirs); -1 if unreadable.")
        .def_property_readonly("cpu_model", &cauchykesai::Platform::cpu_model,
             "CPU model (/proc/cpuinfo model name, e.g. 'Cortex-A78AE').")
        .def_property_readonly("cpu_count", &cauchykesai::Platform::cpu_count,
             "CPU logical core count (/proc/cpuinfo processor segments).")
        .def_property_readonly("mem_total_mb", &cauchykesai::Platform::mem_total_mb,
             "Total memory MB (/proc/meminfo MemTotal, kB->MB); -1 if unreadable.")
        // ── 7 实时方法（每次现读 sysfs）──
        .def("bpu_rate", &cauchykesai::Platform::bpu_rate,
             "Per-core BPU occupancy (sysfs bpu*/ratio, indexed by core id).\n"
             "result[i] = core i occupancy (0-100, 0=idle); length = max core id + 1.\n"
             "Empty list if no sysfs.")
        .def("bpu_freq", &cauchykesai::Platform::bpu_freq,
             "Per-core BPU frequency MHz (devfreq cur_freq, Hz->MHz, indexed by core id).\n"
             "Empty list if no devfreq; unreadable core position is 0.")
        .def("cpu_freq", &cauchykesai::Platform::cpu_freq,
             "Per-CPU frequency MHz (cpufreq scaling_cur_freq, kHz->MHz, indexed by CPU id).\n"
             "Empty list if no cpufreq; unreadable CPU position is 0.")
        .def("temperature", &cauchykesai::Platform::temperature,
             py::arg("name") = "",
             "Temperature (thermal_zone*/{type,temp}). Returns {sensor: °C} (millideg/1000).\n"
             "name='' returns all (only temp>0); name hit returns {name: float}; miss returns {name: -1}.")
        .def("voltage", &cauchykesai::Platform::voltage,
             py::arg("name") = "",
             "Voltage (hwmon*/in*_label + in*_input). Returns {rail: mV}.\n"
             "name='' returns all (skips 'NULL' labels); name hit returns {name: int}; miss returns {name: -1}.")
        .def("ion_memory", &cauchykesai::Platform::ion_memory,
             py::arg("name") = "",
             "Three ION heap occupancy (debugfs /sys/kernel/debug/ion/heaps).\n"
             "Returns {heap: {'used': bytes, 'total': bytes}} for ion_cma/cma_reserved/carveout.\n"
             "name hit (known heap) returns {name: {'used','total'}}; miss returns {name: -1}.")
        .def("ucp_library_path", &cauchykesai::Platform::ucp_library_path,
             "Absolute path of libhbucp.so loaded in this process (from /proc/self/maps); '' if none.")
        // ── summary / repr（s() 留 Python 子类）──
        .def("summary", &cauchykesai::Platform::summary,
             "Full platform info dict: 8 static + 7 realtime + mem_available_mb (native, jsonable).\n"
             "Use .s() for pretty print.")
        .def("s", &cauchykesai::Platform::s,
             "Pretty-print platform info (internal print, returns None).")
        .def("__repr__", &cauchykesai::Platform::repr,
             "Compact one-line repr: Platform(soc=..., bpu_type=..., bpu_cores=N, cpu='...' xN, mem=NMB).")
        ;

    // ── 模块级全局 Platform 单例：import 即构造，CauchyKesai 与用户均共用此实例 ──
    // reference 策略不拷贝：m.platform（实例属性）与此处模块级 platform 指向同一 C++ 对象。
    m.attr("platform") = py::cast(cauchykesai::global_platform(),
                                  py::return_value_policy::reference);

    // ── CauchyKesai ───────────────────────────────────────────────────────
    py::class_<CauchyKesai>(m, "CauchyKesai",
        "BPU inference orchestrator. ION layer = atomic zero-copy capability; Auto = composition.\n"
        "lazy-per-slot: constructor allocates every slot's input/output IONArray (complete + allocated).\n"
        "Auto: m([y,uv]) / m.start([y,uv]) — from_numpy into slot's bound IONArray, then zero-copy start.\n"
        "Zero-copy: m.inputs[slot][idx]=ion (pure assignment, no validation) / m.start() / m.wait_done().\n"
        "Sync (validates bound): m(inputs) / m(task_id) — check_input/output → start → wait → numpy.")
        // ── 单一构造函数（lazy-per-slot：构造时按 desc 建每个 slot 的 IONArray，完善+allocated）──
        .def(py::init<const std::string &, int32_t, int32_t>(),
             py::arg("model_path"), py::arg("n_task") = 1,
             py::arg("model_cnt_select") = 0,
             "Construct model. lazy-per-slot: each slot's input+output IONArray built from desc at construction (complete + allocated).\n"
             "  model_path: path to .hbm model file\n"
             "  n_task: number of concurrent task slots, user-defined, no upper limit (<=0 clamped to 1, default 1); each slot pre-allocates input+output ION at construction\n"
             "  model_cnt_select: model index in packed model (default 0)")
        .def("summary", &CauchyKesai::summary,
             "Model summary dict (input/output shapes, dtypes, strides, quantization, etc.)")
        .def("benchmark", &CauchyKesai::benchmark,
             py::arg("timeout_ms") = 0,
             "Benchmark: run single inference on task 0, return timing dict.\n"
             "  timeout_ms: time budget in ms (default 0 = block until done, never times out)")
        .def("is_busy", &CauchyKesai::is_busy, py::arg("task_id") = 0,
             "Check if a task slot is busy (inference in progress).")
        // ── 异步推理入口（不校验 bound：要快/浅，用户改错自己负责）──
        .def("start", static_cast<void (CauchyKesai::*)(int32_t)>(&CauchyKesai::start),
             py::arg("task_id") = 0,
             "Zero-copy async submit: use slot's currently-bound IONArray (no inputs).\n"
             "Async entry does NOT validate bound (fast/shallow). Call m.wait_done(task_id) / m.wait(task_id) after.\n"
             "  task_id: task slot index\n"
             "  priority: per-slot, set via set_scheduling_params (default 0)")
        .def("start", static_cast<void (CauchyKesai::*)(const std::vector<py::array> &, int32_t)>(&CauchyKesai::start),
             py::arg("inputs"), py::arg("task_id") = 0,
             "Auto-memcpy async submit: from_numpy into slot's bound IONArray, then zero-copy start.\n"
             "Validates input count only. inputs length must == model input count.\n"
             "  inputs: list[np.ndarray] (will be written layout-aware + auto flush into bound IONArray)\n"
             "  task_id: task slot index")
        .def("wait_done", &CauchyKesai::wait_done, py::arg("task_id") = 0, py::arg("timeout_ms") = 0,
             "Wait for async inference to complete (flush_invalidate outputs + release task). Returns None.\n"
             "  timeout_ms: time budget in ms (default 0 = block until done, never times out; see wait docs for budget semantics)")
        .def("wait", &CauchyKesai::wait, py::arg("task_id") = 0, py::arg("timeout_ms") = 0,
             "wait_done + export list[np.ndarray]. Each returned array holds a reference to the IONArray (capsule).\n"
             "  timeout_ms: time budget in ms (default 0 = block until done, never times out)")
        // ── 同步推理入口（校验 bound：check_input/output，不匹配抛）──
        .def("__call__", static_cast<std::vector<py::array> (CauchyKesai::*)(const std::vector<py::array> &, int32_t)>(&CauchyKesai::inference),
             py::arg("inputs"), py::arg("task_id") = 0,
             "Auto sync: validate dtype/ndim/shape + from_numpy + start + wait → list[np.ndarray].")
        .def("__call__", static_cast<std::vector<py::array> (CauchyKesai::*)(int32_t)>(&CauchyKesai::inference),
             py::arg("task_id") = 0,
             "Zero-copy sync: validate bound (check_input/output) + start + wait → list[np.ndarray].\n"
             "Use after m.inputs[slot][idx]=ion (and writing/flushing the buffer yourself).")
        .def_readonly("input_names", &CauchyKesai::input_names,
             "Input tensor names (list of str).")
        .def_readonly("output_names", &CauchyKesai::output_names,
             "Output tensor names (list of str).")
        // ── 模型 / 实例信息（平台/机型/版本等原子能力见 m.platform，类型为 Platform）──
        .def_property_readonly("bpu_core_num", [](CauchyKesai &self) { return self.bpu_core_num(); },
             "Number of BPU cores the model was compiled for (hbDNNGetCompileBpuCoreNum)")
        .def_property_readonly("platform", [](CauchyKesai &self) -> cauchykesai::Platform& {
             return self.platform();
         },
             "Platform atomic-capability instance (model-independent): soc/bpu_type/versions/\n"
             "cores/cpu/mem static + bpu_rate/freq/temp/voltage/ion_memory realtime. See Platform.")
        .def("scheduled_cores", [](CauchyKesai &self, int32_t task_id) { return self.scheduled_cores(task_id); },
             py::arg("task_id") = 0,
             "Currently scheduled BPU core indices for a task slot (from that slot's core mask).\n"
             "Empty list = CORE_ANY (SDK auto-select). Construction default:\n"
             "multi-core model -> [0..N-1], single-core -> [] (CORE_ANY).\n"
             "  task_id: task slot index (default 0). Out of range -> IndexError.")
        .def("scheduled_priority", [](CauchyKesai &self, int32_t task_id) { return self.scheduled_priority(task_id); },
             py::arg("task_id") = 0,
             "Currently scheduled task priority for a task slot (that slot's priorities_[task_id]).\n"
             "Set via set_scheduling_params; construction default 0. Range [0,255] (254/255 preempt).\n"
             "Per-slot query via scheduled_priority(task_id); summary()['scheduled_priority'] is slot 0.\n"
             "  task_id: task slot index (default 0). Out of range -> IndexError.")
        .def("set_scheduling_params", &CauchyKesai::set_scheduling_params,
             py::arg("bpu_cores"), py::arg("priority") = -1, py::arg("task_id") = -1,
             "Set per-slot BPU scheduling params (written to hbUCPSchedParam at SubmitTask):\n"
             "backend core mask + task priority. SDK has no SetModelCoreMask API.\n"
             "Per-slot: each task_id owns its own mask (backends_[task_id]) and priority\n"
             "(priorities_[task_id]); set per slot to run different slots on different core\n"
             "subsets / different priorities concurrently (no cross-slot race).\n"
             "  bpu_cores: list of core indices, e.g. [0,1] or [0,1,2,3] -> bitmask of HB_UCP_BPU_CORE_n\n"
             "  Empty list resets to HB_UCP_BPU_CORE_ANY.\n"
             "  priority: -1 (default) = leave existing priority unchanged; [0,255] = set\n"
             "            (254/255 preempt, needs max_time_per_fc + no L2 cache).\n"
             "  task_id: -1 (default) = broadcast to all slots; [0, n_task) = that slot only.\n"
             "           Out of range -> IndexError.\n"
             "Validated eagerly (raises ValueError/IndexError), faithful to SDK behavior:\n"
             "  - bpu_cores index out of [0,63] -> ValueError at call time;\n"
             "  - multi-core model (bpu_core_num>=2): count must == bpu_core_num, all cores valid;\n"
             "  - single-core model: lenient, mask needs >=1 valid core;\n"
             "  - physical-core out of range -> ValueError (skipped if the chip's physical core\n"
             "    count is unknown; see m.platform.physical_core_num for the value);\n"
             "  - priority not in [-1,255] -> ValueError.")

        // ── 校验能力（同步接口内部调 + Python 暴露给用户主动查；返回 bool 不抛）──
        .def("check_input", &CauchyKesai::check_input, py::arg("ion"), py::arg("idx"),
             "Check if ion can bind to input[idx]: properties_match + BPU byte_offset alignment.\n"
             "Returns True ⟺ bind would succeed (no throw). Same source of truth as sync validation.")
        .def("check_output", &CauchyKesai::check_output, py::arg("ion"), py::arg("idx"),
             "Check if ion can bind to output[idx] (same as check_input, output side).")

        // ── bound IONArray（public 成员，def_readwrite）：m.inputs[slot][idx]=ion 纯赋值，无校验 ──
        // Python 侧 list[list[IONArray]]。零拷贝档直接下标读写 bound；bound 唯一真相。
        .def_readwrite("inputs", &CauchyKesai::bound_inputs,
             "Bound input IONArray per slot: m.inputs[slot][idx] = ion (pure assignment, NO validation).\n"
             "list[list[IONArray]] shape [n_task][input_count]. Read m.inputs[slot][idx] to get current binding.")
        .def_readwrite("outputs", &CauchyKesai::bound_outputs,
             "Bound output IONArray per slot: m.outputs[slot][idx] = ion (pure assignment, NO validation).\n"
             "list[list[IONArray]] shape [n_task][output_count].")

        // ── 模板 desc（模型固有性质，只读）──
        .def_property_readonly("input_descs", [](CauchyKesai &s){ return s.input_descs(); },
             "Input template IONArrayDesc list (model input properties, one per input). Use to build matching IONArray: IONArray(m.input_descs[i]).")
        .def_property_readonly("output_descs", [](CauchyKesai &s){ return s.output_descs(); },
             "Output template IONArrayDesc list.")
        .def_property_readonly("n_task", &CauchyKesai::n_task, "Number of task slots.")
        .def_property_readonly("input_count", &CauchyKesai::input_n, "Number of model inputs.")
        .def_property_readonly("output_count", &CauchyKesai::output_n, "Number of model outputs.")
        ;

    // ── 模块级 C 桥: SDK 芯片/库查询（Python 读不了；无参、不依赖模型）──
    // sysfs/proc 可读的原子能力（占用率、核数、频率、温度、CPU/内存等）已在 Python Platform 实现。
    m.def("soc_name",
        []() { return cauchykesai::soc_name(); },
        "Read the SoC name directly from the chip (hbUCPGetSocName, no model needed).\n"
        "Returns the raw soc name string, e.g. 'D-Robotics RDK S600 MCB V0p2';\n"
        "empty string if unreadable.");
    m.def("dnn_version",
        []() { return cauchykesai::dnn_version(); },
        "DNN library version (hbDNNGetVersion, no model needed). Empty string if unreadable.");
    m.def("bpu_version",
        []() { return cauchykesai::bpu_version(); },
        "BPU runtime version (hbUCPGetVersion, no model needed). Empty string if unreadable.");
    // ── BPU 固件版本（依赖模型加载后才初始化的固件；CauchyKesai.summary() 内部注入）──
    m.def("read_bpu_fw_version",
        []() { return cauchykesai::read_bpu_fw_version(); },
        "BPU firmware version (bpu0/fw_version, e.g. '1.1.26'); '' if unreadable.\n"
        "Firmware initializes only after a model is loaded — read on a CauchyKesai instance.");
}
