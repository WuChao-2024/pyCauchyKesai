#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cauchykesai/pycauchykesai.h"

namespace py = pybind11;

PYBIND11_MODULE(pycauchykesai, m)
{
    m.attr("__version__") = "0.2.0";
    m.attr("__author__")  = "Cauchy - WuChao in D-Robotics";
    m.attr("__date__")    = "2025-05-30";
    m.attr("__doc__")     = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3";

    // ── IONArray: ION 内存管理器 ──────────────────────────────────────────
    //
    // 纯内存管理类，不继承 np.ndarray。
    //   .as_array()          → 返回标准 np.ndarray (零拷贝视图)
    //   .sub_view(dtype, shape, offset) → 元素级偏移子视图
    //   .allocate(cached)    → 懒分配：按 dtype×shape 分配 ION 内存
    //   .phy_addr / .mem_size / .is_cached / .is_allocated
    //   .flush() / .invalidate()
    //   .shape / .dtype / .ndim / .size / .nbytes  (numpy 兼容)
    //
    //   cached=True  → hbUCPMallocCached + CPU cache 一致 (默认)
    //   cached=False → hbUCPMalloc        无 CPU cache, 更高 BPU 带宽
    //   defer=True   → 只记录 dtype+shape, 不分配, 稍后调 .allocate()
    //
    py::class_<IONArray, std::shared_ptr<IONArray>>(m, "IONArray",
        "ION memory manager. Allocates physical memory accessible by both CPU and BPU.\n"
        "Use .as_array() to get a plain np.ndarray for CPU operations.\n"
        "Use .sub_view(dtype, shape, offset) for element-level partitioned views.\n"
        "Use defer=True + .allocate() for lazy allocation.")
        // ── 构造函数 ──
        .def(py::init<py::dtype, std::vector<ssize_t>, bool, bool>(),
             py::arg("dtype"), py::arg("shape"),
             py::arg("cached") = true, py::arg("defer") = false,
             "Create an IONArray.\n"
             "  dtype: element data type (np.dtype)\n"
             "  shape: logical shape (list/tuple of ints)\n"
             "  cached=True → hbUCPMallocCached (default)\n"
             "  cached=False → hbUCPMalloc (no CPU cache, higher BPU bandwidth)\n"
             "  defer=False → allocate immediately (default)\n"
             "  defer=True → store dtype/shape only, call .allocate() later")
        .def(py::init<py::dtype, std::vector<ssize_t>, uint64_t, bool>(),
             py::arg("dtype"), py::arg("shape"), py::arg("byte_size"),
             py::arg("cached") = true,
             "Create an IONArray with explicit byte_size (for padding/alignment).\n"
             "  byte_size must be >= dtype.itemsize × product(shape)")
        // ── 懒分配 ──
        .def("allocate", &IONArray::allocate, py::arg("cached") = true,
             "Allocate ION memory for this IONArray using stored dtype×shape.\n"
             "Use after constructing with defer=True.\n"
             "Throws RuntimeError if already allocated.")
        // ── 内存属性 ──
        .def_property_readonly("phy_addr", &IONArray::phy_addr,
             "BPU physical address (uint64)")
        .def_property_readonly("mem_size", &IONArray::mem_size,
             "ION allocation size in bytes")
        .def_property_readonly("is_cached", &IONArray::is_cached,
             "True if allocated with CPU cache (cached=True)")
        .def_property_readonly("is_allocated", &IONArray::is_allocated,
             "True if ION memory has been allocated")
        .def_property_readonly("dtype", [](IONArray &self) { return self.dtype(); },
             "Element dtype of this IONArray (np.dtype)")
        .def_property_readonly("shape", [](IONArray &self) -> py::tuple {
                auto t = py::tuple(self.shape().size());
                for (size_t i = 0; i < self.shape().size(); i++)
                    t[i] = self.shape()[i];
                return t;
             },
             "Logical shape of this IONArray as tuple, consistent with numpy.ndarray.shape")
        .def_property_readonly("ndim", [](IONArray &self) { return self.ndim(); },
             "Number of dimensions (consistent with numpy.ndarray.ndim)")
        .def_property_readonly("size", [](IONArray &self) { return self.size(); },
             "Total number of elements (consistent with numpy.ndarray.size)")
        .def_property_readonly("nbytes", [](IONArray &self) { return self.nbytes(); },
             "Total bytes (theoretical when unallocated, actual when allocated). "
             "Consistent with numpy.ndarray.nbytes.")
        // ── Cache 操作 ──
        .def("flush", &IONArray::flush,
             "Flush CPU cache: ensures ION memory is up-to-date (no-op on uncached/unallocated)")
        .def("invalidate", &IONArray::invalidate,
             "Invalidate CPU cache: ensures CPU sees latest ION data (no-op on uncached/unallocated)")
        // ── as_array: 导出 np.ndarray ──
        .def("as_array", &IONArray::as_array,
             "Return a plain np.ndarray view of this ION memory (zero-copy).\n"
             "The returned array holds a reference to this IONArray, keeping the\n"
             "ION memory alive as long as the numpy array exists.\n"
             "Throws RuntimeError if IONArray is not allocated.")
        // ── sub_view: 元素级偏移子视图 ──
        .def("sub_view", &IONArray::sub_view,
             py::arg("dtype"), py::arg("shape"), py::arg("offset"),
             "Create a non-owning sub-view at an element offset.\n"
             "  offset: skip N elements of THIS IONArray's dtype (NOT bytes).\n"
             "  byte_offset = offset × this.dtype().itemsize()\n"
             "  The sub-view keeps its parent alive via shared_ptr.\n"
             "  Useful for partitioning ION memory into BPU input/output slots.\n"
             "  Throws RuntimeError if IONArray is not allocated.");

    // ── CauchyKesai ───────────────────────────────────────────────────────
    py::class_<CauchyKesai>(m, "CauchyKesai")
        .def(py::init<const std::string &, int32_t, int32_t>(),
             py::arg("model_path"), py::arg("n_task") = 1,
             py::arg("model_cnt_select") = 0)
        .def(py::init<const std::string &, int32_t, int32_t, bool>(),
             py::arg("model_path"), py::arg("n_task") = 1,
             py::arg("model_cnt_select") = 0, py::arg("_no_alloc") = true)
        .def("s", &CauchyKesai::s)
        .def("t", &CauchyKesai::t)
        .def("is_busy", &CauchyKesai::is_busy, py::arg("task_id") = 0)
        .def("start", &CauchyKesai::start,
             py::arg("inputs"), py::arg("task_id") = 0, py::arg("priority") = 0,
             "Submit async inference with NO validation (advanced users only).\n"
             "Caller must guarantee: task_id/priority valid, task slot free, ION bound.\n"
             "Prefer safe_start() for automatic pre-flight checks.")
        .def("safe_start", &CauchyKesai::safe_start,
             py::arg("inputs"), py::arg("task_id") = 0, py::arg("priority") = 0,
             "Validate task state and submit async inference.\n"
             "Performs all pre-flight checks (task_id, priority, is_busy, ION binding)\n"
             "before calling start(). Recommended over calling start() directly.")
        .def("wait", &CauchyKesai::wait, py::arg("task_id") = 0)
        .def("inference", &CauchyKesai::inference,
             py::arg("inputs"), py::arg("task_id") = 0, py::arg("priority") = 0)
        .def("__call__", &CauchyKesai::inference,
             py::arg("inputs"), py::arg("task_id") = 0, py::arg("priority") = 0)
        .def_readonly("input_tensors", &CauchyKesai::input_tensors)
        .def_readonly("output_tensors", &CauchyKesai::output_tensors)
        .def_readonly("input_names", &CauchyKesai::input_names)
        .def_readonly("output_names", &CauchyKesai::output_names)
        // ── 运行时环境信息 ──
        .def_property_readonly("dnn_version", [](CauchyKesai &self) { return self.dnn_version(); },
             "DNN library version (hbDNNGetVersion)")
        .def_property_readonly("bpu_version", [](CauchyKesai &self) { return self.bpu_version(); },
             "BPU runtime version (hbUCPGetVersion)")
        .def_property_readonly("soc_name", [](CauchyKesai &self) { return self.soc_name(); },
             "SOC chip name (hbUCPGetSocName)")
        .def_property_readonly("model_desc", [](CauchyKesai &self) { return self.model_desc(); },
             "Model description (hbDNNGetModelDesc), 'N/A' if not available")
        .def_property_readonly("bpu_core_num", [](CauchyKesai &self) { return self.bpu_core_num(); },
             "Number of BPU cores the model was compiled for (hbDNNGetCompileBpuCoreNum)")
        .def("set_scheduling_params", &CauchyKesai::set_scheduling_params,
             py::arg("bpu_cores"),
             "Set BPU core mask for multi-core inference.\n"
             "  bpu_cores: list of core indices, e.g. [0, 1, 2] or [0, 1, 2, 3]\n"
             "  Empty list or [] resets to HB_UCP_BPU_CORE_ANY.")
        // ── set_inputs/set_outputs: 接受 IONArray 列表 ──
        .def("set_inputs",
             [](CauchyKesai &self, const py::list &py_ion_inputs, int32_t n_task) {
                 std::vector<IONArray *> ion_ptrs;
                 for (auto item : py_ion_inputs)
                     ion_ptrs.push_back(item.cast<std::shared_ptr<IONArray>>().get());
                 self.set_inputs(ion_ptrs, n_task);
             },
             py::arg("ion_inputs"), py::arg("n_task") = 0,
             "Bind external IONArray inputs for zero-copy (no_alloc mode only).")
        .def("set_outputs",
             [](CauchyKesai &self, const py::list &py_ion_outputs, int32_t n_task) {
                 std::vector<IONArray *> ion_ptrs;
                 for (auto item : py_ion_outputs)
                     ion_ptrs.push_back(item.cast<std::shared_ptr<IONArray>>().get());
                 self.set_outputs(ion_ptrs, n_task);
             },
             py::arg("ion_outputs"), py::arg("n_task") = 0,
             "Bind external IONArray outputs for zero-copy (no_alloc mode only).")
        // ── ion_inputs/ion_outputs: 返回标准模式自动创建的 IONArray ──
        .def("ion_inputs",
             [](CauchyKesai &self, int32_t task_id) -> py::list {
                 auto &inputs = self.owned_inputs();
                 py::list result;
                 if (task_id >= 0 && task_id < (int32_t)inputs.size()) {
                     for (auto &sp : inputs[task_id])
                         result.append(py::cast(sp));
                 }
                 return result;
             },
             py::arg("task_id") = 0,
             "Get auto-allocated IONArray inputs (standard mode). "
             "Returns empty list in no-alloc mode.")
        .def("ion_outputs",
             [](CauchyKesai &self, int32_t task_id) -> py::list {
                 auto &outputs = self.owned_outputs();
                 py::list result;
                 if (task_id >= 0 && task_id < (int32_t)outputs.size()) {
                     for (auto &sp : outputs[task_id])
                         result.append(py::cast(sp));
                 }
                 return result;
             },
             py::arg("task_id") = 0,
             "Get auto-allocated IONArray outputs (standard mode). "
             "Returns empty list in no-alloc mode.");
}
