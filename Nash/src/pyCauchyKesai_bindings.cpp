#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pyCauchyKesai.h"

namespace py = pybind11;

PYBIND11_MODULE(pyCauchyKesai, m)
{
     // 添加模块元信息
     m.attr("__version__") = "0.0.8";                            // 版本号
     m.attr("__author__") = "Cauchy - WuChao in D-Robotics";     // 作者
     m.attr("__date__") = "2025-05-30";                          // 日期
     m.attr("__doc__") = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3"; // 模块描述

     py::class_<CauchyKesai>(m, "CauchyKesai")
         .def(py::init<const std::string &, int32_t, int32_t>(),
              py::arg("model_path"),
              py::arg("n_task") = 1,
              py::arg("model_cnt_select") = 0)
         .def("s", &CauchyKesai::s, "Show BPU Model Summarys.")
         .def("t", &CauchyKesai::t, "Dirty Run once and print performance data.")
         .def("start", &CauchyKesai::start,
              "Start inference with list of numpy arrays and a task ID",
              py::arg("inputs"),
              py::arg("task_id") = 0,
              py::arg("priority") = 0)
         .def("wait", &CauchyKesai::wait,
              "Wait for inference result and return output numpy array without copy",
              py::arg("task_id") = 0)
         .def("inference", &CauchyKesai::inference,
              "Safe Check + Start + Wait.",
              py::arg("inputs"),
              py::arg("task_id") = 0,
              py::arg("priority") = 0)
         .def("__call__", &CauchyKesai::inference,
              "Safe Check + Start + Wait.",
              py::arg("inputs"),
              py::arg("task_id") = 0,
              py::arg("priority") = 0);
}