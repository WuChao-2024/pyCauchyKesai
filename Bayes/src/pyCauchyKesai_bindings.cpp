#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pyCauchyKesai.h"

namespace py = pybind11;

PYBIND11_MODULE(pyCauchyKesai, m)
{
    m.attr("__version__") = "0.0.9_AI_Native";
    m.attr("__author__") = "Cauchy - WuChao in D-Robotics";
    m.attr("__date__") = "2025-05-30";
    m.attr("__doc__") = "GNU AFFERO GENERAL PUBLIC LICENSE Version 3";

    py::class_<CauchyKesai>(m, "CauchyKesai")
        .def(py::init<const std::string &, int32_t, int32_t>(),
             py::arg("model_path"),
             py::arg("n_task") = 1,
             py::arg("model_cnt_select") = 0)
        .def("s", &CauchyKesai::s, "Return model summary as a structured dict (also pretty-prints in terminal).")
        .def("t", &CauchyKesai::t, "Dirty run once and return benchmark result as a structured dict.")
        .def("is_busy", &CauchyKesai::is_busy,
             "Return True if the given task slot is currently running inference.",
             py::arg("task_id") = 0)
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
             py::arg("priority") = 0)
        .def_readonly("input_tensors", &CauchyKesai::input_tensors,
                      "ION memory views for input tensors: list[list[np.ndarray]], shape: [n_task][input_count]")
        .def_readonly("output_tensors", &CauchyKesai::output_tensors,
                      "ION memory views for output tensors: list[list[np.ndarray]], shape: [n_task][output_count]")
        .def_readonly("input_names", &CauchyKesai::input_names,
                      "Input tensor names: list[str]")
        .def_readonly("output_names", &CauchyKesai::output_names,
                      "Output tensor names: list[str]");
}
