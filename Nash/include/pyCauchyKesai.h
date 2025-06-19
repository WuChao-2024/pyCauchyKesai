

#ifndef _PY_CAUCHY_KESAI_S100_FEATUREMAP_TOOLS_
#define _PY_CAUCHY_KESAI_S100_FEATUREMAP_TOOLS_
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <queue>
#include <utility>
#include <vector>
#include <cstring>
#include <filesystem>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// #include "gflags/gflags.h"
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
#define EMPTY ""

#define RDK_CHECK_SUCCESS(value, errmsg) \
  do                                     \
  {                                      \
    /*value can be call of function*/    \
    auto ret_code = value;               \
    if (ret_code != 0)                   \
    {                                    \
      throw std::runtime_error(errmsg);  \
    }                                    \
  } while (0);

// BPU requires w stride alignment, J6P is aligned to 64, and the rest are aligned to 32.
#define ALIGN(value, alignment) (((value) + ((alignment)-1)) & ~((alignment)-1))
#define ALIGN_64(value) ALIGN(value, 64)
#define ALIGN_32(value) ALIGN(value, 32)

namespace py = pybind11;

class __attribute__((visibility("default"))) CauchyKesai
{
public:
  // ACTPloicyRGBEncoder();
  CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select);
  ~CauchyKesai();
  void s();
  void t();
  void start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);
  std::vector<py::array> wait(int32_t task_id);
  std::vector<py::array> inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);
private:
  std::string model_path_;
  const char *modelFileName;
  int32_t n_task_;
  std::vector<int> is_infer;
  std::vector<hbUCPTaskHandle_t> task_handles;
  hbDNNPackedHandle_t packed_dnn_handle;
  int32_t model_count;
  int32_t model_cnt_select_;
  const char **name_list;
  const char *model_name;
  hbDNNHandle_t dnn_handle;
  hbDNNTensorProperties input_properties;
  hbDNNTensorProperties output_properties;

  // 输入头相关
  int32_t input_count;
  std::vector<std::vector<hbDNNTensor>> inputs_hbTensor;
  std::vector<int32_t> inputs_numDimension;
  std::vector<std::vector<size_t>> inputs_shape;
  std::vector<size_t> inputs_byteSize;
  std::vector<std::string> inputs_name;
  std::vector<std::string> inputs_dtype;

  // 输出头相关
  int32_t output_count;
  std::vector<std::vector<hbDNNTensor>> outputs_hbTensor;
  std::vector<int32_t> outputs_numDimension;
  std::vector<std::vector<size_t>> outputs_shape;
  std::vector<std::string> outputs_name;
  std::vector<std::string> outputs_dtype;

  // pybind11 相关
  // std::vector<std::vector<py::array>> results;

  // 统计标志
  double mbs;
};

#endif // _PY_CAUCHY_KESAI_S100_FEATUREMAP_TOOLS_