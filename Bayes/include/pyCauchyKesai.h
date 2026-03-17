
#ifndef _PY_CAUCHY_KESAI_X5_BAYES_
#define _PY_CAUCHY_KESAI_X5_BAYES_

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
#include <atomic>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

#define RDK_CHECK_SUCCESS(value, errmsg) \
  do                                     \
  {                                      \
    auto ret_code = value;               \
    if (ret_code != 0)                   \
    {                                    \
      throw std::runtime_error(errmsg);  \
    }                                    \
  } while (0);

namespace py = pybind11;

class __attribute__((visibility("default"))) CauchyKesai
{
public:
  CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select);
  ~CauchyKesai();
  py::dict s();
  py::dict t();
  bool is_busy(int32_t task_id) const;
  void start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);
  std::vector<py::array> wait(int32_t task_id);
  std::vector<py::array> inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);

  // 暴露给 Python 的 ION 内存视图
  // input_tensors[task_id][input_idx] -> numpy array view of ION memory
  std::vector<std::vector<py::array>> input_tensors;
  // output_tensors[task_id][output_idx] -> numpy array view of ION memory
  std::vector<std::vector<py::array>> output_tensors;
  // 输入输出名称
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

private:
  std::string model_path_;
  const char *modelFileName;
  int32_t n_task_;
  std::vector<std::atomic<int>> is_infer;

  // X5 使用 hbDNNTaskHandle_t（而非 Nash 的 hbUCPTaskHandle_t）
  std::vector<hbDNNTaskHandle_t> task_handles;

  // X5 使用 hbPackedDNNHandle_t（而非 Nash 的 hbDNNPackedHandle_t）
  hbPackedDNNHandle_t packed_dnn_handle;
  int32_t model_count;
  int32_t model_cnt_select_;
  const char **name_list;
  const char *model_name;
  hbDNNHandle_t dnn_handle;
  hbDNNTensorProperties input_properties;
  hbDNNTensorProperties output_properties;

  // 输入头相关
  // 使用裸指针数组（与参考实现一致），避免 std::vector 重分配导致 hbDNNInfer 指针失效
  int32_t input_count;
  std::vector<hbDNNTensor *> inputs_hbTensor;   // [n_task_], 每个是 new hbDNNTensor[input_count]
  std::vector<int32_t> inputs_numDimension;
  std::vector<std::vector<size_t>> inputs_shape;
  std::vector<size_t> inputs_byteSize;
  std::vector<std::string> inputs_name;
  std::vector<std::string> inputs_dtype;
  // 图像类型输入（NV12/RGB/BGR 等）validShape 是逻辑形状，实际内存由 alignedByteSize 决定，
  // 校验和 s() 显示均用展平的一维 (alignedByteSize,) uint8，与 input_tensors 保持一致。
  std::vector<bool> inputs_is_image;

  // 输出头相关
  int32_t output_count;
  std::vector<hbDNNTensor *> outputs_hbTensor;  // [n_task_], 每个是 new hbDNNTensor[output_count]
  std::vector<int32_t> outputs_numDimension;
  std::vector<std::vector<size_t>> outputs_shape;
  std::vector<std::string> outputs_name;
  std::vector<std::string> outputs_dtype;

  // 统计标志
  double mbs;
};

#endif // _PY_CAUCHY_KESAI_X5_BAYES_
