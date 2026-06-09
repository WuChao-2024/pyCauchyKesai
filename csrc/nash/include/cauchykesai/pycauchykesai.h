

#ifndef _PY_CAUCHY_KESAI_NASH_FEATUREMAP_TOOLS_
#define _PY_CAUCHY_KESAI_NASH_FEATUREMAP_TOOLS_
#include <vector>
#include <filesystem>
#include <chrono>
#include <atomic>

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp.h"
#include "hobot/hb_ucp_sys.h"
#include "ion_array.h"

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

// ALIGN_BPU 来自 common/include/bpu_align.h, 由 CMake 的 HB_BPU_ALIGN 控制
#include "../common/bpu_align.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) CauchyKesai
{
public:
  /**
   * @brief 标准构造：自动分配 n_task 组 ION 输入/输出内存
   *
   * @param model_path       .hbm 模型文件路径
   * @param n_task           并发任务槽数量（1-32，默认 1）
   * @param model_cnt_select packed 模型中选择的模型索引（默认 0）
   */
  CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select);

  /**
   * @brief 零内存构造：预分配 n_task 个任务槽，但不分配任何 ION 输入/输出内存
   *
   * 每个任务槽的输入/输出内存均为空，需在推理前通过
   * set_inputs(ion_inputs, n_task=i) / set_outputs(ion_outputs, n_task=i)
   * 为每个槽分别注入外部 IONArray，实现完全零拷贝的推理流程。
   *
   * @param model_path       .hbm 模型文件路径
   * @param n_task           并发任务槽数量（1-32，默认 1）
   * @param model_cnt_select packed 模型中选择的模型索引（默认 0）
   * @param _no_alloc        内部标志，固定传 true 以区分两个构造函数，用户无需关心
   */
  CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select, bool _no_alloc);

  ~CauchyKesai();
  py::dict s();
  py::dict t();
  bool is_busy(int32_t task_id) const;
  void start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);
  std::vector<py::array> wait(int32_t task_id);
  std::vector<py::array> inference(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);

  /**
   * @brief 安全异步推理提交 — 校验后调用 start()
   *
   * 职责: 所有预飞行校验 → 标记任务槽占用 → 调用纯推理 start()
   *   - task_id / priority 边界检查
   *   - 任务槽占用检查 (is_busy)
   *   - no_alloc 模式 ION 内存绑定检查
   *
   * 推荐用户使用 safe_start() 代替直接调用 start()，
   * 以获得完整的错误诊断信息。
   */
  void safe_start(const std::vector<py::array> &inputs, int32_t task_id, int32_t priority);

  /**
   * @brief 设置 BPU 核心调度参数
   *
   * @param bpu_cores BPU 核心索引列表, e.g. [0,1,2,3]. 空列表重置为 ANY.
   */
  void set_scheduling_params(const std::vector<int32_t> &bpu_cores);

  /**
   * @brief 为指定任务槽设置外部 IONArray 输入（零拷贝）
   *
   * 仅在零内存构造模式（_no_alloc=true）下使用。
   * 传入的 IONArray 列表将直接作为 BPU 输入，不发生任何数据拷贝。
   * 调用方需保证 IONArray 的生命周期覆盖整个推理过程。
   *
   * @param ion_inputs  IONArray 列表，长度须等于模型输入数量
   * @param n_task      任务槽索引，范围 [0, n_task-1]（默认 0）
   *
   * @throws std::out_of_range     若 n_task 超出预分配范围
   * @throws std::invalid_argument 若输入数量、dtype 或 shape 不匹配
   */
  void set_inputs(const std::vector<IONArray *> &ion_inputs, int32_t n_task = 0);

  /**
   * @brief 为指定任务槽设置外部 IONArray 输出（零拷贝）
   *
   * 仅在零内存构造模式（_no_alloc=true）下使用。
   * 传入的 IONArray 列表将直接作为 BPU 输出写入目标，不发生任何数据拷贝。
   * 调用方需保证 IONArray 的生命周期覆盖整个推理过程。
   *
   * @param ion_outputs IONArray 列表，长度须等于模型输出数量
   * @param n_task      任务槽索引，范围 [0, n_task-1]（默认 0）
   *
   * @throws std::out_of_range     若 n_task 超出预分配范围
   * @throws std::invalid_argument 若输出数量、dtype 或 shape 不匹配
   */
  void set_outputs(const std::vector<IONArray *> &ion_outputs, int32_t n_task = 0);

  // 获取标准模式自动创建的 IONArray（no-alloc 模式返回空 vector）
  const std::vector<std::vector<std::shared_ptr<IONArray>>>& owned_inputs() const { return owned_ion_inputs_; }
  const std::vector<std::vector<std::shared_ptr<IONArray>>>& owned_outputs() const { return owned_ion_outputs_; }

  // 暴露给 Python 的 ION 内存视图
  // input_tensors[task_id][input_idx] -> numpy array view of ION memory
  std::vector<std::vector<py::array>> input_tensors;
  // output_tensors[task_id][output_idx] -> numpy array view of ION memory
  std::vector<std::vector<py::array>> output_tensors;
  // 输入输出名称
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // ── 运行时环境信息 (在 _init_model 中通过 SDK API 获取) ──
  const std::string& dnn_version() const { return dnn_version_; }
  const std::string& bpu_version() const { return bpu_version_; }
  const std::string& soc_name()    const { return soc_name_; }
  const std::string& model_desc()  const { return model_desc_; }
  int32_t bpu_core_num() const { return bpu_core_num_; }

private:
  // 共享初始化：加载模型、读取输入输出元信息（不分配 ION 内存）
  void _init_model(const std::string &model_path, int32_t model_cnt_select);

  // 对齐 stride 计算：补齐 stride 数组中的 -1 并计算 alignedByteSize
  // 6 处调用点统一使用此函数，消除代码重复
  static void fixup_aligned_stride(hbDNNTensorProperties &props);

  // 内部公共绑定方法：标准构造和 set_inputs/set_outputs 共用
  void _bind_ion_inputs(const std::vector<IONArray *> &ion_inputs, int32_t n_task);
  void _bind_ion_outputs(const std::vector<IONArray *> &ion_outputs, int32_t n_task);

  // 标准模式自动创建的 IONArray（用 shared_ptr 管理，方便暴露给 Python）
  std::vector<std::vector<std::shared_ptr<IONArray>>> owned_ion_inputs_;   // [task][input_idx]
  std::vector<std::vector<std::shared_ptr<IONArray>>> owned_ion_outputs_;  // [task][output_idx]

  std::string model_path_;
  int32_t n_task_;
  // 零内存模式标志：true 表示不自动分配 ION 内存
  bool no_alloc_mode_ = false;
  std::vector<std::atomic<int>> is_infer;
  std::vector<hbUCPTaskHandle_t> task_handles;
  hbDNNPackedHandle_t packed_dnn_handle;
  int32_t model_count;
  int32_t model_cnt_select_;
  const char **name_list;
  const char *model_name;
  hbDNNHandle_t dnn_handle;

  // ── 运行时环境信息 (从 SDK API 获取) ──
  std::string dnn_version_;   // hbDNNGetVersion()
  std::string bpu_version_;   // hbUCPGetVersion() — BPU 统一计算平台
  std::string soc_name_;      // hbUCPGetSocName() — 芯片型号
  std::string model_desc_;    // hbDNNGetModelDesc() — 模型描述
  int32_t bpu_core_num_ = 0;  // hbDNNGetCompileBpuCoreNum() — 编译核数

  // 输入头相关
  int32_t input_count;
  std::vector<std::vector<hbDNNTensor>> inputs_hbTensor;
  std::vector<int32_t> inputs_numDimension;
  std::vector<std::vector<size_t>> inputs_shape;
  std::vector<std::string> inputs_name;
  std::vector<std::string> inputs_dtype;
  std::vector<int64_t> inputs_alignedByteSize;     // alignedByteSize per input
  std::vector<int32_t> inputs_tensorType;           // hbDNNDataType enum value per input
  std::vector<int32_t> inputs_quantiType;           // quantiType per input (NONE=0, SCALE=1)
  std::vector<int32_t> inputs_quantizeAxis;         // quantizeAxis per input
  std::vector<int32_t> inputs_scaleLen;             // scaleLen per input
  std::vector<std::vector<float>> inputs_scaleData; // scaleData[] per input (反量化系数)
  std::vector<int32_t> inputs_zeroPointLen;         // zeroPointLen per input
  std::vector<std::vector<int32_t>> inputs_zeroPointData; // zeroPointData[] per input
  std::vector<std::vector<int64_t>> inputs_stride;  // stride[] per input
  std::vector<std::string> inputs_desc;             // hbDNNGetInputDesc per input

  // 输出头相关
  int32_t output_count;
  std::vector<std::vector<hbDNNTensor>> outputs_hbTensor;
  std::vector<int32_t> outputs_numDimension;
  std::vector<std::vector<size_t>> outputs_shape;
  std::vector<std::string> outputs_name;
  std::vector<std::string> outputs_dtype;
  std::vector<int64_t> outputs_alignedByteSize;     // alignedByteSize per output
  std::vector<int32_t> outputs_tensorType;           // hbDNNDataType enum value per output
  std::vector<int32_t> outputs_quantiType;           // quantiType per output (NONE=0, SCALE=1)
  std::vector<int32_t> outputs_quantizeAxis;         // quantizeAxis per output
  std::vector<int32_t> outputs_scaleLen;             // scaleLen per output
  std::vector<std::vector<float>> outputs_scaleData; // scaleData[] per output (反量化系数)
  std::vector<int32_t> outputs_zeroPointLen;         // zeroPointLen per output
  std::vector<std::vector<int32_t>> outputs_zeroPointData; // zeroPointData[] per output
  std::vector<std::vector<int64_t>> outputs_stride;  // stride[] per output
  std::vector<std::string> outputs_desc;             // hbDNNGetOutputDesc per output

  // 统计标志
  double mbs;
  uint64_t backend_ = HB_UCP_BPU_CORE_ANY;
};

#endif // _PY_CAUCHY_KESAI_NASH_FEATUREMAP_TOOLS_