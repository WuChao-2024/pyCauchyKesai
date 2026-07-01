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
#include "platform.h"  // Platform 类 + read_bpu_fw_version（summary 注入）

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

namespace py = pybind11;

class __attribute__((visibility("default"))) CauchyKesai
{
public:
  /**
   * @brief 构造：按模型 desc 给每个 slot 建好 input/output IONArray（完善 + allocated）
   *
   * lazy-per-slot：构造只建 IONArrayDesc（input_descs/output_descs）+ 每个 slot 的 input/output
   * IONArray（按 desc 立即分配，defer=false）。每个 slot 的 IONArray 一定 allocated 完善。
   * Auto 档 m([y,uv]) 复用 slot 的 IONArray（from_numpy 覆盖数据）；零拷贝档用户用
   * m.inputs[slot][idx]=ion 替换 bound（纯赋值，无校验）。
   *
   * @param model_path       .hbm 模型文件路径
   * @param n_task           并发任务槽数量（≤0 钳到 1，默认 1）；每个 slot 按 desc 建输入+输出 IONArray
   * @param model_cnt_select packed 模型中选择的模型索引（默认 0）
   */
  CauchyKesai(const std::string &model_path, int32_t n_task, int32_t model_cnt_select);

  ~CauchyKesai();

  py::dict summary();
  py::dict benchmark(int32_t timeout_ms = 0);
  bool is_busy(int32_t task_id) const;

  // ── 推理入口（异步不校验 bound；同步 __call__/inference 才校验）──
  // start()：零拷贝提交（不传 inputs，用 slot 已绑定的 IONArray）
  void start(int32_t task_id = 0);
  // start(inputs)：Auto memcpy 提交（from_numpy 写 slot 的 IONArray，再走零拷贝 start）
  void start(const std::vector<py::array> &inputs, int32_t task_id = 0);
  // wait_done：纯等完成（flush_invalidate 输出 + release task），不返回
  void wait_done(int32_t task_id = 0, int32_t timeout_ms = 0);
  // wait：wait_done + 导出 list[ndarray]（Auto 用）
  std::vector<py::array> wait(int32_t task_id = 0, int32_t timeout_ms = 0);
  // inference(inputs)：Auto 同步（校验 + from_numpy + start + wait + 导出）
  std::vector<py::array> inference(const std::vector<py::array> &inputs, int32_t task_id);
  // inference(task_id)：零拷贝同步（校验 bound + start + wait + 导出）
  std::vector<py::array> inference(int32_t task_id = 0);

  /**
   * @brief 设置 BPU 核心调度参数（per-slot：核掩码 + 优先级）
   */
  void set_scheduling_params(const std::vector<int32_t> &bpu_cores, int32_t priority = -1, int32_t task_id = -1);

  // ── 校验能力（C++ 封装，同步接口内部调 + Python 暴露给用户主动查）──
  // check_input/output: ion 能否绑到 input/output[idx]（properties_match + BPU 对齐），返回 bool 不抛
  bool check_input(std::shared_ptr<IONArray> ion, int32_t idx) const;
  bool check_output(std::shared_ptr<IONArray> ion, int32_t idx) const;

  // ── 模板 desc（模型固有性质，只读）──
  std::vector<IONArrayDesc> input_descs() const;
  std::vector<IONArrayDesc> output_descs() const;

  // ── slot 当前绑定的 IONArray（public，Python def_readwrite：m.inputs[slot][idx]=ion 纯赋值，无校验）──
  // py::list（非 std::vector）：pybind11 def_readwrite 对嵌套 vector 元素级赋值写副本不落成员，
  // py::list 引用语义让 m.inputs[slot][idx]=ion 走 Python 原生 __setitem__ 直接落回成员。
  // 元素是 IONArray wrapper（与 shared_ptr 共享所有权）。start/wait 从 py::list cast shared_ptr 拷局部 vector 保活。
  py::list bound_inputs;   // [task][input_idx]
  py::list bound_outputs;  // [task][output_idx]

  // 计数
  int32_t n_task() const { return n_task_; }
  int32_t input_n() const { return input_count; }
  int32_t output_n() const { return output_count; }

  // 输入输出名称
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // ── 模型 / 实例信息（平台/机型/版本等原子能力见 m.platform）──
  int32_t bpu_core_num() const { return bpu_core_num_; }
  std::vector<int32_t> scheduled_cores(int32_t task_id = 0) const;
  int32_t scheduled_priority(int32_t task_id = 0) const;
  // 平台原子能力实例（机型无关；Python m.platform 经 def_property_readonly 暴露）
  cauchykesai::Platform& platform() { return platform_; }

private:
  // 共享初始化：加载模型、读取输入输出元信息、建模板 desc + 每个 slot 的 bound IONArray
  void _init_model(const std::string &model_path, int32_t model_cnt_select);

  // start 内部公共体（抢 slot + 可选 from_numpy + 现建 hbDNNTensor + InferV2 + SubmitTask）
  void _start_impl(int32_t task_id, const std::vector<py::array> *inputs);

  // 模型权威模板 IONArray（性质已 resolve，内存未分配；input_descs/output_descs 的来源）
  std::vector<std::shared_ptr<IONArray>> input_templates_;
  std::vector<std::shared_ptr<IONArray>> output_templates_;

  std::string model_path_;
  int32_t n_task_;
  std::vector<std::atomic<int>> is_infer;
  std::vector<hbUCPTaskHandle_t> task_handles;
  hbDNNPackedHandle_t packed_dnn_handle;
  int32_t model_count;
  int32_t model_cnt_select_;
  const char **name_list;
  const char *model_name;
  hbDNNHandle_t dnn_handle;

  // ── 模型 / 调度信息 ──
  std::string model_desc_;
  int32_t bpu_core_num_ = 0;
  int32_t physical_core_num_ = -1;
  int32_t bpu_align_ = 0;           // S100/S100P=32, S600=64

  int32_t input_count;
  int32_t output_count;

  // 统计标志
  double mbs;
  // per-slot 调度（size = n_task_）
  std::vector<uint64_t> backends_;
  std::vector<int32_t> priorities_;

  // 平台原子能力（机型无关，不依赖模型）。引用全局单例 global_platform()，
  // 多个 CauchyKesai 共用同一实例；summary() 注入 result["platform"]。
  cauchykesai::Platform& platform_;
};

#endif // _PY_CAUCHY_KESAI_NASH_FEATUREMAP_TOOLS_
