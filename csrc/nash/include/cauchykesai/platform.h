#ifndef _CAUCHYKESAI_PLATFORM_H_
#define _CAUCHYKESAI_PLATFORM_H_

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace cauchykesai {

// ── 平台 / 芯片 C 桥（SDK 芯片查询 + 调度器内部辅助）──────────────────
// 这一组是 SDK 芯片 / 库查询（hbUCPGet* / hbDNNGet*），均无参、不依赖加载模型，
// 构造前即可取。CauchyKesai 构造时调 physical_core_count / bpu_align 做核调度校验。
// Platform 类内部也复用这些函数（消除 Python 侧双份真相）。

/**
 * @brief 直接从芯片读 SoC 名称（hbUCPGetSocName，无参，不依赖模型）。
 * @return SoC 名称字符串（如 "D-Robotics RDK S600 MCB V0p2"）；读不到返回空串。
 */
std::string soc_name();

/**
 * @brief DNN 库版本（hbDNNGetVersion，无参，不依赖模型）。
 * @return 版本字符串；读不到返回空串。
 */
std::string dnn_version();

/**
 * @brief BPU 运行时版本（hbUCPGetVersion，无参，不依赖模型）。
 * @return 版本字符串；读不到返回空串。
 */
std::string bpu_version();

/**
 * @brief 芯片物理 BPU 核数（CauchyKesai 调度器内部辅助，非对外原子能力）。
 *
 * 优先读 /sys/devices/system/bpu/core_num，失败回退数 bpu<digits> 目录。
 * CauchyKesai 构造时读一次供核调度校验；对外请用 Platform.physical_core_num。
 * 读不到（非板端 / 无 sysfs）返回 -1。
 */
int32_t physical_core_count();

/**
 * @brief 当前 SoC 的 BPU 对齐字节数（运行时检测，进程内缓存）。
 *
 * 由 soc_name() 的板卡 token 推断（与 Platform.bpu_type 同一套推断逻辑）：
 * S100P / S100 → 32，S600 → 64。soc_name 读不到（非板端）时回退默认 64
 * （kDefaultBpuAlign，仅检测失败的兜底）。
 * CauchyKesai 构造时取一次喂 IONArray 模板。仅供内部使用，不暴露给 Python。
 */
int32_t bpu_align();

// ════════════════════════════════════════════════════════════════════════════
// Platform 类：板卡 / 芯片平台原子能力（机型无关，不依赖加载模型）
//
// 静态属性（构造时缓存一次，只读 property）；实时测量走 method（每次现读 sysfs）。
// CauchyKesai 持有一个实例 (m.platform)；也可独立 Platform() 使用。
// summary() 返原生 py::dict（16 key，已 jsonable）；s()/t() 展示留 Python 子类。
// ════════════════════════════════════════════════════════════════════════════

class __attribute__((visibility("default"))) Platform {
public:
    /// 构造时读一次静态属性（SDK 芯片查询 + sysfs/proc），缓存供只读 property 返回。
    Platform();

    // ── 8 静态属性（只读，构造时缓存）──
    std::string soc_name() const { return _soc_name; }
    std::string bpu_type() const { return _bpu_type; }
    std::string dnn_version() const { return _dnn_version; }
    std::string bpu_version() const { return _bpu_version; }
    int32_t physical_core_num() const { return _physical_core_num; }
    std::string cpu_model() const { return _cpu_model; }
    int32_t cpu_count() const { return _cpu_count; }
    int32_t mem_total_mb() const { return _mem_total_mb; }

    // ── 7 实时方法（每次现读 sysfs）──
    py::list bpu_rate() const;
    py::list bpu_freq() const;
    py::list cpu_freq() const;
    py::dict temperature(const std::string &name = "") const;
    py::dict voltage(const std::string &name = "") const;
    py::dict ion_memory(const std::string &name = "") const;
    std::string ucp_library_path() const;

    // ── summary / repr（s() 反调 Python _render.render_platform，展示逻辑留 Python）──
    py::dict summary() const;
    std::string repr() const;
    void s() const;  // 内部 print(render_platform(summary()))，返 None 等价

private:
    // 静态缓存（构造时读一次）
    std::string _soc_name;
    std::string _bpu_type;
    std::string _dnn_version;
    std::string _bpu_version;
    std::string _cpu_model;
    int32_t _physical_core_num = -1;
    int32_t _cpu_count = 0;
    int32_t _mem_total_mb = -1;

    // ── 内部读取辅助（原 platform.py 私有函数的 C++ 对应）──
    static std::string _infer_arch(const std::string &name);
    static std::tuple<std::string, int32_t> _read_cpuinfo();
    static int32_t _read_mem_total_mb();
    static int32_t _read_mem_available_mb();
};

/**
 * @brief BPU 固件版本（bpu0/fw_version，取 bpu0 代表）；读不到为 "".
 *
 * 依赖模型加载后才初始化的固件——应由 CauchyKesai（模型已加载）上读取，
 * 故 C++ summary() 在此注入。模块级函数（Python 经 bindings 暴露）。
 */
std::string read_bpu_fw_version();

/**
 * @brief 进程级全局 Platform 单例（import 即构造，magic static 线程安全）。
 *
 * 机型无关、不依赖模型；CauchyKesai 复用此实例（不再每实例嵌一份），
 * 模块级也经此暴露为 pycauchykesai.platform。用户仍可独立 Platform() 实例化。
 */
Platform& global_platform();

}  // namespace cauchykesai

#endif  // _CAUCHYKESAI_PLATFORM_H_
