// 平台 / 芯片 C 桥 + Platform 类：板卡 / 芯片平台原子能力（机型无关）。
//
// 两部分：
//   1. namespace cauchykesai 自由函数：SDK 芯片查询（hbUCPGet* / hbDNNGet*）+
//      调度器内部辅助（physical_core_count / bpu_align）。CauchyKesai 构造依赖。
//   2. Platform 类：8 静态 property + 7 实时方法 + summary()。sysfs/proc 读取
//      全在此（原 platform.py 的 C++ 化，消除双份真相）。
#include "cauchykesai/platform.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <hobot/dnn/hb_dnn.h>  // hbDNNGetVersion
#include <hobot/hb_ucp.h>      // hbUCPGetSocName / hbUCPGetVersion

namespace py = pybind11;

namespace cauchykesai {

// ════════════════════════════════════════════════════════════════════════════
// SDK 芯片 / 库查询（自由函数，CauchyKesai 构造 + Platform 类复用）
// ════════════════════════════════════════════════════════════════════════════

std::string soc_name() {
    const char *raw = ::hbUCPGetSocName();  // 无参，芯片直读，不需要模型
    return raw ? std::string(raw) : std::string();
}

std::string dnn_version() {
    const char *raw = ::hbDNNGetVersion();  // 无参，库版本，不需要模型
    return raw ? std::string(raw) : std::string();
}

std::string bpu_version() {
    const char *raw = ::hbUCPGetVersion();  // 无参，BPU 运行时版本，不需要模型
    return raw ? std::string(raw) : std::string();
}

// ── physical_core_count（调度器内部辅助，非对外原子能力）──────────

static constexpr const char *BPU_SYSFS = "/sys/devices/system/bpu";
static constexpr const char *CPU_SYSFS = "/sys/devices/system/cpu";
static constexpr const char *THERMAL_SYSFS = "/sys/class/thermal";
static constexpr const char *HWMON_SYSFS = "/sys/class/hwmon";
static constexpr const char *PROC_CPUINFO = "/proc/cpuinfo";
static constexpr const char *PROC_MEMINFO = "/proc/meminfo";
static constexpr const char *ION_HEAPS = "/sys/kernel/debug/ion/heaps";
static constexpr const char *PROC_MAPS = "/proc/self/maps";

// 监控的三块 ION heap（debugfs /sys/kernel/debug/ion/heaps/<name>，需 root）。
static const std::vector<std::string> ION_HEAP_NAMES = {"ion_cma", "cma_reserved", "carveout"};

// 从目录名解析核编号；仅认 "bpu<纯数字>" 形态，否则返回 -1。
static int32_t parse_core_id(const char *name) {
    if (std::strncmp(name, "bpu", 3) != 0) return -1;
    const char *p = name + 3;
    if (*p == '\0') return -1;  // 正好 "bpu"
    for (const char *q = p; *q; ++q) {
        if (!std::isdigit(static_cast<unsigned char>(*q))) return -1;
    }
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 1023) return -1;
    return static_cast<int32_t>(v);
}

// 从目录名解析 CPU 编号；仅认 "cpu<纯数字>" 形态，否则返回 -1。
static int32_t parse_cpu_id(const char *name) {
    if (std::strncmp(name, "cpu", 3) != 0) return -1;
    const char *p = name + 3;
    if (*p == '\0') return -1;
    for (const char *q = p; *q; ++q) {
        if (!std::isdigit(static_cast<unsigned char>(*q))) return -1;
    }
    char *end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v < 0 || v > 65535) return -1;
    return static_cast<int32_t>(v);
}

// 读一个文件首段整数（O_RDONLY|CLOEXEC）；空串/越界返回 -1。
static int32_t read_int_file(const std::string &path, long max_value) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -1;
    char buf[32] = {0};
    ssize_t n = ::read(fd, buf, sizeof(buf) - 1);
    ::close(fd);
    if (n <= 0) return -1;
    char *end = nullptr;
    long v = std::strtol(buf, &end, 10);
    if (end == buf || v < 0 || v > max_value) return -1;
    return static_cast<int32_t>(v);
}

// 读文件全文到 std::string；失败返回空串。用于需要正则/逐行解析的 sysfs/proc。
static std::string read_text_file(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return std::string();
    std::string out;
    char buf[4096];
    ssize_t n;
    while ((n = ::read(fd, buf, sizeof(buf))) > 0)
        out.append(buf, static_cast<size_t>(n));
    ::close(fd);
    return out;
}

// 读 /sys/devices/system/bpu/core_num（内核导出的权威物理核数）首段正整数。
static int32_t read_top_level_core_num() {
    return read_int_file(std::string(BPU_SYSFS) + "/core_num", 1024);
}

// 回退：数 /sys/devices/system/bpu/bpu<digits> 目录个数。
static int32_t count_bpu_dirs() {
    DIR *d = ::opendir(BPU_SYSFS);
    if (!d) return -1;
    int32_t cnt = 0;
    struct dirent *e;
    while ((e = ::readdir(d)) != nullptr) {
        if (parse_core_id(e->d_name) >= 0) ++cnt;
    }
    ::closedir(d);
    return cnt > 0 ? cnt : -1;
}

int32_t physical_core_count() {
    int32_t v = read_top_level_core_num();
    if (v > 0) return v;
    return count_bpu_dirs();
}

// ── BPU 对齐值（运行时检测，进程内缓存）──────────────────────────────
// 兜底默认对齐值：仅当 hbUCPGetSocName 读不到（非板端）时使用。
// 板上运行时检测恒胜出（S100/S100P=32, S600=64），此 64 仅理论兜底。
constexpr int32_t kDefaultBpuAlign = 64;
int32_t bpu_align() {
    // 机型进程内不变；C++11 magic static 保证只算一次且线程安全。
    static const int32_t cached = []() {
        const std::string name = soc_name();
        // token 匹配（长串优先，避免 S100P 被 S100 截胡；与 Platform._infer_arch 一致）
        if (name.find("S100P") != std::string::npos) return 32;
        if (name.find("S600")  != std::string::npos) return 64;
        if (name.find("S100")  != std::string::npos) return 32;
        return kDefaultBpuAlign;
    }();
    return cached;
}

// ════════════════════════════════════════════════════════════════════════════
// Platform 类内部辅助（sysfs/proc 读取，手写解析不用 <regex>）
// ════════════════════════════════════════════════════════════════════════════

// 排序后的 BPU 核 id 列表（扫 /sys/devices/system/bpu/bpu<digits>）。空 = 无 sysfs。
static std::vector<int32_t> bpu_core_ids() {
    std::vector<int32_t> ids;
    DIR *d = ::opendir(BPU_SYSFS);
    if (!d) return ids;
    struct dirent *e;
    while ((e = ::readdir(d)) != nullptr) {
        int32_t id = parse_core_id(e->d_name);
        if (id >= 0) ids.push_back(id);
    }
    ::closedir(d);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// 排序后的 CPU id 列表（扫 /sys/devices/system/cpu/cpu<digits>）。
static std::vector<int32_t> cpu_ids() {
    std::vector<int32_t> ids;
    DIR *d = ::opendir(CPU_SYSFS);
    if (!d) return ids;
    struct dirent *e;
    while ((e = ::readdir(d)) != nullptr) {
        int32_t id = parse_cpu_id(e->d_name);
        if (id >= 0) ids.push_back(id);
    }
    ::closedir(d);
    std::sort(ids.begin(), ids.end());
    return ids;
}

// 读一块 ION heap 的 (used_bytes, total_bytes)；读不到 (-1, -1)。
//
// 按内容关键字解析（不依赖行号），免疫 deferred free / total orphaned 等尾部行
// 的行号漂移——旧式按行号读对 chunk/system 等 heap 会读错。
//   total: 子串 "heap total size\s+(\d+)"
//   used : 行首空白 + "total" + 空白 + (\d+) + 行尾（等价 Python ^\s+total\s+(\d+)\s*$ MULTILINE）
static std::tuple<int32_t, int32_t> ion_heap_used_total(const std::string &name) {
    std::string txt = read_text_file(std::string(ION_HEAPS) + "/" + name);
    if (txt.empty()) return std::make_tuple(-1, -1);

    int32_t used = -1, total = -1;
    // total: 找子串 "heap total size" 后首个数字
    {
        std::string key = "heap total size";
        std::string::size_type p = txt.find(key);
        if (p != std::string::npos) {
            std::string::size_type i = p + key.size();
            while (i < txt.size() && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
            if (i < txt.size() && std::isdigit(static_cast<unsigned char>(txt[i]))) {
                long v = std::strtol(txt.c_str() + i, nullptr, 10);
                if (v >= 0) total = static_cast<int32_t>(v);
            }
        }
    }
    // used: 逐行找「行首空白 + total + 空白 + 数字 + 行尾」
    {
        std::string::size_type pos = 0;
        while (pos < txt.size()) {
            std::string::size_type eol = txt.find('\n', pos);
            std::string::size_type line_end = (eol == std::string::npos) ? txt.size() : eol;
            std::string::size_type i = pos;
            // 行首可空白
            while (i < line_end && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
            // 匹配 "total"
            static const std::string token = "total";
            if (i + token.size() <= line_end && txt.compare(i, token.size(), token) == 0) {
                std::string::size_type j = i + token.size();
                // token 后须空白
                if (j < line_end && std::isspace(static_cast<unsigned char>(txt[j]))) {
                    while (j < line_end && std::isspace(static_cast<unsigned char>(txt[j]))) ++j;
                    if (j < line_end && std::isdigit(static_cast<unsigned char>(txt[j]))) {
                        long v = std::strtol(txt.c_str() + j, nullptr, 10);
                        if (v >= 0) used = static_cast<int32_t>(v);
                    }
                }
            }
            if (eol == std::string::npos) break;
            pos = eol + 1;
        }
    }
    return std::make_tuple(used, total);
}

// ════════════════════════════════════════════════════════════════════════════
// Platform 类实现
// ════════════════════════════════════════════════════════════════════════════

Platform::Platform() {
    // ── SDK 芯片 / 库查询（自由函数，无参、不依赖模型）──
    _soc_name = cauchykesai::soc_name();
    _dnn_version = cauchykesai::dnn_version();
    _bpu_version = cauchykesai::bpu_version();

    // ── sysfs / proc（构造时读一次缓存）──
    _bpu_type = _infer_arch(_soc_name);
    _physical_core_num = physical_core_count();
    std::tie(_cpu_model, _cpu_count) = _read_cpuinfo();
    _mem_total_mb = _read_mem_total_mb();
}

// ── 全局 Platform 单例（import 即构造；CauchyKesai 与模块级 platform 共用同一实例）──
// magic static（C++11）线程安全；Platform 方法均 const，多 CauchyKesai 只读共享无竞争。
Platform& global_platform() {
    static Platform inst;
    return inst;
}

// ── 静态属性推断 / 读取 ────────────────────────────────────────────

std::string Platform::_infer_arch(const std::string &name) {
    if (name.empty()) return std::string();
    // 长串优先，避免 S100P 被 S100 截胡。
    if (name.find("S100P") != std::string::npos) return "nash-m";
    if (name.find("S600")  != std::string::npos) return "nash-p";
    if (name.find("S100")  != std::string::npos) return "nash-e";
    return std::string();
}

std::tuple<std::string, int32_t> Platform::_read_cpuinfo() {
    std::string txt = read_text_file(PROC_CPUINFO);
    if (txt.empty()) return std::make_tuple(std::string(), 0);

    // model name：首个 "model name" 行冒号后的内容（strip）
    std::string model;
    {
        std::string key = "model name";
        std::string::size_type pos = 0;
        while (pos < txt.size()) {
            std::string::size_type eol = txt.find('\n', pos);
            std::string::size_type line_end = (eol == std::string::npos) ? txt.size() : eol;
            // 行首可空白
            std::string::size_type i = pos;
            while (i < line_end && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
            if (i + key.size() <= line_end && txt.compare(i, key.size(), key) == 0) {
                std::string::size_type j = i + key.size();
                // 跳过空白和冒号
                while (j < line_end && (txt[j] == ':' || std::isspace(static_cast<unsigned char>(txt[j])))) ++j;
                std::string val = txt.substr(j, line_end - j);
                // strip 尾部空白
                while (!val.empty() && std::isspace(static_cast<unsigned char>(val.back()))) val.pop_back();
                if (!val.empty()) { model = val; break; }
            }
            if (eol == std::string::npos) break;
            pos = eol + 1;
        }
    }

    // count：数 "processor:" 行（行首空白 + processor + 空白/冒号）
    int32_t count = 0;
    {
        std::string key = "processor";
        std::string::size_type pos = 0;
        while (pos < txt.size()) {
            std::string::size_type eol = txt.find('\n', pos);
            std::string::size_type line_end = (eol == std::string::npos) ? txt.size() : eol;
            std::string::size_type i = pos;
            while (i < line_end && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
            if (i + key.size() <= line_end && txt.compare(i, key.size(), key) == 0) {
                std::string::size_type j = i + key.size();
                if (j < line_end && (txt[j] == ':' || std::isspace(static_cast<unsigned char>(txt[j]))))
                    ++count;
            }
            if (eol == std::string::npos) break;
            pos = eol + 1;
        }
    }
    if (count == 0) {
        auto ids = cpu_ids();
        count = static_cast<int32_t>(ids.size());
    }
    return std::make_tuple(model, count);
}

int32_t Platform::_read_mem_total_mb() {
    std::string txt = read_text_file(PROC_MEMINFO);
    if (txt.empty()) return -1;
    std::string key = "MemTotal:";
    std::string::size_type p = txt.find(key);
    if (p == std::string::npos) return -1;
    std::string::size_type i = p + key.size();
    while (i < txt.size() && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
    if (i >= txt.size() || !std::isdigit(static_cast<unsigned char>(txt[i]))) return -1;
    long v = std::strtol(txt.c_str() + i, nullptr, 10);
    if (v < 0) return -1;
    return static_cast<int32_t>(v / 1024);  // kB → MB
}

int32_t Platform::_read_mem_available_mb() {
    std::string txt = read_text_file(PROC_MEMINFO);
    if (txt.empty()) return -1;
    std::string key = "MemAvailable:";
    std::string::size_type p = txt.find(key);
    if (p == std::string::npos) return -1;
    std::string::size_type i = p + key.size();
    while (i < txt.size() && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
    if (i >= txt.size() || !std::isdigit(static_cast<unsigned char>(txt[i]))) return -1;
    long v = std::strtol(txt.c_str() + i, nullptr, 10);
    if (v < 0) return -1;
    return static_cast<int32_t>(v / 1024);  // kB → MB
}

// ── 实时方法（每次现读 sysfs）──────────────────────────────────────

py::list Platform::bpu_rate() const {
    py::list out;
    auto ids = bpu_core_ids();
    if (ids.empty()) return out;
    int32_t last = ids.back();
    // 长度 = max核号 + 1（核号连续索引，中间补 0）
    std::vector<int32_t> rates(static_cast<size_t>(last) + 1, 0);
    for (int32_t cid : ids) {
        int32_t r = read_int_file(
            std::string(BPU_SYSFS) + "/bpu" + std::to_string(cid) + "/ratio", 1000000);
        if (r >= 0) rates[static_cast<size_t>(cid)] = r;  // 单核读不到按 0
    }
    for (int32_t v : rates) out.append(v);
    return out;
}

py::list Platform::bpu_freq() const {
    py::list out;
    auto ids = bpu_core_ids();
    if (ids.empty()) return out;
    int32_t last = ids.back();
    std::vector<int32_t> freqs(static_cast<size_t>(last) + 1, 0);
    for (int32_t cid : ids) {
        std::string devfreq_dir = std::string(BPU_SYSFS) + "/bpu" + std::to_string(cid) + "/devfreq";
        DIR *d = ::opendir(devfreq_dir.c_str());
        if (!d) continue;
        struct dirent *e;
        while ((e = ::readdir(d)) != nullptr) {
            std::string sub(e->d_name);
            if (sub == "." || sub == "..") continue;
            int32_t f = read_int_file(devfreq_dir + "/" + sub + "/cur_freq", 1000000000);
            if (f > 0) {
                freqs[static_cast<size_t>(cid)] = f / 1000000;  // Hz → MHz
                break;
            }
        }
        ::closedir(d);
    }
    for (int32_t v : freqs) out.append(v);
    return out;
}

py::list Platform::cpu_freq() const {
    py::list out;
    auto ids = cpu_ids();
    if (ids.empty()) return out;
    int32_t last = ids.back();
    std::vector<int32_t> freqs(static_cast<size_t>(last) + 1, 0);
    for (int32_t cid : ids) {
        int32_t f = read_int_file(
            std::string(CPU_SYSFS) + "/cpu" + std::to_string(cid) + "/cpufreq/scaling_cur_freq",
            1000000000);
        if (f > 0) freqs[static_cast<size_t>(cid)] = f / 1000;  // kHz → MHz
    }
    for (int32_t v : freqs) out.append(v);
    return out;
}

// 遍历 thermal_zone*/{type,temp}。name 命中只读那一个 temp 并立即返回。
py::dict Platform::temperature(const std::string &name) const {
    py::dict out;
    DIR *d = ::opendir(THERMAL_SYSFS);
    if (!d) {
        if (!name.empty()) out[name.c_str()] = py::int_(-1);
        return out;
    }
    // 收集并排序 thermal_zoneN
    std::vector<int32_t> zone_ids;
    struct dirent *e;
    while ((e = ::readdir(d)) != nullptr) {
        std::string zn(e->d_name);
        static const std::string prefix = "thermal_zone";
        if (zn.size() > prefix.size() && zn.compare(0, prefix.size(), prefix) == 0) {
            bool digits = true;
            for (size_t k = prefix.size(); k < zn.size(); ++k)
                if (!std::isdigit(static_cast<unsigned char>(zn[k]))) { digits = false; break; }
            if (digits) zone_ids.push_back(static_cast<int32_t>(std::strtol(zn.c_str() + prefix.size(), nullptr, 10)));
        }
    }
    ::closedir(d);
    std::sort(zone_ids.begin(), zone_ids.end());

    auto read_type = [](const std::string &zpath) -> std::string {
        std::string t = read_text_file(zpath + "/type");
        // strip
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
        size_t start = 0;
        while (start < t.size() && std::isspace(static_cast<unsigned char>(t[start]))) ++start;
        return t.substr(start);
    };

    if (!name.empty()) {
        for (int32_t zid : zone_ids) {
            std::string zpath = std::string(THERMAL_SYSFS) + "/thermal_zone" + std::to_string(zid);
            std::string ttype = read_type(zpath);
            if (ttype == name) {
                int32_t temp = read_int_file(zpath + "/temp", 2000000);
                if (temp > 0)
                    out[name.c_str()] = py::float_(temp / 1000.0);  // 毫度 → °C
                else
                    out[name.c_str()] = py::int_(-1);
                return out;
            }
        }
        out[name.c_str()] = py::int_(-1);  // 未命中
        return out;
    }

    // name 空：返回全部（仅 temp > 0 纳入）
    for (int32_t zid : zone_ids) {
        std::string zpath = std::string(THERMAL_SYSFS) + "/thermal_zone" + std::to_string(zid);
        std::string ttype = read_type(zpath);
        if (ttype.empty()) continue;
        int32_t temp = read_int_file(zpath + "/temp", 2000000);
        if (temp > 0)
            out[ttype.c_str()] = py::float_(temp / 1000.0);
    }
    return out;
}

// 遍历 hwmon*/in*_label + in*_input。跳过 label == "NULL"。
py::dict Platform::voltage(const std::string &name) const {
    py::dict out;
    DIR *d = ::opendir(HWMON_SYSFS);
    if (!d) {
        if (!name.empty()) out[name.c_str()] = py::int_(-1);
        return out;
    }
    std::vector<std::string> hwmon_dirs;
    struct dirent *e;
    while ((e = ::readdir(d)) != nullptr) {
        std::string hn(e->d_name);
        static const std::string prefix = "hwmon";
        if (hn.size() > prefix.size() && hn.compare(0, prefix.size(), prefix) == 0) {
            bool digits = true;
            for (size_t k = prefix.size(); k < hn.size(); ++k)
                if (!std::isdigit(static_cast<unsigned char>(hn[k]))) { digits = false; break; }
            if (digits) hwmon_dirs.push_back(std::string(HWMON_SYSFS) + "/" + hn);
        }
    }
    ::closedir(d);
    std::sort(hwmon_dirs.begin(), hwmon_dirs.end());

    // 从文件名 in<N>_label 提取 N
    auto parse_label_idx = [](const std::string &fname, int32_t &idx) -> bool {
        static const std::string p = "in";
        if (fname.compare(0, p.size(), p) != 0) return false;
        std::string rest = fname.substr(p.size());
        std::string label_suffix = "_label";
        if (rest.size() <= label_suffix.size()) return false;
        if (rest.compare(rest.size() - label_suffix.size(), label_suffix.size(), label_suffix) != 0) return false;
        std::string num = rest.substr(0, rest.size() - label_suffix.size());
        if (num.empty()) return false;
        for (char c : num) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
        idx = static_cast<int32_t>(std::strtol(num.c_str(), nullptr, 10));
        return true;
    };

    auto read_label = [](const std::string &path) -> std::string {
        std::string t = read_text_file(path);
        while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
        size_t start = 0;
        while (start < t.size() && std::isspace(static_cast<unsigned char>(t[start]))) ++start;
        return t.substr(start);
    };

    if (!name.empty()) {
        for (const auto &hwpath : hwmon_dirs) {
            DIR *hd = ::opendir(hwpath.c_str());
            if (!hd) continue;
            struct dirent *he;
            while ((he = ::readdir(hd)) != nullptr) {
                std::string fname(he->d_name);
                int32_t idx;
                if (!parse_label_idx(fname, idx)) continue;
                std::string label = read_label(hwpath + "/" + fname);
                if (label == name) {
                    int32_t val = read_int_file(hwpath + "/in" + std::to_string(idx) + "_input", 1000000);
                    out[name.c_str()] = (val >= 0) ? py::int_(val) : py::int_(-1);
                    ::closedir(hd);
                    return out;
                }
            }
            ::closedir(hd);
        }
        out[name.c_str()] = py::int_(-1);
        return out;
    }

    for (const auto &hwpath : hwmon_dirs) {
        DIR *hd = ::opendir(hwpath.c_str());
        if (!hd) continue;
        struct dirent *he;
        while ((he = ::readdir(hd)) != nullptr) {
            std::string fname(he->d_name);
            int32_t idx;
            if (!parse_label_idx(fname, idx)) continue;
            std::string label = read_label(hwpath + "/" + fname);
            if (label.empty() || label == "NULL") continue;
            int32_t val = read_int_file(hwpath + "/in" + std::to_string(idx) + "_input", 1000000);
            if (val >= 0) out[label.c_str()] = py::int_(val);
        }
        ::closedir(hd);
    }
    return out;
}

// 三块 ION heap 实时占用（debugfs）。name 命中已知 heap 直读那一个。
py::dict Platform::ion_memory(const std::string &name) const {
    py::dict out;
    if (!name.empty()) {
        // 命中已知 heap：返 {name: {"used","total"}}；读不到 used/total 为 -1
        bool known = false;
        for (const auto &h : ION_HEAP_NAMES) if (h == name) { known = true; break; }
        if (!known) {
            out[name.c_str()] = py::int_(-1);  // 非已知 heap → int -1（非 dict）
            return out;
        }
        int32_t used, total;
        std::tie(used, total) = ion_heap_used_total(name);
        py::dict sub;
        sub["used"] = py::int_(used);
        sub["total"] = py::int_(total);
        out[name.c_str()] = sub;
        return out;
    }
    // name 空：返回三块
    for (const auto &h : ION_HEAP_NAMES) {
        int32_t used, total;
        std::tie(used, total) = ion_heap_used_total(h);
        py::dict sub;
        sub["used"] = py::int_(used);
        sub["total"] = py::int_(total);
        out[h.c_str()] = sub;
    }
    return out;
}

// 扫 /proc/self/maps，返回首个 basename 以 libhbucp 开头的库绝对路径。
// maps 行格式：addr perms offset dev inode pathname；手动跳过前 5 段空白分隔字段
// 后取余下（兼容路径含空格，不能用 istringstream >> 按空格截断）。
std::string Platform::ucp_library_path() const {
    std::string txt = read_text_file(PROC_MAPS);
    if (txt.empty()) return std::string();
    std::string::size_type pos = 0;
    while (pos < txt.size()) {
        std::string::size_type eol = txt.find('\n', pos);
        std::string::size_type line_end = (eol == std::string::npos) ? txt.size() : eol;
        // 跳过前 5 段（每段 = 连续非空白 + 跳过空白）
        std::string::size_type i = pos;
        int fields = 0;
        for (; i < line_end && fields < 5; ++fields) {
            while (i < line_end && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;  // 跳前导空白
            while (i < line_end && !std::isspace(static_cast<unsigned char>(txt[i]))) ++i;  // 跳字段
        }
        // 第 6 段：跳前导空白，取到行尾（保留路径含空格）
        while (i < line_end && std::isspace(static_cast<unsigned char>(txt[i]))) ++i;
        if (i < line_end) {
            std::string path = txt.substr(i, line_end - i);
            // basename
            std::string::size_type slash = path.find_last_of('/');
            std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
            static const std::string prefix = "libhbucp";
            if (base.size() >= prefix.size() && base.compare(0, prefix.size(), prefix) == 0)
                return path;
        }
        if (eol == std::string::npos) break;
        pos = eol + 1;
    }
    return std::string();
}

// ── summary / repr ──────────────────────────────────────────────────

py::dict Platform::summary() const {
    py::dict out;
    // ── 静态（构造时缓存）──
    out["soc_name"] = _soc_name;
    out["bpu_type"] = _bpu_type;
    out["dnn_version"] = _dnn_version;
    out["bpu_version"] = _bpu_version;
    out["physical_core_num"] = _physical_core_num;
    out["cpu_model"] = _cpu_model;
    out["cpu_count"] = _cpu_count;
    out["mem_total_mb"] = _mem_total_mb;
    // ── 实时（每次现读 sysfs / proc）──
    out["bpu_rate"] = bpu_rate();
    out["bpu_freq"] = bpu_freq();
    out["cpu_freq"] = cpu_freq();
    out["temperature"] = temperature();
    out["voltage"] = voltage();
    out["ion_memory"] = ion_memory();
    out["mem_available_mb"] = _read_mem_available_mb();
    out["ucp_library_path"] = ucp_library_path();
    return out;
}

std::string Platform::repr() const {
    return "Platform(soc=" + py::repr(py::str(_soc_name)).cast<std::string>() +
           ", bpu_type=" + py::repr(py::str(_bpu_type)).cast<std::string>() +
           ", bpu_cores=" + std::to_string(_physical_core_num) +
           ", cpu=" + py::repr(py::str(_cpu_model)).cast<std::string>() +
           " x" + std::to_string(_cpu_count) +
           ", mem=" + std::to_string(_mem_total_mb) + "MB)";
}

// s()：内部 print(render_platform(summary()))。展示逻辑（_render）留 Python，
// C++ 仅一行转发——避免重写 370 行渲染 + 不破 test_render 独立覆盖。
void Platform::s() const {
    py::module_ rmod = py::module_::import("pyCauchyKesai._render");
    py::object rendered = rmod.attr("render_platform")(this->summary());
    py::print(rendered);
}

// ════════════════════════════════════════════════════════════════════════════
// read_bpu_fw_version（模块级，CauchyKesai.summary 注入）
// ════════════════════════════════════════════════════════════════════════════

std::string read_bpu_fw_version() {
    std::string v = read_text_file(std::string(BPU_SYSFS) + "/bpu0/fw_version");
    // strip
    while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back();
    size_t start = 0;
    while (start < v.size() && std::isspace(static_cast<unsigned char>(v[start]))) ++start;
    return v.substr(start);
}

}  // namespace cauchykesai
