#ifndef ION_ARRAY_H_
#define ION_ARRAY_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>
#include "hobot/hb_ucp_sys.h"

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// IONArray — ION 内存管理器
//
// 纯 C++ 内存管理类，不继承 py::array。
// 职责:
//   1. 分配/持有 ION 物理内存 (hbUCPMalloc / hbUCPMallocCached)
//   2. 将物理地址绑定给 BPU (通过 sys_mem())
//   3. 通过 as_array() 导出 np.ndarray 视图给 CPU 用户
//   4. 通过 sub_view() 创建元素级偏移子视图
//
// 生命周期: 始终由 shared_ptr 管理 (make_shared 创建或 pybind11 持有)
// ─────────────────────────────────────────────────────────────────────────────

class IONArray : public std::enable_shared_from_this<IONArray>
{
public:
    // ── 按 dtype + shape 分配 ION 内存 ──
    // cached=true: 分配 cached ION 内存 (CPU cache 一致)
    // defer=false:  默认立即分配；true 时只记录 dtype+shape，稍后调 allocate()
    IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
             bool cached = true, bool defer = false);

    // ── 按 dtype + shape + 显式字节大小分配 (byte_size >= dtype×shape 的实际大小) ──
    IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
             uint64_t byte_size, bool cached = true);

    // ── Non-owning: 包装已有 hbUCPSysMem (供 sub_view 内部使用) ──
    IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
             const hbUCPSysMem &mem, bool cached = true);

    ~IONArray();

    // 禁止拷贝, 允许移动
    IONArray(const IONArray &) = delete;
    IONArray &operator=(const IONArray &) = delete;
    IONArray(IONArray &&other) noexcept;
    IONArray &operator=(IONArray &&other) noexcept;

    // ── 懒分配 ──
    // 按已存储的 dtype_ × shape_ 分配 ION 内存。
    // 重复调用抛 RuntimeError；通常配合 defer=true 构造使用。
    void allocate(bool cached = true);

    // ── 内存属性（一行 getter，保留 inline）──
    uint64_t phy_addr()        const { return mem_.phyAddr; }
    void    *vir_addr()        const { return mem_.virAddr; }
    uint64_t mem_size()        const { return mem_.memSize; }
    const hbUCPSysMem &sys_mem() const { return mem_; }
    bool is_cached()           const { return cached_; }
    bool is_allocated()        const { return owns_mem_ || mem_.virAddr != nullptr; }

    // ── 类型/形状 (numpy.ndarray 兼容) ──
    const py::dtype              &dtype()  const { return dtype_; }
    int                           ndim()   const { return static_cast<int>(shape_.size()); }
    const std::vector<ssize_t>   &shape()  const { return shape_; }
    ssize_t size() const {
        ssize_t s = 1;
        for (auto d : shape_) s *= d;
        return s;
    }
    ssize_t nbytes() const {
        // 始终返回理论值 (dtype.itemsize × product(shape))，与 numpy.ndarray.nbytes 语义一致
        // 实际 ION 分配大小通过 mem_size() 获取
        ssize_t n = dtype_.itemsize();
        for (auto d : shape_) n *= d;
        return n;
    }

    // ── Cache 操作 ──
    // 未分配时静默跳过 (mem_.virAddr == nullptr)
    void flush() const {
        if (cached_ && mem_.virAddr) hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_CLEAN);
    }
    void invalidate() const {
        if (cached_ && mem_.virAddr) hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_INVALIDATE);
    }

    // ── 导出 np.ndarray 视图 ──
    py::array as_array();

    // ── 元素级偏移子视图 ──
    IONArray sub_view(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                      ssize_t element_offset);

private:
    // sub_view 专用: 携带父引用
    IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
             const hbUCPSysMem &mem, std::shared_ptr<IONArray> parent);

    void _alloc(uint64_t byte_size, bool cached);

    hbUCPSysMem                   mem_{};
    bool                          owns_mem_ = false;
    bool                          cached_   = true;
    py::dtype                     dtype_;
    std::vector<ssize_t>          shape_;
    std::shared_ptr<IONArray>     parent_;   // 子视图持有父引用
};

#endif  // ION_ARRAY_H_
