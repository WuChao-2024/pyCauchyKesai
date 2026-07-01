#ifndef ION_ARRAY_H_
#define ION_ARRAY_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <fenv.h>
#include <algorithm>
#include "hobot/dnn/hb_dnn.h"
#include "hobot/hb_ucp_sys.h"

#include "cauchykesai/ion_memory.h"
#include "cauchykesai/ion_array_desc.h"

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// IONArray — 持有 IONArrayDesc（描述）+ IONMemory（内存），组合关系（has-a）
//
// desc 是 public 直接成员：ion.desc / ion->desc 直接访问，无 getter。
// 不转发 desc 的性质：dtype/shape 等走 ion.desc.dtype（描述的归 desc）。
// IONArray 自己的方法是内存操作：numpy / from_numpy / flush / allocate 等。
// ─────────────────────────────────────────────────────────────────────────────

class IONArray : public std::enable_shared_from_this<IONArray>
{
public:
    IONArrayDesc desc;   // 张量描述（public 直接成员，ion.desc 直接访问）

    // ── 构造（多重载）──
    IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
             bool cached = true, bool defer = false);
    IONArray(const IONArrayDesc &d, bool cached = true, bool defer = false);
    IONArray(const hbDNNTensorProperties &props, int bpu_align,
             bool cached = true, bool defer = false);

    ~IONArray();

    // 禁拷贝，允许移动
    IONArray(const IONArray &) = delete;
    IONArray &operator=(const IONArray &) = delete;
    IONArray(IONArray &&other) noexcept;
    IONArray &operator=(IONArray &&other) noexcept;

    // ── 工厂 ──
    static std::shared_ptr<IONArray> clone(const std::shared_ptr<IONArray> &src,
                                            bool cached = true, bool defer = true);
    static std::shared_ptr<IONArray> from_memory(
        std::shared_ptr<IONMemory> mem, uint64_t byte_offset, const IONArrayDesc &tpl);

    // ── 内存操作（IONArray 本职）──
    void allocate(bool cached = true);

    uint64_t phy_addr() const;
    void    *vir_addr() const;
    uint64_t mem_size() const;
    hbUCPSysMem sys_mem() const;
    hbDNNTensor dnn_tensor() const;   // 投影成 SDK hbDNNTensor（properties + sysMem），供 hbDNNInferV2
    bool is_cached()      const;
    bool is_allocated()   const;
    // memory 成员 public：ion.memory 直接访问

    void flush_clean() const;
    void flush_invalidate() const;

    py::array numpy();
    py::array dequantize();
    void quantize(const py::array &float_arr);
    void from_numpy(const py::array &arr);

    // 留 IONArray：要用 mem_size（内存）校验
    bool properties_match(const IONArray &tpl) const;

    // ── 内存成员（public 直访，无下划线）──
    std::shared_ptr<IONMemory>  memory;           // ION 物理内存（shared_ptr 共享所有权）
    uint64_t                    byte_offset = 0;  // 在 IONMemory 内的字节偏移
};

#endif  // ION_ARRAY_H_
