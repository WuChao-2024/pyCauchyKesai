#ifndef ION_ARRAY_DESC_H_
#define ION_ARRAY_DESC_H_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>
#include <vector>

#include "hobot/dnn/hb_dnn.h"

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// IONArrayDesc — 纯张量描述（无内存），成员 public 直访
//
// 持有 tensor 全部性质（dtype/shape/stride/量化 等），无不变量，成员 public。
// 纯转发 getter 全删（成员直访）；计算方法（ndim/size/nbytes/is_padded_layout/...）
// 名字不和成员冲突，保留；fill_properties 写回 SDK struct。
// IONArray 持有它（has-a），访问走 ion.desc.dtype。
// ─────────────────────────────────────────────────────────────────────────────

class IONArrayDesc
{
public:
    // ── 构造 ──
    // 基本构造：dtype + shape（其余性质安全默认：stride 空、quanti NONE、
    // aligned_byte_size=-1、tensor_type=-1）。参数改名 dt/shp 避免和成员同名。
    IONArrayDesc(const py::dtype &dt, const std::vector<ssize_t> &shp);
    // 完整构造：从 hbDNNTensorProperties resolve（内部用，make_input 模板）
    IONArrayDesc(const hbDNNTensorProperties &props, int bpu_align);

    virtual ~IONArrayDesc() = default;
    IONArrayDesc(const IONArrayDesc &) = default;
    IONArrayDesc &operator=(const IONArrayDesc &) = default;
    IONArrayDesc(IONArrayDesc &&) noexcept = default;
    IONArrayDesc &operator=(IONArrayDesc &&) noexcept = default;

    // ── 计算方法（有逻辑，名字不和成员冲突，保留）──
    int ndim() const { return static_cast<int>(shape.size()); }
    ssize_t size() const {
        ssize_t s = 1;
        for (auto d : shape) s *= d;
        return s;
    }
    ssize_t nbytes() const {
        ssize_t n = dtype.itemsize();
        for (auto d : shape) n *= d;
        return n;
    }
    bool has_stride() const { return !stride.empty(); }
    // 是否 padded（对齐）布局：stride[0] != ∏内维 × itemsize
    bool is_padded_layout() const;
    bool has_tensor_properties() const { return tensor_type >= 0; }
    bool is_quantized() const { return quanti_type == SCALE; }

    // ── 写回 SDK struct（供 hbDNNInferV2 绑性质）──
    void fill_properties(hbDNNTensorProperties &props) const;

    // ── 性质成员（public 直访，无下划线）──
    py::dtype                    dtype;
    std::vector<ssize_t>         shape;
    std::string                  name;
    std::string                  desc;                  // hbDNNGetInputDesc/OutputDesc 文本
    int64_t                      aligned_byte_size = -1;
    int32_t                      tensor_type = -1;
    hbDNNQuantiType              quanti_type = NONE;
    int32_t                      quantize_axis = 0;
    std::vector<int64_t>         stride;
    std::vector<float>           scale;
    std::vector<int32_t>         zero_point;

protected:
    void _validate_shape() const;

private:
    static void _resolve_stride_and_aligned_size(hbDNNTensorProperties &props, int bpu_align);
    static py::dtype _dtype_from_tensor_type(int32_t tensor_type);
};

#endif  // ION_ARRAY_DESC_H_
