#include "cauchykesai/ion_array_desc.h"

#include <stdexcept>

namespace py = pybind11;

// ══════════════════════════════════════════════════════════════════════════════
// _resolve_stride_and_aligned_size: 唯一 stride 计算处
//   无条件补齐 stride 数组中的 -1，再按需计算 alignedByteSize。
//   bpu_align: 对齐值（运行时检测：S100/S100P=32, S600=64）
// ══════════════════════════════════════════════════════════════════════════════

void IONArrayDesc::_resolve_stride_and_aligned_size(hbDNNTensorProperties &props, int bpu_align)
{
    for (int32_t dim_i = props.validShape.numDimensions - 1; dim_i >= 0; --dim_i)
    {
        if (props.stride[dim_i] == -1)
        {
            if (dim_i + 1 >= props.validShape.numDimensions) continue;
            auto cur_stride = props.stride[dim_i + 1] * props.validShape.dimensionSize[dim_i + 1];
            props.stride[dim_i] = ((cur_stride + bpu_align - 1) / bpu_align) * bpu_align;
        }
    }
    if (props.alignedByteSize < 0)
        props.alignedByteSize = props.stride[0] * props.validShape.dimensionSize[0];
}

// ══════════════════════════════════════════════════════════════════════════════
// _dtype_from_tensor_type: hbDNNDataType → py::dtype 推断
// ══════════════════════════════════════════════════════════════════════════════

py::dtype IONArrayDesc::_dtype_from_tensor_type(int32_t tensor_type)
{
    if (tensor_type == HB_DNN_TENSOR_TYPE_S8)       return py::dtype::of<int8_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_U8)  return py::dtype::of<uint8_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_F16) return py::dtype("float16");
    else if (tensor_type == HB_DNN_TENSOR_TYPE_S16) return py::dtype::of<int16_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_U16) return py::dtype::of<uint16_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_F32) return py::dtype::of<float>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_S32) return py::dtype::of<int32_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_U32) return py::dtype::of<uint32_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_F64) return py::dtype::of<double>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_S64) return py::dtype::of<int64_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_U64) return py::dtype::of<uint64_t>();
    else if (tensor_type == HB_DNN_TENSOR_TYPE_BOOL8) return py::dtype::of<bool>();
    else return py::dtype::of<uint8_t>();  // fallback
}

// ══════════════════════════════════════════════════════════════════════════════
// _validate_shape: 维度必须 > 0
// ══════════════════════════════════════════════════════════════════════════════

void IONArrayDesc::_validate_shape() const
{
    for (auto d : shape) {
        if (d <= 0)
            throw std::invalid_argument("tensor shape dimensions must be > 0, got " + std::to_string(d));
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 构造函数
// ══════════════════════════════════════════════════════════════════════════════

// ── 基本构造：dtype + shape（参数改名 dt/shp，避免和成员同名）──
IONArrayDesc::IONArrayDesc(const py::dtype &dt, const std::vector<ssize_t> &shp)
    : dtype(dt), shape(shp)
{
    _validate_shape();
}

// ── 完整构造：从 hbDNNTensorProperties resolve ──
IONArrayDesc::IONArrayDesc(const hbDNNTensorProperties &props, int bpu_align)
    : name(),
      tensor_type(props.tensorType),
      quanti_type(props.quantiType),
      quantize_axis(props.quantizeAxis),
      scale(), zero_point()
{
    // ── 拷贝 validShape → shape ──
    int32_t ndim = props.validShape.numDimensions;
    shape.resize(ndim);
    for (int32_t i = 0; i < ndim; i++)
        shape[i] = static_cast<ssize_t>(props.validShape.dimensionSize[i]);
    _validate_shape();

    // ── 拷贝 stride ──
    stride.resize(ndim);
    for (int32_t i = 0; i < ndim; i++)
        stride[i] = props.stride[i];

    // ── 拷贝量化参数 ──
    if (props.scale.scaleLen > 0) {
        scale.resize(props.scale.scaleLen);
        for (int32_t i = 0; i < props.scale.scaleLen; i++)
            scale[i] = props.scale.scaleData[i];
    }
    if (props.scale.zeroPointLen > 0) {
        zero_point.resize(props.scale.zeroPointLen);
        for (int32_t i = 0; i < props.scale.zeroPointLen; i++)
            zero_point[i] = props.scale.zeroPointData[i];
    }

    // ── 推断 dtype（从 tensorType，需 GIL）──
    dtype = _dtype_from_tensor_type(props.tensorType);

    // ── resolve stride==-1 和 alignedByteSize==-1 ──
    hbDNNTensorProperties resolved_props = props;
    for (int32_t i = 0; i < ndim; i++)
        resolved_props.stride[i] = stride[i];
    _resolve_stride_and_aligned_size(resolved_props, bpu_align);
    for (int32_t i = 0; i < ndim; i++)
        stride[i] = resolved_props.stride[i];
    aligned_byte_size = resolved_props.alignedByteSize;
}

// ══════════════════════════════════════════════════════════════════════════════
// is_padded_layout: stride[0] != ∏内维 × itemsize → 有 padding（对齐布局）
// ══════════════════════════════════════════════════════════════════════════════

bool IONArrayDesc::is_padded_layout() const
{
    if (!has_stride() || shape.empty()) return false;
    int64_t natural = static_cast<int64_t>(dtype.itemsize());
    for (size_t i = 1; i < shape.size(); i++)
        natural *= static_cast<int64_t>(shape[i]);
    return stride[0] != natural;
}

// ══════════════════════════════════════════════════════════════════════════════
// fill_properties: 把自身性质写回 hbDNNTensorProperties（供 hbDNNInferV2）
//   scale/zeroPoint 指针指向本对象内部 vector——本对象生命周期须覆盖推理。
// ══════════════════════════════════════════════════════════════════════════════

void IONArrayDesc::fill_properties(hbDNNTensorProperties &props) const
{
    int32_t ndim_v = static_cast<int32_t>(shape.size());
    props.validShape.numDimensions = ndim_v;
    for (int32_t i = 0; i < ndim_v; i++)
        props.validShape.dimensionSize[i] = static_cast<int32_t>(shape[i]);

    props.tensorType      = static_cast<hbDNNDataType>(tensor_type);
    props.quantiType      = quanti_type;
    props.quantizeAxis    = quantize_axis;
    props.alignedByteSize = aligned_byte_size;

    for (size_t i = 0; i < stride.size() && i < HB_DNN_TENSOR_MAX_DIMENSIONS; i++)
        props.stride[i] = stride[i];

    props.scale.scaleLen      = static_cast<int32_t>(scale.size());
    props.scale.scaleData     = scale.empty() ? nullptr : const_cast<float*>(scale.data());
    props.scale.zeroPointLen  = static_cast<int32_t>(zero_point.size());
    props.scale.zeroPointData = zero_point.empty() ? nullptr : const_cast<int32_t*>(zero_point.data());
}
