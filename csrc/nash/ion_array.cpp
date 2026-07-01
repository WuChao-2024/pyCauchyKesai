#include "cauchykesai/ion_array.h"
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ══════════════════════════════════════════════════════════════════════════════
// 构造函数：性质全进 desc 成员；IONArray 只管 memory/byte_offset
// ══════════════════════════════════════════════════════════════════════════════

// ── 裸构造 dtype + shape ──
IONArray::IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                   bool cached, bool defer)
    : desc(dtype, shape), memory(), byte_offset(0)
{
    if (!defer) allocate(cached);
}

// ── 从 IONArrayDesc 构造（完整性质）──
IONArray::IONArray(const IONArrayDesc &d, bool cached, bool defer)
    : desc(d), memory(), byte_offset(0)
{
    if (!defer) allocate(cached);
}

// ── 从 hbDNNTensorProperties 构造（内部，make_input 模板用）──
IONArray::IONArray(const hbDNNTensorProperties &props, int bpu_align,
                   bool cached, bool defer)
    : desc(props, bpu_align), memory(), byte_offset(0)
{
    if (!defer) {
        memory = std::make_shared<IONMemory>(
            static_cast<uint64_t>(desc.aligned_byte_size), cached);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// from_memory 工厂: 共享 IONMemory + 偏移 + 从描述继承性质
// ══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<IONArray> IONArray::from_memory(
    std::shared_ptr<IONMemory> mem, uint64_t byte_offset, const IONArrayDesc &tpl)
{
    if (!mem)
        throw std::invalid_argument("IONArray::from_memory: memory is nullptr");
    if (byte_offset > mem->size())
        throw std::out_of_range("IONArray::from_memory: byte_offset (" +
            std::to_string(byte_offset) + ") > memory size (" +
            std::to_string(mem->size()) + ")");

    // 从 tpl（desc）构造：性质继承，defer=true 不分配新内存
    auto ion = std::make_shared<IONArray>(tpl, /*cached=*/mem->is_cached(), /*defer=*/true);
    ion->memory      = mem;          // 共享同一 IONMemory
    ion->byte_offset = byte_offset;
    return ion;
}

// ══════════════════════════════════════════════════════════════════════════════
// 克隆工厂: 独立分配（不共享 src 的 memory）
// ══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<IONArray> IONArray::clone(const std::shared_ptr<IONArray> &src,
                                            bool cached, bool defer)
{
    if (!src)
        throw std::invalid_argument("IONArray::clone: source is nullptr");

    // 从 src->desc 构造：性质继承，defer=true 先不分配
    auto ion = std::make_shared<IONArray>(src->desc, cached, /*defer=*/true);
    if (!defer) ion->allocate(cached);  // 独立分配新 IONMemory
    return ion;
}

// ══════════════════════════════════════════════════════════════════════════════
// 析构 — shared_ptr 自动释放 IONMemory
// ══════════════════════════════════════════════════════════════════════════════

IONArray::~IONArray() {}

// ══════════════════════════════════════════════════════════════════════════════
// 移动语义
// ══════════════════════════════════════════════════════════════════════════════

IONArray::IONArray(IONArray &&other) noexcept
    : desc(std::move(other.desc)),
      memory(std::move(other.memory)),
      byte_offset(other.byte_offset)
{
    other.byte_offset = 0;
}

IONArray &IONArray::operator=(IONArray &&other) noexcept
{
    if (this != &other) {
        desc        = std::move(other.desc);
        memory     = std::move(other.memory);
        byte_offset = other.byte_offset;
        other.byte_offset = 0;
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════════
// 懒分配
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::allocate(bool cached)
{
    if (memory && memory->is_allocated())
        throw std::runtime_error("IONArray::allocate: already allocated");
    uint64_t n;
    if (desc.aligned_byte_size > 0)
        n = static_cast<uint64_t>(desc.aligned_byte_size);
    else {
        n = desc.dtype.itemsize();
        for (auto d : desc.shape) n *= static_cast<uint64_t>(d);
    }
    memory = std::make_shared<IONMemory>(n, cached);
}

// ══════════════════════════════════════════════════════════════════════════════
// 内存属性
// ══════════════════════════════════════════════════════════════════════════════

uint64_t IONArray::phy_addr() const
{
    return memory ? memory->phy_addr_at(byte_offset) : 0;
}

void *IONArray::vir_addr() const
{
    return memory ? memory->vir_addr_at(byte_offset) : nullptr;
}

uint64_t IONArray::mem_size() const
{
    return memory ? (memory->size() - byte_offset) : 0;
}

// sys_mem() 返回偏移后 by-value 的 hbUCPSysMem（喂 BPU）
// 未分配时返回空 struct（不抛）——CauchyKesai 构造期对未分配 slot 调用后再分配刷新
hbUCPSysMem IONArray::sys_mem() const
{
    if (!memory)
        return hbUCPSysMem{};
    hbUCPSysMem m = memory->sys_mem();
    m.phyAddr += byte_offset;
    m.virAddr = static_cast<char*>(m.virAddr) + byte_offset;
    m.memSize = (desc.aligned_byte_size > 0)
                    ? static_cast<uint64_t>(desc.aligned_byte_size)
                    : static_cast<uint64_t>(desc.nbytes());
    return m;
}

// ══════════════════════════════════════════════════════════════════════════════
// dnn_tensor: 投影成 SDK hbDNNTensor（properties + sysMem）
//   hbDNNTensor = {hbDNNTensorProperties, hbUCPSysMem} = desc 的 SDK 表示 + memory+offset 的 SDK 表示
//   即 IONArray 的纯投影，无额外信息。供 hbDNNInferV2 现建现用。
//   注意：properties.scale/zeroPoint 指针指向本 IONArray 的 desc 内部 vector——
//   本 IONArray（bound）生命周期须覆盖推理。
// ══════════════════════════════════════════════════════════════════════════════

hbDNNTensor IONArray::dnn_tensor() const
{
    hbDNNTensor t;
    desc.fill_properties(t.properties);
    t.sysMem = sys_mem();
    return t;
}

bool IONArray::is_cached() const { return memory && memory->is_cached(); }
bool IONArray::is_allocated() const { return memory && memory->is_allocated(); }

// ══════════════════════════════════════════════════════════════════════════════
// Cache 操作 — 委托 IONMemory
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::flush_clean() const { if (memory) memory->flush_clean(); }
void IONArray::flush_invalidate() const { if (memory) memory->flush_invalidate(); }

// ══════════════════════════════════════════════════════════════════════════════
// numpy: 布局感知（有 stride 时返回 strided view）
// ══════════════════════════════════════════════════════════════════════════════

py::array IONArray::numpy()
{
    if (!is_allocated())
        throw std::runtime_error("IONArray: not allocated, call allocate() first");

    // 越界校验：offset + 占用 <= mem.size（防 offset 规划错读到邻居段/块外）
    uint64_t need = (desc.aligned_byte_size > 0)
                        ? static_cast<uint64_t>(desc.aligned_byte_size)
                        : static_cast<uint64_t>(desc.nbytes());
    if (byte_offset + need > memory->size())
        throw std::out_of_range("IONArray::numpy: view (offset " +
            std::to_string(byte_offset) + " + need " + std::to_string(need) +
            ") exceeds memory size (" + std::to_string(memory->size()) + ")");

    // capsule 持 shared_ptr<IONArray> → ION 内存不提前释放
    auto self_ptr = shared_from_this();
    auto capsule  = py::capsule(
        new std::shared_ptr<IONArray>(self_ptr),
        [](void *p) { delete static_cast<std::shared_ptr<IONArray>*>(p); });

    void *data_ptr = memory->vir_addr_at(byte_offset);

    if (desc.has_stride()) {
        const auto &st = desc.stride;
        std::vector<ssize_t> np_strides(st.size());
        for (size_t i = 0; i < st.size(); i++)
            np_strides[i] = static_cast<ssize_t>(st[i]);
        return py::array(desc.dtype, desc.shape, np_strides, data_ptr, capsule);
    } else {
        return py::array(desc.dtype, desc.shape, data_ptr, capsule);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// dequantize: buffer (int) → float numpy array
//   NONE 透传；SCALE: f=(x-zero_point)*scale，沿 quantize_axis 广播
// ══════════════════════════════════════════════════════════════════════════════

py::array IONArray::dequantize()
{
    if (!is_allocated())
        throw std::runtime_error("IONArray::dequantize: not allocated, call allocate() first");

    if (desc.quanti_type != SCALE)
        return numpy();

    py::array int_arr = numpy();
    const auto &shp = desc.shape;
    ssize_t total_elems = 1;
    for (auto d : shp) total_elems *= d;

    py::array out_arr(py::dtype::of<float>(), shp);
    auto out_buf = static_cast<float*>(out_arr.mutable_data());

    py::array c_int_arr = py::array::ensure(int_arr, py::array::c_style);
    auto int_buf = static_cast<const int8_t*>(c_int_arr.data());
    int32_t elem_size = desc.dtype.itemsize();
    int32_t tt = desc.tensor_type;

    auto read_elem = [&](ssize_t i) -> float {
        const char *p = reinterpret_cast<const char*>(int_buf) + static_cast<ptrdiff_t>(i) * elem_size;
        if (tt == HB_DNN_TENSOR_TYPE_U8)  { uint8_t v;  std::memcpy(&v, p, 1); return static_cast<float>(v); }
        else if (tt == HB_DNN_TENSOR_TYPE_U16) { uint16_t v; std::memcpy(&v, p, 2); return static_cast<float>(v); }
        else if (tt == HB_DNN_TENSOR_TYPE_U32) { uint32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
        else if (elem_size == 1) { int8_t v;  std::memcpy(&v, p, 1); return static_cast<float>(v); }
        else if (elem_size == 2) { int16_t v; std::memcpy(&v, p, 2); return static_cast<float>(v); }
        else                     { int32_t v; std::memcpy(&v, p, 4); return static_cast<float>(v); }
    };

    const auto &scale = desc.scale;
    const auto &zp = desc.zero_point;
    int32_t slen = static_cast<int32_t>(scale.size());
    int32_t zlen = static_cast<int32_t>(zp.size());

    if (slen == 0) return numpy();

    if (slen == 1 && (zlen == 0 || zlen == 1)) {
        float s = scale[0];
        float z = (zlen > 0) ? static_cast<float>(zp[0]) : 0.0f;
        for (ssize_t i = 0; i < total_elems; i++)
            out_buf[i] = (read_elem(i) - z) * s;
    } else {
        int32_t ndim_val = static_cast<int32_t>(shp.size());
        int32_t axis = desc.quantize_axis;
        std::vector<ssize_t> dim_prod(ndim_val);
        dim_prod[ndim_val - 1] = 1;
        for (int32_t d = ndim_val - 2; d >= 0; d--)
            dim_prod[d] = dim_prod[d + 1] * shp[d + 1];
        for (ssize_t i = 0; i < total_elems; i++) {
            float raw_val = read_elem(i);
            int32_t sidx;
            if (axis >= 0 && axis < ndim_val)
                sidx = static_cast<int32_t>((i / dim_prod[axis]) % shp[axis]);
            else
                sidx = 0;
            if (sidx >= slen) sidx = slen - 1;
            float s = scale[sidx];
            float z = (zlen > 0 && sidx < zlen) ? static_cast<float>(zp[sidx]) : 0.0f;
            out_buf[i] = (raw_val - z) * s;
        }
    }
    return out_arr;
}

// ══════════════════════════════════════════════════════════════════════════════
// quantize: float numpy array → 写入 buffer (int)
//   NONE memcpy；SCALE: int=round(float/scale)+zero_point，clip 到 dtype 范围
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::quantize(const py::array &float_arr)
{
    if (!is_allocated())
        throw std::runtime_error("IONArray::quantize: not allocated, call allocate() first");

    void *buf = memory->vir_addr_at(byte_offset);
    uint64_t buf_remaining = memory->size() - byte_offset;

    if (desc.quanti_type != SCALE) {
        py::array contiguous = py::array::ensure(float_arr, py::array::c_style);
        if (contiguous.nbytes() > static_cast<ssize_t>(buf_remaining))
            throw std::runtime_error("IONArray::quantize: input too large for buffer");
        std::memcpy(buf, contiguous.data(), contiguous.nbytes());
        return;
    }

    if (float_arr.ndim() != desc.ndim())
        throw std::invalid_argument("IONArray::quantize: ndim mismatch");
    const auto &shp = desc.shape;
    for (int i = 0; i < float_arr.ndim(); i++)
        if (float_arr.shape()[i] != shp[i])
            throw std::invalid_argument("IONArray::quantize: shape mismatch at dimension " + std::to_string(i));

    py::array c_float_arr = py::array::ensure(float_arr, py::array::c_style);
    auto float_buf = static_cast<const float*>(c_float_arr.data());

    ssize_t total_elems = 1;
    for (auto d : shp) total_elems *= d;

    int32_t elem_size = desc.dtype.itemsize();
    const auto &scale = desc.scale;
    const auto &zp = desc.zero_point;
    int32_t slen = static_cast<int32_t>(scale.size());
    int32_t zlen = static_cast<int32_t>(zp.size());

    int old_round = fegetround();
    fesetround(FE_TONEAREST);

    int32_t tt = desc.tensor_type;
    bool is_u8 = (tt == HB_DNN_TENSOR_TYPE_U8);
    bool is_s8 = (tt == HB_DNN_TENSOR_TYPE_S8);
    float clip_lo = is_u8 ? 0.0f   : (is_s8 ? -128.0f : -2147483648.0f);
    float clip_hi = is_u8 ? 255.0f : (is_s8 ? 127.0f  : 2147483647.0f);

    if (slen == 1 && (zlen == 0 || zlen == 1)) {
        float s = scale[0];
        float z = (zlen > 0) ? static_cast<float>(zp[0]) : 0.0f;
        for (ssize_t i = 0; i < total_elems; i++) {
            float val = (zlen > 0) ? std::nearbyint(float_buf[i] / s) + z
                                   : std::nearbyint(float_buf[i] / s);
            val = std::max(clip_lo, std::min(clip_hi, val));
            if (elem_size == 1) {
                if (is_u8) static_cast<uint8_t*>(buf)[i] = static_cast<uint8_t>(val);
                else       static_cast<int8_t*>(buf)[i]  = static_cast<int8_t>(val);
            } else if (elem_size == 2) {
                static_cast<int16_t*>(buf)[i] = static_cast<int16_t>(val);
            } else {
                static_cast<int32_t*>(buf)[i] = static_cast<int32_t>(val);
            }
        }
    } else {
        int32_t ndim_val = static_cast<int32_t>(shp.size());
        int32_t axis = desc.quantize_axis;
        std::vector<ssize_t> dim_prod(ndim_val);
        dim_prod[ndim_val - 1] = 1;
        for (int32_t d = ndim_val - 2; d >= 0; d--)
            dim_prod[d] = dim_prod[d + 1] * shp[d + 1];
        for (ssize_t i = 0; i < total_elems; i++) {
            int32_t sidx;
            if (axis >= 0 && axis < ndim_val)
                sidx = static_cast<int32_t>((i / dim_prod[axis]) % shp[axis]);
            else
                sidx = 0;
            if (sidx >= slen) sidx = slen - 1;
            float s = scale[sidx];
            float z = (zlen > 0 && sidx < zlen) ? static_cast<float>(zp[sidx]) : 0.0f;
            float val = (zlen > 0) ? std::nearbyint(float_buf[i] / s) + z
                                   : std::nearbyint(float_buf[i] / s);
            val = std::max(clip_lo, std::min(clip_hi, val));
            if (elem_size == 1) {
                if (is_u8) static_cast<uint8_t*>(buf)[i] = static_cast<uint8_t>(val);
                else       static_cast<int8_t*>(buf)[i]  = static_cast<int8_t>(val);
            } else if (elem_size == 2) {
                static_cast<int16_t*>(buf)[i] = static_cast<int16_t>(val);
            } else {
                static_cast<int32_t*>(buf)[i] = static_cast<int32_t>(val);
            }
        }
    }

    fesetround(old_round);
    flush_clean();
}

// ══════════════════════════════════════════════════════════════════════════════
// from_numpy: 连续 numpy → buffer（布局感知写入）
//   natural 布局 memcpy；padded 布局按 stride 散写。写后 flush。
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::from_numpy(const py::array &arr)
{
    if (!is_allocated())
        throw std::runtime_error("IONArray::from_numpy: not allocated, call allocate() first");

    py::array c = py::array::ensure(arr, py::array::c_style);
    if (c.ndim() != desc.ndim())
        throw std::invalid_argument("IONArray::from_numpy: ndim mismatch");
    const auto &shp = desc.shape;
    for (int i = 0; i < c.ndim(); i++)
        if (c.shape()[i] != shp[i])
            throw std::invalid_argument("IONArray::from_numpy: shape mismatch at dimension " + std::to_string(i));
    if (!desc.dtype.is(c.dtype()))
        throw std::invalid_argument("IONArray::from_numpy: dtype mismatch");

    int32_t es = desc.dtype.itemsize();
    ssize_t total_elems = 1;
    for (auto d : shp) total_elems *= d;

    char *dst = static_cast<char*>(memory->vir_addr_at(byte_offset));
    uint64_t buf_remaining = memory->size() - byte_offset;

    if (!desc.is_padded_layout()) {
        ssize_t nbytes = total_elems * es;
        if (static_cast<uint64_t>(nbytes) > buf_remaining)
            throw std::runtime_error("IONArray::from_numpy: input too large for buffer");
        std::memcpy(dst, c.data(), static_cast<size_t>(nbytes));
    } else {
        const auto &st = desc.stride;
        const char *src = static_cast<const char*>(c.data());
        int n = desc.ndim();
        std::vector<ssize_t> idx(n, 0);
        for (ssize_t linear = 0; linear < total_elems; linear++) {
            int64_t off = 0;
            for (int k = 0; k < n; k++) off += st[k] * static_cast<int64_t>(idx[k]);
            if (static_cast<uint64_t>(off + es) > buf_remaining)
                throw std::out_of_range("IONArray::from_numpy: strided write out of buffer bounds");
            std::memcpy(dst + off, src + linear * es, static_cast<size_t>(es));
            int k = n - 1;
            while (k >= 0) {
                if (++idx[k] < shp[k]) break;
                idx[k] = 0;
                k--;
            }
        }
    }

    flush_clean();
}

// ══════════════════════════════════════════════════════════════════════════════
// properties_match: 校验 this 与 template 性质兼容（用 desc + mem_size）
// ══════════════════════════════════════════════════════════════════════════════

bool IONArray::properties_match(const IONArray &tpl) const
{
    // dtype 必须匹配
    if (!desc.dtype.is(tpl.desc.dtype)) return false;

    // shape 必须匹配
    const auto &a = desc.shape;
    const auto &b = tpl.desc.shape;
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;

    // memSize 必须 >= aligned_byte_size（BPU 需要的缓冲区大小）
    uint64_t this_mem = mem_size();
    if (tpl.desc.aligned_byte_size > 0 &&
        this_mem < static_cast<uint64_t>(tpl.desc.aligned_byte_size))
        return false;

    // 若 this 有 tensor 性质，进一步校验
    if (desc.has_tensor_properties()) {
        if (desc.tensor_type != tpl.desc.tensor_type) return false;
        if (desc.quanti_type != tpl.desc.quanti_type) return false;
    }

    // 布局兼容：模板 padded 时 this 必须带相同 stride
    if (tpl.desc.is_padded_layout() && desc.stride != tpl.desc.stride)
        return false;

    return true;
}
