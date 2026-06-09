#include "cauchykesai/ion_array.h"
#include <stdexcept>

namespace py = pybind11;

// ══════════════════════════════════════════════════════════════════════════════
// 内部辅助
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::_alloc(uint64_t byte_size, bool cached)
{
    int32_t ret;
    if (cached)
        ret = hbUCPMallocCached(&mem_, byte_size, 0);
    else
        ret = hbUCPMalloc(&mem_, byte_size, 0);
    if (ret != 0)
        throw std::runtime_error(std::string("IONArray: ") +
            (cached ? "hbUCPMallocCached" : "hbUCPMalloc") + " failed");
    owns_mem_ = true;
    cached_   = cached;
}

// ══════════════════════════════════════════════════════════════════════════════
// 构造函数
// ══════════════════════════════════════════════════════════════════════════════

// ── 按 dtype + shape 分配 ──
IONArray::IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                   bool cached, bool defer)
    : mem_{}, owns_mem_(false), cached_(cached), dtype_(dtype), shape_(shape)
{
    for (auto d : shape_) {
        if (d <= 0)
            throw std::invalid_argument("IONArray: shape dimensions must be > 0, got " + std::to_string(d));
    }
    if (!defer) allocate(cached);
}

// ── 按 dtype + shape + 显式字节大小分配 ──
IONArray::IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                   uint64_t byte_size, bool cached)
    : mem_{}, owns_mem_(false), cached_(cached), dtype_(dtype), shape_(shape)
{
    for (auto d : shape_) {
        if (d <= 0)
            throw std::invalid_argument("IONArray: shape dimensions must be > 0, got " + std::to_string(d));
    }
    _alloc(byte_size, cached);
}

// ── Non-owning: 包装已有 hbUCPSysMem ──
IONArray::IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                   const hbUCPSysMem &mem, bool cached)
    : mem_(mem), owns_mem_(false), cached_(cached), dtype_(dtype), shape_(shape)
{}

// ── sub_view 内部: 携带父引用 ──
IONArray::IONArray(const py::dtype &dtype, const std::vector<ssize_t> &shape,
                   const hbUCPSysMem &mem, std::shared_ptr<IONArray> parent)
    : mem_(mem), owns_mem_(false),
      cached_(parent ? parent->cached_ : true),
      dtype_(dtype), shape_(shape),
      parent_(std::move(parent))
{}

// ══════════════════════════════════════════════════════════════════════════════
// 析构
// ══════════════════════════════════════════════════════════════════════════════

IONArray::~IONArray()
{
    if (owns_mem_ && mem_.virAddr != nullptr)
        hbUCPFree(&mem_);
}

// ══════════════════════════════════════════════════════════════════════════════
// 移动语义
// ══════════════════════════════════════════════════════════════════════════════

IONArray::IONArray(IONArray &&other) noexcept
    : mem_(other.mem_),
      owns_mem_(other.owns_mem_),
      cached_(other.cached_),
      dtype_(std::move(other.dtype_)),
      shape_(std::move(other.shape_)),
      parent_(std::move(other.parent_))
{
    other.owns_mem_ = false;
    other.mem_      = {};
}

IONArray &IONArray::operator=(IONArray &&other) noexcept
{
    if (this != &other) {
        if (owns_mem_ && mem_.virAddr != nullptr) hbUCPFree(&mem_);
        mem_       = other.mem_;
        owns_mem_  = other.owns_mem_;
        cached_    = other.cached_;
        dtype_     = std::move(other.dtype_);
        shape_     = std::move(other.shape_);
        parent_    = std::move(other.parent_);
        other.owns_mem_ = false;
        other.mem_      = {};
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════════
// 懒分配
// ══════════════════════════════════════════════════════════════════════════════

void IONArray::allocate(bool cached)
{
    if (owns_mem_)
        throw std::runtime_error("IONArray::allocate: already allocated");
    uint64_t n = dtype_.itemsize();
    for (auto d : shape_) n *= static_cast<uint64_t>(d);
    _alloc(n, cached);
}

// ══════════════════════════════════════════════════════════════════════════════
// as_array / sub_view
// ══════════════════════════════════════════════════════════════════════════════

py::array IONArray::as_array()
{
    if (!is_allocated())
        throw std::runtime_error("IONArray: not allocated, call allocate() first");
    // 通过 capsule 持有 shared_ptr<IONArray> → ION 内存不会被提前释放
    auto self_ptr = shared_from_this();
    auto capsule  = py::capsule(
        new std::shared_ptr<IONArray>(self_ptr),
        [](void *p) { delete static_cast<std::shared_ptr<IONArray>*>(p); }
    );
    return py::array(dtype_, shape_, mem_.virAddr, capsule);
}

IONArray IONArray::sub_view(const py::dtype &dtype,
                             const std::vector<ssize_t> &shape,
                             ssize_t element_offset)
{
    if (!is_allocated())
        throw std::runtime_error("IONArray: not allocated, call allocate() first");
    if (element_offset < 0)
        throw std::invalid_argument("sub_view: element_offset must be >= 0, got " + std::to_string(element_offset));
    size_t elem_size    = static_cast<size_t>(dtype_.itemsize());
    uint64_t byte_offset = static_cast<uint64_t>(element_offset) * elem_size;

    size_t dst_elem_size = static_cast<size_t>(dtype.itemsize());
    uint64_t view_bytes  = dst_elem_size;
    for (auto d : shape) view_bytes *= static_cast<uint64_t>(d);

    if (byte_offset + view_bytes > mem_.memSize)
        throw std::out_of_range(
            "sub_view: offset " + std::to_string(element_offset) + " elements (" +
            std::to_string(byte_offset) + " bytes) + view " +
            std::to_string(view_bytes) + " bytes > mem_size " +
            std::to_string(mem_.memSize));

    hbUCPSysMem sub_mem{};
    sub_mem.phyAddr = mem_.phyAddr + byte_offset;
    sub_mem.virAddr = static_cast<char*>(mem_.virAddr) + byte_offset;
    sub_mem.memSize = view_bytes;

    return IONArray(dtype, shape, sub_mem, shared_from_this());
}
