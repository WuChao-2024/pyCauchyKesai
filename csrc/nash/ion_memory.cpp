#include "cauchykesai/ion_memory.h"
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

// ══════════════════════════════════════════════════════════════════════════════
// 内部辅助
// ══════════════════════════════════════════════════════════════════════════════

void IONMemory::_alloc(uint64_t size, bool cached)
{
    int32_t ret;
    if (cached)
        ret = hbUCPMallocCached(&mem_, size, 0);
    else
        ret = hbUCPMalloc(&mem_, size, 0);
    if (ret != 0)
        throw std::runtime_error(std::string("IONMemory: ") +
            (cached ? "hbUCPMallocCached" : "hbUCPMalloc") + " failed");
    owns_  = true;
    cached_ = cached;
}

// ══════════════════════════════════════════════════════════════════════════════
// 构造 / 析构
// ══════════════════════════════════════════════════════════════════════════════

IONMemory::IONMemory(uint64_t size, bool cached)
{
    _alloc(size, cached);
}

IONMemory::IONMemory(hbUCPSysMem mem, bool owns, bool cached)
    : mem_(mem), owns_(owns), cached_(cached)
{
}

IONMemory::~IONMemory()
{
    if (owns_ && mem_.virAddr != nullptr)
    {
        hbUCPFree(&mem_);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Move 语义
// ══════════════════════════════════════════════════════════════════════════════

IONMemory::IONMemory(IONMemory&& other) noexcept
    : mem_(other.mem_), owns_(other.owns_), cached_(other.cached_)
{
    other.mem_ = {};
    other.owns_ = false;
}

IONMemory& IONMemory::operator=(IONMemory&& other) noexcept
{
    if (this != &other)
    {
        // 释放自己持有的资源
        if (owns_ && mem_.virAddr != nullptr)
            hbUCPFree(&mem_);

        mem_    = other.mem_;
        owns_   = other.owns_;
        cached_ = other.cached_;

        other.mem_ = {};
        other.owns_ = false;
    }
    return *this;
}

// ══════════════════════════════════════════════════════════════════════════════
// Cache 操作
// ══════════════════════════════════════════════════════════════════════════════

void IONMemory::flush_clean() const
{
    if (cached_ && mem_.virAddr != nullptr)
        hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_CLEAN);
}

void IONMemory::flush_invalidate() const
{
    if (cached_ && mem_.virAddr != nullptr)
        hbUCPMemFlush(&mem_, HB_SYS_MEM_CACHE_INVALIDATE);
}

// ══════════════════════════════════════════════════════════════════════════════
// numpy: 导出 uint8 numpy array（capsule 保活）
// ══════════════════════════════════════════════════════════════════════════════

py::array IONMemory::numpy()
{
    if (!is_allocated())
        throw std::runtime_error("IONMemory: not allocated");

    auto self_ptr = shared_from_this();
    auto capsule = py::capsule(
        new std::shared_ptr<IONMemory>(self_ptr),
        [](void *p) { delete static_cast<std::shared_ptr<IONMemory>*>(p); }
    );

    std::vector<ssize_t> shape = { static_cast<ssize_t>(mem_.memSize) };
    return py::array(py::dtype::of<uint8_t>(), shape, mem_.virAddr, capsule);
}

// ══════════════════════════════════════════════════════════════════════════════
// non_owning_view
// ══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<IONMemory> IONMemory::non_owning_view() const
{
    return std::make_shared<IONMemory>(mem_, /*owns=*/false, cached_);
}
