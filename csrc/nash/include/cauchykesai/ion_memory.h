#ifndef _ION_MEMORY_H_
#define _ION_MEMORY_H_

#include <memory>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "hobot/hb_ucp_sys.h"

namespace py = pybind11;

class __attribute__((visibility("default"))) IONMemory
    : public std::enable_shared_from_this<IONMemory>
{
public:
    explicit IONMemory(uint64_t size, bool cached = true);
    IONMemory(hbUCPSysMem mem, bool owns = false, bool cached = true);
    ~IONMemory();

    IONMemory(const IONMemory&) = delete;
    IONMemory& operator=(const IONMemory&) = delete;

    IONMemory(IONMemory&&) noexcept;
    IONMemory& operator=(IONMemory&&) noexcept;

    uint64_t phy_addr()   const { return mem_.phyAddr; }
    void*   vir_addr()    const { return mem_.virAddr; }
    uint64_t size()       const { return mem_.memSize; }
    bool    is_cached()   const { return cached_; }
    bool    is_allocated() const { return mem_.virAddr != nullptr; }

    const hbUCPSysMem& sys_mem() const { return mem_; }

    uint64_t phy_addr_at(uint64_t off) const { return mem_.phyAddr + off; }
    void*   vir_addr_at(uint64_t off) const { return static_cast<char*>(mem_.virAddr) + off; }

    void flush_clean() const;
    void flush_invalidate() const;

    py::array numpy();

    /**
     * @brief Create a non-owning view of this IONMemory (shares the same physical memory)
     *
     * The returned IONMemory wraps the same hbUCPSysMem with owns=false.
     * Deleting the view will not free the underlying memory.
     */
    std::shared_ptr<IONMemory> non_owning_view() const;

private:
    void _alloc(uint64_t size, bool cached);

    hbUCPSysMem mem_{};
    bool owns_  = false;
    bool cached_ = true;
};

#endif // _ION_MEMORY_H_
