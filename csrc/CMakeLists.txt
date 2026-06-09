cmake_minimum_required(VERSION 3.18)
project(pycauchykesai LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)

# ============================================================================
# 平台选择: Nash-e | Nash-m | Nash-p
#   Nash-e  : RDK S100, BPU nash-e
#   Nash-m  : RDK S100P, BPU nash-m
#   Nash-p  : RDK S600, BPU nash-p
# ============================================================================
if(NOT DEFINED HB_PLATFORM)
    message(FATAL_ERROR "HB_PLATFORM is required: -DHB_PLATFORM=<Nash-e|Nash-m|Nash-p>")
endif()

# ============================================================================
# Horizon BPU 系统库路径 (板端默认 /usr/hobot/lib)
# ============================================================================
if(NOT DEFINED HB_SYSROOT AND NOT "$ENV{HB_SYSROOT}" STREQUAL "")
    set(HB_SYSROOT "$ENV{HB_SYSROOT}")
endif()
if(NOT HB_SYSROOT)
    set(HB_SYSROOT "/usr/hobot")
endif()
set(HB_LIB_DIR  "${HB_SYSROOT}/lib")
set(HB_INC_DIR  "${HB_SYSROOT}/include")

# ============================================================================
# 平台参数
# ============================================================================
set(PYBIND11_USE_ABI3 ON CACHE BOOL "Build stable ABI module")

if(HB_PLATFORM STREQUAL "Nash-e" OR HB_PLATFORM STREQUAL "Nash-m")
    set(HB_BPU_ALIGN 32)
    set(PLATFORM_DIR nash)
    set(PLATFORM_LIBS hbucp dnn hbrt4 hbtl hb_arm_rpc hlog_wrapper perfetto_sdk)

elseif(HB_PLATFORM STREQUAL "Nash-p")
    set(HB_BPU_ALIGN 64)
    set(PLATFORM_DIR nash)
    set(PLATFORM_LIBS hbucp dnn hbrt4 hbtl hb_arm_rpc hlog_wrapper perfetto_sdk)

else()
    message(FATAL_ERROR "Unknown platform: ${HB_PLATFORM}")
endif()

message(STATUS "========================================")
message(STATUS "  HB_PLATFORM    = ${HB_PLATFORM}")
message(STATUS "  BPU Align      = ${HB_BPU_ALIGN}")
message(STATUS "  HB_SYSROOT     = ${HB_SYSROOT}")
message(STATUS "========================================")

add_compile_definitions(HB_BPU_ALIGN=${HB_BPU_ALIGN})

if(DEFINED SKBUILD_PLATLIB_DIR)
    set(OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}")
else()
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

find_package(pybind11 REQUIRED)

# ============================================================================
# 子模块
# ============================================================================
add_subdirectory(${PLATFORM_DIR})
