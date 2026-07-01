cmake_minimum_required(VERSION 3.18)
project(pycauchykesai LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)

# ============================================================================
# 平台：Nash（nash-e / nash-m / nash-p 三板共用同一份源码与同一组 UCP 库）
#   BPU 对齐值（32/64）由运行时 hbUCPGetSocName 检测（见 platform.cpp bpu_align()），
#   编译期无需任何平台参数。
# ============================================================================
set(PLATFORM_DIR nash)
set(PLATFORM_LIBS hbucp dnn hbrt4 hbtl hb_arm_rpc hlog_wrapper perfetto_sdk)

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
