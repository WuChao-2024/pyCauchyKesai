# project(pyCauchyKesai)
# cmake_minimum_required(VERSION 3.0)

# include_directories(
#     include/hobot
#     include
# )

# link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
# set(CMAKE_SKIP_RPATH FALSE)
# set(CMAKE_BUILD_RPATH "$ORIGIN/lib")  

# find_package(pybind11 REQUIRED)

# add_library(pyCauchyKesai MODULE 
#             src/pyCauchyKesai_bindings.cpp 
#             src/pyCauchyKesai.cpp)
# target_link_libraries(pyCauchyKesai PRIVATE 
#     pybind11::module     
#     hbucp
#     dnn
#     hbrt4
#     hbtl
#     hb_arm_rpc
#     hlog_wrapper
#     perfetto_sdk
# )

project(pyCauchyKesai LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Release)

# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# set(CMAKE_BUILD_TYPE Debug)

if(DEFINED SKBUILD_PLATLIB_DIR)
    set(OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}")   # scikit-build-core 会传入此变量
else()
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

include_directories(
    include/hobot
    include
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")  # 注意：wheel 中 so 在 pyCauchyKesai/lib/

find_package(pybind11 REQUIRED)

# 构建模块
pybind11_add_module(pyCauchyKesai
    src/pyCauchyKesai_bindings.cpp
    src/pyCauchyKesai.cpp
)

target_link_libraries(pyCauchyKesai PRIVATE
    pybind11::module
    hbucp
    dnn
    hbrt4
    hbtl
    hb_arm_rpc
    hlog_wrapper
    perfetto_sdk
)

# —————— 关键：设置输出路径和 RPATH ——————
set_target_properties(pyCauchyKesai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}/pyCauchyKesai"
    PREFIX ""  # 避免 lib 前缀
)

# 安装依赖的 .so 到 pyCauchyKesai/lib/
file(GLOB HOBOT_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so")
foreach(lib ${HOBOT_LIBS})
    get_filename_component(lib_name ${lib} NAME)
    add_custom_command(TARGET pyCauchyKesai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}/pyCauchyKesai/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} "${OUTPUT_DIR}/pyCauchyKesai/lib/${lib_name}"
    )
endforeach()