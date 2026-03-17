project(pyCauchyKesai LANGUAGES C CXX)
cmake_minimum_required(VERSION 3.18)
set(CMAKE_BUILD_TYPE Release)

if(DEFINED SKBUILD_PLATLIB_DIR)
    set(OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}")
else()
    set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# 优先使用项目内的头文件（与 Nash 一致）
include_directories(
    include
)

# 优先使用项目内的动态库（与 Nash 一致）
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 设置 RPATH，让编译出的 .so 在运行时优先从 $ORIGIN/lib 加载动态库
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)          # 不在 build 阶段写入绝对路径
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)  # 直接用 install RPATH 写入 .so
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

find_package(pybind11 REQUIRED)

pybind11_add_module(pyCauchyKesai
    src/pyCauchyKesai_bindings.cpp
    src/pyCauchyKesai.cpp
)

# X5 只需要链接 dnn 和 hbrt_bayes_aarch64
# 其他系统库（libalog、libhbmem、libcnn_intf）由系统提供
target_link_libraries(pyCauchyKesai PRIVATE
    pybind11::module
    dnn
    hbrt_bayes_aarch64
)

set_target_properties(pyCauchyKesai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}/pyCauchyKesai"
    PREFIX ""
)

# 打包时将 lib/*.so 复制到 wheel 的 pyCauchyKesai/lib/ 目录（与 Nash 一致）
file(GLOB X5_LIBS "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so")
foreach(lib ${X5_LIBS})
    get_filename_component(lib_name ${lib} NAME)
    add_custom_command(TARGET pyCauchyKesai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}/pyCauchyKesai/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} "${OUTPUT_DIR}/pyCauchyKesai/lib/${lib_name}"
    )
endforeach()
