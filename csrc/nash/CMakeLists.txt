# Nash (nash-e/nash-m/nash-p) — CauchyKesai BPU inference

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobot
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party  # vendored nlohmann/json (MIT), 三板通用无系统依赖
)

# 链接只用仓库自带的 UCP 库 (csrc/nash/lib)，不依赖系统 /usr/hobot/lib。
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")

pybind11_add_module(pycauchykesai
    bindings.cpp
    cauchy_kesai.cpp
    platform.cpp
    ion_array_desc.cpp
    ion_array.cpp
    ion_memory.cpp
)

target_link_libraries(pycauchykesai PRIVATE
    pybind11::module
    ${PLATFORM_LIBS}
)

set_target_properties(pycauchykesai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}/pyCauchyKesai"
    PREFIX ""
)

# 将仓库自带的 UCP 库 (csrc/nash/lib/*.so) 打包进 wheel。
# 编译/链接/打包全部只依赖 csrc/nash/lib，无任何系统 /usr/hobot 路径。
# RPATH=$ORIGIN/lib 运行时从包内 lib/ 加载。
file(GLOB HB_SOS "${CMAKE_CURRENT_SOURCE_DIR}/lib/*.so")
foreach(lib ${HB_SOS})
    get_filename_component(lib_name ${lib} NAME)
    add_custom_command(TARGET pycauchykesai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}/pyCauchyKesai/lib"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} "${OUTPUT_DIR}/pyCauchyKesai/lib/${lib_name}"
    )
endforeach()
