# Nash (nash-e/nash-m/nash-p) — CauchyKesai BPU inference

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hobot
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

link_directories(${HB_LIB_DIR})

set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")

pybind11_add_module(pycauchykesai
    bindings.cpp
    cauchy_kesai.cpp
    ion_array.cpp
)

target_compile_definitions(pycauchykesai PRIVATE HB_BPU_ALIGN=${HB_BPU_ALIGN})

target_link_libraries(pycauchykesai PRIVATE
    pybind11::module
    ${PLATFORM_LIBS}
)

set_target_properties(pycauchykesai PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}/pyCauchyKesai"
    PREFIX ""
)

# 将 Horizon BPU 系统库打包进 wheel (从系统路径拷贝)
# 仅在正式打包时拷贝 — editable install 时 RPATH $ORIGIN/lib:/usr/hobot/lib
# 会自动回退到 /usr/hobot/lib，无需在源树里留 82MB 副本。
if(NOT DEFINED ENV{SKBUILD_EDITABLE})
    file(GLOB HB_SOS "${HB_LIB_DIR}/libhbucp*.so" "${HB_LIB_DIR}/libdnn*.so"
         "${HB_LIB_DIR}/libhbrt4*.so" "${HB_LIB_DIR}/libhbtl*.so"
         "${HB_LIB_DIR}/libhb_arm_rpc*.so" "${HB_LIB_DIR}/libhlog_wrapper*.so"
         "${HB_LIB_DIR}/libperfetto_sdk*.so" "${HB_LIB_DIR}/libhbhpl*.so"
         "${HB_LIB_DIR}/libhbvp*.so" "${HB_LIB_DIR}/libhbdsp_plugin*.so")
    foreach(lib ${HB_SOS})
        get_filename_component(lib_name ${lib} NAME)
        add_custom_command(TARGET pycauchykesai POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUTPUT_DIR}/pyCauchyKesai/lib"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} "${OUTPUT_DIR}/pyCauchyKesai/lib/${lib_name}"
        )
    endforeach()
endif()
