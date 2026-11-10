# The Python extension module, built only when KLASP_BUILD_PYTHON is ON.
#
# klasp_api.cpp includes Kokkos and is compiled like the rest of klasp;
# bindings.cpp includes nanobind and no Kokkos, and bypasses the device compiler.

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

# DOWNLOAD_ONLY plus find_package is nanobind's as-a-submodule recipe.
CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    GIT_TAG v3.1.0
    DOWNLOAD_ONLY YES
)
find_package(nanobind CONFIG REQUIRED
    PATHS "${nanobind_SOURCE_DIR}/cmake" NO_DEFAULT_PATH)

nanobind_add_module(_klasp
    NB_STATIC
    src/bindings.cpp
    src/klasp_api.cpp
)
target_include_directories(_klasp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(_klasp PRIVATE klasp_core nlohmann_json::nlohmann_json)
target_compile_definitions(_klasp PRIVATE KLASP_PY_BACKEND="${KLASP_BACKEND}")

# Under CUDA, Kokkos makes nvcc_wrapper the C++ compiler; bindings.cpp has no
# device code, so it passes --host-only (a wrapper-only flag) when the wrapper
# is the compiler.
get_filename_component(_klasp_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
if(_klasp_cxx_name STREQUAL "nvcc_wrapper")
    set_source_files_properties(src/bindings.cpp
        PROPERTIES COMPILE_OPTIONS "--host-only")
endif()

# Mirror the wheel layout so PYTHONPATH=<build>/python can import klasp.
set(KLASP_PY_STAGE "${CMAKE_CURRENT_BINARY_DIR}/klasp")
set_target_properties(_klasp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${KLASP_PY_STAGE}")
add_custom_command(TARGET _klasp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/klasp/__init__.py" "${KLASP_PY_STAGE}/__init__.py"
    COMMENT "Staging the klasp Python package"
    VERBATIM
)

# Supported install path is `pip install .`, which places this in the wheel's
# platlib beside python/klasp.
install(TARGETS _klasp LIBRARY DESTINATION klasp COMPONENT python)

if(KLASP_BUILD_TESTS)
    add_test(NAME python_module
        COMMAND ${Python_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_klasp.py"
                "${CMAKE_SOURCE_DIR}/example-inputs/acetic_acid.json")
    set_tests_properties(python_module PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
        TIMEOUT 900)
endif()
